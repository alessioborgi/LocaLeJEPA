# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Cluster-structure evaluation of frozen encoder embeddings -- no GPU, no checkpoints.

Consumes the feature caches written by ``eval_indomain.py`` (which produces the
SPHERE-JEPA Table-4 numbers) and answers the follow-up question: *what do the
clusters behind those numbers actually look like, and how good are they?*

Two outputs.

**Metrics** (``metrics.csv``) -- deliberately no silhouette. Silhouette scores
compactness, and BallReg exists to spread points uniformly inside a ball, so it
penalises that method by construction and ranks RectLpJEPA above backbones that
beat it on every accuracy measure. Instead:

  knn20 / linear   in-domain accuracy, train bank -> val query (Table 4 protocol)
  kmeans_ari       agreement of k-means(10) with the true classes. k-means can
                   only cut out compact convex cells, so this asks "are the
                   classes blobs?"
  spectral_ari     same score for spectral clustering on the k-NN graph, which
                   follows connectivity instead. Reading the two together
                   separates "compact" from "well-separated" -- they disagree,
                   and the disagreement is the interesting part.
  purity@r         P(neighbour r has the same class), r = 1..100. A curve, not a
                   scalar: distinguishes a clean core from a clean far field.
  orc_gap          within- minus cross-class Ollivier-Ricci curvature.
  orc_auroc        AUROC of -kappa as a detector of class-crossing edges.
                   Threshold-free, unlike prune-rate statistics, which on this
                   data fire on a handful of edges and cannot separate backbones.
  frag_k*          connected components per class vs graph k.

**Layouts** (``coords.npz``) -- t-SNE, UMAP, and EmbedOR with and without
ORC-ManL pruning, for the class-coloured figures. Several methods rather than
one because EmbedOR's curvature reweighting stretches negative-curvature edges
by construction, so structure that appears only in its panel is the method
rather than the data. ``plot_embedding_structure.py`` reports how much class
structure each layout retains, which makes that check quantitative.

    python eval_structure.py                     # all layouts
    python eval_structure.py --layouts tsne umap # skip the slow EmbedOR runs

**Metric** -- ``--metric cosine`` (default) treats features as free vectors and
z-scores them. ``--metric poincare`` treats them as points in the Poincare ball
the hyperbolic encoder maps onto, and uses geodesic distances throughout: the
k-NN graph the curvature is computed on, the neighbour rankings, the spectral
affinity and all three layouts. Use it only for H-ViT backbones, and run both to
see what the imposed geometry buys -- cosine reads only the direction of a
feature, and the norm is where a hyperbolic model stores hierarchy depth:

    python eval_structure.py --cache logs/eval_indomain_imagenette_geometric/cache \
                             --out logs/eval_structure_geometric_poincare \
                             --metric poincare --hyp-c 0.1

Two columns are approximations rather than exact hyperbolic quantities, both
flagged in the functions that compute them: ``kmeans_*`` (no closed-form Frechet
mean -- run in the tangent space) and the Table-4 ``knn*``/``linear`` columns
(a different vote rule and a tangent-space head, so they are comparable across
*backbones* but not across *metrics*). ``loo_knn20`` is the one accuracy column
computed identically under both.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.eval.embedor import build_knn_graph, compute_orc, run_embedor  # noqa: E402
from src.eval.knn_probe import weighted_knn  # noqa: E402
from src.eval.poincare import DEFAULT_C, check_in_ball, logmap0, poincare_distances  # noqa: E402

DEFAULT_CACHE = "logs/eval_indomain_imagenette_ksweep/cache"
DEFAULT_OUT = "logs/eval_structure"
LAYOUTS = ("tsne", "umap", "embedor", "embedor_pruned")
METRICS = ("cosine", "poincare")

# A cache extracted from an unconverged checkpoint scores far below the ~0.89 the
# converged Imagenette backbones reach; refuse to plot those silently.
MIN_LOO_KNN = 0.75

_SPLIT_RE = re.compile(r"_(train|val)_\d+_ep")


def standardize(X: np.ndarray) -> np.ndarray:
    sd = X.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - X.mean(0, keepdims=True)) / sd


def discover_snapshots(cache_dir: str, epoch: int) -> dict[str, dict[str, str]]:
    """{backbone: {"val": path}} from a SNAPSHOT cache written by eval_snapshots.py.

    That cache is laid out ``<cache>/<name>/val_<crop>_ep<N>.pt`` -- val only, one
    file per training epoch -- whereas eval_indomain.py writes
    ``<ds>_<split>_<crop>_ep<N>.pt`` with both splits. Selecting an epoch here is
    what lets the layouts and cluster metrics be read at a training horizon
    (e.g. epoch 99, the manuscript's 100-epoch reading) rather than only at the
    best checkpoint. The Table-4 columns are unavailable, since no train split
    was cached.
    """
    found: dict[str, dict[str, str]] = {}
    for path in sorted(glob.glob(os.path.join(cache_dir, "*", f"val_*_ep{epoch}.pt"))):
        found[os.path.basename(os.path.dirname(path))] = {"val": path}
    return found


def available_epochs(cache_dir: str) -> list[int]:
    """Epochs present in a snapshot cache, for error messages."""
    eps = set()
    for path in glob.glob(os.path.join(cache_dir, "*", "val_*_ep*.pt")):
        m = re.search(r"_ep(\d+)\.pt$", path)
        if m:
            eps.add(int(m.group(1)))
    return sorted(eps)


def discover(cache_dir: str) -> dict[str, dict[str, str]]:
    """{backbone: {split: path}} from ``<cache>/<name>/<ds>_<split>_<crop>_ep<N>.pt``."""
    found: dict[str, dict[str, str]] = {}
    for path in sorted(glob.glob(os.path.join(cache_dir, "*", "*.pt"))):
        m = _SPLIT_RE.search(os.path.basename(path))
        if m:
            found.setdefault(os.path.basename(os.path.dirname(path)), {})[m.group(1)] = path
    return found


def load(path: str) -> tuple[np.ndarray, np.ndarray]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return blob["X"].float().numpy(), blob["y"].numpy().astype(int)


def _knn_indices(X: np.ndarray, k: int, D: np.ndarray | None) -> np.ndarray:
    """Indices of the k nearest neighbours of each row, self excluded."""
    if D is not None:
        Dm = np.array(D, dtype=np.float64, copy=True)
        np.fill_diagonal(Dm, np.inf)
        idx = np.argpartition(Dm, kth=k - 1, axis=1)[:, :k]
        row = np.arange(len(Dm))[:, None]
        return idx[row, np.argsort(Dm[row, idx], axis=1)]
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X)
    return nn.kneighbors(X, return_distance=False)[:, 1:]


def loo_knn(X: np.ndarray, y: np.ndarray, k: int = 20, D: np.ndarray | None = None) -> float:
    """Leave-one-out majority vote among the k nearest neighbours.

    Unweighted on purpose: it is the one accuracy column computed identically
    under both metrics, so cosine and Poincare runs can be compared on it
    directly (the Table-4 k-NN columns cannot -- see :func:`table4`).
    """
    idx = _knn_indices(X, k, D)
    return float((np.array([np.bincount(y[r]).argmax() for r in idx]) == y).mean())


def table4_poincare(Xtr, ytr, Xva, yva, c: float, k_values=(10, 20, 100, 200),
                    chunk: int = 512) -> dict:
    """Table-4 protocol under the intrinsic ball metric.

    Two deliberate departures from the cosine path, both of which make the
    numbers non-comparable with it (compare ``loo_knn20`` across metrics
    instead, and the clustering/curvature columns, which are metric-faithful on
    both sides):

    * the k-NN vote is an unweighted majority over the k nearest bank points.
      The cosine path weights votes by ``exp(sim / 0.07)``, and a temperature in
      units of hyperbolic distance -- which is unbounded -- has no equivalent
      default.
    * the linear head is fitted in the tangent space at the origin
      (``logmap0``), the standard linear probe for a ball-valued representation.
      A hyperplane in the ambient coordinates is not a hyperbolic decision
      boundary, and the tangent map is exactly the encoder's own ``expmap0``
      inverted.
    """
    from sklearn.linear_model import LogisticRegression

    max_k = max(k_values)
    votes = np.empty((len(yva), max_k), dtype=int)
    for start in range(0, len(yva), chunk):
        stop = min(start + chunk, len(yva))
        d = poincare_distances(Xva[start:stop], Xtr, c=c)
        idx = np.argpartition(d, kth=max_k - 1, axis=1)[:, :max_k]
        row = np.arange(stop - start)[:, None]
        votes[start:stop] = ytr[idx[row, np.argsort(d[row, idx], axis=1)]]

    out = {}
    for k in k_values:
        pred = np.array([np.bincount(v).argmax() for v in votes[:, :k]])
        out[f"knn{k}_top1"] = round(100.0 * float((pred == yva).mean()), 2)

    Ttr, Tva = logmap0(Xtr, c), logmap0(Xva, c)
    mu, sd = Ttr.mean(0, keepdims=True), Ttr.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    clf = LogisticRegression(max_iter=3000, C=1.0).fit((Ttr - mu) / sd, ytr)
    out["linear_top1"] = round(100.0 * clf.score((Tva - mu) / sd, yva), 2)
    out["knn_rule"] = "majority-poincare"
    return out


def table4(Xtr, ytr, Xva, yva) -> dict:
    """k-NN + linear on the real protocol: fit/index on train, score on val."""
    from sklearn.linear_model import LogisticRegression

    res = weighted_knn(
        torch.from_numpy(Xtr).float(), torch.from_numpy(ytr).long(),
        torch.from_numpy(Xva).float(), torch.from_numpy(yva).long(),
        num_classes=int(max(ytr.max(), yva.max())) + 1, k_values=[10, 20, 100, 200],
        temperature=0.07, device="cpu", chunk_size=512, bank_dtype=torch.float32,
    )
    out = {f"knn{k}_top1": round(v["top1"], 2) for k, v in res.items()}
    # eval_indomain.py sweeps an SGD lr grid on GPU; this deterministic
    # multinomial fit is the CPU stand-in used for ranking. Quote the numbers in
    # results.csv from eval_indomain.py for the paper table itself.
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, ytr)
    out["linear_top1"] = round(100.0 * clf.score(Xva, yva), 2)
    out["knn_rule"] = "weighted-cosine"
    return out


def clustering_scores(X: np.ndarray, y: np.ndarray, seed: int,
                      D: np.ndarray | None = None, c: float = DEFAULT_C) -> dict:
    """k-means (compactness) and spectral (connectivity) agreement with the classes.

    Under the Poincare metric, spectral clustering is exact -- it only needs the
    k-NN graph, which is built from the geodesic distances. k-means is not: it
    needs centroids, and a Frechet mean on the ball has no closed form. It is
    therefore run in the tangent space at the origin, which preserves radial
    structure exactly and distorts only the tangential part. Read the k-means
    column as an approximation on the hyperbolic side; the spectral one is not.
    """
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    from sklearn.preprocessing import normalize

    n = int(y.max()) + 1
    Xn = logmap0(X, c) if D is not None else normalize(X)  # else: cosine geometry
    out = {}
    km = KMeans(n_clusters=n, n_init=10, random_state=seed).fit_predict(Xn)
    out["kmeans_ari"] = round(adjusted_rand_score(y, km), 4)
    out["kmeans_ami"] = round(adjusted_mutual_info_score(y, km), 4)
    try:
        if D is not None:
            sc = SpectralClustering(n_clusters=n, affinity="precomputed_nearest_neighbors",
                                    n_neighbors=15, random_state=seed,
                                    assign_labels="kmeans").fit_predict(D)
        else:
            sc = SpectralClustering(n_clusters=n, affinity="nearest_neighbors", n_neighbors=15,
                                    random_state=seed, assign_labels="kmeans").fit_predict(Xn)
        out["spectral_ari"] = round(adjusted_rand_score(y, sc), 4)
        out["spectral_ami"] = round(adjusted_mutual_info_score(y, sc), 4)
    except Exception as exc:
        out["spectral_error"] = repr(exc)
    return out


def purity_profile(X: np.ndarray, y: np.ndarray, max_rank: int = 100,
                   D: np.ndarray | None = None) -> np.ndarray:
    idx = _knn_indices(X, max_rank, D)
    return (y[idx] == y[:, None]).mean(axis=0)


def curvature_scores(X: np.ndarray, y: np.ndarray, n_neighbors: int, n_jobs: int,
                     D: np.ndarray | None = None) -> dict:
    """Ollivier-Ricci curvature statistics of the k-NN graph.

    Only the graph depends on the metric: the curvature itself uses uniform
    1-hop measures and a hop-count ground metric, so switching to Poincare
    distances changes *which* edges exist, not how kappa is computed.
    """
    from sklearn.metrics import roc_auc_score

    G = build_knn_graph(None if D is not None else np.ascontiguousarray(X, dtype=np.float64),
                        n_neighbors, distances=D)
    orc = compute_orc(G, n_jobs=n_jobs)
    kappa = np.array([orc[(min(u, v), max(u, v))] for u, v in G.edges()])
    cross = np.array([int(y[u] != y[v]) for u, v in G.edges()])

    out = {
        "orc_within": round(float(kappa[cross == 0].mean()), 4) if (cross == 0).any() else float("nan"),
        "orc_cross": round(float(kappa[cross == 1].mean()), 4) if (cross == 1).any() else float("nan"),
        "n_edges": int(kappa.size),
        "edge_cross_frac": round(float(cross.mean()), 4),
    }
    out["orc_gap"] = round(out["orc_within"] - out["orc_cross"], 4)
    if 0 < cross.sum() < cross.size:
        out["orc_auroc"] = round(float(roc_auc_score(cross, -kappa)), 4)
    return out


def fragmentation_sweep(X: np.ndarray, y: np.ndarray, ks: list[int],
                        D: np.ndarray | None = None) -> dict[str, float]:
    import networkx as nx

    Xc = None if D is not None else np.ascontiguousarray(X, dtype=np.float64)
    out = {}
    for k in ks:
        G = build_knn_graph(Xc, k, distances=D)
        out[f"frag_k{k}"] = round(float(np.mean([
            nx.number_connected_components(G.subgraph(np.flatnonzero(y == c).tolist()))
            for c in np.unique(y)
        ])), 2)
    return out


def layout(X: np.ndarray, y: np.ndarray, method: str, seed: int, n_jobs: int,
           prune_delta: float, prune_lda: float, D: np.ndarray | None = None) -> np.ndarray:
    """2-D layout of the embeddings; ``D`` switches every method to that metric."""
    if method == "tsne":
        from sklearn.manifold import TSNE

        if D is not None:
            # A PCA init is undefined without coordinates, hence init="random".
            return TSNE(n_components=2, perplexity=40, metric="precomputed", init="random",
                        random_state=seed).fit_transform(D)
        return TSNE(n_components=2, perplexity=40, metric="cosine", init="pca",
                    random_state=seed).fit_transform(X)
    if method == "umap":
        import umap

        return umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                         metric="precomputed" if D is not None else "cosine",
                         random_state=seed).fit_transform(D if D is not None else X)
    if method in ("embedor", "embedor_pruned"):
        return run_embedor(
            X, y, n_neighbors=15, perplexity=40, p=3.0, epochs=300, seed=seed,
            prune=(method == "embedor_pruned"), prune_delta=prune_delta,
            prune_lda=prune_lda, n_jobs=n_jobs, device="cpu", verbose=False,
            distances=D,
        ).coords
    raise ValueError(method)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default=DEFAULT_CACHE,
                    help="feature-cache root written by eval_indomain.py")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--layouts", nargs="*", default=list(LAYOUTS), choices=list(LAYOUTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--prune-delta", type=float, default=0.8,
                    help="ORC-ManL curvature confidence; candidates have kappa < -1+4(1-delta)")
    ap.add_argument("--prune-lda", type=float, default=0.01,
                    help="ORC-ManL metric distortion; detour bound (1-lda)pi^2/(2 sqrt(24 lda))")
    ap.add_argument("--only", default=None, help="comma-separated backbone names")
    ap.add_argument("--epoch", type=int, default=None,
                    help="read a SNAPSHOT cache (eval_snapshots.py layout) at this training "
                         "epoch instead of the eval_indomain.py best-checkpoint cache. "
                         "Val split only, so the Table-4 columns are omitted. Early-epoch "
                         "features score below the default --min-loo gate, so lower it")
    ap.add_argument("--metric", default="cosine", choices=list(METRICS),
                    help="geometry every metric and layout is computed in. "
                         "'poincare' is for hyperbolic backbones only (H-ViT): it uses "
                         "the raw, UNSTANDARDISED features, since z-scoring moves points "
                         "out of the ball")
    ap.add_argument("--min-loo", type=float, default=MIN_LOO_KNN,
                    help=f"skip a backbone whose LOO k-NN falls below this "
                         f"(default {MIN_LOO_KNN}: an unconverged checkpoint). Lower it "
                         f"only deliberately -- a converged Imagenette backbone is ~0.89")
    ap.add_argument("--hyp-c", type=float, default=DEFAULT_C,
                    help="Poincare-ball curvature parameter; must match the encoder "
                         "config (hvit_small_p8.yaml: hyp_c=0.1)")
    args = ap.parse_args()
    hyper = args.metric == "poincare"

    only = {n.strip() for n in args.only.split(",")} if args.only else None
    if args.epoch is not None:
        caches = discover_snapshots(args.cache, args.epoch)
        if not caches:
            eps = available_epochs(args.cache)
            sys.exit(f"no snapshot features for epoch {args.epoch} under {args.cache}\n"
                     + (f"available epochs: {eps}" if eps else
                        "no snapshot cache here at all -- this flag expects the cache/ "
                        "directory written by eval_snapshots.py."))
        print(f"snapshot cache at epoch {args.epoch}: {len(caches)} backbones "
              f"(val split only -- no Table-4 columns)")
    else:
        caches = discover(args.cache)
        if not caches:
            sys.exit(f"no feature caches under {args.cache}\n"
                     f"Run eval_indomain.py on the cluster first, then rsync its cache/ directory.")

    os.makedirs(args.out, exist_ok=True)
    rows, purity, coords = [], {}, {}

    for name, splits in sorted(caches.items()):
        if only and name not in only:
            continue
        if "val" not in splits:
            print(f"  {name:14s} no val cache -- skipping")
            continue

        Xva_raw, yva = load(splits["val"])
        if hyper:
            # Standardising would leave the ball, so the hyperbolic path keeps the
            # raw features and carries the geometry in Dva instead.
            check_in_ball(Xva_raw, args.hyp_c)
            Xva, Dva = Xva_raw, poincare_distances(Xva_raw, c=args.hyp_c)
        else:
            Xva, Dva = standardize(Xva_raw), None
        loo = loo_knn(Xva, yva, D=Dva)
        if loo < args.min_loo:
            print(f"  {name:14s} LOO k-NN {loo:.3f} < {args.min_loo} -- unconverged checkpoint, skipping")
            continue

        t0 = time.time()
        row = {"backbone": name, "metric": args.metric,
               "epoch": args.epoch if args.epoch is not None else "best",
               "n_val": len(yva), "dim": Xva.shape[1], "loo_knn20": round(100 * loo, 2)}

        if "train" in splits:
            Xtr_raw, ytr = load(splits["train"])
            row["n_train"] = len(ytr)
            if hyper:
                check_in_ball(Xtr_raw, args.hyp_c)
                row.update(table4_poincare(Xtr_raw, ytr, Xva_raw, yva, args.hyp_c))
            else:
                # One standardisation fitted on train and applied to both, so the
                # bank and the queries live in the same frame.
                mu, sd = Xtr_raw.mean(0, keepdims=True), Xtr_raw.std(0, keepdims=True)
                sd[sd == 0] = 1.0
                row.update(table4((Xtr_raw - mu) / sd, ytr, (Xva_raw - mu) / sd, yva))
        else:
            print(f"  {name:14s} no train cache -- Table-4 columns unavailable")

        row.update(clustering_scores(Xva, yva, args.seed, D=Dva, c=args.hyp_c))
        row.update(curvature_scores(Xva, yva, 15, args.n_jobs, D=Dva))
        row.update(fragmentation_sweep(Xva, yva, [5, 10, 15, 25, 50], D=Dva))
        rows.append(row)
        purity[name] = purity_profile(Xva, yva, D=Dva).tolist()

        coords[name] = {"labels": yva}
        for m in args.layouts:
            tm = time.time()
            coords[name][m] = layout(Xva, yva, m, args.seed, args.n_jobs,
                                     args.prune_delta, args.prune_lda, D=Dva)
            print(f"  {name:14s} {m:15s} {time.time() - tm:5.1f}s")

        print(f"  {name:14s} done {time.time() - t0:5.0f}s | "
              f"knn20 {row.get('knn20_top1', float('nan')):.1f} "
              f"ARI km {row['kmeans_ari']:.3f} / spec {row.get('spectral_ari', float('nan')):.3f}")

    if not rows:
        sys.exit("no usable backbones found")

    cols = sorted({k for r in rows for k in r},
                  key=lambda c: (c not in ("backbone", "metric", "epoch"), c))
    with open(os.path.join(args.out, "metrics.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "purity.json"), "w", encoding="utf-8") as fh:
        json.dump(purity, fh)
    np.savez_compressed(os.path.join(args.out, "coords.npz"),
                        **{f"{n}__{m}": v for n, d in coords.items() for m, v in d.items()})
    print(f"\nwrote {args.out}/metrics.csv, purity.json, coords.npz  ({len(rows)} backbones)")


if __name__ == "__main__":
    main()
