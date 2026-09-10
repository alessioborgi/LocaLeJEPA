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

DEFAULT_CACHE = "logs/eval_indomain_imagenette_ksweep/cache"
DEFAULT_OUT = "logs/eval_structure"
LAYOUTS = ("tsne", "umap", "embedor", "embedor_pruned")

# A cache extracted from an unconverged checkpoint scores far below the ~0.89 the
# converged Imagenette backbones reach; refuse to plot those silently.
MIN_LOO_KNN = 0.75

_SPLIT_RE = re.compile(r"_(train|val)_\d+_ep")


def standardize(X: np.ndarray) -> np.ndarray:
    sd = X.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - X.mean(0, keepdims=True)) / sd


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


def loo_knn(X: np.ndarray, y: np.ndarray, k: int = 20) -> float:
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X)
    _, idx = nn.kneighbors(X)
    return float((np.array([np.bincount(y[r]).argmax() for r in idx[:, 1:]]) == y).mean())


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
    return out


def clustering_scores(X: np.ndarray, y: np.ndarray, seed: int) -> dict:
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    from sklearn.preprocessing import normalize

    n = int(y.max()) + 1
    Xn = normalize(X)  # cosine geometry, matching the k-NN metric
    out = {}
    km = KMeans(n_clusters=n, n_init=10, random_state=seed).fit_predict(Xn)
    out["kmeans_ari"] = round(adjusted_rand_score(y, km), 4)
    out["kmeans_ami"] = round(adjusted_mutual_info_score(y, km), 4)
    try:
        sc = SpectralClustering(n_clusters=n, affinity="nearest_neighbors", n_neighbors=15,
                                random_state=seed, assign_labels="kmeans").fit_predict(Xn)
        out["spectral_ari"] = round(adjusted_rand_score(y, sc), 4)
        out["spectral_ami"] = round(adjusted_mutual_info_score(y, sc), 4)
    except Exception as exc:
        out["spectral_error"] = repr(exc)
    return out


def purity_profile(X: np.ndarray, y: np.ndarray, max_rank: int = 100) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=max_rank + 1, metric="cosine").fit(X)
    _, idx = nn.kneighbors(X)
    return (y[idx[:, 1:]] == y[:, None]).mean(axis=0)


def curvature_scores(X: np.ndarray, y: np.ndarray, n_neighbors: int, n_jobs: int) -> dict:
    from sklearn.metrics import roc_auc_score

    G = build_knn_graph(np.ascontiguousarray(X, dtype=np.float64), n_neighbors)
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


def fragmentation_sweep(X: np.ndarray, y: np.ndarray, ks: list[int]) -> dict[str, float]:
    import networkx as nx

    Xc = np.ascontiguousarray(X, dtype=np.float64)
    out = {}
    for k in ks:
        G = build_knn_graph(Xc, k)
        out[f"frag_k{k}"] = round(float(np.mean([
            nx.number_connected_components(G.subgraph(np.flatnonzero(y == c).tolist()))
            for c in np.unique(y)
        ])), 2)
    return out


def layout(X: np.ndarray, y: np.ndarray, method: str, seed: int, n_jobs: int,
           prune_delta: float, prune_lda: float) -> np.ndarray:
    if method == "tsne":
        from sklearn.manifold import TSNE

        return TSNE(n_components=2, perplexity=40, metric="cosine", init="pca",
                    random_state=seed).fit_transform(X)
    if method == "umap":
        import umap

        return umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine",
                         random_state=seed).fit_transform(X)
    if method in ("embedor", "embedor_pruned"):
        return run_embedor(
            X, y, n_neighbors=15, perplexity=40, p=3.0, epochs=300, seed=seed,
            prune=(method == "embedor_pruned"), prune_delta=prune_delta,
            prune_lda=prune_lda, n_jobs=n_jobs, device="cpu", verbose=False,
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
    args = ap.parse_args()

    only = {n.strip() for n in args.only.split(",")} if args.only else None
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
        Xva = standardize(Xva_raw)
        loo = loo_knn(Xva, yva)
        if loo < MIN_LOO_KNN:
            print(f"  {name:14s} LOO k-NN {loo:.3f} < {MIN_LOO_KNN} -- unconverged checkpoint, skipping")
            continue

        t0 = time.time()
        row = {"backbone": name, "n_val": len(yva), "dim": Xva.shape[1],
               "loo_knn20": round(100 * loo, 2)}

        if "train" in splits:
            Xtr_raw, ytr = load(splits["train"])
            # One standardisation fitted on train and applied to both, so the
            # bank and the queries live in the same frame.
            mu, sd = Xtr_raw.mean(0, keepdims=True), Xtr_raw.std(0, keepdims=True)
            sd[sd == 0] = 1.0
            row["n_train"] = len(ytr)
            row.update(table4((Xtr_raw - mu) / sd, ytr, (Xva_raw - mu) / sd, yva))
        else:
            print(f"  {name:14s} no train cache -- Table-4 columns unavailable")

        row.update(clustering_scores(Xva, yva, args.seed))
        row.update(curvature_scores(Xva, yva, 15, args.n_jobs))
        row.update(fragmentation_sweep(Xva, yva, [5, 10, 15, 25, 50]))
        rows.append(row)
        purity[name] = purity_profile(Xva, yva).tolist()

        coords[name] = {"labels": yva}
        for m in args.layouts:
            tm = time.time()
            coords[name][m] = layout(Xva, yva, m, args.seed, args.n_jobs,
                                     args.prune_delta, args.prune_lda)
            print(f"  {name:14s} {m:15s} {time.time() - tm:5.1f}s")

        print(f"  {name:14s} done {time.time() - t0:5.0f}s | "
              f"knn20 {row.get('knn20_top1', float('nan')):.1f} "
              f"ARI km {row['kmeans_ari']:.3f} / spec {row.get('spectral_ari', float('nan')):.3f}")

    if not rows:
        sys.exit("no usable backbones found")

    cols = sorted({k for r in rows for k in r}, key=lambda c: (c != "backbone", c))
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
