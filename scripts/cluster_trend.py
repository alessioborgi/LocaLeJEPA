# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Cluster-quality scalars as a function of TRAINING epoch.

``eval_structure.py`` describes one checkpoint. This walks the whole snapshot
cache and computes the same scalars at every epoch, so the question becomes a
trend rather than a snapshot: does the geometry a backbone induces change while
the probe accuracy climbs, or does accuracy move on its own?

That distinction is the open one in ``paper/k_ablation.tex`` -- on plain ViT the
neighborhood count moved both probes while every structural descriptor stayed
flat, and no mechanism connected them. A curve can separate "flat because
nothing happens" from "flat because it saturated before epoch 100".

Reuses ``eval_structure.py`` for every metric, so a number here and a number
there are the same computation. Layouts are NOT computed (see
``eval_structure.py --epoch``); this is scalars only, ~7.5s per run-epoch.

    python scripts/cluster_trend.py                       # epochs <= 100, ~4 min
    python scripts/cluster_trend.py --max-epoch 800       # everything, ~28 min
    python scripts/cluster_trend.py --metric poincare     # hyperbolic backbones

Rows are appended as they are computed and existing (run, epoch) pairs are
skipped, so an interrupted run resumes and a widened --max-epoch only computes
what is new.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval_structure as ES  # noqa: E402

DEFAULT_CACHE = "logs/eval_snapshots_geometric/cache"
DEFAULT_OUT = "logs/cluster_trend"
FIELDS = ["run", "epoch", "metric", "n_val", "loo_knn20", "kmeans_ari", "kmeans_ami",
          "spectral_ari", "spectral_ami", "orc_within", "orc_cross", "orc_gap",
          "orc_auroc", "edge_cross_frac", "n_edges", "frag_k5", "frag_k15", "frag_k50",
          "purity_r1", "purity_r10", "purity_r50"]
_EP_RE = re.compile(r"_ep(\d+)\.pt$")


def discover(cache_dir: str) -> dict[str, dict[int, str]]:
    """{run: {epoch: path}} over a snapshot cache."""
    runs: dict[str, dict[int, str]] = {}
    for path in sorted(glob.glob(os.path.join(cache_dir, "*", "val_*_ep*.pt"))):
        m = _EP_RE.search(path)
        if m:
            runs.setdefault(os.path.basename(os.path.dirname(path)), {})[int(m.group(1))] = path
    return runs


def existing(path: str) -> set[tuple[str, int]]:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as fh:
        return {(r["run"], int(r["epoch"])) for r in csv.DictReader(fh)}


def evaluate(path: str, seed: int, n_jobs: int, metric: str, hyp_c: float) -> dict:
    X_raw, y = ES.load(path)
    if metric == "poincare":
        ES.check_in_ball(X_raw, hyp_c)
        X, D = X_raw, ES.poincare_distances(X_raw, c=hyp_c)
    else:
        X, D = ES.standardize(X_raw), None

    row = {"n_val": len(y), "loo_knn20": round(100 * ES.loo_knn(X, y, D=D), 3)}
    row.update(ES.clustering_scores(X, y, seed, D=D, c=hyp_c))
    row.update(ES.curvature_scores(X, y, 15, n_jobs, D=D))
    row.update(ES.fragmentation_sweep(X, y, [5, 15, 50], D=D))
    prof = ES.purity_profile(X, y, 50, D=D)
    row.update({"purity_r1": round(float(prof[0]), 4),
                "purity_r10": round(float(prof[9]), 4),
                "purity_r50": round(float(prof[49]), 4)})
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=DEFAULT_CACHE, help="snapshot cache from eval_snapshots.py")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--max-epoch", type=int, default=100)
    ap.add_argument("--min-epoch", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth available snapshot")
    ap.add_argument("--only", default=None, help="comma-separated run names")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--metric", default="cosine", choices=list(ES.METRICS))
    ap.add_argument("--hyp-c", type=float, default=ES.DEFAULT_C)
    ap.add_argument("--overwrite", action="store_true", help="recompute rows already present")
    args = ap.parse_args()

    runs = discover(args.cache)
    if args.only:
        want = {n.strip() for n in args.only.split(",")}
        runs = {k: v for k, v in runs.items() if k in want}
    if not runs:
        sys.exit(f"no snapshot features under {args.cache}")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "metrics.csv")
    done = set() if args.overwrite else existing(out_path)
    fresh = args.overwrite or not os.path.exists(out_path)

    todo = [(run, ep, p)
            for run, eps in sorted(runs.items())
            for ep, p in sorted(eps.items())[:: max(args.stride, 1)]
            if args.min_epoch <= ep <= args.max_epoch and (run, ep) not in done]
    if not todo:
        sys.exit(f"nothing to do -- {len(done)} rows already in {out_path} "
                 f"(use --overwrite, or widen --max-epoch)")
    print(f"{len(todo)} run-epochs to evaluate ({len(done)} already done), "
          f"metric={args.metric}, ~{7.5 * len(todo) / 60:.0f} min")

    with open(out_path, "w" if fresh else "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if fresh:
            w.writeheader()
        for i, (run, ep, path) in enumerate(todo, 1):
            t = time.time()
            row = {"run": run, "epoch": ep, "metric": args.metric,
                   **evaluate(path, args.seed, args.n_jobs, args.metric, args.hyp_c)}
            w.writerow(row)
            fh.flush()  # so an interrupted run keeps everything computed so far
            print(f"  [{i:>3}/{len(todo)}] {run:22s} ep{ep:<4} "
                  f"knn={row['loo_knn20']:5.2f} ARI km={row['kmeans_ari']:.3f} "
                  f"spec={row.get('spectral_ari', float('nan')):.3f} "
                  f"dk={row['orc_gap']:.3f} ({time.time() - t:.0f}s)")

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
