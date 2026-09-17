# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Cluster-quality scalars against training epoch, from ``scripts/cluster_trend.py``.

  cluster_trend        one panel per scalar, one line per run, epoch on x.
  cluster_trend_scatter  each structural scalar against k-NN top-1, which asks
                       the question the panels cannot: when a structural measure
                       moves, is it moving *with* accuracy or independently of
                       it? Points from one run are joined in epoch order, so a
                       curve lying along a shared trend line means the structure
                       is a function of accuracy rather than of the objective.

That second figure exists because of the open item in ``paper/k_ablation.tex``:
the neighborhood count moved both probes while every structural descriptor
stayed flat, and nothing connected the two. Flat-in-epoch and
flat-against-accuracy are different claims, and only the scatter tests the one
that matters.

    python visualisations/plot_cluster_trend.py
    python visualisations/plot_cluster_trend.py --data logs/cluster_trend --xmax 100
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from plot_embedding_structure import _family  # noqa: E402  shared colours/labels


def _style(name: str) -> tuple[str, str, str]:
    """Colour, linestyle, label. The shared palette keys on the OBJECTIVE, so the
    matched and paper recipes of one objective collide; here they are two lines
    on one axis (RDMReg's two recipes differ by 32 accuracy points), so the paper
    recipe is dashed."""
    colour, label = _family(name)
    return colour, ("--" if name.endswith("_paper") else "-"), label

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 11,
})
MUTED = "#4a4a4a"

PANELS = [
    ("loo_knn20", "LOO k-NN top-1 (%)", "accuracy"),
    ("kmeans_ari", "k-means ARI", "compactness-based clustering"),
    ("spectral_ari", "spectral ARI", "connectivity-based clustering"),
    ("orc_gap", "Δκ  (within − cross)", "curvature class separation"),
    ("orc_auroc", "κ-AUROC", "class-crossing edge detection"),
    ("edge_cross_frac", "cross-class edge fraction", "k-NN graph leakage (lower = purer)"),
]


def load(path: str) -> dict[str, list[dict]]:
    if not os.path.exists(path):
        sys.exit(f"no metrics at {path} -- run scripts/cluster_trend.py first")
    runs: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            row = {"epoch": int(r["epoch"])}
            for k, v in r.items():
                if k not in ("run", "epoch", "metric"):
                    try:
                        row[k] = float(v)
                    except (TypeError, ValueError):
                        row[k] = None
            runs.setdefault(r["run"], []).append(row)
    return {k: sorted(v, key=lambda r: r["epoch"]) for k, v in runs.items()}


def _series(rows, key):
    pts = [(r["epoch"], r[key]) for r in rows if r.get(key) is not None]
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def trend_figure(runs, out: str, xmax: int | None) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for ax, (key, ylabel, sub) in zip(axes.ravel(), PANELS):
        for name, rows in sorted(runs.items()):
            ep, val = _series(rows, key)
            if not len(ep):
                continue
            colour, ls, label = _style(name)
            ax.plot(ep, val, marker="o", ms=3.5, lw=1.8, color=colour, ls=ls, label=label)
        ax.set_xlabel("training epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(sub, color=MUTED, fontsize=11)
        ax.grid(alpha=0.25)
        if xmax:
            ax.set_xlim(right=xmax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Cluster quality vs training epoch", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    _save(fig, out)


def scatter_figure(runs, out: str) -> None:
    """Structure against accuracy: does structure track the probe, or not?"""
    keys = [p for p in PANELS if p[0] != "loo_knn20"]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4.3))
    for ax, (key, ylabel, _) in zip(np.atleast_1d(axes), keys):
        xs, ys = [], []
        for name, rows in sorted(runs.items()):
            pts = [(r["loo_knn20"], r[key]) for r in rows
                   if r.get(key) is not None and r.get("loo_knn20") is not None]
            if not pts:
                continue
            colour, ls, label = _style(name)
            x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
            ax.plot(x, y, marker="o", ms=4, lw=1.2, alpha=0.9, color=colour, ls=ls, label=label)
            xs.append(x); ys.append(y)
        if xs:
            # A single pooled fit: runs lying ON it have structure explained by
            # accuracy alone; a run sitting off it differs in geometry at matched
            # accuracy, which is the only way a structural claim survives.
            X, Y = np.concatenate(xs), np.concatenate(ys)
            order = np.argsort(X)
            slope, intercept = np.polyfit(X, Y, 1)
            r = float(np.corrcoef(X, Y)[0, 1])
            ax.plot(X[order], slope * X[order] + intercept, "--", color=MUTED, lw=1.2,
                    label=f"pooled fit (r={r:.2f})")
        ax.set_xlabel("LOO k-NN top-1 (%)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Structure against accuracy — points are epochs, joined in order",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    _save(fig, out)


def _save(fig, stem: str) -> None:
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=165, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png / .pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="logs/cluster_trend")
    ap.add_argument("--fig-dir", default=os.path.join(_HERE, "figures", "cluster_trend"))
    ap.add_argument("--xmax", type=int, default=None, help="clip the epoch axis")
    ap.add_argument("--only", default=None, help="comma-separated run names")
    args = ap.parse_args()

    runs = load(os.path.join(args.data, "metrics.csv"))
    if args.only:
        want = {n.strip() for n in args.only.split(",")}
        runs = {k: v for k, v in runs.items() if k in want}
    if not runs:
        sys.exit("no runs to plot")
    span = sorted({r["epoch"] for rows in runs.values() for r in rows})
    print(f"{len(runs)} runs, epochs {span[0]}..{span[-1]} ({len(span)} points)")

    trend_figure(runs, os.path.join(args.fig_dir, "cluster_trend"), args.xmax)
    scatter_figure(runs, os.path.join(args.fig_dir, "cluster_trend_scatter"))


if __name__ == "__main__":
    main()
