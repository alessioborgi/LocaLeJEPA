# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Learning curves for the seeded k-sweep: does the representation become
separable faster as the BallReg locality k grows?

Reads the per-run CSVs from ``eval_snapshots.py`` and produces:

  1. ``embedor_lc_knn``    -- leave-one-out k-NN top-1 vs epoch, one band per
     group (k2 / k4 / k8 / LeJEPA), seed spread shaded.
  2. ``embedor_lc_orcgap`` -- within/cross curvature gap Delta-kappa vs epoch.
  3. ``embedor_lc_summary``-- area under the k-NN curve (capped at --aulc-epochs)
     and epochs-to-threshold, per group, with seed error bars.

    python visualisations/plot_embedor_learning_curves.py [--metric knn_loo_top1] [--threshold 0.85] [--aulc-epochs 200]
"""

import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))  # np>=2 renamed trapz

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
FIG_DIR = os.path.join(_HERE, "figures")
LC_DIR = os.path.join(_ROOT, "logs", "eval_embedor_imagenette", "learning_curves")

GROUP_ORDER = ["lejepa", "k2", "k4", "k8"]
GROUP_LABEL = {"lejepa": "LeJEPA (SIGReg)", "k2": "BallJEPA k2",
               "k4": "BallJEPA k4", "k8": "BallJEPA k8"}
GROUP_COLOR = {"lejepa": "#8c8c8c", "k2": "#f4a259", "k4": "#e07a1f", "k8": "#2a9d8f"}

_NAME_RE = re.compile(r"^(?:ball_l01_(k\d+)|(lejepa))_s(\d+)$")


def _load() -> dict[str, dict[int, list[dict]]]:
    """{group: {seed: [row, ...]}} from the per-run CSVs."""
    if not os.path.isdir(LC_DIR):
        sys.exit(f"no learning-curve data at {LC_DIR} -- run eval_snapshots.py first")
    out: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for fname in sorted(os.listdir(LC_DIR)):
        if not fname.endswith(".csv") or fname == "learning_curves.csv":
            continue
        m = _NAME_RE.match(fname[:-4])
        if not m:
            continue
        group = m.group(1) or m.group(2)
        seed = int(m.group(3))
        import csv
        with open(os.path.join(LC_DIR, fname), encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                out[group][seed].append(
                    {"epoch": int(r["epoch"]),
                     **{k: float(r[k]) for k in ("knn_loo_top1", "orc_gap", "orc_within",
                                                 "orc_cross", "edge_cross_frac")}}
                )
    for g in out:
        for s in out[g]:
            out[g][s].sort(key=lambda d: d["epoch"])
    return out


def _grid(rows_by_seed: dict[int, list[dict]], metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Common epoch grid + (seeds, epochs) metric array, NaN where a seed lacks a point."""
    epochs = sorted({d["epoch"] for rows in rows_by_seed.values() for d in rows})
    ep = np.array(epochs, float)
    mat = np.full((len(rows_by_seed), len(epochs)), np.nan)
    for si, seed in enumerate(sorted(rows_by_seed)):
        by_ep = {d["epoch"]: d[metric] for d in rows_by_seed[seed]}
        for ei, e in enumerate(epochs):
            if e in by_ep:
                mat[si, ei] = by_ep[e]
    return ep, mat


def _curve_figure(data, metric: str, ylabel: str, title: str, out: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for g in GROUP_ORDER:
        if g not in data:
            continue
        ep, mat = _grid(data[g], metric)
        mean = np.nanmean(mat, axis=0)
        lo, hi = np.nanmin(mat, axis=0), np.nanmax(mat, axis=0)
        ax.fill_between(ep, lo, hi, color=GROUP_COLOR[g], alpha=0.18, linewidth=0)
        ax.plot(ep, mean, color=GROUP_COLOR[g], lw=2, label=f"{GROUP_LABEL[g]}  (n={mat.shape[0]})")
    ax.set_xlabel("epoch", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, out)


def _summary_figure(data, threshold: float, aulc_epochs: int, out: str) -> None:
    groups = [g for g in GROUP_ORDER if g in data]
    aulc, ttt = defaultdict(list), defaultdict(list)
    for g in groups:
        for seed, rows in data[g].items():
            ep = np.array([r["epoch"] for r in rows], float)
            v = np.array([r["knn_loo_top1"] for r in rows], float)
            mask = ep <= aulc_epochs
            if mask.sum() >= 2:
                aulc[g].append(_trapz(v[mask], ep[mask]) / (ep[mask][-1] - ep[mask][0]))
            hit = ep[v >= threshold]
            ttt[g].append(float(hit[0]) if hit.size else np.nan)

    x = np.arange(len(groups))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, vals, ttl, ylab in [
        (axes[0], aulc, f"mean k-NN top-1 over epochs 0-{aulc_epochs}\n(higher = learns faster)", "normalised AULC"),
        (axes[1], ttt, f"epochs to k-NN top-1 ≥ {threshold}\n(lower = learns faster)", "epochs"),
    ]:
        means = [np.nanmean(vals[g]) for g in groups]
        errs = [np.nanstd(vals[g]) for g in groups]
        ax.bar(x, means, yerr=errs, capsize=4,
               color=[GROUP_COLOR[g] for g in groups])
        for xi, g in zip(x, groups):
            ax.scatter([xi] * len(vals[g]), vals[g], color="#222", s=14, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([GROUP_LABEL[g].replace(" ", "\n", 1) for g in groups], fontsize=9)
        ax.set_title(ttl, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylab, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("k-sweep learning speed — dots = seeds, bars = mean ± sd",
                 fontsize=13, fontweight="bold")
    _save(fig, out)

    print(f"\n{'group':14s} {'AULC(0-'+str(aulc_epochs)+')':>14s} {'epochs>=%.2f' % threshold:>14s}")
    for g in groups:
        print(f"{GROUP_LABEL[g]:14s} {np.nanmean(aulc[g]):14.4f} "
              f"{np.nanmean(ttt[g]):14.1f}  (seeds: {[round(x, 1) for x in ttt[g]]})")


def _save(fig, out: str) -> None:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    fig.savefig(os.path.splitext(out)[0] + ".pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--aulc-epochs", type=int, default=200)
    args = ap.parse_args()

    data = _load()
    if not data:
        sys.exit(f"no parseable run CSVs in {LC_DIR}")
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#2a2f37"})

    _curve_figure(data, "knn_loo_top1", "leave-one-out k-NN top-1",
                  "Representation separability vs epoch — seeded k-sweep",
                  os.path.join(FIG_DIR, "embedor_lc_knn.png"))
    _curve_figure(data, "orc_gap", "within − cross curvature gap  Δκ",
                  "Curvature class-separation vs epoch — seeded k-sweep",
                  os.path.join(FIG_DIR, "embedor_lc_orcgap.png"))
    _summary_figure(data, args.threshold, args.aulc_epochs,
                    os.path.join(FIG_DIR, "embedor_lc_summary.png"))


if __name__ == "__main__":
    main()
