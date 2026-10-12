# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Torus reconstruction across backbones AND training sizes.

Grid of tori: one row per frozen backbone (sorted best -> worst by full-data
angular error), one column per training-set size. Every cell fits the ridge
angle-probe on that many frames and renders the *same* held-out test frames on
the 2-torus, colored by angular error on a shared scale. Reveals whether a
backbone (e.g. localballreg) dominates regardless of how much probe data it gets.

    python plot_yoda_torus_compare.py [out.png]
"""

import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# This file lives in <repo>/visualisations/; anchor to the repo root so `src`
# and sibling-module imports resolve regardless of the caller's working dir.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
FIG_DIR = os.path.join(_HERE, "figures")

from plot_yoda_torus import BACKBONES, style_axes, torus_xyz
from plot_yoda_torus_sweep import ERR_VMAX, fit_at_size, light_torus_surface

SIZES_SHOWN = [10, 50, 100, 500, 8097]
HIGHLIGHT = "localballreg_l0.1_k2"


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(FIG_DIR, "yoda_torus_compare.png")

    # Load every backbone's cached features once.
    data = {}
    for name, path in BACKBONES.items():
        blob = torch.load(path, map_location="cpu")
        X, y = blob["X"].numpy().astype(np.float64), blob["y"].numpy().astype(np.float64)
        if y.ndim == 1 or y.shape[1] < 2:
            continue
        data[name] = (X, y)

    # Precompute MAE for every (backbone, size); sort rows by full-data MAE.
    print("computing probe fits ...")
    results = {}  # name -> {ts: (pred, err, eff, mae)}
    full_mae = {}
    for name, (X, y) in data.items():
        results[name] = {}
        for ts in SIZES_SHOWN:
            _, pred, err, eff = fit_at_size(X, y, ts)
            results[name][ts] = (pred, err, eff, err.mean())
        full_mae[name] = results[name][SIZES_SHOWN[-1]][3]
        print(f"  {name:22s} full-data MAE={full_mae[name]:5.2f}°")
    order = sorted(data, key=lambda n: full_mae[n])

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#2a2f37"})
    nrow, ncol = len(order), len(SIZES_SHOWN)
    fig = plt.figure(figsize=(2.7 * ncol, 2.5 * nrow))

    scat = None
    for r, name in enumerate(order):
        is_hl = name == HIGHLIGHT
        X, y = data[name]
        for c, ts in enumerate(SIZES_SHOWN):
            pred, err, eff, mae = results[name][ts]
            ax = fig.add_subplot(nrow, ncol, r * ncol + c + 1, projection="3d")
            light_torus_surface(ax)
            px, py, pz = torus_xyz(pred[:, 0], pred[:, 1])
            scat = ax.scatter(px, py, pz, c=err, cmap="viridis_r", s=4, alpha=0.85,
                              vmin=0, vmax=ERR_VMAX, depthshade=False)
            style_axes(ax, "")
            tcol = "#b4530f" if is_hl else "#2a2f37"
            ax.set_title(f"MAE {mae:.1f}°", fontsize=11, fontweight="bold",
                         color=tcol, pad=-2)
            # Column headers (training size) on the top row.
            if r == 0:
                hdr = f"{ts} frames" if ts != 8097 else "full pool"
                ax.text2D(0.5, 1.28, hdr, transform=ax.transAxes, ha="center",
                          fontsize=13, fontweight="bold", color="#141a20")
            # Row labels (backbone name) on the first column.
            if c == 0:
                ax.text2D(-0.14, 0.5, name, transform=ax.transAxes, va="center",
                          ha="center", rotation=90, fontsize=12,
                          fontweight="bold" if is_hl else "normal", color=tcol)

    cax = fig.add_axes([0.35, 0.055, 0.32, 0.012])
    cb = fig.colorbar(scat, cax=cax, orientation="horizontal")
    cb.set_label("per-frame angular error (deg) — held-out test  ·  yellow = accurate",
                 fontsize=10)
    cb.ax.tick_params(labelsize=9)

    # No figure title: the LaTeX caption names the figure. The per-panel MAE titles and
    # the row/column labels stay, since they carry data.
    fig.tight_layout(rect=(0.02, 0.07, 1, 0.99), h_pad=1.4, w_pad=0.4)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    out_pdf = os.path.splitext(out)[0] + ".pdf"
    fig.savefig(out_pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} and {out_pdf}")


if __name__ == "__main__":
    main()
