# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Compact main-paper version of the torus figures.

One row, five panels, all on the same held-out test frames of a fixed probe
split:

  (a) ground-truth label torus (colour = figurine-0 angle),
  (b) LeJEPA / global SIGReg probe reconstruction, coloured by angular error,
  (c) LocaLeJEPA probe reconstruction on the same colour scale,
  (d) regional advantage map  err(LeJEPA) - err(LocaLeJEPA)  at the full pool,
  (e) the same advantage map with only 100 probe-training frames.

The appendix keeps the full per-backbone / per-budget grids
(plot_yoda_torus_compare.py, plot_yoda_torus_advantage.py); this file only
condenses them.

    python visualisations/plot_yoda_torus_main.py [out.png]
    python visualisations/plot_yoda_torus_main.py --panels [out_dir]
The second form writes the five panels as separate title-free images
(a_label_torus, b_lejepa, c_localejepa, d_advantage_full, e_advantage_100),
each with its own colour bar, for assembly with LaTeX subcaptions; it prints
the per-panel statistics (MAE, mean advantage, win rate) for the captions.
"""
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
FIG_DIR = os.path.join(_HERE, "figures")

from plot_yoda_torus import BACKBONES, draw_torus_surface, style_axes, torus_xyz
from plot_yoda_torus_sweep import fit_at_size
from plot_yoda_torus_advantage import bin_grid, paint_torus, smooth_circular, CMAP

HERO, BASE = "localballreg_l0.1_k2", "vit_sigreg"
FULL, SMALL = 8097, 100
ERR_VMAX = 15.0  # shared colour scale for the two reconstruction panels (deg)


def load(name):
    blob = torch.load(BACKBONES[name], map_location="cpu")
    return blob["X"].numpy().astype(np.float64), blob["y"].numpy().astype(np.float64)


def compute():
    Xh, y = load(HERO)
    Xb, _ = load(BASE)
    fits = {}
    for name, X in ((HERO, Xh), (BASE, Xb)):
        for ts in (FULL, SMALL):
            fits[(name, ts)] = fit_at_size(X, y, ts)
    return y, fits


def draw_panel(fig, ax, cax, kind, y, fits):
    """Draw one of the five panels on ``ax`` (3-d) with its colour bar on ``cax``.
    Returns a dict of statistics for the caption."""
    stats = {}
    if kind == "a":
        draw_torus_surface(ax)
        gx, gy, gz = torus_xyz(y[:, 0], y[:, 1])
        ax.scatter(gx, gy, gz, c=y[:, 0], cmap="twilight", s=4, alpha=0.8, vmin=0, vmax=2 * np.pi, depthshade=False)
        cb = fig.colorbar(cm.ScalarMappable(Normalize(0, 2 * np.pi), "twilight"), cax=cax,
                          orientation="horizontal", ticks=[0, np.pi, 2 * np.pi])
        cb.ax.set_xticklabels(["0", "\u03c0", "2\u03c0"]); cb.set_label("figurine-0 angle", fontsize=8.5, labelpad=2)
        stats["n_frames"] = int(len(y))
    elif kind in ("b", "c"):
        name = BASE if kind == "b" else HERO
        _, pred, err, ntr = fits[(name, FULL)]
        draw_torus_surface(ax)
        px, py, pz = torus_xyz(pred[:, 0], pred[:, 1])
        sc = ax.scatter(px, py, pz, c=err, cmap="viridis_r", s=5, alpha=0.9, vmin=0, vmax=ERR_VMAX, depthshade=False)
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal"); cb.set_label("angular error (\u00b0)", fontsize=8.5, labelpad=2)
        stats["mae"] = float(err.mean()); stats["n_train"] = int(ntr)
    else:
        ts = FULL if kind == "d" else SMALL
        test_idx = fits[(HERO, FULL)][0]
        gt = y[test_idx]
        d = fits[(BASE, ts)][2] - fits[(HERO, ts)][2]
        g = smooth_circular(bin_grid(gt[:, 0], gt[:, 1], d))
        vmax = max(1.0, float(np.percentile(np.abs(g[~np.isnan(g)]), 96)))
        paint_torus(ax, g, vmax)
        cb = fig.colorbar(cm.ScalarMappable(Normalize(-vmax, vmax), CMAP), cax=cax, orientation="horizontal")
        cb.set_label("err(LeJEPA) \u2212 err(LocaLeJEPA) (\u00b0)", fontsize=8.5, labelpad=2)
        stats["mean_adv"] = float(d.mean()); stats["win_frac"] = float((d > 0).mean()); stats["n_train"] = int(fits[(HERO, ts)][3])
    cb.ax.tick_params(labelsize=8, length=2)
    style_axes(ax, "")
    ax.set_xlim(-3.3, 3.3); ax.set_ylim(-3.3, 3.3); ax.set_zlim(-1.3, 1.3)
    return stats


PANELS = (("a", "a_label_torus"), ("b", "b_lejepa"), ("c", "c_localejepa"),
          ("d", "d_advantage_full"), ("e", "e_advantage_100"))


def main_panels(out_dir):
    """Five separate title-free images, one per panel, plus their statistics."""
    y, fits = compute()
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#2a2f37"})
    os.makedirs(out_dir, exist_ok=True)
    for kind, fname in PANELS:
        fig = plt.figure(figsize=(3.4, 3.4))
        gs = fig.add_gridspec(2, 1, height_ratios=(1, 0.05), hspace=0.0, left=0.02, right=0.98, top=0.99, bottom=0.14)
        ax = fig.add_subplot(gs[0, 0], projection="3d")
        cax = fig.add_subplot(gs[1, 0])
        pos = cax.get_position(); cax.set_position([pos.x0 + 0.2 * pos.width, pos.y0, 0.6 * pos.width, pos.height])
        stats = draw_panel(fig, ax, cax, kind, y, fits)
        pos = ax.get_position()
        ax.set_position([pos.x0 - 0.12 * pos.width, pos.y0 - 0.2 * pos.height, 1.24 * pos.width, 1.3 * pos.height])
        base = os.path.join(out_dir, fname)
        fig.savefig(base + ".png", dpi=200, facecolor="white")
        fig.savefig(base + ".pdf", facecolor="white")
        plt.close(fig)
        print(f"wrote {base}.png/.pdf  {stats}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--panels":
        return main_panels(sys.argv[2] if len(sys.argv) > 2 else os.path.join(FIG_DIR, "yoda_torus_main_panels"))
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(FIG_DIR, "yoda_torus_main.png")
    y, fits = compute()
    test_idx = fits[(HERO, FULL)][0]
    gt = y[test_idx]
    theta, phi = gt[:, 0], gt[:, 1]

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#2a2f37"})
    fig = plt.figure(figsize=(16, 3.7))
    gs = fig.add_gridspec(2, 5, height_ratios=(1, 0.05), wspace=0.05, hspace=0.0,
                          left=0.01, right=0.99, top=0.90, bottom=0.17)

    def title(ax, t):
        ax.set_title(t, fontsize=10.5, fontweight="bold", pad=0, color="#2a2f37", y=0.93)
        # zoom the torus so it fills the panel (style_axes leaves wide margins)
        ax.set_xlim(-3.3, 3.3); ax.set_ylim(-3.3, 3.3); ax.set_zlim(-1.3, 1.3)
        pos = ax.get_position()
        ax.set_position([pos.x0 - 0.08 * pos.width, pos.y0 - 0.22 * pos.height, 1.16 * pos.width, 1.3 * pos.height])

    def hbar(col, mappable, label, ticks=None, ticklabels=None):
        cax = fig.add_subplot(gs[1, col])
        pos = cax.get_position()
        cax.set_position([pos.x0 + 0.25 * pos.width, pos.y0, 0.5 * pos.width, pos.height])
        cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=ticks)
        if ticklabels is not None:
            cb.ax.set_xticklabels(ticklabels)
        cb.ax.tick_params(labelsize=8, length=2)
        cb.set_label(label, fontsize=8.5, labelpad=2)

    # (a) ground truth
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    draw_torus_surface(ax)
    gx, gy, gz = torus_xyz(y[:, 0], y[:, 1])
    ax.scatter(gx, gy, gz, c=y[:, 0], cmap="twilight", s=4, alpha=0.8, vmin=0, vmax=2 * np.pi, depthshade=False)
    style_axes(ax, ""); title(ax, "(a) Label torus\n(all 8,097 frames)")
    hbar(0, cm.ScalarMappable(Normalize(0, 2 * np.pi), "twilight"), "figurine-0 angle",
         ticks=[0, np.pi, 2 * np.pi], ticklabels=["0", "\u03c0", "2\u03c0"])

    # (b), (c) reconstructions at the full pool
    for col, name, t in ((1, BASE, "(b) LeJEPA (SIGReg)"), (2, HERO, "(c) LocaLeJEPA")):
        _, pred, err, ntr = fits[(name, FULL)]
        ax = fig.add_subplot(gs[0, col], projection="3d")
        draw_torus_surface(ax)
        px, py, pz = torus_xyz(pred[:, 0], pred[:, 1])
        sc = ax.scatter(px, py, pz, c=err, cmap="viridis_r", s=5, alpha=0.9, vmin=0, vmax=ERR_VMAX, depthshade=False)
        style_axes(ax, ""); title(ax, f"{t}\nheld-out MAE {err.mean():.2f}\u00b0")
        hbar(col, sc, "angular error (\u00b0)")

    # (d), (e) advantage maps
    for col, ts, lab in ((3, FULL, "(d) Advantage, full pool"), (4, SMALL, "(e) Advantage, 100 frames")):
        d = fits[(BASE, ts)][2] - fits[(HERO, ts)][2]
        g = smooth_circular(bin_grid(theta, phi, d))
        vmax = max(1.0, float(np.percentile(np.abs(g[~np.isnan(g)]), 96)))
        ax = fig.add_subplot(gs[0, col], projection="3d")
        paint_torus(ax, g, vmax)
        style_axes(ax, ""); title(ax, f"{lab}\n{d.mean():+.2f}\u00b0 mean, wins {100 * (d > 0).mean():.0f}% of frames")
        hbar(col, cm.ScalarMappable(Normalize(-vmax, vmax), CMAP), "err(LeJEPA) \u2212 err(LocaLeJEPA) (\u00b0)")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=170, facecolor="white")
    fig.savefig(os.path.splitext(out)[0] + ".pdf", facecolor="white")
    plt.close(fig)
    print(f"wrote {out} (+pdf)")


if __name__ == "__main__":
    main()
