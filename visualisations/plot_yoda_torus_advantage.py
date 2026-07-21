# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""How much better is the reference backbone than every other one, on the torus?

Makes the *improvement* the signal instead of yellow-on-yellow absolute error.
The reference ("hero") defaults to localballreg_l0.1_k2 -- the best backbone on
the full-data probe -- and EVERY other frozen backbone is compared by the
per-frame angular-error difference  d = err(other) - err(hero)  on the SAME
held-out test frames. Sibling local variants (dynamic-k, k4, lambda0.3) are
included as competitors, so the plot also shows how the reference fares against
the rest of its own family. Because d is a noisy per-frame red/blue speckle, we
*bin d over torus regions* and paint the torus surface with the regional mean --
so a systematic advantage shows up as a smooth red field rather than
salt-and-pepper. Red = the hero is more accurate in that region; blue = the
competitor is. A per-column symmetric adaptive scale keeps the gap visible at
every training budget. Panels annotate mean d (deg the hero gains) and the
fraction of frames it wins.

    python plot_yoda_torus_advantage.py [out.png]
    python plot_yoda_torus_advantage.py --hero localballreg_l0.1_dynk out.png

Pass --hero to re-anchor the whole figure on a different backbone (e.g. if a new
variant overtakes localballreg_l0.1_k2 on the full-data probe).
"""

import argparse
import os
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

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
from plot_yoda_torus_sweep import fit_at_size

DEFAULT_HERO = "localballreg_l0.1_k2"
SIZES_SHOWN = [10, 50, 100, 500, 8097]
CMAP = "RdBu_r"  # red = positive d = hero better
NBINS = 20       # torus (angle0, angle1) grid resolution for regional averaging


def bin_grid(theta, phi, d, nbins=NBINS):
    """Regional mean of d over an (angle0, angle1) torus grid; NaN for empty cells."""
    edges = np.linspace(0, 2 * np.pi, nbins + 1)
    ti = np.clip(np.digitize(theta, edges) - 1, 0, nbins - 1)
    pj = np.clip(np.digitize(phi, edges) - 1, 0, nbins - 1)
    s = np.zeros((nbins, nbins))
    c = np.zeros((nbins, nbins))
    np.add.at(s, (ti, pj), d)
    np.add.at(c, (ti, pj), 1.0)
    return np.divide(s, c, out=np.full_like(s, np.nan), where=c > 0)


def smooth_circular(grid):
    """One pass of NaN-aware 3x3 mean smoothing, wrapping around the torus."""
    valid = ~np.isnan(grid)
    filled = np.where(valid, grid, 0.0)
    acc = np.zeros_like(filled)
    cnt = np.zeros_like(filled)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            acc += np.roll(np.roll(filled, di, 0), dj, 1)
            cnt += np.roll(np.roll(valid.astype(float), di, 0), dj, 1)
    out = np.divide(acc, cnt, out=np.full_like(acc, np.nan), where=cnt > 0)
    return out


def paint_torus(ax, grid, vmax, nbins=NBINS):
    """Paint the torus surface with the regional-mean grid via a diverging cmap."""
    edges = np.linspace(0, 2 * np.pi, nbins + 1)
    T, P = np.meshgrid(edges, edges, indexing="ij")
    x, y, z = torus_xyz(T, P)
    fc = plt.get_cmap(CMAP)(Normalize(-vmax, vmax)(grid))
    fc[np.isnan(grid)] = (0.92, 0.92, 0.93, 0.25)  # empty regions -> faint gray
    ax.plot_surface(x, y, z, facecolors=fc, rstride=1, cstride=1,
                    linewidth=0, antialiased=True, shade=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?", default=os.path.join(FIG_DIR, "yoda_torus_advantage.png"),
                    help="output .png path (a sibling .pdf is written alongside)")
    ap.add_argument("--hero", default=DEFAULT_HERO,
                    help=f"reference backbone the figure is anchored on (default: {DEFAULT_HERO})")
    args = ap.parse_args()
    out, hero = args.out, args.hero
    if hero not in BACKBONES:
        raise SystemExit(f"unknown --hero {hero!r}; choose from {list(BACKBONES)}")

    data = {}
    for name, path in BACKBONES.items():
        blob = torch.load(path, map_location="cpu")
        X, y = blob["X"].numpy().astype(np.float64), blob["y"].numpy().astype(np.float64)
        if y.ndim == 2 and y.shape[1] >= 2:
            data[name] = (X, y)

    # Per-frame error for every backbone at every size (fixed seed -> shared test set).
    print("computing probe fits ...")
    err = {}   # name -> {ts: per_frame_err}
    full_mae = {}
    test_gt = None
    for name, (X, y) in data.items():
        err[name] = {}
        for ts in SIZES_SHOWN:
            test_idx, _, e, _ = fit_at_size(X, y, ts)
            err[name][ts] = e
            if test_gt is None:
                test_gt = y[test_idx]  # (n_test, 2) ground-truth angles
        full_mae[name] = err[name][SIZES_SHOWN[-1]].mean()
        print(f"  {name:22s} full MAE={full_mae[name]:5.2f}°")

    # Competitors: every backbone except the hero itself -- including the hero's own
    # local-regularizer siblings (dynamic-k, k4, lambda0.3), so the figure shows the
    # within-family comparison too. Sorted best -> worst by full-data error.
    competitors = sorted((n for n in data if n != hero), key=lambda n: full_mae[n])
    if full_mae[hero] > min(full_mae[n] for n in competitors):
        better = min(competitors, key=lambda n: full_mae[n])
        print(f"  NOTE: {better} has a lower full-data MAE ({full_mae[better]:.2f}°) "
              f"than the chosen hero {hero} ({full_mae[hero]:.2f}°); "
              f"consider --hero {better}.")

    theta, phi = test_gt[:, 0], test_gt[:, 1]

    # Precompute the regional-mean grids and per-column symmetric scales.
    grids = {}
    col_vmax = {}
    for ts in SIZES_SHOWN:
        vmaxes = []
        for name in competitors:
            g = smooth_circular(bin_grid(theta, phi, err[name][ts] - err[hero][ts]))
            grids[(name, ts)] = g
            vmaxes.append(np.abs(g[~np.isnan(g)]))
        col_vmax[ts] = max(1.0, float(np.percentile(np.concatenate(vmaxes), 96)))

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#2a2f37"})
    nrow, ncol = len(competitors), len(SIZES_SHOWN)
    fig = plt.figure(figsize=(2.8 * ncol, 2.5 * nrow))

    for r, name in enumerate(competitors):
        for c, ts in enumerate(SIZES_SHOWN):
            d = err[name][ts] - err[hero][ts]  # >0 => hero more accurate
            ax = fig.add_subplot(nrow, ncol, r * ncol + c + 1, projection="3d")
            paint_torus(ax, grids[(name, ts)], col_vmax[ts])
            style_axes(ax, "")
            win = d.mean()
            frac = float((d > 0).mean()) * 100
            ax.set_title(f"{win:+.1f}°  ·  {frac:.0f}% frames", fontsize=10.5,
                         fontweight="bold", color="#b4530f" if win > 0 else "#2f5aa8", pad=-2)
            if r == 0:
                hdr = f"{ts} frames" if ts != 8097 else "full pool"
                ax.text2D(0.5, 1.30, hdr, transform=ax.transAxes, ha="center",
                          fontsize=13, fontweight="bold", color="#141a20")
                ax.text2D(0.5, 1.16, f"scale ±{col_vmax[ts]:.0f}°", transform=ax.transAxes,
                          ha="center", fontsize=9, color="#8a94a0")
            if c == 0:
                ax.text2D(-0.16, 0.5, f"vs {name}", transform=ax.transAxes, va="center",
                          ha="center", rotation=90, fontsize=12, color="#2a2f37")

    # One representative diverging colorbar (scale is per-column; see headers).
    cax = fig.add_axes([0.30, 0.05, 0.42, 0.013])
    cb = fig.colorbar(cm.ScalarMappable(Normalize(-1, 1), CMAP), cax=cax, orientation="horizontal",
                      ticks=[-1, 0, 1])
    cb.ax.set_xticklabels(["competitor better", "tie", f"{hero} better"])
    cb.ax.tick_params(labelsize=10)
    cb.set_label(f"per-frame angular-error advantage  (red = {hero} wins the frame)",
                 fontsize=10, labelpad=6)

    fig.suptitle(
        f"Where {hero} wins across the Torus",
        fontsize=16, fontweight="bold", y=0.985, color="#141a20")
    fig.tight_layout(rect=(0.02, 0.06, 1, 0.95), h_pad=1.4, w_pad=0.4)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    out_pdf = os.path.splitext(out)[0] + ".pdf"
    fig.savefig(out_pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} and {out_pdf}")


if __name__ == "__main__":
    main()
