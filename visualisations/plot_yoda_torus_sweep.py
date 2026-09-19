# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Torus reconstruction vs. training-set size for a frozen JEPA backbone.

Fits the ridge angle-probe on a growing number of training frames (5, 10, 20,
..., full pool) -- exactly the low-data sweep of src/eval_angle_probe_lowdata.py --
and, for each size, renders the reconstructed (angle0, angle1) of the *same*
held-out test frames on the 2-torus, colored by angular error. Watch the torus
snap from noise into a clean donut as the probe gets more data.

    python plot_yoda_torus_sweep.py [backbone_name] [out.png]
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

from plot_yoda_torus import (
    BACKBONES,
    RIDGE_GRID,
    TEST_FRAC,
    VAL_FRAC,
    _aug,
    fit_probe,
    style_axes,
    torus_xyz,
)

SIZES = [5, 10, 20, 50, 100, 500, 800, 2000, 4000, 8097]
ERR_VMAX = 90.0  # deg; a uniform-random probe averages ~90 deg circular error


def light_torus_surface(ax):
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, 2 * np.pi, 60)
    u, v = np.meshgrid(u, v)
    x, y, z = torus_xyz(u, v)
    ax.plot_surface(x, y, z, rstride=4, cstride=4, color="#c9ced6",
                    alpha=0.07, linewidth=0, antialiased=True, shade=False)


def fit_at_size(X, y, k, seed=0):
    """Fit both angle probes on the first k pool frames; predict on the test split.

    Uses the same fixed seed for every k, so the test/val split is held constant
    and only the amount of training data grows (matching the low-data protocol).
    """
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(n * TEST_FRAC))
    n_val = int(round((n - n_test) * VAL_FRAC))
    test_idx, val_idx, pool_idx = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
    tr_idx = pool_idx[:min(k, len(pool_idx))]

    mu, sd = X[tr_idx].mean(0), X[tr_idx].std(0) + 1e-6
    Xtr1, Xval1, Xte1 = _aug((X[tr_idx] - mu) / sd), _aug((X[val_idx] - mu) / sd), _aug((X[test_idx] - mu) / sd)

    preds, errs = [], []
    for a in range(y.shape[1]):
        W = fit_probe(Xtr1, y[tr_idx, a], Xval1, y[val_idx, a])
        cs = Xte1 @ W
        th = np.arctan2(cs[:, 1], cs[:, 0]) % (2 * np.pi)
        preds.append(th)
        errs.append(np.degrees(np.abs(np.angle(np.exp(1j * (th - y[test_idx, a]))))))
    return test_idx, np.stack(preds, 1), np.stack(errs, 1).mean(1), len(tr_idx)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "localballreg_l0.1_k2"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(FIG_DIR, "yoda_torus_sweep.png")
    if name not in BACKBONES:
        raise SystemExit(f"unknown backbone {name!r}; choose from {list(BACKBONES)}")

    blob = torch.load(BACKBONES[name], map_location="cpu")
    X, y = blob["X"].numpy().astype(np.float64), blob["y"].numpy().astype(np.float64)
    if y.ndim == 1 or y.shape[1] < 2:
        raise SystemExit("need two angles to draw a 2-torus")

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#2a2f37"})
    ncol = 5
    nrow = int(np.ceil(len(SIZES) / ncol))
    fig = plt.figure(figsize=(3.15 * ncol, 3.2 * nrow))

    scat = None
    for i, ts in enumerate(SIZES):
        test_idx, pred, err, eff = fit_at_size(X, y, ts)
        mae = err.mean()
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
        light_torus_surface(ax)
        px, py, pz = torus_xyz(pred[:, 0], pred[:, 1])
        scat = ax.scatter(px, py, pz, c=err, cmap="viridis_r", s=5, alpha=0.85,
                          vmin=0, vmax=ERR_VMAX, depthshade=False)
        label = f"{eff} train frames" if eff == ts else f"{eff} frames (full pool)"
        style_axes(ax, f"{label}\nMAE {mae:.1f}°")
        print(f"  n={ts:>5} eff={eff:>5}  MAE={mae:6.2f}°")

    cax = fig.add_axes([0.35, 0.05, 0.32, 0.018])
    cb = fig.colorbar(scat, cax=cax, orientation="horizontal")
    cb.set_label("per-frame angular error (deg) — held-out test", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    fig.suptitle(
        f"Torus reconstruction sharpens with training data — {name}\n"
        "ridge angle-probe on frozen features · fixed held-out test set · growing train size",
        fontsize=14, fontweight="bold", y=0.99, color="#141a20")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    out_pdf = os.path.splitext(out)[0] + ".pdf"
    fig.savefig(out_pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} and {out_pdf}")


if __name__ == "__main__":
    main()
