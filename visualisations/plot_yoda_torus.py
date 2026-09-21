# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Visualize the Yoda rotation-angle label space as the 2-torus it actually is.

Each frame carries two figurine rotation angles (angle0, angle1) in [0, 2pi), so
the joint label space is a 2-torus T^2. We embed it in R^3 and show:

  (1) the ground-truth data on the torus (one point per frame), and
  (2) how a frozen JEPA backbone's ridge angle-probe reconstructs those angles
      on held-out frames, colored by angular error.

    python plot_yoda_torus.py [backbone_name] [out.png]
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

# This file lives in <repo>/visualisations/; anchor to the repo root so feature
# caches and `src` imports resolve regardless of the caller's working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
FIG_DIR = os.path.join(_HERE, "figures")  # all rendered figures land here

from src.eval.angle_probe import _circular_metrics

# Reuse the same feature caches the low-data probe reads (repo-root relative).
# Where the features are: the index `src/eval_angle_probe.py` writes beside its results, keyed
# by run name. Point YODA_EVAL_DIR at another eval folder to draw from it. The previous version
# hardcoded eleven pre-migration cache filenames that no longer exist (campaign plan C5c).
YODA_EVAL_DIR = os.environ.get("YODA_EVAL_DIR", os.path.join(_ROOT, "logs", "eval_angle_probe_torus_campaign"))


def load_index(eval_dir: str = YODA_EVAL_DIR) -> dict[str, str]:
    """run name -> absolute cache path, from ``features_index.json``; empty if the eval has not run."""
    import json

    path = os.path.join(eval_dir, "features_index.json")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        index = json.load(handle)
    return {name: (entry["cache"] if os.path.isabs(entry["cache"]) else os.path.join(_ROOT, entry["cache"]))
            for name, entry in index.items()}


BACKBONES = load_index()

R_MAJOR, R_MINOR = 3.0, 1.0  # ring radius (angle1) and tube radius (angle0)
TEST_FRAC, VAL_FRAC = 0.2, 0.2
RIDGE_GRID = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)


def torus_xyz(theta, phi, r_major=R_MAJOR, r_minor=R_MINOR):
    """Embed torus angles into R^3. theta = tube (angle0), phi = ring (angle1)."""
    x = (r_major + r_minor * np.cos(theta)) * np.cos(phi)
    y = (r_major + r_minor * np.cos(theta)) * np.sin(phi)
    z = r_minor * np.sin(theta)
    return x, y, z


def _aug(a):
    return np.concatenate([a, np.ones((a.shape[0], 1))], axis=1)


def fit_probe(Xtr1, th_tr, Xval1, th_val):
    """Ridge (cos,sin) probe with bias; lambda picked on val by angular error."""
    Ytr = np.stack([np.cos(th_tr), np.sin(th_tr)], axis=1)
    d = Xtr1.shape[1]
    A = Xtr1.T @ Xtr1
    b = Xtr1.T @ Ytr
    eye = np.eye(d)
    eye[-1, -1] = 0.0
    best_err, best_W = np.inf, None
    for lam in RIDGE_GRID:
        W = np.linalg.solve(A + lam * eye, b)
        err = _circular_metrics(th_val, Xval1 @ W)["mae_deg"]
        if err < best_err:
            best_err, best_W = err, W
    return best_W


def predict_angles(X, y, seed=0):
    """Fit both angle probes on a train split; return (test_idx, pred_angles, gt)."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(n * TEST_FRAC))
    n_val = int(round((n - n_test) * VAL_FRAC))
    test_idx, val_idx, tr_idx = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]

    mu, sd = X[tr_idx].mean(0), X[tr_idx].std(0) + 1e-6
    Xtr1, Xval1, Xte1 = _aug((X[tr_idx] - mu) / sd), _aug((X[val_idx] - mu) / sd), _aug((X[test_idx] - mu) / sd)

    preds, errs = [], []
    for a in range(y.shape[1]):
        W = fit_probe(Xtr1, y[tr_idx, a], Xval1, y[val_idx, a])
        cs = Xte1 @ W
        preds.append(np.arctan2(cs[:, 1], cs[:, 0]) % (2 * np.pi))
        errs.append(np.degrees(np.abs(np.angle(np.exp(1j * (preds[-1] - y[test_idx, a])))) ) )
    return test_idx, np.stack(preds, 1), np.stack(errs, 1)


def draw_torus_surface(ax):
    u = np.linspace(0, 2 * np.pi, 90)
    v = np.linspace(0, 2 * np.pi, 90)
    u, v = np.meshgrid(u, v)
    x, y, z = torus_xyz(u, v)
    ax.plot_surface(x, y, z, rstride=3, cstride=3, color="#c9ced6",
                    alpha=0.10, linewidth=0, antialiased=True, shade=False)
    ax.plot_wireframe(x, y, z, rstride=9, cstride=9, color="#aeb6c2",
                      linewidth=0.35, alpha=0.35)


def style_axes(ax, title):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=6, color="#2a2f37")
    ax.set_box_aspect((1, 1, 0.42))
    ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2); ax.set_zlim(-2.6, 2.6)
    ax.view_init(elev=34, azim=-58)
    ax.set_axis_off()


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "vit_local_l01_k2_s0"
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(FIG_DIR, "yoda_torus.png")
    if name not in BACKBONES:
        raise SystemExit(f"unknown backbone {name!r}; choose from {list(BACKBONES)} "
                         f"(index: {os.path.join(YODA_EVAL_DIR, 'features_index.json')})")

    blob = torch.load(BACKBONES[name], map_location="cpu")
    X, y = blob["X"].numpy().astype(np.float64), blob["y"].numpy().astype(np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if y.shape[1] < 2:
        raise SystemExit("need two angles to draw a 2-torus")

    th_all, phi_all = y[:, 0], y[:, 1]
    test_idx, pred, err = predict_angles(X, y)
    mae = err.mean()

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#2a2f37"})
    fig = plt.figure(figsize=(16, 6.2))

    # Panel 1: ground-truth torus, colored by figurine-0 angle (tube angle).
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    draw_torus_surface(ax1)
    gx, gy, gz = torus_xyz(th_all, phi_all)
    p1 = ax1.scatter(gx, gy, gz, c=th_all, cmap="twilight", s=6, alpha=0.8,
                     vmin=0, vmax=2 * np.pi, depthshade=False)
    style_axes(ax1, "Ground truth · figurine 0")
    cb1 = fig.colorbar(cm.ScalarMappable(Normalize(0, 2 * np.pi), "twilight"),
                       ax=ax1, fraction=0.03, pad=0.02, ticks=[0, np.pi, 2 * np.pi])
    cb1.ax.set_yticklabels(["0", "π", "2π"]); cb1.ax.tick_params(labelsize=9)

    # Panel 2: ground-truth torus, colored by figurine-1 angle (ring angle).
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    draw_torus_surface(ax2)
    ax2.scatter(gx, gy, gz, c=phi_all, cmap="hsv", s=6, alpha=0.8,
                vmin=0, vmax=2 * np.pi, depthshade=False)
    style_axes(ax2, "Ground truth · figurine 1")
    cb2 = fig.colorbar(cm.ScalarMappable(Normalize(0, 2 * np.pi), "hsv"),
                       ax=ax2, fraction=0.03, pad=0.02, ticks=[0, np.pi, 2 * np.pi])
    cb2.ax.set_yticklabels(["0", "π", "2π"]); cb2.ax.tick_params(labelsize=9)

    # Panel 3: frozen-backbone probe reconstruction on held-out frames, by error.
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    draw_torus_surface(ax3)
    px, py, pz = torus_xyz(pred[:, 0], pred[:, 1])
    per_frame_err = err.mean(1)
    p3 = ax3.scatter(px, py, pz, c=per_frame_err, cmap="viridis_r", s=8, alpha=0.9,
                     vmin=0, vmax=max(20.0, np.percentile(per_frame_err, 95)), depthshade=False)
    # Drop the "_l<lambda>_k<clusters>" suffix from the display label; the exact
    # hyperparameters live in the LaTeX caption instead.
    style_axes(ax3, f"Probe reconstruction · {name.split('_l')[0]}")
    cb3 = fig.colorbar(p3, ax=ax3, fraction=0.03, pad=0.02)
    cb3.set_label("angular error (deg)", fontsize=9); cb3.ax.tick_params(labelsize=9)

    fig.suptitle("The rotation-angle label space is a 2-torus",
                 fontsize=16, fontweight="bold", y=0.98, color="#141a20")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
    # Always also emit a vector PDF alongside the raster PNG.
    out_pdf = os.path.splitext(out)[0] + ".pdf"
    fig.savefig(out_pdf, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} and {out_pdf}  (backbone={name}, test MAE={mae:.2f}°, n_test={len(test_idx)})")


if __name__ == "__main__":
    main()
