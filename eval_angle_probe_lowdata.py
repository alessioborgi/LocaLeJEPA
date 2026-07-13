# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Low-data learning-curve probe on the bobblehead (Yoda) rotation angles.

For every frozen backbone we reuse its cached 512-d features and sweep the number
of *training* frames given to the linear (ridge) angle probe. For each train size
and each of N seeds we: draw a fresh random 20%/16%/rest partition, sample the
train subset from the pool, select ridge strength on the val split, and evaluate
on the held-out 20% test split. We report mean +/- std (over seeds) of test R^2
and mean angular error, per angle, per backbone.

Because a 20% test split is always held out, the training pool caps at ~5182
frames; requested sizes >= that (e.g. 8097 "full dataset") use the whole pool.
The effective train count is recorded per size.

    python eval_angle_probe_lowdata.py
"""

import csv
import json
import os

import numpy as np
import torch

from src.eval.angle_probe import _circular_metrics

OUT_DIR = "logs/eval_angle_probe_lowdata"
SIZES = [5, 10, 20, 50, 100, 500, 800, 2000, 4000, 8097]
N_SEEDS = 10
RIDGE_GRID = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3)
TEST_FRAC = 0.2
VAL_FRAC = 0.2

# name -> cached features (written by the various eval_angle_probe.py runs)
BACKBONES = {
    "svit_susreg":      "logs/eval_angle_probe_full/cache/svit_susreg_features.pt",
    "nvit_sigreg":      "logs/eval_angle_probe_full/cache/nvit_sigreg_features.pt",
    "nvit_subreg":      "logs/eval_angle_probe_full/cache/nvit_subreg_features.pt",
    "vit_sigreg":       "logs/eval_angle_probe_yoda_vit/cache/vit_sigreg_features.pt",
    "vit_subreg":       "logs/eval_angle_probe_yoda_vit/cache/vit_subreg_features.pt",
    # collapse-fixed local backbones (healthy sweep checkpoints, lamb=0.1)
    "localballreg_l0.1_k2": "logs/eval_angle_probe_yoda_local_healthy/cache/localballreg_l0.1_k2_features.pt",
    "localballreg_l0.1_k4": "logs/eval_angle_probe_yoda_local_healthy/cache/localballreg_l0.1_k4_features.pt",
    "localballreg_l0.3_k2": "logs/eval_angle_probe_yoda_local_healthy/cache/localballreg_l0.3_k2_features.pt",
    "localsigreg_l0.1_k4":  "logs/eval_angle_probe_yoda_local_healthy/cache/localsigreg_l0.1_k4_features.pt",
}
ANGLE_LABELS = ["figure 0", "figure 1"]


def _aug(a):
    return np.concatenate([a, np.ones((a.shape[0], 1))], axis=1)


def _fit_eval(Xtr1, Ytr, Xval1, th_val, Xte1, th_te):
    """Ridge with bias, lambda picked on val by angular error; metrics on test.

    Computes A = X^T X and b = X^T Y once and reuses them across the ridge grid.
    """
    d = Xtr1.shape[1]
    A = Xtr1.T @ Xtr1
    b = Xtr1.T @ Ytr
    eye = np.eye(d)
    eye[-1, -1] = 0.0  # do not regularise the bias column
    best_err, best_W = np.inf, None
    for lam in RIDGE_GRID:
        W = np.linalg.solve(A + lam * eye, b)
        err = _circular_metrics(th_val, Xval1 @ W)["mae_deg"]
        if err < best_err:
            best_err, best_W = err, W
    return _circular_metrics(th_te, Xte1 @ best_W)


def run_backbone(X, angles):
    n = X.shape[0]
    n_ang = angles.shape[1]
    # results[size][angle] -> {"r2": [...seeds], "mae": [...seeds]}; plus eff train counts
    res = {ts: [{"r2": [], "mae": []} for _ in range(n_ang)] for ts in SIZES}
    eff = {ts: [] for ts in SIZES}

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        n_test = int(round(n * TEST_FRAC))
        n_val = int(round((n - n_test) * VAL_FRAC))
        test_idx = perm[:n_test]
        val_idx = perm[n_test:n_test + n_val]
        pool_idx = perm[n_test + n_val:]

        Xte = X[test_idx]
        Xval = X[val_idx]
        for ts in SIZES:
            k = min(ts, len(pool_idx))
            train_idx = pool_idx[:k]
            eff[ts].append(k)
            Xtr = X[train_idx]
            mu = Xtr.mean(0)
            sd = Xtr.std(0) + 1e-6
            Xtr1 = _aug((Xtr - mu) / sd)
            Xval1 = _aug((Xval - mu) / sd)
            Xte1 = _aug((Xte - mu) / sd)
            for a in range(n_ang):
                th_tr = angles[train_idx, a]
                Ytr = np.stack([np.cos(th_tr), np.sin(th_tr)], axis=1)
                m = _fit_eval(Xtr1, Ytr, Xval1, angles[val_idx, a], Xte1, angles[test_idx, a])
                res[ts][a]["r2"].append(m["r2"])
                res[ts][a]["mae"].append(m["mae_deg"])
    return res, eff


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary = {"sizes": SIZES, "n_seeds": N_SEEDS, "backbones": {}}

    for name, path in BACKBONES.items():
        blob = torch.load(path, map_location="cpu")
        X, y = blob["X"].numpy().astype(np.float64), blob["y"].numpy().astype(np.float64)
        if y.ndim == 1:
            y = y[:, None]
        res, eff = run_backbone(X, y)
        n_ang = y.shape[1]
        summary["backbones"][name] = {"n_frames": int(X.shape[0]), "sizes": {}}
        print(f"\n=== {name}  (N={X.shape[0]}) ===")
        header = "  size  eff_tr | " + " | ".join(
            f"{ANGLE_LABELS[a]:>8s} R2         MAE" for a in range(n_ang))
        print(header)
        for ts in SIZES:
            eff_tr = int(round(np.mean(eff[ts])))
            entry = {"eff_train": eff_tr, "angles": {}}
            row = f"  {ts:5d}  {eff_tr:6d} |"
            for a in range(n_ang):
                r2 = np.array(res[ts][a]["r2"]); mae = np.array(res[ts][a]["mae"])
                entry["angles"][f"angle{a}"] = {
                    "r2_mean": float(r2.mean()), "r2_std": float(r2.std(ddof=1)),
                    "mae_mean": float(mae.mean()), "mae_std": float(mae.std(ddof=1)),
                }
                row += f"  {r2.mean():.3f}±{r2.std(ddof=1):.3f}  {mae.mean():5.2f}±{mae.std(ddof=1):4.2f} |"
            print(row)
            summary["backbones"][name]["sizes"][ts] = entry

    with open(os.path.join(OUT_DIR, "results_lowdata.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # tidy CSV: one row per (backbone, size, angle)
    with open(os.path.join(OUT_DIR, "results_lowdata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["backbone", "train_size", "eff_train", "angle",
                    "r2_mean", "r2_std", "mae_mean", "mae_std", "n_seeds"])
        for name, b in summary["backbones"].items():
            for ts in SIZES:
                e = b["sizes"][ts]
                for a_key, m in e["angles"].items():
                    w.writerow([name, ts, e["eff_train"], a_key,
                                f'{m["r2_mean"]:.6f}', f'{m["r2_std"]:.6f}',
                                f'{m["mae_mean"]:.4f}', f'{m["mae_std"]:.4f}', N_SEEDS])
    print(f"\nWritten {OUT_DIR}/results_lowdata.json and results_lowdata.csv")


if __name__ == "__main__":
    main()
