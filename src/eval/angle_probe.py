# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Linear probe for the bobblehead rotation angles on frozen features.

A purely supervised, frozen-backbone evaluation: a **linear** map is fit from
the frozen 512-d features to ``(cos θ, sin θ)`` of each rotation angle, and the
predicted angle is read off with ``atan2``.  This measures how linearly
decodable the two rotation factors are from each representation — the simplest
"frozen baseline" comparison across backbones.

Protocol: train / val / test split, ridge regression in closed form, ridge
strength selected on the validation split by mean angular error, metrics
reported on the held-out test split.
"""

from __future__ import annotations

import numpy as np


def _ridge_solve(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge regression. ``X`` is augmented with a bias column."""
    d = X.shape[1]
    reg = lam * np.eye(d)
    reg[-1, -1] = 0.0  # do not regularise the bias term
    return np.linalg.solve(X.T @ X + reg, X.T @ Y)


def _circular_metrics(theta_true: np.ndarray, pred_cossin: np.ndarray) -> dict:
    target = np.stack([np.cos(theta_true), np.sin(theta_true)], axis=1)
    ss_res = ((target - pred_cossin) ** 2).sum()
    ss_tot = ((target - target.mean(0)) ** 2).sum()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    theta_pred = np.arctan2(pred_cossin[:, 1], pred_cossin[:, 0])
    err = np.abs(np.angle(np.exp(1j * (theta_pred - theta_true))))  # wrapped to [0, π]
    return {
        "r2": float(r2),
        "mae_deg": float(np.degrees(err.mean())),
        "median_deg": float(np.degrees(np.median(err))),
    }


def circular_linear_probe(
    X: np.ndarray,
    angles: np.ndarray,
    test_frac: float = 0.2,
    val_frac: float = 0.2,
    ridge_grid: tuple[float, ...] = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3),
    standardize: bool = True,
    seed: int = 0,
) -> tuple[dict, dict]:
    """Fit a linear probe per angle and evaluate on a held-out test split.

    Args:
        X: ``(N, D)`` frozen features.
        angles: ``(N,)`` or ``(N, k)`` true angles in radians.
        test_frac, val_frac: split fractions (of the whole / of the non-test).
        ridge_grid: ridge strengths searched on the validation split.
        standardize: z-score features using train statistics.

    Returns:
        ``(metrics, viz)`` — ``metrics[f"angle{k}"]`` has train/val/test dicts and
        the selected ridge; ``viz[f"angle{k}"]`` holds held-out test true/pred
        angles for plotting.
    """
    X = np.asarray(X, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    if angles.ndim == 1:
        angles = angles[:, None]

    n = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(n * test_frac))
    n_val = int(round((n - n_test) * val_frac))
    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]

    Xtr, Xval, Xte = X[train_idx], X[val_idx], X[test_idx]
    if standardize:
        mu = Xtr.mean(0)
        sd = Xtr.std(0) + 1e-6
        Xtr, Xval, Xte = (Xtr - mu) / sd, (Xval - mu) / sd, (Xte - mu) / sd

    def aug(a):
        return np.concatenate([a, np.ones((a.shape[0], 1))], axis=1)

    Xtr1, Xval1, Xte1 = aug(Xtr), aug(Xval), aug(Xte)

    metrics: dict = {}
    viz: dict = {}
    for k in range(angles.shape[1]):
        th_tr, th_val, th_te = angles[train_idx, k], angles[val_idx, k], angles[test_idx, k]
        Ytr = np.stack([np.cos(th_tr), np.sin(th_tr)], axis=1)

        # Select ridge strength on the validation split (by angular error).
        best_lam, best_val_err, best_W = None, np.inf, None
        for lam in ridge_grid:
            W = _ridge_solve(Xtr1, Ytr, lam)
            val_err = _circular_metrics(th_val, Xval1 @ W)["mae_deg"]
            if val_err < best_val_err:
                best_lam, best_val_err, best_W = lam, val_err, W

        metrics[f"angle{k}"] = {
            "ridge": best_lam,
            "train": _circular_metrics(th_tr, Xtr1 @ best_W),
            "val": _circular_metrics(th_val, Xval1 @ best_W),
            "test": _circular_metrics(th_te, Xte1 @ best_W),
        }
        pred_te = Xte1 @ best_W
        viz[f"angle{k}"] = {
            "true": th_te,
            "pred": np.arctan2(pred_te[:, 1], pred_te[:, 0]),
            "pred_cos": pred_te[:, 0],
            "pred_sin": pred_te[:, 1],
        }

    return metrics, viz


def low_data_curve(
    X: np.ndarray,
    angles: np.ndarray,
    train_sizes: tuple[int, ...] = (25, 50, 100, 200, 400, 800),
    n_seeds: int = 5,
    test_frac: float = 0.2,
    val_frac: float = 0.2,
    ridge_grid: tuple[float, ...] = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3),
    standardize: bool = True,
    base_seed: int = 0,
) -> dict:
    """Probe learning curve: test angular error vs number of training frames.

    A fixed test/val split is held out; for each ``train_size`` we sample that
    many training frames (averaged over ``n_seeds`` resamples), select ridge on
    val, and evaluate on test.  Weaker representations degrade faster as labels
    become scarce — the standard way to separate near-saturated backbones.

    Returns ``{"train_sizes": [...], "angle{k}": {"mae_mean": [...],
    "mae_std": [...], "r2_mean": [...]}}``.
    """
    X = np.asarray(X, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    if angles.ndim == 1:
        angles = angles[:, None]
    n = X.shape[0]

    rng0 = np.random.default_rng(base_seed)
    perm = rng0.permutation(n)
    n_test = int(round(n * test_frac))
    n_val = int(round((n - n_test) * val_frac))
    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    pool_idx = perm[n_test + n_val:]

    if standardize:
        mu = X[pool_idx].mean(0)
        sd = X[pool_idx].std(0) + 1e-6
        Xs = (X - mu) / sd
    else:
        Xs = X

    def aug(a):
        return np.concatenate([a, np.ones((a.shape[0], 1))], axis=1)

    Xval1, Xte1 = aug(Xs[val_idx]), aug(Xs[test_idx])

    out: dict = {"train_sizes": list(train_sizes)}
    for k in range(angles.shape[1]):
        th_val, th_te = angles[val_idx, k], angles[test_idx, k]
        mae_mean, mae_std, r2_mean = [], [], []
        for ts in train_sizes:
            maes, r2s = [], []
            for s in range(n_seeds):
                rng = np.random.default_rng(base_seed + 1 + s)
                take = rng.choice(pool_idx, size=min(ts, len(pool_idx)), replace=False)
                Xtr1 = aug(Xs[take])
                Ytr = np.stack([np.cos(angles[take, k]), np.sin(angles[take, k])], axis=1)
                best_err, best_W = np.inf, None
                for lam in ridge_grid:
                    W = _ridge_solve(Xtr1, Ytr, lam)
                    err = _circular_metrics(th_val, Xval1 @ W)["mae_deg"]
                    if err < best_err:
                        best_err, best_W = err, W
                m = _circular_metrics(th_te, Xte1 @ best_W)
                maes.append(m["mae_deg"])
                r2s.append(m["r2"])
            mae_mean.append(float(np.mean(maes)))
            mae_std.append(float(np.std(maes)))
            r2_mean.append(float(np.mean(r2s)))
        out[f"angle{k}"] = {"mae_mean": mae_mean, "mae_std": mae_std, "r2_mean": r2_mean}
    return out


def pca_2d(X: np.ndarray, standardize: bool = True) -> np.ndarray:
    """2-D PCA projection of features (for qualitative structure plots)."""
    X = np.asarray(X, dtype=np.float64)
    if standardize:
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    X = X - X.mean(0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return X @ vt[:2].T
