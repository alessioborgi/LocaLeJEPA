# Copyright (c) 2025 Alessio Borgioli
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""L-BFGS linear probe with regularization sweep.

Protocol (matching SSL evaluation literature, Yu et al. NeurIPS 2024)
----------------------------------------------------------------------
- Carve a validation split (val_frac=0.2) from the training features by
  seeded random selection; used only for the λ sweep.
- 45 log-spaced regularization values λ ∈ [1e-6, 1e5]
- sklearn LogisticRegression with solver='lbfgs', C = 1/λ
- Select the best λ on the validation split.
- Retrain with the best λ on the full training set (train + val).
- Report accuracy at best λ on the held-out test set.
"""

import logging

import numpy as np

logger = logging.getLogger()


def run_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    n_reg:    int   = 45,
    max_iter: int   = 5000,
    val_frac: float = 0.2,
    seed:     int   = 0,
) -> tuple:
    """Sweep λ on a val split, retrain on full train, return (best_acc, best_lambda).

    Parameters
    ----------
    X_train, y_train : training features [N_tr, D] and integer labels [N_tr]
    X_test,  y_test  : test features [N_te, D] and integer labels [N_te]
    n_reg            : number of regularization values to sweep (default 45)
    max_iter         : max L-BFGS iterations per fit (default 5000)
    val_frac         : fraction of training set held out for the λ sweep (default 0.2)
    seed             : random seed for the train/val split

    Returns
    -------
    best_acc    : float — top-1 accuracy on the test set (retrained with best λ)
    best_lambda : float — regularization value that achieved best val accuracy
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as e:
        raise ImportError(
            'scikit-learn is required for the linear probe. '
            'Install it with: pip install scikit-learn'
        ) from e

    rng   = np.random.default_rng(seed)
    n     = len(X_train)
    n_val = max(1, int(n * val_frac))
    val_idx = rng.choice(n, size=n_val, replace=False)
    tr_mask = np.ones(n, dtype=bool)
    tr_mask[val_idx] = False

    X_tr, y_tr   = X_train[tr_mask],  y_train[tr_mask]
    X_val, y_val = X_train[val_idx],   y_train[val_idx]

    lambdas      = np.logspace(-6, 5, n_reg)
    best_val_acc = 0.0
    best_lam     = lambdas[0]

    for lam in lambdas:
        clf = LogisticRegression(
            C=float(1.0 / lam),
            max_iter=max_iter,
            solver='lbfgs',
            tol=1e-4,
        )
        clf.fit(X_tr, y_tr)
        val_acc = float(clf.score(X_val, y_val))
        logger.debug('λ=%.2e  val_acc=%.4f', lam, val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_lam     = lam

    logger.info('Best val_acc=%.4f at λ=%.2e — retraining on full train set',
                best_val_acc, best_lam)

    clf_final = LogisticRegression(
        C=float(1.0 / best_lam),
        max_iter=max_iter,
        solver='lbfgs',
        tol=1e-4,
    )
    clf_final.fit(X_train, y_train)
    best_acc = float(clf_final.score(X_test, y_test))

    logger.info('Test acc=%.4f at λ=%.2e (C=%.2e)', best_acc, best_lam, 1.0 / best_lam)
    return best_acc, best_lam
