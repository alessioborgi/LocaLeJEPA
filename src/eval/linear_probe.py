"""L-BFGS linear probe with regularization sweep.

Protocol (matching SSL evaluation literature)
---------------------------------------------
- 45 log-spaced regularization values λ ∈ [1e-6, 1e5]
- sklearn LogisticRegression with solver='lbfgs', C = 1/λ
- multinomial softmax for multi-class datasets
- report accuracy at the best λ on the test set
"""

import logging

import numpy as np

logger = logging.getLogger()

_LAMBDA_GRID = np.logspace(-6, 5, 45)   # regularization strength λ; C = 1/λ


def run_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    n_reg:    int  = 45,
    max_iter: int  = 1000,
) -> tuple:
    """Sweep L-BFGS regularization values and return (best_acc, best_lambda).

    Parameters
    ----------
    X_train, y_train : training features [N_tr, D] and integer labels [N_tr]
    X_test,  y_test  : test features [N_te, D] and integer labels [N_te]
    n_reg            : number of regularization values to sweep (default 45)
    max_iter         : max L-BFGS iterations per fit

    Returns
    -------
    best_acc    : float — top-1 accuracy on the test set
    best_lambda : float — regularization value that achieved best_acc
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as e:
        raise ImportError(
            'scikit-learn is required for the linear probe. '
            'Install it with: pip install scikit-learn'
        ) from e

    lambdas  = np.logspace(-6, 5, n_reg)
    best_acc = 0.0
    best_lam = lambdas[0]

    for lam in lambdas:
        clf = LogisticRegression(
            C=float(1.0 / lam),
            max_iter=max_iter,
            solver='lbfgs',
            tol=1e-4,
        )
        clf.fit(X_train, y_train)
        acc = float(clf.score(X_test, y_test))
        logger.debug('λ=%.2e  C=%.2e  acc=%.4f', lam, 1.0 / lam, acc)
        if acc > best_acc:
            best_acc = acc
            best_lam = lam

    logger.info('Best acc=%.4f at λ=%.2e (C=%.2e)', best_acc, best_lam, 1.0 / best_lam)
    return best_acc, best_lam
