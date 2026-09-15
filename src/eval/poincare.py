# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Poincare-ball distances for evaluating hyperbolic backbones.

``src.models.components.hyperbolic_vit.HyperbolicViT`` returns points *inside*
the Poincare ball of curvature ``-c`` (radius ``1/sqrt(c)``), not free vectors.
Every default in the evaluation stack -- k-NN, k-means, the graph the
Ollivier-Ricci curvature is computed on, t-SNE/UMAP -- uses cosine similarity,
which reads only the direction of a feature and discards its norm. The norm is
exactly where a hyperbolic model puts hierarchy depth: distance from the origin
grows without bound as a point approaches the boundary, while its cosine
geometry is unchanged. This module supplies the intrinsic alternative.

Geodesic distance on the ball (Ungar, eq. for the Mobius gyrodistance):

    d_c(u, v) = (1/sqrt(c)) * arcosh(1 + 2c |u - v|^2
                                       / ((1 - c|u|^2) (1 - c|v|^2)))

Two consequences worth keeping in mind when reading results:

* The distance is unbounded, and it blows up near the boundary. Two points at
  norm 0.99/sqrt(c) in *opposite* directions are astronomically further apart
  than two points at norm 0.1/sqrt(c). Cosine sees the same angle in both cases.
* Standardising features per dimension (the z-scoring the Euclidean path uses)
  moves points out of the ball and makes the distance undefined. Use the raw
  cached features with this metric; :func:`check_in_ball` enforces it.

``logmap0`` is provided for the operations that need *vectors* rather than
distances -- k-means centroids and the linear probe. It is the exact inverse of
the ``expmap0`` the encoder ends with, so it recovers the pre-map tangent
representation: radial distances are preserved exactly (up to the conventional
factor 2, irrelevant to any scale-invariant method), while distances between two
off-origin points are only approximated. That approximation is standard practice
for hyperbolic k-means and is flagged wherever it is used.
"""

from __future__ import annotations

import numpy as np

DEFAULT_C = 0.1  # configs/model/encoder/hvit_small_p8.yaml


def ball_radius(c: float = DEFAULT_C) -> float:
    """Radius of the Poincare ball of curvature ``-c``."""
    return float(c) ** -0.5


def check_in_ball(X: np.ndarray, c: float = DEFAULT_C, tol: float = 1e-6) -> None:
    """Raise unless every row of ``X`` lies inside the ball of curvature ``-c``.

    The usual cause of a failure is feeding standardised features, or a
    Euclidean backbone, to the hyperbolic metric -- both produce a silently
    meaningless number rather than an error, so this is checked up front.
    """
    norms = np.linalg.norm(np.asarray(X, dtype=np.float64), axis=1)
    radius = ball_radius(c)
    worst = float(norms.max())
    if worst >= radius * (1.0 + tol):
        raise ValueError(
            f"features are not Poincare-ball points for c={c}: max norm {worst:.4f} "
            f">= ball radius {radius:.4f}. Either these are Euclidean features "
            f"(use --metric cosine), or they were standardised before this call "
            f"(the hyperbolic path must use the raw cached features), or the "
            f"curvature is wrong (--hyp-c must match the encoder config)."
        )


def _sq_norms(X: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", X, X)


def poincare_distances(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    c: float = DEFAULT_C,
    chunk: int = 2048,
) -> np.ndarray:
    """Pairwise geodesic distances, ``(len(X), len(Y))`` -- ``Y=None`` means ``X``.

    Computed in float64 and chunked over rows of ``X``: the (1 - c|x|^2) factors
    approach zero near the boundary, so the ratio inside arcosh is large and
    catastrophically sensitive to rounding in float32.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    Y = X if Y is None else np.ascontiguousarray(Y, dtype=np.float64)
    sqrt_c = float(c) ** 0.5

    # Conformal factors, floored: a point clamped exactly onto the numerical
    # boundary would otherwise divide by zero and produce inf distances.
    gx = np.maximum(1.0 - c * _sq_norms(X), 1e-12)
    gy = np.maximum(1.0 - c * _sq_norms(Y), 1e-12)
    ny = _sq_norms(Y)

    out = np.empty((X.shape[0], Y.shape[0]), dtype=np.float64)
    for start in range(0, X.shape[0], chunk):
        stop = min(start + chunk, X.shape[0])
        block = X[start:stop]
        # NumPy 1.26 on macOS Accelerate raises a spurious "divide by zero
        # encountered in matmul" on ordinary finite inputs; the output is
        # correct. Silenced here only, and the result is checked below.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            gram = block @ Y.T
        sq = _sq_norms(block)[:, None] + ny[None, :] - 2.0 * gram
        np.maximum(sq, 0.0, out=sq)  # kill the -1e-16 the expansion can produce
        arg = 1.0 + 2.0 * c * sq / (gx[start:stop, None] * gy[None, :])
        np.maximum(arg, 1.0, out=arg)  # arcosh domain
        out[start:stop] = np.arccosh(arg) / sqrt_c
    if Y is X:
        np.fill_diagonal(out, 0.0)
    if not np.isfinite(out).all():
        raise FloatingPointError(
            "non-finite Poincare distance -- points are numerically on the ball "
            "boundary; check --hyp-c against the encoder config."
        )
    return out


def logmap0(X: np.ndarray, c: float = DEFAULT_C) -> np.ndarray:
    """Inverse of ``expmap0``: ball points -> tangent vectors at the origin.

    ``|logmap0(x)| = d_c(0, x) / 2``, so radial structure survives exactly and
    only the tangential part is distorted. Use it for methods that need
    coordinates (k-means, a linear head), never as a substitute for
    :func:`poincare_distances` in a pairwise computation.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    sqrt_c = float(c) ** 0.5
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-15)
    scaled = np.clip(sqrt_c * norm, 0.0, 1.0 - 1e-12)
    return np.arctanh(scaled) * X / (sqrt_c * norm)


if __name__ == "__main__":  # smoke test: python -m src.eval.poincare
    import torch

    from src.models.components.hyperbolic import expmap0

    rng = np.random.default_rng(0)
    c = DEFAULT_C
    U = rng.normal(size=(200, 16)) * 0.7
    X = expmap0(torch.from_numpy(U), c=c).numpy()
    check_in_ball(X, c)

    D = poincare_distances(X, c=c)
    assert np.allclose(np.diag(D), 0.0, atol=1e-9)
    assert np.allclose(D, D.T, atol=1e-9), "distance must be symmetric"
    # Triangle inequality on random triples.
    i, j, k = (rng.integers(0, len(X), 3000) for _ in range(3))
    assert (D[i, j] <= D[i, k] + D[k, j] + 1e-9).all(), "triangle inequality violated"
    # Radial consistency: d(0, x) == 2 |logmap0(x)|.
    d0 = poincare_distances(np.zeros((1, X.shape[1])), X, c=c)[0]
    assert np.allclose(d0, 2.0 * np.linalg.norm(logmap0(X, c), axis=1), atol=1e-8)
    # logmap0 inverts the encoder's expmap0.
    assert np.allclose(logmap0(X, c), U, atol=1e-6)
    # The point cosine cannot see: same direction, different depth.
    near, far = np.array([[0.1] + [0.0] * 15]), np.array([[3.1] + [0.0] * 15])
    print(f"d(0, 0.1u) = {poincare_distances(np.zeros_like(near), near, c=c)[0, 0]:.3f}   "
          f"d(0, 3.1u) = {poincare_distances(np.zeros_like(far), far, c=c)[0, 0]:.3f}   "
          "(cosine distance is 0 for both)")
    print("poincare.py: all checks passed")
