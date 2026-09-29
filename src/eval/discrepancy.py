# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""How close does a FROZEN model actually get to the distribution its loss was driving it to?

The regularizers drive the projected embedding toward a target law -- an isotropic Gaussian for
SIGReg, the uniform distribution on a Euclidean ball for LocaLeReg -- but nothing has ever measured
whether a trained model *arrives* there. Every existing eval-time measure is about downstream
accuracy or about rank (`src/eval/rankme.py`), and RankMe saturates on the 16-d projector output
(`results/RankMe_results.md:152-155`), so it cannot separate two non-collapsed runs.

This module closes that gap by evaluating the SAME sketched Epps-Pulley statistic the loss uses, on
cached frozen features, as a diagnostic rather than an objective. It is the premise test for the
capacity campaign (decision 24): if a ViT-S/8 already reaches the target under the *global*
statistic, then the global constraint was satisfiable and locality was never load-bearing.

**No new mathematics is written here.** Every characteristic function, the radius convention and the
statistic itself come from `src/losses.py` unchanged. What this module adds is the three corrections
that turn a training loss into a comparable metric:

1. **Two numbers, because there is no single N-independent one.** `_sketch_test` ends
   `* z.size(-2)`, which is the Epps-Pulley normalisation: it makes the value a *test statistic*,
   with a fixed distribution when the fit is good and divergence when it is not. Measured here at
   d=16 over N = 512 to 8192, the raw statistic stays near 1.0 for samples drawn FROM the ball while
   growing 1.09 -> 3.98 for Gaussian samples scored against the ball. Divided by N it is instead an
   *effect size*: it falls toward 0 when the fit is good and converges to the true squared
   characteristic-function distance when it is not. So this module reports **both** --
   ``statistic`` for "is the mismatch detectable at this sample size", ``distance`` for "how far
   away is it" -- rather than silently picking one. Neither is comparable across different N; both
   are comparable across arms at equal N, which is how they are used.
2. **A pinned seed.** `_sketch_test` draws fresh random slice directions on every call, so two
   unseeded calls on the same features disagree. Every call here is seeded, and `num_slices`
   defaults well above the training default of 256.
3. **An explicit centring choice**, because centring lives in the wrappers rather than the
   statistic: `ballreg_loss` defaults to `center=False`, `local_ballreg_loss` bakes in
   `center=True`, and `local_sigreg_loss` deliberately does not centre. A metric that inherited
   whichever convention its arm happened to use would not be comparable across arms, so the
   convention is a parameter and is reported alongside the value.

**What this does NOT measure.** The training loss is computed on augmented views, with
per-neighbourhood centring, on the training split. This is computed on one clean view of the
evaluation split. It answers "does the frozen projector output match the target law", which is the
question the campaign needs -- but it is not a re-evaluation of the training objective, and anything
that reports it must say so.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import torch

from src.losses import (
    _ball_d_cf,
    _gaussian_cf,
    _resolve_ball_radius,
    _sketch_test,
    local_loss,
)

TARGETS = ("ball", "gaussian")
Target = Literal["ball", "gaussian"]

# Above the training default of 256: this is a measurement, not a gradient signal, so the slice
# count costs nothing here and a noisy metric would be worse than a slow one.
DEFAULT_NUM_SLICES = 1024
DEFAULT_KNOTS = 17
DEFAULT_T_MAX = 3.0
DEFAULT_N_QUAD = 64
DEFAULT_SEED = 0


def _target_phi(
    target: Target, d: int, radius: float | None, n_quad: int
) -> tuple[Callable[[torch.Tensor], torch.Tensor], float | None]:
    """Resolve a target name to its characteristic function, and the radius it used.

    The ball radius follows `src/losses.py`'s convention exactly: `radius=None` resolves to
    `sqrt(d + 2)`, the value at which the uniform ball has identity covariance, so the ball and the
    isotropic Gaussian are normalised the same way and their discrepancies are comparable.
    """
    if target == "gaussian":
        return _gaussian_cf, None
    if target == "ball":
        R = _resolve_ball_radius(d, radius)
        return (lambda t: _ball_d_cf(t, d=d, radius=R, n_quad=n_quad)), R
    raise ValueError(f"target must be one of {TARGETS}, got {target!r}")


def discrepancy(
    Z: torch.Tensor,
    target: Target,
    *,
    center: bool = False,
    radius: float | None = None,
    knots: int = DEFAULT_KNOTS,
    num_slices: int = DEFAULT_NUM_SLICES,
    t_max: float = DEFAULT_T_MAX,
    n_quad: int = DEFAULT_N_QUAD,
    seed: int = DEFAULT_SEED,
) -> dict[str, float]:
    """How far is ``Z`` from ``target``, under the sliced Epps-Pulley statistic?

    Parameters
    ----------
    Z : (N, D) frozen projected features. A 2-D matrix is what the feature cache holds; note
        `ballreg_loss` itself refuses anything but (V, N, D), which is why this module calls
        `_sketch_test` directly rather than routing through the loss wrappers.
    target : "ball" or "gaussian".
    center : subtract the sample mean before testing, constraining shape and leaving location free.
    radius : ball radius; None resolves to sqrt(D + 2) (identity covariance).
    seed : pinned immediately before the slice draw, so the value is reproducible.

    Returns
    -------
    ``{"statistic", "distance", "n_samples"}``.

    ``statistic`` is the value `src/losses.py` would compute -- the same scale as the training loss,
    N-normalised in the Epps-Pulley sense, so it stays put under a good fit and diverges under a bad
    one. ``distance`` is ``statistic / n_samples``, an estimate of the squared
    characteristic-function distance, which goes to zero for a perfect fit.

    **Read them at equal N.** Neither is comparable across sample counts: under a good fit the
    statistic is flat while the distance falls as 1/N, and under a mismatch the statistic grows while
    the distance converges. Both are comparable across arms evaluated on the same split, which is the
    only comparison this metric is used for.
    """
    if Z.ndim != 2:
        raise ValueError(f"discrepancy expects (N, D) features, got {tuple(Z.shape)}")
    n_samples, d = Z.shape
    if n_samples < 2:
        raise ValueError(f"need at least 2 samples to estimate a characteristic function, got {n_samples}")

    Z = Z.float()
    if center:
        Z = Z - Z.mean(dim=-2, keepdim=True)

    phi, _ = _target_phi(target, d, radius, n_quad)
    torch.manual_seed(seed)
    statistic = float(_sketch_test(Z, phi, knots=knots, num_slices=num_slices, t_max=t_max))
    return {
        "statistic": statistic,
        "distance": statistic / n_samples,
        "n_samples": float(n_samples),
    }


def discrepancy_both_targets(Z: torch.Tensor, **kwargs: Any) -> dict[str, dict[str, float]]:
    """Score ``Z`` against BOTH targets, so arms are comparable rather than self-scored.

    A LocaLeJEPA run measured only against the ball and a LeJEPA run measured only against the
    Gaussian would each be graded on its own curve, and the two numbers would say nothing about each
    other. Reporting both for every arm is what makes the column mean something across the table.
    """
    return {target: discrepancy(Z, target, **kwargs) for target in TARGETS}


def local_discrepancy(
    Z: torch.Tensor,
    target: Target,
    *,
    num_clusters: int,
    center: bool = True,
    radius: float | None = None,
    knots: int = DEFAULT_KNOTS,
    num_slices: int = DEFAULT_NUM_SLICES,
    t_max: float = DEFAULT_T_MAX,
    n_quad: int = DEFAULT_N_QUAD,
    seed: int = DEFAULT_SEED,
    **local_kwargs: Any,
) -> tuple[float, dict[str, float]]:
    """The same statistic inside k-means neighbourhoods, plus the partition that produced it.

    Returns ``(value, stats)``. ``stats`` is `local_loss`'s own out-param and carries
    ``configured_clusters``, ``effective_clusters``, ``kept_fraction`` and ``fallback``.

    **Read ``fallback`` before reading the value.** When every cluster falls below
    `min_cluster_size`, `local_loss` silently scores the whole batch globally instead
    (`src/losses.py:309-310`); the value then looks like a good local score and is not one. A caller
    that ignores ``fallback`` will mistake a degenerate partition for a well-matched one.

    **The value is NOT normalised by N** and is comparable only between cells with the same
    ``(N, num_clusters)``. `local_loss` averages per-cluster statistics, each already scaled by its
    own cluster's size, and it does not expose those sizes -- so there is no single divisor that
    would make this comparable across partitions. The N-normalised global figure from
    :func:`discrepancy` is the cross-arm number; this one is a within-cell diagnostic.
    """
    if Z.ndim != 2:
        raise ValueError(f"local_discrepancy expects (N, D) features, got {tuple(Z.shape)}")
    d = Z.shape[-1]
    phi, resolved_radius = _target_phi(target, d, radius, n_quad)

    def inner(cluster: torch.Tensor) -> torch.Tensor:
        block = cluster.float()
        if center:
            block = block - block.mean(dim=-2, keepdim=True)
        return _sketch_test(block, phi, knots=knots, num_slices=num_slices, t_max=t_max)

    stats: dict[str, float] = {}
    torch.manual_seed(seed)
    # local_loss requires (V, N, D); one clean view is V=1.
    value = local_loss(
        Z.float().unsqueeze(0), loss=inner, num_clusters=num_clusters, stats=stats, **local_kwargs
    )
    stats["radius"] = float(resolved_radius) if resolved_radius is not None else float("nan")
    return float(value), stats


@torch.no_grad()
def covariance_metrics(Z: torch.Tensor) -> dict[str, float]:
    """Isotropy and whitening diagnostics, ported unchanged from the training module.

    `src/models/lejepa_module.py:191-209` already computes exactly these and logs them under
    `train/`, but they were unreachable from any eval path, so the signals
    `results/BallRegRadius_results.md` reads to detect collapse ("the diagonal is below its own
    target by 5e3") could never be read on a frozen checkpoint. This is that function on an
    ``(N, D)`` matrix instead of ``(V, N, D)``; the arithmetic is identical.

    For a target with identity covariance -- which both the Gaussian and the ball at
    ``R = sqrt(d + 2)`` have -- a healthy run has ``cov_diag_mean`` near 1 and the off-diagonal
    means near 0.
    """
    if Z.ndim != 2:
        raise ValueError(f"covariance_metrics expects (N, D) features, got {tuple(Z.shape)}")
    if Z.shape[0] < 2:
        raise ValueError(f"need at least 2 samples for a covariance, got {Z.shape[0]}")

    z_all = Z.float()
    std_mean = z_all.std(dim=0).mean()
    norm_mean = z_all.norm(dim=1).mean()
    mean = z_all.mean(dim=0)
    z_centered = z_all - mean
    cov = z_centered.T @ z_centered / (z_all.size(0) - 1)
    cov_diag = torch.diag(cov)
    cov_offdiag = cov - torch.diag(cov_diag)
    return {
        "std_mean": float(std_mean),
        "norm_mean": float(norm_mean),
        "mean_abs": float(mean.abs().mean()),
        "cov_diag_mean": float(cov_diag.mean()),
        "cov_offdiag_mean": float(cov_offdiag.abs().mean()),
        "cov_offdiag_sq_mean": float(cov_offdiag.square().mean()),
    }
