# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone loss functions shared across LocaLeJEPA methods.

All functions accept projections in (V, N, D) layout -- V views, N samples,
D embedding dimension -- which is the uniform interface across all encoders.

invariance_loss  -- pulls every view toward the mean of the reference views
sigreg_loss      -- Sketched Isotropic Gaussian Regularization (SIGReg / LeJEPA)
ballreg_loss     -- Sketched uniform-ball regularization
local_loss       -- any of the tests above computed independently on k-means local neighborhoods
local_sigreg_loss  -- local_loss with sigreg_loss
local_ballreg_loss -- local_loss with ballreg_loss centered per cluster (LocaLeJEPA)
susreg_loss      -- Sketched Uniform Spherical Regularization (SUSReg / SPHERE-JEPA)
subreg_loss      -- Subspace Gaussian Regularization (Sub-JEPA)
rdmreg_loss      -- Rectified Distribution Matching Regularization (Rectified LpJEPA)
nt_xent_loss     -- Normalised Temperature-scaled Cross Entropy (SimCLR)
"""

import math
from collections.abc import Callable
from functools import lru_cache, partial
from typing import Any

import torch
import torch.nn.functional as F
from numpy.polynomial.legendre import leggauss


def invariance_loss(proj: torch.Tensor, num_global: int | None = None, normalize: bool = False) -> torch.Tensor:
    """L_inv: pull every view toward the mean of the reference (global) views.

    Parameters
    ----------
    proj       : (V, N, D) stacked projections from all views
    num_global : if given, reference is the mean of proj[:num_global];
                 if None, reference is the mean of all views
    normalize  : L2-normalize the projections first (SPHERE-JEPA convention for SUSReg)

    Returns
    -------
    Scalar tensor -- mean squared distance from each view to the reference.
    """
    if normalize:
        proj = F.normalize(proj, dim=-1)
    ref = proj[:num_global].mean(0) if num_global is not None else proj.mean(0)
    return (proj - ref).square().mean()


def sigreg_loss(proj: torch.Tensor, knots: int = 17, num_slices: int = 256, t_max: float = 3.0) -> torch.Tensor:
    """SIGReg: Sketched Isotropic Gaussian Regularization (LeJEPA).

    Enforces an isotropic Gaussian distribution on projection embeddings via
    the Epps-Pulley characteristic function test along random 1-D projections.
    Integrates on [0, t_max] using the trapezoid rule and exploits ECF symmetry
    by doubling the inner weights (so [0, t_max] equals the paper's [-t_max, t_max]).

    Parameters
    ----------
    proj       : (V, N, D) stacked projections from all views (any dtype)
    knots      : number of integration knots (default 17)
    num_slices : number of random sketch directions |A| (default 256; the LeJEPA
                 paper recommends 1024 for ViT-L/14 on ImageNet-1K)
    t_max      : half-width of the integration domain (default 3.0; paper uses 5.0)

    Returns
    -------
    Scalar tensor (float32).
    """
    proj = proj.float()
    device = proj.device

    # Integration grid and trapezoid weights on [0, t_max].
    t = torch.linspace(0, t_max, knots, device=device)
    dt = t_max / (knots - 1)
    weights = torch.full((knots,), 2.0 * dt, device=device)
    # Boundary correction for trapezoid rule.
    weights[[0, -1]] = dt
    # Target CF of N(0,1): phi(t) = exp(-t^2 / 2). Also used as integration weight.
    phi = torch.exp(-t.square() / 2.0)
    weights = weights * phi

    # Random unit-norm sketch directions in R^D.
    A = torch.randn(proj.size(-1), num_slices, device=device)
    A = A.div_(A.norm(p=2, dim=0))
    # 1-D projections at each integration knot: (V, N, num_slices, knots).
    x_t = (proj @ A).unsqueeze(-1) * t
    # Squared ECF error: real part + imaginary part.
    err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
    return ((err @ weights) * proj.size(-2)).mean()


def _validate_kmeans_input(x: torch.Tensor, num_clusters: int) -> tuple[int, int, int]:
    if x.ndim != 2:
        raise ValueError(f"k-means expects a 2D point cloud (N, D), got {tuple(x.shape)}")

    N, D = x.shape
    if N < 1:
        raise ValueError("k-means requires at least one point")

    K = min(max(int(num_clusters), 1), N)
    return N, D, K


@torch.no_grad()
def _torch_kmeans_labels(
    x: torch.Tensor,
    num_clusters: int,
    *,
    iters: int = 10,
    normalize: bool = True,
) -> torch.Tensor:
    """Assign a point cloud to k-means clusters using pure PyTorch.

    Parameters
    ----------
    x           : (N, D) point cloud used only for cluster assignment
    num_clusters: requested number of clusters; clipped to [1, N]
    iters       : number of Lloyd iterations
    normalize   : if True, L2-normalize points/centroids for cosine-like
                  neighborhoods

    Returns
    -------
    labels : (N,) long tensor with values in [0, K)
    """
    N, D, K = _validate_kmeans_input(x, num_clusters)
    x = x.float()
    if normalize:
        x = F.normalize(x, dim=-1)

    centroids = x[torch.randperm(N, device=x.device)[:K]]
    labels = torch.zeros(N, dtype=torch.long, device=x.device)

    for _ in range(max(int(iters), 1)):
        labels = torch.cdist(x, centroids).argmin(dim=-1)

        sums = torch.zeros_like(centroids)
        sums.scatter_add_(0, labels[:, None].expand(-1, D), x)

        counts = torch.bincount(labels, minlength=K)
        nonempty = counts > 0

        new_centroids = centroids.clone()
        new_centroids[nonempty] = sums[nonempty] / counts[nonempty, None]
        if normalize:
            new_centroids = F.normalize(new_centroids, dim=-1)

        centroids = new_centroids

    return labels


@torch.no_grad()
def _fast_pytorch_kmeans_labels(
    x: torch.Tensor,
    num_clusters: int,
    *,
    iters: int = 10,
    normalize: bool = True,
) -> torch.Tensor:
    """Assign k-means labels using fast-pytorch-kmeans."""
    _, _, K = _validate_kmeans_input(x, num_clusters)

    try:
        from fast_pytorch_kmeans import KMeans
    except ImportError as exc:
        raise ImportError(
            "kmeans_backend='fast_pytorch_kmeans' but fast-pytorch-kmeans could not be imported. "
            "Install it or set model.reg_loss.kmeans_backend=torch."
        ) from exc

    mode = "cosine" if normalize else "euclidean"
    kmeans = KMeans(n_clusters=K, max_iter=max(int(iters), 1), mode=mode, verbose=0)
    return kmeans.fit_predict(x.float()).long()


def local_loss(
    proj: torch.Tensor,
    *,
    loss: Callable[[torch.Tensor], torch.Tensor],
    num_clusters: int = 8,
    num_global: int | None = None,
    kmeans_iters: int = 10,
    min_cluster_size: int = 4,
    weighted: bool = False,
    normalize_for_kmeans: bool = True,
    kmeans_backend: str = "fast_pytorch_kmeans",
) -> torch.Tensor:
    """A distribution-matching test applied over k-means neighborhoods in embedding space.

    Clustering is done over one detached point per sample, obtained by averaging
    either the reference/global views ``proj[:num_global]`` or all views. ``loss``
    is then computed independently on each local subset ``proj[:, mask, :]``.

    Parameters
    ----------
    proj                : (V, N, D) stacked projections from all views
    loss                : (V, n, D) -> scalar test applied to each neighborhood
    num_clusters        : number of k-means neighborhoods to form
    num_global          : if given, cluster on mean of the first num_global views
    kmeans_iters        : Lloyd iterations for per-batch k-means
    min_cluster_size    : clusters smaller than this are skipped as too noisy
    weighted            : if True, average cluster losses by cluster size;
                          otherwise each non-empty local neighborhood is equal
    normalize_for_kmeans: if True, use cosine-like k-means assignments while
                          still computing the test on raw projections
    kmeans_backend      : "fast_pytorch_kmeans" uses the external accelerated
                          package; "torch" uses the in-repo fallback

    Returns
    -------
    Scalar tensor (float32). Falls back to ``loss`` on the whole batch if every
    cluster is too small after filtering.
    """
    if proj.ndim != 3:
        raise ValueError(f"local_loss expects proj with shape (V, N, D), got {tuple(proj.shape)}")

    V, N, _ = proj.shape
    if N < 1:
        raise ValueError("local_loss requires at least one sample")

    if num_global is not None:
        if num_global < 1 or num_global > V:
            raise ValueError(f"num_global must be in [1, {V}], got {num_global}")
        cluster_points = proj[:num_global].mean(dim=0).detach()
    else:
        cluster_points = proj.mean(dim=0).detach()

    if kmeans_backend == "fast_pytorch_kmeans":
        labels = _fast_pytorch_kmeans_labels(
            cluster_points,
            num_clusters=num_clusters,
            iters=kmeans_iters,
            normalize=normalize_for_kmeans,
        )
    elif kmeans_backend == "torch":
        labels = _torch_kmeans_labels(
            cluster_points,
            num_clusters=num_clusters,
            iters=kmeans_iters,
            normalize=normalize_for_kmeans,
        )
    else:
        raise ValueError(f"Unknown kmeans_backend: {kmeans_backend!r}. Choose 'fast_pytorch_kmeans' or 'torch'.")

    losses: list[torch.Tensor] = []
    weights: list[int] = []
    min_size = max(int(min_cluster_size), 1)
    K = min(max(int(num_clusters), 1), N)
    # One host synchronization for every cluster size, instead of one per cluster.
    sizes = torch.bincount(labels, minlength=K).tolist()

    for cluster_idx in range(K):
        if sizes[cluster_idx] < min_size:
            continue

        mask = labels == cluster_idx
        losses.append(loss(proj[:, mask, :]))
        weights.append(sizes[cluster_idx])

    if len(losses) == 0:
        return loss(proj)

    local_losses = torch.stack(losses)
    if weighted:
        w = torch.tensor(weights, device=proj.device, dtype=local_losses.dtype)
        return (local_losses * (w / w.sum())).sum()

    return local_losses.mean()


@lru_cache(maxsize=32)
def _ball_quadrature(d: int, n_quad: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gauss-Legendre nodes and density weights, which depend on neither the grid nor the radius."""
    x_np, w_np = leggauss(n_quad)
    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    w = torch.tensor(w_np, dtype=torch.float32, device=device)

    weights = (1.0 - x.square()).clamp(min=0.0).pow((d - 1) / 2.0) * w
    return x, weights, weights.sum()


def _resolve_ball_radius(d: int, radius: float | None) -> float:
    """Resolve and validate the radius of a d-dimensional target ball."""
    if d < 1:
        raise ValueError(f"ball dimension must be >= 1, got {d}")

    R = float((d + 2) ** 0.5 if radius is None else radius)
    if R <= 0:
        raise ValueError(f"ball radius must be positive, got {R}")
    return R


def _ball_d_cf(
    t: torch.Tensor,
    d: int,
    radius: float | None = None,
    n_quad: int = 64,
) -> torch.Tensor:
    """Characteristic function of a 1-D projection of a uniform d-ball.

    If ``X`` is uniform on the centered ``d``-ball with radius ``R`` and ``a``
    is unit-norm, then ``a^T X`` has density proportional to
    ``(1 - (s/R)^2)^((d-1)/2)`` on ``[-R, R]``. This computes
    ``E[cos(t a^T X)]`` by Gauss-Legendre quadrature on ``[-1, 1]``.

    The default ``radius=sqrt(d+2)`` gives ``Cov[X] = I``.
    """
    R = _resolve_ball_radius(d, radius)
    x, weights, norm = _ball_quadrature(d, n_quad, str(t.device))

    cos_vals = torch.cos((R * t).unsqueeze(-1) * x.unsqueeze(0))
    return (cos_vals * weights.unsqueeze(0)).sum(-1) / norm


def ballreg_loss(
    proj: torch.Tensor,
    knots: int = 17,
    num_slices: int = 256,
    t_max: float = 3.0,
    radius: float | None = None,
    center: bool = False,
    scale_grid: bool = False,
    n_quad: int = 64,
) -> torch.Tensor:
    """Sketched uniform-ball regularization.

    This is the same Epps-Pulley-style characteristic-function statistic as
    SIGReg, but the target distribution is uniform on a centered d-dimensional
    ball instead of isotropic Gaussian. The Gaussian window is retained as the
    integration weight; only the target characteristic function is replaced.

    Parameters
    ----------
    proj  : (V, N, D) stacked projections from all views
    knots : number of integration knots
    num_slices: number of random sketch directions |A|
    t_max : half-width of the integration domain
    radius: ball radius. If None, uses sqrt(D + 2), which makes Cov[Unif ball] = I.
    center: if True, subtract per-view sample means before computing the test.
            This is useful for local clusters where the target should describe
            local residual shape rather than force every cluster centroid to zero.
    scale_grid: if True, measure frequency in units of 1/sigma_target instead of
                absolute units, i.e. evaluate the characteristic functions at
                ``t * sqrt(d + 2) / R``. See the comment at the grid below.
    n_quad: Gauss-Legendre quadrature nodes for the target CF

    Returns
    -------
    Scalar tensor (float32).
    """
    if proj.ndim != 3:
        raise ValueError(f"ballreg_loss expects proj with shape (V, N, D), got {tuple(proj.shape)}")

    proj = proj.float()
    if center:
        proj = proj - proj.mean(dim=-2, keepdim=True)

    device = proj.device
    d = proj.shape[-1]
    R = _resolve_ball_radius(d, radius)

    # Integration grid and trapezoid weights on [0, t_max].
    u = torch.linspace(0, t_max, knots, device=device)
    dt = t_max / (knots - 1)
    trap = torch.full((knots,), 2.0 * dt, device=device)
    trap[[0, -1]] = dt

    # Keep the same Gaussian integration window used by SIGReg/SUSReg.
    window = torch.exp(-u.square() / 2.0)
    weights = trap * window

    # The target CF only turns over at t ~ 1/sigma_target = sqrt(d + 2) / R, so on a
    # grid fixed in absolute units a small ball is nearly indistinguishable from a
    # collapsed distribution and the statistic barely penalizes collapse. Scaling the
    # grid equalizes that power across radii; it is a no-op at R = sqrt(d + 2).
    t = u * ((d + 2) ** 0.5 / R) if scale_grid else u

    with torch.no_grad():
        target_phi = _ball_d_cf(t, d=d, radius=R, n_quad=n_quad)

    # Random unit-norm sketch directions in R^D.
    A = torch.randn(d, num_slices, device=device)
    A = A.div_(A.norm(p=2, dim=0))
    # 1-D projections at each integration knot: (V, N, num_slices, knots).
    x_t = (proj @ A).unsqueeze(-1) * t
    # Squared ECF error: real part + imaginary part.
    err = (x_t.cos().mean(-3) - target_phi).square() + x_t.sin().mean(-3).square()
    return ((err @ weights) * proj.size(-2)).mean()


def local_sigreg_loss(proj: torch.Tensor, *, loss: Callable[[torch.Tensor], torch.Tensor] = sigreg_loss, **kwargs: Any) -> torch.Tensor:
    """SIGReg over k-means neighborhoods; see ``local_loss``."""
    return local_loss(proj, loss=loss, **kwargs)


def local_ballreg_loss(
    proj: torch.Tensor,
    *,
    loss: Callable[[torch.Tensor], torch.Tensor] = partial(ballreg_loss, center=True),
    **kwargs: Any,
) -> torch.Tensor:
    """Uniform-ball regularization over k-means neighborhoods, centered per cluster; see ``local_loss``."""
    return local_loss(proj, loss=loss, **kwargs)


def _rho_d_cf(t: torch.Tensor, d: int, n_quad: int = 64) -> torch.Tensor:
    """Characteristic function of the spherical projection density for dimension d.

    For X uniform on the (d-1)-sphere, the scalar projection a^T X has density
    proportional to (1 - s^2)^((d-3)/2) on (-1, 1). The CF is real by symmetry.
    Computed via Gauss-Legendre quadrature on [-1, 1].

    Parameters
    ----------
    t     : (knots,) integration points at which to evaluate the CF
    d     : ambient dimension (d >= 2)
    n_quad: number of Gauss-Legendre quadrature nodes

    Returns
    -------
    phi : (knots,) real-valued CF, float32, on the same device as t.
    """
    x_np, w_np = leggauss(n_quad)
    # Quadrature nodes and weights on [-1, 1].
    x = torch.tensor(x_np, dtype=torch.float32, device=t.device)
    w = torch.tensor(w_np, dtype=torch.float32, device=t.device)

    exponent = (d - 3) / 2.0
    # Unnormalized density at each quadrature node: (n_quad,).
    density = (1.0 - x.square()).clamp(min=0.0).pow(exponent)
    # Normalization constant via quadrature.
    norm = (density * w).sum()

    # CF = E[cos(t*s)] = integral of cos(t*s) * density(s) ds / norm.
    # cos_vals shape: (knots, n_quad).
    cos_vals = torch.cos(t.unsqueeze(-1) * x.unsqueeze(0))
    # Integrate and normalize to get CF values: (knots,).
    phi = (cos_vals * (density * w).unsqueeze(0)).sum(-1) / norm
    return phi


def susreg_loss(proj: torch.Tensor, knots: int = 17) -> torch.Tensor:
    """SUSReg: Sketched Uniform Spherical Regularization (SPHERE-JEPA).

    Enforces uniformity on the unit hypersphere by testing, via the Epps-Pulley
    statistic, that L2-normalized projections match the spherical projection
    density rather than the Gaussian prior used by SIGReg. Uses a Gaussian
    integration weight w(t) = exp(-t^2/2) independently of the target CF.

    For d > 256 the CLT approximation N(0, 1/d) is used instead of the exact CF.

    Reference: Nicollier et al., "SPHERE-JEPA", arXiv 2605.26900 (2026).

    Parameters
    ----------
    proj  : (V, N, D) stacked projections from all views (any dtype)
    knots : number of integration knots (default 17)

    Returns
    -------
    Scalar tensor (float32).
    """
    # Normalize projections to unit norm before computing the loss.
    proj = F.normalize(proj.float(), dim=-1)
    device = proj.device
    d = proj.shape[-1]

    # Integration grid and trapezoid weights on [0, 3].
    t = torch.linspace(0, 3, knots, device=device)
    dt = 3.0 / (knots - 1)
    trap = torch.full((knots,), 2.0 * dt, device=device)
    # Boundary correction for trapezoid rule.
    trap[[0, -1]] = dt
    # Gaussian integration weight, independent of the target CF.
    gauss_w = torch.exp(-t.square() / 2.0)
    weights = trap * gauss_w

    with torch.no_grad():
        # Target CF: exact for d <= 256, CLT Gaussian approximation for d > 256.
        phi_Y = torch.exp(-t.square() / (2.0 * d)) if d > 256 else _rho_d_cf(t, d)

    # Random unit-norm sketch directions in R^d.
    A = torch.randn(d, 256, device=device)
    A = A.div_(A.norm(p=2, dim=0))
    # 1-D projections at each integration knot: (V, N, 256, knots).
    x_t = (proj @ A).unsqueeze(-1) * t
    # Squared ECF error: real part + imaginary part.
    err = (x_t.cos().mean(-3) - phi_Y).square() + x_t.sin().mean(-3).square()
    return ((err @ weights) * proj.size(-2)).mean()


def subreg_loss(
    proj: torch.Tensor,
    proj_matrices: torch.Tensor,
    knots: int = 17,
) -> torch.Tensor:
    """SubReg: Subspace Gaussian Regularization (Sub-JEPA).

    Applies the Epps-Pulley Gaussian test independently in K frozen orthonormal
    subspaces rather than the full ambient space. Each subspace uses its own
    random sketch direction set, matching the regularization to the intrinsic
    dimensionality of the representation manifold.

    Reference: Zhao et al., "Sub-JEPA", arXiv 2605.09241 (2026).

    Parameters
    ----------
    proj         : (V, N, D) stacked projections from all views
    proj_matrices: (K, ds, D) frozen row-orthonormal projection matrices,
                   registered as a buffer on LeJEPAModule
    knots        : number of integration knots (default 17)

    Returns
    -------
    Scalar tensor (float32).
    """
    proj = proj.float()
    V, N, D = proj.shape
    K, ds, _ = proj_matrices.shape
    device = proj.device

    # Integration grid and trapezoid weights on [0, 3].
    t = torch.linspace(0, 3, knots, device=device)
    dt = 3.0 / (knots - 1)
    weights = torch.full((knots,), 2.0 * dt, device=device)
    # Boundary correction for trapezoid rule.
    weights[[0, -1]] = dt
    # Target CF of N(0,1): phi(t) = exp(-t^2 / 2). Also used as integration weight.
    phi = torch.exp(-t.square() / 2.0)
    weights = weights * phi

    # Project into each subspace via einsum: result[k,v,n,m] = sum_d proj[v,n,d] * P[k,m,d].
    # Output shape: (K, V, N, ds).
    z_sub = torch.einsum("vnd,kmd->kvnm", proj, proj_matrices)

    # Random unit-norm sketch directions in R^ds, shared across all subspaces.
    # Shape: (ds, 256).
    A = torch.randn(ds, 256, device=device)
    A = A.div_(A.norm(p=2, dim=0))

    # 1-D projections per subspace: (K, V, N, 256).
    scalars = z_sub @ A
    # Broadcast with integration knots: (K, V, N, 256, knots).
    x_t = scalars.unsqueeze(-1) * t
    # Squared ECF error: real part + imaginary part, averaged over views.
    err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
    return ((err @ weights) * N).mean()


def sigma_gn(p: float) -> float:
    """Scale sigma_GN so the Generalized Gaussian GN_p(0, sigma) has unit variance.

    This is the default scale used in Rectified LpJEPA (paper Section 4.5). It
    fixes the pre-rectification variance to 1; the post-rectification variance of
    RGN_p(mu, sigma_GN) then varies with mu and p. Setting p=1 gives 1/sqrt(2)
    (Laplace) and p=2 gives 1.0 (Gaussian).
    """
    return math.gamma(1.0 / p) ** 0.5 / (p ** (1.0 / p) * math.gamma(3.0 / p) ** 0.5)


def sample_generalized_gaussian(
    shape: tuple[int, ...],
    p: float,
    loc: float = 0.0,
    scale: float = 1.0,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample from the Generalized Gaussian GN_p(loc, scale).

    Uses the exact fast paths for the Laplace (p=1) and Gaussian (p=2) special
    cases, and the sign/Gamma reparameterization of Nardon & Pianca (2009) for a
    generic p > 0 (paper Algorithm 1, before rectification)::

        S ~ Unif{-1, +1},  G ~ Gamma(1/p, 1),  X = loc + scale * S * (p G)^{1/p}

    Rectify the output with ``relu`` to obtain a Rectified Generalized Gaussian
    RGN_p(loc, scale).
    """
    if p <= 0:
        raise ValueError(f"lp_norm_parameter p must be > 0, got {p}")

    if p == 1.0:
        loc_t = torch.as_tensor(loc, device=device, dtype=dtype)
        scale_t = torch.as_tensor(scale, device=device, dtype=dtype)
        return torch.distributions.Laplace(loc_t, scale_t).sample(shape)
    if p == 2.0:
        return loc + scale * torch.randn(shape, device=device, dtype=dtype)

    sign = torch.empty(shape, device=device, dtype=dtype).bernoulli_(0.5).mul_(2.0).sub_(1.0)
    gamma = torch.distributions.Gamma(1.0 / p, 1.0).sample(shape).to(device=device, dtype=dtype)
    return loc + scale * sign * (p * gamma).pow(1.0 / p)


def rdmreg_loss(
    proj: torch.Tensor,
    *,
    lp_norm_parameter: float = 1.0,
    mean_shift_value: float = 0.0,
    sigma: float | None = None,
    num_projections: int = 256,
    rectify_target: bool = True,
) -> torch.Tensor:
    """RDMReg: Rectified Distribution Matching Regularization (Rectified LpJEPA).

    Matches the feature distribution to a (Rectified) Generalized Gaussian target
    ``prod_i RGN_p(mu, sigma)`` using a two-sample sliced 2-Wasserstein distance,
    the Cramer-Wold device instantiated with random 1-D projections. Because the
    RGG family is not closed under linear combinations, matching must be
    nonparametric two-sample; here the 1-D distance is the sorted-sample L2^2
    (paper Eq. 16).

    Reference: Kuang et al., "Rectified LpJEPA", arXiv:2602.01456 (2026).

    Parameters
    ----------
    proj              : (V, N, D) stacked projections from all views. When
                        matching the Rectified target these should already be
                        rectified (ReLU) features.
    lp_norm_parameter : ell_p parameter p of the Generalized Gaussian target
    mean_shift_value  : location mu of the target; more negative mu => sparser
    sigma             : target scale; if None, uses sigma_GN(p)
    num_projections   : number of random 1-D sketch directions on S^{D-1}
    rectify_target    : if True, ReLU the target samples (Rectified LpJEPA); if
                        False, match the plain Generalized Gaussian (LpJEPA)

    Returns
    -------
    Scalar tensor (float32), averaged over the sliced distances and views.
    """
    if proj.ndim != 3:
        raise ValueError(f"rdmreg_loss expects proj with shape (V, N, D), got {tuple(proj.shape)}")

    proj = proj.float()
    V, N, D = proj.shape
    device = proj.device
    scale = sigma_gn(lp_norm_parameter) if sigma is None else float(sigma)

    # Random unit-norm sketch directions on the ell_2 sphere S^{D-1}: (P, D).
    directions = torch.randn(num_projections, D, device=device)
    directions = F.normalize(directions, dim=1)

    losses = []
    for v in range(V):
        z = proj[v]  # (N, D)
        target = sample_generalized_gaussian(
            (N, D),
            lp_norm_parameter,
            loc=mean_shift_value,
            scale=scale,
            device=device,
            dtype=proj.dtype,
        )
        if rectify_target:
            target = F.relu(target)

        # 1-D projected marginals: (N, P).
        z_proj = z @ directions.T
        target_proj = target @ directions.T

        # Sliced 2-Wasserstein: sort along the batch dim, take mean squared gap.
        z_sorted, _ = z_proj.sort(dim=0)
        target_sorted, _ = target_proj.sort(dim=0)
        losses.append((z_sorted - target_sorted).square().mean())

    return torch.stack(losses).mean()


@torch.no_grad()
def l1_sparsity_metric(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """ell_1 sparsity metric (1/D) * (||z||_1 / ||z||_2)^2, averaged over samples.

    Attains 1/D for maximally sparse vectors and 1 for dense uniform vectors.
    ``z`` is any (..., D) tensor; the metric is averaged over all leading dims.
    """
    D = z.shape[-1]
    z = z.float()
    l1 = z.norm(p=1, dim=-1)
    l2 = z.norm(p=2, dim=-1)
    return ((1.0 / D) * (l1 / (l2 + eps)).square()).mean()


@torch.no_grad()
def l0_sparsity_metric(z: torch.Tensor) -> torch.Tensor:
    """ell_0 sparsity metric (1/D) * ||z||_0: mean fraction of nonzero entries."""
    D = z.shape[-1]
    return ((z != 0).sum(dim=-1).float() / D).mean()


def nt_xent_loss(proj: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """NT-Xent contrastive loss for SimCLR.

    Parameters
    ----------
    proj        : (V=2, N, D) projections from exactly two views
    temperature : softmax temperature (default 0.07)

    Returns
    -------
    Scalar cross-entropy loss.
    """
    if proj.shape[0] != 2:
        raise ValueError(f"NT-Xent requires exactly 2 views, got {proj.shape[0]}")

    # Normalize both views to unit norm.
    z1 = F.normalize(proj[0].float(), dim=-1)
    z2 = F.normalize(proj[1].float(), dim=-1)

    N = z1.shape[0]
    # Concatenate views and compute all-pairs cosine similarity: (2N, 2N).
    z = torch.cat([z1, z2], dim=0)
    sim = (z @ z.T) / temperature

    # Exclude self-similarity from the softmax denominator.
    sim.fill_diagonal_(float("-inf"))

    labels = torch.cat(
        [
            torch.arange(N, 2 * N, device=proj.device),
            torch.arange(0, N, device=proj.device),
        ]
    )
    return F.cross_entropy(sim, labels)
