# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone loss functions shared across BundLeJEPA methods.

All functions accept projections in (V, N, D) layout -- V views, N samples,
D embedding dimension -- which is the uniform interface across all encoders.

invariance_loss  -- pulls every view toward the mean of the reference views
sigreg_loss      -- Sketched Isotropic Gaussian Regularization (SIGReg / LeJEPA)
local_sigreg_loss -- SIGReg computed independently on k-means local neighborhoods
ballreg_loss     -- Sketched uniform-ball regularization
local_ballreg_loss -- uniform-ball regularization on k-means local neighborhoods
susreg_loss      -- Sketched Uniform Spherical Regularization (SUSReg / SPHERE-JEPA)
subreg_loss      -- Subspace Gaussian Regularization (Sub-JEPA)
nt_xent_loss     -- Normalised Temperature-scaled Cross Entropy (SimCLR)
"""

import torch
import torch.nn.functional as F
from numpy.polynomial.legendre import leggauss


def invariance_loss(proj: torch.Tensor, num_global: int | None = None) -> torch.Tensor:
    """L_inv: pull every view toward the mean of the reference (global) views.

    Parameters
    ----------
    proj       : (V, N, D) stacked projections from all views
    num_global : if given, reference is the mean of proj[:num_global];
                 if None, reference is the mean of all views

    Returns
    -------
    Scalar tensor -- mean squared distance from each view to the reference.
    """
    ref = proj[:num_global].mean(0) if num_global is not None else proj.mean(0)
    return (proj - ref).square().mean()


def sigreg_loss(proj: torch.Tensor, knots: int = 17) -> torch.Tensor:
    """SIGReg: Sketched Isotropic Gaussian Regularization (LeJEPA).

    Enforces an isotropic Gaussian distribution on projection embeddings via
    the Epps-Pulley characteristic function test along random 1-D projections.
    Integrates on [0, t_max] using the trapezoid rule and exploits ECF symmetry
    by doubling the inner weights.

    Parameters
    ----------
    proj  : (V, N, D) stacked projections from all views (any dtype)
    knots : number of integration knots (default 17)

    Returns
    -------
    Scalar tensor (float32).
    """
    proj = proj.float()
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

    # Random unit-norm sketch directions in R^D.
    A = torch.randn(proj.size(-1), 256, device=device)
    A = A.div_(A.norm(p=2, dim=0))
    # 1-D projections at each integration knot: (V, N, 256, knots).
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
            "local_sigreg_loss was configured with kmeans_backend='fast_pytorch_kmeans', "
            "but fast-pytorch-kmeans could not be imported. Install it or set "
            "local_sigreg_kmeans_backend='torch'."
        ) from exc

    mode = "cosine" if normalize else "euclidean"
    kmeans = KMeans(n_clusters=K, max_iter=max(int(iters), 1), mode=mode, verbose=0)
    return kmeans.fit_predict(x.float()).long()


def local_sigreg_loss(
    proj: torch.Tensor,
    *,
    num_clusters: int = 8,
    num_global: int | None = None,
    kmeans_iters: int = 10,
    min_cluster_size: int = 4,
    weighted: bool = False,
    normalize_for_kmeans: bool = True,
    kmeans_backend: str = "fast_pytorch_kmeans",
    knots: int = 17,
) -> torch.Tensor:
    """Local SIGReg over k-means neighborhoods in embedding space.

    Clustering is done over one detached point per sample, obtained by averaging
    either the reference/global views ``proj[:num_global]`` or all views. SIGReg
    is then computed independently on each local subset ``proj[:, mask, :]``.

    Parameters
    ----------
    proj                : (V, N, D) stacked projections from all views
    num_clusters        : number of k-means neighborhoods to form
    num_global          : if given, cluster on mean of the first num_global views
    kmeans_iters        : Lloyd iterations for per-batch k-means
    min_cluster_size    : clusters smaller than this are skipped as too noisy
    weighted            : if True, average cluster losses by cluster size;
                          otherwise each non-empty local neighborhood is equal
    normalize_for_kmeans: if True, use cosine-like k-means assignments while
                          still computing SIGReg on raw projections
    kmeans_backend      : "fast_pytorch_kmeans" uses the external accelerated
                          package; "torch" uses the in-repo fallback
    knots               : SIGReg integration knots

    Returns
    -------
    Scalar tensor (float32). Falls back to global SIGReg if every cluster is too
    small after filtering.
    """
    if proj.ndim != 3:
        raise ValueError(f"local_sigreg_loss expects proj with shape (V, N, D), got {tuple(proj.shape)}")

    V, N, _ = proj.shape
    if N < 1:
        raise ValueError("local_sigreg_loss requires at least one sample")

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
        raise ValueError(
            f"Unknown kmeans_backend: {kmeans_backend!r}. Choose 'fast_pytorch_kmeans' or 'torch'."
        )

    losses: list[torch.Tensor] = []
    weights: list[int] = []
    min_size = max(int(min_cluster_size), 1)
    K = min(max(int(num_clusters), 1), N)

    for cluster_idx in range(K):
        mask = labels == cluster_idx
        cluster_size = int(mask.sum().item())
        if cluster_size < min_size:
            continue

        losses.append(sigreg_loss(proj[:, mask, :], knots=knots))
        weights.append(cluster_size)

    if len(losses) == 0:
        return sigreg_loss(proj, knots=knots)

    local_losses = torch.stack(losses)
    if weighted:
        w = torch.tensor(weights, device=proj.device, dtype=local_losses.dtype)
        return (local_losses * (w / w.sum())).sum()

    return local_losses.mean()


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
    if d < 1:
        raise ValueError(f"ball dimension must be >= 1, got {d}")

    R = float((d + 2) ** 0.5 if radius is None else radius)
    if R <= 0:
        raise ValueError(f"ball radius must be positive, got {R}")

    x_np, w_np = leggauss(n_quad)
    x = torch.tensor(x_np, dtype=torch.float32, device=t.device)
    w = torch.tensor(w_np, dtype=torch.float32, device=t.device)

    exponent = (d - 1) / 2.0
    density = (1.0 - x.square()).clamp(min=0.0).pow(exponent)
    norm = (density * w).sum()

    cos_vals = torch.cos((R * t).unsqueeze(-1) * x.unsqueeze(0))
    return (cos_vals * (density * w).unsqueeze(0)).sum(-1) / norm


def ballreg_loss(
    proj: torch.Tensor,
    knots: int = 17,
    radius: float | None = None,
    center: bool = False,
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
    radius: ball radius. If None, uses sqrt(D + 2), which makes Cov[Unif ball] = I.
    center: if True, subtract per-view sample means before computing the test.
            This is useful for local clusters where the target should describe
            local residual shape rather than force every cluster centroid to zero.
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

    # Integration grid and trapezoid weights on [0, 3].
    t = torch.linspace(0, 3, knots, device=device)
    dt = 3.0 / (knots - 1)
    trap = torch.full((knots,), 2.0 * dt, device=device)
    trap[[0, -1]] = dt

    # Keep the same Gaussian integration window used by SIGReg/SUSReg.
    window = torch.exp(-t.square() / 2.0)
    weights = trap * window

    with torch.no_grad():
        target_phi = _ball_d_cf(t, d=d, radius=radius, n_quad=n_quad)

    # Random unit-norm sketch directions in R^D.
    A = torch.randn(d, 256, device=device)
    A = A.div_(A.norm(p=2, dim=0))
    # 1-D projections at each integration knot: (V, N, 256, knots).
    x_t = (proj @ A).unsqueeze(-1) * t
    # Squared ECF error: real part + imaginary part.
    err = (x_t.cos().mean(-3) - target_phi).square() + x_t.sin().mean(-3).square()
    return ((err @ weights) * proj.size(-2)).mean()


def local_ballreg_loss(
    proj: torch.Tensor,
    *,
    num_clusters: int = 8,
    num_global: int | None = None,
    kmeans_iters: int = 10,
    min_cluster_size: int = 4,
    weighted: bool = False,
    normalize_for_kmeans: bool = True,
    kmeans_backend: str = "fast_pytorch_kmeans",
    knots: int = 17,
    radius: float | None = None,
    center_clusters: bool = True,
    n_quad: int = 64,
) -> torch.Tensor:
    """Uniform-ball regularization over k-means neighborhoods.

    Clusters are assigned exactly as in ``local_sigreg_loss``. Each local subset
    is then tested against a uniform-ball target with ``ballreg_loss``.

    ``center_clusters`` defaults to True because local neighborhoods generally
    have nonzero centroids; centering makes the target constrain local residual
    shape rather than pull every cluster center to the origin.
    """
    if proj.ndim != 3:
        raise ValueError(f"local_ballreg_loss expects proj with shape (V, N, D), got {tuple(proj.shape)}")

    V, N, _ = proj.shape
    if N < 1:
        raise ValueError("local_ballreg_loss requires at least one sample")

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
        raise ValueError(
            f"Unknown kmeans_backend: {kmeans_backend!r}. Choose 'fast_pytorch_kmeans' or 'torch'."
        )

    losses: list[torch.Tensor] = []
    weights: list[int] = []
    min_size = max(int(min_cluster_size), 1)
    K = min(max(int(num_clusters), 1), N)

    for cluster_idx in range(K):
        mask = labels == cluster_idx
        cluster_size = int(mask.sum().item())
        if cluster_size < min_size:
            continue

        losses.append(
            ballreg_loss(
                proj[:, mask, :],
                knots=knots,
                radius=radius,
                center=center_clusters,
                n_quad=n_quad,
            )
        )
        weights.append(cluster_size)

    if len(losses) == 0:
        return ballreg_loss(proj, knots=knots, radius=radius, center=center_clusters, n_quad=n_quad)

    local_losses = torch.stack(losses)
    if weighted:
        w = torch.tensor(weights, device=proj.device, dtype=local_losses.dtype)
        return (local_losses * (w / w.sum())).sum()

    return local_losses.mean()


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
