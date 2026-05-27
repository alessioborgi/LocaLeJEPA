# Copyright (c) 2025 Alessio Borgi, Pavel Snopov
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone loss functions shared across BundLeJEPA methods.

All functions accept projections in (V, N, D) layout — V views, N samples,
D embedding dimension — which is the uniform interface across all encoders.

invariance_loss  -- pulls every view toward the mean of the reference views
sigreg_loss      -- Sketched Isotropic Gaussian Regularization (SIGReg / LeJEPA)
susreg_loss      -- Sketched Uniform Spherical Regularization (SUSReg / SPHERE-JEPA)
subreg_loss      -- Subspace Gaussian Regularization (Sub-JEPA)
nt_xent_loss     -- Normalised Temperature-scaled Cross Entropy (SimCLR)
"""

import numpy as np
import torch
import torch.nn.functional as F
from numpy.polynomial.legendre import leggauss


def invariance_loss(proj: torch.Tensor, num_global: int | None = None) -> torch.Tensor:
    """L_inv: pull every view toward the mean of the reference (global) views.

    Parameters
    ----------
    proj       : (V, N, D) — stacked projections from all views
    num_global : if given, the reference is the mean of proj[:num_global];
                 if None, the reference is the mean of all views

    Returns
    -------
    Scalar tensor — mean squared distance from each view to the reference.
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
    proj  : (V, N, D) — stacked projections from all views (any dtype)
    knots : number of integration knots (default 17)

    Returns
    -------
    Scalar tensor (float32).
    """
    proj   = proj.float()
    device = proj.device

    t       = torch.linspace(0, 3, knots, device=device)
    dt      = 3.0 / (knots - 1)
    weights = torch.full((knots,), 2.0 * dt, device=device)
    weights[[0, -1]] = dt
    phi     = torch.exp(-t.square() / 2.0)
    weights = weights * phi

    A   = torch.randn(proj.size(-1), 256, device=device)
    A   = A.div_(A.norm(p=2, dim=0))
    x_t = (proj @ A).unsqueeze(-1) * t          # (V, N, 256, knots)
    err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
    return ((err @ weights) * proj.size(-2)).mean()


def _rho_d_cf(t: torch.Tensor, d: int, n_quad: int = 64) -> torch.Tensor:
    """Characteristic function of the spherical projection density ρ_d.

    For X ~ Unif(S^{d-1}) and a ∈ S^{d-1}, the scalar a^T X has density
    ρ_d(s) ∝ (1 - s^2)^{(d-3)/2} on (-1, 1). The CF is real by symmetry.
    Computed via Gauss-Legendre quadrature on [-1, 1].

    Parameters
    ----------
    t     : (knots,) integration points at which to evaluate φ_{ρ_d}
    d     : dimension of the hypersphere S^{d-1} (d ≥ 2)
    n_quad: number of Gauss-Legendre quadrature nodes

    Returns
    -------
    phi : (knots,) real-valued CF of ρ_d, float32, on the same device as t.
    """
    x_np, w_np = leggauss(n_quad)
    x = torch.tensor(x_np, dtype=torch.float32, device=t.device)   # (n_quad,)
    w = torch.tensor(w_np, dtype=torch.float32, device=t.device)   # (n_quad,)

    exponent = (d - 3) / 2.0
    density = (1.0 - x.square()).clamp(min=0.0).pow(exponent)      # (n_quad,)
    norm = (density * w).sum()

    # φ(t) = E[cos(t·s)] = ∫ cos(t·s) ρ_d(s) ds / ∫ ρ_d(s) ds
    cos_vals = torch.cos(t.unsqueeze(-1) * x.unsqueeze(0))         # (knots, n_quad)
    phi = (cos_vals * (density * w).unsqueeze(0)).sum(-1) / norm   # (knots,)
    return phi


def susreg_loss(proj: torch.Tensor, knots: int = 17) -> torch.Tensor:
    """SUSReg: Sketched Uniform Spherical Regularization (SPHERE-JEPA).

    Enforces uniformity on S^{d-1} by testing, via the Epps-Pulley statistic,
    that L2-normalised projections match the spherical projection density ρ_d
    rather than the Gaussian prior used by SIGReg. Uses a Gaussian integration
    weight w(t) = e^{-t²/2} independently of the target CF.

    For d > 256 the central-limit approximation ρ_d ≈ N(0, 1/d) is used.

    Reference: Nicollier et al., "SPHERE-JEPA", arXiv 2605.26900 (2026).

    Parameters
    ----------
    proj  : (V, N, D) — stacked projections from all views (any dtype)
    knots : number of integration knots (default 17)

    Returns
    -------
    Scalar tensor (float32).
    """
    proj   = F.normalize(proj.float(), dim=-1)   # project to S^{d-1}
    device = proj.device
    d      = proj.shape[-1]

    t  = torch.linspace(0, 3, knots, device=device)
    dt = 3.0 / (knots - 1)
    trap    = torch.full((knots,), 2.0 * dt, device=device)
    trap[[0, -1]] = dt
    gauss_w = torch.exp(-t.square() / 2.0)   # integration weight w(t)
    weights = trap * gauss_w

    with torch.no_grad():
        phi_Y = (
            torch.exp(-t.square() / (2.0 * d))   # Gaussian approx for large d
            if d > 256
            else _rho_d_cf(t, d)
        )

    A   = torch.randn(d, 256, device=device)
    A   = A.div_(A.norm(p=2, dim=0))
    x_t = (proj @ A).unsqueeze(-1) * t          # (V, N, 256, knots)
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
    random sketch direction set, matching the regularisation to the intrinsic
    dimensionality of the representation manifold.

    Reference: Zhao et al., "Sub-JEPA", arXiv 2605.09241 (2026).

    Parameters
    ----------
    proj         : (V, N, D) — stacked projections from all views
    proj_matrices: (K, ds, D) — frozen row-orthonormal projection matrices,
                   registered as a buffer on LeJEPAModule
    knots        : number of integration knots (default 17)

    Returns
    -------
    Scalar tensor (float32).
    """
    proj = proj.float()
    V, N, D = proj.shape
    K, ds, _ = proj_matrices.shape
    device   = proj.device

    t       = torch.linspace(0, 3, knots, device=device)
    dt      = 3.0 / (knots - 1)
    weights = torch.full((knots,), 2.0 * dt, device=device)
    weights[[0, -1]] = dt
    phi     = torch.exp(-t.square() / 2.0)
    weights = weights * phi

    # Project into each subspace: result[k,v,n,m] = Σ_d proj[v,n,d] * P[k,m,d]
    z_sub = torch.einsum('vnd,kmd->kvnm', proj, proj_matrices)   # (K, V, N, ds)

    # Random sketch directions in R^{ds}, one shared set across subspaces
    A = torch.randn(ds, 256, device=device)
    A = A.div_(A.norm(p=2, dim=0))                               # (ds, 256)

    scalars = z_sub @ A                                           # (K, V, N, 256)
    x_t = scalars.unsqueeze(-1) * t                              # (K, V, N, 256, knots)
    err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
    return ((err @ weights) * N).mean()


def nt_xent_loss(proj: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """NT-Xent contrastive loss for SimCLR.

    Parameters
    ----------
    proj        : (V=2, N, D) — projections from exactly two views
    temperature : softmax temperature τ (default 0.07)

    Returns
    -------
    Scalar cross-entropy loss.
    """
    if proj.shape[0] != 2:
        raise ValueError(f'NT-Xent requires exactly 2 views, got {proj.shape[0]}')

    z1 = F.normalize(proj[0].float(), dim=-1)   # (N, D)
    z2 = F.normalize(proj[1].float(), dim=-1)

    N   = z1.shape[0]
    z   = torch.cat([z1, z2], dim=0)            # (2N, D)
    sim = (z @ z.T) / temperature               # (2N, 2N)

    sim.fill_diagonal_(float('-inf'))

    labels = torch.cat([
        torch.arange(N, 2 * N, device=proj.device),
        torch.arange(0, N,     device=proj.device),
    ])
    return F.cross_entropy(sim, labels)
