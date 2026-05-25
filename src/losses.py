# Copyright (c) 2025 Alessio Borgi, Pavel Snopov
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone loss functions shared across BundLeJEPA methods.

All functions accept projections in (V, N, D) layout — V views, N samples,
D embedding dimension — which is the uniform interface across all encoders.

invariance_loss  -- pulls every view toward the mean of the reference views
sigreg_loss      -- Sketched Isotropic Gaussian Regularization (SIGReg)
nt_xent_loss     -- Normalised Temperature-scaled Cross Entropy (SimCLR)
align_loss       -- AUH alignment: mean L2^alpha distance over positive pairs (Wang & Isola, ICML 2020)
uniform_loss     -- AUH uniformity: log Gaussian potential kernel (Wang & Isola, ICML 2020)
sfrik_loss       -- SFRIK uniformity: truncated rotation-invariant kernel MMD (Zheng et al., ICLR 2023)
"""

import torch
import torch.nn.functional as F


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
    """SIGReg: Sketched Isotropic Gaussian Regularization.

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


def align_loss(proj: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """L_align from Wang & Isola (ICML 2020): mean L2^alpha over positive (same-sample) view pairs.

    Parameters
    ----------
    proj  : (V, N, D) — stacked projections from all views
    alpha : exponent (default 2); original paper uses alpha=2

    Returns
    -------
    Scalar tensor (float32).
    """
    z = F.normalize(proj.float(), dim=-1)
    V = z.shape[0]
    idx = torch.triu_indices(V, V, offset=1, device=z.device)
    diffs = z[idx[0]] - z[idx[1]]          # (num_pairs, N, D)
    return diffs.norm(p=2, dim=-1).pow(alpha).mean()


def uniform_loss(proj: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """L_uniform from Wang & Isola (ICML 2020): log average Gaussian potential on S^{d-1}.

    Parameters
    ----------
    proj : (V, N, D) — stacked projections from all views
    t    : bandwidth of the Gaussian kernel (default 2)

    Returns
    -------
    Scalar tensor (float32).
    """
    z = F.normalize(proj.float().flatten(0, 1), dim=-1)   # (V*N, D)
    sq_pdist = torch.pdist(z, p=2).pow(2)                 # N*(N-1)/2 unique pairs
    return sq_pdist.mul(-t).exp().mean().log()


def sfrik_loss(
    proj: torch.Tensor,
    b1: float = 1.0,
    b2: float = 40.0,
    b3: float = 40.0,
) -> torch.Tensor:
    """SFRIK uniformity loss from Zheng et al. (ICLR 2023).

    Truncated rotation-invariant kernel MMD on S^{d-1} using the first three
    zonal spherical harmonics (Legendre polynomials P1, P2, P3):

        k(z, z') = b1·P1(u) + b2·P2(u) + b3·P3(u)

    where u = z^T z' and the polynomials are the dimension-q Gegenbauer variants:
        P1(u) = u
        P2(u) = (q·u² - 1) / (q - 1)
        P3(u) = ((q+2)·u³ - 3u) / (q - 1)

    Parameters
    ----------
    proj : (V, N, D) — stacked projections from all views
    b1   : weight on the degree-1 term (default 1.0)
    b2   : weight on the degree-2 term (default 40.0)
    b3   : weight on the degree-3 term (default 40.0)

    Returns
    -------
    Scalar tensor (float32).
    """
    z = F.normalize(proj.float().flatten(0, 1), dim=-1)   # (V*N, D)
    q = z.shape[-1]
    gram = z @ z.T                                         # (V*N, V*N)
    p1 = gram
    p2 = (q * gram.square() - 1) / (q - 1)
    p3 = ((q + 2) * gram.pow(3) - 3 * gram) / (q - 1)
    return (b1 * p1 + b2 * p2 + b3 * p3).mean()


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

    # Mask self-similarities so they never become the argmax
    sim.fill_diagonal_(float('-inf'))

    # Positive pair for sample i in [0,N) is i+N, and vice-versa
    labels = torch.cat([
        torch.arange(N, 2 * N, device=proj.device),
        torch.arange(0, N,     device=proj.device),
    ])
    return F.cross_entropy(sim, labels)
