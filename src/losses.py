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
