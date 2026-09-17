# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Conformal Autoencoder (CAE) for intrinsic-dimension detection.

Faithful implementation of Algorithm 1 / Eq. (6) from Kevrekidis, Ahmad,
Maggioni, Villar & Kevrekidis, "Thinner Latent Spaces: Detecting Dimension and
Imposing Invariance with Conformal Autoencoders" (arXiv:2408.16138).

An over-specified ``n``-dimensional latent layer is regularised so that the
gradients of distinct latent coordinates (w.r.t. the input) are mutually
orthogonal::

    E_CAE = (1/N) Σ_i ||x_i - x̂_i||²                              (reconstruction)
          + α (1/N) Σ_i Σ_{j>k} |⟨∇ν_j(x_i), ∇ν_k(x_i)⟩|²        (orthogonality)

Under this objective the network exhibits *mode collapse*: redundant latent
coordinates have their input-gradient driven to zero, so the number of latent
coordinates with non-vanishing gradient norm reveals the intrinsic dimension of
the data manifold (locally, the tangent-space dimension).

In the bobblehead experiment (paper §3.3) the input to the CAE is a frozen
feature representation of the images (PCA-60 in the paper; here a frozen JEPA
backbone embedding).  Two figures rotating at independent frequencies trace a
2-torus, which embeds minimally in R^4, so ~4 latent coordinates stay active.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Architecture (Definition B.1 / Table 2: tanh layers followed by linear layers)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """MLP: ``n_tanh`` tanh-activated layers followed by ``n_linear`` linear layers.

    Matches the paper's bobblehead architecture ("5 Tanh-activated layers
    followed by two linear layers, all of width w").
    """

    def __init__(self, in_dim: int, out_dim: int, width: int, n_tanh: int = 5, n_linear: int = 2):
        super().__init__()
        if n_linear < 1:
            raise ValueError("n_linear must be >= 1 (the output projection).")
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(n_tanh):
            layers.append(nn.Linear(prev, width))
            layers.append(nn.Tanh())
            prev = width
        # Linear tail; only the final layer maps to out_dim.
        for i in range(n_linear):
            this_out = out_dim if i == n_linear - 1 else width
            layers.append(nn.Linear(prev, this_out))
            prev = this_out
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformalAutoencoder(nn.Module):
    """Encoder/decoder pair with an over-specified latent layer (Definition 2.2)."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        width: int = 256,
        n_tanh: int = 5,
        n_linear: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = FeedForward(input_dim, latent_dim, width, n_tanh, n_linear)
        self.decoder = FeedForward(latent_dim, input_dim, width, n_tanh, n_linear)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


# ---------------------------------------------------------------------------
# Orthogonality term (Eq. 6, second sum) via automatic differentiation
# ---------------------------------------------------------------------------

def latent_input_jacobian(model: ConformalAutoencoder, x: torch.Tensor) -> torch.Tensor:
    """Per-sample Jacobian ∂ν/∂x of the latent code w.r.t. the input.

    Args:
        x: ``(B, n)`` batch with ``requires_grad`` enabled.

    Returns:
        ``(B, L, n)`` tensor of latent-coordinate input-gradients.
    """
    z = model.encode(x)  # (B, L)
    L = z.shape[1]
    grads = []
    for j in range(L):
        # d(sum_b z[b, j]) / dx[b'] is non-zero only for b' == b (samples are
        # independent), so this recovers the per-sample gradient ∇ν_j(x_b).
        (g,) = torch.autograd.grad(z[:, j].sum(), x, create_graph=True, retain_graph=True)
        grads.append(g)
    return torch.stack(grads, dim=1)  # (B, L, n)


def orthogonality_loss(jac: torch.Tensor, norm: str = "l2") -> torch.Tensor:
    """Mean over the batch of the off-diagonal Gram penalty (Eq. 6, second term).

    ``norm="l2"`` gives the squared inner products Σ_{j>k} ⟨∇ν_j, ∇ν_k⟩² as in
    Eq. (6).  ``norm="l1"`` uses Σ_{j>k} |⟨∇ν_j, ∇ν_k⟩|, which the paper suggests
    (Algorithm 1 note, Example C.1) to more aggressively encourage lower-
    dimensional latent spaces.
    """
    # Gram matrix of latent gradients per sample: (B, L, L).
    gram = torch.bmm(jac, jac.transpose(1, 2))
    L = gram.shape[1]
    iu, ju = torch.triu_indices(L, L, offset=1, device=jac.device)
    pairs = gram[:, iu, ju]  # (B, n_pairs)
    if norm == "l1":
        return pairs.abs().sum(dim=1).mean()
    return (pairs ** 2).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Training (Algorithm 1)
# ---------------------------------------------------------------------------

@dataclass
class CAEConfig:
    latent_dim: int = 8
    width: int = 256
    n_tanh: int = 5
    n_linear: int = 2
    alpha: float = 1.0            # orthogonality weight (Eq. 6)
    ortho_norm: str = "l2"        # "l2" (Eq. 6) or "l1" (encourages lower dim)
    lr: float = 1e-3
    epochs: int = 2000
    batch_size: int = 256
    standardize: bool = True      # z-score inputs (paper rescales to unit cube)
    grad_norm_eval_batch: int = 1024
    active_frac: float = 0.1      # latent active if grad-norm > active_frac * max
    seed: int = 0
    log_every: int = 100
    patience: int = 0             # early stop if mean loss has not improved for
                                  # `patience` logged checkpoints (0 disables)
    min_delta: float = 1e-6       # minimum loss improvement to reset patience


@dataclass
class CAEResult:
    inferred_dim: int
    active_components: list[int]
    grad_norms: np.ndarray              # (L,) final E_x||∇ν_i||
    grad_norm_history: np.ndarray       # (n_logs, L)
    latent_variances: np.ndarray        # (L,)
    loss_history: np.ndarray            # (n_logs,)
    recon_history: np.ndarray           # (n_logs,)
    ortho_history: np.ndarray           # (n_logs,)
    epochs_logged: np.ndarray           # (n_logs,)
    latents: np.ndarray                 # (N, L) final latent codes (full dataset)
    final_recon: float
    extras: dict = field(default_factory=dict)


@torch.no_grad()
def _latent_codes(model: ConformalAutoencoder, X: torch.Tensor, device, batch_size: int) -> torch.Tensor:
    model.eval()
    out = []
    for (xb,) in DataLoader(TensorDataset(X), batch_size=batch_size):
        out.append(model.encode(xb.to(device)).cpu())
    return torch.cat(out, dim=0)


def _grad_norms(model: ConformalAutoencoder, X: torch.Tensor, device, n_eval: int) -> torch.Tensor:
    """E_x ||∇ν_i(x)|| per latent component, estimated on up to ``n_eval`` points."""
    model.eval()
    idx = torch.randperm(X.shape[0])[: min(n_eval, X.shape[0])]
    x = X[idx].to(device).clone().requires_grad_(True)
    jac = latent_input_jacobian(model, x)          # (B, L, n)
    return jac.norm(dim=2).mean(dim=0).detach().cpu()  # (L,)


def train_conformal_autoencoder(
    X: torch.Tensor,
    cfg: CAEConfig,
    device: torch.device,
) -> tuple[ConformalAutoencoder, CAEResult]:
    """Run Algorithm 1 on feature matrix ``X`` of shape ``(N, n)``."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    X = X.float()
    if cfg.standardize:
        mean = X.mean(dim=0, keepdim=True)
        std = X.std(dim=0, keepdim=True).clamp_min(1e-6)
        Xn = (X - mean) / std
    else:
        mean = torch.zeros(1, X.shape[1])
        std = torch.ones(1, X.shape[1])
        Xn = X

    model = ConformalAutoencoder(
        input_dim=Xn.shape[1], latent_dim=cfg.latent_dim,
        width=cfg.width, n_tanh=cfg.n_tanh, n_linear=cfg.n_linear,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loader = DataLoader(TensorDataset(Xn), batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    gn_hist, loss_hist, recon_hist, ortho_hist, ep_log = [], [], [], [], []
    best_loss = float("inf")
    stale = 0

    for epoch in range(cfg.epochs):
        model.train()
        ep_loss = ep_recon = ep_ortho = 0.0
        n_batches = 0
        for (xb,) in loader:
            xb = xb.to(device).requires_grad_(True)
            xhat, _ = model(xb)
            recon = F.mse_loss(xhat, xb)
            jac = latent_input_jacobian(model, xb)
            ortho = orthogonality_loss(jac, norm=cfg.ortho_norm)
            loss = recon + cfg.alpha * ortho

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            ep_loss += float(loss.detach())
            ep_recon += float(recon.detach())
            ep_ortho += float(ortho.detach())
            n_batches += 1

        if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
            gn = _grad_norms(model, Xn, device, cfg.grad_norm_eval_batch)
            gn_hist.append(gn.numpy())
            mean_loss = ep_loss / n_batches
            loss_hist.append(mean_loss)
            recon_hist.append(ep_recon / n_batches)
            ortho_hist.append(ep_ortho / n_batches)
            ep_log.append(epoch)

            if cfg.patience > 0:
                if best_loss - mean_loss > cfg.min_delta:
                    best_loss = mean_loss
                    stale = 0
                else:
                    stale += 1
                    if stale >= cfg.patience:
                        break

    # Final diagnostics.
    final_gn = _grad_norms(model, Xn, device, X.shape[0])
    Z = _latent_codes(model, Xn, device, cfg.batch_size)
    variances = Z.var(dim=0, unbiased=False)

    thresh = cfg.active_frac * float(final_gn.max())
    active = [i for i in range(cfg.latent_dim) if float(final_gn[i]) > thresh]

    result = CAEResult(
        inferred_dim=len(active),
        active_components=active,
        grad_norms=final_gn.numpy(),
        grad_norm_history=np.asarray(gn_hist),
        latent_variances=variances.numpy(),
        loss_history=np.asarray(loss_hist),
        recon_history=np.asarray(recon_hist),
        ortho_history=np.asarray(ortho_hist),
        epochs_logged=np.asarray(ep_log),
        latents=Z.numpy(),
        final_recon=float(recon_hist[-1]) if recon_hist else float("nan"),
        extras={"feature_mean": mean.numpy(), "feature_std": std.numpy()},
    )
    return model, result


# ---------------------------------------------------------------------------
# Quantitative recovery of the rotation angles (when labels are available)
# ---------------------------------------------------------------------------

def circular_recovery(latents: np.ndarray, angles: np.ndarray, active: list[int]) -> dict:
    """How well do the active latents recover each true rotation angle?

    For each angle column we linearly regress ``(cos θ, sin θ)`` on the active
    latent coordinates, decode a predicted angle via ``atan2``, and report the
    mean absolute angular error (radians) and the (cos,sin) R².  This is the
    natural circular analogue of the paper's "colour the embedding by the true
    angle" sanity check (Figs. 3, 6).

    Args:
        latents: ``(N, L)`` latent codes.
        angles:  ``(N,)`` or ``(N, k)`` true angles in radians.
        active:  indices of the active latent coordinates.
    """
    if angles.ndim == 1:
        angles = angles[:, None]
    Z = latents[:, active] if active else latents
    Z = np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=1)  # affine term

    out = {}
    for k in range(angles.shape[1]):
        theta = angles[:, k]
        target = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (N, 2)
        coef, *_ = np.linalg.lstsq(Z, target, rcond=None)
        pred = Z @ coef
        ss_res = ((target - pred) ** 2).sum()
        ss_tot = ((target - target.mean(0)) ** 2).sum()
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        pred_theta = np.arctan2(pred[:, 1], pred[:, 0])
        ang_err = np.abs(np.angle(np.exp(1j * (pred_theta - theta))))
        out[f"angle{k}"] = {
            "cos_sin_r2": float(r2),
            "mean_abs_angular_error_rad": float(ang_err.mean()),
            "mean_abs_angular_error_deg": float(np.degrees(ang_err.mean())),
        }
    return out
