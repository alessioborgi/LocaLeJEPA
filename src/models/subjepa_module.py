# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from src.losses import invariance_loss, subreg_loss
from src.models.jepa_module import JEPAModule


class SubJEPAModule(JEPAModule):
    """LightningModule for Sub-JEPA pretraining (Zhao et al., arXiv:2605.09241).

    Sub-JEPA replaces the full-dimensional Gaussian regularizer with Subspace
    Gaussian Regularization (``subreg``): projections are mapped through
    ``num_subspaces`` frozen row-orthonormal matrices and regularized within each
    low-dimensional subspace::

        loss = lamb * subreg_loss(proj, proj_matrices) + (1 - lamb) * invariance_loss(proj)

    The projection matrices are registered as a buffer so they move with the
    module across devices and are saved in checkpoints, but are never updated
    by the optimizer.

    The encoder can be any ``nn.Module`` that maps flattened image views
    ``(N * V, C, H, W)`` to embeddings ``(N * V, embedding_dim)``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer],
        projector: nn.Module | None = None,
        online_probe: nn.Module | None = None,
        embedding_dim: int = 512,
        projection_dim: int = 128,
        num_classes: int = 10,
        lamb: float = 0.02,
        num_subspaces: int = 4,
        num_global_crops: int | None = None,
        use_online_probe: bool = True,
        probe_lr: float = 1e-3,
        probe_weight_decay: float = 1e-7,
        warmup_epochs: int = 0,
        compile: bool = False,
    ) -> None:
        super().__init__(
            encoder=encoder,
            optimizer=optimizer,
            projector=projector,
            online_probe=online_probe,
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            num_classes=num_classes,
            lamb=lamb,
            num_global_crops=num_global_crops,
            use_online_probe=use_online_probe,
            probe_lr=probe_lr,
            probe_weight_decay=probe_weight_decay,
            warmup_epochs=warmup_epochs,
            compile=compile,
        )
        self.save_hyperparameters(ignore=["encoder", "projector", "online_probe", "optimizer"], logger=False)

        self.num_subspaces = num_subspaces

        # Frozen row-orthonormal projection matrices (K, ds, D).
        # Initialised on CPU here; Lightning moves them to the correct device automatically.
        K = num_subspaces
        ds = projection_dim // K
        if ds < 1:
            raise ValueError(
                f"num_subspaces={K} is too large for projection_dim={projection_dim}; need num_subspaces <= projection_dim."
            )
        matrices = []
        for _ in range(K):
            g = torch.randn(projection_dim, ds)
            # QR gives Q with orthonormal columns: (D, ds).
            q, _ = torch.linalg.qr(g)
            # Transpose to get orthonormal rows: (ds, D).
            matrices.append(q.T)
        self.register_buffer("proj_matrices", torch.stack(matrices, dim=0))

    def model_step(self, views: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        emb, proj = self.forward(views)

        inv_loss = invariance_loss(proj, num_global=self.num_global_crops)
        reg = subreg_loss(proj, self.proj_matrices)

        # Weighted combination: loss = lamb * reg + (1 - lamb) * inv.
        lejepa_loss = reg * self.lamb + inv_loss * (1.0 - self.lamb)

        cov = self._covariance_metrics(proj)

        return emb, {
            "loss": lejepa_loss,
            "lejepa_loss": lejepa_loss,
            "invariance_loss": inv_loss,
            "reg_loss": reg,
            "subreg_loss": reg,
            "inv_contrib": inv_loss * (1.0 - self.lamb),
            "reg_contrib": reg * self.lamb,
            **cov,
        }

    def _log_reg_metrics(self, metrics: dict[str, Any]) -> None:
        self.log("train/subreg_loss", metrics["subreg_loss"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/reg_contrib", metrics["reg_contrib"], sync_dist=True, on_step=True, on_epoch=True)


if __name__ == "__main__":
    _ = SubJEPAModule(
        encoder=nn.Identity(), projector=nn.Identity(), optimizer=torch.optim.AdamW, online_probe=nn.Identity()
    )
