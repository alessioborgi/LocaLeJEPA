# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses import invariance_loss, susreg_loss
from src.models.jepa_module import JEPAModule


class SphereJEPAModule(JEPAModule):
    """LightningModule for SPHERE-JEPA pretraining (Nicollier et al., arXiv:2605.26900).

    SPHERE-JEPA replaces the isotropic Gaussian regularizer with Sketched
    Uniform Spherical Regularization (``susreg``), pushing the embedding
    distribution toward the uniform measure on the unit hypersphere. Both the
    invariance and regularization losses operate on L2-normalized projections::

        proj_n  = F.normalize(proj, dim=-1)
        loss    = lamb * susreg_loss(proj) + (1 - lamb) * invariance_loss(proj_n)

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

    def model_step(self, views: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        emb, proj = self.forward(views)

        # Both the invariance and regularization losses operate on normalized
        # embeddings (approximating squared geodesic distance on the sphere).
        proj_for_inv = F.normalize(proj, dim=-1)

        inv_loss = invariance_loss(proj_for_inv, num_global=self.num_global_crops)
        reg = susreg_loss(proj)

        # Weighted combination: loss = lamb * reg + (1 - lamb) * inv.
        lejepa_loss = reg * self.lamb + inv_loss * (1.0 - self.lamb)

        cov = self._covariance_metrics(proj)

        return emb, {
            "loss": lejepa_loss,
            "lejepa_loss": lejepa_loss,
            "invariance_loss": inv_loss,
            "reg_loss": reg,
            "susreg_loss": reg,
            "inv_contrib": inv_loss * (1.0 - self.lamb),
            "reg_contrib": reg * self.lamb,
            **cov,
        }

    def _log_reg_metrics(self, metrics: dict[str, Any]) -> None:
        self.log("train/susreg_loss", metrics["susreg_loss"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/reg_contrib", metrics["reg_contrib"], sync_dist=True, on_step=True, on_epoch=True)


if __name__ == "__main__":
    _ = SphereJEPAModule(
        encoder=nn.Identity(), projector=nn.Identity(), optimizer=torch.optim.AdamW, online_probe=nn.Identity()
    )
