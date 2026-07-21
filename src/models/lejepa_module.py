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

from src.losses import invariance_loss, local_ballreg_loss, local_sigreg_loss, sigreg_loss, susreg_loss
from src.models.jepa_module import JEPAModule


class LeJEPAModule(JEPAModule):
    """LightningModule for LeJEPA pretraining (Balestriero & LeCun, arXiv:2309.02347).

    The default regularizer is Sketched Isotropic Gaussian Regularization
    (``sigreg``), the original LeJEPA objective. The regularizer can also be
    swapped with ``susreg``, ``local_sigreg``, or ``local_ballreg`` via
    ``loss_type`` while keeping the same LeJEPA training scaffold. The total
    loss is::

        loss = lamb * reg_loss(proj) + (1 - lamb) * invariance_loss(proj)

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
        loss_type: str = "sigreg",
        num_global_crops: int | None = None,
        local_sigreg_num_clusters: int = 8,
        local_sigreg_kmeans_iters: int = 10,
        local_sigreg_min_cluster_size: int = 4,
        local_sigreg_weighted: bool = False,
        local_sigreg_normalize_for_kmeans: bool = True,
        local_sigreg_kmeans_backend: str = "fast_pytorch_kmeans",
        local_ballreg_radius: float | None = None,
        local_ballreg_center_clusters: bool = True,
        local_ballreg_n_quad: int = 64,
        dynamic_num_clusters: bool = False,
        dynamic_num_clusters_start: int = 1,
        dynamic_num_clusters_max: int = 8,
        dynamic_num_clusters_step_epochs: int = 100,
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

        self.loss_type = loss_type
        self.local_sigreg_num_clusters = local_sigreg_num_clusters
        self.local_sigreg_kmeans_iters = local_sigreg_kmeans_iters
        self.local_sigreg_min_cluster_size = local_sigreg_min_cluster_size
        self.local_sigreg_weighted = local_sigreg_weighted
        self.local_sigreg_normalize_for_kmeans = local_sigreg_normalize_for_kmeans
        self.local_sigreg_kmeans_backend = local_sigreg_kmeans_backend
        self.local_ballreg_radius = local_ballreg_radius
        self.local_ballreg_center_clusters = local_ballreg_center_clusters
        self.local_ballreg_n_quad = local_ballreg_n_quad
        self.dynamic_num_clusters = dynamic_num_clusters
        self.dynamic_num_clusters_start = dynamic_num_clusters_start
        self.dynamic_num_clusters_max = dynamic_num_clusters_max
        self.dynamic_num_clusters_step_epochs = dynamic_num_clusters_step_epochs
        if self.dynamic_num_clusters:
            # Start the local-neighborhood count at its scheduled epoch-0 value so
            # the very first training step already uses the intended k.
            self.local_sigreg_num_clusters = self._scheduled_num_clusters(0)

    def _scheduled_num_clusters(self, epoch: int) -> int:
        """Step schedule for the local k-means neighborhood count k.

        ``k = min(max, start + epoch // step_epochs)``. With the defaults
        (start=1, step=100, max=8) this gives k=1 at epoch 0, k=2 at epoch 100,
        incrementing by one every 100 epochs and capping at k=8 from epoch 700 on.
        """
        k = self.dynamic_num_clusters_start + epoch // self.dynamic_num_clusters_step_epochs
        return int(min(self.dynamic_num_clusters_max, k))

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        if self.dynamic_num_clusters:
            self.local_sigreg_num_clusters = self._scheduled_num_clusters(self.current_epoch)
            self.log(
                "train/num_clusters", float(self.local_sigreg_num_clusters), sync_dist=True, on_epoch=True
            )

    def model_step(self, views: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        emb, proj = self.forward(views)

        # For SUSReg, match the SPHERE-JEPA convention: invariance operates on
        # normalized projections while the regularizer normalizes internally.
        proj_for_inv = F.normalize(proj, dim=-1) if self.loss_type == "susreg" else proj
        inv_loss = invariance_loss(proj_for_inv, num_global=self.num_global_crops)

        reg_metrics: dict[str, torch.Tensor] = {}
        if self.loss_type == "sigreg":
            reg = sigreg_loss(proj)
            reg_metrics["sigreg_loss"] = reg
        elif self.loss_type == "susreg":
            reg = susreg_loss(proj)
            reg_metrics["susreg_loss"] = reg
        elif self.loss_type == "local_sigreg":
            reg = local_sigreg_loss(
                proj,
                num_clusters=self.local_sigreg_num_clusters,
                num_global=self.num_global_crops,
                kmeans_iters=self.local_sigreg_kmeans_iters,
                min_cluster_size=self.local_sigreg_min_cluster_size,
                weighted=self.local_sigreg_weighted,
                normalize_for_kmeans=self.local_sigreg_normalize_for_kmeans,
                kmeans_backend=self.local_sigreg_kmeans_backend,
            )
            reg_metrics["local_sigreg_loss"] = reg
        elif self.loss_type == "local_ballreg":
            reg = local_ballreg_loss(
                proj,
                num_clusters=self.local_sigreg_num_clusters,
                num_global=self.num_global_crops,
                kmeans_iters=self.local_sigreg_kmeans_iters,
                min_cluster_size=self.local_sigreg_min_cluster_size,
                weighted=self.local_sigreg_weighted,
                normalize_for_kmeans=self.local_sigreg_normalize_for_kmeans,
                kmeans_backend=self.local_sigreg_kmeans_backend,
                radius=self.local_ballreg_radius,
                center_clusters=self.local_ballreg_center_clusters,
                n_quad=self.local_ballreg_n_quad,
            )
            reg_metrics["local_ballreg_loss"] = reg
        else:
            raise ValueError(
                f"Unknown loss_type: {self.loss_type!r}. "
                "Choose 'sigreg', 'susreg', 'local_sigreg', or 'local_ballreg'."
            )

        # Weighted combination: loss = lamb * reg + (1 - lamb) * inv.
        lejepa_loss = reg * self.lamb + inv_loss * (1.0 - self.lamb)

        cov = self._covariance_metrics(proj)

        return emb, {
            "loss": lejepa_loss,
            "lejepa_loss": lejepa_loss,
            "invariance_loss": inv_loss,
            "reg_loss": reg,
            "inv_contrib": inv_loss * (1.0 - self.lamb),
            "reg_contrib": reg * self.lamb,
            **cov,
            **reg_metrics,
        }

    def _log_reg_metrics(self, metrics: dict[str, Any]) -> None:
        reg_key = f"{self.loss_type}_loss"
        self.log(f"train/{reg_key}", metrics[reg_key], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/reg_contrib", metrics["reg_contrib"], sync_dist=True, on_step=True, on_epoch=True)


if __name__ == "__main__":
    _ = LeJEPAModule(
        encoder=nn.Identity(), projector=nn.Identity(), optimizer=torch.optim.AdamW, online_probe=nn.Identity()
    )
