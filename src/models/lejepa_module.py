# Copyright (c) 2026 Pavel Snopov
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.losses import align_loss, invariance_loss, sfrik_loss, sigreg_loss, uniform_loss


class LeJEPAModule(LightningModule):
    """LightningModule for LeJEPA pretraining.

    The module composes a Hydra-instantiated image encoder with a projector head.
    The encoder can be any ``nn.Module`` as long as it maps flattened image views
    ``(N * V, C, H, W)`` to embeddings ``(N * V, embedding_dim)``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        projector: nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer],
        online_probe: nn.Module | None = None,
        embedding_dim: int = 512,
        projection_dim: int = 128,
        num_classes: int = 10,
        lamb: float = 0.02,
        loss_type: str = "sigreg",
        alpha: float = 2.0,
        t: float = 2.0,
        b1: float = 1.0,
        b2: float = 40.0,
        b3: float = 40.0,
        num_global_crops: int | None = None,
        use_online_probe: bool = True,
        probe_lr: float = 1e-3,
        probe_weight_decay: float = 1e-7,
        warmup_epochs: int = 0,
        compile: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["encoder", "projector", "online_probe", "optimizer"], logger=False)

        self.encoder = encoder
        self.projector = projector
        self.online_probe = online_probe if use_online_probe else None
        self.optimizer_fn = optimizer

        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.num_classes = num_classes
        self.lamb = lamb
        self.loss_type = loss_type
        self.alpha = alpha
        self.t = t
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.num_global_crops = num_global_crops
        self.use_online_probe = use_online_probe
        self.probe_lr = probe_lr
        self.probe_weight_decay = probe_weight_decay
        self.warmup_epochs = warmup_epochs
        self.compile_model = compile

    def encode(self, views: torch.Tensor) -> torch.Tensor:
        """Encode multi-view images into embeddings.

        Args:
            views: Tensor shaped ``(N, V, C, H, W)``.

        Returns:
            Tensor shaped ``(N * V, embedding_dim)``.
        """
        if views.ndim != 5:
            raise RuntimeError(f"Expected views with shape (N, V, C, H, W), got {tuple(views.shape)}")

        batch_size, num_views = views.shape[:2]
        flat_views = views.flatten(0, 1)

        emb = self.encoder(flat_views)

        expected_shape = (batch_size * num_views, self.embedding_dim)
        if emb.shape != expected_shape:
            raise RuntimeError(f"Expected embeddings with shape (N * V, {self.embedding_dim}), got {tuple(emb.shape)}")

        return emb

    def forward(self, views: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode multi-view images and compute projector outputs for LeJEPA loss.

        Args:
            views: Tensor shaped ``(N, V, C, H, W)``.

        Returns:
            ``emb`` shaped ``(N * V, embedding_dim)`` and ``proj`` shaped ``(V, N, projection_dim)``.
        """
        batch_size, num_views = views.shape[:2]
        emb = self.encode(views)
        proj = self.projector(emb)
        if proj.ndim != 2 or proj.shape[-1] != self.projection_dim:
            raise RuntimeError(f"Projector must return (N * V, {self.projection_dim}), got {tuple(proj.shape)}")

        proj = proj.reshape(batch_size, num_views, self.projection_dim).transpose(0, 1)
        return emb, proj

    def model_step(self, views: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute LeJEPA pretrain losses from multi-view images.

        Args:
            views: Tensor shaped ``(N, V, C, H, W)``.

        Returns:
            Dictionary containing:
            Backbone embeddings shaped ``(N * V, embedding_dim)`` and scalar loss metrics.
        """
        emb, proj = self.forward(views)

        inv_loss = invariance_loss(proj, num_global=self.num_global_crops)

        # ? Why do we compute reg loss on proj instead of emb? In le-wm paper they compute it over emb
        reg_metrics: dict[str, torch.Tensor] = {}
        if self.loss_type == "sigreg":
            reg = sigreg_loss(proj)
            reg_metrics["sigreg_loss"] = reg
        elif self.loss_type == "auh":
            a_loss = align_loss(proj, alpha=self.alpha)
            u_loss = uniform_loss(proj, t=self.t)
            reg = a_loss + u_loss
            reg_metrics["align_loss"] = a_loss
            reg_metrics["uniform_loss"] = u_loss
        elif self.loss_type == "sfrik":
            reg = sfrik_loss(proj, b1=self.b1, b2=self.b2, b3=self.b3)
            reg_metrics["sfrik_loss"] = reg
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type!r}. Choose 'sigreg', 'auh', or 'sfrik'.")

        lejepa_loss = reg * self.lamb + inv_loss * (1.0 - self.lamb)

        with torch.no_grad():
            z_all = proj.flatten(0, 1)  # (N*V, D)
            std_mean = z_all.std(dim=0).mean()
            norm_mean = z_all.norm(dim=1).mean()
            mean = z_all.mean(dim=0)
            z_centered = z_all - mean
            cov = z_centered.T @ z_centered / (z_all.size(0) - 1)  # (D, D)
            cov_diag = torch.diag(cov)
            cov_offdiag = cov - torch.diag(cov_diag)
            mean_abs = mean.abs().mean()
            cov_diag_mean = cov_diag.mean()
            cov_offdiag_mean = cov_offdiag.abs().mean()
            cov_offdiag_sq_mean = cov_offdiag.square().mean()

        metrics = {
            "loss": lejepa_loss,
            "lejepa_loss": lejepa_loss,
            "invariance_loss": inv_loss,
            "reg_loss": reg,
            "inv_contrib": inv_loss * (1.0 - self.lamb),
            "reg_contrib": reg * self.lamb,
            "std_mean": std_mean,
            "norm_mean": norm_mean,
            "mean_abs": mean_abs,
            "cov_diag_mean": cov_diag_mean,
            "cov_offdiag_mean": cov_offdiag_mean,
            "cov_offdiag_sq_mean": cov_offdiag_sq_mean,
            **reg_metrics,
        }

        return emb, metrics

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        views, labels = batch
        emb, metrics = self.model_step(views)

        self.log("train/lejepa_loss", metrics["lejepa_loss"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/invariance_loss", metrics["invariance_loss"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/loss", metrics["loss"], prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/inv_contrib", metrics["inv_contrib"], sync_dist=True, on_step=True, on_epoch=True)
        if self.loss_type == "sigreg":
            self.log("train/sigreg_loss", metrics["sigreg_loss"], sync_dist=True, on_step=True, on_epoch=True)
            self.log("train/sigreg_contrib", metrics["reg_contrib"], sync_dist=True, on_step=True, on_epoch=True)
        elif self.loss_type == "auh":
            self.log("train/align_loss", metrics["align_loss"], sync_dist=True, on_step=True, on_epoch=True)
            self.log("train/uniform_loss", metrics["uniform_loss"], sync_dist=True, on_step=True, on_epoch=True)
            self.log("train/auh_contrib", metrics["reg_contrib"], sync_dist=True, on_step=True, on_epoch=True)
        elif self.loss_type == "sfrik":
            self.log("train/sfrik_loss", metrics["sfrik_loss"], sync_dist=True, on_step=True, on_epoch=True)
            self.log("train/sfrik_contrib", metrics["reg_contrib"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/std_mean", metrics["std_mean"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/norm_mean", metrics["norm_mean"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/mean_abs", metrics["mean_abs"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/cov_diag_mean", metrics["cov_diag_mean"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/cov_offdiag_mean", metrics["cov_offdiag_mean"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/cov_offdiag_sq_mean", metrics["cov_offdiag_sq_mean"], sync_dist=True, on_step=True, on_epoch=True)

        loss = metrics["loss"]

        if self.online_probe is not None:
            labels = labels.to(emb.device, non_blocking=True)
            y_rep = labels.repeat_interleave(views.shape[1])
            y_hat = self.online_probe(emb.detach())

            metrics["probe_loss"] = F.cross_entropy(y_hat, y_rep)
            self.log("train/probe_loss", metrics["probe_loss"], sync_dist=True, on_step=True, on_epoch=True)
            loss += metrics["probe_loss"]

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        if self.online_probe is not None:
            views, labels = batch
            emb = self.encode(views)
            labels = labels.to(emb.device, non_blocking=True)
            y_hat = self.online_probe(emb)

            probe_acc = (y_hat.argmax(dim=1) == labels).float().mean()
            self.log("val/probe_acc", probe_acc, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        if self.online_probe is not None:
            views, labels = batch
            emb = self.encode(views)
            labels = labels.to(emb.device, non_blocking=True)
            y_hat = self.online_probe(emb)

            probe_acc = (y_hat.argmax(dim=1) == labels).float().mean()
            self.log("test/probe_acc", probe_acc, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)

    def setup(self, stage: str | None = None) -> None:
        if self.compile_model and stage == "fit":
            self.encoder = torch.compile(self.encoder)
            self.projector = torch.compile(self.projector)
            if self.online_probe is not None:
                self.online_probe = torch.compile(self.online_probe)

    def optimizer_step(self, *args: Any, **kwargs: Any) -> None:
        super().optimizer_step(*args, **kwargs)
        # nGPT: snap NormLinear weight rows back to S^{d-1} after every optimiser update
        if hasattr(self.encoder, "norm_weights_"):
            self.encoder.norm_weights_()

    def configure_optimizers(self) -> dict[str, Any]:
        param_groups: list[dict[str, Any]] = [
            {"params": list(self.encoder.parameters()) + list(self.projector.parameters())},
        ] + (
            [{"params": self.online_probe.parameters(), "lr": self.probe_lr, "weight_decay": self.probe_weight_decay}]
            if self.online_probe is not None
            else []
        )

        optimizer = self.optimizer_fn(params=param_groups)
        scheduler = self._build_scheduler(optimizer)  # TODO: this should be configurable as well
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _build_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        max_epochs = int(self.trainer.max_epochs) if self.trainer.max_epochs is not None else 1
        steps_per_epoch = max(1, total_steps // max_epochs)
        warmup_steps = min(total_steps, steps_per_epoch * self.warmup_epochs)

        if warmup_steps <= 0:
            return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-3)

        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
        if warmup_steps >= total_steps:
            return warmup

        cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-3,
        )
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


if __name__ == "__main__":
    _ = LeJEPAModule(encoder=nn.Identity(), projector=nn.Identity(), optimizer=torch.optim.AdamW, online_probe=nn.Identity())
