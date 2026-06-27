# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared LightningModule base for all JEPA-style pretraining variants.

JEPAModule provides the common scaffold used by LeJEPA, SphereJEPA, SubJEPA,
and BundLeJEPA:
  - encode() and forward() for multi-view embedding
  - covariance diagnostics helper
  - training_step skeleton with pluggable _log_reg_metrics() hook
  - validation_step and test_step with online probe accuracy
  - optimizer + cosine scheduler with linear warmup
  - nGPT norm_weights_() call in optimizer_step

Subclasses must implement:
  - model_step(views): compute the loss, return (emb, metrics dict)
  - _log_reg_metrics(metrics): log regularizer-specific scalars to W&B

Subclasses may override:
  - _probe_training_loss(emb, views, labels, metrics): online probe in training
  - setup(), configure_optimizers() if extra components need compilation/params
"""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class JEPAModule(LightningModule):
    """Shared scaffold for JEPA-style self-supervised pretraining."""

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
        super().__init__()

        self.encoder = encoder
        self.projector = projector if projector is not None else nn.Identity()
        self.online_probe = online_probe if use_online_probe else None
        self.optimizer_fn = optimizer

        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.num_classes = num_classes
        self.lamb = lamb
        self.num_global_crops = num_global_crops
        self.use_online_probe = use_online_probe
        self.probe_lr = probe_lr
        self.probe_weight_decay = probe_weight_decay
        self.warmup_epochs = warmup_epochs
        self.compile_model = compile

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, views: torch.Tensor) -> torch.Tensor:
        """Encode multi-view images into embeddings.

        Args:
            views: (N, V, C, H, W)

        Returns:
            (N * V, embedding_dim)
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
        """Encode and project multi-view images.

        Args:
            views: (N, V, C, H, W)

        Returns:
            emb (N * V, embedding_dim) and proj (V, N, projection_dim)
        """
        batch_size, num_views = views.shape[:2]
        emb = self.encode(views)
        proj = self.projector(emb)
        if proj.ndim != 2 or proj.shape[-1] != self.projection_dim:
            raise RuntimeError(f"Projector must return (N * V, {self.projection_dim}), got {tuple(proj.shape)}")

        # Reshape to (V, N, D) layout expected by all loss functions.
        proj = proj.reshape(batch_size, num_views, self.projection_dim).transpose(0, 1)
        return emb, proj

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def model_step(self, views: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute pretrain losses. Subclasses must implement."""
        raise NotImplementedError

    def _log_reg_metrics(self, metrics: dict[str, torch.Tensor]) -> None:
        """Log regularizer-specific scalars. Subclasses override."""
        pass

    def _probe_training_loss(
        self,
        emb: torch.Tensor,
        views: torch.Tensor,
        labels: torch.Tensor,
        metrics: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute and log the online probe loss. Returns the scalar loss to add.

        Default: uses only the first-view embedding matched to original labels.
        LeJEPAModule overrides this to probe across all views with repeated labels.
        """
        batch_size, num_views = views.shape[:2]
        # Extract first-view embedding for the probe.
        emb_first = emb.reshape(batch_size, num_views, self.embedding_dim)[:, 0]
        y_hat = self.online_probe(emb_first.detach())
        probe_loss = F.cross_entropy(y_hat, labels)
        metrics["probe_loss"] = probe_loss
        self.log("train/probe_loss", probe_loss, sync_dist=True, on_step=True, on_epoch=True)
        return probe_loss

    # ------------------------------------------------------------------
    # Diagnostics helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _covariance_metrics(self, proj: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute embedding covariance diagnostics from (V, N, D) projections."""
        # Flatten all views into a single batch: (N*V, D).
        z_all = proj.flatten(0, 1)
        std_mean = z_all.std(dim=0).mean()
        norm_mean = z_all.norm(dim=1).mean()
        mean = z_all.mean(dim=0)
        z_centered = z_all - mean
        cov = z_centered.T @ z_centered / (z_all.size(0) - 1)
        cov_diag = torch.diag(cov)
        cov_offdiag = cov - torch.diag(cov_diag)
        return {
            "std_mean": std_mean,
            "norm_mean": norm_mean,
            "mean_abs": mean.abs().mean(),
            "cov_diag_mean": cov_diag.mean(),
            "cov_offdiag_mean": cov_offdiag.abs().mean(),
            "cov_offdiag_sq_mean": cov_offdiag.square().mean(),
        }

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        views, labels = batch
        emb, metrics = self.model_step(views)

        # Common logging shared by all JEPA variants.
        self.log("train/lejepa_loss", metrics["lejepa_loss"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/invariance_loss", metrics["invariance_loss"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/loss", metrics["loss"], prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/inv_contrib", metrics["inv_contrib"], sync_dist=True, on_step=True, on_epoch=True)

        # Regularizer-specific logging delegated to subclass.
        self._log_reg_metrics(metrics)

        # Covariance diagnostics.
        self.log("train/std_mean", metrics["std_mean"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/norm_mean", metrics["norm_mean"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/mean_abs", metrics["mean_abs"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/cov_diag_mean", metrics["cov_diag_mean"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/cov_offdiag_mean", metrics["cov_offdiag_mean"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/cov_offdiag_sq_mean", metrics["cov_offdiag_sq_mean"], sync_dist=True, on_step=True, on_epoch=True)

        loss = metrics["loss"]

        if self.online_probe is not None:
            labels = labels.to(emb.device, non_blocking=True)
            loss = loss + self._probe_training_loss(emb, views, labels, metrics)

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
        # nGPT: keep encoder weight rows on the unit hypersphere after each optimizer step.
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
        scheduler = self._build_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
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

        cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-3)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
