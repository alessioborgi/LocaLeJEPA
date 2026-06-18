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
from lightning import LightningModule
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.losses import invariance_loss, sigreg_loss, subreg_loss, susreg_loss


class BundLeJEPAModule(LightningModule):
    """LightningModule for BundLeJEPA pretraining (LeJEPA + equivariance).

    BundLeJEPA extends LeJEPA with a third loss component that encourages
    *equivariance* to a chosen group of transformations, following Equivariant
    Self-Supervised Learning (E-SSL, Dangovski et al., ICLR 2022). The total
    objective combines three terms::

        loss = (1 - lamb) * inv_loss + lamb * reg_loss + eq_lambda * eq_loss

    where

    inv_loss : invariance loss -- identical to LeJEPA (pull every view toward
               the reference mean).
    reg_loss : distributional regularizer (``sigreg`` / ``susreg`` / ``subreg``)
               -- identical to LeJEPA. ``sigreg`` is the default; this term is a
               placeholder slated to change.
    eq_loss  : equivariance loss -- a predictor head ``p2`` predicts which
               transformation was applied to a view from its backbone embedding.
               We apply each of the ``num_transforms`` four-fold rotations to
               *all* views (not a single small crop as in the original E-SSL
               paper), encode them, and classify the rotation with cross entropy.

    Inference is unchanged from LeJEPA: a single view is encoded and the online
    probe / downstream head reads the backbone embedding directly.

    Supported regularization loss types
    -----------------------------------
    sigreg : Sketched Isotropic Gaussian Regularization (original LeJEPA).
    susreg : Sketched Uniform Spherical Regularization (SPHERE-JEPA).
    subreg : Subspace Gaussian Regularization (Sub-JEPA). Requires
             ``num_subspaces`` to be set. Uses frozen orthonormal projection
             matrices registered as a buffer.
    """

    def __init__(
        self,
        encoder: nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer],
        projector: nn.Module | None = None,
        equivariance_predictor: nn.Module | None = None,
        online_probe: nn.Module | None = None,
        embedding_dim: int = 512,
        projection_dim: int = 128,
        num_classes: int = 10,
        lamb: float = 0.02,
        loss_type: str = "sigreg",
        num_subspaces: int = 4,
        num_global_crops: int | None = None,
        use_equivariance: bool = True,
        eq_lambda: float = 0.4,
        num_transforms: int = 4,
        use_online_probe: bool = True,
        probe_lr: float = 1e-3,
        probe_weight_decay: float = 1e-7,
        warmup_epochs: int = 0,
        compile: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(
            ignore=["encoder", "projector", "equivariance_predictor", "online_probe", "optimizer"], logger=False
        )

        self.encoder = encoder
        self.projector = projector if projector is not None else nn.Identity()
        self.online_probe = online_probe if use_online_probe else None
        self.optimizer_fn = optimizer

        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.num_classes = num_classes
        self.lamb = lamb
        self.loss_type = loss_type
        self.num_subspaces = num_subspaces
        self.num_global_crops = num_global_crops
        self.use_equivariance = use_equivariance
        self.eq_lambda = eq_lambda
        self.num_transforms = num_transforms
        self.use_online_probe = use_online_probe
        self.probe_lr = probe_lr
        self.probe_weight_decay = probe_weight_decay
        self.warmup_epochs = warmup_epochs
        self.compile_model = compile

        # Equivariance predictor p2: maps a backbone embedding to logits over the
        # num_transforms transformation classes (e.g. 4 for four-fold rotations).
        if use_equivariance:
            self.equivariance_predictor = (
                equivariance_predictor
                if equivariance_predictor is not None
                else self._default_equivariance_predictor(embedding_dim, num_transforms)
            )
        else:
            self.equivariance_predictor = None

        # Sub-JEPA: frozen row-orthonormal projection matrices (K, ds, D).
        # Initialised on CPU here; Lightning moves them to the correct device
        # automatically when the module is transferred.
        if loss_type == "subreg":
            K = num_subspaces
            ds = projection_dim // K
            if ds < 1:
                raise ValueError(
                    f"num_subspaces={K} is too large for projection_dim={projection_dim}; need num_subspaces ≤ projection_dim."
                )
            matrices = []
            for _ in range(K):
                g = torch.randn(projection_dim, ds)
                # QR gives Q with orthonormal columns: (D, ds).
                q, _ = torch.linalg.qr(g)
                # Transpose to get orthonormal rows: (ds, D).
                matrices.append(q.T)
            self.register_buffer("proj_matrices", torch.stack(matrices, dim=0))
        else:
            self.register_buffer("proj_matrices", None)

    @staticmethod
    def _default_equivariance_predictor(embedding_dim: int, num_transforms: int) -> nn.Module:
        """Build the default E-SSL predictor head (2-layer MLP + linear classifier).

        Mirrors the predictor in Dangovski et al.: each hidden linear layer is
        followed by LayerNorm and ReLU, and a final linear head produces the
        per-transformation logits.
        """
        hidden = 2048
        return nn.Sequential(
            nn.Linear(embedding_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_transforms),
        )

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
        """Encode multi-view images and compute projector outputs for the LeJEPA loss.

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

    def equivariance_loss(self, views: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """E-SSL equivariance loss over *all* views.

        For every view we construct the ``num_transforms`` four-fold rotations,
        encode them with the shared backbone, and train the predictor ``p2`` to
        classify the applied rotation via cross entropy. Using all views (rather
        than the single small crop of the original E-SSL paper) gives the
        predictor every available augmentation of each image.

        Args:
            views: Tensor shaped ``(N, V, C, H, W)``. Spatial dims must be square
                so that ``rot90`` preserves the tensor shape.

        Returns:
            Scalar cross-entropy loss and scalar rotation-prediction accuracy.
        """
        batch_size, num_views = views.shape[:2]
        # Flatten all views into a single batch: (N * V, C, H, W).
        flat = views.flatten(0, 1)

        # Build the rotated views: rotation k by k * 90 degrees, stacked along batch.
        rotated = torch.cat(
            [torch.rot90(flat, k, dims=(-2, -1)) for k in range(self.num_transforms)],
            dim=0,
        )  # (T * N * V, C, H, W)
        # Per-image target = which rotation was applied.
        labels = torch.arange(self.num_transforms, device=views.device).repeat_interleave(batch_size * num_views)

        emb = self.encoder(rotated)  # (T * N * V, embedding_dim)
        logits = self.equivariance_predictor(emb)  # (T * N * V, num_transforms)

        eq_loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            eq_acc = (logits.argmax(dim=1) == labels).float().mean()
        return eq_loss, eq_acc

    def model_step(self, views: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute BundLeJEPA pretrain losses from multi-view images.

        Args:
            views: Tensor shaped ``(N, V, C, H, W)``.

        Returns:
            Backbone embeddings shaped ``(N * V, embedding_dim)`` and scalar loss metrics.
        """
        emb, proj = self.forward(views)

        # For SPHERE-JEPA both the invariance and regularisation losses operate on
        # normalised embeddings (approximating squared geodesic distance).
        proj_for_inv = F.normalize(proj, dim=-1) if self.loss_type == "susreg" else proj

        inv_loss = invariance_loss(proj_for_inv, num_global=self.num_global_crops)

        reg_metrics: dict[str, torch.Tensor] = {}
        if self.loss_type == "sigreg":
            reg = sigreg_loss(proj)
            reg_metrics["sigreg_loss"] = reg
        elif self.loss_type == "susreg":
            reg = susreg_loss(proj)
            reg_metrics["susreg_loss"] = reg
        elif self.loss_type == "subreg":
            reg = subreg_loss(proj, self.proj_matrices)
            reg_metrics["subreg_loss"] = reg
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type!r}. Choose 'sigreg', 'susreg', or 'subreg'.")

        # LeJEPA weighted combination: lamb * reg + (1 - lamb) * inv.
        lejepa_loss = reg * self.lamb + inv_loss * (1.0 - self.lamb)

        # E-SSL equivariance term, added on top of the LeJEPA objective.
        if self.equivariance_predictor is not None:
            eq_loss, eq_acc = self.equivariance_loss(views)
        else:
            eq_loss = torch.zeros((), device=lejepa_loss.device)
            eq_acc = torch.zeros((), device=lejepa_loss.device)

        eq_contrib = eq_loss * self.eq_lambda
        total_loss = lejepa_loss + eq_contrib

        with torch.no_grad():
            # Flatten views into a single batch for covariance diagnostics: (N*V, D).
            z_all = proj.flatten(0, 1)
            std_mean = z_all.std(dim=0).mean()
            norm_mean = z_all.norm(dim=1).mean()
            mean = z_all.mean(dim=0)
            z_centered = z_all - mean
            cov = z_centered.T @ z_centered / (z_all.size(0) - 1)
            cov_diag = torch.diag(cov)
            cov_offdiag = cov - torch.diag(cov_diag)
            mean_abs = mean.abs().mean()
            cov_diag_mean = cov_diag.mean()
            cov_offdiag_mean = cov_offdiag.abs().mean()
            cov_offdiag_sq_mean = cov_offdiag.square().mean()

        metrics = {
            "loss": total_loss,
            "bundlejepa_loss": total_loss,
            "lejepa_loss": lejepa_loss,
            "invariance_loss": inv_loss,
            "reg_loss": reg,
            "equivariance_loss": eq_loss,
            "equivariance_acc": eq_acc,
            "inv_contrib": inv_loss * (1.0 - self.lamb),
            "reg_contrib": reg * self.lamb,
            "eq_contrib": eq_contrib,
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

        self.log("train/bundlejepa_loss", metrics["bundlejepa_loss"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/lejepa_loss", metrics["lejepa_loss"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/invariance_loss", metrics["invariance_loss"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/loss", metrics["loss"], prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/inv_contrib", metrics["inv_contrib"], sync_dist=True, on_step=True, on_epoch=True)

        if self.loss_type == "sigreg":
            self.log("train/sigreg_loss", metrics["sigreg_loss"], sync_dist=True, on_step=True, on_epoch=True)
            self.log("train/reg_contrib", metrics["reg_contrib"], sync_dist=True, on_step=True, on_epoch=True)
        elif self.loss_type == "susreg":
            self.log("train/susreg_loss", metrics["susreg_loss"], sync_dist=True, on_step=True, on_epoch=True)
            self.log("train/reg_contrib", metrics["reg_contrib"], sync_dist=True, on_step=True, on_epoch=True)
        elif self.loss_type == "subreg":
            self.log("train/subreg_loss", metrics["subreg_loss"], sync_dist=True, on_step=True, on_epoch=True)
            self.log("train/reg_contrib", metrics["reg_contrib"], sync_dist=True, on_step=True, on_epoch=True)

        if self.equivariance_predictor is not None:
            self.log("train/equivariance_loss", metrics["equivariance_loss"], sync_dist=True, on_step=True, on_epoch=True)
            self.log("train/equivariance_acc", metrics["equivariance_acc"], sync_dist=True, on_step=True, on_epoch=True)
            self.log("train/eq_contrib", metrics["eq_contrib"], sync_dist=True, on_step=True, on_epoch=True)

        self.log("train/std_mean", metrics["std_mean"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/norm_mean", metrics["norm_mean"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/mean_abs", metrics["mean_abs"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/cov_diag_mean", metrics["cov_diag_mean"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/cov_offdiag_mean", metrics["cov_offdiag_mean"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/cov_offdiag_sq_mean", metrics["cov_offdiag_sq_mean"], sync_dist=True, on_step=True, on_epoch=True)

        loss = metrics["loss"]

        if self.online_probe is not None:
            labels = labels.to(emb.device, non_blocking=True)
            batch_size, num_views = views.shape[:2]
            emb_first_view = emb.reshape(batch_size, num_views, self.embedding_dim)[:, 0]
            y_hat = self.online_probe(emb_first_view.detach())

            metrics["probe_loss"] = F.cross_entropy(y_hat, labels)
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
            if self.equivariance_predictor is not None:
                self.equivariance_predictor = torch.compile(self.equivariance_predictor)
            if self.online_probe is not None:
                self.online_probe = torch.compile(self.online_probe)

    def optimizer_step(self, *args: Any, **kwargs: Any) -> None:
        super().optimizer_step(*args, **kwargs)
        # nGPT: keep encoder weight rows on the unit hypersphere after each optimizer step.
        if hasattr(self.encoder, "norm_weights_"):
            self.encoder.norm_weights_()

    def configure_optimizers(self) -> dict[str, Any]:
        backbone_params = list(self.encoder.parameters()) + list(self.projector.parameters())
        if self.equivariance_predictor is not None:
            backbone_params += list(self.equivariance_predictor.parameters())

        param_groups: list[dict[str, Any]] = [
            {"params": backbone_params},
        ] + (
            [{"params": self.online_probe.parameters(), "lr": self.probe_lr, "weight_decay": self.probe_weight_decay}]
            if self.online_probe is not None
            else []
        )

        optimizer = self.optimizer_fn(params=param_groups)
        scheduler = self._build_scheduler(optimizer)
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
    _ = BundLeJEPAModule(
        encoder=nn.Identity(), projector=nn.Identity(), optimizer=torch.optim.AdamW, online_probe=nn.Identity()
    )
