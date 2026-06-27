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

from src.losses import invariance_loss, sigreg_loss, subreg_loss, susreg_loss
from src.models.jepa_module import JEPAModule


class BundLeJEPAModule(JEPAModule):
    """LightningModule for BundLeJEPA pretraining (LeJEPA + equivariance).

    BundLeJEPA extends LeJEPA with an equivariance loss that encourages the
    encoder to be sensitive to a chosen group of transformations, following
    Equivariant Self-Supervised Learning (E-SSL, Dangovski et al., ICLR 2022).
    The total objective is::

        loss = (1 - lamb) * inv_loss + lamb * reg_loss + eq_lambda * eq_loss

    Supported regularizers (``loss_type``):
      sigreg : Sketched Isotropic Gaussian Regularization (default, original LeJEPA).
      susreg : Sketched Uniform Spherical Regularization (SPHERE-JEPA).
      subreg : Subspace Gaussian Regularization (Sub-JEPA), requires ``num_subspaces``.

    The equivariance predictor ``p2`` classifies which of ``num_transforms``
    four-fold rotations was applied to each encoded view. The encoder, projector,
    and equivariance predictor are optimized jointly.
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
        self.save_hyperparameters(
            ignore=["encoder", "projector", "equivariance_predictor", "online_probe", "optimizer"], logger=False
        )

        self.loss_type = loss_type
        self.num_subspaces = num_subspaces
        self.use_equivariance = use_equivariance
        self.eq_lambda = eq_lambda
        self.num_transforms = num_transforms

        # Equivariance predictor p2: maps backbone embeddings to rotation logits.
        if use_equivariance:
            self.equivariance_predictor = (
                equivariance_predictor
                if equivariance_predictor is not None
                else self._default_equivariance_predictor(embedding_dim, num_transforms)
            )
        else:
            self.equivariance_predictor = None

        # Sub-JEPA: frozen row-orthonormal projection matrices (K, ds, D).
        # Initialised on CPU here; Lightning moves them to the correct device automatically.
        if loss_type == "subreg":
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
        else:
            self.register_buffer("proj_matrices", None)

    @staticmethod
    def _default_equivariance_predictor(embedding_dim: int, num_transforms: int) -> nn.Module:
        """Build the default E-SSL predictor head (2-layer MLP + linear classifier)."""
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

    def equivariance_loss(self, views: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """E-SSL equivariance loss over all views.

        Applies ``num_transforms`` four-fold rotations to every view, encodes
        them, and trains the predictor p2 to classify the rotation via cross
        entropy. Spatial dims must be square so that rot90 preserves shape.

        Args:
            views: (N, V, C, H, W)

        Returns:
            Scalar cross-entropy loss and scalar rotation-prediction accuracy.
        """
        batch_size, num_views = views.shape[:2]
        # Flatten all views into a single batch: (N * V, C, H, W).
        flat = views.flatten(0, 1)

        # Stack rotation-k copies along the batch dimension: (T * N * V, C, H, W).
        rotated = torch.cat(
            [torch.rot90(flat, k, dims=(-2, -1)) for k in range(self.num_transforms)],
            dim=0,
        )
        # Per-image target label = which rotation was applied.
        labels = torch.arange(self.num_transforms, device=views.device).repeat_interleave(batch_size * num_views)

        emb = self.encoder(rotated)
        logits = self.equivariance_predictor(emb)

        eq_loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            eq_acc = (logits.argmax(dim=1) == labels).float().mean()
        return eq_loss, eq_acc

    def model_step(self, views: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        emb, proj = self.forward(views)

        # For SPHERE-JEPA the invariance loss operates on normalized embeddings.
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

        # LeJEPA combination: lamb * reg + (1 - lamb) * inv.
        lejepa_loss = reg * self.lamb + inv_loss * (1.0 - self.lamb)

        # E-SSL equivariance term.
        if self.equivariance_predictor is not None:
            eq_loss, eq_acc = self.equivariance_loss(views)
        else:
            eq_loss = torch.zeros((), device=lejepa_loss.device)
            eq_acc = torch.zeros((), device=lejepa_loss.device)

        eq_contrib = eq_loss * self.eq_lambda
        total_loss = lejepa_loss + eq_contrib

        cov = self._covariance_metrics(proj)

        return emb, {
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
            **cov,
            **reg_metrics,
        }

    def _log_reg_metrics(self, metrics: dict[str, Any]) -> None:
        self.log("train/bundlejepa_loss", metrics["bundlejepa_loss"], sync_dist=True, on_step=True, on_epoch=True)

        reg_key = f"{self.loss_type}_loss"
        self.log(f"train/{reg_key}", metrics[reg_key], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/reg_contrib", metrics["reg_contrib"], sync_dist=True, on_step=True, on_epoch=True)

        if self.equivariance_predictor is not None:
            self.log("train/equivariance_loss", metrics["equivariance_loss"], sync_dist=True, on_step=True, on_epoch=True)
            self.log("train/equivariance_acc", metrics["equivariance_acc"], sync_dist=True, on_step=True, on_epoch=True)
            self.log("train/eq_contrib", metrics["eq_contrib"], sync_dist=True, on_step=True, on_epoch=True)

    def setup(self, stage: str | None = None) -> None:
        super().setup(stage)
        # Also compile the equivariance predictor if requested.
        if self.compile_model and stage == "fit" and self.equivariance_predictor is not None:
            self.equivariance_predictor = torch.compile(self.equivariance_predictor)

    def configure_optimizers(self) -> dict[str, Any]:
        # Include equivariance_predictor parameters in the backbone group.
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
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


if __name__ == "__main__":
    _ = BundLeJEPAModule(
        encoder=nn.Identity(), projector=nn.Identity(), optimizer=torch.optim.AdamW, online_probe=nn.Identity()
    )
