# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

<<<<<<< HEAD
"""Rectified LpJEPA pretraining module.

Rectified LpJEPA (Kuang et al., "Rectified LpJEPA: Joint-Embedding Predictive
Architectures with Sparse and Maximum-Entropy Representations", arXiv:2602.01456,
2026) generalizes LeJEPA by matching the *rectified* feature distribution to a
Rectified Generalized Gaussian (RGG) target via a two-sample sliced Wasserstein
loss (RDMReg) instead of the isotropic-Gaussian SIGReg of LeJEPA.

Two design ingredients distinguish it from LeJEPA:
  1. features are rectified, ``z = ReLU(projector(encoder(x)))``, so both the
     invariance term and the regularizer operate on non-negative features;
  2. the regularizer is RDMReg -- a sliced 2-Wasserstein distance to
     ``prod_i RGN_p(mu, sigma)`` -- because the RGG family is not closed under
     linear projections and therefore cannot use a closed-form characteristic
     function target the way SIGReg/SUSReg do.

Setting ``target_distribution='lp_distribution'`` disables rectification and
matches the plain Generalized Gaussian, recovering the LpJEPA baseline (which
reduces to LeJEPA at p=2). The loss follows the paper's two-weight form::

    loss = invariance_loss_weight * inv + rdm_reg_loss_weight * RDMReg
"""

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses import (
    invariance_loss,
    l0_sparsity_metric,
    l1_sparsity_metric,
    rdmreg_loss,
    sigma_gn,
)
from src.models.jepa_module import JEPAModule


class RectifiedLpJEPAModule(JEPAModule):
    """LightningModule for Rectified LpJEPA pretraining (arXiv:2602.01456).

    The encoder can be any ``nn.Module`` mapping flattened image views
    ``(N * V, C, H, W)`` to embeddings ``(N * V, embedding_dim)``. The projector
    output is rectified in-module, so a plain (non-ReLU-terminated) projector is
    expected -- matching the rest of the JEPA variants in this repo.
=======
from typing import Any

import torch
import torch.nn.functional as F

from src.losses import l0_sparsity_metric, l1_sparsity_metric
from src.models.lejepa_module import LeJEPAModule


class RectifiedLpJEPAModule(LeJEPAModule):
    """Rectified LpJEPA (arXiv:2602.01456): ``LeJEPAModule`` with the projector output
    rectified before both losses and absolute loss weights instead of ``lamb``::

        z = relu(proj)
        loss = invariance_loss_weight * inv_loss(z) + rdm_reg_loss_weight * reg_loss(z)

    ``reg_loss`` is ``rdmreg_loss`` (``model/reg_loss=rdmreg``). ``rectify=False`` gives plain LpJEPA.
>>>>>>> 504dbb4b1bdcef4de8af30d106b1e20689fb1f81
    """

    def __init__(
        self,
<<<<<<< HEAD
        encoder: nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer],
        projector: nn.Module | None = None,
        online_probe: nn.Module | None = None,
        embedding_dim: int = 512,
        projection_dim: int = 128,
        num_classes: int = 10,
        lamb: float = 0.02,
        num_global_crops: int | None = None,
        # --- Rectified LpJEPA / RDMReg hyperparameters ---
        target_distribution: str = "rectified_lp_distribution",
        lp_norm_parameter: float = 1.0,
        mean_shift_value: float = 0.0,
        mode_of_sigma: str = "sigma_GN",
        num_projections: int = 256,
        invariance_loss_weight: float = 25.0,
        rdm_reg_loss_weight: float = 125.0,
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

        if target_distribution not in ("rectified_lp_distribution", "lp_distribution"):
            raise ValueError(
                f"Unknown target_distribution: {target_distribution!r}. "
                "Choose 'rectified_lp_distribution' (Rectified LpJEPA) or 'lp_distribution' (LpJEPA)."
            )

        self.target_distribution = target_distribution
        self.lp_norm_parameter = lp_norm_parameter
        self.mean_shift_value = mean_shift_value
        self.mode_of_sigma = mode_of_sigma
        self.num_projections = num_projections
        self.invariance_loss_weight = invariance_loss_weight
        self.rdm_reg_loss_weight = rdm_reg_loss_weight
        # Match/rectify iff we target the Rectified Generalized Gaussian.
        self.rectify = target_distribution == "rectified_lp_distribution"

        # Target scale sigma. sigma_GN fixes the pre-rectification GN variance to
        # 1 (the paper default). sigma_RGN (unit post-rectification variance)
        # requires a bisection solve on the RGG variance; it is intentionally not
        # wired here to avoid the mpmath dependency -- pass an explicit float via
        # mode_of_sigma to override.
        if mode_of_sigma == "sigma_GN":
            self.chosen_sigma = sigma_gn(lp_norm_parameter)
        else:
            try:
                self.chosen_sigma = float(mode_of_sigma)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"mode_of_sigma must be 'sigma_GN' or a float-like scale, got {mode_of_sigma!r}."
                ) from exc

    def model_step(self, views: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        emb, proj = self.forward(views)

        # Rectify features so invariance and RDMReg both operate on non-negative
        # representations (Rectified LpJEPA); left untouched for the LpJEPA case.
        z = F.relu(proj) if self.rectify else proj

        inv_loss = invariance_loss(z, num_global=self.num_global_crops)
        reg = rdmreg_loss(
            z,
            lp_norm_parameter=self.lp_norm_parameter,
            mean_shift_value=self.mean_shift_value,
            sigma=self.chosen_sigma,
            num_projections=self.num_projections,
            rectify_target=self.rectify,
        )
=======
        *args: Any,
        rectify: bool = True,
        invariance_loss_weight: float = 25.0,
        rdm_reg_loss_weight: float = 125.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.rectify = rectify
        self.invariance_loss_weight = invariance_loss_weight
        self.rdm_reg_loss_weight = rdm_reg_loss_weight

    def model_step(self, views: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        emb, proj = self.forward(views)
        z = F.relu(proj) if self.rectify else proj
        inv_loss = self.inv_loss(z)
        reg = self.reg_loss(z)
>>>>>>> 504dbb4b1bdcef4de8af30d106b1e20689fb1f81

        inv_contrib = inv_loss * self.invariance_loss_weight
        reg_contrib = reg * self.rdm_reg_loss_weight
        loss = inv_contrib + reg_contrib

<<<<<<< HEAD
        cov = self._covariance_metrics(z)

=======
>>>>>>> 504dbb4b1bdcef4de8af30d106b1e20689fb1f81
        return emb, {
            "loss": loss,
            "lejepa_loss": loss,
            "invariance_loss": inv_loss,
<<<<<<< HEAD
            "reg_loss": reg,
            "rdmreg_loss": reg,
=======
            self.reg_name: reg,
>>>>>>> 504dbb4b1bdcef4de8af30d106b1e20689fb1f81
            "inv_contrib": inv_contrib,
            "reg_contrib": reg_contrib,
            "l1_sparsity": l1_sparsity_metric(z.flatten(0, 1)),
            "l0_sparsity": l0_sparsity_metric(z.flatten(0, 1)),
<<<<<<< HEAD
            **cov,
        }

    def _log_reg_metrics(self, metrics: dict[str, Any]) -> None:
        self.log("train/rdmreg_loss", metrics["rdmreg_loss"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/reg_contrib", metrics["reg_contrib"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/l1_sparsity", metrics["l1_sparsity"], sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/l0_sparsity", metrics["l0_sparsity"], sync_dist=True, on_step=True, on_epoch=True)


if __name__ == "__main__":
    _ = RectifiedLpJEPAModule(
        encoder=nn.Identity(), projector=nn.Identity(), optimizer=torch.optim.AdamW, online_probe=nn.Identity()
    )
=======
            **self._covariance_metrics(z),
        }
>>>>>>> 504dbb4b1bdcef4de8af30d106b1e20689fb1f81
