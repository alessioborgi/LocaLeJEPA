# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
    """

    def __init__(
        self,
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

        inv_contrib = inv_loss * self.invariance_loss_weight
        reg_contrib = reg * self.rdm_reg_loss_weight
        loss = inv_contrib + reg_contrib

        return emb, {
            "loss": loss,
            "lejepa_loss": loss,
            "invariance_loss": inv_loss,
            self.reg_name: reg,
            "inv_contrib": inv_contrib,
            "reg_contrib": reg_contrib,
            "l1_sparsity": l1_sparsity_metric(z.flatten(0, 1)),
            "l0_sparsity": l0_sparsity_metric(z.flatten(0, 1)),
            **self._covariance_metrics(z),
        }
