# Copyright (c) 2026 Pavel Snopov
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.nn.init import trunc_normal_

from src.models.components.tensor_ops import apply_masks, repeat_interleave_batch


class IJEPAModule(LightningModule):
    """LightningModule for I-JEPA pretraining.

    The batch contract follows the original I-JEPA mask collator:
    ``(udata, masks_enc, masks_pred)``, where ``udata[0]`` is the image tensor.
    """

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer],
        ema: Sequence[float] = (0.996, 1.0),
        lr: float = 1e-3,
        start_lr: float = 2e-4,
        final_lr: float = 1e-6,
        weight_decay: float = 0.04,
        final_weight_decay: float = 0.4,
        warmup_epochs: int = 40,
        ipe_scale: float = 1.0,
        compile: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["encoder", "predictor"], logger=False)

        self.encoder = encoder
        self.predictor = predictor
        self.target_encoder = copy.deepcopy(self.encoder)
        self.optimizer_fn = optimizer

        self.ema = tuple(float(v) for v in ema)
        self.lr = lr
        self.start_lr = start_lr
        self.final_lr = final_lr
        self.weight_decay = weight_decay
        self.final_weight_decay = final_weight_decay
        self.warmup_epochs = warmup_epochs
        self.ipe_scale = ipe_scale
        self.compile_model = compile
        self.total_schedule_steps = 1
        self.warmup_steps = 0

        self._init_weights()

        self.target_encoder.requires_grad_(False)

        self.register_buffer("scheduler_step", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("ema_step", torch.zeros((), dtype=torch.long), persistent=True)

    def forward_target(
        self,
        images: torch.Tensor,
        masks_enc: list[torch.Tensor],
        masks_pred: list[torch.Tensor],
    ) -> torch.Tensor:
        with torch.no_grad():
            target = self.target_encoder(images)
            target = F.layer_norm(target, (target.size(-1),))
            batch_size = target.size(0)
            target = apply_masks(target, masks_pred)
            target = repeat_interleave_batch(target, batch_size, repeat=len(masks_enc))
        return target

    def model_step(
        self,
        images: torch.Tensor,
        masks_enc: list[torch.Tensor],
        masks_pred: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        target = self.forward_target(images, masks_enc, masks_pred)
        context = self.encoder(images, masks_enc)
        pred = self.predictor(context, masks_enc, masks_pred)
        loss = F.smooth_l1_loss(pred, target)
        return {"loss": loss}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        udata, masks_enc, masks_pred = batch
        images = udata[0] if isinstance(udata, (tuple, list)) else udata

        masks_enc = list(masks_enc)
        masks_pred = list(masks_pred)

        metrics = self.model_step(images, masks_enc, masks_pred)
        loss = metrics["loss"]

        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/mask_enc", float(masks_enc[0].shape[-1]), sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/mask_pred", float(masks_pred[0].shape[-1]), sync_dist=True, on_step=True, on_epoch=True)

        return loss

    def setup(self, stage: str | None = None) -> None:
        if self.compile_model and stage == "fit":
            self.encoder = torch.compile(self.encoder)
            self.predictor = torch.compile(self.predictor)
            self.target_encoder = torch.compile(self.target_encoder)

    def on_fit_start(self) -> None:
        total_steps = int(self.trainer.estimated_stepping_batches)
        max_epochs = self.trainer.max_epochs
        if max_epochs is None or max_epochs <= 0:
            raise RuntimeError("I-JEPA warmup_epochs requires Trainer.max_epochs to be a positive integer.")
        steps_per_epoch = max(1, total_steps // max_epochs)
        self.warmup_steps = self.warmup_epochs * steps_per_epoch
        self.total_schedule_steps = max(1, int(self.ipe_scale * total_steps))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        encoder_decay = [param for name, param in self.encoder.named_parameters() if "bias" not in name and param.ndim != 1]
        predictor_decay = [param for name, param in self.predictor.named_parameters() if "bias" not in name and param.ndim != 1]
        encoder_no_decay = [param for name, param in self.encoder.named_parameters() if "bias" in name or param.ndim == 1]
        predictor_no_decay = [param for name, param in self.predictor.named_parameters() if "bias" in name or param.ndim == 1]

        return self.optimizer_fn(
            params=[
                {"params": encoder_decay, "lr": self.start_lr, "weight_decay": self.weight_decay},
                {"params": predictor_decay, "lr": self.start_lr, "weight_decay": self.weight_decay},
                {"params": encoder_no_decay, "lr": self.start_lr, "weight_decay": 0.0, "WD_exclude": True},
                {"params": predictor_no_decay, "lr": self.start_lr, "weight_decay": 0.0, "WD_exclude": True},
            ]
        )

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        optimizer = self.optimizers()
        step = int(self.scheduler_step.item()) + 1  # type: ignore
        if step < self.warmup_steps:
            progress = float(step) / float(max(1, self.warmup_steps))
            lr = self.start_lr + progress * (self.lr - self.start_lr)
        else:
            progress = float(step - self.warmup_steps) / float(max(1, self.total_schedule_steps - self.warmup_steps))
            lr = max(self.final_lr, self.final_lr + (self.lr - self.final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)))

        progress = float(step) / float(max(1, self.total_schedule_steps))
        wd = self.final_weight_decay + (self.weight_decay - self.final_weight_decay) * 0.5 * (1.0 + math.cos(math.pi * progress))
        wd = max(self.final_weight_decay, wd) if self.final_weight_decay <= self.weight_decay else min(self.final_weight_decay, wd)

        for group in optimizer.param_groups:  # type: ignore
            group["lr"] = lr
            if not group.get("WD_exclude", False):
                group["weight_decay"] = wd
        self.scheduler_step.add_(1)  # type: ignore

        self.log("train/lr", lr, sync_dist=True, on_step=True, on_epoch=False)
        self.log("train/weight_decay", wd, sync_dist=True, on_step=True, on_epoch=False)

    def on_before_zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        step = min(int(self.ema_step.item()), self.total_schedule_steps)  # type: ignore
        momentum = self.ema[0] + step * (self.ema[1] - self.ema[0]) / float(max(1, self.total_schedule_steps))
        with torch.no_grad():
            for encoder_param, target_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                target_param.data.mul_(momentum).add_((1.0 - momentum) * encoder_param.detach().data)
        self.ema_step.add_(1)  # type: ignore
        self.log("train/ema_momentum", momentum, sync_dist=True, on_step=True, on_epoch=False)

    def _init_weights(self) -> None:
        def init_module(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

        for module in self.encoder.modules():
            init_module(module)
        for module in self.predictor.modules():
            init_module(module)
        self.target_encoder.load_state_dict(self.encoder.state_dict())


if __name__ == "__main__":
    from components.vision_transformer import VisionTransformer, VisionTransformerPredictor

    _ = IJEPAModule(
        encoder=VisionTransformer(embed_dim=192, num_heads=3),
        predictor=VisionTransformerPredictor(num_patches=196, embed_dim=192, num_heads=3),
        optimizer=torch.optim.AdamW,
    )
