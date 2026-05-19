# Copyright (c) 2026 Pavel Snopov
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule


class PhCRegressionModule(LightningModule):
    """Downstream PhC DOS regression with optional encoder fine-tuning."""

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer],
        embedding_dim: int = 1024,
        scheduler: Callable[..., Any] | None = None,
        pretrained_ckpt_path: str | None = None,
        freeze_encoder: bool = False,
        strict_encoder_load: bool = True,
        scheduler_interval: str = "epoch",
        scheduler_monitor: str | None = "val/loss",
        compile: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["encoder", "predictor", "optimizer", "scheduler"], logger=False)

        self.encoder = encoder
        self.predictor = predictor
        self.optimizer_fn = optimizer
        self.scheduler_fn = scheduler

        self.embedding_dim = embedding_dim
        self.pretrained_ckpt_path = pretrained_ckpt_path
        self.freeze_encoder = freeze_encoder
        self.strict_encoder_load = strict_encoder_load
        self.scheduler_interval = scheduler_interval
        self.scheduler_monitor = scheduler_monitor
        self.compile_model = compile

        self._encoder_initialized = False

        self.val_preds: list[torch.Tensor] = []
        self.val_targets: list[torch.Tensor] = []
        self.test_preds: list[torch.Tensor] = []
        self.test_targets: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise RuntimeError(f"Expected PhC inputs with shape (N, 1, H, W), got {tuple(x.shape)}")

        emb = self.encoder(x)
        if emb.ndim != 2 or emb.shape[-1] != self.embedding_dim:
            raise RuntimeError(
                f"Expected encoder outputs with shape (N, {self.embedding_dim}), got {tuple(emb.shape)}"
            )

        return self.predictor(emb)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, target = batch
        pred = self(x)
        if pred.shape != target.shape:
            raise RuntimeError(f"Expected predictor outputs to match targets, got {tuple(pred.shape)} and {tuple(target.shape)}")

        loss = F.mse_loss(pred, target)
        mae = F.l1_loss(pred, target)

        self.log("train/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
        self.log("train/mae", mae, sync_dist=True, on_step=True, on_epoch=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_preds = []
        self.val_targets = []

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        x, target = batch
        pred = self(x)
        self._check_target_shape(pred, target)
        self.val_preds.append(pred.detach().cpu())
        self.val_targets.append(target.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("val", self.val_preds, self.val_targets)

    def on_test_epoch_start(self) -> None:
        self.test_preds = []
        self.test_targets = []

    def test_step(self, batch: Any, batch_idx: int) -> None:
        x, target = batch
        pred = self(x)
        self._check_target_shape(pred, target)
        self.test_preds.append(pred.detach().cpu())
        self.test_targets.append(target.detach().cpu())

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics("test", self.test_preds, self.test_targets)

    def setup(self, stage: str | None = None) -> None:
        if not self._encoder_initialized:
            if self.pretrained_ckpt_path is not None:
                self._load_pretrained_encoder()
            self._set_encoder_trainable(not self.freeze_encoder)
            self._encoder_initialized = True

        if self.compile_model and stage == "fit":
            self.encoder = torch.compile(self.encoder)
            self.predictor = torch.compile(self.predictor)

    def configure_optimizers(self) -> dict[str, Any] | torch.optim.Optimizer:
        params = list(self.predictor.parameters()) + ([] if self.freeze_encoder else list(self.encoder.parameters()))
        optimizer = self.optimizer_fn(params=params)

        if self.scheduler_fn is None:
            return optimizer

        scheduler = self.scheduler_fn(optimizer=optimizer)
        scheduler_config: dict[str, Any] = {
            "scheduler": scheduler,
            "interval": self.scheduler_interval,
            "frequency": 1,
        }
        if self.scheduler_monitor is not None:
            scheduler_config["monitor"] = self.scheduler_monitor

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def _load_pretrained_encoder(self) -> None:
        ckpt_path = Path(self.pretrained_ckpt_path).expanduser()
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

        encoder_state = {key.removeprefix("encoder."): value for key, value in state_dict.items() if key.startswith("encoder.")}
        if not encoder_state:
            encoder_state = state_dict

        self.encoder.load_state_dict(encoder_state, strict=self.strict_encoder_load)

    def _set_encoder_trainable(self, trainable: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable

    @staticmethod
    def _check_target_shape(pred: torch.Tensor, target: torch.Tensor) -> None:
        if pred.shape != target.shape:
            raise RuntimeError(f"Expected predictor outputs to match targets, got {tuple(pred.shape)} and {tuple(target.shape)}")

    def _log_epoch_metrics(self, stage: str, preds: list[torch.Tensor], targets: list[torch.Tensor]) -> None:
        pred = torch.cat(preds, dim=0)
        target = torch.cat(targets, dim=0)

        loss = F.mse_loss(pred, target)
        mae = F.l1_loss(pred, target)
        eps_dos = 100.0 * (pred - target).abs().sum() / target.abs().sum().clamp_min(torch.finfo(pred.dtype).eps)

        self.log(f"{stage}/loss", loss, prog_bar=(stage == "val"), sync_dist=False)
        self.log(f"{stage}/mae", mae, sync_dist=False)
        self.log(f"{stage}/eps_dos", eps_dos, prog_bar=True, sync_dist=False)
