# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src import losses


def _accepts_keyword(fn: Callable[..., Any], name: str) -> bool:
    """True if ``fn`` declares ``name`` as a real keyword parameter.

    A bare ``**kwargs`` does not count: a function that merely swallows ``stats`` would leave it
    empty, and the module would log nothing while believing the regularizer had reported.
    ``functools.partial`` objects resolve through to the wrapped signature, which is how the
    Hydra-instantiated ``local_*`` regularizers (``_partial_: true``) are detected.
    """
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    parameter = parameters.get(name)
    return parameter is not None and parameter.kind in (
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )


class LeJEPAModule(LightningModule):
    """LightningModule for LeJEPA pretraining (Balestriero & LeCun, arXiv:2511.08544).

    ``reg_loss`` and ``inv_loss`` are callables ``(proj) -> scalar`` on ``(V, N, D)``
    projections (``src.losses``; ``reg_loss`` is configured through ``model/reg_loss``,
    default SIGReg). The total loss is::

        loss = lamb * reg_loss(proj) + (1 - lamb) * inv_loss(proj)

    The encoder can be any ``nn.Module`` that maps flattened image views
    ``(N * V, C, H, W)`` to embeddings ``(N * V, embedding_dim)``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        optimizer: Callable[..., torch.optim.Optimizer],
        reg_loss: Callable[..., torch.Tensor] = losses.sigreg_loss,
        inv_loss: Callable[..., torch.Tensor] = losses.invariance_loss,
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
        lr_eta_min_ratio: float = 0.01,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "projector", "online_probe", "optimizer", "reg_loss", "inv_loss"], logger=False)

        self.encoder = encoder
        self.projector = projector if projector is not None else nn.Identity()
        self.online_probe = online_probe if use_online_probe else None
        self.optimizer_fn = optimizer
        self.reg_loss = reg_loss
        self.inv_loss = inv_loss
        self.reg_name = getattr(reg_loss, "func", reg_loss).__name__

        # Decided once: the local regularizers report their realised partition through a `stats`
        # out-param (src/losses.py::local_loss). DynamicNumClusters re-wraps `reg_loss` in a
        # partial every epoch, which keeps the parameter, so the answer does not go stale.
        self._reg_takes_stats = _accepts_keyword(reg_loss, "stats")

        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.lamb = lamb
        self.use_online_probe = use_online_probe
        self.probe_lr = probe_lr
        self.probe_weight_decay = probe_weight_decay
        self.warmup_epochs = warmup_epochs
        self.lr_eta_min_ratio = lr_eta_min_ratio
        self.compile_model = compile

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _view_groups(views: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]) -> tuple[list[torch.Tensor], int, int]:
        """Normalize the views batch to a list of (N, V_i, C, H_i, W_i) resolution groups.

        A single (N, V, C, H, W) tensor is the historical layout. A list/tuple of such
        tensors is the mixed-resolution multi-crop layout (see MultiCropTransform):
        each group shares one resolution, the global views come first, and the
        logical view index runs over the groups in order.

        Returns:
            (groups, batch_size, total number of views)
        """
        groups = list(views) if isinstance(views, (list, tuple)) else [views]
        if not groups:
            raise RuntimeError("Expected at least one group of views")
        for group in groups:
            if group.ndim != 5:
                raise RuntimeError(f"Expected views with shape (N, V, C, H, W), got {tuple(group.shape)}")
            if group.shape[0] != groups[0].shape[0]:
                raise RuntimeError("All view groups must share the same batch size")
        batch_size = groups[0].shape[0]
        num_views = sum(group.shape[1] for group in groups)
        return groups, batch_size, num_views

    def encode(self, views: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Encode multi-view images into embeddings.

        Args:
            views: (N, V, C, H, W), or a list/tuple of such tensors, one per
                resolution group (mixed-resolution multi-crop).

        Returns:
            (N * V, embedding_dim), sample-major: row ``n * V + v`` is view ``v`` of
            sample ``n`` (global views first).
        """
        groups, batch_size, _ = self._view_groups(views)
        group_embs = []
        for group in groups:
            emb_g = self.encoder(group.flatten(0, 1))
            if emb_g.ndim != 2 or emb_g.shape[-1] != self.embedding_dim:
                raise RuntimeError(f"Expected embeddings with shape (N * V, {self.embedding_dim}), got {tuple(emb_g.shape)}")
            group_embs.append(emb_g.reshape(batch_size, group.shape[1], self.embedding_dim))
        return torch.cat(group_embs, dim=1).flatten(0, 1)

    def forward(self, views: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode and project multi-view images.

        Args:
            views: (N, V, C, H, W), or a list/tuple of per-resolution groups.

        Returns:
            emb (N * V, embedding_dim) and proj (V, N, projection_dim)
        """
        _, batch_size, num_views = self._view_groups(views)
        emb = self.encode(views)
        proj = self.projector(emb)
        if proj.ndim != 2 or proj.shape[-1] != self.projection_dim:
            raise RuntimeError(f"Projector must return (N * V, {self.projection_dim}), got {tuple(proj.shape)}")

        # Reshape to (V, N, D) layout expected by all loss functions.
        proj = proj.reshape(batch_size, num_views, self.projection_dim).transpose(0, 1)
        return emb, proj

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def model_step(self, views: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        emb, proj = self.forward(views)
        inv_loss = self.inv_loss(proj)
        reg_stats: dict[str, float] = {}
        reg = self.reg_loss(proj, stats=reg_stats) if self._reg_takes_stats else self.reg_loss(proj)

        lejepa_loss = reg * self.lamb + inv_loss * (1.0 - self.lamb)

        return emb, {
            "loss": lejepa_loss,
            "lejepa_loss": lejepa_loss,
            "invariance_loss": inv_loss,
            self.reg_name: reg,
            "inv_contrib": inv_loss * (1.0 - self.lamb),
            "reg_contrib": reg * self.lamb,
            **self._covariance_metrics(proj),
            # Realised-K diagnostics, logged as train/local/*. Absent for a global regularizer,
            # so an empty column means "not a local run", never "K was as configured".
            **{f"local/{key}": value for key, value in reg_stats.items()},
        }

    @torch.no_grad()
    def _covariance_metrics(self, proj: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute embedding covariance diagnostics from (V, N, D) projections."""
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
        loss = metrics["loss"]

        if self.use_online_probe:
            # Probe on the first view of every sample, matched to the original labels.
            _, batch_size, num_views = self._view_groups(views)
            emb_first = emb.reshape(batch_size, num_views, self.embedding_dim)[:, 0]
            labels = labels.to(emb.device, non_blocking=True)
            metrics["probe_loss"] = F.cross_entropy(self.online_probe(emb_first.detach()), labels)
            loss = loss + metrics["probe_loss"]

        for key, value in metrics.items():
            self.log(f"train/{key}", value, prog_bar=key == "loss", sync_dist=True, on_step=True, on_epoch=True)

        return loss

    def _probe_eval_step(self, batch: Any, prefix: str) -> None:
        if not self.use_online_probe:
            return
        views, labels = batch
        emb = self.encode(views)
        labels = labels.to(emb.device, non_blocking=True)
        probe_acc = (self.online_probe(emb).argmax(dim=1) == labels).float().mean()
        self.log(f"{prefix}/probe_acc", probe_acc, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._probe_eval_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._probe_eval_step(batch, "test")

    def setup(self, stage: str | None = None) -> None:
        if self.compile_model and stage == "fit":
            self.encoder = torch.compile(self.encoder)
            self.projector = torch.compile(self.projector)
            if self.use_online_probe:
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
            if self.use_online_probe
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
        eta_min = self.lr_eta_min_ratio * optimizer.param_groups[0]["lr"]

        if warmup_steps <= 0:
            return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=eta_min)

        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
        if warmup_steps >= total_steps:
            return warmup

        cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=eta_min)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


if __name__ == "__main__":
    module = LeJEPAModule(encoder=nn.Identity(), projector=nn.Identity(), optimizer=torch.optim.AdamW, online_probe=nn.Identity())
    print(module)
