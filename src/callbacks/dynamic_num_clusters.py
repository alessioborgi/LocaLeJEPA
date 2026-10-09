# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

from lightning import Callback, LightningModule, Trainer


class DynamicNumClusters(Callback):
    """Step schedule for the local regularizer's k-means neighborhood count.

    Every ``step_epochs`` epochs k moves by ``step`` from ``start`` towards ``end`` (up or
    down, clamped at ``end``) and is set on ``pl_module.reg_loss``. With the defaults
    (start=1, end=8, step_epochs=100, step=1) k=1 at epoch 0, k=2 at epoch 100, +1 every
    100 epochs, k=8 from epoch 700 on.
    """

    def __init__(self, start: int = 1, end: int = 8, step_epochs: int = 100, step: int = 1) -> None:
        self.start = start
        self.end = end
        self.step_epochs = step_epochs
        self.step = step

    def k(self, epoch: int) -> int:
        """The neighborhood count in force during ``epoch``."""
        moved = self.step * (epoch // self.step_epochs)
        return min(self.end, self.start + moved) if self.end >= self.start else max(self.end, self.start - moved)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # F87: Lightning skips on_train_epoch_start for an epoch it resumes in the middle of, and the
        # launcher resumes mid-epoch after every wall; without this the rest of that epoch would train at
        # the freshly built module's K (the config's num_clusters), not the schedule's. Not logged here:
        # the epoch-start hook logs K, and this hook runs before any training step. Printed instead, so a
        # successor's job log shows the K its resumed epoch trains at: train/num_clusters gets no row for
        # that epoch (the regularizer's per-step train/local/configured_clusters does record it).
        k = self.k(trainer.current_epoch)
        pl_module.reg_loss = partial(pl_module.reg_loss, num_clusters=k)
        print(f"DynamicNumClusters: K={k} at train start (epoch {trainer.current_epoch})", flush=True)

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        k = self.k(trainer.current_epoch)
        pl_module.reg_loss = partial(pl_module.reg_loss, num_clusters=k)
        pl_module.log("train/num_clusters", float(k), sync_dist=True, on_epoch=True)
