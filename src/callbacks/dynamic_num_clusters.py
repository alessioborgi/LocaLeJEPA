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

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        moved = self.step * (trainer.current_epoch // self.step_epochs)
        k = min(self.end, self.start + moved) if self.end >= self.start else max(self.end, self.start - moved)
        pl_module.reg_loss = partial(pl_module.reg_loss, num_clusters=k)
        pl_module.log("train/num_clusters", float(k), sync_dist=True, on_epoch=True)
