# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

from lightning import Callback, LightningModule, Trainer


class DynamicNumClusters(Callback):
    """Step schedule for the local regularizer's k-means neighborhood count.

    ``k = min(max, start + epoch // step_epochs)``, set on ``pl_module.reg_loss``
    at the start of every epoch. With the defaults (start=1, step=100, max=8)
    k=1 at epoch 0, k=2 at epoch 100, +1 every 100 epochs, k=8 from epoch 700 on.
    """

    def __init__(self, start: int = 1, max: int = 8, step_epochs: int = 100) -> None:
        self.start = start
        self.max = max
        self.step_epochs = step_epochs

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        k = min(self.max, self.start + trainer.current_epoch // self.step_epochs)
        pl_module.reg_loss = partial(pl_module.reg_loss, num_clusters=k)
        pl_module.log("train/num_clusters", float(k), sync_dist=True, on_epoch=True)
