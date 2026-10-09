"""A run resumed mid-epoch keeps its dynamic-K schedule (F87).

``DynamicNumClusters`` set K only in ``on_train_epoch_start``, and Lightning 2.6.6 skips that hook for
an epoch it resumes in the middle of (``fit_loop`` with ``restarted_mid_epoch``). The campaign launcher
always resumes mid-epoch after a wall -- from ``on_exception.ckpt``, the checkpoint with the largest
``global_step`` -- so the rest of that epoch trained at the freshly built module's K, the config's
``num_clusters: 1``, instead of the schedule's K. These fits are tiny and on CPU.
"""

from __future__ import annotations

import pytest
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import OnExceptionCheckpoint

from src.callbacks.dynamic_num_clusters import DynamicNumClusters


class KRecorder(LightningModule):
    """Records the K its regularizer would be called with at every training step. A fresh module
    starts at K=1, as a run built from a dynamic-K config does (``num_clusters: 1``)."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)
        self.reg_loss = lambda num_clusters=1: num_clusters
        self.seen: list[tuple[int, int, int]] = []

    def training_step(self, batch, batch_idx):
        self.seen.append((self.current_epoch, batch_idx, self.reg_loss()))
        return self.layer(batch[0]).pow(2).mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class Wall(Callback):
    """The SLURM wall, as Lightning sees it: an exception in the middle of an epoch."""

    def __init__(self, epoch: int, batch: int) -> None:
        self.epoch, self.batch = epoch, batch

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch == self.epoch and batch_idx == self.batch:
            raise RuntimeError("wall")


def loader():
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(16, 2)), batch_size=4)


def schedule():
    """K=1 in epoch 0 and K=10 from epoch 1: the paper's 1 -> 10, compressed to two epochs."""
    return DynamicNumClusters(start=1, end=10, step_epochs=1, step=9)


def trainer(tmp_path, callbacks):
    return Trainer(max_epochs=2, callbacks=callbacks, logger=False, enable_progress_bar=False,
                   enable_model_summary=False, accelerator="cpu", devices=1,
                   default_root_dir=str(tmp_path), num_sanity_val_steps=0, limit_val_batches=0)


def test_the_steps_after_a_mid_epoch_resume_see_the_schedules_k(tmp_path):
    first = KRecorder()
    with pytest.raises(RuntimeError, match="wall"):
        trainer(tmp_path, [schedule(), OnExceptionCheckpoint(dirpath=str(tmp_path))]
                + [Wall(epoch=1, batch=1)]).fit(first, loader())
    assert [k for e, _, k in first.seen if e == 1] == [10, 10], "before the wall the schedule was in force"
    ckpt = tmp_path / "on_exception.ckpt"
    assert ckpt.is_file()

    resumed = KRecorder()                                   # a fresh module: K=1, as the launcher builds it
    trainer(tmp_path, [schedule()]).fit(resumed, loader(), ckpt_path=str(ckpt))
    after = [(e, b, k) for e, b, k in resumed.seen if e == 1]
    assert after, "the resumed run trained some of epoch 1"
    assert all(k == 10 for _, _, k in after), f"resumed steps of epoch 1 must see K=10, saw {after}"


def test_a_fresh_run_still_starts_at_the_schedules_first_k(tmp_path):
    module = KRecorder()
    trainer(tmp_path, [schedule()]).fit(module, loader())
    assert {k for e, _, k in module.seen if e == 0} == {1}
    assert {k for e, _, k in module.seen if e == 1} == {10}
