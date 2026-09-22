"""last.ckpt is the final epoch, whatever the online probe did (F50).

Lightning 2.6.6 writes a callback's last.ckpt only in the call that also saved a top-k checkpoint,
i.e. only when the monitored metric improved. With one callback carrying `monitor`, `save_top_k=1`
and `save_last=True`, last.ckpt was a copy of best.ckpt: the first ablation wave ended with
last.ckpt at epoch 97 of 100, and every FASRC run of the draft had the same settings. The
callbacks group now has a monitor-less callback for last.ckpt. These fits are tiny and on CPU.
"""

from __future__ import annotations

import torch
from hydra.utils import instantiate
from lightning.pytorch import LightningModule, Trainer

import scripts.aicr.campaign as C


class Constant(LightningModule):
    """A model whose validation metric follows a schedule we choose."""

    def __init__(self, schedule):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)
        self.schedule = list(schedule)

    def training_step(self, batch, batch_idx):
        return self.layer(batch[0]).pow(2).mean()

    def validation_step(self, batch, batch_idx):
        self.log("val/probe_acc", float(self.schedule[self.current_epoch]), on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def loader():
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(8, 2)), batch_size=4)


def callbacks_from_config(ckpt_dir):
    cfg = C.compose_config(["experiment=ablations/base", f"callbacks.model_checkpoint.dirpath={ckpt_dir}",
                            f"callbacks.last_checkpoint.dirpath={ckpt_dir}", f"callbacks.on_exception.dirpath={ckpt_dir}"])
    return [instantiate(cfg.callbacks[name]) for name in ("model_checkpoint", "last_checkpoint", "on_exception")]


def fit(tmp_path, schedule):
    ckpt_dir = tmp_path / "ckpt"
    trainer = Trainer(max_epochs=len(schedule), callbacks=callbacks_from_config(ckpt_dir), logger=False,
                      enable_progress_bar=False, enable_model_summary=False, accelerator="cpu", devices=1,
                      default_root_dir=str(tmp_path), num_sanity_val_steps=0)
    trainer.fit(Constant(schedule), loader(), loader())
    return ckpt_dir


def epoch_of(path):
    return int(torch.load(path, map_location="cpu", weights_only=False)["epoch"])


def test_last_is_the_final_epoch_when_the_metric_never_improves(tmp_path):
    ckpt_dir = fit(tmp_path, schedule=[0.9, 0.5, 0.5, 0.5])
    assert epoch_of(ckpt_dir / "last.ckpt") == 3, "the final epoch, regardless of the monitor"
    assert epoch_of(ckpt_dir / "best.ckpt") == 0, "the online probe's best epoch"
    assert sorted(p.name for p in ckpt_dir.iterdir()) == ["best.ckpt", "last.ckpt"], "no -v1 files, nothing else"


def test_last_and_best_coincide_only_when_the_last_epoch_is_the_best(tmp_path):
    ckpt_dir = fit(tmp_path, schedule=[0.1, 0.2, 0.3])
    assert epoch_of(ckpt_dir / "last.ckpt") == 2 and epoch_of(ckpt_dir / "best.ckpt") == 2


def test_lightning_auto_resumes_from_an_existing_last_ckpt_and_overwrites_it(tmp_path):
    """With an OnExceptionCheckpoint among the callbacks, Lightning turns fit(ckpt_path=None) into
    ckpt_path="last" and resumes from the file it finds (F50). Documented here, and the resumed
    run keeps writing last.ckpt in place -- no last-v1.ckpt."""
    ckpt_dir = fit(tmp_path, schedule=[0.5, 0.5])
    assert epoch_of(ckpt_dir / "last.ckpt") == 1
    trainer = Trainer(max_epochs=4, callbacks=callbacks_from_config(ckpt_dir), logger=False, enable_progress_bar=False,
                      enable_model_summary=False, accelerator="cpu", devices=1, default_root_dir=str(tmp_path), num_sanity_val_steps=0)
    trainer.fit(Constant([0.5, 0.5, 0.5, 0.5]), loader(), loader())          # no ckpt_path: Lightning picks last.ckpt
    assert epoch_of(ckpt_dir / "last.ckpt") == 3 and trainer.current_epoch == 4
    assert sorted(p.name for p in ckpt_dir.iterdir()) == ["best.ckpt", "last.ckpt"]


def test_the_postflight_artifact_check_accepts_the_new_last_and_rejected_the_old_one(tmp_path):
    ckpt_dir = fit(tmp_path, schedule=[0.9, 0.5, 0.5])
    ok, _ = C.verify_artifact(ckpt_dir / "last.ckpt", max_epochs=3)
    assert ok
    ok, message = C.verify_artifact(ckpt_dir / "best.ckpt", max_epochs=3)
    assert not ok and "epoch 0, not 2" in message
