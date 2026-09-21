"""End-to-end training on synthetic data, through the real Hydra entry point.

Everything else in this suite tests a part. This runs ``src.train.train`` itself -- config
composition, datamodule, module, loss, online probe, checkpointing, reload and the test loop --
on a handful of batches of generated images, so the wiring between the parts is exercised too.

The most important assertion is the last one: **nothing may be written inside the repository.**
That is the guard against the defaults-resolve-into-the-checkout problem, and on the cluster it
is the difference between a tidy run and filling a lab-shared quota.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from conftest import write_imagefolder

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "configs"

pytestmark = pytest.mark.slow


@pytest.fixture
def tiny_imagenette(tmp_path):
    """Imagenette's expected layout, small enough to train on in seconds. Ten classes because
    the experiment configs set ``num_classes: 10`` and the online probe's head must match."""
    base = tmp_path / "imagenette" / "imagenette2-160"
    for split in ("train", "val"):
        write_imagefolder(base / split, classes=10, per_class=4, size=32)
    return tmp_path


def run_training(experiment: str, data_root: Path, out_dir: Path, extra: list[str] | None = None):
    os.environ["PROJECT_ROOT"] = str(REPO)
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="train",
                overrides=[
                    f"experiment={experiment}",
                    "trainer=cpu",
                    "trainer.max_epochs=1",
                    "+trainer.limit_train_batches=3",
                    "+trainer.limit_val_batches=1",
                    "+trainer.limit_test_batches=1",
                    # RichProgressBar is in callbacks/default, so drop the callback rather
                    # than disabling the bar (Lightning rejects the combination).
                    "~callbacks.rich_progress_bar",
                    "trainer.accumulate_grad_batches=1",
                    "data.batch_size=20",          # >= k * min_cluster_size, so the local path
                    "data.num_workers=0",          # is genuinely taken -- see test_losses_local
                    "data.crop_size=32",
                    "logger=csv",
                    "extras.enforce_tags=false",
                    "extras.print_config=false",
                    f"paths.data_dir={data_root}",
                    f"paths.log_dir={out_dir}",
                    # paths.output_dir interpolates ${hydra:runtime.output_dir}, which only
                    # exists under @hydra.main. compose() does not populate HydraConfig, so the
                    # resolved paths are supplied directly instead of faking the resolver.
                    f"paths.output_dir={out_dir}/run",
                    f"paths.work_dir={out_dir}",
                    *(extra or []),
                ],
            )
            from src.train import train

            return train(cfg)
    finally:
        GlobalHydra.instance().clear()


@pytest.mark.parametrize("experiment", [
    "lejepa/vit_imagenette_sigreg",
    "lejepa/vit_imagenette_localballreg_l01_k4",
])
def test_training_runs_end_to_end(experiment, tiny_imagenette, tmp_path):
    """Both arms of the headline comparison, through the real entry point."""
    out = tmp_path / "out"
    metrics, objects = run_training(experiment, tiny_imagenette, out)

    loss = metrics.get("train/loss_epoch", metrics.get("train/loss"))
    assert loss is not None, f"no training loss was logged; got {sorted(metrics)}"
    assert torch.isfinite(torch.as_tensor(float(loss))), "loss must be finite"
    assert objects["model"] is not None and objects["datamodule"] is not None


def test_a_checkpoint_is_written_and_reloadable(tiny_imagenette, tmp_path):
    """The checkpoint is what every downstream evaluation consumes, and ``train()`` reloads it
    for the test loop -- so a checkpoint that cannot be read fails here rather than days later."""
    out = tmp_path / "out"
    run_training("lejepa/vit_imagenette_sigreg", tiny_imagenette, out)

    checkpoints = list(out.rglob("*.ckpt"))
    assert checkpoints, "training must write at least one checkpoint"

    blob = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    assert "state_dict" in blob and "epoch" in blob
    assert any(k.startswith("encoder.") for k in blob["state_dict"]), (
        "the encoder weights must be present under the prefix load_backbone strips"
    )


def test_nothing_is_written_inside_the_repository(tiny_imagenette, tmp_path):
    """THE guard. Every default in ``configs/paths/default.yaml`` resolves under PROJECT_ROOT,
    i.e. the checkout. On the cluster the checkout sits on a lab-shared quota, and filling it
    kills other people's running jobs -- so a run that writes there is a real incident, not an
    untidiness.
    """
    before = {p for p in (REPO / "logs", REPO / "data", REPO / "checkpoints") if p.exists()}

    run_training("lejepa/vit_imagenette_sigreg", tiny_imagenette, tmp_path / "out")

    after = {p for p in (REPO / "logs", REPO / "data", REPO / "checkpoints") if p.exists()}
    assert after == before, (
        f"training created {after - before} inside the repository; paths.data_dir and "
        "paths.log_dir must be redirected (configs/local/default.yaml does this on the cluster)"
    )


def test_the_local_regularizer_actually_takes_its_local_path(tiny_imagenette, tmp_path, monkeypatch):
    """A 20-sample batch with k=4 and min_cluster_size=4 clears the fallback threshold, so this
    run exercises the clustering rather than silently degrading to the global test. Verified by
    counting how many times the inner loss is called per step.
    """
    import src.losses as losses

    calls = []
    original = losses.local_loss

    def counting(proj, *, loss, **kwargs):
        inner_calls = []

        def spy(subset):
            inner_calls.append(int(subset.shape[1]))
            return loss(subset)

        out = original(proj, loss=spy, **kwargs)
        calls.append(inner_calls)
        return out

    monkeypatch.setattr(losses, "local_loss", counting)
    monkeypatch.setattr("src.losses.local_ballreg_loss.__defaults__", None, raising=False)

    run_training("lejepa/vit_imagenette_localballreg_l01_k4", tiny_imagenette, tmp_path / "out")

    assert calls, "local_loss was never called"
    fell_back = [c for c in calls if len(c) == 1 and c[0] == 20]
    assert len(fell_back) < len(calls), (
        f"every step fell back to the global test: {calls[:3]}"
    )


@pytest.mark.parametrize("seed, should_seed", [(0, True), (42, True), (None, False)])
def test_seed_zero_actually_seeds(tiny_imagenette, tmp_path, monkeypatch, seed, should_seed):
    """``seed: 0`` must seed.

    ``src/train.py`` guarded on truthiness, so ``seed: 0`` -- the most natural seed anyone picks --
    silently skipped ``seed_everything`` entirely, and the run proceeded unseeded while its config
    said otherwise. Latent today (training configs use 42 or null), but it is the same defect as
    the linear probe's unseeded head and it would have been discovered the hard way.

    ``None`` must still skip: that is the documented "do not seed" value, and forcing a seed on
    every run would change what ``seed: null`` means.
    """
    import src.train as train_module

    seen = []
    monkeypatch.setattr(train_module.L, "seed_everything",
                        lambda value, workers=False: seen.append(value))

    out = tmp_path / "out"
    run_training("lejepa/vit_imagenette_sigreg", tiny_imagenette, out,
                 extra=[f"seed={'null' if seed is None else seed}"])

    assert seen == ([seed] if should_seed else [])


def test_a_rehearsed_sigterm_writes_the_exception_checkpoint(tiny_imagenette, tmp_path):
    """The calibration's interruption is a REAL SIGTERM to this process, handled by Lightning's own
    signal connector: ``SelfInterrupt`` fires at global_step 2, the epoch loop raises
    ``SIGTERMException`` (a ``SystemExit``), and ``OnExceptionCheckpoint`` writes
    ``on_exception.ckpt`` holding exactly that step -- the file the launcher's resume decision
    reads. Simulating the signal would prove nothing about the path a preemption takes."""
    import torch

    ckpt_dir = tmp_path / "ckpt"
    with pytest.raises(SystemExit):
        run_training("lejepa/vit_imagenette_sigreg", tiny_imagenette, tmp_path / "out", extra=[
            "callbacks=calibrate",          # the harness already drops rich_progress_bar
            "callbacks.self_interrupt.at_step=2", "callbacks.throughput.warmup_steps=0",
            "+trainer.max_steps=6", f"callbacks.on_exception.dirpath={ckpt_dir}",
            f"callbacks.model_checkpoint.dirpath={ckpt_dir}",
        ])
    blob = torch.load(ckpt_dir / "on_exception.ckpt", map_location="cpu", weights_only=False)
    assert blob["global_step"] == 2
    # (the tiny fixture has two batches per epoch, so an epoch boundary is also reached and
    # last.ckpt exists too -- irrelevant here; the on-exception file is what the launcher reads)
