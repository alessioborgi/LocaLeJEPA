"""The calibration callbacks: throughput with the data-wait fraction, and a rehearsed SIGTERM.

The number that matters is not steps/s but how much of the wall the loop spent *waiting for
data* -- a decode-starved 128 px run (plan §"Schedule", lever 7) looks like a slow GPU to steps/s
and like an idle one to the cluster's watchdog (F40). The clock is faked so every figure here is
exact, not a tolerance.
"""

from __future__ import annotations

import json
import os
import signal

import pytest
import torch

import src.callbacks.throughput as T
from src.callbacks.throughput import SelfInterrupt, Throughput


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def perf_counter(self) -> float:
        return self.now


class Logger:
    def __init__(self) -> None:
        self.logged: list[tuple[dict, int]] = []

    def log_metrics(self, metrics, step=None):
        self.logged.append((dict(metrics), step))


class FakeTrainer:
    def __init__(self, world_size: int = 1, steps_per_epoch: int = 100) -> None:
        self.world_size = world_size
        self.num_training_batches = steps_per_epoch
        self.global_step = 0
        self.logger = Logger()


@pytest.fixture
def clock(monkeypatch) -> Clock:
    clock = Clock()
    monkeypatch.setattr(T, "time", clock)
    return clock


def batch(n: int = 8):
    return (torch.zeros(n, 4, 3, 2, 2), torch.zeros(n, dtype=torch.long))


def run_steps(cb: Throughput, trainer: FakeTrainer, clock: Clock, *, steps: int, compute: float, wait: float):
    """Each step: ``compute`` seconds inside the batch, then ``wait`` seconds before the next."""
    for step in range(steps):
        cb.on_train_batch_start(trainer, None, batch(), step)
        clock.now += compute
        trainer.global_step = step + 1
        cb.on_train_batch_end(trainer, None, None, batch(), step)
        clock.now += wait


# --------------------------------------------------------------------------------------
# the figures
# --------------------------------------------------------------------------------------

def test_warmup_is_excluded_and_the_wait_fraction_is_exact(clock):
    cb = Throughput(warmup_steps=2, print_every=0)
    trainer = FakeTrainer(world_size=2, steps_per_epoch=100)
    cb.on_train_start(trainer, None)
    run_steps(cb, trainer, clock, steps=5, compute=1.0, wait=0.5)

    s = cb.summary()
    assert s["measured_steps"] == 3, "the first two steps are warm-up"
    assert cb.compute_seconds == pytest.approx(3.0) and cb.wait_seconds == pytest.approx(1.5)
    assert s["data_wait_fraction"] == pytest.approx(1.5 / 4.5)
    assert s["steps_per_s"] == pytest.approx(3 / 4.5)
    assert s["samples_per_s"] == pytest.approx(3 * 8 * 2 / 4.5), "samples count every rank"
    assert s["mean_step_s"] == pytest.approx(1.5)
    assert s["epoch_s"] == pytest.approx(100 / (3 / 4.5))
    assert s["hours_per_100_epochs"] == pytest.approx(100 * 150 / 3600)


def test_a_compute_bound_run_reports_zero_wait(clock):
    cb = Throughput(warmup_steps=0, print_every=0)
    trainer = FakeTrainer()
    cb.on_train_start(trainer, None)
    run_steps(cb, trainer, clock, steps=4, compute=2.0, wait=0.0)
    assert cb.summary()["data_wait_fraction"] == pytest.approx(0.0)
    assert cb.summary()["steps_per_s"] == pytest.approx(0.5)


def test_summary_before_any_measured_step_is_defined_and_finite(clock):
    cb = Throughput(warmup_steps=5)
    cb.on_train_start(FakeTrainer(), None)
    s = cb.summary()
    assert s["measured_steps"] == 0 and s["steps_per_s"] == 0.0
    assert s["data_wait_fraction"] is None and s["epoch_s"] is None and s["mean_step_s"] is None


def test_train_end_prints_one_json_line_and_logs_the_numbers(clock, capsys):
    cb = Throughput(warmup_steps=0, print_every=0)
    trainer = FakeTrainer(steps_per_epoch=10)
    cb.on_train_start(trainer, None)
    run_steps(cb, trainer, clock, steps=2, compute=1.0, wait=1.0)
    cb.on_train_end(trainer, None)

    line = [l for l in capsys.readouterr().out.splitlines() if l.startswith("THROUGHPUT ")][0]
    payload = json.loads(line[len("THROUGHPUT "):])
    # two computes of 1.0 s and ONE counted wait: the wait after the last batch has no next
    # batch start to attribute it to, so the fraction is 1/3, not 1/2
    assert payload["measured_steps"] == 2 and payload["data_wait_fraction"] == pytest.approx(1 / 3)
    metrics, step = trainer.logger.logged[-1]
    assert metrics["calib/data_wait_fraction"] == pytest.approx(1 / 3) and step == 2
    assert all(isinstance(v, (int, float)) for v in metrics.values())


def test_interim_summaries_are_printed_every_print_every_steps(clock, capsys):
    cb = Throughput(warmup_steps=0, print_every=2)
    trainer = FakeTrainer()
    cb.on_train_start(trainer, None)
    run_steps(cb, trainer, clock, steps=5, compute=1.0, wait=0.0)
    interim = [l for l in capsys.readouterr().out.splitlines() if l.startswith("throughput ")]
    assert len(interim) == 2


@pytest.mark.parametrize("batch_, expected", [
    ((torch.zeros(8, 4, 3, 2, 2), torch.zeros(8)), 8),
    (([torch.zeros(6, 2, 3, 4, 4), torch.zeros(6, 6, 3, 2, 2)], torch.zeros(6)), 6),
    (("not a tensor", 0), 0),
])
def test_batch_size_reads_the_sample_axis_of_either_layout(batch_, expected):
    assert T._batch_size(batch_) == expected


# --------------------------------------------------------------------------------------
# the rehearsed SIGTERM
# --------------------------------------------------------------------------------------

def test_self_interrupt_sends_sigterm_to_this_process_once(monkeypatch):
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(T.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    cb = SelfInterrupt(at_step=3)
    trainer = FakeTrainer()
    for step in range(1, 7):
        trainer.global_step = step
        cb.on_train_batch_end(trainer, None, None, batch(), step - 1)
    assert sent == [(os.getpid(), signal.SIGTERM)], "exactly once, at step 3, to ourselves"


def test_self_interrupt_is_off_by_default(monkeypatch):
    sent: list = []
    monkeypatch.setattr(T.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    cb = SelfInterrupt()
    trainer = FakeTrainer()
    trainer.global_step = 10_000
    cb.on_train_batch_end(trainer, None, None, batch(), 0)
    assert sent == []


def test_the_calibrate_callback_group_composes_and_instantiates(tmp_path):
    from hydra.utils import instantiate
    from scripts.aicr.campaign import compose_config

    cfg = compose_config(["experiment=ablations/base", "callbacks=calibrate", "callbacks.self_interrupt.at_step=150"])
    assert isinstance(instantiate(cfg.callbacks.throughput), Throughput)
    interrupt = instantiate(cfg.callbacks.self_interrupt)
    assert isinstance(interrupt, SelfInterrupt) and interrupt.at_step == 150
    assert "model_checkpoint" in cfg.callbacks and "on_exception" in cfg.callbacks, "the defaults stay"
