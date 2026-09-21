# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Calibration callbacks: what a run computes at, and a rehearsal of being interrupted.

``Throughput`` separates two things a bare steps/s figure conflates -- time the GPU spent
computing and time the loop spent waiting for the next batch. A decode-starved run (16 cores
per GPU decoding full-size JPEGs, plan §"Schedule") looks like a slow GPU to steps/s and like an
*idle* GPU to the cluster's watchdog (F40); the data-wait fraction is the number that tells them
apart and decides whether the loader lever is pulled. Every figure is printed with the geometry
it was measured at, because a throughput number without its batch, views and crop is what sized
a previous campaign from the wrong model (research_rules/01).

``SelfInterrupt`` sends the process a real SIGTERM at a chosen step, so the calibration job
rehearses the preemption path end to end -- Lightning's own handler, the ``SIGTERMException``,
``OnExceptionCheckpoint``, the launcher's resume decision -- instead of a simulation of it.
"""

from __future__ import annotations

import json
import os
import signal
import time
from typing import Any

from lightning import Callback


class Throughput(Callback):
    """Steps/s, samples/s and the data-wait fraction, from wall-clock stamps around each batch.

    Parameters
    ----------
    warmup_steps : steps excluded from every figure (cuDNN autotuning, worker spin-up, the
                   first ``torch.compile`` graph -- none of which the campaign pays per step).
    print_every  : print an interim summary every this many measured steps (0: only at the end).
    """

    def __init__(self, warmup_steps: int = 20, print_every: int = 50) -> None:
        super().__init__()
        self.warmup_steps = int(warmup_steps)
        self.print_every = int(print_every)
        self.reset()

    def reset(self) -> None:
        self.measured_steps = 0
        self.compute_seconds = 0.0
        self.wait_seconds = 0.0
        self.samples = 0
        self._batch_started: float | None = None
        self._last_batch_end: float | None = None
        self._seen_steps = 0
        self.world_size = 1
        self.steps_per_epoch: int | None = None

    # ----------------------------------------------------------------------- hooks
    def on_train_start(self, trainer, pl_module) -> None:
        self.reset()
        self.world_size = int(getattr(trainer, "world_size", 1) or 1)
        batches = getattr(trainer, "num_training_batches", None)
        self.steps_per_epoch = int(batches) if isinstance(batches, int) and batches > 0 else None

    def on_train_batch_start(self, trainer, pl_module, batch: Any, batch_idx: int) -> None:
        now = time.perf_counter()
        if self._seen_steps >= self.warmup_steps and self._last_batch_end is not None:
            self.wait_seconds += now - self._last_batch_end
        self._batch_started = now

    def on_train_batch_end(self, trainer, pl_module, outputs: Any, batch: Any, batch_idx: int) -> None:
        now = time.perf_counter()
        if self._seen_steps >= self.warmup_steps and self._batch_started is not None:
            self.compute_seconds += now - self._batch_started
            self.measured_steps += 1
            self.samples += _batch_size(batch) * self.world_size
            if self.print_every and self.measured_steps % self.print_every == 0:
                print("throughput " + json.dumps(self.summary()), flush=True)
        self._seen_steps += 1
        self._last_batch_end = now

    def on_train_end(self, trainer, pl_module) -> None:
        summary = self.summary()
        print("THROUGHPUT " + json.dumps(summary), flush=True)
        logger = getattr(trainer, "logger", None)
        if logger is not None and self.measured_steps:
            numeric = {f"calib/{k}": v for k, v in summary.items() if isinstance(v, (int, float)) and v is not None}
            logger.log_metrics(numeric, step=int(getattr(trainer, "global_step", 0) or 0))

    # ----------------------------------------------------------------------- the numbers
    def summary(self) -> dict[str, Any]:
        wall = self.compute_seconds + self.wait_seconds
        steps_per_s = self.measured_steps / wall if wall > 0 else 0.0
        epoch_s = self.steps_per_epoch / steps_per_s if (self.steps_per_epoch and steps_per_s) else None
        return {
            "measured_steps": self.measured_steps,
            "warmup_steps": self.warmup_steps,
            "world_size": self.world_size,
            "steps_per_s": steps_per_s,
            "samples_per_s": self.samples / wall if wall > 0 else 0.0,
            "mean_step_s": wall / self.measured_steps if self.measured_steps else None,
            "data_wait_fraction": self.wait_seconds / wall if wall > 0 else None,
            "steps_per_epoch": self.steps_per_epoch,
            "epoch_s": epoch_s,
            "hours_per_100_epochs": (100 * epoch_s / 3600) if epoch_s else None,
        }


def _batch_size(batch: Any) -> int:
    """Samples in a training batch: ``(views, labels)`` where views is (N, V, C, H, W) or a list
    of such groups (mixed-resolution multi-crop). Unknown shapes count as zero rather than guess."""
    views = batch[0] if isinstance(batch, (tuple, list)) and batch else batch
    if isinstance(views, (tuple, list)) and views:
        views = views[0]
    shape = getattr(views, "shape", None)
    return int(shape[0]) if shape is not None and len(shape) >= 1 else 0


class SelfInterrupt(Callback):
    """Send this process a real SIGTERM once ``global_step`` reaches ``at_step``.

    Every rank does it to itself, which is what ``srun --signal`` produces on the cluster. The
    handler Lightning installed turns it into ``SIGTERMException`` at the next loop boundary,
    ``OnExceptionCheckpoint`` writes ``on_exception.ckpt``, and the launcher's resume decision
    is exercised for real on the second launch. ``at_step=None`` disables it.
    """

    def __init__(self, at_step: int | None = None) -> None:
        super().__init__()
        self.at_step = None if at_step is None else int(at_step)
        self.fired = False

    def on_train_batch_end(self, trainer, pl_module, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self.at_step is None or self.fired:
            return
        if int(getattr(trainer, "global_step", 0) or 0) >= self.at_step:
            self.fired = True
            print(f"SelfInterrupt: sending SIGTERM to pid {os.getpid()} at global_step {trainer.global_step}", flush=True)
            os.kill(os.getpid(), signal.SIGTERM)
