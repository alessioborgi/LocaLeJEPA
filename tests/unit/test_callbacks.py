"""The dynamic-k schedule.

``DynamicNumClusters`` rewrites ``pl_module.reg_loss`` at every epoch start. The schedule itself
is simple arithmetic; the interesting part is *how* it applies, because it wraps the regularizer
in a fresh ``functools.partial`` each epoch rather than rebinding a value.
"""

from __future__ import annotations

from functools import partial

import pytest
import torch

from src.callbacks.dynamic_num_clusters import DynamicNumClusters


class FakeTrainer:
    def __init__(self, epoch: int) -> None:
        self.current_epoch = epoch


class FakeModule:
    """Captures the num_clusters the regularizer would actually be called with."""

    def __init__(self) -> None:
        self.reg_loss = lambda proj, num_clusters=None: num_clusters
        self.logged: dict[str, float] = {}

    def log(self, name, value, **kwargs):
        self.logged[name] = float(value)

    def effective_k(self):
        return self.reg_loss(torch.zeros(1))


def advance(callback: DynamicNumClusters, module: FakeModule, epoch: int) -> int:
    callback.on_train_epoch_start(FakeTrainer(epoch), module)
    return module.effective_k()


# --------------------------------------------------------------------------------------
# the schedule
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("epoch, expected", [(0, 1), (99, 1), (100, 2), (250, 3), (700, 8), (999, 8)])
def test_default_schedule_steps_k_every_hundred_epochs(epoch, expected):
    """start=1, end=8, step=1, step_epochs=100: k=1 at epoch 0, +1 per 100 epochs, capped at 8.
    Documented in the docstring; pinned here so the arithmetic cannot drift."""
    assert advance(DynamicNumClusters(), FakeModule(), epoch) == expected


def test_k_is_clamped_at_the_endpoint():
    assert advance(DynamicNumClusters(start=1, end=4, step_epochs=10), FakeModule(), 10_000) == 4


def test_a_descending_schedule_is_supported():
    """``end`` below ``start`` counts down -- the reason ``max`` was renamed to ``end``. The
    reversed ramp is one of the arms in the k-schedule write-up."""
    callback = DynamicNumClusters(start=16, end=1, step_epochs=10, step=2)
    assert advance(callback, FakeModule(), 0) == 16
    assert advance(callback, FakeModule(), 10) == 14
    assert advance(callback, FakeModule(), 1000) == 1, "clamped at the endpoint, not past it"


def test_step_larger_than_one_moves_faster():
    callback = DynamicNumClusters(start=1, end=16, step_epochs=10, step=4)
    assert advance(callback, FakeModule(), 10) == 5
    assert advance(callback, FakeModule(), 20) == 9


def test_k_is_logged_for_the_run_record():
    """The k actually in force is logged, which is the only runtime record of it -- there is no
    other report of the effective cluster count."""
    module = FakeModule()
    DynamicNumClusters().on_train_epoch_start(FakeTrainer(300), module)
    assert module.logged["train/num_clusters"] == 4.0


# --------------------------------------------------------------------------------------
# how it rebinds the regularizer
# --------------------------------------------------------------------------------------

def test_the_latest_epochs_k_wins_over_earlier_ones():
    """Each epoch wraps the *previous* regularizer in a new partial rather than replacing a
    value, so the wrappers nest. The outermost keyword wins, so the schedule is still correct --
    assert that directly, since correctness here is not obvious from the implementation."""
    callback, module = DynamicNumClusters(step_epochs=100), FakeModule()
    for epoch in (0, 100, 200, 300):
        callback.on_train_epoch_start(FakeTrainer(epoch), module)
    assert module.effective_k() == 4, "epoch 300 -> k=4, not the k from an earlier epoch"


def test_repeated_rebinding_does_not_accumulate_wrappers():
    """REFUTES cycle-0 candidate C0-2.

    ``pl_module.reg_loss = partial(pl_module.reg_loss, num_clusters=k)`` *looks* like it wraps
    the previous partial, which would nest 800 deep over an 800-epoch Imagenette run and add a
    call frame to every training step. It does not: CPython's ``partial.__new__`` detects that
    its target is already a ``partial`` and **flattens**, merging the keyword dicts with the
    outer call winning.

    So the depth stays at one however many epochs pass, and the schedule is correct for the same
    reason. Pinned because the code reads as though it leaks and the refutation is not obvious --
    without this test someone will "fix" it again.
    """
    callback, module = DynamicNumClusters(step_epochs=1), FakeModule()
    for epoch in range(50):
        callback.on_train_epoch_start(FakeTrainer(epoch), module)

    depth = 0
    fn = module.reg_loss
    while isinstance(fn, partial):
        depth += 1
        fn = fn.func
    assert depth == 1, f"partial should flatten to a single wrapper, found {depth}"
    assert module.effective_k() == 8, "and the schedule still resolves to the latest k"
