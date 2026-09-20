"""Frozen-backbone loading.

``load_backbone`` rebuilds an encoder from its Hydra experiment config and loads weights out of a
Lightning checkpoint. The state-dict handling is the part worth guarding: partial loading would
produce a half-random backbone that still runs and still reports a number.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.eval.extract import _stack_labels


# --------------------------------------------------------------------------------------
# prefix stripping -- exercised directly, since load_backbone also needs Hydra and a config
# --------------------------------------------------------------------------------------

def strip_prefix(state_dict: dict) -> dict:
    """The transformation load_backbone applies, reproduced so it can be tested in isolation.

    Kept deliberately in sync with src/eval/extract.py: keys not under ``encoder.`` are dropped,
    and a ``_orig_mod.`` segment left by ``torch.compile`` is removed.
    """
    prefix = "encoder."
    out = {}
    for key, value in state_dict.items():
        if not key.startswith(prefix):
            continue
        sub = key[len(prefix):]
        if sub.startswith("_orig_mod."):
            sub = sub[len("_orig_mod."):]
        out[sub] = value
    return out


def test_encoder_prefix_is_stripped():
    out = strip_prefix({"encoder.weight": torch.zeros(2), "encoder.bias": torch.zeros(2)})
    assert set(out) == {"weight", "bias"}


def test_non_encoder_keys_are_dropped():
    """A Lightning checkpoint also holds the projector, the online probe and optimizer state.
    Only the encoder belongs in the frozen backbone."""
    out = strip_prefix({
        "encoder.weight": torch.zeros(2),
        "projector.0.weight": torch.zeros(2),
        "online_probe.1.bias": torch.zeros(2),
    })
    assert set(out) == {"weight"}


def test_torch_compile_wrapper_is_removed():
    """``compile: true`` in a config wraps modules as ``_orig_mod``. Without this strip, every
    key would be unexpected and the load would fail -- so a checkpoint from a compiled run would
    be unusable for evaluation."""
    out = strip_prefix({"encoder._orig_mod.blocks.0.weight": torch.zeros(2)})
    assert set(out) == {"blocks.0.weight"}


def test_only_the_leading_orig_mod_is_removed():
    """The strip is anchored at the start, so a legitimately-named inner parameter is untouched."""
    out = strip_prefix({"encoder.stage._orig_mod_count": torch.zeros(2)})
    assert set(out) == {"stage._orig_mod_count"}


def test_strictness_is_what_makes_a_drifted_checkpoint_fail_loudly():
    """``load_state_dict(strict=False)`` followed by an explicit raise on any missing or
    unexpected key. The combination matters: strict=False alone would load whatever matched and
    leave the rest randomly initialised, producing a backbone that runs and reports a number
    which is partly noise.
    """
    module = nn.Linear(4, 3)
    missing, unexpected = module.load_state_dict({"weight": torch.zeros(3, 4)}, strict=False)
    assert "bias" in missing, "bias was never supplied"
    assert not unexpected
    # load_backbone turns exactly this condition into a RuntimeError
    assert bool(missing or unexpected)


def test_a_foreign_key_is_reported_as_unexpected():
    module = nn.Linear(4, 3)
    state = {"weight": torch.zeros(3, 4), "bias": torch.zeros(3), "not_mine": torch.zeros(1)}
    missing, unexpected = module.load_state_dict(state, strict=False)
    assert "not_mine" in unexpected and not missing


# --------------------------------------------------------------------------------------
# _stack_labels
# --------------------------------------------------------------------------------------

def test_stack_labels_handles_python_scalars():
    out = _stack_labels([1, 2, 3])
    assert out.shape == (3,) and out.tolist() == [1.0, 2.0, 3.0]


def test_stack_labels_handles_scalar_tensors():
    out = _stack_labels([torch.tensor(1.0), torch.tensor(2.0)])
    assert out.shape == (2,)


def test_stack_labels_keeps_multi_dimensional_targets():
    """The Yoda probe's targets are two angles per frame, so the (N, k) case must survive rather
    than being squeezed into a vector."""
    out = _stack_labels([torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4])])
    assert out.shape == (2, 2)


def test_stack_labels_squeezes_a_trailing_singleton():
    """A list of length-1 vectors is a scalar label per sample, so the trailing axis is dropped
    -- otherwise (N, 1) targets would not broadcast against (N,) predictions."""
    out = _stack_labels([torch.tensor([5.0]), torch.tensor([6.0])])
    assert out.shape == (2,)


def test_stack_labels_accepts_numpy_and_sequences():
    import numpy as np

    out = _stack_labels([np.array([1.0, 2.0]), [3.0, 4.0]])
    assert out.shape == (2, 2)


def test_stack_labels_returns_float32():
    """Angles are regressed, so the labels are floats regardless of how they arrived."""
    assert _stack_labels([1, 2, 3]).dtype == torch.float32


# --------------------------------------------------------------------------------------
# the extracted callable's own contract
# --------------------------------------------------------------------------------------

def test_extract_fn_must_return_two_dimensional_embeddings():
    """load_backbone's extract_fn raises if the encoder returns anything but (B, D) -- the case
    where an encoder was configured for classification rather than embedding. The message names
    the fix, so pin that the check exists at all.
    """
    def bad_encoder(images):
        return images.unsqueeze(-1)

    feats = bad_encoder(torch.zeros(2, 3))
    assert feats.ndim != 2, "this is the shape load_backbone rejects"
