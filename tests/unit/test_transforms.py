"""Multi-view and multi-crop transforms.

The contract that matters is **view ordering**: ``MultiCropTransform`` promises global views
first, and ``invariance_loss``'s ``num_global`` slice takes that on trust. The second contract is
the return *type*, which decides whether ``encode`` takes its single-tensor or grouped path.
"""

from __future__ import annotations

import torch

from src.data.transforms.image import MultiCropTransform, MultiViewTransform


class Tagged:
    """A transform that ignores its input and returns a constant, so views are identifiable."""

    def __init__(self, value: float, size: int = 2) -> None:
        self.value, self.size = value, size
        self.calls = 0

    def __call__(self, _image) -> torch.Tensor:
        self.calls += 1
        return torch.full((3, self.size, self.size), self.value)


class Counting:
    """Returns a different value on each call, so repeated application is visible."""

    def __init__(self, size: int = 2) -> None:
        self.calls = 0
        self.size = size

    def __call__(self, _image) -> torch.Tensor:
        self.calls += 1
        return torch.full((3, self.size, self.size), float(self.calls))


# --------------------------------------------------------------------------------------
# MultiViewTransform
# --------------------------------------------------------------------------------------

def test_multiview_stacks_the_requested_number_of_views():
    out = MultiViewTransform(Tagged(1.0), num_views=4)(object())
    assert out.shape == (4, 3, 2, 2)


def test_multiview_applies_the_transform_independently_per_view():
    """Four *independent* draws, not one draw broadcast four times -- otherwise every view of a
    sample would be identical and the invariance loss would be trivially zero."""
    base = Counting()
    out = MultiViewTransform(base, num_views=4)(object())
    assert base.calls == 4
    assert len(out.flatten(1)[:, 0].unique()) == 4, "each view must be a separate draw"


def test_multiview_with_a_single_view():
    """num_views=1 is the eval transform's setting; it must still produce a leading view axis."""
    assert MultiViewTransform(Tagged(1.0), num_views=1)(object()).shape == (1, 3, 2, 2)


# --------------------------------------------------------------------------------------
# MultiCropTransform -- ordering and return type
# --------------------------------------------------------------------------------------

def test_multicrop_without_locals_returns_a_single_tensor():
    """No local transform means one resolution, so the historical (V, C, H, W) layout applies
    and ``encode`` takes its single-tensor path."""
    out = MultiCropTransform(Tagged(1.0), local_transform=None, num_global_crops=2)(object())
    assert isinstance(out, torch.Tensor) and out.shape == (2, 3, 2, 2)


def test_multicrop_same_resolution_returns_a_single_stacked_tensor():
    """Locals at the *same* resolution are concatenated rather than grouped -- the shapes match,
    so there is nothing to separate."""
    out = MultiCropTransform(Tagged(1.0, size=4), Tagged(2.0, size=4),
                             num_global_crops=2, num_local_crops=6)(object())
    assert isinstance(out, torch.Tensor) and out.shape == (8, 3, 4, 4)


def test_multicrop_same_resolution_puts_globals_first():
    """THE ordering contract. ``num_global`` slices ``proj[:k]``, so if locals came first the
    invariance loss would anchor on the wrong views -- silently, with no shape error."""
    out = MultiCropTransform(Tagged(1.0, size=4), Tagged(2.0, size=4),
                             num_global_crops=2, num_local_crops=3)(object())
    values = out.flatten(1)[:, 0]
    assert values[:2].tolist() == [1.0, 1.0], "the first two views must be the globals"
    assert values[2:].tolist() == [2.0, 2.0, 2.0]


def test_multicrop_mixed_resolution_returns_a_tuple_of_groups():
    """Different resolutions cannot be stacked, so the transform returns (globals, locals) and
    ``encode`` runs the encoder once per group. This is the ViT-L/14 paper setting."""
    out = MultiCropTransform(Tagged(1.0, size=8), Tagged(2.0, size=4),
                             num_global_crops=2, num_local_crops=8)(object())
    assert isinstance(out, tuple) and len(out) == 2

    globals_, locals_ = out
    assert globals_.shape == (2, 3, 8, 8)
    assert locals_.shape == (8, 3, 4, 4)
    assert globals_.flatten(1)[:, 0].tolist() == [1.0, 1.0], "globals are the first element"


def test_multicrop_calls_each_transform_the_requested_number_of_times():
    g, l = Counting(size=8), Counting(size=4)
    MultiCropTransform(g, l, num_global_crops=2, num_local_crops=6)(object())
    assert (g.calls, l.calls) == (2, 6)


def test_multicrop_with_zero_locals_ignores_the_local_transform():
    """num_local_crops=0 must not call the local transform at all."""
    local = Counting()
    out = MultiCropTransform(Tagged(1.0), local, num_global_crops=3, num_local_crops=0)(object())
    assert local.calls == 0
    assert isinstance(out, torch.Tensor) and out.shape[0] == 3
