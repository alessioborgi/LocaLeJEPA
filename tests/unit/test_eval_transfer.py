"""Transfer-dataset split conventions.

``TRANSFER_DATASETS`` is an ``OrderedDict`` and its ordering is load-bearing: it fixes the column
order of every transfer table, and ``_average`` refuses to average unless all seven are present.
"""

from __future__ import annotations

import pytest
import torch

from src.eval.transfer import TRANSFER_DATASETS, _targets, get_transfer_splits


class FakeDataset:
    """Minimal stand-in exposing one of the label attributes ``_targets`` looks for."""

    def __init__(self, labels, attr: str = "targets") -> None:
        setattr(self, attr, labels)
        self._labels_list = labels

    def __len__(self) -> int:
        return len(self._labels_list)

    def __getitem__(self, index):
        return torch.zeros(3, 2, 2), self._labels_list[index]


# --------------------------------------------------------------------------------------
# the dataset registry
# --------------------------------------------------------------------------------------

def test_the_seven_reported_datasets_are_registered():
    assert set(TRANSFER_DATASETS) == {
        "dtd", "aircraft", "cifar10", "cifar100", "flowers", "food", "pets"
    }


def test_registry_order_is_stable():
    """An OrderedDict, and the table's column order follows it. Reordering would silently
    permute every published transfer row."""
    assert list(TRANSFER_DATASETS) == [
        "dtd", "aircraft", "cifar10", "cifar100", "flowers", "food", "pets"
    ]


def test_each_entry_carries_a_display_name_and_class_count():
    for key, (display, num_classes) in TRANSFER_DATASETS.items():
        assert isinstance(display, str) and display
        assert isinstance(num_classes, int) and num_classes > 1


def test_class_counts_match_the_published_protocol():
    """Pinned literally against the numbers in the transfer write-up: a wrong count would make
    a probe's output layer the wrong width and quietly cap accuracy."""
    expected = {"dtd": 47, "aircraft": 100, "cifar10": 10, "cifar100": 100,
                "flowers": 102, "food": 101, "pets": 37}
    assert {k: v[1] for k, v in TRANSFER_DATASETS.items()} == expected


# --------------------------------------------------------------------------------------
# label extraction -- the fallback chain
# --------------------------------------------------------------------------------------

def test_targets_attribute_is_preferred():
    assert _targets(FakeDataset([0, 1, 2, 1], "targets")) == [0, 1, 2, 1]


@pytest.mark.parametrize("attr", ["_labels", "labels"])
def test_alternative_label_attributes_are_found(attr):
    """torchvision is inconsistent across datasets -- some expose ``targets``, others
    ``_labels`` or ``labels``. The chain exists so the caller does not have to care."""
    assert _targets(FakeDataset([3, 1, 4], attr)) == [3, 1, 4]


def test_targets_falls_back_to_iterating_the_dataset():
    """The slow path: a dataset exposing no label attribute at all is decoded item by item.
    Correct but expensive, so it is worth knowing which datasets take it."""

    class Opaque:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return torch.zeros(3, 2, 2), index % 2

    assert _targets(Opaque()) == [0, 1, 0]


def test_targets_handles_tensor_labels():
    assert _targets(FakeDataset(torch.tensor([0, 1, 1]), "targets")) == [0, 1, 1]


def test_targets_recurses_into_a_concat_dataset():
    """Several transfer splits are train+val concatenations, so the labels must be gathered
    across both parts rather than from the wrapper."""
    from torch.utils.data import ConcatDataset

    combined = ConcatDataset([FakeDataset([0, 1]), FakeDataset([2, 3])])
    assert _targets(combined) == [0, 1, 2, 3]


# --------------------------------------------------------------------------------------
# split construction
# --------------------------------------------------------------------------------------

def test_an_unknown_key_raises_before_touching_disk():
    """Validation precedes any filesystem access, so a typo in a config fails instantly rather
    than after a download attempt or a confusing FileNotFoundError."""
    with pytest.raises(ValueError, match="Unknown transfer dataset|unknown"):
        get_transfer_splits("imagenet", "/definitely/not/here", crop_size=32, download=False)


def test_every_registered_key_is_handled_by_the_dispatch():
    """Each registered key must reach its own branch rather than the catch-all. A missing branch
    would surface only when that dataset was first evaluated -- late, and on the cluster."""
    for key in TRANSFER_DATASETS:
        with pytest.raises((FileNotFoundError, RuntimeError, OSError)):
            get_transfer_splits(key, "/definitely/not/here", crop_size=32, download=False)
