"""``get_transfer_splits``: the seven SPHERE-JEPA Table 1 datasets and their split convention.

The module's whole contribution is *which torchvision splits get concatenated into "train"*. That
convention is what makes the numbers comparable to SimCLR / BYOL / DINO, and it is invisible in
any output: using DTD's ``train`` alone instead of ``train+val`` halves the probe's training set
and simply produces a lower number, with nothing to indicate why.

torchvision is replaced with a recorder here. No archive is downloaded, and what is asserted is
exactly what this module decides: the split names, the concatenation, the shared transform, and
the class-count check.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import ConcatDataset

from src.eval.transfer import TRANSFER_DATASETS, _targets, get_transfer_splits


# The published convention, restated independently of the implementation.
# key -> (torchvision splits concatenated into "train", test split, classes)
EXPECTED = {
    "dtd": (["train", "val"], "test", 47),
    "aircraft": (["trainval"], "test", 100),
    "cifar10": ([True], False, 10),          # CIFAR takes train=True/False, not a split name
    "cifar100": ([True], False, 100),
    "flowers": (["train", "val"], "test", 102),
    "food": (["train"], "test", 101),
    "pets": (["trainval"], "test", 37),
}


class Recorder:
    """Stands in for one torchvision dataset class and records its construction."""

    calls: list[dict] = []

    def __init__(self, name, n_classes):
        self.name, self.n_classes = name, n_classes

    def __call__(self, root=None, transform=None, download=None, split=None, train=None,
                 partition=None, annotation_level=None, target_types=None):
        Recorder.calls.append({"dataset": self.name, "split": split, "train": train,
                               "partition": partition, "annotation_level": annotation_level,
                               "target_types": target_types, "root": root,
                               "transform": transform, "download": download})
        return FakeSplit(self.n_classes)


class FakeSplit(torch.utils.data.Dataset):
    """One label per class, so ``max(targets) + 1`` equals the declared class count."""

    def __init__(self, n_classes):
        self.targets = list(range(n_classes))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.zeros(3, 4, 4), self.targets[index]


@pytest.fixture
def torchvision(monkeypatch):
    import src.eval.transfer as module

    Recorder.calls = []
    fake = type("tvd", (), {
        "DTD": Recorder("DTD", 47),
        "FGVCAircraft": Recorder("FGVCAircraft", 100),
        "CIFAR10": Recorder("CIFAR10", 10),
        "CIFAR100": Recorder("CIFAR100", 100),
        "Flowers102": Recorder("Flowers102", 102),
        "Food101": Recorder("Food101", 101),
        "OxfordIIITPet": Recorder("OxfordIIITPet", 37),
    })
    monkeypatch.setattr(module, "tvd", fake)
    return module


# --------------------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------------------

def test_the_seven_table_one_datasets_are_registered_in_order():
    """An ``OrderedDict``, and the order is the published table's column order. The transfer
    average is taken over exactly these seven keys, so adding an eighth silently changes what
    'mean' means in every reported row."""
    assert list(TRANSFER_DATASETS) == ["dtd", "aircraft", "cifar10", "cifar100",
                                       "flowers", "food", "pets"]


def test_the_registered_class_counts_are_the_published_ones():
    """47 / 100 / 10 / 100 / 102 / 101 / 37. These size the probe's output layer; a wrong count
    either truncates the label space or leaves dead logits, and neither raises."""
    assert {k: v[1] for k, v in TRANSFER_DATASETS.items()} == {k: v[2] for k, v in EXPECTED.items()}


def test_an_unknown_key_lists_the_valid_ones():
    """The keys are ``flowers`` and ``food``, not ``flowers102`` and ``food101`` -- a slip worth
    an explicit message, since the obvious guesses are both wrong."""
    with pytest.raises(ValueError, match="Unknown transfer dataset 'flowers102'"):
        get_transfer_splits("flowers102", root="/tmp", crop_size=32)


# --------------------------------------------------------------------------------------
# the split convention
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", list(TRANSFER_DATASETS))
def test_each_dataset_uses_its_published_split_convention(torchvision, key):
    """The core assertion of this file, one dataset at a time. Checked against the table in the
    module docstring, restated at the top of this file so the two are independent."""
    train_splits, test_split, n_classes = EXPECTED[key]
    _, _, observed_classes = get_transfer_splits(key, root="/data", crop_size=32)

    assert observed_classes == n_classes
    field = "train" if key.startswith("cifar") else "split"
    assert [call[field] for call in Recorder.calls] == [*train_splits, test_split]


@pytest.mark.parametrize("key", ["dtd", "flowers"])
def test_train_and_validation_are_concatenated_for_the_two_small_datasets(torchvision, key):
    """DTD and Flowers-102 have tiny official train splits (1880 and 1020 images). The benchmark
    convention trains the probe on train+val; using train alone is the difference between a
    comparable number and a low one."""
    train, test, _ = get_transfer_splits(key, root="/data", crop_size=32)

    assert isinstance(train, ConcatDataset) and len(train.datasets) == 2
    assert not isinstance(test, ConcatDataset), "the test split is never concatenated"


def test_dtd_uses_partition_one(torchvision):
    """DTD ships ten train/val/test partitions. The benchmark reports partition 1; averaging over
    all ten, or silently taking torchvision's default, gives a different number under the same
    name."""
    get_transfer_splits("dtd", root="/data", crop_size=32)
    assert {call["partition"] for call in Recorder.calls} == {1}


def test_aircraft_uses_the_variant_annotation_level(torchvision):
    """FGVC-Aircraft has three label granularities -- variant (100), family (70), manufacturer
    (30). Only ``variant`` gives the 100 classes Table 1 reports, and the other two would each
    produce a valid, higher accuracy."""
    get_transfer_splits("aircraft", root="/data", crop_size=32)
    assert {call["annotation_level"] for call in Recorder.calls} == {"variant"}


def test_pets_uses_category_targets_not_segmentation(torchvision):
    """``target_types`` defaults to ``category`` in torchvision, but it is pinned explicitly
    because ``segmentation`` would return masks where the probe expects labels."""
    get_transfer_splits("pets", root="/data", crop_size=32)
    assert {call["target_types"] for call in Recorder.calls} == {"category"}


# --------------------------------------------------------------------------------------
# shared configuration across every split
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", list(TRANSFER_DATASETS))
def test_train_and_test_share_one_transform_object(torchvision, key):
    """Built once and passed to every split. If train and test were transformed differently, the
    probe would be fitted and evaluated in different feature spaces -- which lowers accuracy
    without failing, and would be nearly impossible to attribute."""
    get_transfer_splits(key, root="/data", crop_size=32)
    transforms = {id(call["transform"]) for call in Recorder.calls}
    assert len(transforms) == 1


def test_the_transform_is_the_in_domain_eval_transform(torchvision):
    """The same function the in-domain probe uses. Transfer and in-domain numbers appear in the
    same tables, so they have to be produced through the same pipeline."""
    from src.eval.indomain import build_eval_transform

    get_transfer_splits("cifar10", root="/data", crop_size=64)
    built = Recorder.calls[0]["transform"]
    assert repr(built) == repr(build_eval_transform(64))


@pytest.mark.parametrize("download", [True, False])
def test_downloading_is_off_unless_asked(torchvision, download):
    """Compute nodes on AICR have no outbound network. A default of True would turn a missing
    archive into a hung job rather than an immediate, readable failure -- and the archives are
    fetched on the login node ahead of time."""
    get_transfer_splits("cifar10", root="/data", crop_size=32, download=download)
    assert {call["download"] for call in Recorder.calls} == {download}


def test_the_root_is_shared_and_normalised(torchvision):
    """One root for all seven, passed through ``Path`` so a trailing slash or a doubled separator
    does not produce a second cache directory alongside the populated one."""
    get_transfer_splits("cifar10", root="/data//sub/", crop_size=32)
    assert {call["root"] for call in Recorder.calls} == {"/data/sub"}


# --------------------------------------------------------------------------------------
# the class-count guard
# --------------------------------------------------------------------------------------

def test_a_label_space_that_disagrees_with_the_registry_is_refused(torchvision, monkeypatch):
    """The guard that catches a corrupted or partial download: if the archive only unpacked 8 of
    CIFAR-10's classes, the probe would train on 8, evaluate on 8, and report a *higher* accuracy
    than the real benchmark. Refused with both numbers named.
    """
    import src.eval.transfer as module

    monkeypatch.setattr(module.tvd, "CIFAR10", Recorder("CIFAR10", 8))
    with pytest.raises(RuntimeError, match=r"cifar10: expected 10 classes, labels span 8"):
        get_transfer_splits("cifar10", root="/data", crop_size=32)


def test_the_guard_looks_at_both_splits(torchvision, monkeypatch):
    """``max(max(train), max(test))``. A test split missing its last class is just as invalidating
    as a train split missing one, and only the combined maximum catches both."""
    import src.eval.transfer as module

    class Asymmetric(Recorder):
        def __call__(self, **kwargs):
            Recorder.calls.append(kwargs)
            return FakeSplit(10 if kwargs.get("train") else 6)

    monkeypatch.setattr(module.tvd, "CIFAR10", Asymmetric("CIFAR10", 10))
    get_transfer_splits("cifar10", root="/data", crop_size=32)   # train reaches 10: accepted


# --------------------------------------------------------------------------------------
# _targets
# --------------------------------------------------------------------------------------

def test_targets_are_read_without_decoding_images():
    """Reading labels via ``dataset[i]`` decodes and transforms every image -- minutes per dataset
    and a JPEG decode for a number already held in memory. The attribute names differ across
    torchvision classes, hence the three-way lookup."""
    for attribute in ("targets", "_labels", "labels"):
        dataset = type("D", (), {attribute: [3, 1, 2]})()
        assert _targets(dataset) == [3, 1, 2]


def test_targets_are_read_in_preference_order():
    """``targets`` wins over ``_labels``. Some classes define both, and they are not always the
    same thing; the order has to be fixed rather than dictionary-dependent."""
    dataset = type("D", (), {"targets": [1, 1], "_labels": [9, 9]})()
    assert _targets(dataset) == [1, 1]


def test_targets_of_a_concatenation_are_the_concatenation_of_targets():
    """In split order. A ``ConcatDataset`` has no ``targets`` of its own, so without this branch
    the fallback would decode every image in DTD and Flowers."""
    parts = ConcatDataset([FakeSplit(3), FakeSplit(2)])
    assert _targets(parts) == [0, 1, 2, 0, 1]


def test_targets_fall_back_to_indexing_when_no_attribute_exists():
    """The last resort, kept because a future torchvision class may expose neither name. Slow but
    correct, which is the right trade for a path that should never be taken."""

    class NoAttributes(torch.utils.data.Dataset):
        def __len__(self):
            return 3

        def __getitem__(self, index):
            return torch.zeros(1), index * 2

    assert _targets(NoAttributes()) == [0, 2, 4]
