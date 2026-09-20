"""Datamodule construction and the guards that run before any data is touched.

The world-size divisibility checks are the interesting ones: they exist because a global batch
that does not divide across ranks would silently give different ranks different batch sizes,
changing the effective batch and -- for the local regularizers -- the k-means input.
"""

from __future__ import annotations

import pytest

from src.data.imagenet1k_datamodule import ImageNet1KDataModule, _apply_class_subset
from src.data.imagenette_datamodule import IMAGENETTE_VARIANTS, ImagenetteDataModule


class FakeTrainer:
    def __init__(self, world_size: int) -> None:
        self.world_size = world_size


def identity(x):
    return x


# --------------------------------------------------------------------------------------
# Imagenette
# --------------------------------------------------------------------------------------

def test_all_three_variants_are_known():
    assert set(IMAGENETTE_VARIANTS) == {"full_size", "320px", "160px"}
    for url, folder in IMAGENETTE_VARIANTS.values():
        assert url.startswith("https://") and folder.startswith("imagenette2")


def test_unknown_variant_raises_when_the_root_is_resolved():
    module = ImagenetteDataModule(train_transform=identity, variant="999px")
    with pytest.raises(ValueError, match="Unknown Imagenette variant"):
        _ = module._dataset_root


@pytest.mark.parametrize("split", ["test", "validation", "train2"])
def test_unknown_split_raises_at_construction(split):
    """Fails at construction rather than at setup, so a typo surfaces before a job is submitted
    rather than after it has queued."""
    with pytest.raises(ValueError, match="Unknown Imagenette split"):
        ImagenetteDataModule(train_transform=identity, val_split=split)


def test_num_classes_is_ten():
    assert ImagenetteDataModule(train_transform=identity).num_classes == 10


def test_val_batch_size_defaults_to_the_train_batch_size():
    module = ImagenetteDataModule(train_transform=identity, batch_size=128)
    assert module.val_batch_size == 128


def test_test_split_defaults_to_the_val_split():
    module = ImagenetteDataModule(train_transform=identity, val_split="val")
    assert module.test_split == "val"


def test_persistent_workers_follows_the_worker_count():
    assert ImagenetteDataModule(train_transform=identity, num_workers=4).persistent_workers is True
    assert ImagenetteDataModule(train_transform=identity, num_workers=0).persistent_workers is False


def test_batch_size_must_divide_the_world_size(imagenette_tree):
    """A global batch of 100 across 3 ranks cannot be split evenly. Silently rounding would
    change the effective batch -- and for the local regularizers the per-step batch *is* the
    k-means input, so it would change the method."""
    module = ImagenetteDataModule(train_transform=identity, data_dir=str(imagenette_tree),
                                  batch_size=100)
    module.trainer = FakeTrainer(world_size=3)
    with pytest.raises(RuntimeError, match="not divisible by the number of devices"):
        module.setup()


def test_batch_size_is_divided_across_ranks(imagenette_tree):
    module = ImagenetteDataModule(train_transform=identity, data_dir=str(imagenette_tree),
                                  batch_size=256)
    module.trainer = FakeTrainer(world_size=4)
    module.setup()
    assert module.batch_size_per_device == 64


def test_dataloader_before_setup_raises(imagenette_tree):
    module = ImagenetteDataModule(train_transform=identity, data_dir=str(imagenette_tree))
    with pytest.raises(RuntimeError, match="setup\\(\\) must create data_train"):
        module.train_dataloader()


def test_val_and_test_loaders_are_none_without_an_eval_transform(imagenette_tree):
    """Returning None rather than raising is deliberate: the I-JEPA configs train without a
    validation split."""
    module = ImagenetteDataModule(train_transform=identity, data_dir=str(imagenette_tree))
    module.setup()
    assert module.val_dataloader() is None and module.test_dataloader() is None


def test_setup_is_idempotent(imagenette_tree):
    """Lightning may call setup more than once; rebuilding the dataset each time would discard
    any state and waste the directory scan."""
    module = ImagenetteDataModule(train_transform=identity, data_dir=str(imagenette_tree))
    module.setup()
    first = module.data_train
    module.setup()
    assert module.data_train is first


# --------------------------------------------------------------------------------------
# ImageNet-1K
# --------------------------------------------------------------------------------------

def test_imagenet_num_classes_defaults_to_1000():
    assert ImageNet1KDataModule(train_transform=identity, data_dir="/x").num_classes == 1000


def test_imagenet_num_classes_follows_a_class_subset_file(tmp_path):
    subset = tmp_path / "subset.txt"
    subset.write_text("n001\nn002\n\nn003\n")
    module = ImageNet1KDataModule(train_transform=identity, data_dir="/x",
                                  class_subset_file=str(subset))
    assert module.num_classes == 3, "blank lines must not be counted"


def test_copy_data_is_an_unimplemented_trap(imagenet_tree):
    """``copy_data=True`` raises NotImplementedError with its body left as a string literal.
    Every shipped config sets it false, so the branch is dead -- but it is a trap for anyone who
    flips the flag, and the failure should stay loud rather than silently copying nothing.
    """
    module = ImageNet1KDataModule(train_transform=identity, data_dir=str(imagenet_tree),
                                  copy_data=True)
    with pytest.raises(NotImplementedError):
        module.setup()


def test_class_subset_remaps_labels_to_a_dense_range(imagenet_tree, tmp_path):
    """Filtering to a subset must renumber labels 0..k-1 in sorted class order; leaving the
    original indices would make a k-class probe index out of range."""
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(str(imagenet_tree / "imagenet_full_size" / "061417" / "train"))
    subset_file = tmp_path / "subset.txt"
    subset_file.write_text("n000\nn002\n")

    filtered = _apply_class_subset(dataset, subset_file)
    labels = sorted({t for _, t in filtered.samples})
    assert labels == [0, 1], "two retained classes must become labels 0 and 1"
    assert len(filtered.samples) == 8, "4 images from each of the two retained classes"


def test_class_subset_rejects_a_non_imagefolder_dataset():
    with pytest.raises(TypeError, match="ImageFolder-style dataset"):
        _apply_class_subset(object(), "/x")
