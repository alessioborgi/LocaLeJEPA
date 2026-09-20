"""The ImageNet-1K datamodule: the one that matters for the campaign.

Every headline number in this project comes from a run whose data went through here. The failure
mode is not a crash -- it is a completed 100-epoch run over subtly wrong data, and the tests below
are chosen for exactly that: label ordering, the class-subset remap, the ``copy_data`` trap, and
the global-vs-per-device batch size.

The ``imagenet_tree`` fixture builds a real (three-class, twelve-image) ImageFolder in ``tmp_path``
at the ``imagenet_full_size/061417`` sub-path the module expects, so nothing is mocked out.
"""

from __future__ import annotations

import pytest
import torch

from conftest import FakeTrainer
from src.data.imagenet1k_datamodule import ImageNet, ImageNet1KDataModule


def build(imagenet_tree, **kwargs):
    defaults = dict(train_transform=lambda x: x, data_dir=str(imagenet_tree), num_workers=0,
                    copy_data=False)
    return ImageNet1KDataModule(**{**defaults, **kwargs})


def to_tensor(image):
    from torchvision import transforms

    return transforms.ToTensor()(image)


# --------------------------------------------------------------------------------------
# the ImageNet dataset wrapper
# --------------------------------------------------------------------------------------

def test_the_train_and_val_suffixes_select_different_directories(imagenet_tree):
    """``train/`` and ``val/`` under the same root. Pointing both at ``train/`` would evaluate on
    the training set and produce a beautifully high, entirely meaningless number."""
    root = str(imagenet_tree)
    train = ImageNet(root=root, transform=to_tensor, train=True, copy_data=False)
    val = ImageNet(root=root, transform=to_tensor, train=False, copy_data=False)

    assert train.root.endswith("train/") and val.root.endswith("val/")
    assert {path for path, _ in train.samples}.isdisjoint({path for path, _ in val.samples})


def test_index_targets_builds_a_per_class_index(imagenet_tree):
    """``target_indices[c]`` is the sample positions of class ``c``. This is what class-balanced
    subsampling draws from, so a wrong index silently biases the subset -- and the resulting probe
    accuracy is simply lower, with nothing to attribute it to."""
    dataset = ImageNet(root=str(imagenet_tree), transform=to_tensor, train=True,
                       copy_data=False, index_targets=True)

    assert len(dataset.target_indices) == len(dataset.classes) == 3
    for class_index, positions in enumerate(dataset.target_indices):
        assert positions, "every class must be represented"
        for position in positions:
            assert int(dataset.samples[position][1]) == class_index


def test_the_index_covers_every_sample_exactly_once(imagenet_tree):
    """A partition, not merely a lookup: a sample appearing in two classes' lists or in none
    would make a 'balanced' subset the wrong size and the wrong composition."""
    dataset = ImageNet(root=str(imagenet_tree), transform=to_tensor, train=True,
                       copy_data=False, index_targets=True)

    flat = [position for positions in dataset.target_indices for position in positions]
    assert sorted(flat) == list(range(len(dataset.samples)))


def test_indexing_targets_is_off_by_default(imagenet_tree):
    """It converts ``samples`` to a numpy array of strings, which changes the type a caller gets
    back. Over 1.28M samples it also costs time and memory, so it stays opt-in."""
    dataset = ImageNet(root=str(imagenet_tree), transform=to_tensor, train=True, copy_data=False)
    assert not hasattr(dataset, "target_indices")
    assert isinstance(dataset.samples, list)


def test_copy_data_defaults_to_off(imagenet_tree):
    """It used to default to **True**, which raised ``NotImplementedError`` -- so constructing
    ``ImageNet`` directly, rather than through the datamodule, failed for no stated reason. The
    datamodule always passed False and all 25 configs set false, so the default was simply wrong.
    """
    dataset = ImageNet(root=str(imagenet_tree), transform=to_tensor, train=True)
    assert len(dataset.samples) > 0


def test_asking_for_node_local_copying_says_what_to_do_instead(imagenet_tree):
    """Still unimplemented -- ``copy_imgnt_locally`` exists nowhere in the repo -- but the error
    now names the working alternative.

    This is a feature someone will reach for on AICR, where it is genuinely the right idea: the
    dataset is on /scratch and the compute nodes have fast local NVMe. The answer is to stage in
    the job script before training starts, not inside the datamodule.
    """
    with pytest.raises(NotImplementedError, match="Stage the dataset to node-local storage"):
        ImageNet(root=str(imagenet_tree), transform=to_tensor, train=True, copy_data=True)


def test_the_datamodule_never_requests_copying_for_the_eval_splits(imagenet_tree):
    """``copy_data=self.copy_data if train else False``. The eval splits always read in place, so
    a configuration with copying on would fail on the train split only -- and the guard means the
    val split is never even attempted with it."""
    module = build(imagenet_tree, eval_transform=to_tensor, train_transform=to_tensor)
    module.setup()
    assert module.data_val is not None


# --------------------------------------------------------------------------------------
# num_classes
# --------------------------------------------------------------------------------------

def test_num_classes_is_a_thousand_by_default(imagenet_tree):
    assert build(imagenet_tree).num_classes == 1000


def test_num_classes_counts_the_lines_of_a_class_subset_file(imagenet_tree, tmp_path):
    """The 100-class ablations size their probe from this. Counting blank lines -- a trailing
    newline is normal -- would give 101 classes and a dead logit."""
    subset = tmp_path / "subset.txt"
    subset.write_text("n000\nn001\n\n  \nn002\n")
    assert build(imagenet_tree, class_subset_file=str(subset)).num_classes == 3


def test_a_missing_subset_file_falls_back_to_a_thousand(imagenet_tree, tmp_path):
    """The path is checked for existence rather than assumed. A typo'd path would otherwise take
    down a job at setup; instead it silently trains on all 1000 -- pinned here as the behaviour,
    since it is the kind of default that is worth knowing about before reading a result table."""
    assert build(imagenet_tree, class_subset_file=str(tmp_path / "absent.txt")).num_classes == 1000


# --------------------------------------------------------------------------------------
# setup and the loaders
# --------------------------------------------------------------------------------------

def test_setup_builds_train_val_and_test(imagenet_tree):
    """val and test are both built from the ``val/`` directory -- ImageNet-1K's test labels are
    not public, so the val split stands in for both. Stated here so the equality is not read as
    a bug."""
    module = build(imagenet_tree, train_transform=to_tensor, eval_transform=to_tensor)
    module.setup()

    assert module.data_train is not None
    assert len(module.data_val) == len(module.data_test)


def test_setup_is_idempotent(imagenet_tree):
    """Lightning calls setup once per stage. Re-walking 1.28M files on each would add minutes to
    every job for no benefit."""
    module = build(imagenet_tree, train_transform=to_tensor, eval_transform=to_tensor)
    module.setup("fit")
    first = module.data_train
    module.setup("validate")
    assert module.data_train is first


def test_the_eval_splits_are_skipped_without_an_eval_transform(imagenet_tree):
    module = build(imagenet_tree, train_transform=to_tensor)
    module.setup()
    assert module.data_val is None and module.data_test is None
    assert module.val_dataloader() is None and module.test_dataloader() is None


def test_the_global_batch_size_is_divided_across_ranks(imagenet_tree):
    """The campaign runs 4-GPU DDP. ``data.batch_size=64`` in a config is the global batch; if it
    were treated as per-device the effective batch would be 256 and the LeJEPA regularizer -- a
    distributional statistic over the batch -- would be estimated from four times the samples at
    a quarter of the step count. A different experiment, same config file."""
    module = build(imagenet_tree, train_transform=to_tensor, eval_transform=to_tensor,
                   batch_size=64, val_batch_size=32)
    trainer = FakeTrainer(estimated_stepping_batches=10, max_epochs=1)
    trainer.world_size = 4
    module.trainer = trainer
    module.setup()

    assert module.batch_size_per_device == 16
    assert module.val_batch_size_per_device == 8
    assert module.train_dataloader().batch_size == 16


def test_an_indivisible_batch_size_is_refused(imagenet_tree):
    module = build(imagenet_tree, train_transform=to_tensor, batch_size=10)
    trainer = FakeTrainer(estimated_stepping_batches=10, max_epochs=1)
    trainer.world_size = 4
    module.trainer = trainer

    with pytest.raises(RuntimeError, match=r"Batch size \(10\) is not divisible"):
        module.setup()


def test_an_indivisible_validation_batch_size_is_refused(imagenet_tree):
    module = build(imagenet_tree, train_transform=to_tensor, eval_transform=to_tensor,
                   batch_size=8, val_batch_size=10)
    trainer = FakeTrainer(estimated_stepping_batches=10, max_epochs=1)
    trainer.world_size = 4
    module.trainer = trainer

    with pytest.raises(RuntimeError, match=r"Validation batch size \(10\) is not divisible"):
        module.setup()


def test_the_train_loader_shuffles_and_drops_the_last_partial_batch(imagenet_tree):
    from torch.utils.data import RandomSampler, SequentialSampler

    module = build(imagenet_tree, train_transform=to_tensor, eval_transform=to_tensor,
                   batch_size=2)
    module.setup()

    train, val = module.train_dataloader(), module.val_dataloader()
    assert isinstance(train.sampler, RandomSampler) and train.drop_last is True
    assert isinstance(val.sampler, SequentialSampler) and val.drop_last is False


def test_the_collator_is_attached_to_training_only(imagenet_tree):
    collator = object()
    module = build(imagenet_tree, train_transform=to_tensor, eval_transform=to_tensor,
                   collator=collator)
    module.setup()

    assert module.train_dataloader().collate_fn is collator
    assert module.val_dataloader().collate_fn is not collator


def test_the_train_loader_refuses_to_build_before_setup(imagenet_tree):
    with pytest.raises(RuntimeError, match=r"setup\(\) must create data_train"):
        build(imagenet_tree).train_dataloader()


def test_a_batch_comes_out_with_the_expected_shape(imagenet_tree):
    """End to end through a real DataLoader over real JPEGs -- the one place the whole chain is
    exercised rather than inspected."""
    module = build(imagenet_tree, train_transform=to_tensor, batch_size=2)
    module.setup()

    images, labels = next(iter(module.train_dataloader()))
    assert images.shape == (2, 3, 8, 8) and labels.shape == (2,)
    assert labels.dtype == torch.int64


# --------------------------------------------------------------------------------------
# the class subset
# --------------------------------------------------------------------------------------

def test_the_class_subset_keeps_only_the_listed_synsets(imagenet_tree, tmp_path):
    subset = tmp_path / "subset.txt"
    subset.write_text("n000\nn002\n")

    module = build(imagenet_tree, train_transform=to_tensor, class_subset_file=str(subset))
    module.setup()

    assert len(module.data_train) == 8, "two of the three classes, four images each"


def test_the_class_subset_remaps_labels_to_a_dense_range(imagenet_tree, tmp_path):
    """THE ONE THAT MATTERS. ImageFolder assigns indices over all 1000 synsets; a 100-class
    subset must be relabelled 0..99 or the probe's output layer is indexed out of range -- or,
    with a 1000-wide layer, trains 900 dead logits and reports a meaningless accuracy.

    The remap is by *sorted synset name*, which is the same rule ImageFolder itself uses, so the
    subset's label order matches what a 100-class ImageFolder would have produced.
    """
    subset = tmp_path / "subset.txt"
    subset.write_text("n002\nn000\n")          # deliberately unsorted in the file

    module = build(imagenet_tree, train_transform=to_tensor, class_subset_file=str(subset))
    module.setup()

    labels = sorted({target for _, target in module.data_train.samples})
    assert labels == [0, 1], "dense and zero-based"

    for path, target in module.data_train.samples:
        expected = {"n000": 0, "n002": 1}[path.split("/")[-2]]
        assert target == expected, "sorted synset order, not file order"


def test_the_remapped_targets_attribute_agrees_with_the_samples(imagenet_tree, tmp_path):
    """``targets`` and ``samples`` are both read downstream -- by the sampler and by the loader
    respectively. If they disagreed, a class-balanced sampler would select by one label and the
    model would be trained on another."""
    subset = tmp_path / "subset.txt"
    subset.write_text("n000\nn001\n")

    module = build(imagenet_tree, train_transform=to_tensor, class_subset_file=str(subset))
    module.setup()

    assert module.data_train.targets == [target for _, target in module.data_train.samples]


def test_a_synset_in_the_file_but_not_on_disk_is_ignored(imagenet_tree, tmp_path):
    """A 100-class list against a partially-staged dataset. The listed-but-absent synset simply
    contributes nothing, rather than raising a KeyError halfway through the remap."""
    subset = tmp_path / "subset.txt"
    subset.write_text("n000\nn999\n")

    module = build(imagenet_tree, train_transform=to_tensor, class_subset_file=str(subset))
    module.setup()
    assert len(module.data_train) == 4


def test_the_class_subset_rejects_a_non_imagefolder_dataset():
    from src.data.imagenet1k_datamodule import _apply_class_subset

    with pytest.raises(TypeError, match="expects an ImageFolder-style dataset"):
        _apply_class_subset(object(), "whatever.txt")
