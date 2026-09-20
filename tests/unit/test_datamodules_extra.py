"""STL10 and the bobblehead torus dataset.

Neither is in the main ImageNet campaign -- STL10 backs the I-JEPA configs, bobblehead backs the
conformal-autoencoder dimension-detection experiment -- but both are Tier A: they decide what data
reaches a model, and a datamodule defect produces a completed run over the wrong images rather
than an error.

No download happens here. STL10 is exercised against a stand-in that records how torchvision was
called, which is the whole of what the datamodule contributes; bobblehead gets a real zip built in
``tmp_path``, because its loader's job *is* unpacking one.
"""

from __future__ import annotations

import zipfile

import numpy as np
import pytest
import torch


# ======================================================================================
# STL10
# ======================================================================================

class FakeSTL10:
    """Records every construction. Ten items so a DataLoader can be built over it."""

    instances: list[dict] = []

    def __init__(self, root, split, download, transform=None):
        self.root, self.split, self.download, self.transform = root, split, download, transform
        FakeSTL10.instances.append({"root": root, "split": split, "download": download,
                                    "transform": transform})

    def __len__(self):
        return 10

    def __getitem__(self, index):
        return torch.zeros(3, 8, 8), index % 10


@pytest.fixture
def stl10(monkeypatch):
    import src.data.stl10_datamodule as module

    FakeSTL10.instances = []
    monkeypatch.setattr(module, "STL10", FakeSTL10)
    return module


def build_stl10(stl10, **kwargs):
    defaults = dict(train_transform=lambda x: x, data_dir="data", num_workers=0)
    return stl10.STL10DataModule(**{**defaults, **kwargs})


@pytest.mark.parametrize("split", ["train", "test", "unlabeled", "train+unlabeled"])
def test_every_real_stl10_split_is_accepted(stl10, split):
    assert build_stl10(stl10, train_split=split).train_split == split


def test_an_unknown_split_is_refused_at_construction(stl10):
    """torchvision would raise the same thing eventually, but only after ``prepare_data`` has
    started a download. Failing in the constructor means a config typo costs a second."""
    with pytest.raises(ValueError, match="Unknown STL10 split 'unlabelled'"):
        build_stl10(stl10, train_split="unlabelled")


def test_a_bad_split_is_caught_in_the_val_and_test_positions_too(stl10):
    """All three are validated in one loop; a guard that only checked ``train_split`` would let
    the eval splits through, and those are what the reported numbers come from."""
    with pytest.raises(ValueError, match="Unknown STL10 split"):
        build_stl10(stl10, val_split="validation")


def test_the_test_split_defaults_to_the_validation_split(stl10):
    """So ``val_split=test`` alone gives a working three-way configuration -- which is how every
    I-JEPA config is written."""
    assert build_stl10(stl10, val_split="test").test_split == "test"
    assert build_stl10(stl10, val_split="test", test_split="train").test_split == "train"


def test_the_validation_batch_size_defaults_to_the_training_one(stl10):
    assert build_stl10(stl10, batch_size=64).val_batch_size == 64
    assert build_stl10(stl10, batch_size=64, val_batch_size=8).val_batch_size == 8


@pytest.mark.parametrize("workers, expected", [(0, False), (4, True)])
def test_persistent_workers_follows_the_worker_count_unless_set(stl10, workers, expected):
    """``persistent_workers=True`` with ``num_workers=0`` is a hard error in PyTorch, so the
    default has to be derived rather than fixed. It is still overridable."""
    assert build_stl10(stl10, num_workers=workers).persistent_workers is expected
    assert build_stl10(stl10, num_workers=workers, persistent_workers=False).persistent_workers is False


def test_prepare_data_downloads_only_the_splits_that_will_be_used(stl10):
    """Each STL10 split is a separate multi-gigabyte archive. Downloading the eval splits when no
    ``eval_transform`` was configured would cost bandwidth and disk for data the run never reads.
    """
    build_stl10(stl10, train_split="unlabeled", val_split="test").prepare_data()
    assert [i["split"] for i in FakeSTL10.instances] == ["unlabeled"], "no eval_transform given"

    FakeSTL10.instances = []
    build_stl10(stl10, train_split="unlabeled", val_split="test",
                eval_transform=lambda x: x).prepare_data()
    assert [i["split"] for i in FakeSTL10.instances] == ["unlabeled", "test"]


def test_prepare_data_does_not_fetch_the_same_split_twice(stl10):
    """``test_split`` defaults to ``val_split``, so the naive version downloads 'test' twice."""
    build_stl10(stl10, train_split="train", val_split="test",
                eval_transform=lambda x: x).prepare_data()
    assert [i["split"] for i in FakeSTL10.instances].count("test") == 1


def test_setup_builds_datasets_without_downloading(stl10):
    """``download=False`` in ``setup``: on a multi-node job every rank runs ``setup`` while only
    rank zero runs ``prepare_data``. A download here would be N concurrent writes to one cache
    directory."""
    module = build_stl10(stl10, val_split="test", eval_transform=lambda x: x)
    module.setup()

    setup_calls = [i for i in FakeSTL10.instances if i["download"] is False]
    assert len(setup_calls) == 3, "train, val and test"
    assert all(call["download"] is False for call in setup_calls)


def test_setup_is_idempotent(stl10):
    """Lightning calls ``setup`` once per stage -- fit, then validate, then test. Rebuilding the
    dataset each time would re-scan the archive and, worse, discard any state."""
    module = build_stl10(stl10, val_split="test", eval_transform=lambda x: x)
    module.setup("fit")
    first = module.data_train
    module.setup("test")
    assert module.data_train is first


def test_eval_datasets_are_skipped_without_an_eval_transform(stl10):
    """An I-JEPA config has no eval transform. Building the eval datasets anyway would hand
    raw PIL images to a dataloader that then fails at collation, halfway into the first epoch."""
    module = build_stl10(stl10, val_split="test")
    module.setup()
    assert module.data_train is not None
    assert module.data_val is None and module.data_test is None
    assert module.val_dataloader() is None and module.test_dataloader() is None


def test_the_per_device_batch_size_is_divided_by_the_world_size(stl10):
    """``batch_size`` in a config is the *global* batch. Getting this wrong scales the effective
    batch by the device count, which changes the optimisation problem without changing anything
    visible in the config."""
    from conftest import FakeTrainer

    module = build_stl10(stl10, batch_size=64, val_batch_size=32)
    trainer = FakeTrainer(estimated_stepping_batches=10, max_epochs=1)
    trainer.world_size = 4
    module.trainer = trainer
    module.setup()

    assert module.batch_size_per_device == 16
    assert module.val_batch_size_per_device == 8


def test_an_indivisible_batch_size_is_refused(stl10):
    """Silent flooring would give ranks different batch sizes and a global batch that is not the
    configured one. Refused instead, with both numbers in the message."""
    from conftest import FakeTrainer

    module = build_stl10(stl10, batch_size=10)
    trainer = FakeTrainer(estimated_stepping_batches=10, max_epochs=1)
    trainer.world_size = 4
    module.trainer = trainer

    with pytest.raises(RuntimeError, match=r"Batch size \(10\) is not divisible"):
        module.setup()


def test_an_indivisible_validation_batch_size_is_refused(stl10):
    from conftest import FakeTrainer

    module = build_stl10(stl10, batch_size=8, val_batch_size=10, val_split="test",
                         eval_transform=lambda x: x)
    trainer = FakeTrainer(estimated_stepping_batches=10, max_epochs=1)
    trainer.world_size = 4
    module.trainer = trainer

    with pytest.raises(RuntimeError, match=r"Validation batch size \(10\) is not divisible"):
        module.setup()


def test_the_train_loader_shuffles_and_the_eval_loaders_do_not(stl10):
    """Shuffled evaluation is not wrong on its own, but combined with feature caching keyed by
    position it silently misaligns features and labels."""
    module = build_stl10(stl10, batch_size=2, val_split="test", eval_transform=lambda x: x)
    module.setup()

    from torch.utils.data import RandomSampler, SequentialSampler

    assert isinstance(module.train_dataloader().sampler, RandomSampler)
    assert isinstance(module.val_dataloader().sampler, SequentialSampler)
    assert isinstance(module.test_dataloader().sampler, SequentialSampler)


def test_the_train_loader_drops_the_last_partial_batch_by_default(stl10):
    """LeJEPA's regularizer is a distributional statistic over the batch; a final batch of 3
    would contribute a badly estimated one. The eval loaders must NOT drop, or samples vanish
    from the reported denominator."""
    module = build_stl10(stl10, batch_size=4, val_split="test", eval_transform=lambda x: x)
    module.setup()

    assert module.train_dataloader().drop_last is True
    assert module.val_dataloader().drop_last is False


def test_the_collator_reaches_the_train_loader_only(stl10):
    """The I-JEPA mask collator applies to training batches. Attaching it to the eval loaders
    would hand masks to a probe that expects images."""
    collator = object()
    module = build_stl10(stl10, collator=collator, val_split="test", eval_transform=lambda x: x)
    module.setup()

    assert module.train_dataloader().collate_fn is collator
    assert module.val_dataloader().collate_fn is not collator


def test_the_train_loader_refuses_to_build_before_setup(stl10):
    """Rather than returning a loader over ``None``, whose failure surfaces as an opaque
    collation error from a worker process."""
    with pytest.raises(RuntimeError, match="setup\\(\\) must create data_train"):
        build_stl10(stl10).train_dataloader()


def test_stl10_has_ten_classes(stl10):
    assert build_stl10(stl10).num_classes == 10


def test_the_transforms_are_kept_out_of_the_saved_hyperparameters(stl10):
    """``save_hyperparameters`` writes into every checkpoint. A transform object there makes the
    checkpoint depend on the classes being importable, so an unrelated refactor makes old
    checkpoints unloadable."""
    module = build_stl10(stl10, val_split="test", eval_transform=lambda x: x)
    for excluded in ("train_transform", "eval_transform", "collator"):
        assert excluded not in module.hparams


# ======================================================================================
# bobblehead: the rotating-figures torus
# ======================================================================================

from src.data.bobblehead_datamodule import (  # noqa: E402
    BobbleheadDataset,
    build_eval_transform,
    extract_bobblehead,
    get_bobblehead_dataset,
)

N_FRAMES = 6


@pytest.fixture
def archive(tmp_path):
    """A real zip of the real shape: ``color_image_export.npy`` of (N, H, W, 3) uint8 plus a
    ``labels.npy`` stored transposed as (2, N), which is how the published archive stores it."""
    images = np.random.randint(0, 255, size=(N_FRAMES, 24, 32, 3), dtype=np.uint8)
    angles = np.stack([np.linspace(0, 6.2, N_FRAMES), np.linspace(0, 3.1, N_FRAMES)])

    staging = tmp_path / "staging"
    staging.mkdir()
    np.save(staging / "color_image_export.npy", images)
    np.save(staging / "labels.npy", angles)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with zipfile.ZipFile(data_dir / "yoda_dog_images_and_labels.zip", "w") as zf:
        for name in ("color_image_export.npy", "labels.npy"):
            zf.write(staging / name, arcname=name)

    return data_dir, images, angles


def test_the_archive_is_extracted_and_the_image_array_located(archive, tmp_path):
    data_dir, images, _ = archive
    found = extract_bobblehead(data_dir / "yoda_dog_images_and_labels.zip", tmp_path / "out")

    assert found.name == "color_image_export.npy"
    np.testing.assert_array_equal(np.load(found), images)


def test_extraction_is_skipped_when_the_array_is_already_there(archive, tmp_path):
    """8097 frames at 240x320x3 is 1.8 GB. Re-extracting on every evaluation would dominate the
    runtime of a job whose actual work is a forward pass."""
    data_dir, _, _ = archive
    out = tmp_path / "out"
    first = extract_bobblehead(data_dir / "yoda_dog_images_and_labels.zip", out)
    mtime = first.stat().st_mtime_ns

    (data_dir / "yoda_dog_images_and_labels.zip").unlink()   # gone: nothing to extract from
    assert extract_bobblehead(data_dir / "yoda_dog_images_and_labels.zip", out) == first
    assert first.stat().st_mtime_ns == mtime


def test_a_missing_archive_says_which_path_it_looked_at(tmp_path):
    with pytest.raises(FileNotFoundError, match="Bobblehead archive not found"):
        extract_bobblehead(tmp_path / "absent.zip", tmp_path / "out")


def test_a_truncated_archive_is_diagnosed_as_an_incomplete_download(tmp_path):
    """The failure mode this guard exists for: a 1.8 GB download interrupted by a walltime limit
    leaves a partial file. ``zipfile`` would raise ``BadZipFile``, which reads like corruption;
    the message here names the actual cause and the actual fix."""
    partial = tmp_path / "partial.zip"
    partial.write_bytes(b"PK\x03\x04" + b"\x00" * 64)

    with pytest.raises(RuntimeError, match="download may be in progress or truncated"):
        extract_bobblehead(partial, tmp_path / "out")


def test_the_image_array_is_found_inside_a_subdirectory(tmp_path):
    """Some copies of the archive nest everything one level down. The rglob fallback handles it;
    without that the extraction 'succeeds' and then fails on a missing file."""
    images = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    staging = tmp_path / "s"
    staging.mkdir()
    np.save(staging / "color_image_export.npy", images)

    zip_path = tmp_path / "nested.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(staging / "color_image_export.npy", arcname="inner/color_image_export.npy")

    found = extract_bobblehead(zip_path, tmp_path / "out")
    assert found.parent.name == "inner"


def test_an_archive_without_the_image_array_lists_what_it_did_contain(tmp_path):
    zip_path = tmp_path / "wrong.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("readme.txt", "not the data")

    with pytest.raises(FileNotFoundError, match="color_image_export.npy not found"):
        extract_bobblehead(zip_path, tmp_path / "out")


def test_transposed_labels_are_returned_one_row_per_frame(archive, tmp_path):
    """THE ORIENTATION TRAP. The published ``labels.npy`` is (2, 8097) -- two angle rows -- not
    (8097, 2). Indexing the untransposed array by frame would return the two angles of frames 0
    and 1 for *every* sample, which is a perfectly-shaped, entirely wrong label set that the
    conformal autoencoder would happily fit.

    Only detectable by value, which is why the fixture uses two distinguishable linspaces.
    """
    data_dir, _, angles = archive
    dataset = BobbleheadDataset(data_dir=data_dir, extract_dir=tmp_path / "out", crop_size=8)

    assert dataset.has_labels
    assert dataset.labels.shape == (N_FRAMES, 2)
    for frame in range(N_FRAMES):
        _, label = dataset[frame]
        torch.testing.assert_close(label, torch.tensor(angles[:, frame], dtype=torch.float32))


def test_a_dataset_without_labels_falls_back_to_the_frame_index(tmp_path):
    """Labels are optional -- the dimension-detection task is unsupervised. The frame index is a
    time coordinate, which is what the recovered-torus plots are coloured by."""
    images = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    staging = tmp_path / "s"
    staging.mkdir()
    np.save(staging / "color_image_export.npy", images)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with zipfile.ZipFile(data_dir / "yoda_dog_images_and_labels.zip", "w") as zf:
        zf.write(staging / "color_image_export.npy", arcname="color_image_export.npy")

    dataset = BobbleheadDataset(data_dir=data_dir, extract_dir=tmp_path / "out", crop_size=8)
    assert not dataset.has_labels
    assert float(dataset[2][1]) == 2.0


def test_subsampling_keeps_temporal_order(archive, tmp_path):
    """The manifold structure *is* the temporal structure -- the frames trace a path on a torus.
    Shuffling or reversing would not change any shape, and would destroy the experiment."""
    data_dir, _, angles = archive
    dataset = BobbleheadDataset(data_dir=data_dir, extract_dir=tmp_path / "out",
                                crop_size=8, subsample=2)

    assert dataset.indices == [0, 2, 4]
    torch.testing.assert_close(dataset[1][1], torch.tensor(angles[:, 2], dtype=torch.float32))


def test_max_frames_truncates_after_subsampling(archive, tmp_path):
    """Order of operations: subsample then truncate. The other order would give a different,
    denser window of the trajectory for the same two arguments."""
    data_dir, _, _ = archive
    dataset = BobbleheadDataset(data_dir=data_dir, extract_dir=tmp_path / "out",
                                crop_size=8, subsample=2, max_frames=2)
    assert dataset.indices == [0, 2]
    assert len(dataset) == 2


def test_a_subsample_of_zero_does_not_divide_by_zero(archive, tmp_path):
    """``max(1, subsample)``. A config that computes the stride and lands on 0 gets stride 1
    rather than a ``range`` step error."""
    data_dir, _, _ = archive
    assert BobbleheadDataset(data_dir=data_dir, extract_dir=tmp_path / "out",
                             crop_size=8, subsample=0).indices == list(range(N_FRAMES))


def test_a_wrongly_shaped_image_array_is_refused(tmp_path):
    """Greyscale or channel-first frames would still index and still transform, into a model that
    then sees structured noise."""
    staging = tmp_path / "s"
    staging.mkdir()
    np.save(staging / "color_image_export.npy", np.zeros((4, 8, 8), dtype=np.uint8))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with zipfile.ZipFile(data_dir / "yoda_dog_images_and_labels.zip", "w") as zf:
        zf.write(staging / "color_image_export.npy", arcname="color_image_export.npy")

    with pytest.raises(RuntimeError, match=r"expected \(N, H, W, 3\)"):
        BobbleheadDataset(data_dir=data_dir, extract_dir=tmp_path / "out", crop_size=8)


def test_frames_are_transformed_into_normalised_square_tensors(archive, tmp_path):
    """The default transform must match the one the backbones were trained under -- ImageNet
    statistics and a centre crop. Features extracted through a different pipeline are not
    comparable to anything else in the campaign."""
    data_dir, _, _ = archive
    dataset = BobbleheadDataset(data_dir=data_dir, extract_dir=tmp_path / "out", crop_size=16)

    image, _ = dataset[0]
    assert image.shape == (3, 16, 16) and image.dtype == torch.float32
    assert float(image.min()) < 0.0, "normalised, not raw [0,1]"


def test_a_custom_transform_replaces_the_default(archive, tmp_path):
    data_dir, _, _ = archive
    dataset = BobbleheadDataset(data_dir=data_dir, extract_dir=tmp_path / "out",
                                transform=lambda image: torch.tensor(image.size))
    torch.testing.assert_close(dataset[0][0], torch.tensor([32, 24]))


def test_the_convenience_constructor_passes_its_arguments_through(archive, tmp_path):
    data_dir, _, _ = archive
    dataset = get_bobblehead_dataset(data_dir=data_dir, crop_size=8,
                                     extract_dir=tmp_path / "out", max_frames=3)
    assert isinstance(dataset, BobbleheadDataset) and len(dataset) == 3


def test_the_eval_transform_is_deterministic():
    """No random augmentation: two passes over the same frame must give the same tensor, or the
    'manifold' being recovered includes the augmentation noise."""
    from PIL import Image

    transform = build_eval_transform(16)
    image = Image.fromarray(np.random.randint(0, 255, (24, 32, 3), dtype=np.uint8), mode="RGB")
    torch.testing.assert_close(transform(image), transform(image), rtol=0, atol=0)
