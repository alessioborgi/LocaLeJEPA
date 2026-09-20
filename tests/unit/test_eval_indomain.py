"""In-domain split construction and the evaluation transform.

The transform here must match ``configs/data/transform/image/lejepa_eval.yaml`` exactly: if the
two drifted, frozen features would be extracted through a different pipeline than the online
probe saw during training, and the two numbers would stop being comparable without anything
failing.
"""

from __future__ import annotations

import pytest
import torch
from torchvision import transforms

from src.eval.indomain import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_eval_transform,
    extract_classification_features,
    get_indomain_splits,
    subsample_per_class,
)


# --------------------------------------------------------------------------------------
# the evaluation transform
# --------------------------------------------------------------------------------------

def test_eval_transform_is_resize_centrecrop_totensor_normalize():
    """Deterministic and in this order. Any random augmentation here would make feature
    extraction non-reproducible; a missing normalisation would silently shift every feature."""
    pipeline = build_eval_transform(96).transforms
    assert [type(t) for t in pipeline] == [
        transforms.Resize, transforms.CenterCrop, transforms.ToTensor, transforms.Normalize
    ]
    assert pipeline[0].size == 96
    assert pipeline[1].size == (96, 96)


def test_eval_transform_uses_the_imagenet_statistics():
    """These must equal the values in lejepa_eval.yaml. Pinned literally, because a drift here
    is invisible -- features would simply be offset."""
    assert IMAGENET_MEAN == [0.485, 0.456, 0.406]
    assert IMAGENET_STD == [0.229, 0.224, 0.225]
    normalize = build_eval_transform(64).transforms[-1]
    assert list(normalize.mean) == IMAGENET_MEAN and list(normalize.std) == IMAGENET_STD


def test_eval_transform_output_shape_and_determinism(imagefolder):
    from PIL import Image

    image = Image.open(next(imagefolder.glob("*/*.JPEG")))
    transform = build_eval_transform(32)
    first, second = transform(image), transform(image)
    assert first.shape == (3, 32, 32)
    torch.testing.assert_close(first, second, rtol=0, atol=0), "must be deterministic"


# --------------------------------------------------------------------------------------
# split construction
# --------------------------------------------------------------------------------------

def test_imagenette_splits(imagenette_tree):
    train, val, num_classes = get_indomain_splits(
        "imagenette", str(imagenette_tree), crop_size=16, imagenette_variant="160px"
    )
    assert num_classes == 3 and len(train) == 12 and len(val) == 12


def test_imagenet_splits(imagenet_tree):
    train, val, num_classes = get_indomain_splits("imagenet1k", str(imagenet_tree), crop_size=16)
    assert num_classes == 3 and len(train) == 12


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown in-domain dataset"):
        get_indomain_splits("cifar10", "/nonexistent", crop_size=16)


def test_unknown_imagenette_variant_raises_before_touching_disk():
    """Validation happens before any path is opened, so a typo fails immediately rather than
    after a confusing FileNotFoundError."""
    with pytest.raises(ValueError, match="Unknown Imagenette variant"):
        get_indomain_splits("imagenette", "/nonexistent", crop_size=16,
                            imagenette_variant="512px")


def test_mismatched_class_mappings_raise(tmp_path):
    """THE guard that matters here. If train and val disagree on class_to_idx, every reported
    accuracy is computed against permuted labels -- a silently wrong number, not a crash.
    """
    from conftest import write_imagefolder

    base = tmp_path / "imagenette" / "imagenette2-160"
    write_imagefolder(base / "train", classes=3, per_class=2)
    write_imagefolder(base / "val", classes=3, per_class=2)
    (base / "val" / "n002").rename(base / "val" / "zzz_different_name")

    with pytest.raises(RuntimeError, match="class-to-index mapping"):
        get_indomain_splits("imagenette", str(tmp_path / "imagenette"), crop_size=16)


# --------------------------------------------------------------------------------------
# subsampling
# --------------------------------------------------------------------------------------

def test_subsample_caps_each_class(imagefolder):
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(str(imagefolder), transform=build_eval_transform(16))
    subset = subsample_per_class(dataset, per_class=2, seed=0)
    assert len(subset) == 6, "3 classes x 2"

    labels = [dataset.samples[i][1] for i in subset.indices]
    assert sorted(labels) == [0, 0, 1, 1, 2, 2]


def test_subsample_is_deterministic_for_a_given_seed(imagefolder):
    """The k-NN bank is built from this subset, so a different draw is a different reported
    number. Reproducibility here is the whole point of the seed argument."""
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(str(imagefolder), transform=build_eval_transform(16))
    a = subsample_per_class(dataset, per_class=2, seed=0).indices
    b = subsample_per_class(dataset, per_class=2, seed=0).indices
    c = subsample_per_class(dataset, per_class=2, seed=1).indices
    assert a == b
    assert a != c, "and a different seed must actually draw differently"


def test_subsample_indices_are_sorted(imagefolder):
    """Sorted indices preserve the dataset's original order, which keeps the cached feature
    matrix aligned with anything else indexed the same way."""
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(str(imagefolder), transform=build_eval_transform(16))
    indices = subsample_per_class(dataset, per_class=3, seed=0).indices
    assert indices == sorted(indices)


def test_subsample_keeps_everything_when_the_cap_exceeds_the_class_size(imagefolder):
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(str(imagefolder), transform=build_eval_transform(16))
    assert len(subsample_per_class(dataset, per_class=999, seed=0)) == len(dataset)


# --------------------------------------------------------------------------------------
# feature extraction and its cache
# --------------------------------------------------------------------------------------

def test_features_are_stored_as_float16_with_int64_labels(imagefolder):
    """fp16 halves a 1.28M x 512 bank to ~1.3 GB. The dtype is part of the on-disk contract."""
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(str(imagefolder), transform=build_eval_transform(8))
    feats, labels = extract_classification_features(
        lambda images: images.flatten(1)[:, :5], dataset, batch_size=4, num_workers=0
    )
    assert feats.dtype == torch.float16 and labels.dtype == torch.int64
    assert feats.shape == (12, 5) and labels.shape == (12,)


def test_the_cache_round_trips(imagefolder, tmp_path):
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(str(imagefolder), transform=build_eval_transform(8))
    cache = tmp_path / "cache" / "feat.pt"
    calls = []

    def extract(images):
        calls.append(1)
        return images.flatten(1)[:, :5]

    first = extract_classification_features(extract, dataset, batch_size=4, num_workers=0,
                                            cache_path=str(cache))
    assert cache.exists()

    calls.clear()
    second = extract_classification_features(extract, dataset, batch_size=4, num_workers=0,
                                             cache_path=str(cache))
    assert calls == [], "a present cache must short-circuit extraction entirely"
    torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)


def test_labels_follow_the_dataset_order_not_the_loader(imagefolder):
    """``shuffle=False`` in the loader, so cached labels must equal ImageFolder's targets in
    order -- which is what lets a cached bank be compared against a freshly built one."""
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(str(imagefolder), transform=build_eval_transform(8))
    _, labels = extract_classification_features(
        lambda images: images.flatten(1)[:, :3], dataset, batch_size=5, num_workers=0
    )
    assert labels.tolist() == [t for _, t in dataset.samples]
