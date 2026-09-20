# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""In-domain classification splits and frozen-feature extraction.

"In-domain" means the dataset the backbone was pretrained on: the train split
becomes the k-NN feature bank / linear-probe training set, and the validation
split is the reported set — the SPHERE-JEPA Table 3 setting (pretrain on
ImageNet-1K, evaluate on ImageNet-1K).

Both splits use the deterministic eval transform (resize + centre crop +
normalise), i.e. exactly the view the online probe sees in ``validation_step``.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.eval.io import atomic_torch_save, load_tensor_cache

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_eval_transform(crop_size: int) -> transforms.Compose:
    """Deterministic eval transform, identical to ``configs/.../lejepa_eval.yaml``."""
    return transforms.Compose(
        [
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_indomain_splits(
    dataset: str,
    data_dir: str,
    crop_size: int,
    image_folder: str = "imagenet_full_size/061417/",
    imagenette_variant: str = "160px",
) -> tuple[Dataset, Dataset, int]:
    """Return ``(train_split, val_split, num_classes)`` for a pretraining dataset.

    Args:
        dataset: ``"imagenet1k"`` or ``"imagenette"``.
        data_dir: Dataset root (``data/imagenet`` or ``data/imagenette``).
        crop_size: Resolution the backbone was pretrained at.
        image_folder: ImageNet sub-path holding ``train/`` and ``val/``.
        imagenette_variant: Imagenette release directory name suffix.
    """
    transform = build_eval_transform(crop_size)
    root = Path(data_dir)

    if dataset == "imagenet1k":
        base = root / image_folder
        train = ImageFolder(root=str(base / "train"), transform=transform)
        val = ImageFolder(root=str(base / "val"), transform=transform)
    elif dataset == "imagenette":
        variants = {"160px": "imagenette2-160", "320px": "imagenette2-320", "full": "imagenette2"}
        if imagenette_variant not in variants:
            raise ValueError(f"Unknown Imagenette variant {imagenette_variant!r}; choose one of {sorted(variants)}.")
        base = root / variants[imagenette_variant]
        train = ImageFolder(root=str(base / "train"), transform=transform)
        val = ImageFolder(root=str(base / "val"), transform=transform)
    else:
        raise ValueError(f"Unknown in-domain dataset {dataset!r}; choose 'imagenet1k' or 'imagenette'.")

    if train.class_to_idx != val.class_to_idx:
        raise RuntimeError("Train and val splits disagree on the class-to-index mapping.")

    return train, val, len(train.classes)


def subsample_per_class(dataset: ImageFolder, per_class: int, seed: int = 0) -> Subset:
    """Take at most ``per_class`` samples of each class (deterministic given seed).

    Used to shrink the k-NN feature bank when a full-train extraction is too
    expensive. This *changes* the protocol relative to Table 3, so callers must
    report it.
    """
    generator = torch.Generator().manual_seed(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, (_, target) in enumerate(dataset.samples):
        by_class[int(target)].append(index)

    kept: list[int] = []
    for target in sorted(by_class):
        indices = by_class[target]
        if len(indices) > per_class:
            picks = torch.randperm(len(indices), generator=generator)[:per_class]
            indices = [indices[int(i)] for i in picks]
        kept.extend(indices)
    kept.sort()
    return Subset(dataset, kept)


@torch.no_grad()
def extract_classification_features(
    extract_fn,
    dataset: Dataset,
    batch_size: int = 256,
    num_workers: int = 8,
    cache_path: str | None = None,
    log_every: int = 200,
    tag: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract ``(features, labels)`` for a classification dataset.

    Features are stored as float16 (halves the 1.28M x 512 ImageNet bank to
    ~1.3 GB) and labels as int64. If ``cache_path`` exists it is loaded instead.
    """
    blob = load_tensor_cache(cache_path)
    if blob is not None:
        logger.info("%s: loaded cached features %s from %s", tag, tuple(blob["X"].shape), cache_path)
        return blob["X"], blob["y"]

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    feats: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for step, (images, targets) in enumerate(loader):
        feats.append(extract_fn(images).half())
        labels.append(torch.as_tensor(targets, dtype=torch.long))
        if log_every and step % log_every == 0:
            logger.info("%s: %d/%d batches", tag, step, len(loader))

    features = torch.cat(feats, dim=0)
    targets = torch.cat(labels, dim=0)

    if cache_path is not None:
        atomic_torch_save({"X": features, "y": targets}, cache_path)
        logger.info("%s: cached features %s to %s", tag, tuple(features.shape), cache_path)

    return features, targets
