# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Transfer (out-of-domain) linear-probing datasets — SPHERE-JEPA Table 1.

Table 1 of SPHERE-JEPA (Nicollier et al., 2026) reports frozen-backbone linear
probing accuracy of ImageNet-1K-pretrained encoders on seven downstream
classification datasets: DTD, FGVC-Aircraft, CIFAR-10, CIFAR-100, Flowers-102,
Food-101 and Oxford-IIIT Pets, plus their mean. Every dataset is wrapped by
torchvision, so this module only fixes the *split convention* (the standard
"train on everything that is not the test split, report on the test split"
used by the SimCLR / BYOL / DINO transfer benchmarks) and applies the same
deterministic eval transform as the in-domain probe.

Standard splits (torchvision names):

  ======== ===================== ========== ======= ======================
  key      dataset               train      test    classes
  ======== ===================== ========== ======= ======================
  dtd      DTD (partition 1)     train+val  test    47
  aircraft FGVC-Aircraft variant trainval   test    100
  cifar10  CIFAR-10              train      test    10
  cifar100 CIFAR-100             train      test    100
  flowers  Flowers-102           train+val  test    102
  food     Food-101              train      test    101
  pets     Oxford-IIIT Pets      trainval   test    37
  ======== ===================== ========== ======= ======================
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from torch.utils.data import ConcatDataset, Dataset
from torchvision import datasets as tvd

from src.eval.indomain import build_eval_transform

# key -> (display name, number of classes)
TRANSFER_DATASETS: "OrderedDict[str, tuple[str, int]]" = OrderedDict(
    [
        ("dtd", ("DTD", 47)),
        ("aircraft", ("Aircraft", 100)),
        ("cifar10", ("CIFAR10", 10)),
        ("cifar100", ("CIFAR100", 100)),
        ("flowers", ("Flowers", 102)),
        ("food", ("Food", 101)),
        ("pets", ("Pets", 37)),
    ]
)


def _targets(dataset: Dataset) -> list[int]:
    """Integer labels of a torchvision classification dataset without decoding images."""
    for attr in ("targets", "_labels", "labels"):
        if hasattr(dataset, attr):
            return [int(t) for t in getattr(dataset, attr)]
    if isinstance(dataset, ConcatDataset):
        out: list[int] = []
        for part in dataset.datasets:
            out.extend(_targets(part))
        return out
    return [int(dataset[i][1]) for i in range(len(dataset))]


def get_transfer_splits(
    key: str,
    root: str,
    crop_size: int,
    download: bool = False,
) -> tuple[Dataset, Dataset, int]:
    """Return ``(train_split, test_split, num_classes)`` for one Table 1 dataset.

    Args:
        key: one of :data:`TRANSFER_DATASETS`.
        root: directory holding (or receiving) the torchvision downloads.
        crop_size: resolution the backbone was pretrained at (eval transform).
        download: fetch the archives if missing (needs internet: login node).
    """
    if key not in TRANSFER_DATASETS:
        raise ValueError(f"Unknown transfer dataset {key!r}; choose one of {list(TRANSFER_DATASETS)}.")
    transform = build_eval_transform(crop_size)
    root = str(Path(root))
    common = {"root": root, "transform": transform, "download": download}

    if key == "dtd":
        train = ConcatDataset([
            tvd.DTD(split="train", partition=1, **common),
            tvd.DTD(split="val", partition=1, **common),
        ])
        test = tvd.DTD(split="test", partition=1, **common)
    elif key == "aircraft":
        train = tvd.FGVCAircraft(split="trainval", annotation_level="variant", **common)
        test = tvd.FGVCAircraft(split="test", annotation_level="variant", **common)
    elif key == "cifar10":
        train = tvd.CIFAR10(train=True, **common)
        test = tvd.CIFAR10(train=False, **common)
    elif key == "cifar100":
        train = tvd.CIFAR100(train=True, **common)
        test = tvd.CIFAR100(train=False, **common)
    elif key == "flowers":
        train = ConcatDataset([tvd.Flowers102(split="train", **common), tvd.Flowers102(split="val", **common)])
        test = tvd.Flowers102(split="test", **common)
    elif key == "food":
        train = tvd.Food101(split="train", **common)
        test = tvd.Food101(split="test", **common)
    elif key == "pets":
        train = tvd.OxfordIIITPet(split="trainval", target_types="category", **common)
        test = tvd.OxfordIIITPet(split="test", target_types="category", **common)
    else:  # pragma: no cover - guarded above
        raise AssertionError(key)

    num_classes = TRANSFER_DATASETS[key][1]
    observed = max(max(_targets(train)), max(_targets(test))) + 1
    if observed != num_classes:
        raise RuntimeError(f"{key}: expected {num_classes} classes, labels span {observed}.")
    return train, test, num_classes
