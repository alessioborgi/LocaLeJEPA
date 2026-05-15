# Copyright (c) 2026 Pavel Snopov
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder


class ImageNet(ImageFolder):
    """ImageNet ImageFolder wrapper with optional local copy support."""

    def __init__(
        self,
        root: str | Path,
        image_folder: str = "imagenet_full_size/061417/",
        tar_file: str = "imagenet_full_size-061417.tar.gz",
        transform: Callable[[Any], Any] | None = None,
        train: bool = True,
        job_id: str | None = None,
        local_rank: int | None = None,
        copy_data: bool = True,
        index_targets: bool = False,
    ) -> None:
        suffix = "train/" if train else "val/"
        data_path = None
        root = str(root)

        if copy_data:
            raise NotImplementedError
        """
            data_path = copy_imgnt_locally( # TODO: implement copy_imgnt_locally()
                root=root,
                suffix=suffix,
                image_folder=image_folder,
                tar_file=tar_file,
                job_id=job_id,
                local_rank=local_rank,
            )
        """
        if (not copy_data) or (data_path is None):
            data_path = os.path.join(root, image_folder, suffix)

        super().__init__(root=data_path, transform=transform)

        if index_targets:
            self.targets = []
            for sample in self.samples:
                self.targets.append(sample[1])
            self.targets = np.array(self.targets)
            self.samples = np.array(self.samples)

            self.target_indices = []
            for target in range(len(self.classes)):
                indices = np.squeeze(np.argwhere(self.targets == target)).tolist()
                self.target_indices.append(indices)


class ImageNet1KDataModule(LightningDataModule):
    """ImageNet datamodule with Hydra-configured transforms and optional collator.

    LeJEPA configs use multi-view train/eval transforms. I-JEPA configs use a
    single-image train transform plus mask collator and omit validation/test.
    """

    def __init__(
        self,
        train_transform: Callable[[Any], Any],
        data_dir: str,
        eval_transform: Callable[[Any], Any] | None = None,
        collator: Callable[[Any], Any] | None = None,
        image_folder: str = "imagenet_full_size/061417/",
        tar_file: str = "imagenet_full_size-061417.tar.gz",
        batch_size: int = 256,
        val_batch_size: int | None = None,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool | None = False,
        drop_last: bool = True,
        copy_data: bool = False,
        job_id: str | None = None,
        local_rank: int | None = None,
        class_subset_file: str | None = None,
        crop_size: int = 224,
    ) -> None:
        super().__init__()

        val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        persistent_workers = num_workers > 0 if persistent_workers is None else persistent_workers

        self.save_hyperparameters(ignore=["train_transform", "eval_transform", "collator"], logger=False)

        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.collator = collator
        self.data_dir = Path(data_dir)
        self.image_folder = image_folder
        self.tar_file = tar_file
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.drop_last = drop_last
        self.copy_data = copy_data
        self.job_id = job_id
        self.local_rank = local_rank
        self.class_subset_file = class_subset_file
        self.crop_size = crop_size

        self.batch_size_per_device = batch_size
        self.val_batch_size_per_device = val_batch_size

        self.data_train: Dataset[Any] | None = None
        self.data_val: Dataset[Any] | None = None
        self.data_test: Dataset[Any] | None = None

    @property
    def num_classes(self) -> int:
        if self.class_subset_file is not None and Path(self.class_subset_file).exists():
            with Path(self.class_subset_file).open("r", encoding="utf-8") as file:
                return sum(1 for line in file if line.strip())
        return 1000

    def setup(self, stage: str | None = None) -> None:
        if self.trainer is not None:
            world_size = self.trainer.world_size
            if self.batch_size % world_size != 0:
                raise RuntimeError(f"Batch size ({self.batch_size}) is not divisible by the number of devices ({world_size}).")
            if self.eval_transform is not None and self.val_batch_size % world_size != 0:
                raise RuntimeError(
                    f"Validation batch size ({self.val_batch_size}) is not divisible by the number of devices ({world_size})."
                )
            self.batch_size_per_device = self.batch_size // world_size
            self.val_batch_size_per_device = self.val_batch_size // world_size

        if self.data_train is None:
            self.data_train = self._build_dataset(train=True, transform=self.train_transform)
        if self.eval_transform is not None and self.data_val is None:
            self.data_val = self._build_dataset(train=False, transform=self.eval_transform)
        if self.eval_transform is not None and self.data_test is None:
            self.data_test = self._build_dataset(train=False, transform=self.eval_transform)

    def train_dataloader(self) -> DataLoader[Any]:
        if self.data_train is None:
            raise RuntimeError("setup() must create data_train before train_dataloader() is called.")
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader[Any] | None:
        if self.data_val is None:
            return None
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.val_batch_size_per_device,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader[Any] | None:
        if self.data_test is None:
            return None
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.val_batch_size_per_device,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    def _build_dataset(self, train: bool, transform: Callable[[Any], Any] | None) -> Dataset[Any]:
        dataset: Dataset[Any] = ImageNet(
            root=self.data_dir,
            image_folder=self.image_folder,
            tar_file=self.tar_file,
            transform=transform,
            train=train,
            job_id=self.job_id,
            local_rank=self.local_rank,
            copy_data=self.copy_data if train else False,
            index_targets=False,
        )

        if self.class_subset_file is not None:
            dataset = _apply_class_subset(dataset, self.class_subset_file)
        return dataset


def _apply_class_subset(dataset: Dataset[Any], class_subset_file: str | Path) -> Dataset[Any]:
    """Filter an ImageFolder-style dataset to listed synset folders."""
    if not isinstance(dataset, ImageFolder):
        raise TypeError("class_subset_file expects an ImageFolder-style dataset.")

    with Path(class_subset_file).open("r", encoding="utf-8") as file:
        valid_classes = {line.strip() for line in file if line.strip()}

    sorted_valid = sorted(valid_classes)
    new_idx = {
        dataset.class_to_idx[class_name]: index for index, class_name in enumerate(sorted_valid) if class_name in dataset.class_to_idx
    }
    dataset.samples = [(path, new_idx[target]) for path, target in dataset.samples if target in new_idx]
    dataset.targets = [target for _, target in dataset.samples]
    return dataset


if __name__ == "__main__":
    _ = ImageNet1KDataModule(train_transform=lambda x: x, eval_transform=lambda x: x, data_dir="data")
    print("ImageNet1KDataModule successfully initialized.")
