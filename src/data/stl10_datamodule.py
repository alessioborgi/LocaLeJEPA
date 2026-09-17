# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import STL10


class STL10DataModule(LightningDataModule):
    """STL10 datamodule with Hydra-configured transforms and optional collator.

    LeJEPA configs use multi-view transforms and eval data. I-JEPA configs use a
    single-image transform plus mask collator and can omit validation/test.
    """

    def __init__(
        self,
        train_transform: Callable[[Any], Any],
        data_dir: str = "data",
        train_split: str = "unlabeled",
        val_split: Optional[str] = None,
        test_split: Optional[str] = None,
        eval_transform: Optional[Callable[[Any], Any]] = None,
        collator: Optional[Callable[[Any], Any]] = None,
        batch_size: int = 256,
        val_batch_size: Optional[int] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: Optional[bool] = None,
        drop_last: bool = True,
        crop_size: int = 224,
    ) -> None:
        super().__init__()

        val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        test_split = test_split if test_split is not None else val_split
        persistent_workers = num_workers > 0 if persistent_workers is None else persistent_workers

        for split in (train_split, val_split, test_split):
            if split is not None and split not in {"train", "test", "unlabeled", "train+unlabeled"}:
                raise ValueError(f"Unknown STL10 split {split!r}.")

        self.save_hyperparameters(ignore=["train_transform", "eval_transform", "collator"], logger=False)

        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.collator = collator
        self.data_dir = Path(data_dir)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.drop_last = drop_last
        self.crop_size = crop_size

        self.batch_size_per_device = batch_size
        self.val_batch_size_per_device = val_batch_size

        self.data_train: Optional[Dataset[Any]] = None
        self.data_val: Optional[Dataset[Any]] = None
        self.data_test: Optional[Dataset[Any]] = None

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self) -> None:
        STL10(root=str(self.data_dir), split=self.train_split, download=True)
        if self.val_split is not None and self.eval_transform is not None:
            STL10(root=str(self.data_dir), split=self.val_split, download=True)
        if self.test_split is not None and self.eval_transform is not None and self.test_split != self.val_split:
            STL10(root=str(self.data_dir), split=self.test_split, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        eval_transform = self.eval_transform

        if self.trainer is not None:
            world_size = self.trainer.world_size
            if self.batch_size % world_size != 0:
                raise RuntimeError(f"Batch size ({self.batch_size}) is not divisible by the number of devices ({world_size}).")
            if self.eval_transform is not None and self.val_split is not None and self.val_batch_size % world_size != 0:
                raise RuntimeError(
                    f"Validation batch size ({self.val_batch_size}) is not divisible by the number of devices ({world_size})."
                )
            self.batch_size_per_device = self.batch_size // world_size
            self.val_batch_size_per_device = self.val_batch_size // world_size

        if self.data_train is None:
            self.data_train = STL10(
                root=str(self.data_dir),
                split=self.train_split,
                download=False,
                transform=self.train_transform,
            )
        if self.val_split is not None and eval_transform is not None and self.data_val is None:
            self.data_val = STL10(
                root=str(self.data_dir),
                split=self.val_split,
                download=False,
                transform=eval_transform,
            )
        if self.test_split is not None and eval_transform is not None and self.data_test is None:
            self.data_test = STL10(
                root=str(self.data_dir),
                split=self.test_split,
                download=False,
                transform=eval_transform,
            )

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

    def val_dataloader(self) -> Optional[DataLoader[Any]]:
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

    def test_dataloader(self) -> Optional[DataLoader[Any]]:
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


if __name__ == "__main__":
    _ = STL10DataModule(train_transform=lambda x: x, eval_transform=lambda x: x, val_split="test")
    print("STL10DataModule successfully initialized.")
