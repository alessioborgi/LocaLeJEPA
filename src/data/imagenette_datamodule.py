# Copyright (c) 2026 Pavel Snopov
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, Callable, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive

IMAGENETTE_VARIANTS = {
    "full_size": ("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz", "imagenette2"),
    "320px": ("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz", "imagenette2-320"),
    "160px": ("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz", "imagenette2-160"),
}


class ImagenetteDataset(Dataset):
    """Imagenette loaded from the official fastai archive.

    Shared image-folder dataset for LeJEPA and I-JEPA datamodules.
    """

    def __init__(self, root: str | Path, split: str, transform: Callable[[Any], Any]):
        self.dataset = ImageFolder(root=Path(root) / _split_dir(split), transform=transform)

    def __getitem__(self, index: int):
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)


class ImagenetteDataModule(LightningDataModule):
    """Imagenette datamodule with Hydra-configured transforms and optional collator.

    LeJEPA configs pass multi-view train/eval transforms. I-JEPA configs pass a
    single-image train transform and a mask collator, with validation/test left off.
    """

    def __init__(
        self,
        train_transform: Callable[[Any], Any],
        data_dir: str = "data/imagenette",
        variant: str = "160px",
        train_split: str = "train",
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

        self.save_hyperparameters(ignore=["train_transform", "eval_transform", "collator"], logger=False)

        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.collator = collator
        self.data_dir = Path(data_dir)
        self.variant = variant
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
        """Download/cache Imagenette once before distributed setup."""
        url, root_name = self._variant_info()
        if not self._dataset_root.exists():
            download_and_extract_archive(url, download_root=str(self.data_dir), filename=f"{root_name}.tgz")

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
            self.data_train = ImagenetteDataset(self._dataset_root, self.train_split, transform=self.train_transform)
        if self.val_split is not None and eval_transform is not None and self.data_val is None:
            self.data_val = ImagenetteDataset(self._dataset_root, self.val_split, transform=eval_transform)
        if self.test_split is not None and eval_transform is not None and self.data_test is None:
            self.data_test = ImagenetteDataset(self._dataset_root, self.test_split, transform=eval_transform)

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

    @property
    def _dataset_root(self) -> Path:
        _, root_name = self._variant_info()
        return self.data_dir / root_name

    def _variant_info(self) -> tuple[str, str]:
        try:
            return IMAGENETTE_VARIANTS[self.variant]
        except KeyError as error:
            options = ", ".join(sorted(IMAGENETTE_VARIANTS))
            raise ValueError(f"Unknown Imagenette variant {self.variant!r}. Choose one of: {options}.") from error


def _split_dir(split: str) -> str:
    if split == "validation":
        return "val"
    if split in {"train", "val"}:
        return split
    raise ValueError(f"Unknown Imagenette split {split!r}. Choose 'train', 'val', or 'validation'.")


if __name__ == "__main__":
    _ = ImagenetteDataModule(train_transform=lambda x: x, eval_transform=lambda x: x, val_split="validation")
    print("ImagenetteDataModule successfully initialized.")
