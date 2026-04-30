# Copyright (c) 2026 Pavel Snopov
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

# TODO: generate synthetic PhC data via a separate script and save as .npz


class PhCDataset(Dataset):
    """Photonic crystal dataset loaded from a ``.npz`` archive.

    Expected archive keys:
        images: ``(N, 32, 32)`` permittivity maps
        dos: ``(N, 400)`` density-of-states vectors
    """

    def __init__(
        self,
        path: str | Path,
        split: str = "train",
        train_frac: float = 0.6,
        seed: int = 42,
        eps_min: float = 1.0,
        eps_max: float = 20.0,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        data = np.load(path)
        images = data["images"].astype(np.float32)
        dos = data["dos"].astype(np.float32)

        if images.max() > 1.0 or images.min() < 0.0:
            images = (images - eps_min) / (eps_max - eps_min)
        if images.min() < 0.0 or images.max() > 1.0:
            raise ValueError("PhC images must be normalized to [0, 1] or raw permittivity values in [eps_min, eps_max].")

        if split not in {"train", "test"}:
            raise ValueError(f"Unknown PhC split {split!r}. Choose 'train' or 'test'.")

        n_samples = len(images)
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(n_samples)
        cut = int(n_samples * train_frac)
        indices = permutation[:cut] if split == "train" else permutation[cut:]

        self.images = torch.from_numpy(images[indices]).unsqueeze(1)
        self.dos = torch.from_numpy(dos[indices])
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.dos[index]


class PhCDataModule(LightningDataModule):
    """PhC datamodule with Hydra-configured tensor transforms and optional collator.

    LeJEPA configs pass a multi-view DOS-invariant tensor transform. Future
    masked-prediction variants can override the transform and collator from
    Hydra without changing this datamodule.
    """

    def __init__(
        self,
        path: str,
        train_transform: Callable[[torch.Tensor], torch.Tensor],
        eval_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        train_split: str = "train",
        val_split: Optional[str] = None,
        test_split: Optional[str] = None,
        collator: Optional[Callable[[Any], Any]] = None,
        batch_size: int = 512,
        val_batch_size: Optional[int] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: Optional[bool] = None,
        drop_last: bool = True,
        train_frac: float = 0.6,
        seed: int = 42,
        eps_min: float = 1.0,
        eps_max: float = 20.0,
        crop_size: int = 32,
    ) -> None:
        super().__init__()

        val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        test_split = test_split if test_split is not None else val_split
        persistent_workers = num_workers > 0 if persistent_workers is None else persistent_workers

        for split in (train_split, val_split, test_split):
            if split is not None and split not in {"train", "test"}:
                raise ValueError(f"Unknown PhC split {split!r}. Choose 'train' or 'test'.")

        self.save_hyperparameters(ignore=["train_transform", "eval_transform", "collator"], logger=False)

        self.path = Path(path)
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.collator = collator
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.drop_last = drop_last
        self.train_frac = train_frac
        self.seed = seed
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.crop_size = crop_size

        self.batch_size_per_device = batch_size
        self.val_batch_size_per_device = val_batch_size

        self.data_train: Optional[Dataset[Any]] = None
        self.data_val: Optional[Dataset[Any]] = None
        self.data_test: Optional[Dataset[Any]] = None

    @property
    def target_dim(self) -> int:
        return 400

    def prepare_data(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"PhC dataset archive does not exist: {self.path}")

    def setup(self, stage: Optional[str] = None) -> None:
        eval_transform = self.eval_transform

        if self.trainer is not None:
            world_size = self.trainer.world_size
            if self.batch_size % world_size != 0:
                raise RuntimeError(f"Batch size ({self.batch_size}) is not divisible by the number of devices ({world_size}).")
            if self.val_split is not None and self.eval_transform is not None and self.val_batch_size % world_size != 0:
                raise RuntimeError(
                    f"Validation batch size ({self.val_batch_size}) is not divisible by the number of devices ({world_size})."
                )
            self.batch_size_per_device = self.batch_size // world_size
            self.val_batch_size_per_device = self.val_batch_size // world_size

        if self.data_train is None:
            self.data_train = PhCDataset(
                path=self.path,
                split=self.train_split,
                train_frac=self.train_frac,
                seed=self.seed,
                eps_min=self.eps_min,
                eps_max=self.eps_max,
                transform=self.train_transform,
            )
        if self.val_split is not None and eval_transform is not None and self.data_val is None:
            self.data_val = PhCDataset(
                path=self.path,
                split=self.val_split,
                train_frac=self.train_frac,
                seed=self.seed,
                eps_min=self.eps_min,
                eps_max=self.eps_max,
                transform=eval_transform,
            )
        if self.test_split is not None and eval_transform is not None and self.data_test is None:
            self.data_test = PhCDataset(
                path=self.path,
                split=self.test_split,
                train_frac=self.train_frac,
                seed=self.seed,
                eps_min=self.eps_min,
                eps_max=self.eps_max,
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
    _ = PhCDataModule(path="data/phc.npz", train_transform=lambda x: x)
    print("PhCDataModule successfully initialized.")
