# Copyright (c) 2026 Pavel Snopov
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

EPS_MIN = 1.0
EPS_MAX = 20.0


def _numeric_key(value: str) -> tuple[int, str]:
    return (int(value), value) if value.isdigit() else (10**12, value)


class PhCDataset(Dataset):
    """Photonic crystal dataset loaded from an ESSL/SIB-CL ``.h5`` archive.

    Expected archive groups:
        unitcell/mpbepsimage/<id>: ``(32, 32)`` permittivity map
        unitcell/epsavg/<id>: scalar average permittivity
        mpbcal/DOS/<id>: density-of-states vector, optional for unlabeled data
    """

    FILES = {
        "blob": {
            "labeled": {
                "train": "sq-blob-v3-tm.h5",
                "test": "sq-blob-test2k-tm.h5",
            },
            "unlabeled": {
                "train": "sq-blob-ul-15k-tm.h5",
            },
        },
        "gpm": {
            "labeled": {
                "train": "mf1-sg3-tm.h5",
            },
            "unlabeled": {
                "train": "mf1-sg3-25k-ul-s99-tm.h5",
            },
        },
    }

    def __init__(
        self,
        path: str | Path,
        dataset: str,
        split: str,
        target_mode: str = "unlabeled",
        train_size: Optional[int] = None,
        valid_size: int = 0,
        test_size: Optional[int] = None,
        ldos: int = 0,
        udos: int = 400,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        path = Path(path)
        if path.suffix not in {".h5", ".hdf5"}:
            raise ValueError(f"PhCDataset expects an HDF5 file, got {path}.")
        if target_mode not in {"labeled", "unlabeled"}:
            raise ValueError(f"Unknown PhC target_mode {target_mode!r}. Choose 'labeled' or 'unlabeled'.")
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Unknown PhC split {split!r}. Choose 'train', 'valid', or 'test'.")
        if not 0 <= ldos < udos:
            raise ValueError(f"Expected 0 <= ldos < udos, got ldos={ldos}, udos={udos}.")

        with h5py.File(path, "r") as handle:
            image_group = handle["unitcell/mpbepsimage"]
            sample_ids = sorted(image_group.keys(), key=_numeric_key)
            sample_ids = self._select_sample_ids(
                sample_ids=sample_ids,
                dataset=dataset,
                split=split,
                train_size=train_size,
                valid_size=valid_size,
                test_size=test_size,
            )

            dos_group = handle.get("mpbcal/DOS")
            epsavg_group = handle["unitcell/epsavg"]
            images = []
            dos = []

            for sample_id in sample_ids:
                image = image_group[sample_id][()].astype(np.float32)
                if image.max() > 1.0 or image.min() < 0.0:
                    image = (image - EPS_MIN) / (EPS_MAX - EPS_MIN)
                if image.min() < 0.0 or image.max() > 1.0:
                    raise ValueError("PhC images must be normalized to [0, 1] or raw permittivity values in [1, 20].")
                images.append(image)

                if dos_group is not None and sample_id in dos_group:
                    epsavg = np.asarray(epsavg_group[sample_id][()], dtype=np.float32)
                    dos.append((dos_group[sample_id][()][ldos:udos] / np.sqrt(epsavg)).astype(np.float32))
                elif target_mode == "labeled":
                    raise KeyError(f"PhC file {path} does not contain DOS labels for sample {sample_id}.")
                else:
                    dos.append(np.zeros(udos - ldos, dtype=np.float32))

        images = np.stack(images).astype(np.float32)
        dos = np.stack(dos).astype(np.float32)

        self.images = torch.from_numpy(images).unsqueeze(1)
        self.dos = torch.from_numpy(dos)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.dos[index]

    @staticmethod
    def _select_sample_ids(
        sample_ids: list[str],
        dataset: str,
        split: str,
        train_size: Optional[int],
        valid_size: int,
        test_size: Optional[int],
    ) -> list[str]:
        if train_size is None:
            if split == "train" or dataset == "blob" and split == "test":
                return sample_ids if test_size is None else sample_ids[:test_size]
            raise ValueError(f"PhC split {split!r} requires train_size to reproduce E-SSL index ranges.")

        start = 1
        if split == "train":
            stop = start + train_size
        elif split == "valid":
            start += train_size
            stop = start + valid_size
        elif dataset == "blob":
            stop = start + (test_size if test_size is not None else len(sample_ids))
        else:
            start += train_size + valid_size
            stop = start + (test_size if test_size is not None else len(sample_ids))

        selected_ids = [str(index) for index in range(start, stop)]
        if not selected_ids:
            raise ValueError(f"Selected empty PhC split {split!r}.")
        return selected_ids


class PhCDataModule(LightningDataModule):
    """PhC SSL datamodule following E-SSL dataset and target-mode routing."""

    def __init__(
        self,
        dataset: str,
        target_mode: str,
        data_dir: str,
        train_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        eval_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        train_split: str = "train",
        val_split: Optional[str] = None,
        test_split: Optional[str] = None,
        batch_size: int = 512,
        val_batch_size: Optional[int] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: Optional[bool] = None,
        drop_last: bool = True,
        train_size: Optional[int] = None,
        valid_size: int = 0,
        test_size: Optional[int] = None,
        ldos: int = 0,
        udos: int = 400,
    ) -> None:
        super().__init__()

        if dataset not in {"blob", "gpm"}:
            raise ValueError(f"Unknown PhC dataset {dataset!r}. Choose 'blob' or 'gpm'.")
        if target_mode not in {"labeled", "unlabeled"}:
            raise ValueError(f"Unknown PhC target_mode {target_mode!r}. Choose 'labeled' or 'unlabeled'.")
        for split in (train_split, val_split, test_split):
            if split is not None and split not in {"train", "valid", "test"}:
                raise ValueError(f"Unknown PhC split {split!r}. Choose 'train', 'valid', or 'test'.")

        val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        persistent_workers = num_workers > 0 if persistent_workers is None else persistent_workers

        self.save_hyperparameters(ignore=["train_transform", "eval_transform"], logger=False)

        self.dataset = dataset
        self.target_mode = target_mode
        self.data_dir = Path(data_dir)
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.drop_last = drop_last
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.ldos = ldos
        self.udos = udos

        self.batch_size_per_device = batch_size
        self.val_batch_size_per_device = val_batch_size

        self.data_train: Optional[Dataset[Any]] = None
        self.data_val: Optional[Dataset[Any]] = None
        self.data_test: Optional[Dataset[Any]] = None

    @property
    def target_dim(self) -> int:
        return self.udos - self.ldos

    def prepare_data(self) -> None:
        splits = [self.train_split]
        if self.val_split is not None:
            splits.append(self.val_split)
        if self.test_split is not None:
            splits.append(self.test_split)

        required_paths = []
        for split in splits:
            path = self._resolve_path(split)
            if path not in required_paths:
                required_paths.append(path)

        if all(path.exists() for path in required_paths):
            return

        from scripts.download_phc import download_phc

        download_phc(output_dir=self.data_dir)
        missing_paths = [path for path in required_paths if not path.exists()]
        if missing_paths:
            missing = ", ".join(str(path) for path in missing_paths)
            raise FileNotFoundError(f"PhC dataset archive(s) do not exist: {missing}")

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            world_size = self.trainer.world_size
            if self.batch_size % world_size != 0:
                raise RuntimeError(f"Batch size ({self.batch_size}) is not divisible by the number of devices ({world_size}).")
            if self.val_batch_size % world_size != 0:
                raise RuntimeError(
                    f"Validation batch size ({self.val_batch_size}) is not divisible by the number of devices ({world_size})."
                )
            self.batch_size_per_device = self.batch_size // world_size
            self.val_batch_size_per_device = self.val_batch_size // world_size

        if self.data_train is None:
            self.data_train = PhCDataset(
                path=self._resolve_path(self.train_split),
                dataset=self.dataset,
                split=self.train_split,
                target_mode=self.target_mode,
                train_size=self.train_size,
                valid_size=self.valid_size,
                test_size=self.test_size,
                ldos=self.ldos,
                udos=self.udos,
                transform=self.train_transform,
            )
        if self.val_split is not None and self.data_val is None:
            self.data_val = PhCDataset(
                path=self._resolve_path(self.val_split),
                dataset=self.dataset,
                split=self.val_split,
                target_mode=self.target_mode,
                train_size=self.train_size,
                valid_size=self.valid_size,
                test_size=self.test_size,
                ldos=self.ldos,
                udos=self.udos,
                transform=self.eval_transform,
            )
        if self.test_split is not None and self.data_test is None:
            self.data_test = PhCDataset(
                path=self._resolve_path(self.test_split),
                dataset=self.dataset,
                split=self.test_split,
                target_mode=self.target_mode,
                train_size=self.train_size,
                valid_size=self.valid_size,
                test_size=self.test_size,
                ldos=self.ldos,
                udos=self.udos,
                transform=self.eval_transform,
            )

    def train_dataloader(self) -> DataLoader[Any]:
        if self.data_train is None:
            raise RuntimeError("setup() must create data_train before train_dataloader() is called.")
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            shuffle=True,
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

    def _resolve_path(self, split: str) -> Path:
        file_split = "train" if split == "valid" else split
        try:
            filename = PhCDataset.FILES[self.dataset][self.target_mode][file_split]
        except KeyError as exc:
            raise ValueError(
                f"No PhC file is defined for dataset={self.dataset!r}, target_mode={self.target_mode!r}, split={split!r}."
            ) from exc
        return self.data_dir / filename


if __name__ == "__main__":
    _ = PhCDataModule(dataset="blob", target_mode="unlabeled", data_dir="data/phc", train_transform=lambda x: x)
    print("PhCDataModule successfully initialized.")
