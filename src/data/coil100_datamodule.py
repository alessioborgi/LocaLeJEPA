import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.utils import download_and_extract_archive

COIL100_URL = "https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip"
FILENAME = re.compile(r"^obj(\d+)__(\d+)\.png$", re.IGNORECASE)
NUM_OBJECTS = 100
POSES_PER_OBJECT = 72


class COIL100Dataset(Dataset):
    """COIL-100 frames for one split, with per-frame object id and turntable angle."""

    def __init__(
        self,
        root: str | Path,
        transform: Callable[[Any], Any],
        split: str = "train",
        val_pose_stride: int = 6,
    ) -> None:
        if split not in {"train", "val", "all"}:
            raise ValueError(f"Unknown COIL-100 split {split!r}. Choose 'train', 'val' or 'all'.")

        records = []
        for path in sorted(Path(root).rglob("obj*.png")):
            match = FILENAME.match(path.name)
            if match is None:
                continue
            obj, degrees = int(match.group(1)), int(match.group(2))
            pose = degrees // 5
            held_out = pose % val_pose_stride == 0
            if split == "all" or (split == "val") == held_out:
                records.append((path, obj, degrees))

        if not records:
            raise FileNotFoundError(
                f"No COIL-100 frames under {root}. Expected files named obj<N>__<degrees>.png; see the module docstring for the layout."
            )

        self.transform = transform
        self.paths = [r[0] for r in records]
        self.objects = np.array([r[1] for r in records], dtype=np.int64)
        self.angles = np.array([math.radians(r[2]) for r in records], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image), int(self.objects[index]) - 1


class COIL100DataModule(LightningDataModule):
    """COIL-100 datamodule with Hydra-configured multi-view transforms.

    ``val_pose_stride`` holds out every n-th pose of every object, so the online
    probe has to identify an object from viewpoints it never saw.
    """

    def __init__(
        self,
        train_transform: Callable[[Any], Any],
        data_dir: str = "data/coil-100",
        eval_transform: Callable[[Any], Any] | None = None,
        val_pose_stride: int = 6,
        batch_size: int = 256,
        val_batch_size: int | None = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool | None = None,
        drop_last: bool = True,
        crop_size: int = 128,
    ) -> None:
        super().__init__()

        val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        persistent_workers = num_workers > 0 if persistent_workers is None else persistent_workers

        self.save_hyperparameters(ignore=["train_transform", "eval_transform"], logger=False)

        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.data_dir = Path(data_dir)
        self.val_pose_stride = val_pose_stride
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.drop_last = drop_last
        self.crop_size = crop_size

        self.batch_size_per_device = batch_size
        self.val_batch_size_per_device = val_batch_size
        self.data_train: Dataset[Any] | None = None
        self.data_val: Dataset[Any] | None = None

    @property
    def num_classes(self) -> int:
        return NUM_OBJECTS

    def prepare_data(self) -> None:
        if any(self.data_dir.glob("obj*.png")):
            return
        download_and_extract_archive(COIL100_URL, download_root=str(self.data_dir.parent))

    def setup(self, stage: str | None = None) -> None:
        if self.trainer is not None:
            world_size = self.trainer.world_size
            if self.batch_size % world_size != 0:
                raise RuntimeError(f"Batch size ({self.batch_size}) is not divisible by the number of devices ({world_size}).")
            self.batch_size_per_device = self.batch_size // world_size
            self.val_batch_size_per_device = self.val_batch_size // world_size

        if self.data_train is None:
            self.data_train = COIL100Dataset(self.data_dir, self.train_transform, "train", self.val_pose_stride)
        if self.eval_transform is not None and self.data_val is None:
            self.data_val = COIL100Dataset(self.data_dir, self.eval_transform, "val", self.val_pose_stride)

    def train_dataloader(self) -> DataLoader[Any]:
        if self.data_train is None:
            raise RuntimeError("setup() must create data_train before train_dataloader() is called.")
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any] | None:
        if self.data_val is None:
            return None
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.val_batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            shuffle=False,
        )
