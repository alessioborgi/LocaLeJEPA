# Copyright (c) 2026 Pavel Snopov
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive

from src.transforms import (
    make_lejepa_eval_transform,
    make_lejepa_transforms,
    make_multicrop_transforms,
)

IMAGENETTE_VARIANTS = {
    "full_size": ("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz", "imagenette2"),
    "320px": ("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz", "imagenette2-320"),
    "160px": ("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz", "imagenette2-160"),
}


class ImagenetteDataset(Dataset):
    """Imagenette loaded from the official fastai archive.

    Accepts any callable transform so it can be driven by either the
    LeJEPA MultiViewTransform (train) or the eval variant (val).
    """

    def __init__(self, root: str | Path, split: str, transform: Callable[[Any], Any]):
        self.dataset = ImageFolder(root=Path(root) / _split_dir(split), transform=transform)

    def __getitem__(self, index: int):
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)


class ImagenetteDataModule(LightningDataModule):
    """LightningDataModule for Imagenette.

    The default setup mirrors the existing LeJEPA Imagenette config:
    official fastai ``160px`` archive, four train views at
    128x128, and one deterministic eval view.
    """

    def __init__(
        self,
        data_dir: str = "data/imagenette",
        variant: str = "160px",
        train_split: str = "train",
        val_split: str = "validation",
        batch_size: int = 256,
        val_batch_size: Optional[int] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: Optional[bool] = None,
        drop_last: bool = True,
        crop_size: int = 128,
        crop_scale: Sequence[float] = (0.08, 1.0),
        num_views: int = 4,
        num_global_crops: Optional[int] = None,
        num_local_crops: int = 0,
        global_crop_scale: Sequence[float] = (0.3, 1.0),
        local_crop_scale: Sequence[float] = (0.05, 0.3),
        record_metadata: bool = False,
        use_gaussian_blur: bool = True,
        use_horizontal_flip: bool = True,
        use_color_distortion: bool = True,
        color_jitter_strength: float = 0.8,
    ) -> None:
        super().__init__()

        # * I do the normalization here in order to save the actual values of the hyperparameters
        val_batch_size = val_batch_size if val_batch_size is not None else batch_size
        persistent_workers = num_workers > 0 if persistent_workers is None else persistent_workers
        crop_scale = tuple(crop_scale)
        global_crop_scale = tuple(global_crop_scale)
        local_crop_scale = tuple(local_crop_scale)

        self.save_hyperparameters(logger=False)

        self.data_dir = Path(data_dir)
        self.variant = variant
        # ? I would remove this boilerplate code and just use `self.hparams` or dataclasses
        # ? But the use of `self.hparams` is not aligned with Pylance type checking,
        # ? and dataclasses increases the complexity of the codebase
        self.train_split = train_split
        self.val_split = val_split
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.drop_last = drop_last
        self.crop_size = crop_size
        self.crop_scale = crop_scale
        self.num_views = num_views
        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops
        self.global_crop_scale = global_crop_scale
        self.local_crop_scale = local_crop_scale
        self.record_metadata = record_metadata
        self.use_gaussian_blur = use_gaussian_blur
        self.use_horizontal_flip = use_horizontal_flip
        self.use_color_distortion = use_color_distortion
        self.color_jitter_strength = color_jitter_strength

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

    def setup(self, stage: Optional[str] = None) -> None:  # ? Use stage-specific logic?
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
            self.data_train = ImagenetteDataset(self._dataset_root, self.train_split, transform=self._train_transform())
        if self.data_val is None:
            self.data_val = ImagenetteDataset(self._dataset_root, self.val_split, transform=self._eval_transform())
        if self.data_test is None:
            self.data_test = ImagenetteDataset(self._dataset_root, self.val_split, transform=self._eval_transform())

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,  # type: ignore
            batch_size=self.batch_size_per_device,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,  # type: ignore
            batch_size=self.val_batch_size_per_device,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,  # type: ignore
            batch_size=self.val_batch_size_per_device,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    def _train_transform(self):
        if self.num_global_crops is not None:
            return make_multicrop_transforms(
                global_crop_size=self.crop_size,
                num_global_crops=self.num_global_crops,
                num_local_crops=self.num_local_crops,
                global_crop_scale=self.global_crop_scale,
                local_crop_scale=self.local_crop_scale,
                use_color_distortion=self.use_color_distortion,
                color_jitter=self.color_jitter_strength,
                use_gaussian_blur=self.use_gaussian_blur,
                use_horizontal_flip=self.use_horizontal_flip,
                record_metadata=self.record_metadata,
            )

        return make_lejepa_transforms(
            crop_size=self.crop_size,
            crop_scale=self.crop_scale,
            num_views=self.num_views,
            gaussian_blur=self.use_gaussian_blur,
            horizontal_flip=self.use_horizontal_flip,
            color_distortion=self.use_color_distortion,
            color_jitter=self.color_jitter_strength,
        )

    def _eval_transform(self):
        return make_lejepa_eval_transform(crop_size=self.crop_size)

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
    _ = ImagenetteDataModule()
    print("ImagenetteDataModule successfully initialized.")
