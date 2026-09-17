# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Bobblehead (Yoda + bulldog) rotating-figures dataset.

This is the image data set used in Kevrekidis et al., "Thinner Latent Spaces"
(arXiv:2408.16138, §3.3) — originally from Lederman & Talmon's alternating-
diffusion work.  Two figurines rotate at independent, fixed (unknown) angular
frequencies in front of a camera, so the image manifold is a **2-torus**
(parametrised by the two rotation angles), embedded in pixel space.

The archive ``yoda_dog_images_and_labels.zip`` stores the frames as a single
NumPy array ``color_image_export.npy`` of shape ``(8097, 240, 320, 3)`` and
dtype ``uint8`` (8097 RGB frames, 240×320).  Frames are in temporal order, so
the array index doubles as a time index.  Any label array (rotation angles) is
loaded if present but is *optional*: the Conformal-Autoencoder dimension-
detection task is unsupervised; labels are only used for the recovery sanity
check.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_ZIP = "yoda_dog_images_and_labels.zip"
IMAGES_ENTRY = "color_image_export.npy"


def build_eval_transform(crop_size: int = 128) -> Callable[[Image.Image], torch.Tensor]:
    """Deterministic eval transform matching the LeJEPA eval pipeline.

    Resizes the short side to ``crop_size``, centre-crops to a square, converts
    to a tensor and applies ImageNet normalisation (the backbones were trained
    with these stats).
    """
    return transforms.Compose([
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def extract_bobblehead(zip_path: str | Path, target_dir: str | Path) -> Path:
    """Extract the bobblehead archive once into ``target_dir``.

    Returns the path to the extracted images ``.npy``.  Raises a clear error if
    the archive is missing or still being downloaded (truncated zip).
    """
    zip_path = Path(zip_path)
    target_dir = Path(target_dir)
    img_path = target_dir / IMAGES_ENTRY

    if img_path.exists():
        return img_path

    if not zip_path.is_file():
        raise FileNotFoundError(f"Bobblehead archive not found: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise RuntimeError(
            f"{zip_path} is not a complete/valid zip yet (download may be in "
            f"progress or truncated). Re-run once the download finishes."
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)

    if not img_path.exists():
        # The image entry might live in a sub-directory; locate it.
        matches = list(target_dir.rglob(IMAGES_ENTRY))
        if matches:
            return matches[0]
        raise FileNotFoundError(
            f"{IMAGES_ENTRY} not found inside {zip_path}. "
            f"Extracted entries: {[p.name for p in target_dir.iterdir()]}"
        )
    return img_path


def _find_labels(target_dir: Path, n_frames: int, labels_name: Optional[str]) -> Optional[np.ndarray]:
    """Best-effort discovery of a per-frame label array (rotation angles).

    If ``labels_name`` is given, load that file. Otherwise scan ``target_dir``
    for any ``.npy``/``.csv`` (other than the images) whose leading axis matches
    the number of frames. Returns ``None`` if nothing suitable is found.
    """
    candidates: list[Path] = []
    if labels_name:
        candidates = [target_dir / labels_name]
    else:
        for p in sorted(target_dir.rglob("*")):
            if p.name == IMAGES_ENTRY:
                continue
            if p.suffix.lower() in (".npy", ".csv", ".txt"):
                candidates.append(p)

    for p in candidates:
        if not p.is_file():
            continue
        try:
            if p.suffix.lower() == ".npy":
                arr = np.load(p, allow_pickle=False)
            else:
                arr = np.loadtxt(p, delimiter="," if p.suffix.lower() == ".csv" else None)
        except Exception:
            continue
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1 and arr.shape[0] == n_frames:
            return arr[:, None]
        if arr.ndim == 2:
            # Labels may be stored as (N, k) or transposed (k, N) — e.g. the
            # bobblehead labels.npy is (2, 8097): two rotation-angle rows.
            if arr.shape[0] == n_frames:
                return arr
            if arr.shape[1] == n_frames:
                return arr.T
    return None


class BobbleheadDataset(Dataset):
    """RGB frames of the two rotating figures.

    Each item is ``(image_tensor, label)`` where ``label`` is the rotation-angle
    vector when available, otherwise the integer frame index (a stand-in time
    coordinate useful for colouring the recovered torus).
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        zip_name: str = DEFAULT_ZIP,
        extract_dir: Optional[str | Path] = None,
        crop_size: int = 128,
        transform: Optional[Callable[[Any], Any]] = None,
        labels_name: Optional[str] = None,
        max_frames: Optional[int] = None,
        subsample: int = 1,
    ) -> None:
        data_dir = Path(data_dir)
        zip_path = data_dir / zip_name
        target_dir = Path(extract_dir) if extract_dir is not None else data_dir / "bobblehead"

        img_path = extract_bobblehead(zip_path, target_dir)
        self.images = np.load(img_path, mmap_mode="r")  # (N, 240, 320, 3) uint8
        if self.images.ndim != 4 or self.images.shape[-1] != 3:
            raise RuntimeError(f"Unexpected image array shape {self.images.shape}; expected (N, H, W, 3).")

        n_total = self.images.shape[0]
        labels = _find_labels(target_dir, n_total, labels_name)
        self.labels = labels  # (N, k) or None

        # Optional temporal subsampling / truncation (keeps order).
        self.indices = list(range(0, n_total, max(1, subsample)))
        if max_frames is not None:
            self.indices = self.indices[:max_frames]

        self.transform = transform if transform is not None else build_eval_transform(crop_size)
        self.has_labels = labels is not None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        frame = np.ascontiguousarray(self.images[idx])  # (240, 320, 3) uint8
        image = self.transform(Image.fromarray(frame, mode="RGB"))
        if self.has_labels:
            label = torch.as_tensor(self.labels[idx], dtype=torch.float32)
        else:
            label = torch.tensor(float(idx))  # frame index as time coordinate
        return image, label


def get_bobblehead_dataset(
    data_dir: str | Path = "data",
    crop_size: int = 128,
    **kwargs: Any,
) -> BobbleheadDataset:
    """Convenience constructor used by the evaluation entry point."""
    return BobbleheadDataset(data_dir=data_dir, crop_size=crop_size, **kwargs)


if __name__ == "__main__":
    ds = get_bobblehead_dataset()
    print(f"BobbleheadDataset: {len(ds)} frames, labels={ds.has_labels}")
    x, y = ds[0]
    print("sample image tensor:", tuple(x.shape), "label:", y)
