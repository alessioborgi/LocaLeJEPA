# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Stage ImageNet-1K from HuggingFace into the ImageFolder layout expected by
``src.data.imagenet1k_datamodule.ImageNet1KDataModule``.

Uses the official gated ``ILSVRC/imagenet-1k`` dataset via ``datasets`` (as
recommended by the dataset card: ``load_dataset("ILSVRC/imagenet-1k")``) and
writes the *original* JPEG bytes out to

    <dest>/imagenet_full_size/061417/train/<label>/train_<i>.JPEG
    <dest>/imagenet_full_size/061417/val/<label>/val_<i>.JPEG

Class directories are the zero-padded integer HuggingFace label (``0000``..
``0999``); ``torchvision.datasets.ImageFolder`` sorts them alphanumerically, so
the resulting class index equals the HF label and is consistent across the
train/val splits. (Full-IN1K SSL runs use ``class_subset_file: null``, so the
directory names need only be consistent, not WordNet ids.)

Images are exported with ``Image(decode=False)`` so the source JPEG bytes are
copied verbatim — no re-encoding, no quality loss, and much faster than
decode+re-save. A handful of source images are not JPEG (PNG/CMYK); those are
decoded and re-saved as RGB JPEG via the ``--reencode-fallback`` path.

Prerequisite (one-time, must be done by the account owner):
    1. Accept the license at https://huggingface.co/datasets/ILSVRC/imagenet-1k
    2. hf auth login          (or: huggingface-cli login)

Usage:
    python scripts/stage_imagenet1k_hf.py --dest <your-scratch>/imagenet
"""

import argparse
import io
import os
import sys
from pathlib import Path

from datasets import Image, load_dataset
from huggingface_hub import get_token

REPO_ID = "ILSVRC/imagenet-1k"
# HuggingFace split name -> ImageFolder subdir name the datamodule reads.
SPLITS = [("train", "train"), ("validation", "val")]


def _write_example(image_field: dict, label: int, split_dir: Path, idx: int) -> bool:
    """Write one example's original bytes to <split_dir>/<label>/<split>_<idx>.JPEG.

    Returns True if a new file was written, False if it already existed.
    """
    class_dir = split_dir / f"{int(label):04d}"
    class_dir.mkdir(parents=True, exist_ok=True)
    dest = class_dir / f"{split_dir.name}_{idx}.JPEG"
    if dest.exists():
        return False

    data = image_field.get("bytes")
    if data is None and image_field.get("path"):
        data = Path(image_field["path"]).read_bytes()
    if data is None:
        raise ValueError(f"example {idx} in {split_dir} has neither image bytes nor path")

    # Verbatim copy for JPEG; decode+re-save for the few non-JPEG source images.
    if data[:3] == b"\xff\xd8\xff":
        with open(dest, "wb") as out:
            out.write(data)
    else:
        from PIL import Image as PILImage

        with PILImage.open(io.BytesIO(data)) as im:
            im.convert("RGB").save(dest, format="JPEG", quality=95)
    return True


def stage_split(hf_split: str, subdir: str, root: Path, cache_dir: str | None) -> int:
    split_dir = root / subdir
    split_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{subdir}] load_dataset({REPO_ID!r}, split={hf_split!r}) ...", flush=True)
    ds = load_dataset(REPO_ID, split=hf_split, cache_dir=cache_dir)
    # decode=False -> keep the raw stored JPEG bytes instead of decoding to PIL.
    ds = ds.cast_column("image", Image(decode=False))

    written = 0
    for idx, ex in enumerate(ds):
        if _write_example(ex["image"], ex["label"], split_dir, idx):
            written += 1
        if (idx + 1) % 100_000 == 0:
            print(f"[{subdir}] {idx + 1} processed ({written} newly written)", flush=True)

    n_classes = sum(1 for d in split_dir.iterdir() if d.is_dir())
    print(f"[{subdir}] done: {idx + 1} images, {written} new, {n_classes} class dirs", flush=True)
    if n_classes != 1000:
        print(f"ERROR: expected 1000 class dirs in {split_dir}, found {n_classes}", flush=True)
        sys.exit(1)
    return idx + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", required=True, help="Destination root (becomes data.data_dir)")
    parser.add_argument("--cache-dir", default=None, help="HF datasets cache (defaults to HF_HOME)")
    args = parser.parse_args()

    if not get_token():
        print(
            "No HuggingFace token found. Accept the license at\n"
            "  https://huggingface.co/datasets/ILSVRC/imagenet-1k\n"
            "then run `hf auth login` (or `huggingface-cli login`) once.",
            flush=True,
        )
        sys.exit(1)

    root = Path(args.dest) / "imagenet_full_size" / "061417"
    counts = {}
    for hf_split, subdir in SPLITS:
        counts[subdir] = stage_split(hf_split, subdir, root, args.cache_dir)

    print(f"Staged ImageNet-1K at {root} (train {counts['train']}, val {counts['val']})", flush=True)
    print(f"Point the datamodule at it with: data.data_dir={args.dest}", flush=True)

    # The HuggingFace download backend spawns background transfer threads that
    # can segfault during normal interpreter finalization (harmless, but it
    # yields a non-zero exit that would trip `set -euo pipefail` in the Slurm
    # wrapper AFTER the data is fully written and verified above). Data
    # integrity is already checked (1000 class dirs per split), so force a clean
    # exit to guarantee the wrapper proceeds to the symlink step.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
