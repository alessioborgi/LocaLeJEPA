#!/usr/bin/env python
"""Pre-flight check on a staged ImageNet-1K tree, before a campaign depends on it.

``scripts/stage_imagenet1k_hf.py`` verifies one thing when it finishes: that each split has 1000
class directories. That is far too weak to launch weeks of GPU time behind. A download that was
interrupted after touching every class -- which is the *likely* interruption, since the shards are
not ordered by class -- passes it with images missing from every one.

What this checks instead, cheapest and most decisive first:

1. **Both splits have the same 1000 class directories.** ``get_indomain_splits`` asserts
   ``train.class_to_idx == val.class_to_idx``; a mismatch is a hard failure hours into an
   evaluation, and it is free to detect now.
2. **Names are zero-padded to four digits.** ``ImageFolder`` sorts class directories as *strings*
   and assigns labels in that order, so ``10`` sorting before ``2`` permutes every label. The
   model trains perfectly and evaluates to noise.
3. **val has exactly 50 images per class.** ImageNet-1K's validation split is exactly 50 per
   class, 50,000 total -- the sharpest integrity check the dataset offers, and it catches a
   partial download that item 1 would not.
4. **train totals 1,281,167**, with per-class counts inside the real [732, 1300] range.
5. **No zero-byte files**, which is what an interrupted write leaves.
6. **JPEG magic number** on a sample, which catches truncation without decoding 1.28M files.
   ``--decode N`` additionally opens N images per split for a real decode.

Usage:
    python scripts/verify_imagenet_staging.py --root $IN1K_ROOT
    python scripts/verify_imagenet_staging.py --root $IN1K_ROOT --decode 200
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

# ImageNet-1K (ILSVRC2012), the published figures.
EXPECTED_CLASSES = 1000
EXPECTED = {"train": 1_281_167, "val": 50_000}
VAL_PER_CLASS = 50
TRAIN_PER_CLASS_RANGE = (732, 1300)
JPEG_MAGIC = b"\xff\xd8\xff"


def scan_split(split_dir: Path) -> tuple[dict[str, int], list[str]]:
    """Per-class file counts and any zero-byte files. One ``scandir`` per class directory."""
    counts: dict[str, int] = {}
    empty: list[str] = []
    for entry in sorted(os.scandir(split_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        n = 0
        for f in os.scandir(entry.path):
            if not f.is_file():
                continue
            n += 1
            if f.stat().st_size == 0:
                empty.append(f.path)
        counts[entry.name] = n
    return counts, empty


def _files_in_order(class_dir: Path) -> list[str]:
    """The class's files in NAME order, so a seeded draw picks the same file on every machine.

    F44: `rng.choice` over a bare `os.scandir` picked a different file on the cluster's NFS than
    on the workstation's APFS with the same seed -- readdir order is a filesystem property, not a
    stable one -- and the suite's planted-garbage test was green on one machine and red on the
    other. Same defect class as the probe's unseeded head (`b97840f`): a seed that reaches a
    draw over an unstable order is not a seed.
    """
    return sorted(f.path for f in os.scandir(class_dir) if f.is_file())


def check_magic(split_dir: Path, counts: dict[str, int], sample: int, rng) -> list[str]:
    """Read the first three bytes of `sample` files spread across classes."""
    bad = []
    classes = [c for c, n in counts.items() if n]
    for name in rng.sample(classes, min(sample, len(classes))):
        files = _files_in_order(split_dir / name)
        if not files:
            continue
        path = rng.choice(files)
        with open(path, "rb") as handle:
            if handle.read(3) != JPEG_MAGIC:
                bad.append(path)
    return bad


def verify(root: Path, decode: int = 0, sample: int = 200, seed: int = 0) -> list[str]:
    """Return a list of problems. Empty means the tree is usable."""
    rng = random.Random(seed)
    problems: list[str] = []
    per_split: dict[str, dict[str, int]] = {}

    for split in ("train", "val"):
        split_dir = root / split
        if not split_dir.is_dir():
            problems.append(f"missing split directory: {split_dir}")
            continue

        counts, empty = scan_split(split_dir)
        per_split[split] = counts

        if len(counts) != EXPECTED_CLASSES:
            problems.append(f"{split}: {len(counts)} class dirs, expected {EXPECTED_CLASSES}")

        bad_names = [c for c in counts if not (len(c) == 4 and c.isdigit())]
        if bad_names:
            problems.append(
                f"{split}: {len(bad_names)} class dirs are not 4-digit zero-padded "
                f"(e.g. {bad_names[:5]}). ImageFolder sorts as strings, so labels would permute"
            )

        total = sum(counts.values())
        if total != EXPECTED[split]:
            # signed as (found - expected), so a shortfall reads negative. The other direction
            # is technically the same number and reads as its opposite.
            problems.append(f"{split}: {total:,} images, expected {EXPECTED[split]:,} "
                            f"({total - EXPECTED[split]:+,})")

        if empty:
            problems.append(f"{split}: {len(empty)} zero-byte file(s), e.g. {empty[:3]}")

        if split == "val":
            wrong = {c: n for c, n in counts.items() if n != VAL_PER_CLASS}
            if wrong:
                problems.append(
                    f"val: {len(wrong)} class(es) do not have exactly {VAL_PER_CLASS} images "
                    f"(e.g. {dict(list(wrong.items())[:5])})"
                )
        else:
            lo, hi = TRAIN_PER_CLASS_RANGE
            out = {c: n for c, n in counts.items() if not (lo <= n <= hi)}
            if out:
                problems.append(f"train: {len(out)} class(es) outside [{lo}, {hi}] "
                                f"(e.g. {dict(list(out.items())[:5])})")

        if counts:
            bad = check_magic(split_dir, counts, sample, rng)
            if bad:
                problems.append(f"{split}: {len(bad)} sampled file(s) are not JPEG, e.g. {bad[:3]}")

        if decode and counts:
            from PIL import Image

            failed = []
            classes = [c for c, n in counts.items() if n]
            for name in rng.sample(classes, min(decode, len(classes))):
                files = _files_in_order(split_dir / name)
                if not files:
                    continue
                path = rng.choice(files)
                try:
                    with Image.open(path) as im:
                        im.convert("RGB").load()
                except Exception as error:  # noqa: BLE001
                    failed.append(f"{path} ({type(error).__name__})")
            if failed:
                problems.append(f"{split}: {len(failed)} sampled file(s) failed to decode, "
                                f"e.g. {failed[:3]}")

    if len(per_split) == 2:
        only_train = sorted(set(per_split["train"]) - set(per_split["val"]))
        only_val = sorted(set(per_split["val"]) - set(per_split["train"]))
        if only_train or only_val:
            problems.append(
                f"train and val have different class sets "
                f"(train-only: {only_train[:5]}, val-only: {only_val[:5]}). "
                f"get_indomain_splits asserts class_to_idx equality and will fail"
            )

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True,
                        help="the directory containing train/ and val/ "
                             "(i.e. <dest>/imagenet_full_size/061417)")
    parser.add_argument("--decode", type=int, default=0,
                        help="also fully decode this many images per split (slower)")
    parser.add_argument("--sample", type=int, default=200,
                        help="how many files per split to magic-number check")
    args = parser.parse_args()

    root = Path(args.root)
    print(f"verifying {root} ...", flush=True)
    problems = verify(root, decode=args.decode, sample=args.sample)

    if not problems:
        print("\nOK: the staged tree matches ImageNet-1K in every check.")
        return 0

    print(f"\n{len(problems)} PROBLEM(S):")
    for problem in problems:
        print(f"  - {problem}")
    print("\nDo not launch a campaign against this tree.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
