# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Download the seven SPHERE-JEPA Table 1 transfer datasets with torchvision, then verify them.

Run it with the repo on the Python path (``PYTHONPATH=<clone>``; the batch wrapper does that).
Every download ends by resetting the modification times of the whole tree (F49: the scratch
purge deletes by mtime, and extracted archives carry old ones).

Run it as a batch job (``scripts/aicr/slurm_download_transfer.sh``): on AICR the compute nodes
have outbound internet and the login node kills sustained downloads, so the old instruction
"run on a login node" was wrong there. The eval jobs then read the datasets with
``download=False``.

    python scripts/download_transfer_datasets.py --root <scratch>/transfer_datasets
    python scripts/download_transfer_datasets.py --root ... --only dtd,aircraft
    python scripts/download_transfer_datasets.py --root ... --verify

``--verify`` re-opens every dataset OFFLINE and records its split sizes and class count in
``<root>/.verified`` (JSON). It replaces the existence-only ``.ready`` marker, which a partial
download could earn: a file on disk proves a process started, not that it finished
(research_rules/05).
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.transfer import TRANSFER_DATASETS, get_transfer_splits
from src.utils.touch import touch_tree


def remove_hollow_dirs(root: str) -> list[str]:
    """Remove every directory under ``root`` that holds no regular file -- what the scratch purge
    leaves behind (F49: files go, directories stay). torchvision takes such a directory for a
    downloaded dataset, skips the extraction and then fails to read it; four of the seven did on
    2026-09-22. Returns the names removed."""
    removed = []
    if not os.path.isdir(root):
        return removed
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path) and not os.path.islink(path) and not any(
            files for _, _, files in os.walk(path)
        ):
            shutil.rmtree(path)
            removed.append(name)
    return removed


def download(root: str, keys: list[str]) -> list[str]:
    failed = []
    hollow = remove_hollow_dirs(root)
    if hollow:
        print(f"=== removed {len(hollow)} hollow director{'y' if len(hollow) == 1 else 'ies'} (purge leftovers): "
              f"{', '.join(hollow)} ===", flush=True)
    for key in keys:
        started = time.time()
        print(f"=== {key}: downloading to {root} ===", flush=True)
        try:
            train, test, num_classes = get_transfer_splits(key, root, crop_size=128, download=True)
            print(f"=== {key}: train {len(train)} | test {len(test)} | classes {num_classes} "
                  f"| {(time.time() - started) / 60:.1f} min ===", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"=== {key}: FAILED: {exc!r} ===", flush=True)
            failed.append(key)
    # F49: the scratch purge is by mtime and extraction keeps the archives' old mtimes; without this
    # the extracted files are gone at the next sweep while the archives and directories remain
    print(f"=== touched {touch_tree(root)} entries under {root} (scratch purge is by mtime, F49) ===", flush=True)
    return failed


def verify(root: str, keys: list[str]) -> dict[str, dict[str, int]]:
    """Every dataset re-opened with ``download=False``; ``get_transfer_splits`` itself raises
    when the labels do not span the expected class count, and empty splits are refused here."""
    report: dict[str, dict[str, int]] = {}
    problems: list[str] = []
    for key in keys:
        try:
            train, test, num_classes = get_transfer_splits(key, root, crop_size=128, download=False)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{key}: {exc!r}")
            continue
        if len(train) == 0 or len(test) == 0:
            problems.append(f"{key}: empty split (train {len(train)}, test {len(test)})")
            continue
        report[key] = {"train": len(train), "test": len(test), "classes": num_classes}
        print(f"=== {key}: verified train {len(train)} | test {len(test)} | classes {num_classes} ===", flush=True)
    if problems:
        raise SystemExit("VERIFICATION FAILED:\n  " + "\n  ".join(problems))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--only", default=None, help="Comma-separated subset of dataset keys.")
    parser.add_argument("--verify", action="store_true", help="Re-open every dataset offline and write .verified.")
    args = parser.parse_args()
    keys = [k.strip() for k in args.only.split(",")] if args.only else list(TRANSFER_DATASETS)

    if args.verify:
        report = verify(args.root, keys)
        if not args.only:
            marker = Path(args.root) / ".verified"
            tmp = marker.with_suffix(".tmp")
            tmp.write_text(json.dumps({"datasets": report, "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1) + "\n")
            os.replace(tmp, marker)
            print(f"wrote {marker}")
        print("All datasets verified.")
        return

    failed = download(args.root, keys)
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)
    print("All datasets downloaded; run again with --verify before any evaluation reads them.")


if __name__ == "__main__":
    main()
