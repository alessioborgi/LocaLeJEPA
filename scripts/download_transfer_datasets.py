# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Download the seven SPHERE-JEPA Table 1 transfer datasets with torchvision.

Run on a login node (compute nodes have no internet); the eval jobs then read
the datasets with ``download=False``.

    python scripts/download_transfer_datasets.py --root <your-scratch>/transfer_datasets
    python scripts/download_transfer_datasets.py --root ... --only dtd,aircraft
"""

import argparse
import sys
import time
from pathlib import Path

from src.eval.transfer import TRANSFER_DATASETS, get_transfer_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--only", default=None, help="Comma-separated subset of dataset keys.")
    args = parser.parse_args()
    keys = [k.strip() for k in args.only.split(",")] if args.only else list(TRANSFER_DATASETS)
    failed = []
    for key in keys:
        started = time.time()
        print(f"=== {key}: downloading to {args.root} ===", flush=True)
        try:
            train, test, num_classes = get_transfer_splits(key, args.root, crop_size=128, download=True)
            print(f"=== {key}: train {len(train)} | test {len(test)} | classes {num_classes} "
                  f"| {(time.time() - started) / 60:.1f} min ===", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"=== {key}: FAILED: {exc!r} ===", flush=True)
            failed.append(key)
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)
    if not args.only:
        # Marker consumed by the Slurm launchers (they wait for it before evaluating).
        open(Path(args.root) / ".ready", "w").close()
    print("All datasets ready.")


if __name__ == "__main__":
    main()
