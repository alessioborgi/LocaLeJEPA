# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Build an ``eval_indomain`` config that re-reads the ablation checkpoints with the DRAFT's probes.

Why this exists (2026-09-22, `state/10-PAPER-STATE.md`): the ablation tables rank cells by the
ONLINE probe's final-epoch accuracy, while the draft's neighbourhood-count table ranks the same
kind of runs by a post-hoc k-NN (k=20) and a linear head fitted on the frozen train split. The two
rankings disagree -- on the online probe the ``K=1`` control wins -- and an online probe is known
to under-read a representation that is well structured but not yet linearly separable. Re-reading
the checkpoints we already have with the draft's probes decides which of the two it is, for ~20
GPU-hours instead of the ~1100 a longer rerun would cost.

The generated config is DATA: one ``backbones`` entry per ablation cell, pointing at an absolute
checkpoint path, in the grid's own order. A cell is included only when its checkpoint exists and
the launcher's ``.verified`` marker is beside it (plan G5/V4), so an eval can never run on a
half-trained cell. No override is emitted: ``load_backbone`` composes ``train`` with
``experiment=<name>`` and instantiates ONLY ``model.encoder``, which is the same ViT-S/8 for every
cell of the grid -- lambda, K, the batch and the logger do not reach it.

    python scripts/aicr/make_ablation_eval_config.py \
        --grid $SCR/grids/ablation_grid.json --ckpt-root $SCR/checkpoints_ablations \
        --data-dir $SCR/data/imagenette --out $SCR/eval/ablation_indomain.yaml \
        --names $SCR/eval/ablation_names.txt --out-dir $SCR/logs/eval_ablation

``--names`` is the array index -> backbone name map the job script reads; it is written in the
same order as the config's ``backbones``, so task *i* evaluates line *i+1*.
"""

import argparse
import json
import os
from pathlib import Path

# The draft's protocol, copied from configs/eval/indomain_imagenette.yaml so the numbers are
# comparable to `tab:k-ablation`: cosine-weighted k-NN reported at k=20, and a linear head whose
# learning rate is selected on a held-out tenth of the train split (never on the reported split).
KNN = {"k_values": [10, 20, 100, 200], "temperature": 0.07, "report_k": 20, "chunk_size": 512}
LINEAR = {"lr_grid": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0], "epochs": 100, "batch_size": 1024,
          "weight_decay": 0.0, "momentum": 0.9, "standardize": True, "val_frac": 0.1}


def select_cells(runs, ckpt_root: Path, kind: str, require_verified: bool = True):
    """``(entries, skipped)`` in the grid's order. A cell enters only with BOTH its checkpoint and,
    unless explicitly waived, the ``.verified`` marker the launcher writes after its postflight."""
    entries, skipped = [], []
    for run in runs:
        name = run["run_name"]
        run_dir = ckpt_root / name
        ckpt = run_dir / f"{kind}.ckpt"
        if not ckpt.is_file():
            skipped.append(f"{name}: no {kind}.ckpt")
            continue
        if require_verified and not (run_dir / ".verified").is_file():
            skipped.append(f"{name}: not verified")
            continue
        entries.append({"name": name, "experiment": run["experiment"], "checkpoint": str(ckpt)})
    return entries, skipped


def build_config(entries, *, data_dir: str, out_dir: str, cache_dir: str, crop_size: int,
                 variant: str, batch_size: int, num_workers: int, seed: int) -> dict:
    return {
        "device": None, "seed": seed,
        "dataset": "imagenette", "data_dir": data_dir, "imagenette_variant": variant,
        "crop_size": crop_size, "batch_size": batch_size, "num_workers": num_workers,
        "output_folder": out_dir, "cache_dir": cache_dir,
        "train_subset_per_class": None,
        "knn": dict(KNN), "linear": dict(LINEAR),
        "backbones": entries,
    }


def main(argv=None) -> int:
    import yaml

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--ckpt-root", type=Path, required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", type=Path, required=True, help="the eval config to write")
    p.add_argument("--names", type=Path, required=True, help="array index -> backbone name map")
    p.add_argument("--out-dir", required=True, help="output_folder of the eval (results go here)")
    p.add_argument("--cache-dir", default=None, help="feature cache (default: <out-dir>/cache)")
    p.add_argument("--checkpoint-kind", default="last", choices=("last", "best"))
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--variant", default="160px")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-unverified", action="store_true",
                   help="include cells without a .verified marker (diagnostic only; never for a table)")
    args = p.parse_args(argv)

    runs = json.loads(args.grid.read_text(encoding="utf-8"))["runs"]
    entries, skipped = select_cells(runs, args.ckpt_root, args.checkpoint_kind,
                                    require_verified=not args.allow_unverified)
    cfg = build_config(entries, data_dir=args.data_dir, out_dir=args.out_dir,
                       cache_dir=args.cache_dir or os.path.join(args.out_dir, "cache"),
                       crop_size=args.crop_size, variant=args.variant, batch_size=args.batch_size,
                       num_workers=args.num_workers, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    args.names.parent.mkdir(parents=True, exist_ok=True)
    args.names.write_text("".join(f"{e['name']}\n" for e in entries), encoding="utf-8")

    print(f"{len(entries)} cell(s) of {len(runs)} -> {args.out}")
    print(f"array indices 0-{len(entries) - 1} -> {args.names}" if entries else "NOTHING TO EVALUATE")
    if skipped:
        print(f"skipped {len(skipped)}: " + "; ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else ""))
    return 0 if entries else 1


if __name__ == "__main__":
    raise SystemExit(main())
