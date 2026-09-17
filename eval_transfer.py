# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Transfer linear-probing evaluation of pretrained JEPA backbones.

Reproduces SPHERE-JEPA (Nicollier et al., 2026) Table 1: an ImageNet-1K
pretrained backbone is frozen and a linear classifier is trained on its
features for each of seven downstream datasets (DTD, Aircraft, CIFAR-10,
CIFAR-100, Flowers, Food, Pets); top-1 accuracy on each test split and their
mean ("Avg") are reported.

Per backbone the features of every dataset are extracted once with the
deterministic eval transform, cached to disk, and probed with the same linear
probe as the in-domain evaluation (``src/eval/linear_probe.py``). Results are
written incrementally to ``<output_folder>/per_backbone/<name>.json`` so an
interrupted job resumes at the first unfinished dataset.

Usage
-----
    python eval_transfer.py --config configs/eval/transfer_in1k_ep100.yaml
    python eval_transfer.py --config ... --only hvit_sigreg --datasets dtd,pets
    python eval_transfer.py --config ... --merge-only
"""

import argparse
import csv
import json
import logging
import os
import sys
import time

import torch
import yaml

from src.eval.extract import load_backbone
from src.eval.indomain import extract_classification_features
from src.eval.linear_probe import run_linear_probe
from src.eval.transfer import TRANSFER_DATASETS, get_transfer_splits

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()


def _per_backbone_dir(out_dir: str) -> str:
    path = os.path.join(out_dir, "per_backbone")
    os.makedirs(path, exist_ok=True)
    return path


def _backbone_file(name: str, out_dir: str) -> str:
    return os.path.join(_per_backbone_dir(out_dir), f"{name}.json")


def _load_backbone_entry(name: str, out_dir: str) -> dict | None:
    path = _backbone_file(name, out_dir)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    return None


def _save_backbone(name: str, entry: dict, out_dir: str) -> None:
    """One file per backbone so concurrent per-backbone jobs never clobber."""
    path = _backbone_file(name, out_dir)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as file:
        json.dump(entry, file, indent=2)
    os.replace(tmp, path)


def _collect_results(out_dir: str, order: list[str]) -> dict:
    folder = _per_backbone_dir(out_dir)
    found = {}
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".json"):
            with open(os.path.join(folder, filename), encoding="utf-8") as file:
                found[filename[: -len(".json")]] = json.load(file)
    ordered = {name: found[name] for name in order if name in found}
    ordered.update({name: entry for name, entry in found.items() if name not in ordered})
    return ordered


def _average(entry: dict, keys: list[str]) -> float | None:
    """Mean top-1 over ``keys``; ``None`` unless every dataset is present."""
    vals = [entry["datasets"][k]["top1"] for k in keys if k in entry["datasets"]]
    return sum(vals) / len(vals) if len(vals) == len(keys) else None


def _write_results(results: dict, out_dir: str, keys: list[str]) -> None:
    if not results:
        return
    for entry in results.values():
        entry["avg_top1"] = _average(entry, keys)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    with open(os.path.join(out_dir, "results.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        header = ["backbone", "experiment", "checkpoint", "epoch", "feat_dim"]
        header += [f"{k}_top1" for k in keys] + ["avg_top1"] + [f"{k}_top5" for k in keys] + [f"{k}_lr" for k in keys]
        writer.writerow(header)
        for name, entry in results.items():
            ds = entry["datasets"]
            row = [name, entry["experiment"], entry["checkpoint"], entry.get("epoch"), entry["feat_dim"]]
            row += [f'{ds[k]["top1"]:.2f}' if k in ds else "" for k in keys]
            row += [f'{entry["avg_top1"]:.2f}' if entry["avg_top1"] is not None else ""]
            row += [f'{ds[k]["top5"]:.2f}' if k in ds else "" for k in keys]
            row += [ds[k]["best_lr"] if k in ds else "" for k in keys]
            writer.writerow(row)


def _print_table(results: dict, keys: list[str]) -> None:
    names = [TRANSFER_DATASETS[k][0] for k in keys]
    width = 30 + 9 * (len(keys) + 2)
    print("\n" + "=" * width)
    print("Transfer linear probing, top-1 % (SPHERE-JEPA Table 1 protocol)")
    print("=" * width)
    print(f'{"Backbone":28s} {"ep":>4s} ' + " ".join(f"{n:>8s}" for n in names) + f' {"Avg":>8s}')
    print("-" * width)
    for name, entry in results.items():
        ds = entry["datasets"]
        cells = [f'{ds[k]["top1"]:8.1f}' if k in ds else f'{"—":>8s}' for k in keys]
        avg = _average(entry, keys)
        avg_cell = f"{avg:8.1f}" if avg is not None else f'{"—":>8s}'
        print(f'{name:28s} {str(entry.get("epoch", "?")):>4s} ' + " ".join(cells) + f" {avg_cell}")
    print("=" * width)


def main(cfg: dict, only: list[str] | None, datasets: list[str] | None, merge_only: bool = False,
         force: bool = False) -> None:
    device = torch.device(cfg.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    crop_size = int(cfg.get("crop_size", 128))
    out_dir = cfg.get("output_folder", "logs/eval_transfer")
    cache_dir = cfg.get("cache_dir")
    os.makedirs(out_dir, exist_ok=True)

    keys = list(cfg.get("datasets") or TRANSFER_DATASETS)
    unknown = [k for k in keys if k not in TRANSFER_DATASETS]
    if unknown:
        raise ValueError(f"Unknown transfer datasets in config: {unknown}")
    config_order = [entry["name"] for entry in cfg["backbones"]]

    if merge_only:
        merged = _collect_results(out_dir, config_order)
        _print_table(merged, keys)
        _write_results(merged, out_dir, keys)
        logger.info("Merged %d backbone result files into %s", len(merged), out_dir)
        return

    run_keys = [k for k in keys if not datasets or k in datasets]
    linear_cfg = cfg.get("linear", {})
    data_root = cfg["data_root"]
    failures: list[str] = []

    for entry in cfg["backbones"]:
        name = entry["name"]
        if only and name not in only:
            continue
        if not os.path.isfile(entry["checkpoint"]):
            logger.error("%s: checkpoint %s not found — skipping.", name, entry["checkpoint"])
            continue

        epoch = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False, mmap=True).get("epoch")
        previous = None if force else _load_backbone_entry(name, out_dir)
        if previous is not None and previous.get("epoch") != epoch:
            logger.warning("%s: existing results are for epoch %s, checkpoint is epoch %s — recomputing.",
                           name, previous.get("epoch"), epoch)
            previous = None
        done = dict(previous["datasets"]) if previous else {}
        todo = [k for k in run_keys if k not in done]
        if not todo:
            logger.info("%s: all datasets already evaluated — skipping.", name)
            continue

        logger.info("=== backbone: %s (%s) | datasets: %s ===", name, entry["checkpoint"], ", ".join(todo))
        started = time.time()
        try:
            extract_fn, feat_dim = load_backbone(
                experiment=entry["experiment"],
                checkpoint_path=entry["checkpoint"],
                device=device,
                crop_size=crop_size,
                overrides=entry.get("overrides"),
            )
        except Exception:
            logger.exception("%s: failed to load backbone — skipping.", name)
            failures.append(name)
            continue
        logger.info("%s: feature dim %d, checkpoint epoch %s", name, feat_dim, epoch)

        record = {
            "experiment": entry["experiment"],
            "checkpoint": entry["checkpoint"],
            "epoch": epoch,
            "feat_dim": feat_dim,
            "crop_size": crop_size,
            "linear_config": {
                "lr_grid": linear_cfg.get("lr_grid", [0.01, 0.03, 0.1, 0.3, 1.0]),
                "epochs": int(linear_cfg.get("epochs", 100)),
                "batch_size": int(linear_cfg.get("batch_size", 256)),
                "weight_decay": float(linear_cfg.get("weight_decay", 0.0)),
                "val_frac": float(linear_cfg.get("val_frac", 0.1)),
                "standardize": bool(linear_cfg.get("standardize", True)),
            },
            "datasets": done,
        }

        for key in todo:
            ds_started = time.time()
            display = TRANSFER_DATASETS[key][0]
            try:
                train_ds, test_ds, num_classes = get_transfer_splits(key, data_root, crop_size, download=False)
            except Exception:
                logger.exception("%s/%s: dataset unavailable — skipping (download it on a login node).", name, key)
                failures.append(f"{name}/{key}")
                continue
            logger.info("%s/%s: train %d | test %d | classes %d", name, display, len(train_ds), len(test_ds), num_classes)

            def cache(split: str) -> str | None:
                if not cache_dir:
                    return None
                return os.path.join(cache_dir, name, f"{key}_{split}_{crop_size}_ep{epoch}.pt")

            X_train, y_train = extract_classification_features(
                extract_fn, train_ds, batch_size=cfg.get("batch_size", 256),
                num_workers=cfg.get("num_workers", 8), cache_path=cache("train"), tag=f"{name}/{key}/train",
                log_every=100,
            )
            X_test, y_test = extract_classification_features(
                extract_fn, test_ds, batch_size=cfg.get("batch_size", 256),
                num_workers=cfg.get("num_workers", 8), cache_path=cache("test"), tag=f"{name}/{key}/test",
                log_every=100,
            )
            linear = run_linear_probe(
                X_train, y_train, X_test, y_test, num_classes=num_classes,
                lr_grid=record["linear_config"]["lr_grid"],
                epochs=record["linear_config"]["epochs"],
                batch_size=record["linear_config"]["batch_size"],
                weight_decay=record["linear_config"]["weight_decay"],
                momentum=float(linear_cfg.get("momentum", 0.9)),
                standardize=record["linear_config"]["standardize"],
                val_frac=record["linear_config"]["val_frac"],
                device=device, seed=int(cfg.get("seed", 0)),
            )
            logger.info(
                "%s/%s: linear top-1 = %.2f%%  top-5 = %.2f%%  (lr=%g, train top-1 = %.2f%%, %.1f min)",
                name, display, linear["top1"], linear["top5"], linear["best_lr"], linear["train_top1"],
                (time.time() - ds_started) / 60,
            )
            record["datasets"][key] = {
                "name": display,
                "num_classes": num_classes,
                "n_train": int(X_train.shape[0]),
                "n_test": int(X_test.shape[0]),
                "top1": linear["top1"],
                "top5": linear["top5"],
                "train_top1": linear["train_top1"],
                "best_lr": linear["best_lr"],
                "search": linear["search"],
                "runtime_min": (time.time() - ds_started) / 60,
            }
            record["avg_top1"] = _average(record, keys)
            _save_backbone(name, record, out_dir)
            del X_train, X_test

        record["runtime_min"] = (previous or {}).get("runtime_min", 0.0) + (time.time() - started) / 60
        record["avg_top1"] = _average(record, keys)
        _save_backbone(name, record, out_dir)
        del extract_fn
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results = _collect_results(out_dir, config_order)
    _print_table(results, keys)
    _write_results(results, out_dir, keys)
    logger.info("Results written to %s", out_dir)
    if failures:
        logger.error("%d failure(s): %s", len(failures), ", ".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BundLeJEPA transfer linear-probing evaluation (SPHERE-JEPA Table 1)")
    parser.add_argument("--config", type=str, default="configs/eval/transfer_in1k_ep100.yaml")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated backbone names to run (default: all in the config).")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset keys to run (default: all in the config).")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip evaluation; just rebuild results.json/csv from per-backbone files.")
    parser.add_argument("--force", action="store_true", help="Recompute datasets that already have results.")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    main(
        config,
        [n.strip() for n in args.only.split(",")] if args.only else None,
        [d.strip() for d in args.datasets.split(",")] if args.datasets else None,
        merge_only=args.merge_only,
        force=args.force,
    )
