# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""In-domain k-NN + linear evaluation of pretrained JEPA backbones.

Reproduces the evaluation protocol of SPHERE-JEPA (Nicollier et al., 2026)
Table 3: a backbone pretrained on a dataset is frozen and evaluated *on that
same dataset* with the two classical probes,

  * ``k-NN (%)``   -- similarity-weighted k-NN vote over the train feature bank
                      (cosine similarity, exp(sim/T) weights, T = 0.07);
  * ``Linear (%)`` -- a single linear layer trained on the frozen features.

Both read the same features, extracted once per backbone with the deterministic
eval transform and cached to disk.

Usage
-----
    python eval_indomain.py --config configs/eval/indomain_in1k.yaml
    python eval_indomain.py --config configs/eval/indomain_in1k.yaml --only hvit_sigreg
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
from src.eval.indomain import extract_classification_features, get_indomain_splits, subsample_per_class
from src.eval.knn_probe import weighted_knn
from src.eval.linear_probe import run_linear_probe

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()


def _per_backbone_dir(out_dir: str) -> str:
    path = os.path.join(out_dir, "per_backbone")
    os.makedirs(path, exist_ok=True)
    return path


def _save_backbone(name: str, entry: dict, out_dir: str) -> None:
    """One file per backbone so concurrent per-backbone jobs never clobber."""
    with open(os.path.join(_per_backbone_dir(out_dir), f"{name}.json"), "w", encoding="utf-8") as file:
        json.dump(entry, file, indent=2)


def _collect_results(out_dir: str, order: list[str]) -> dict:
    """Rebuild the full result set from the per-backbone files, config order first."""
    folder = _per_backbone_dir(out_dir)
    found = {}
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".json"):
            with open(os.path.join(folder, filename), encoding="utf-8") as file:
                found[filename[: -len(".json")]] = json.load(file)
    ordered = {name: found[name] for name in order if name in found}
    ordered.update({name: entry for name, entry in found.items() if name not in ordered})
    return ordered


def _write_results(results: dict, out_dir: str) -> None:
    if not results:
        return
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    k_report = sorted({int(k) for entry in results.values() for k in entry["knn"]})
    with open(os.path.join(out_dir, "results.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        header = ["backbone", "experiment", "checkpoint", "epoch", "feat_dim", "n_train", "n_val"]
        header += [f"knn{k}_top1" for k in k_report] + [f"knn{k}_top5" for k in k_report]
        header += ["linear_top1", "linear_top5", "linear_best_lr"]
        writer.writerow(header)
        for name, entry in results.items():
            row = [
                name, entry["experiment"], entry["checkpoint"], entry.get("epoch"),
                entry["feat_dim"], entry["n_train"], entry["n_val"],
            ]
            row += [f'{entry["knn"][str(k)]["top1"]:.2f}' if str(k) in entry["knn"] else "" for k in k_report]
            row += [f'{entry["knn"][str(k)]["top5"]:.2f}' if str(k) in entry["knn"] else "" for k in k_report]
            row += [f'{entry["linear"]["top1"]:.2f}', f'{entry["linear"]["top5"]:.2f}', entry["linear"]["best_lr"]]
            writer.writerow(row)


def _print_table(results: dict, dataset: str, report_k: int) -> None:
    print("\n" + "=" * 84)
    print(f"In-domain evaluation on {dataset} (SPHERE-JEPA Table 3 protocol)")
    print("=" * 84)
    print(f'{"Backbone":28s} {"ep":>4s} {f"k-NN@{report_k} (%)":>14s} {"Linear (%)":>11s} {"kNN top5":>9s} {"Lin top5":>9s}')
    print("-" * 84)
    for name, entry in results.items():
        knn = entry["knn"].get(str(report_k)) or next(iter(entry["knn"].values()))
        print(
            f'{name:28s} {str(entry.get("epoch", "?")):>4s} {knn["top1"]:14.1f} '
            f'{entry["linear"]["top1"]:11.1f} {knn["top5"]:9.1f} {entry["linear"]["top5"]:9.1f}'
        )
    print("=" * 84)


def main(cfg: dict, only: list[str] | None, merge_only: bool = False) -> None:
    device = torch.device(cfg.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    crop_size = int(cfg.get("crop_size", 128))
    out_dir = cfg.get("output_folder", "logs/eval_indomain")
    cache_dir = cfg.get("cache_dir")
    os.makedirs(out_dir, exist_ok=True)

    dataset_name = cfg.get("dataset", "imagenet1k")
    report_k_cfg = int(cfg.get("knn", {}).get("report_k", 20))

    if merge_only:
        # Rebuild the summary from per-backbone files after parallel jobs finish.
        merged = _collect_results(out_dir, [entry["name"] for entry in cfg["backbones"]])
        _print_table(merged, dataset_name, report_k_cfg)
        _write_results(merged, out_dir)
        logger.info("Merged %d backbone result files into %s", len(merged), out_dir)
        return

    train_ds, val_ds, num_classes = get_indomain_splits(
        dataset=dataset_name,
        data_dir=cfg["data_dir"],
        crop_size=crop_size,
        image_folder=cfg.get("image_folder", "imagenet_full_size/061417/"),
        imagenette_variant=cfg.get("imagenette_variant", "160px"),
    )

    per_class = cfg.get("train_subset_per_class")
    if per_class:
        full_train = len(train_ds)
        train_ds = subsample_per_class(train_ds, int(per_class), seed=int(cfg.get("seed", 0)))
        logger.warning(
            "Train bank SUBSAMPLED to %d img/class: %d of %d images (%.1f%%). "
            "This departs from the full-train Table 3 protocol.",
            per_class, len(train_ds), full_train, 100.0 * len(train_ds) / full_train,
        )
    logger.info(
        "Dataset %s | train %d | val %d | classes %d | crop %d",
        dataset_name, len(train_ds), len(val_ds), num_classes, crop_size,
    )

    knn_cfg = cfg.get("knn", {})
    linear_cfg = cfg.get("linear", {})

    config_order = [entry["name"] for entry in cfg["backbones"]]
    failures: list[str] = []

    for entry in cfg["backbones"]:
        name = entry["name"]
        if only and name not in only:
            continue
        if not os.path.isfile(entry["checkpoint"]):
            logger.error("%s: checkpoint %s not found — skipping.", name, entry["checkpoint"])
            continue

        logger.info("=== backbone: %s (%s) ===", name, entry["checkpoint"])
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
            # A checkpoint whose architecture has drifted from its config must not
            # take the remaining backbones down with it.
            logger.exception("%s: failed to load backbone — skipping.", name)
            failures.append(name)
            continue
        epoch = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False).get("epoch")
        logger.info("%s: feature dim %d, checkpoint epoch %s", name, feat_dim, epoch)

        # Key the cache on the checkpoint epoch too: a backbone re-evaluated
        # after further pretraining must not silently reuse stale features.
        def cache(split: str, _name: str = name, _epoch=epoch) -> str | None:
            if not cache_dir:
                return None
            return os.path.join(cache_dir, _name, f"{dataset_name}_{split}_{crop_size}_ep{_epoch}.pt")

        X_val, y_val = extract_classification_features(
            extract_fn, val_ds, batch_size=cfg.get("batch_size", 256),
            num_workers=cfg.get("num_workers", 8), cache_path=cache("val"), tag=f"{name}/val",
        )
        X_train, y_train = extract_classification_features(
            extract_fn, train_ds, batch_size=cfg.get("batch_size", 256),
            num_workers=cfg.get("num_workers", 8), cache_path=cache("train"), tag=f"{name}/train",
        )
        logger.info("%s: features extracted in %.1f min", name, (time.time() - started) / 60)

        knn = weighted_knn(
            X_train, y_train, X_val, y_val, num_classes=num_classes,
            k_values=knn_cfg.get("k_values", [10, 20, 100, 200]),
            temperature=float(knn_cfg.get("temperature", 0.07)),
            device=device, chunk_size=int(knn_cfg.get("chunk_size", 128)),
        )
        for k, metrics in knn.items():
            logger.info("%s: k-NN k=%-3d top-1 = %.2f%%  top-5 = %.2f%%", name, k, metrics["top1"], metrics["top5"])

        linear = run_linear_probe(
            X_train, y_train, X_val, y_val, num_classes=num_classes,
            lr_grid=linear_cfg.get("lr_grid", [0.01, 0.03, 0.1, 0.3, 1.0]),
            epochs=int(linear_cfg.get("epochs", 50)),
            batch_size=int(linear_cfg.get("batch_size", 4096)),
            weight_decay=float(linear_cfg.get("weight_decay", 0.0)),
            momentum=float(linear_cfg.get("momentum", 0.9)),
            standardize=bool(linear_cfg.get("standardize", True)),
            val_frac=float(linear_cfg.get("val_frac", 0.02)),
            device=device, seed=int(cfg.get("seed", 0)),
        )
        logger.info(
            "%s: linear top-1 = %.2f%%  top-5 = %.2f%%  (lr=%g, train top-1 = %.2f%%)",
            name, linear["top1"], linear["top5"], linear["best_lr"], linear["train_top1"],
        )

        _save_backbone(name, {
            "experiment": entry["experiment"],
            "checkpoint": entry["checkpoint"],
            "epoch": epoch,
            "feat_dim": feat_dim,
            "dataset": dataset_name,
            "crop_size": crop_size,
            "n_train": int(X_train.shape[0]),
            "n_val": int(X_val.shape[0]),
            "train_subset_per_class": per_class,
            "knn": {str(k): v for k, v in knn.items()},
            "knn_temperature": float(knn_cfg.get("temperature", 0.07)),
            "linear": linear,
            "runtime_min": (time.time() - started) / 60,
        }, out_dir)

        del extract_fn, X_train, X_val
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results = _collect_results(out_dir, config_order)
    _print_table(results, dataset_name, report_k_cfg)
    _write_results(results, out_dir)
    logger.info("Results written to %s", out_dir)
    if failures:
        logger.error("%d backbone(s) failed and are absent from the table: %s", len(failures), ", ".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BundLeJEPA in-domain k-NN + linear evaluation")
    parser.add_argument("--config", type=str, default="configs/eval/indomain_in1k.yaml")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated backbone names to run (default: all in the config).")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip evaluation; just rebuild results.json/csv from per-backbone files.")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    main(config, [n.strip() for n in args.only.split(",")] if args.only else None, merge_only=args.merge_only)
