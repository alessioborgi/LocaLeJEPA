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
    python src/eval_indomain.py --config configs/eval/indomain_in1k.yaml
    python src/eval_indomain.py --config configs/eval/indomain_in1k.yaml --only hvit_sigreg
"""

import argparse
import csv
import json
import logging
import os
import sys
import time

import rootutils
import torch
import yaml

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.eval.extract import load_backbone  # noqa: E402
from src.eval.indomain import extract_classification_features, get_indomain_splits, subsample_per_class
from src.eval.knn_probe import weighted_knn
from src.eval.linear_probe import run_linear_probe
from src.eval.io import atomic_write
from src.eval.io import feature_cache_path

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()


def _per_backbone_dir(out_dir: str) -> str:
    path = os.path.join(out_dir, "per_backbone")
    os.makedirs(path, exist_ok=True)
    return path


def _save_backbone(name: str, entry: dict, out_dir: str) -> None:
    """One file per backbone so concurrent per-backbone jobs never clobber."""
    with atomic_write(os.path.join(_per_backbone_dir(out_dir), f"{name}.json"), "w", encoding="utf-8") as file:
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
    with atomic_write(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    k_report = sorted({int(k) for entry in results.values() for k in entry["knn"]})
    with atomic_write(os.path.join(out_dir, "results.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        header = ["backbone", "experiment", "checkpoint", "epoch", "feat_dim", "n_train", "n_val"]
        header += [f"knn{k}_top1" for k in k_report] + [f"knn{k}_top5" for k in k_report]
        header += ["linear_top1", "linear_top5", "linear_best_lr", "linear_best_wd", "linear_best_epoch", "probe_select", "checkpoint_kind"]
        writer.writerow(header)
        for name, entry in results.items():
            row = [
                name, entry["experiment"], entry["checkpoint"], entry.get("epoch"),
                entry["feat_dim"], entry["n_train"], entry["n_val"],
            ]
            row += [f'{entry["knn"][str(k)]["top1"]:.2f}' if str(k) in entry["knn"] else "" for k in k_report]
            row += [f'{entry["knn"][str(k)]["top5"]:.2f}' if str(k) in entry["knn"] else "" for k in k_report]
            row += [f'{entry["linear"]["top1"]:.2f}', f'{entry["linear"]["top5"]:.2f}', entry["linear"]["best_lr"],
                    entry["linear"].get("best_weight_decay"), entry["linear"].get("best_epoch"),
                    entry.get("probe_select", "train_slice"), entry.get("checkpoint_kind")]
            writer.writerow(row)


def _print_table(results: dict, dataset: str, report_k: int) -> None:
    print("\n" + "=" * 84)
    print(f"In-domain evaluation on {dataset} (SPHERE-JEPA Table 3 protocol)")
    print("=" * 84)
    print(f'{"Backbone":28s} {"ep":>4s} {f"k-NN@{report_k} (%)":>14s} {"Linear (%)":>11s} {"kNN top5":>9s} {"Lin top5":>9s}')
    print("-" * 84)
    for name, entry in results.items():
        # A missing k renders as an em dash. It must NEVER fall back to another k: the previous
    # `.get(k) or next(iter(...))` reported the SMALLEST available k (the dict is built in
    # ascending order) under the requested k's heading, with no marker. Ledger T.suite-P2-7.
        knn = entry["knn"].get(str(report_k))
        top1 = f'{knn["top1"]:14.1f}' if knn else f'{"—":>14s}'
        top5 = f'{knn["top5"]:9.1f}' if knn else f'{"—":>9s}'
        print(
            f'{name:28s} {str(entry.get("epoch", "?")):>4s} {top1} '
            f'{entry["linear"]["top1"]:11.1f} {top5} {entry["linear"]["top5"]:9.1f}'
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
    # Two pipeline discriminators (research_rules/05), written into every result and into the
    # cache key: which representation was probed and which probe protocol was used. "encoder" +
    # "sgd" is ours; "cls2" + "lejepa" is LeJEPA's protocol (campaign decision 3), reported in a
    # separate, labelled column and never mixed with the other.
    feature = str(cfg.get("feature", "encoder"))
    probe = str(linear_cfg.get("probe", "sgd"))
    # a third discriminator (decision 16): how the probe's settings were chosen. "train_slice" is
    # the strict headline; "eval" selects lr x wd x epoch on the reported split and is an upper
    # bound of LeJEPA's published protocol -- reported only in its own labelled column.
    select_on = str(linear_cfg.get("select_on", "train_slice"))
    wd_grid = linear_cfg.get("wd_grid")

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
                feature=feature,
            )
        except Exception:
            # A checkpoint whose architecture has drifted from its config must not
            # take the remaining backbones down with it.
            logger.exception("%s: failed to load backbone — skipping.", name)
            failures.append(name)
            continue
        epoch = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False).get("epoch")
        # which checkpoint of the run this is (last / best / ...): recorded from the file, so a
        # generator labels a column from the data and never from a filename it was handed
        checkpoint_kind = os.path.splitext(os.path.basename(entry["checkpoint"]))[0]
        logger.info("%s: feature dim %d, checkpoint epoch %s (%s)", name, feat_dim, epoch, checkpoint_kind)

        # The key carries a digest of the checkpoint FILE, not just the backbone name and the
        # epoch it reports. Ledger C0-4, reproduced 2026-09-20: two different checkpoints sharing
        # a config `name` and a stored `epoch` collided on one cache file, and the second
        # evaluation silently reported features computed from the first. Built in exactly one
        # place (src/eval/io.py) because eval_rankme reconstructs these same filenames to read
        # them, and a divergence there is a silent skip rather than an error.
        def cache(split: str, _name: str = name, _epoch=epoch,
                  _ckpt: str = entry["checkpoint"]) -> str | None:
            return feature_cache_path(cache_dir, _name, dataset_name, split, crop_size,
                                      _epoch, _ckpt, feature=feature)

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
            device=device, seed=int(cfg.get("seed", 0)), probe=probe,
            select_on=select_on, wd_grid=[float(w) for w in wd_grid] if wd_grid else None,
        )
        logger.info(
            "%s: linear top-1 = %.2f%%  top-5 = %.2f%%  (%s; lr=%g, wd=%g, epoch %s, train top-1 = %.2f%%)",
            name, linear["top1"], linear["top5"], select_on, linear["best_lr"], linear["best_weight_decay"],
            linear["best_epoch"], linear["train_top1"],
        )

        _save_backbone(name, {
            "experiment": entry["experiment"],
            "checkpoint": entry["checkpoint"],
            "epoch": epoch,
            "feat_dim": feat_dim,
            "feature": feature,
            "probe": probe,
            "probe_select": select_on,
            "checkpoint_kind": checkpoint_kind,
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
    parser = argparse.ArgumentParser(description="LocaLeJEPA in-domain k-NN + linear evaluation")
    parser.add_argument("--config", type=str, default="configs/eval/indomain_in1k.yaml")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated backbone names to run (default: all in the config).")
    parser.add_argument("--merge-only", action="store_true",
                        help="Skip evaluation; just rebuild results.json/csv from per-backbone files.")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    main(config, [n.strip() for n in args.only.split(",")] if args.only else None, merge_only=args.merge_only)
