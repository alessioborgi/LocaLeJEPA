# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""RankMe (Garrido et al., ICML 2023) of the pretrained JEPA backbones.

For every backbone the effective rank of its frozen features is measured,
label-free, on 25,600 ImageNet-1K train images (the paper's source-dataset
protocol, Appendix G) and — like the paper's Tables S20-S25 — on the train
split of each transfer dataset. The headline number is RankMe of the 512-d
encoder output (the *representation* the k-NN / linear probes consume); RankMe
of the 16-d projector output (*embedding*, what the paper measures) is reported
alongside but is capped at 16 by our projector width.

Features are reused from the in-domain / transfer evaluation caches where they
exist (no GPU needed); backbones without a cache are embedded on the fly and
the 25,600-image subset is cached under ``cache_dir``. Results go to
``<output_folder>/per_backbone/<name>.json`` (one file per backbone, so
per-backbone Slurm tasks never clobber each other).

Usage
-----
    python src/eval_rankme.py --config configs/eval/rankme_in1k_ep100.yaml
    python src/eval_rankme.py --config ... --only hvit_sigreg,d4rn50_sigreg
    python src/eval_rankme.py --config ... --cached-only      # skip backbones needing a GPU
    python src/eval_rankme.py --config ... --merge-only
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
from torch.utils.data import Subset
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.eval.extract import load_backbone  # noqa: E402
from src.eval.indomain import extract_classification_features, get_indomain_splits  # noqa: E402
from src.eval.rankme import load_projector, project_features, rankme, subsample_indices, subsample_rows  # noqa: E402
from src.eval.transfer import TRANSFER_DATASETS, get_transfer_splits  # noqa: E402

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()


def _per_backbone_dir(out_dir: str) -> str:
    path = os.path.join(out_dir, "per_backbone")
    os.makedirs(path, exist_ok=True)
    return path


def _save_backbone(name: str, entry: dict, out_dir: str) -> None:
    path = os.path.join(_per_backbone_dir(out_dir), f"{name}.json")
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


def _write_results(results: dict, out_dir: str, dataset_keys: list[str]) -> None:
    if not results:
        return
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    with open(os.path.join(out_dir, "results.csv"), "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        header = ["backbone", "experiment", "checkpoint", "epoch", "feat_dim", "proj_dim", "n_source",
                  "rankme_repr", "classical_rank_repr", "rankme_repr_centered", "rankme_emb", "classical_rank_emb"]
        header += [f"rankme_repr_{k}" for k in dataset_keys] + [f"rankme_emb_{k}" for k in dataset_keys]
        writer.writerow(header)
        for name, entry in results.items():
            src = entry["source"]
            row = [name, entry["experiment"], entry["checkpoint"], entry.get("epoch"), entry["feat_dim"],
                   entry.get("proj_dim"), src["repr"]["n"],
                   f'{src["repr"]["rankme"]:.2f}', src["repr"]["classical_rank"],
                   f'{src["repr_centered"]["rankme"]:.2f}',
                   f'{src["emb"]["rankme"]:.2f}' if src.get("emb") else "",
                   src["emb"]["classical_rank"] if src.get("emb") else ""]
            tgt = entry.get("targets", {})
            row += [f'{tgt[k]["repr"]["rankme"]:.2f}' if k in tgt else "" for k in dataset_keys]
            row += [f'{tgt[k]["emb"]["rankme"]:.2f}' if k in tgt and tgt[k].get("emb") else "" for k in dataset_keys]
            writer.writerow(row)


def _print_table(results: dict, dataset_keys: list[str]) -> None:
    print("\n" + "=" * (46 + 9 * (4 + len(dataset_keys))))
    print("RankMe on 512-d representations (25,600 ImageNet-1K train images) + per-target-dataset RankMe")
    print("=" * (46 + 9 * (4 + len(dataset_keys))))
    head = f'{"backbone":30s} {"ep":>4s} {"RankMe":>8s} {"rank":>6s} {"emb":>6s} ' + " ".join(f"{k[:8]:>8s}" for k in dataset_keys)
    print(head)
    print("-" * len(head))
    for name, entry in results.items():
        src = entry["source"]
        emb = f'{src["emb"]["rankme"]:6.2f}' if src.get("emb") else "     -"
        tgt = entry.get("targets", {})
        cells = " ".join(f'{tgt[k]["repr"]["rankme"]:8.1f}' if k in tgt else f'{"-":>8s}' for k in dataset_keys)
        print(f'{name:30s} {str(entry.get("epoch", "?")):>4s} {src["repr"]["rankme"]:8.2f} '
              f'{src["repr"]["classical_rank"]:6d} {emb} {cells}')
    print("=" * len(head))


def _strip_spectrum(block: dict, keep: bool) -> dict:
    return block if keep else {k: v for k, v in block.items() if k != "spectrum"}


def _load_cached(path: str | None) -> torch.Tensor | None:
    if path and os.path.isfile(path):
        blob = torch.load(path, map_location="cpu")
        logger.info("loaded cached features %s from %s", tuple(blob["X"].shape), path)
        return blob["X"]
    return None


def main(cfg: dict, only: list[str] | None, merge_only: bool = False, cached_only: bool = False) -> None:
    device = torch.device(cfg.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    crop_size = int(cfg.get("crop_size", 128))
    out_dir = cfg.get("output_folder", "logs/eval_rankme")
    os.makedirs(out_dir, exist_ok=True)
    n_samples = int(cfg.get("n_samples", 25600))
    seed = int(cfg.get("seed", 0))
    dataset_name = cfg.get("dataset", "imagenet1k")
    transfer_keys = list(cfg.get("transfer_datasets", []))
    config_order = [entry["name"] for entry in cfg["backbones"]]

    if merge_only:
        merged = _collect_results(out_dir, config_order)
        _print_table(merged, transfer_keys)
        _write_results(merged, out_dir, transfer_keys)
        logger.info("Merged %d backbone result files into %s", len(merged), out_dir)
        return

    indomain_cache = cfg.get("indomain_cache_dir")
    transfer_cache = cfg.get("transfer_cache_dir")
    cache_dir = cfg.get("cache_dir")
    keep_spectrum = bool(cfg.get("save_spectrum", True))

    train_ds = None  # ImageFolder built lazily: only needed when a backbone has no cached features.
    failures: list[str] = []

    for entry in cfg["backbones"]:
        name = entry["name"]
        if only and name not in only:
            continue
        if not os.path.isfile(entry["checkpoint"]):
            logger.error("%s: checkpoint %s not found — skipping.", name, entry["checkpoint"])
            continue
        epoch = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False, mmap=True).get("epoch")
        logger.info("=== backbone: %s (%s, epoch %s) ===", name, entry["checkpoint"], epoch)
        started = time.time()

        # ---- source (pretraining) dataset features: cache -> subset cache -> GPU extraction
        full_cache = os.path.join(indomain_cache, name, f"{dataset_name}_train_{crop_size}_ep{epoch}.pt") if indomain_cache else None
        subset_cache = os.path.join(cache_dir, name, f"{dataset_name}_train{n_samples}_{crop_size}_ep{epoch}.pt") if cache_dir else None
        X_src = _load_cached(subset_cache)
        source_from = "subset_cache"
        if X_src is None:
            X_full = _load_cached(full_cache)
            if X_full is not None:
                X_src = subsample_rows(X_full, n_samples, seed)
                source_from = "indomain_cache"
                del X_full
        extract_fn = None
        if X_src is None:
            if cached_only:
                logger.warning("%s: no cached %s features and --cached-only set — skipping.", name, dataset_name)
                continue
            try:
                extract_fn, _ = load_backbone(
                    experiment=entry["experiment"], checkpoint_path=entry["checkpoint"], device=device,
                    crop_size=crop_size, overrides=entry.get("overrides"),
                )
            except Exception:
                logger.exception("%s: failed to load backbone — skipping.", name)
                failures.append(name)
                continue
            if train_ds is None:
                train_ds, _, _ = get_indomain_splits(
                    dataset=dataset_name, data_dir=cfg["data_dir"], crop_size=crop_size,
                    image_folder=cfg.get("image_folder", "imagenet_full_size/061417/"),
                )
            # Same rows subsample_rows() takes from the full cache: identical subset either way.
            subset = Subset(train_ds, subsample_indices(len(train_ds), n_samples, seed))
            X_src, _ = extract_classification_features(
                extract_fn, subset, batch_size=cfg.get("batch_size", 256), num_workers=cfg.get("num_workers", 8),
                cache_path=subset_cache, tag=f"{name}/{dataset_name}/train{n_samples}", log_every=20,
            )
            source_from = "extracted"

        # ---- projector -> 16-d embeddings (what the paper measures)
        projector, rectify = None, False
        try:
            projector, rectify = load_projector(
                entry["experiment"], entry["checkpoint"], crop_size=crop_size, overrides=entry.get("overrides"),
            )
        except Exception:
            logger.exception("%s: could not rebuild the projector — embedding RankMe omitted.", name)

        def both(X: torch.Tensor) -> dict:
            out = {"repr": _strip_spectrum(rankme(X), keep_spectrum),
                   "repr_centered": _strip_spectrum(rankme(X, center=True), False)}
            if projector is not None:
                out["emb"] = _strip_spectrum(rankme(project_features(projector, X, rectify)), keep_spectrum)
            return out

        source = both(X_src)
        source["from"] = source_from
        logger.info("%s: %s RankMe(repr) = %.2f (classical %d, centered %.2f)%s", name, dataset_name,
                    source["repr"]["rankme"], source["repr"]["classical_rank"], source["repr_centered"]["rankme"],
                    f', RankMe(emb) = {source["emb"]["rankme"]:.2f}' if "emb" in source else "")

        # ---- target datasets (transfer train splits)
        targets = {}
        for key in transfer_keys:
            path = os.path.join(transfer_cache, name, f"{key}_train_{crop_size}_ep{epoch}.pt") if transfer_cache else None
            X_t = _load_cached(path)
            if X_t is None:
                if cached_only or not cfg.get("transfer_data_root"):
                    logger.warning("%s/%s: no cached features — skipping this target.", name, key)
                    continue
                if extract_fn is None:
                    try:
                        extract_fn, _ = load_backbone(
                            experiment=entry["experiment"], checkpoint_path=entry["checkpoint"], device=device,
                            crop_size=crop_size, overrides=entry.get("overrides"),
                        )
                    except Exception:
                        logger.exception("%s: failed to load backbone — targets skipped.", name)
                        break
                try:
                    ds_train, _, _ = get_transfer_splits(key, cfg["transfer_data_root"], crop_size, download=False)
                except Exception:
                    logger.exception("%s/%s: dataset unavailable — skipping.", name, key)
                    continue
                try:
                    X_t, _ = extract_classification_features(
                        extract_fn, ds_train, batch_size=cfg.get("batch_size", 256), num_workers=cfg.get("num_workers", 8),
                        cache_path=path, tag=f"{name}/{key}/train", log_every=50,
                    )
                except Exception:
                    # e.g. image files missing under an intact metadata layout (netscratch sweep).
                    logger.exception("%s/%s: feature extraction failed — skipping this target.", name, key)
                    continue
            X_t = subsample_rows(X_t, n_samples, seed)
            targets[key] = both(X_t)
            targets[key]["n_total"] = int(X_t.shape[0])
            logger.info("%s/%s: RankMe(repr) = %.2f (n=%d)", name, key, targets[key]["repr"]["rankme"], X_t.shape[0])

        record = {
            "experiment": entry["experiment"],
            "checkpoint": entry["checkpoint"],
            "epoch": epoch,
            "family": entry.get("family"),
            "feat_dim": int(X_src.shape[1]),
            "proj_dim": int(source["emb"]["dim"]) if "emb" in source else None,
            "rectified_embeddings": rectify,
            "source_dataset": dataset_name,
            "crop_size": crop_size,
            "n_samples": n_samples,
            "seed": seed,
            "source": source,
            "targets": targets,
            "runtime_min": (time.time() - started) / 60,
        }
        _save_backbone(name, record, out_dir)
        logger.info("%s: done in %.1f min", name, record["runtime_min"])

    merged = _collect_results(out_dir, config_order)
    _print_table(merged, transfer_keys)
    _write_results(merged, out_dir, transfer_keys)
    if failures:
        logger.error("Failed backbones: %s", ", ".join(failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--only", default=None, help="Comma-separated backbone names to evaluate.")
    parser.add_argument("--cached-only", action="store_true", help="Only use cached features (no GPU extraction).")
    parser.add_argument("--merge-only", action="store_true", help="Rebuild results.{json,csv} from per_backbone/.")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    main(config, [n for n in args.only.split(",") if n] if args.only else None, args.merge_only, args.cached_only)
