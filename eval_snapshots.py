# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Learning-curve evaluation over periodic training snapshots.

For each ``snap_epoch*.ckpt`` under a run's ``snapshots/`` directory: rebuild the
frozen encoder, extract features on the held-out split, and record

  * ``knn_loo_top1`` -- leave-one-out cosine k-NN accuracy (representation
    separability, decoupled from the online probe's own optimisation)
  * ``orc_gap``      -- within- minus cross-class Ollivier-Ricci curvature gap
    (:func:`src.eval.embedor.curvature_gap`)

so the k-sweep's "probe learns faster as k grows" observation can be quantified
(area under the curve, epochs-to-threshold) with seed error bars.

    python eval_snapshots.py --config configs/eval_snapshots_ksweep.yaml
    python eval_snapshots.py --config configs/eval_snapshots_ksweep.yaml --only ball_l01_k8_s0
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
import re
import sys
import time

import numpy as np
import torch
import yaml

from src.eval.embedor import curvature_gap
from src.eval.extract import load_backbone
from src.eval.indomain import extract_classification_features, get_indomain_splits

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()

_EPOCH_RE = re.compile(r"epoch(\d+)")


def _knn_loo_top1(X: np.ndarray, y: np.ndarray, k: int = 20) -> float:
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X)
    _, idx = nn.kneighbors(X)
    pred = np.array([np.bincount(y[row]).argmax() for row in idx[:, 1:]])
    return float((pred == y).mean())


def _standardize(X: np.ndarray) -> np.ndarray:
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (X - X.mean(axis=0, keepdims=True)) / sd


def _snapshots(run_dir: str, stride: int) -> list[tuple[int, str]]:
    found = []
    for path in glob.glob(os.path.join(run_dir, "snapshots", "snap_epoch*.ckpt")):
        m = _EPOCH_RE.search(os.path.basename(path))
        if m:
            found.append((int(m.group(1)), path))
    found.sort()
    return found[:: max(stride, 1)]


def main(cfg: dict, only: list[str] | None) -> None:
    device = torch.device(cfg.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    crop_size = int(cfg.get("crop_size", 128))
    out_dir = cfg["output_folder"]
    cache_dir = cfg.get("cache_dir")
    os.makedirs(out_dir, exist_ok=True)

    n_neighbors = int(cfg.get("n_neighbors", 15))
    knn_k = int(cfg.get("knn_k", 20))
    n_jobs = int(cfg.get("n_jobs", 8))
    stride = int(cfg.get("epoch_stride", 1))

    _, val_ds, num_classes = get_indomain_splits(
        dataset=cfg.get("dataset", "imagenette"), data_dir=cfg["data_dir"],
        crop_size=crop_size, imagenette_variant=cfg.get("imagenette_variant", "160px"),
    )
    dataset = {"val": val_ds}[cfg.get("split", "val")]
    logger.info("Dataset %s | %d imgs | %d classes", cfg.get("dataset"), len(dataset), num_classes)

    all_rows: list[dict] = []
    for run in cfg["runs"]:
        name = run["name"]
        if only and name not in only:
            continue
        snaps = _snapshots(run["dir"], stride)
        if not snaps:
            logger.warning("%s: no snapshots under %s/snapshots -- skipping.", name, run["dir"])
            continue
        logger.info("=== %s: %d snapshots (epochs %d..%d) ===",
                    name, len(snaps), snaps[0][0], snaps[-1][0])

        run_rows: list[dict] = []
        for epoch, ckpt in snaps:
            started = time.time()
            try:
                extract_fn, _ = load_backbone(
                    experiment=run["experiment"], checkpoint_path=ckpt, device=device,
                    crop_size=crop_size, feature="encoder",
                )
            except Exception:
                logger.exception("%s ep%d: load failed -- skipping snapshot.", name, epoch)
                continue

            cache_path = (
                os.path.join(cache_dir, name, f"val_{crop_size}_ep{epoch}.pt") if cache_dir else None
            )
            X, y = extract_classification_features(
                extract_fn, dataset, batch_size=cfg.get("batch_size", 256),
                num_workers=cfg.get("num_workers", 8), cache_path=cache_path,
                tag=f"{name}/ep{epoch}", log_every=0,
            )
            X = X.float().numpy()
            y = y.numpy().astype(int)
            if cfg.get("standardize", True):
                X = _standardize(X)

            knn = _knn_loo_top1(X, y, knn_k)
            cg = curvature_gap(X, y, n_neighbors=n_neighbors, n_jobs=n_jobs)
            row = {"run": name, "experiment": run["experiment"], "epoch": epoch,
                   "knn_loo_top1": round(knn, 5),
                   **{k: round(v, 5) if isinstance(v, float) else v for k, v in cg.items()}}
            run_rows.append(row)
            all_rows.append(row)
            logger.info("%s ep%-4d | knn=%.3f orc_gap=%.3f comps=%d (%.0fs)",
                        name, epoch, knn, cg["orc_gap"], cg["n_components"], time.time() - started)

            del extract_fn
            if device.type == "cuda":
                torch.cuda.empty_cache()

        _write_csv(run_rows, os.path.join(out_dir, f"{name}.csv"))

    if all_rows:
        _write_csv(all_rows, os.path.join(out_dir, "learning_curves.csv"))
    logger.info("Wrote %d rows to %s", len(all_rows), out_dir)


def _write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["run", "experiment", "epoch", "knn_loo_top1", "orc_within", "orc_cross",
            "orc_gap", "edge_cross_frac", "n_edges", "n_components"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learning-curve evaluation over training snapshots")
    parser.add_argument("--config", type=str, default="configs/eval_snapshots_ksweep.yaml")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated run names.")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    main(config, [n.strip() for n in args.only.split(",")] if args.only else None)
