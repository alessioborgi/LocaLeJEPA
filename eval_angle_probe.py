# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Linear-probe comparison of frozen backbones on the bobblehead rotation angles.

For each frozen checkpoint we extract 512-d features and fit a linear probe that
predicts ``(cos θ, sin θ)`` of the two rotation angles (Yoda, bulldog).  We
report held-out test R² and mean angular error, and render comparison figures.

Usage
-----
    python eval_angle_probe.py --config configs/eval_angle_probe.yaml
"""

import argparse
import csv
import json
import logging
import os
import sys

import numpy as np
import torch
import yaml

from src.data.bobblehead_datamodule import build_eval_transform, get_bobblehead_dataset
from src.eval.angle_probe import circular_linear_probe, pca_2d
from src.eval.extract import extract_features, load_backbone

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()

ANGLE_NAMES = ["angle0", "angle1"]


def _plots(per_backbone, out_dir, angle_labels):
    """Comparison + per-backbone qualitative figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logger.warning("matplotlib unavailable, skipping plots: %s", exc)
        return

    os.makedirs(out_dir, exist_ok=True)
    names = list(per_backbone.keys())
    n_ang = max(len(b["metrics"]) for b in per_backbone.values())
    ang_keys = [f"angle{k}" for k in range(n_ang)]
    x = np.arange(len(names))
    w = 0.8 / n_ang

    # (1) Headline bar chart: test mean angular error per backbone × angle.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j, ak in enumerate(ang_keys):
        vals = [per_backbone[n]["metrics"][ak]["test"]["mae_deg"] for n in names]
        bars = ax.bar(x + j * w, vals, w, label=angle_labels[j])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + w * (n_ang - 1) / 2)
    ax.set_xticklabels(names)
    ax.set_ylabel("test mean angular error (deg)  ↓ better")
    ax.set_title("Linear probe: rotation-angle error per backbone")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compare_angular_error.png"), dpi=130)
    plt.close(fig)

    # (2) Test R² per backbone × angle.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j, ak in enumerate(ang_keys):
        vals = [per_backbone[n]["metrics"][ak]["test"]["r2"] for n in names]
        bars = ax.bar(x + j * w, vals, w, label=angle_labels[j])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + w * (n_ang - 1) / 2)
    ax.set_xticklabels(names)
    ax.set_ylabel(r"test $(\cos,\sin)$ $R^2$  ↑ better")
    ax.set_title("Linear probe: rotation-angle $R^2$ per backbone")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "compare_r2.png"), dpi=130)
    plt.close(fig)

    # (3) True-vs-predicted angle scatter grid (rows=backbone, cols=angle).
    fig, axes = plt.subplots(len(names), n_ang, figsize=(4 * n_ang, 3.6 * len(names)), squeeze=False)
    for i, n in enumerate(names):
        for j, ak in enumerate(ang_keys):
            v = per_backbone[n]["viz"][ak]
            axx = axes[i][j]
            axx.scatter(np.degrees(v["true"]), np.degrees(v["pred"]), s=3, alpha=0.4)
            axx.plot([-180, 180], [-180, 180], "r--", lw=1)
            mae = per_backbone[n]["metrics"][ak]["test"]["mae_deg"]
            axx.set_title(f"{n} · {angle_labels[j]} (MAE={mae:.1f}°)", fontsize=9)
            axx.set_xlabel("true angle (deg)")
            axx.set_ylabel("pred angle (deg)")
    fig.suptitle("Linear probe: true vs predicted angle (held-out test)", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "true_vs_pred.png"), dpi=130)
    plt.close(fig)

    # (4) PCA(2) of features coloured by each true angle (qualitative structure).
    fig, axes = plt.subplots(len(names), n_ang, figsize=(4.5 * n_ang, 4 * len(names)), squeeze=False)
    for i, n in enumerate(names):
        P = per_backbone[n]["pca"]
        ang = per_backbone[n]["angles_full"]
        for j in range(n_ang):
            axx = axes[i][j]
            sc = axx.scatter(P[:, 0], P[:, 1], c=ang[:, j], s=4, cmap="twilight")
            axx.set_title(f"{n} · PCA coloured by {angle_labels[j]}", fontsize=9)
            axx.set_xlabel("PC1")
            axx.set_ylabel("PC2")
            fig.colorbar(sc, ax=axx, fraction=0.046)
    fig.suptitle("Feature-space PCA coloured by true rotation angle", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_pca.png"), dpi=130)
    plt.close(fig)

    logger.info("Figures written to %s", out_dir)


def main(cfg: dict) -> None:
    device = torch.device(cfg.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    crop_size = cfg.get("crop_size", 128)
    out_dir = cfg.get("output_folder", "logs/eval_angle_probe")
    os.makedirs(out_dir, exist_ok=True)
    angle_labels = cfg.get("angle_labels", ["angle 0", "angle 1"])

    dataset = get_bobblehead_dataset(
        data_dir=cfg.get("data_dir", "data"),
        crop_size=crop_size,
        transform=build_eval_transform(crop_size),
        subsample=cfg.get("subsample", 2),
        max_frames=cfg.get("max_frames"),
    )
    logger.info("Dataset: %d frames | labels available: %s", len(dataset), dataset.has_labels)
    if not dataset.has_labels:
        raise RuntimeError("Linear angle probe requires labels (labels.npy) — none found.")

    probe_kwargs = cfg.get("probe", {})
    per_backbone = {}

    for entry in cfg["backbones"]:
        name = entry["name"]
        logger.info("=== backbone: %s ===", name)
        extract_fn, feat_dim = load_backbone(
            experiment=entry["experiment"], checkpoint_path=entry["checkpoint"],
            device=device, crop_size=crop_size,
        )
        cache_path = os.path.join(cfg["cache_dir"], f"{name}_features.pt") if cfg.get("cache_dir") else None
        X, y = extract_features(
            extract_fn, dataset, batch_size=cfg.get("batch_size", 256),
            num_workers=cfg.get("num_workers", 8), device=device, cache_path=cache_path,
        )
        Xn, angles = X.numpy(), y.numpy()
        if angles.ndim == 1:
            angles = angles[:, None]

        metrics, viz = circular_linear_probe(Xn, angles, **probe_kwargs)
        for k, ak in enumerate([f"angle{i}" for i in range(angles.shape[1])]):
            t = metrics[ak]["test"]
            logger.info("%s · %s: test R²=%.3f  MAE=%.1f°  (median %.1f°, ridge=%g)",
                        name, angle_labels[k] if k < len(angle_labels) else ak,
                        t["r2"], t["mae_deg"], t["median_deg"], metrics[ak]["ridge"])

        per_backbone[name] = {
            "feat_dim": feat_dim,
            "metrics": metrics,
            "viz": viz,
            "pca": pca_2d(Xn),
            "angles_full": angles,
        }
        del extract_fn
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _plots(per_backbone, out_dir, angle_labels)

    # ── summary table ────────────────────────────────────────────────────────
    n_ang = max(len(b["metrics"]) for b in per_backbone.values())
    print("\n" + "=" * 78)
    header = f'{"Backbone":16s}' + "".join(
        f'  {("a"+str(k)+" R2"):>8s}  {("a"+str(k)+" MAE°"):>9s}' for k in range(n_ang)
    )
    print(header)
    print("-" * 78)
    for name, b in per_backbone.items():
        row = f"{name:16s}"
        for k in range(n_ang):
            t = b["metrics"][f"angle{k}"]["test"]
            row += f'  {t["r2"]:8.3f}  {t["mae_deg"]:9.1f}'
        print(row)
    print("=" * 78)

    # ── persist ──────────────────────────────────────────────────────────────
    summary = {
        name: {"feat_dim": b["feat_dim"],
               "metrics": {ak: b["metrics"][ak] for ak in b["metrics"]}}
        for name, b in per_backbone.items()
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        head = ["backbone"]
        for k in range(n_ang):
            head += [f"angle{k}_test_r2", f"angle{k}_test_mae_deg", f"angle{k}_test_median_deg", f"angle{k}_ridge"]
        w.writerow(head)
        for name, b in per_backbone.items():
            row = [name]
            for k in range(n_ang):
                m = b["metrics"][f"angle{k}"]
                row += [f'{m["test"]["r2"]:.6f}', f'{m["test"]["mae_deg"]:.4f}',
                        f'{m["test"]["median_deg"]:.4f}', m["ridge"]]
            w.writerow(row)
    logger.info("Results written to %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BundLeJEPA linear angle probe")
    parser.add_argument("--config", type=str, default="configs/eval_angle_probe.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
