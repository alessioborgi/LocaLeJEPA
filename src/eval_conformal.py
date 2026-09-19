# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Conformal-Autoencoder dimension detection on frozen JEPA features.

Replicates the bobblehead experiment of Kevrekidis et al., "Thinner Latent
Spaces" (arXiv:2408.16138, §3.3) but swaps the paper's PCA-60 front-end for a
**frozen pretrained JEPA backbone** as the fixed feature extractor:

    images --(frozen backbone)--> features --(Conformal Autoencoder)--> latent

The CAE (Algorithm 1 / Eq. 6) over-specifies its latent layer and regularises
the latent-gradient orthogonality, so redundant coordinates collapse and the
number of *active* latent coordinates reveals the intrinsic dimension of the
image manifold.  The two figures rotate independently, tracing a 2-torus
(minimally embedded in R^4), so we expect ~4 active coordinates.

The script loops over several frozen checkpoints (e.g. SphereJEPA+SUSReg,
nViT+SIGReg, nViT+SubReg) so their representations can be compared on this task.

Usage
-----
    python src/eval_conformal.py --config configs/eval/conformal.yaml
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
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.bobblehead_datamodule import build_eval_transform, get_bobblehead_dataset  # noqa: E402
from src.eval.conformal_autoencoder import (  # noqa: E402
    CAEConfig,
    circular_recovery,
    train_conformal_autoencoder,
)
from src.eval.extract import extract_features, load_backbone  # noqa: E402

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger()


def _maybe_plot(name, result, angles, out_dir):
    """Save the paper-style diagnostic figures (best-effort; needs matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logger.warning("matplotlib unavailable, skipping plots: %s", exc)
        return

    os.makedirs(out_dir, exist_ok=True)

    # (1) Gradient-norm trajectory per latent (cf. Fig. 1a / mode collapse).
    fig, ax = plt.subplots(figsize=(6, 4))
    for j in range(result.grad_norm_history.shape[1]):
        ax.plot(result.epochs_logged, result.grad_norm_history[:, j], label=f"ν{j}")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\mathbb{E}_x\,\|\nabla\nu_i\|$")
    ax.set_title(f"{name}: latent-gradient norms (active = {result.inferred_dim})")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_gradnorm.png"), dpi=120)
    plt.close(fig)

    # (2) Per-latent variance bar (cf. Fig. 11b).
    fig, ax = plt.subplots(figsize=(6, 4))
    var = np.clip(result.latent_variances, 1e-12, None)
    ax.bar(range(len(var)), np.log10(var))
    ax.set_xlabel("latent component index")
    ax.set_ylabel(r"$\log_{10}$ Var$(\nu_i)$")
    ax.set_title(f"{name}: latent variances")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_variance.png"), dpi=120)
    plt.close(fig)

    # (3) Torus scatter: first two active latents, coloured by a true angle / time.
    active = result.active_components
    if len(active) >= 2:
        color = angles[:, 0] if (angles is not None and angles.ndim == 2) else (
            angles if angles is not None else np.arange(result.latents.shape[0])
        )
        fig, ax = plt.subplots(figsize=(5, 5))
        sc = ax.scatter(result.latents[:, active[0]], result.latents[:, active[1]],
                        c=color, s=4, cmap="twilight")
        ax.set_xlabel(f"ν{active[0]}")
        ax.set_ylabel(f"ν{active[1]}")
        ax.set_title(f"{name}: latent (first 2 active)")
        fig.colorbar(sc, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{name}_latent_scatter.png"), dpi=120)
        plt.close(fig)


def main(cfg: dict) -> None:
    device = torch.device(cfg.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    crop_size = cfg.get("crop_size", 128)
    data_dir = cfg.get("data_dir", "data")
    out_dir = cfg.get("output_folder", "logs/eval_conformal")
    os.makedirs(out_dir, exist_ok=True)

    logger.info("Loading bobblehead frames from %s", data_dir)
    dataset = get_bobblehead_dataset(
        data_dir=data_dir,
        crop_size=crop_size,
        transform=build_eval_transform(crop_size),
        labels_name=cfg.get("labels_name"),
        max_frames=cfg.get("max_frames"),
        subsample=cfg.get("subsample", 1),
    )
    logger.info("Dataset: %d frames | labels available: %s", len(dataset), dataset.has_labels)

    cae_kwargs = cfg.get("cae", {})
    results = {}

    for entry in cfg["backbones"]:
        name = entry["name"]
        logger.info("=== backbone: %s ===", name)

        extract_fn, feat_dim = load_backbone(
            experiment=entry["experiment"],
            checkpoint_path=entry["checkpoint"],
            device=device,
            crop_size=crop_size,
        )
        logger.info("%s: feature dim = %d", name, feat_dim)

        cache_path = None
        if cfg.get("cache_dir"):
            cache_path = os.path.join(cfg["cache_dir"], f"{name}_features.pt")
        X, y = extract_features(
            extract_fn, dataset,
            batch_size=cfg.get("batch_size", 256),
            num_workers=cfg.get("num_workers", 8),
            device=device,
            cache_path=cache_path,
        )
        logger.info("%s: features %s", name, tuple(X.shape))

        cae_cfg = CAEConfig(**cae_kwargs)
        _, result = train_conformal_autoencoder(X, cae_cfg, device)
        logger.info(
            "%s: inferred dim = %d (active %s) | final recon = %.3e",
            name, result.inferred_dim, result.active_components, result.final_recon,
        )

        # Optional quantitative recovery of rotation angles (radians).
        recovery = None
        angles = None
        if dataset.has_labels:
            angles = y.numpy()
            recovery = circular_recovery(result.latents, angles, result.active_components)
            logger.info("%s: angle recovery = %s", name, recovery)

        _maybe_plot(name, result, angles, out_dir)

        results[name] = {
            "feat_dim": feat_dim,
            "inferred_dim": result.inferred_dim,
            "active_components": result.active_components,
            "grad_norms": result.grad_norms.tolist(),
            "latent_variances": result.latent_variances.tolist(),
            "final_recon": result.final_recon,
            "recovery": recovery,
        }
        # Free GPU memory between backbones.
        del extract_fn
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f'{"Backbone":28s}  {"feat":>5s}  {"inferred dim":>12s}  {"recon":>10s}')
    print("-" * 64)
    for name, r in results.items():
        print(f'{name:28s}  {r["feat_dim"]:5d}  {r["inferred_dim"]:12d}  {r["final_recon"]:10.2e}')
    print("=" * 64)

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["backbone", "feat_dim", "inferred_dim", "final_recon"])
        for name, r in results.items():
            w.writerow([name, r["feat_dim"], r["inferred_dim"], f'{r["final_recon"]:.6e}'])
    logger.info("Results written to %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LocaLeJEPA conformal-autoencoder dimension detection")
    parser.add_argument("--config", type=str, default="configs/eval/conformal.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
