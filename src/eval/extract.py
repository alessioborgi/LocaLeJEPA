# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Frozen-backbone loading and feature extraction.

The pretrained JEPA encoders are rebuilt from their Hydra *experiment* configs
(exactly as during training, ``src/train.py``) and their weights are loaded from
the Lightning checkpoint (``state_dict`` keys prefixed with ``encoder.``).  The
returned ``extract_fn`` maps a batch of images ``(B, C, H, W)`` to embeddings
``(B, embedding_dim)`` with the backbone kept **frozen** (``eval`` + no grad).

This mirrors the role of the PCA-60 front-end in Kevrekidis et al., "Thinner
Latent Spaces" (arXiv:2408.16138): a fixed, non-trained feature extractor whose
output is then handed to the Conformal Autoencoder.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

ExtractFn = Callable[[torch.Tensor], torch.Tensor]


def load_backbone(
    experiment: str,
    checkpoint_path: str,
    device: torch.device,
    crop_size: int = 128,
    config_dir: Optional[str] = None,
    project_root: Optional[str] = None,
    overrides: Optional[list[str]] = None,
    feature: str = "encoder",
) -> tuple[ExtractFn, int]:
    """Rebuild a pretrained encoder from its experiment config and load weights.

    ``feature="patch"`` returns post-norm spatial ViT tokens, excluding prefix
    tokens and output heads. H-ViT uses its Euclidean body. Only this mode
    interpolates positional embeddings when the requested resolution changes.

    Args:
        experiment: Hydra experiment name, e.g. ``"lejepa/nvit_imagenette_sigreg"``.
        checkpoint_path: Path to a Lightning ``.ckpt`` produced by ``src/train.py``.
        device: Device to place the frozen encoder on.
        crop_size: Input resolution the encoder was trained with (128 here).
        config_dir: Absolute path to the ``configs`` directory. Defaults to
            ``<project_root>/configs``.
        project_root: Project root used to resolve ``${oc.env:PROJECT_ROOT}`` in
            the Hydra ``paths`` config. Defaults to two levels above this file.
        overrides: Extra Hydra overrides applied on top of the experiment, e.g.
            ``["model.encoder.pool=group"]``. Needed when the encoder config has
            moved on since the checkpoint was trained, so the rebuilt module must
            be pinned to the architecture variant the weights belong to.

    Returns:
        ``(extract_fn, embedding_dim)`` where ``extract_fn(images) -> features``.
    """
    # Imported lazily so the rest of the package does not hard-depend on Hydra.
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if config_dir is None:
        config_dir = os.path.join(project_root, "configs")
    # paths/default.yaml resolves ${oc.env:PROJECT_ROOT}; set it if absent.
    os.environ.setdefault("PROJECT_ROOT", project_root)

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=os.path.abspath(config_dir)):
        cfg = compose(
            config_name="train",
            overrides=[f"experiment={experiment}", f"data.crop_size={crop_size}", *(overrides or [])],
        )
    if feature not in ("encoder", "patch"):
        raise ValueError(f"Unknown feature mode: {feature}")
    encoder = hydra.utils.instantiate(cfg.model.encoder)
    embedding_dim = int(cfg.model.embedding_dim)
    patch_body = None
    if feature == "patch":
        patch_body = encoder.body if hasattr(encoder, "to_poincare") else encoder
        if not all(hasattr(patch_body, attr) for attr in
                   ("forward_features", "patch_embed", "num_prefix_tokens", "num_features")):
            raise ValueError("Patch extraction requires a timm ViT or HyperbolicViT")
        embedding_dim = int(patch_body.num_features)
    GlobalHydra.instance().clear()

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    prefix = "encoder."
    # Strip the Lightning module prefix; also tolerate a torch.compile ``_orig_mod.`` wrap.
    encoder_sd = {}
    for key, value in state_dict.items():
        if not key.startswith(prefix):
            continue
        sub = key[len(prefix):]
        if sub.startswith("_orig_mod."):
            sub = sub[len("_orig_mod."):]
        encoder_sd[sub] = value

    if patch_body is not None:
        pos_key = "body.pos_embed" if patch_body is not encoder else "pos_embed"
        if pos_key in encoder_sd and patch_body.pos_embed is not None:
            source, target = encoder_sd[pos_key], patch_body.pos_embed
            if source.shape != target.shape:
                from timm.layers import resample_abs_pos_embed
                prefix = 0 if getattr(patch_body, "no_embed_class", False) else patch_body.num_prefix_tokens
                count = source.shape[1] - prefix
                side = int(count ** 0.5)
                if side * side != count:
                    raise ValueError("Patch PCA requires a square checkpoint positional grid")
                encoder_sd[pos_key] = resample_abs_pos_embed(
                    source, new_size=patch_body.patch_embed.grid_size,
                    old_size=(side, side), num_prefix_tokens=prefix,
                )

    missing, unexpected = encoder.load_state_dict(encoder_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Encoder state_dict mismatch for {experiment}: "
            f"{len(missing)} missing, {len(unexpected)} unexpected. "
            f"missing[:5]={list(missing)[:5]} unexpected[:5]={list(unexpected)[:5]}"
        )

    # Move before eval(): escnn conv layers expand their filters from a basis
    # cache that is *shared between module instances*, and they do so inside
    # eval(). Once an earlier equivariant backbone in the same process has been
    # moved to CUDA, that shared basis lives on CUDA, so calling eval() while
    # these weights are still on CPU einsums a CPU tensor against a CUDA one.
    encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)

    @torch.no_grad()
    def extract_fn(images: torch.Tensor) -> torch.Tensor:
        images = images.to(device, non_blocking=True)
        if patch_body is not None:
            return vit_spatial_patch_tokens(patch_body, images).float().cpu()
        feats = encoder(images)
        if feats.ndim != 2:
            raise RuntimeError(
                f"Expected encoder to return (B, D) embeddings, got {tuple(feats.shape)}. "
                "The encoder must be configured with num_classes = embedding_dim (LeJEPA mode)."
            )
        return feats.float().cpu()

    return extract_fn, embedding_dim


def vit_spatial_patch_tokens(body, images):
    """Post-norm ViT tokens in raster order, excluding CLS/register tokens."""
    tokens = body.forward_features(images)
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
        raise ValueError("Expected forward_features to return (B,N,D) tokens")
    tokens = tokens[:, body.num_prefix_tokens:]
    expected = int(body.patch_embed.grid_size[0] * body.patch_embed.grid_size[1])
    if tokens.shape[1] != expected:
        raise ValueError(f"Expected {expected} spatial tokens, got {tokens.shape[1]}")
    return tokens


def _stack_labels(labels: list) -> torch.Tensor:
    """Stack a list of per-sample labels into a single tensor.

    Accepts scalars, 1-D vectors, or tensors; returns shape ``(N,)`` or ``(N, k)``.
    """
    out = []
    for lab in labels:
        if isinstance(lab, torch.Tensor):
            out.append(lab.flatten().float())
        elif isinstance(lab, (int, float)):
            out.append(torch.tensor([float(lab)]))
        else:  # numpy array / list / tuple
            out.append(torch.as_tensor(lab, dtype=torch.float32).flatten())
    stacked = torch.stack(out, dim=0)
    return stacked.squeeze(-1) if stacked.shape[-1] == 1 else stacked


def extract_features(
    extract_fn: ExtractFn,
    dataset: Dataset,
    batch_size: int = 256,
    num_workers: int = 8,
    device: Optional[torch.device] = None,  # kept for signature parity; device lives in extract_fn
    cache_path: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract (and optionally cache) features for every sample in ``dataset``.

    Args:
        extract_fn: Callable returned by :func:`load_backbone`.
        dataset: Yields ``(image, label)`` pairs.
        batch_size, num_workers: DataLoader settings.
        cache_path: If set and the file exists, load from it; otherwise compute
            and save ``{"X": features, "y": labels}`` there.

    Returns:
        ``(X, y)`` with ``X`` of shape ``(N, embedding_dim)``.
    """
    if cache_path is not None and os.path.isfile(cache_path):
        blob = torch.load(cache_path, map_location="cpu")
        return blob["X"], blob["y"]

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    feats: list[torch.Tensor] = []
    labels: list = []
    for images, targets in loader:
        feats.append(extract_fn(images))
        if isinstance(targets, torch.Tensor):
            for row in targets:
                labels.append(row)
        else:
            labels.extend(list(targets))

    X = torch.cat(feats, dim=0)
    y = _stack_labels(labels)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"X": X, "y": y}, cache_path)

    return X, y
