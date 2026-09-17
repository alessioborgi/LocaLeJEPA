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
        feature: ``"encoder"`` (default) returns the frozen ``embedding_dim`` backbone
            output -- the representation used for every downstream probe. ``"projector"``
            chains the trained projector head and returns its ``projection_dim`` output --
            the space the JEPA regularizer (SIGReg / BallReg / RDMReg) actually acts on.
            ``"patch"`` returns post-final-norm spatial tokens (B,H*W,D),
            excluding all prefix tokens and bypassing pooling/output heads.
            H-ViT uses the Euclidean transformer body, not ball coordinates.
            Note the RectLpJEPA projector output is *pre*-ReLU (the module rectifies in
            ``model_step``, not in the projector module).

    Returns:
        ``(extract_fn, feat_dim)`` where ``extract_fn(images) -> features`` and
        ``feat_dim`` is ``embedding_dim`` (encoder) or ``projection_dim`` (projector).
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
    if feature not in ("encoder", "projector", "patch"):
        raise ValueError(f"feature must be 'encoder', 'projector', or 'patch', got {feature!r}")

    encoder = hydra.utils.instantiate(cfg.model.encoder)
    embedding_dim = int(cfg.model.embedding_dim)
    patch_body = None
    if feature == "patch":
        # H-ViT's spatial tokens belong to its Euclidean transformer body;
        # its tangent head and Poincare map act only on the pooled output.
        patch_body = encoder.body if hasattr(encoder, "to_poincare") else encoder
        if not all(hasattr(patch_body, a) for a in
                   ("forward_features", "patch_embed", "num_prefix_tokens", "num_features")):
            raise ValueError("Patch extraction supports timm ViTs and this repo's HyperbolicViT only")
    projector = (
        hydra.utils.instantiate(cfg.model.projector)
        if feature == "projector" and cfg.model.get("projector") is not None
        else None
    )
    feat_dim = int(cfg.model.projection_dim) if feature == "projector" else embedding_dim
    if feature == "patch":
        feat_dim = int(patch_body.num_features)
    GlobalHydra.instance().clear()

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    # A checkpoint trained at 128px has a 16x16 positional grid (plus CLS),
    # while a 224px visualization asks for 28x28.  Timm's standard remedy is
    # bicubic interpolation of the spatial positional table.  This is enabled
    # only for patch extraction: ordinary encoder loading remains strict, so a
    # changed resolution cannot silently alter downstream evaluation.
    if feature == "patch":
        pos_key = next((k for k in ("encoder.pos_embed", "encoder.body.pos_embed") if k in state_dict), None)
        if pos_key is not None:
            target = patch_body.pos_embed
            source = state_dict[pos_key]
            if source.shape != target.shape:
                import torch.nn.functional as F
                n_prefix = int(patch_body.num_prefix_tokens)
                n_src = source.shape[1] - n_prefix
                n_dst = target.shape[1] - n_prefix
                src_side, dst_side = int(n_src ** 0.5), int(n_dst ** 0.5)
                if src_side * src_side != n_src or dst_side * dst_side != n_dst:
                    raise RuntimeError(f"Cannot interpolate non-square positional grids: {source.shape} -> {target.shape}")
                prefix = source[:, :n_prefix]
                spatial = source[:, n_prefix:].reshape(1, src_side, src_side, -1).permute(0, 3, 1, 2)
                spatial = F.interpolate(spatial, size=(dst_side, dst_side), mode="bicubic", align_corners=False)
                state_dict[pos_key] = torch.cat([prefix, spatial.permute(0, 2, 3, 1).reshape(1, n_dst, -1)], dim=1)

    def _submodule_sd(prefix: str) -> dict:
        """Pull one Lightning submodule's params, tolerating a torch.compile wrap."""
        out = {}
        for key, value in state_dict.items():
            if not key.startswith(prefix):
                continue
            sub = key[len(prefix):]
            if sub.startswith("_orig_mod."):
                sub = sub[len("_orig_mod."):]
            out[sub] = value
        return out

    missing, unexpected = encoder.load_state_dict(_submodule_sd("encoder."), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Encoder state_dict mismatch for {experiment}: "
            f"{len(missing)} missing, {len(unexpected)} unexpected. "
            f"missing[:5]={list(missing)[:5]} unexpected[:5]={list(unexpected)[:5]}"
        )
    if projector is not None:
        p_missing, p_unexpected = projector.load_state_dict(_submodule_sd("projector."), strict=False)
        if p_missing or p_unexpected:
            raise RuntimeError(
                f"Projector state_dict mismatch for {experiment}: "
                f"{len(p_missing)} missing, {len(p_unexpected)} unexpected. "
                f"missing[:5]={list(p_missing)[:5]} unexpected[:5]={list(p_unexpected)[:5]}"
            )

    # Move before eval(): escnn conv layers expand their filters from a basis
    # cache that is *shared between module instances*, and they do so inside
    # eval(). Once an earlier equivariant backbone in the same process has been
    # moved to CUDA, that shared basis lives on CUDA, so calling eval() while
    # these weights are still on CPU einsums a CPU tensor against a CUDA one.
    modules = [encoder] + ([projector] if projector is not None else [])
    for mod in modules:
        mod.to(device)
        mod.eval()
        for param in mod.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def extract_fn(images: torch.Tensor) -> torch.Tensor:
        images = images.to(device, non_blocking=True)
        if patch_body is not None:
            return spatial_patch_tokens(patch_body, images).float().cpu()
        feats = encoder(images)
        if feats.ndim != 2:
            raise RuntimeError(
                f"Expected encoder to return (B, D) embeddings, got {tuple(feats.shape)}. "
                "The encoder must be configured with num_classes = embedding_dim (LeJEPA mode)."
            )
        if projector is not None:
            feats = projector(feats)
        return feats.float().cpu()

    return extract_fn, feat_dim


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


def spatial_patch_tokens(body, images):
    """Post-norm timm ViT tokens, excluding CLS/register tokens, in raster order."""
    tokens = body.forward_features(images)
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
        raise ValueError("Expected forward_features to return (B,N,D) tokens")
    tokens = tokens[:, body.num_prefix_tokens:]
    expected = int(body.patch_embed.grid_size[0] * body.patch_embed.grid_size[1])
    if tokens.shape[1] != expected:
        raise ValueError(f"Expected {expected} spatial tokens, got {tokens.shape[1]}")
    return tokens
