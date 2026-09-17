# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""RankMe: label-free effective rank of a feature matrix.

Garrido et al., "RankMe: Assessing the Downstream Performance of Pretrained
Self-Supervised Representations by Their Rank" (ICML 2023), Eq. (1)-(2):

    RankMe(Z) = exp(-sum_k p_k log p_k),   p_k = sigma_k(Z) / ||sigma(Z)||_1 + eps

i.e. the exponential of the Shannon entropy of the normalised singular-value
distribution (Roy & Vetterli, 2007). The paper estimates it on 25,600 samples
of the *source* (pretraining) dataset and shows the classical threshold rank
(Press et al., 2007) agrees with it (Appendix F). Both are implemented here,
on the raw feature matrix exactly as in the paper (no centring) plus a centred
variant for reference.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

# Eq. (2) additive epsilon; float32 round-off constant for the classical rank.
RANKME_EPS = 1e-7
CLASSICAL_EPS = 1e-7


def singular_values(Z: torch.Tensor) -> torch.Tensor:
    """Singular values of ``Z`` (N, K) in descending order, computed in float64."""
    if Z.ndim != 2:
        raise ValueError(f"Expected a 2-D feature matrix, got shape {tuple(Z.shape)}")
    return torch.linalg.svdvals(Z.to(torch.float64))


def rankme_from_singular_values(sigma: torch.Tensor, eps: float = RANKME_EPS) -> float:
    """Entropic effective rank of a spectrum (RankMe, Eq. 1)."""
    p = sigma / sigma.sum() + eps
    return float(torch.exp(-(p * p.log()).sum()))


def classical_rank_from_singular_values(sigma: torch.Tensor, shape: tuple[int, int], eps: float = CLASSICAL_EPS) -> int:
    """Threshold rank: number of sigma_k > max(sigma) * max(N, K) * eps."""
    threshold = sigma.max() * max(shape) * eps
    return int((sigma > threshold).sum())


def rankme(Z: torch.Tensor, center: bool = False) -> dict:
    """RankMe, classical rank and the normalised spectrum of a feature matrix.

    Args:
        Z: Features ``(N, K)``; any float dtype (cast to float64 internally).
        center: Subtract the per-dimension mean first. The paper does **not**
            centre; a centred rank ignores a single large mean direction.

    Returns:
        ``{"rankme", "classical_rank", "n", "dim", "spectrum"}`` where ``spectrum``
        is ``sigma / sigma_max`` as a Python list (for spectrum plots).
    """
    Z = Z.to(torch.float64)
    if center:
        Z = Z - Z.mean(dim=0, keepdim=True)
    sigma = singular_values(Z)
    return {
        "rankme": rankme_from_singular_values(sigma),
        "classical_rank": classical_rank_from_singular_values(sigma, tuple(Z.shape)),
        "n": int(Z.shape[0]),
        "dim": int(Z.shape[1]),
        "spectrum": (sigma / sigma[0]).tolist(),
    }


def subsample_rows(X: torch.Tensor, n: int, seed: int = 0) -> torch.Tensor:
    """Deterministic random subset of at most ``n`` rows (same rows for every backbone)."""
    if X.shape[0] <= n:
        return X
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randperm(X.shape[0], generator=generator)[:n].sort().values
    return X[idx]


def subsample_indices(total: int, n: int, seed: int = 0) -> list[int]:
    """The row indices :func:`subsample_rows` would pick — to embed only those images."""
    if total <= n:
        return list(range(total))
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(total, generator=generator)[:n].sort().values.tolist()


def load_projector(
    experiment: str,
    checkpoint_path: str,
    crop_size: int = 128,
    config_dir: Optional[str] = None,
    project_root: Optional[str] = None,
    overrides: Optional[list[str]] = None,
) -> tuple[torch.nn.Module, bool]:
    """Rebuild the pretraining projector (encoder output -> embeddings) and load its weights.

    Mirrors :func:`src.eval.extract.load_backbone` for the ``projector.`` keys of
    the Lightning checkpoint. Returns ``(projector, rectify)`` where ``rectify``
    is True for Rectified LpJEPA, whose loss sees ``ReLU(projector(.))``
    (``src/models/rectifiedlpjepa_module.py``).
    """
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if config_dir is None:
        config_dir = os.path.join(project_root, "configs")
    os.environ.setdefault("PROJECT_ROOT", project_root)

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=os.path.abspath(config_dir)):
        cfg = compose(
            config_name="train",
            overrides=[f"experiment={experiment}", f"data.crop_size={crop_size}", *(overrides or [])],
        )
    projector = hydra.utils.instantiate(cfg.model.projector)
    # Same rule as RectifiedLpJEPAModule.__init__: rectify iff the target is the RGN.
    rectify = cfg.model.get("target_distribution") == "rectified_lp_distribution"
    GlobalHydra.instance().clear()

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    prefix = "projector."
    proj_sd = {}
    for key, value in ckpt["state_dict"].items():
        if key.startswith(prefix):
            sub = key[len(prefix):]
            if sub.startswith("_orig_mod."):
                sub = sub[len("_orig_mod."):]
            proj_sd[sub] = value
    missing, unexpected = projector.load_state_dict(proj_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Projector state_dict mismatch for {experiment}: {len(missing)} missing, "
            f"{len(unexpected)} unexpected. missing[:5]={list(missing)[:5]} unexpected[:5]={list(unexpected)[:5]}"
        )
    projector.eval()
    for param in projector.parameters():
        param.requires_grad_(False)
    return projector, rectify


@torch.no_grad()
def project_features(projector: torch.nn.Module, X: torch.Tensor, rectify: bool, batch_size: int = 8192) -> torch.Tensor:
    """Push frozen encoder features through the (eval-mode) projector on CPU."""
    outs = []
    for start in range(0, X.shape[0], batch_size):
        z = projector(X[start:start + batch_size].float())
        outs.append(torch.relu(z) if rectify else z)
    return torch.cat(outs, dim=0)
