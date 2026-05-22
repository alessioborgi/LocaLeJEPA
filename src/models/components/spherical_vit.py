# Copyright (c) 2026 Alessio Borgi
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""SphericalViT — Vision Transformer with spherical harmonic positional encoding
and optional spherical relative position bias (RPB).

Drop-in replacement for VisionTransformer / timm ViT in LeJEPA and I-JEPA.

Key differences from standard ViT:
  • Positional encoding: real spherical harmonics Y_l^m evaluated at each patch's
    (θ, φ) on the unit sphere, projected to embed_dim via a learned linear layer,
    instead of fixed 2-D sinusoidal encoding.  The harmonic features are fixed;
    only the linear projection is learned.
  • Spherical RPB (optional): each attention layer adds a per-head learnable bias
    indexed by the great-circle distance bin between query and key patches.
  • num_classes interface matching timm ViT: when num_classes > 0, a GAP + linear
    head produces (B, num_classes) for LeJEPA.  When num_classes == 0 the raw token
    sequence (B, num_patches, embed_dim) is returned for I-JEPA compatibility.
"""

import math
from collections.abc import Callable, Sequence
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from src.models.components.tensor_ops import apply_masks
from src.models.components.vision_transformer import DropPath, MLP, PatchEmbed, _to_2tuple


# ---------------------------------------------------------------------------
# Spherical harmonic helpers (pure NumPy, runs once at model construction)
# ---------------------------------------------------------------------------

def _associated_legendre(l_max: int, x: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """Associated Legendre polynomials P_l^m(x), 0 ≤ m ≤ l ≤ l_max.

    Uses the standard three-term recurrences without Condon–Shortley phase,
    matching the real-spherical-harmonic convention adopted below.
    """
    Plm: dict[tuple[int, int], np.ndarray] = {}
    sin_t = np.sqrt(np.maximum(1.0 - x**2, 0.0))

    Plm[(0, 0)] = np.ones_like(x)

    # Diagonal: P_m^m = (2m-1) * sin_theta * P_{m-1}^{m-1}
    for m in range(1, l_max + 1):
        Plm[(m, m)] = (2 * m - 1) * sin_t * Plm[(m - 1, m - 1)]

    # Sub-diagonal: P_{m+1}^m = (2m+1) * x * P_m^m
    for m in range(l_max):
        Plm[(m + 1, m)] = (2 * m + 1) * x * Plm[(m, m)]

    # General three-term recurrence for the remaining entries
    for l in range(2, l_max + 1):
        for m in range(0, l - 1):
            Plm[(l, m)] = (
                (2 * l - 1) * x * Plm[(l - 1, m)] - (l + m - 1) * Plm[(l - 2, m)]
            ) / (l - m)

    return Plm


def _real_sh_matrix(H: int, W: int, max_degree: int) -> np.ndarray:
    """Evaluate real spherical harmonics at the centre of each patch.

    Args:
        H: Number of patch rows.
        W: Number of patch columns.
        max_degree: Maximum harmonic degree L.

    Returns:
        Array of shape ``(H*W, (max_degree+1)**2)`` with harmonic values ordered
        as ``[(l=0,m=0), (l=1,m=-1), (l=1,m=0), (l=1,m=1), (l=2,m=-2), ...]``.
    """
    N = H * W
    # Patch centres in spherical coordinates
    theta = np.pi * (np.arange(H, dtype=np.float64) + 0.5) / H   # (H,)  colatitude
    phi   = 2.0 * np.pi * (np.arange(W, dtype=np.float64) + 0.5) / W  # (W,)  azimuth
    theta_flat = np.repeat(theta, W)  # (N,)
    phi_flat   = np.tile(phi, H)      # (N,)
    cos_theta  = np.cos(theta_flat)   # (N,)

    Plm = _associated_legendre(max_degree, cos_theta)

    num_harmonics = (max_degree + 1) ** 2
    sh = np.zeros((N, num_harmonics), dtype=np.float64)
    col = 0
    for l in range(max_degree + 1):
        for m in range(-l, l + 1):
            abs_m = abs(m)
            # Orthonormal normalisation constant K_l^m
            fact_ratio = math.factorial(l - abs_m) / math.factorial(l + abs_m)
            K = math.sqrt((2 * l + 1) / (4.0 * math.pi) * fact_ratio)
            if m == 0:
                sh[:, col] = K * Plm[(l, 0)]
            elif m > 0:
                sh[:, col] = math.sqrt(2.0) * K * Plm[(l, m)] * np.cos(m * phi_flat)
            else:
                sh[:, col] = math.sqrt(2.0) * K * Plm[(l, abs_m)] * np.sin(abs_m * phi_flat)
            col += 1

    return sh.astype(np.float32)


def _dist_bins(H: int, W: int, num_bins: int) -> np.ndarray:
    """Great-circle distance bins for every pair of patch centres.

    Returns:
        Integer array of shape ``(H*W, H*W)`` with bin indices in ``[0, num_bins)``.
    """
    theta = np.pi * (np.arange(H, dtype=np.float64) + 0.5) / H
    phi   = 2.0 * np.pi * (np.arange(W, dtype=np.float64) + 0.5) / W
    theta_flat = np.repeat(theta, W)   # (N,)
    phi_flat   = np.tile(phi, H)       # (N,)

    cos_t = np.cos(theta_flat)
    sin_t = np.sin(theta_flat)
    # cos(angular distance) = sin_t_i*sin_t_j*cos(phi_i-phi_j) + cos_t_i*cos_t_j
    cos_dp   = np.cos(phi_flat[:, None] - phi_flat[None, :])  # (N, N)
    cos_dist = sin_t[:, None] * sin_t[None, :] * cos_dp + cos_t[:, None] * cos_t[None, :]
    cos_dist = np.clip(cos_dist, -1.0, 1.0)
    dist = np.arccos(cos_dist)  # (N, N)  in [0, pi]

    bins = np.floor(dist / math.pi * (num_bins - 1)).astype(np.int32)
    return np.clip(bins, 0, num_bins - 1)


# ---------------------------------------------------------------------------
# SphericalHarmonicPE
# ---------------------------------------------------------------------------

class SphericalHarmonicPE(nn.Module):
    """Positional encoding from real spherical harmonics projected to embed_dim.

    The harmonic features are fixed buffers; only the linear projection is learned.
    """

    def __init__(self, H: int, W: int, embed_dim: int, max_degree: int = 8) -> None:
        super().__init__()
        sh = _real_sh_matrix(H, W, max_degree)
        self.register_buffer("sh_feats", torch.from_numpy(sh))  # (N, K)
        self.proj = nn.Linear((max_degree + 1) ** 2, embed_dim, bias=False)
        self._H = H
        self._W = W
        self._max_degree = max_degree

    def forward(self) -> torch.Tensor:
        """Return positional encoding of shape ``(1, N, embed_dim)``."""
        return self.proj(self.sh_feats).unsqueeze(0)  # type: ignore[arg-type]

    def recompute(self, H: int, W: int) -> torch.Tensor:
        """Compute PE analytically for a different grid without modifying buffers."""
        sh = torch.from_numpy(_real_sh_matrix(H, W, self._max_degree)).to(
            device=self.sh_feats.device, dtype=self.sh_feats.dtype
        )
        return self.proj(sh).unsqueeze(0)  # (1, H*W, D)


# ---------------------------------------------------------------------------
# SphericalAttention
# ---------------------------------------------------------------------------

class SphericalAttention(nn.Module):
    """Multi-head self-attention with an optional learnable spherical RPB.

    The RPB adds a per-head scalar bias to each logit based on the great-circle
    distance bin between the two patch positions.  It is silently skipped when
    the sequence length does not match the stored grid (e.g. after masking).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        H: int | None = None,
        W: int | None = None,
        num_bins: int = 64,
        use_sph_attn_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self._full_N = (H * W) if (H is not None and W is not None) else None
        self.use_sph_attn_bias = use_sph_attn_bias and self._full_N is not None
        if self.use_sph_attn_bias:
            bins = _dist_bins(H, W, num_bins)  # type: ignore[arg-type]
            self.register_buffer("dist_bins", torch.from_numpy(bins).long())  # (N, N)
            self.bias_table = nn.Parameter(torch.zeros(num_heads, num_bins))

    def _rpb(self, n: int) -> torch.Tensor | None:
        if not self.use_sph_attn_bias or n != self._full_N:
            return None
        # bias_table: (H, num_bins), dist_bins: (N, N) → index → (H, N, N) → (1, H, N, N)
        return self.bias_table[:, self.dist_bins].unsqueeze(0)  # type: ignore[index]

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        rpb = self._rpb(N)
        if rpb is not None:
            attn = attn + rpb
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if return_attention:
            return attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


# ---------------------------------------------------------------------------
# SphericalBlock
# ---------------------------------------------------------------------------

class SphericalBlock(nn.Module):
    """Transformer block backed by SphericalAttention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        H: int | None = None,
        W: int | None = None,
        num_bins: int = 64,
        use_sph_attn_bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SphericalAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            H=H,
            W=W,
            num_bins=num_bins,
            use_sph_attn_bias=use_sph_attn_bias,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        attn = self.attn(self.norm1(x), return_attention=return_attention)
        if return_attention:
            return attn
        x = x + self.drop_path(attn)
        return x + self.drop_path(self.mlp(self.norm2(x)))


# ---------------------------------------------------------------------------
# SphericalViT
# ---------------------------------------------------------------------------

class SphericalViT(nn.Module):
    """Vision Transformer with spherical harmonic positional encoding.

    Drop-in replacement for VisionTransformer and timm ViT backbones in both
    LeJEPA (num_classes > 0) and I-JEPA (num_classes == 0) training pipelines.

    Args:
        img_size: Input image size (square assumed).
        patch_size: Patch size in pixels.
        in_chans: Number of input channels.
        embed_dim: Transformer embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden-dim expansion factor.
        qkv_bias: Add bias to QKV projections.
        qk_scale: Override default QK scale (head_dim**-0.5).
        drop_rate: Dropout applied inside MLP.
        attn_drop_rate: Dropout applied to attention weights.
        drop_path_rate: Stochastic depth rate (linearly scaled across blocks).
        norm_layer: Normalisation layer factory.
        init_std: Standard deviation for weight initialisation.
        num_classes: If > 0, add GAP + linear head → output (B, num_classes).
            Set to ``${model.embedding_dim}`` in Hydra configs for LeJEPA.
            Set to 0 to return raw token sequence (B, num_patches, embed_dim) for I-JEPA.
        sh_max_degree: Maximum spherical harmonic degree L.  Produces (L+1)² features
            per patch that are projected to embed_dim.  Default 8 → 81 harmonics.
        use_sph_attn_bias: Add a great-circle-distance RPB to every attention layer.
        sph_attn_num_bins: Number of distance bins for the RPB lookup table.
    """

    def __init__(
        self,
        img_size: int | Sequence[int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        init_std: float = 0.02,
        num_classes: int = 0,
        sh_max_degree: int = 8,
        use_sph_attn_bias: bool = True,
        sph_attn_num_bins: int = 64,
    ) -> None:
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_classes = num_classes

        img_h, img_w = _to_2tuple(img_size)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        H_p = img_h // patch_size
        W_p = img_w // patch_size
        self._H_p = H_p
        self._W_p = W_p

        self.pos_encoding = SphericalHarmonicPE(H_p, W_p, embed_dim, max_degree=sh_max_degree)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                SphericalBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    H=H_p,
                    W=W_p,
                    num_bins=sph_attn_num_bins,
                    use_sph_attn_bias=use_sph_attn_bias,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Classification head for LeJEPA mode (GAP + linear, matching timm ViT interface)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else None

        self.init_std = init_std
        self.apply(self._init_weights)
        self.fix_init_weight()

    # ------------------------------------------------------------------
    # Weight initialisation (identical to VisionTransformer)
    # ------------------------------------------------------------------

    def fix_init_weight(self) -> None:
        def rescale(param: torch.Tensor, layer_id: int) -> None:
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    # ------------------------------------------------------------------
    # Positional encoding with variable-size support
    # ------------------------------------------------------------------

    def _get_pos_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Return PE for the current token grid, recomputing analytically if needed."""
        N = x.shape[1]
        if N == self._H_p * self._W_p:
            return self.pos_encoding()
        # Non-default crop size: derive grid from N assuming square
        side = int(math.isqrt(N))
        return self.pos_encoding.recompute(side, side)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        masks: list[torch.Tensor] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: ``(B, C, H, W)`` image tensor.
            masks: Optional index masks for I-JEPA token masking.

        Returns:
            ``(B, num_classes)`` when ``num_classes > 0`` (LeJEPA mode), or
            ``(B, num_patches, embed_dim)`` when ``num_classes == 0`` (I-JEPA mode).
        """
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        x = self.patch_embed(x)
        x = x + self._get_pos_encoding(x)

        if masks is not None:
            x = apply_masks(x, masks)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        if self.head is not None:
            x = x.mean(dim=1)   # GAP over patch tokens
            x = self.head(x)

        return x
