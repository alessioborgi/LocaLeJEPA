# Copyright (c) 2026 Alessio Borgi
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""NormalizedViT (nViT) — nGPT ViT with learned positional embeddings.

Compared with SphericalViT this model replaces:
  • Spherical harmonic PE → learned unit-norm PE table (one d-dim vector per patch).
  • Great-circle RPB      → no RPB; standard scaled dot-product attention.

All nGPT components (QK-norm, SwiGLU, spherical LERP residuals, NormLinear
weight rows on S^{d-1}) are identical to SphericalViT.

Reference: Loshchilov et al., "nGPT: Normalized Transformer with Representation
Learning on the Hypersphere", ICLR 2025.  https://arxiv.org/abs/2410.01131

Training notes (same as SphericalViT):
  • Set weight_decay = 0.
  • Call model.norm_weights_() after every optimizer step.
  • Higher learning rates (3–5×) are safe vs. standard ViT.

Drop-in replacement for SphericalViT in LeJEPA and I-JEPA configs.
"""

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.spherical_vit import NormLinear, SwiGLUMLP, l2norm
from src.models.components.tensor_ops import apply_masks
from src.models.components.vision_transformer import PatchEmbed, _to_2tuple


class NViTAttention(nn.Module):
    """Multi-head self-attention on the hypersphere — no spherical RPB.

    Differences from SphericalAttention:
      • No great-circle distance relative position bias.
      • QK scales stored directly at ``head_dim**0.25`` (reference nViT convention)
        so the initial effective attention logit is ``head_dim**0.5 * cos(q, k)``,
        matching the nGPT paper eq. 16.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.Wq = NormLinear(dim, dim)
        self.Wk = NormLinear(dim, dim)
        self.Wv = NormLinear(dim, dim)
        self.Wo = NormLinear(dim, dim)

        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

        # per-head per-dim QK scales; init at head_dim**0.25 so that
        # (q*s) @ (k*s)^T = head_dim**0.5 * cosine(q_unit, k_unit) initially
        scale_init = self.head_dim ** 0.25
        self.q_scale = nn.Parameter(torch.full((num_heads, 1, self.head_dim), scale_init))
        self.k_scale = nn.Parameter(torch.full((num_heads, 1, self.head_dim), scale_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.Wq(x).reshape(B, N, H, D).transpose(1, 2)  # (B, H, N, D)
        k = self.Wk(x).reshape(B, N, H, D).transpose(1, 2)
        v = self.Wv(x).reshape(B, N, H, D).transpose(1, 2)

        # QK-norm then scale; scale=1. because scaling is absorbed into q_scale / k_scale
        q = l2norm(q) * self.q_scale
        k = l2norm(k) * self.k_scale

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0,
            scale=1.0,
        )
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.Wo(out))


class NViTBlock(nn.Module):
    """Transformer block with nGPT spherical LERP residuals (no RPB)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        drop: float = 0.0,
        alpha_init: float = 0.05,
    ) -> None:
        super().__init__()
        self.attn = NViTAttention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.mlp = SwiGLUMLP(dim, mlp_ratio=mlp_ratio, drop=drop)

        # per-dim eigen learning rates stored at alpha_init/sqrt(dim); restored in forward
        alpha_scale = alpha_init / math.sqrt(dim)
        self._alpha_restore = math.sqrt(dim)
        self.alpha_A = nn.Parameter(torch.full((dim,), alpha_scale))
        self.alpha_M = nn.Parameter(torch.full((dim,), alpha_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha_A = self.alpha_A * self._alpha_restore
        x = l2norm(x.lerp(l2norm(self.attn(x)), alpha_A))

        alpha_M = self.alpha_M * self._alpha_restore
        x = l2norm(x.lerp(l2norm(self.mlp(x)), alpha_M))
        return x


class NormalizedViT(nn.Module):
    """nGPT Vision Transformer with learned positional embeddings.

    All token representations live on the unit hypersphere S^{d-1} at every
    layer.  Positional encoding is a learned table of unit-norm vectors, one
    per patch position, rather than the spherical harmonic projection used by
    SphericalViT.  Attention uses standard SDPA without a relative position bias.

    **Training recipe** (identical to SphericalViT):

    1. Set ``weight_decay = 0``.
    2. After each optimiser step call ``model.norm_weights_()`` to keep
       :class:`NormLinear` weight rows and the PE table on the sphere.
    3. Higher learning rates (3–5×) are safe compared with a standard ViT.

    Args:
        img_size: Input image resolution (square assumed).
        patch_size: Patch side length in pixels.
        in_chans: Number of input channels.
        embed_dim: Transformer embedding dimension *d*.
        depth: Number of transformer blocks *L*.
        num_heads: Number of attention heads.
        mlp_ratio: FFN hidden-dim expansion (SwiGLU convention: actual width
            is ``int(embed_dim * mlp_ratio * 2/3)``).
        drop_rate: Dropout inside MLP and attention output projection.
        attn_drop_rate: Dropout applied to attention weights.
        num_classes: If > 0, add GAP + NormLinear head → ``(B, num_classes)``
            for LeJEPA.  Set to 0 to return ``(B, num_patches, embed_dim)``
            for I-JEPA.
        alpha_init: Initial effective eigen learning rate for spherical residuals.
            Defaults to ``1/depth`` (nGPT recommendation).
        qkv_bias, qk_scale, drop_path_rate, norm_layer, init_std,
        sh_max_degree, use_sph_attn_bias, sph_attn_num_bins: Accepted for
            config drop-in compatibility with SphericalViT; all are ignored.
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
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        num_classes: int = 0,
        alpha_init: float | None = None,
        # SphericalViT-compatible kwargs — accepted but ignored:
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_path_rate: float = 0.0,
        norm_layer: object = None,
        init_std: float = 0.02,
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
        num_patches = (img_h // patch_size) * (img_w // patch_size)

        # Learned unit-norm PE table: shape (num_patches, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(num_patches, embed_dim))
        with torch.no_grad():
            self.pos_emb.copy_(F.normalize(self.pos_emb, dim=-1))

        _alpha = (1.0 / depth) if alpha_init is None else alpha_init
        self.blocks = nn.ModuleList([
            NViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop_rate,
                drop=drop_rate,
                alpha_init=_alpha,
            )
            for _ in range(depth)
        ])

        if num_classes > 0:
            self.head = NormLinear(embed_dim, num_classes)
            self._sz_restore = math.sqrt(num_classes)
            self.logit_scale = nn.Parameter(
                torch.full((num_classes,), num_classes ** -0.5)
            )
        else:
            self.head = None
            self.logit_scale = None
            self._sz_restore = 1.0

    def norm_weights_(self) -> None:
        """Snap all NormLinear weight rows and the PE table back to unit norm.

        Call after every optimiser step.
        """
        with torch.no_grad():
            self.pos_emb.copy_(F.normalize(self.pos_emb, dim=-1))
        for m in self.modules():
            if isinstance(m, NormLinear):
                m.norm_weights_()

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
            Output tokens are unit-norm vectors on S^{d-1}.
        """
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        # Step 1: extract patch tokens, add unit-norm PE, project onto S^{d-1}
        x = self.patch_embed(x)
        pos_emb = F.normalize(self.pos_emb, dim=-1)
        x = l2norm(x + pos_emb)

        # Step 2: (optional) keep only context/target subsets for I-JEPA masking
        if masks is not None:
            x = apply_masks(x, masks)

        # Step 3: pass through all transformer blocks; each keeps tokens on S^{d-1}
        for block in self.blocks:
            x = block(x)

        if self.head is not None:
            # Step 4 (LeJEPA): global average pool, re-normalise, apply head with temperature
            x = l2norm(x.mean(dim=1))
            x = self.head(x) * (self.logit_scale * self._sz_restore)

        return x
