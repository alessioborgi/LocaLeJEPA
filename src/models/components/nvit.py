# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""NormalizedViT (nViT) -- nGPT ViT with learned positional embeddings.

Self-contained implementation closely following the reference nViT by Loshchilov
et al., "nGPT: Normalized Transformer with Representation Learning on the
Hypersphere", ICLR 2025 (arXiv:2410.01131).

Framework additions over the reference:
- num_classes=0 mode returns (B, num_patches, embed_dim) for I-JEPA.
- masks argument in forward for I-JEPA token masking.
- Same __init__ signature as SphericalViT for drop-in Hydra config use.

Training notes (same as the reference):
- Set weight_decay = 0.
- Call model.norm_weights_() after every optimizer step.
- Higher learning rates (3-5x) are safe versus standard ViT.
"""

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from src.models.components.tensor_ops import apply_masks


def l2norm(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(t, dim=dim, p=2)


class L2Norm(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return l2norm(t, dim=self.dim)


class NormLinear(nn.Module):
    """Bias-free linear layer with unit-norm weight rows or columns.

    norm_dim_in=True (default): each output neuron weight vector has unit norm.
    norm_dim_in=False: each input feature weight vector has unit norm.
    """

    def __init__(self, dim: int, dim_out: int, norm_dim_in: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim_out, bias=False)
        parametrize.register_parametrization(
            self.linear, "weight", L2Norm(dim=-1 if norm_dim_in else 0)
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @torch.no_grad()
    def norm_weights_(self) -> None:
        # Snap the raw parameter back to its normalized value.
        original = self.linear.parametrizations.weight.original
        original.copy_(self.weight)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        dim_head: int = 64,
        heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dim_inner = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout

        self.to_q = NormLinear(dim, dim_inner)
        self.to_k = NormLinear(dim, dim_inner)
        self.to_v = NormLinear(dim, dim_inner)
        self.to_out = NormLinear(dim_inner, dim, norm_dim_in=False)

        # Init at dim_head**0.25 so initial logit scale equals dim_head**0.5 * cos(q, k).
        scale_init = dim_head ** 0.25
        self.q_scale = nn.Parameter(torch.ones(heads, 1, dim_head) * scale_init)
        self.k_scale = nn.Parameter(torch.ones(heads, 1, dim_head) * scale_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        H, D = self.heads, self.dim_head

        # Project and split into heads: (B, H, N, D).
        q = self.to_q(x).reshape(B, N, H, D).transpose(1, 2)
        k = self.to_k(x).reshape(B, N, H, D).transpose(1, 2)
        v = self.to_v(x).reshape(B, N, H, D).transpose(1, 2)

        # Unit-norm QK then apply per-head learned scale; scale=1.0 in SDPA.
        q = l2norm(q) * self.q_scale
        k = l2norm(k) * self.k_scale

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, scale=1.0
        )
        out = out.transpose(1, 2).reshape(B, N, H * D)
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, *, dim_inner: int, dropout: float = 0.0) -> None:
        super().__init__()
        # SwiGLU: actual hidden width is 2/3 of the nominal expansion.
        dim_inner = int(dim_inner * 2 / 3)

        self.dim = dim
        self.dropout = nn.Dropout(dropout)

        self.to_hidden = NormLinear(dim, dim_inner)
        self.to_gate   = NormLinear(dim, dim_inner)
        self.to_out    = NormLinear(dim_inner, dim, norm_dim_in=False)

        self.hidden_scale = nn.Parameter(torch.ones(dim_inner))
        self.gate_scale   = nn.Parameter(torch.ones(dim_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.to_hidden(x) * self.hidden_scale
        # Gate branch scaled by sqrt(dim) to maintain unit-sphere geometry.
        gate   = self.to_gate(x) * self.gate_scale * (self.dim ** 0.5)
        hidden = F.silu(gate) * hidden
        hidden = self.dropout(hidden)
        return self.to_out(hidden)


class NormalizedViT(nn.Module):
    """nGPT Vision Transformer with learned positional embeddings.

    All token representations live on the unit hypersphere at every layer.
    Positional encoding is a table of unit-norm vectors stored as the weight
    rows of a NormLinear(embed_dim, num_patches) layer, exactly as in the
    reference nViT implementation.

    Args:
        img_size:       Input image resolution (square assumed).
        patch_size:     Patch side length in pixels.
        in_chans:       Number of input channels.
        embed_dim:      Transformer embedding dimension d.
        depth:          Number of transformer blocks.
        num_heads:      Number of attention heads.
        mlp_ratio:      Nominal FFN hidden-dim expansion (SwiGLU actual width
                        is int(embed_dim * mlp_ratio * 2/3)).
        drop_rate:      Dropout inside MLP and attention output projection.
        attn_drop_rate: Dropout applied to attention weights.
        num_classes:    If > 0, adds GAP + NormLinear head returning
                        (B, num_classes) for LeJEPA. Set to 0 to return patch
                        tokens (B, num_patches, embed_dim) for I-JEPA.
        alpha_init:     Initial eigen learning rate for spherical residuals.
                        Defaults to 1/depth.
        qkv_bias, qk_scale, drop_path_rate, norm_layer, init_std,
        sh_max_degree, use_sph_attn_bias, sph_attn_num_bins:
                        Accepted for drop-in compatibility with SphericalViT
                        configs; all are silently ignored.
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
        # SphericalViT-compatible kwargs, accepted but ignored:
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

        if isinstance(img_size, Sequence):
            img_h, img_w = img_size[0], img_size[1]
        else:
            img_h = img_w = img_size

        patch_dim   = in_chans * patch_size * patch_size
        num_patches = (img_h // patch_size) * (img_w // patch_size)

        self.num_features = self.embed_dim = embed_dim
        self.num_heads    = num_heads
        self.num_classes  = num_classes
        self.patch_size   = patch_size
        # scale = sqrt(embed_dim), used to restore alpha and logit magnitude.
        self.scale        = embed_dim ** 0.5

        # Linear patch projection; columns of weight are unit-norm (norm_dim_in=False).
        self.to_patch_embedding = NormLinear(patch_dim, embed_dim, norm_dim_in=False)

        # PE table: weight rows are unit-norm PE vectors, one per patch position.
        # Weight shape is (num_patches, embed_dim); row i is the PE for position i.
        self.abs_pos_emb = NormLinear(embed_dim, num_patches)

        # Default alpha follows the nGPT recommendation of 1/depth.
        residual_lerp_scale_init = (1.0 / depth) if alpha_init is None else alpha_init

        dim_head  = embed_dim // num_heads
        dim_inner = int(embed_dim * mlp_ratio)

        self.layers = nn.ModuleList([])
        self.residual_lerp_scales = nn.ParameterList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(embed_dim, dim_head=dim_head, heads=num_heads, dropout=attn_drop_rate),
                FeedForward(embed_dim, dim_inner=dim_inner, dropout=drop_rate),
            ]))
            # Alpha stored divided by scale; restored in forward to keep parameter magnitudes small.
            self.residual_lerp_scales.append(nn.ParameterList([
                nn.Parameter(torch.ones(embed_dim) * residual_lerp_scale_init / self.scale),
                nn.Parameter(torch.ones(embed_dim) * residual_lerp_scale_init / self.scale),
            ]))

        if num_classes > 0:
            self.logit_scale = nn.Parameter(torch.ones(num_classes))
            self.to_pred     = NormLinear(embed_dim, num_classes)
        else:
            self.logit_scale = None
            self.to_pred     = None

    @torch.no_grad()
    def norm_weights_(self) -> None:
        # Snap all NormLinear parameters back to unit norm after each optimizer step.
        for module in self.modules():
            if isinstance(module, NormLinear):
                module.norm_weights_()

    def forward(
        self,
        x: torch.Tensor,
        masks: list[torch.Tensor] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:     (B, C, H, W) image tensor.
            masks: Optional index masks for I-JEPA token masking.

        Returns:
            (B, num_classes) when num_classes > 0 (LeJEPA mode), or
            (B, num_patches, embed_dim) when num_classes == 0 (I-JEPA mode).
        """
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        B, C, H, W = x.shape
        p = self.patch_size

        # Rearrange image into flat patch vectors: (B, num_patches, C*p*p).
        tokens = x.reshape(B, C, H // p, p, W // p, p)
        tokens = tokens.permute(0, 2, 4, 1, 3, 5).reshape(B, (H // p) * (W // p), C * p * p)
        tokens = self.to_patch_embedding(tokens)

        # Fetch PE vectors for the current sequence length and project to unit norm.
        seq_len = tokens.shape[1]
        pos_emb = self.abs_pos_emb.weight[torch.arange(seq_len, device=x.device)]
        tokens  = l2norm(tokens + pos_emb)

        if masks is not None:
            tokens = apply_masks(tokens, masks)

        for (attn, ff), (attn_alpha, ff_alpha) in zip(self.layers, self.residual_lerp_scales):
            # Spherical LERP residual: interpolate on the unit hypersphere.
            attn_out = l2norm(attn(tokens))
            tokens   = l2norm(tokens.lerp(attn_out, attn_alpha * self.scale))

            ff_out = l2norm(ff(tokens))
            tokens = l2norm(tokens.lerp(ff_out, ff_alpha * self.scale))

        if self.to_pred is not None:
            # LeJEPA mode: global average pool, re-normalize, then classify.
            pooled = l2norm(tokens.mean(dim=1))
            return self.to_pred(pooled) * self.logit_scale * self.scale

        return tokens
