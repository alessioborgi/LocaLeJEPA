# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""SphericalViT — nGPT-style Vision Transformer where token representations live
on the unit hypersphere S^{d-1} throughout the network.

Combines:
  • Spherical harmonic positional encoding: real spherical harmonics Y_l^m
    evaluated at each patch centre, projected to embed_dim.  The harmonic
    features are fixed buffers; only the linear projection is learned.
  • Great-circle distance relative position bias (RPB): every attention layer
    adds a per-head learnable bias indexed by the angular distance bin between
    query and key patch centres.
  • nGPT architecture (Loshchilov et al., ICLR 2025): all weight matrices use
    L2-normalised rows (NormLinear), attention queries and keys are per-head
    L2-normalised with a learnable per-head scale (QK-norm), the MLP uses
    SwiGLU with multiplicative scaling factors, and every residual update is a
    spherical LERP followed by re-normalisation so that hidden states remain on
    S^{d-1} at every layer.

Training notes:
  • Call ``model.norm_weights_()`` after each optimiser step to snap NormLinear
    weight rows back to unit norm and avoid drift from the parametrised view.
  • Set weight decay to 0 — the unit-norm constraint renders it redundant.
  • Higher learning rates than standard ViT are safe (nGPT reports 4–20× faster
    convergence in step count).

Drop-in replacement for VisionTransformer / timm ViT in LeJEPA and I-JEPA.
"""

import math
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.tensor_ops import apply_masks
from src.models.components.vision_transformer import PatchEmbed, _to_2tuple


def l2norm(x: torch.Tensor) -> torch.Tensor:
    """Normalise the last dimension of *x* to unit norm."""
    return F.normalize(x, dim=-1)


class NormLinear(nn.Module):
    """Linear layer whose weight rows are constrained to the unit hypersphere.

    Normalisation is applied on every forward pass.  Call ``norm_weights_()``
    after each optimiser step to keep the stored weight tensor on the sphere
    and avoid drift between the live normalised view and the stored original.
    No bias term: the nGPT spherical formulation does not use them.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        # Step 1: random initialisation followed by immediate projection onto S^{d-1}
        nn.init.normal_(self.weight)
        with torch.no_grad():
            self.weight.copy_(F.normalize(self.weight, dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Every matrix–vector product is a cosine similarity, so outputs are bounded in [-1, 1]
        return F.linear(x, F.normalize(self.weight, dim=-1))

    def norm_weights_(self) -> None:
        """Snap weight rows to unit norm (call after each optimiser step)."""
        with torch.no_grad():
            self.weight.copy_(F.normalize(self.weight, dim=-1))


def _associated_legendre(l_max: int, x: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """Associated Legendre polynomials P_l^m(x), 0 ≤ m ≤ l ≤ l_max.

    Uses three standard recurrences without Condon–Shortley phase, matching
    the real-spherical-harmonic convention adopted in ``_real_sh_matrix``:

      Diagonal:     P_m^m     = (2m-1) · sin_θ · P_{m-1}^{m-1}
      Sub-diagonal: P_{m+1}^m = (2m+1) · x · P_m^m
      General:      P_l^m     = ((2l-1)·x·P_{l-1}^m − (l+m-1)·P_{l-2}^m) / (l−m)
    """
    Plm: dict[tuple[int, int], np.ndarray] = {}

    # Step 1: derive sin θ from the input cos θ, clamping to avoid NaN from float rounding
    sin_t = np.sqrt(np.maximum(1.0 - x**2, 0.0))

    # Step 2: seed the recurrence with the trivial base case P_0^0 = 1
    Plm[(0, 0)] = np.ones_like(x)

    # Step 3: fill the diagonal entries P_m^m using the sin θ recurrence
    for m in range(1, l_max + 1):
        Plm[(m, m)] = (2 * m - 1) * sin_t * Plm[(m - 1, m - 1)]

    # Step 4: fill the sub-diagonal entries P_{m+1}^m
    for m in range(l_max):
        Plm[(m + 1, m)] = (2 * m + 1) * x * Plm[(m, m)]

    # Step 5: fill all remaining entries with the general three-term recurrence
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

    # Step 1: place patch centres uniformly on the sphere using pixel-centre convention
    theta = np.pi * (np.arange(H, dtype=np.float64) + 0.5) / H       # colatitude ∈ (0, π)
    phi   = 2.0 * np.pi * (np.arange(W, dtype=np.float64) + 0.5) / W  # azimuth    ∈ (0, 2π)
    theta_flat = np.repeat(theta, W)   # (N,)
    phi_flat   = np.tile(phi, H)       # (N,)
    cos_theta  = np.cos(theta_flat)    # argument expected by the Legendre recurrence

    # Step 2: compute all required associated Legendre polynomials
    Plm = _associated_legendre(max_degree, cos_theta)

    # Step 3: assemble the real spherical harmonics column by column
    num_harmonics = (max_degree + 1) ** 2
    sh = np.zeros((N, num_harmonics), dtype=np.float64)
    col = 0
    for l in range(max_degree + 1):
        for m in range(-l, l + 1):
            abs_m = abs(m)
            # orthonormal prefactor shared by all three cases
            fact_ratio = math.factorial(l - abs_m) / math.factorial(l + abs_m)
            K = math.sqrt((2 * l + 1) / (4.0 * math.pi) * fact_ratio)
            if m == 0:
                sh[:, col] = K * Plm[(l, 0)]
            elif m > 0:
                # cosine harmonic — real part of the complex Y_l^m
                sh[:, col] = math.sqrt(2.0) * K * Plm[(l, m)] * np.cos(m * phi_flat)
            else:
                # sine harmonic — imaginary part of the complex Y_l^{|m|}
                sh[:, col] = math.sqrt(2.0) * K * Plm[(l, abs_m)] * np.sin(abs_m * phi_flat)
            col += 1

    return sh.astype(np.float32)


def _dist_bins(H: int, W: int, num_bins: int) -> np.ndarray:
    """Great-circle distance bins for every pair of patch centres.

    Returns:
        Integer array of shape ``(H*W, H*W)`` with bin indices in ``[0, num_bins)``.
    """
    # Step 1: place patch centres on the sphere (same grid as in _real_sh_matrix)
    theta = np.pi * (np.arange(H, dtype=np.float64) + 0.5) / H
    phi   = 2.0 * np.pi * (np.arange(W, dtype=np.float64) + 0.5) / W
    theta_flat = np.repeat(theta, W)
    phi_flat   = np.tile(phi, H)

    cos_t = np.cos(theta_flat)
    sin_t = np.sin(theta_flat)

    # Step 2: compute pairwise great-circle distance via the spherical law of cosines
    cos_dp   = np.cos(phi_flat[:, None] - phi_flat[None, :])
    cos_dist = sin_t[:, None] * sin_t[None, :] * cos_dp + cos_t[:, None] * cos_t[None, :]
    # Step 3: clamp to [-1, 1] before arccos to guard against float rounding outside the domain
    cos_dist = np.clip(cos_dist, -1.0, 1.0)
    dist = np.arccos(cos_dist)   # great-circle angle in [0, π]

    # Step 4: discretise distances into uniform bins and clip to valid range
    bins = np.floor(dist / math.pi * (num_bins - 1)).astype(np.int32)
    return np.clip(bins, 0, num_bins - 1)


class SphericalHarmonicPE(nn.Module):
    """Positional encoding from real spherical harmonics projected to embed_dim.

    The harmonic features are fixed buffers; only the linear projection is learned.
    """

    def __init__(self, H: int, W: int, embed_dim: int, max_degree: int = 8) -> None:
        super().__init__()
        sh = _real_sh_matrix(H, W, max_degree)
        # register as buffer so it moves to the right device but is not a trainable parameter
        self.register_buffer("sh_feats", torch.from_numpy(sh))
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
        return self.proj(sh).unsqueeze(0)


class SwiGLUMLP(nn.Module):
    """SwiGLU feedforward block with NormLinear projections.

    Hidden dimension is ``int(dim * mlp_ratio * 2/3)`` so that total parameter
    count matches a standard GELU MLP with the same ``mlp_ratio``.

    The gate projection *v* is rescaled by ``sqrt(dim)`` (nGPT section 2.4) to keep
    inputs to SiLU in the non-linear regime when activations are bounded in
    [-1, 1] from the normalised representation.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        # 2/3 factor compensates for the extra gate branch so param count equals a GELU MLP
        hidden = int(dim * mlp_ratio * 2 / 3)
        self.Wu = NormLinear(dim, hidden)   # value branch
        self.Wv = NormLinear(dim, hidden)   # gate branch
        self.Wo = NormLinear(hidden, dim)   # output projection back to embed_dim
        # per-neuron learnable scales: su for value, sv for gate
        self.su = nn.Parameter(torch.ones(hidden))
        self.sv = nn.Parameter(torch.ones(hidden))
        # precompute √dim: multiplying sv by this rescales the gate so SiLU stays non-linear
        self._dim_sqrt = math.sqrt(dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: compute scaled value and gate projections
        u = self.Wu(x) * self.su
        v = self.Wv(x) * (self.sv * self._dim_sqrt)
        # Step 2: apply SwiGLU gating (element-wise product of value and activated gate)
        return self.drop(self.Wo(u * F.silu(v)))


class SphericalAttention(nn.Module):
    """Multi-head self-attention on the hypersphere with optional spherical RPB.

    Key differences from standard MHSA:

    * All projection matrices use :class:`NormLinear` (unit-norm weight rows).
    * Queries and keys are per-head L2-normalised then element-wise scaled by a
      learnable per-head vector ``sqk`` (QK-norm, nGPT section 2.3).  ``sqk`` is
      stored at ``1/√head_dim`` and restored by ``×√head_dim`` each forward pass
      so the effective initial value is 1 and Adam sees gradients of the right
      magnitude.  This bounds every attention logit as a weighted cosine similarity.
    * Attention logits are scaled by ``sqrt(head_dim)`` (nGPT convention) rather
      than ``1/sqrt(head_dim)``; the learnable ``sqk`` absorbs the initial scale.
    * Spherical RPB: great-circle distance bin bias added before softmax (silently
      skipped when the sequence length does not match the stored grid, e.g. after
      masking).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        H: int | None = None,
        W: int | None = None,
        num_bins: int = 64,
        use_sph_attn_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # separate Q/K/V/O projections so each matrix has independent NormLinear rows
        self.Wq = NormLinear(dim, dim)
        self.Wk = NormLinear(dim, dim)
        self.Wv = NormLinear(dim, dim)
        self.Wo = NormLinear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # sqk is stored at 1/√head_dim and multiplied by √head_dim in forward,
        # so the effective initial value is 1 while Adam operates on a unit-scale parameter
        self._sqk_restore = self.head_dim ** 0.5
        self.sqk = nn.Parameter(
            torch.full((num_heads, 1, self.head_dim), self.head_dim ** -0.5)
        )

        self._full_N = (H * W) if (H is not None and W is not None) else None
        self.use_sph_attn_bias = use_sph_attn_bias and self._full_N is not None
        if self.use_sph_attn_bias:
            bins = _dist_bins(H, W, num_bins)  # type: ignore[arg-type]
            # store bin indices as a non-trainable buffer; bias values are the learned parameters
            self.register_buffer("dist_bins", torch.from_numpy(bins).long())
            self.bias_table = nn.Parameter(torch.zeros(num_heads, num_bins))

    def _rpb(self, n: int) -> torch.Tensor | None:
        if not self.use_sph_attn_bias or n != self._full_N:
            # skip RPB when the sequence has been masked to a non-standard length
            return None
        return self.bias_table[:, self.dist_bins].unsqueeze(0)  # type: ignore[index]

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        # Step 1: project tokens into Q, K, V and split into per-head tensors (B, H, N, D)
        q = self.Wq(x).reshape(B, N, H, D).transpose(1, 2)
        k = self.Wk(x).reshape(B, N, H, D).transpose(1, 2)
        v = self.Wv(x).reshape(B, N, H, D).transpose(1, 2)

        # Step 2: QK-norm — L2-normalise each head's Q and K, then scale by restored sqk
        sqk = self.sqk * self._sqk_restore
        q = l2norm(q) * sqk
        k = l2norm(k) * sqk

        # Step 3: compute attention scores; scale by √head_dim (not 1/√) because Q/K are unit-norm
        attn = (q @ k.transpose(-2, -1)) * self._sqk_restore

        # Step 4: add great-circle distance bias before softmax (skipped if sequence is masked)
        rpb = self._rpb(N)
        if rpb is not None:
            attn = attn + rpb
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if return_attention:
            return attn

        # Step 5: aggregate values and project back to embed_dim
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.Wo(x))


class SphericalBlock(nn.Module):
    """Transformer block with nGPT spherical residuals.

    Each sub-layer update keeps the hidden state on S^{d-1}::

        h_sub = l2norm(sub(h))
        h     = l2norm(lerp(h, h_sub, alpha))

    The per-dimension eigen learning rates ``alpha_A`` / ``alpha_M`` are
    stored at ``alpha_init / sqrt(dim)`` and restored to ``alpha_init`` during
    the forward pass, following the nGPT scale trick for Adam compatibility.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        drop: float = 0.0,
        H: int | None = None,
        W: int | None = None,
        num_bins: int = 64,
        use_sph_attn_bias: bool = True,
        alpha_init: float = 0.05,
    ) -> None:
        super().__init__()
        self.attn = SphericalAttention(
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
            H=H,
            W=W,
            num_bins=num_bins,
            use_sph_attn_bias=use_sph_attn_bias,
        )
        self.mlp = SwiGLUMLP(dim, mlp_ratio=mlp_ratio, drop=drop)

        # eigen learning rates stored at alpha_init/√dim so Adam gradient magnitudes are
        # comparable across different embed_dim values; restored by ×√dim in forward
        alpha_scale = alpha_init / math.sqrt(dim)
        self._alpha_restore = math.sqrt(dim)
        self.alpha_A = nn.Parameter(torch.full((dim,), alpha_scale))  # one rate per dim, attention
        self.alpha_M = nn.Parameter(torch.full((dim,), alpha_scale))  # one rate per dim, MLP

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        attn_out = self.attn(x, return_attention=return_attention)
        if return_attention:
            return attn_out

        # Step 1: spherical residual for the attention sub-layer
        # restore the effective eigen learning rate, then LERP on S^{d-1} and re-normalise
        alpha_A = self.alpha_A * self._alpha_restore
        x = l2norm(x.lerp(l2norm(attn_out), alpha_A))

        # Step 2: spherical residual for the MLP sub-layer (same pattern)
        alpha_M = self.alpha_M * self._alpha_restore
        x = l2norm(x.lerp(l2norm(self.mlp(x)), alpha_M))

        return x


class SphericalViT(nn.Module):
    """nGPT Vision Transformer with spherical harmonic positional encoding.

    All token representations live on the unit hypersphere S^{d-1} at every
    layer.  See the module docstring for full architecture details.

    **Training recipe:**

    1. Set ``weight_decay = 0`` — unit-norm constraints replace regularisation.
    2. After each optimiser step call ``model.norm_weights_()`` to keep
       :class:`NormLinear` weight rows on the sphere.
    3. Higher learning rates (3–5×) are safe compared with a standard ViT.

    Args:
        img_size: Input image resolution (square assumed).
        patch_size: Patch side length in pixels.
        in_chans: Number of input channels.
        embed_dim: Transformer embedding dimension *d*.
        depth: Number of transformer blocks *L*.
        num_heads: Number of attention heads.
        mlp_ratio: FFN hidden-dim expansion; actual width is
            ``int(embed_dim × mlp_ratio × 2/3)`` (SwiGLU convention).
        drop_rate: Dropout inside MLP and attention projection.
        attn_drop_rate: Dropout applied to attention weights.
        num_classes: If > 0, add GAP + NormLinear head → ``(B, num_classes)``
            for LeJEPA.  Set to 0 to return ``(B, num_patches, embed_dim)``
            for I-JEPA.
        sh_max_degree: Maximum spherical harmonic degree *L*;
            produces ``(L+1)²`` features projected to embed_dim.
        use_sph_attn_bias: Add great-circle-distance RPB to every attention layer.
        sph_attn_num_bins: Number of distance bins for the RPB lookup table.
        alpha_init: Initial effective eigen learning rate for spherical residuals.
            Defaults to ``1/depth`` (nGPT recommendation).
        qkv_bias, qk_scale, drop_path_rate, norm_layer, init_std: Accepted for
            config compatibility; unused in the nGPT formulation.
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
        sh_max_degree: int = 8,
        use_sph_attn_bias: bool = True,
        sph_attn_num_bins: int = 64,
        alpha_init: float | None = None,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_path_rate: float = 0.0,
        norm_layer: object = None,
        init_std: float = 0.02,
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

        # nGPT default: spread the total angular budget of 1 radian evenly across all layers
        _alpha = (1.0 / depth) if alpha_init is None else alpha_init
        self.blocks = nn.ModuleList(
            [
                SphericalBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop_rate,
                    drop=drop_rate,
                    H=H_p,
                    W=W_p,
                    num_bins=sph_attn_num_bins,
                    use_sph_attn_bias=use_sph_attn_bias,
                    alpha_init=_alpha,
                )
                for _ in range(depth)
            ]
        )

        if num_classes > 0:
            # LeJEPA mode: GAP + NormLinear head with a learnable per-class temperature
            self.head = NormLinear(embed_dim, num_classes)
            # logit_scale stored at 1/√num_classes, restored by ×√num_classes in forward;
            # this temperature is necessary because dot products of unit vectors are in [-1, 1],
            # which would otherwise produce a near-uniform softmax without rescaling
            self._sz_restore = math.sqrt(num_classes)
            self.logit_scale = nn.Parameter(
                torch.full((num_classes,), num_classes ** -0.5)
            )
        else:
            # I-JEPA mode: return the raw token sequence (already unit-norm)
            self.head = None
            self.logit_scale = None
            self._sz_restore = 1.0

    def norm_weights_(self) -> None:
        """Snap all NormLinear weight rows back to unit norm.

        Call after every optimiser step.  NormLinear normalises its weights
        live in ``forward``, but the stored ``weight`` tensor drifts from the
        unit sphere as the optimiser accumulates updates.  This method re-snaps
        those rows so the stored and live views stay consistent, which matters
        for weight-based operations (saving checkpoints, computing norms, etc.).
        """
        for m in self.modules():
            if isinstance(m, NormLinear):
                m.norm_weights_()

    def _get_pos_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Return PE for the current token count, recomputing for non-standard grids."""
        N = x.shape[1]
        if N == self._H_p * self._W_p:
            return self.pos_encoding()
        # variable-resolution input: recompute harmonics on a square grid of the same token count
        side = int(math.isqrt(N))
        return self.pos_encoding.recompute(side, side)

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

        # Step 1: extract patch tokens and add spherical harmonic PE, then project onto S^{d-1}
        x = self.patch_embed(x)
        x = l2norm(x + self._get_pos_encoding(x))

        # Step 2: (optional) keep only the context or target token subsets for I-JEPA masking
        if masks is not None:
            x = apply_masks(x, masks)

        # Step 3: pass through all transformer blocks; each block keeps tokens on S^{d-1}
        for block in self.blocks:
            x = block(x)

        if self.head is not None:
            # Step 4 (LeJEPA): global average pool, re-normalise, apply head with temperature
            x = l2norm(x.mean(dim=1))
            x = self.head(x) * (self.logit_scale * self._sz_restore)

        return x
