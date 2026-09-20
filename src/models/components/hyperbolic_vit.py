# Copyright (c) 2022 Alexander Ermolov
# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo,
# Randall Balestriero, Claudio Battiloro

"""Hyperbolic ViT encoder adapted from the official ``htdt/hyp_metric`` code.

Reference: Ermolov et al., "Hyperbolic Vision Transformers: Combining
Improvements in Metric Learning", CVPR 2022 (arXiv:2203.10833).
Official code: https://github.com/htdt/hyp_metric (MIT license).
"""

import math
import warnings

import timm
import torch
import torch.nn as nn

from src.models.components.hyperbolic import ToPoincare


class HyperbolicViT(nn.Module):
    """timm ViT with a Poincare-ball metric-learning head.

    Args:
        model_name: timm ViT model name.
        embedding_dim: Output dimension of the Poincare ball.
        hyp_c: Positive ``c`` for constant sectional curvature ``-c``.
        clip_r: Maximum tangent-vector norm before the exponential map. The
            official implementation uses ``2.3``.
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch16_224",
        embedding_dim: int = 128,
        pretrained: bool = False,
        img_size: int = 224,
        in_chans: int = 3,
        drop_path_rate: float = 0.0,
        hyp_c: float = 0.1,
        clip_r: float | None = 2.3,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}.")

        self.body = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_chans,
            drop_path_rate=drop_path_rate,
            img_size=img_size,
        )

        # hyp_c and clip_r are NOT independently sweepable. expmap0 composes tanh with a
        # projection that clamps at (1 - 1e-3)/sqrt(c), so once tanh(sqrt(c) * clip_r) approaches
        # 1 every tangent vector lands on the same shell and all magnitude information is
        # destroyed -- silently, with no error and a perfectly healthy-looking loss. At c=1.0
        # with 16-dim features of ordinary norm the map is already saturated; the configured
        # c=0.1 / clip_r=2.3 reaches 0.62 of the ball radius and is fine.
        if clip_r is not None:
            reach = math.tanh(math.sqrt(hyp_c) * clip_r)
            if reach > 0.95:
                warnings.warn(
                    f"HyperbolicViT: hyp_c={hyp_c} with clip_r={clip_r} reaches {reach:.4f} of "
                    f"the ball radius, so the exponential map is saturated -- distinct features "
                    f"will collapse onto one shell and their magnitudes will be lost. Lower "
                    f"hyp_c or clip_r (the campaign uses 0.1 / 2.3, which reaches 0.62).",
                    RuntimeWarning, stacklevel=2,
                )

        self.tangent_head = nn.Linear(self.body.num_features, embedding_dim)
        self.to_poincare = ToPoincare(c=hyp_c, clip_radius=clip_r)
        nn.init.zeros_(self.tangent_head.bias)
        nn.init.orthogonal_(self.tangent_head.weight)

    @property
    def ball_radius(self) -> float:
        return self.to_poincare.ball_radius

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.body(x)
        return self.to_poincare(self.tangent_head(features))
