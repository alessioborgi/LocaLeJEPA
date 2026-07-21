# Copyright (c) 2022 Alexander Ermolov
# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo,
# Randall Balestriero, Claudio Battiloro
# SPDX-License-Identifier: MIT

"""Numerically stable operations for the Poincare ball model adapted from https://github.com/htdt/hyp_metric."""

import math

import torch
import torch.nn as nn


def project(x: torch.Tensor, c: float, eps: float = 1e-3) -> torch.Tensor:
    """Project points into the open Poincare ball for curvature ``-c``."""
    max_norm = (1.0 - eps) / math.sqrt(c)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)
    scale = (max_norm / norm).clamp(max=1.0)
    return x * scale


def expmap0(u: torch.Tensor, c: float) -> torch.Tensor:
    """Map a tangent vector at the origin to the Poincare ball."""
    sqrt_c = math.sqrt(c)
    norm = u.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(u.dtype).tiny)
    mapped = torch.tanh(sqrt_c * norm) * u / (sqrt_c * norm)
    return project(mapped, c=c)


class ToPoincare(nn.Module):
    """Map Euclidean features to a Poincare ball."""

    def __init__(self, c: float = 0.1, clip_radius: float | None = 2.3) -> None:
        super().__init__()
        if c <= 0:
            raise ValueError(f"Poincare-ball curvature parameter c must be positive, got {c}.")
        if clip_radius is not None and clip_radius <= 0:
            raise ValueError(f"clip_radius must be positive or None, got {clip_radius}.")
        self.c = float(c)
        self.clip_radius = clip_radius

    @property
    def ball_radius(self) -> float:
        return self.c**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.clip_radius is not None:
            norm = x.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)
            x = x * (self.clip_radius / norm).clamp(max=1.0)
        return expmap0(x, c=self.c)

    def extra_repr(self) -> str:
        return f"c={self.c}, clip_radius={self.clip_radius}"
