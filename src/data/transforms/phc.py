# Copyright (c) 2025 Alessio Borgi, Pavel Snopov
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable

import torch


class RandomC4v:
    """Sample one C4v symmetry operation for a square PhC tensor."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        k = int(torch.randint(4, (1,)).item())
        x = torch.rot90(x, k, dims=(-2, -1))
        if torch.rand(1).item() < 0.5:
            x = x.flip(-1)
        return x


class RandomMirror:
    """Sample one axis mirror or identity."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        op = torch.randint(3, (1,)).item()
        if op == 0:
            return x.flip(-1)
        if op == 1:
            return x.flip(-2)
        return x


class RandomRoll:
    """Apply a continuous periodic translation to a PhC unit cell."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        dy = int(torch.randint(height, (1,)).item())
        dx = int(torch.randint(width, (1,)).item())
        return x.roll((dy, dx), dims=(-2, -1))


class RandomRefractiveScaling:
    """Scale normalized permittivity values while keeping them in bounds.

    The PhC data is expected to be normalized as ``(eps - eps_min) /
    (eps_max - eps_min)``. This transform samples ``s`` or ``1 / s`` and
    retries until the scaled physical permittivity stays within bounds.
    """

    def __init__(
        self,
        max_scale: float = 2.0,
        eps_min: float = 1.0,
        eps_max: float = 20.0,
        max_attempts: int = 100,
    ) -> None:
        self.max_scale = max_scale
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.max_attempts = max_attempts

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        eps = x * (self.eps_max - self.eps_min) + self.eps_min

        for _ in range(self.max_attempts):
            scale = 1.0 + (self.max_scale - 1.0) * torch.rand((), device=x.device, dtype=x.dtype)
            if torch.rand((), device=x.device) < 0.5:
                scale = scale.reciprocal()

            scaled = eps * scale
            if scaled.min() >= self.eps_min and scaled.max() <= self.eps_max:
                return (scaled - self.eps_min) / (self.eps_max - self.eps_min)

        return x


class PhCMultiViewTransform:
    """Apply a PhC transform independently and stack multiple DOS-invariant views."""

    def __init__(self, base_transform: Callable[[torch.Tensor], torch.Tensor], num_views: int = 2) -> None:
        self.base_transform = base_transform
        self.num_views = num_views

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.base_transform(x) for _ in range(self.num_views)])
