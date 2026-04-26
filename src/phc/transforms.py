# Copyright (c) 2025 Alessio Borgioli
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchvision import transforms


class RandomC4v:
    """Uniformly sample one of the 8 C4v symmetry operations.

    4 rotations (0°, 90°, 180°, 270°) × 2 (identity + horizontal flip).
    Operates on a (C, H, W) float tensor; H and W must be equal.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        k    = torch.randint(4, (1,)).item()
        x    = torch.rot90(x, k, dims=(-2, -1))
        if torch.rand(1).item() < 0.5:
            x = x.flip(-1)
        return x


class RandomMirror:
    """Uniformly sample a horizontal mirror, vertical mirror, or identity.

    Useful as an additional symmetry op for Gpm-type crystals.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        op = torch.randint(3, (1,)).item()
        if op == 0:
            return x.flip(-1)
        if op == 1:
            return x.flip(-2)
        return x  # identity


class RandomRoll:
    """Toroidal (periodic) translation by a uniformly random (dy, dx) offset."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2], x.shape[-1]
        dy   = torch.randint(H, (1,)).item()
        dx   = torch.randint(W, (1,)).item()
        return x.roll((dy, dx), dims=(-2, -1))


class RandomRefractiveScaling:
    """Scale permittivity pixel values by a uniform random factor in [lo, hi].

    Simulates variation in refractive index while preserving crystal topology.
    """

    def __init__(self, lo: float = 0.8, hi: float = 1.2):
        self.lo = lo
        self.hi = hi

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.lo + (self.hi - self.lo) * torch.rand(1).item()
        return x * scale


class PhCMultiViewTransform:
    """Generate V independently augmented views of a unit-cell tensor.

    Each sample in the DataLoader receives a (V, 1, H, W) stacked tensor.
    """

    def __init__(self, base_transform, num_views: int = 2):
        self.base = base_transform
        self.V    = num_views

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.base(x) for _ in range(self.V)])


def make_phc_single_transform(
    use_c4v:          bool  = True,
    use_mirror:       bool  = False,
    use_roll:         bool  = True,
    use_refractive:   bool  = True,
    refractive_range: tuple = (0.8, 1.2),
):
    """Single-view augmentation pipeline for I-JEPA (returns one (1, 32, 32) tensor).

    The I-JEPA masking scheme creates multiple context/target views by masking
    patches of a single augmented image, so only one view per sample is needed.
    """
    ops = []
    if use_c4v:
        ops.append(RandomC4v())
    if use_mirror:
        ops.append(RandomMirror())
    if use_roll:
        ops.append(RandomRoll())
    if use_refractive:
        ops.append(RandomRefractiveScaling(*refractive_range))
    return transforms.Compose(ops) if ops else (lambda x: x)


def make_phc_transforms(
    num_views:        int   = 2,
    use_c4v:          bool  = True,
    use_mirror:       bool  = False,
    use_roll:         bool  = True,
    use_refractive:   bool  = True,
    refractive_range: tuple = (0.8, 1.2),
) -> PhCMultiViewTransform:
    """Build the augmentation pipeline for PhC self-supervised pretraining.

    The base transform receives a (1, 32, 32) float32 tensor normalised to [0, 1].
    Symmetry ops are applied in order: C4v → mirror → roll → refractive scaling.
    """
    ops = []
    if use_c4v:
        ops.append(RandomC4v())
    if use_mirror:
        ops.append(RandomMirror())
    if use_roll:
        ops.append(RandomRoll())
    if use_refractive:
        ops.append(RandomRefractiveScaling(*refractive_range))

    base = transforms.Compose(ops) if ops else (lambda x: x)
    return PhCMultiViewTransform(base, num_views=num_views)
