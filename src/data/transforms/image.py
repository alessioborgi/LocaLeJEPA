# Copyright (c) 2026 Pavel Snopov
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from collections.abc import Callable
from typing import Any

import torch


class MultiViewTransform:
    """Apply one image transform independently several times and stack views."""

    def __init__(self, base_transform: Callable[[Any], torch.Tensor], num_views: int) -> None:
        self.base_transform = base_transform
        self.num_views = num_views

    def __call__(self, image: Any) -> torch.Tensor:
        return torch.stack([self.base_transform(image) for _ in range(self.num_views)])


class MultiCropTransform:
    """Apply global transforms first, then local transforms, and stack views.

    The returned view order is part of the LeJEPA contract: the first
    ``num_global_crops`` views are global anchors for the invariance loss.
    """

    def __init__(
        self,
        global_transform: Callable[[Any], torch.Tensor],
        local_transform: Callable[[Any], torch.Tensor] | None = None,
        num_global_crops: int = 2,
        num_local_crops: int = 0,
    ) -> None:
        self.global_transform = global_transform
        self.local_transform = local_transform
        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops

    def __call__(self, image: Any) -> torch.Tensor:
        views = []

        for transform, num_crops in ((self.global_transform, self.num_global_crops), (self.local_transform, self.num_local_crops)):
            if transform is None:
                continue
            for _ in range(num_crops):
                views.append(transform(image))
        return torch.stack(views)
