# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
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

    def __call__(self, image: Any) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        global_views = [self.global_transform(image) for _ in range(self.num_global_crops)]
        local_views = (
            [self.local_transform(image) for _ in range(self.num_local_crops)]
            if self.local_transform is not None
            else []
        )
        if not local_views:
            return torch.stack(global_views)
        if global_views[0].shape == local_views[0].shape:
            # Same resolution for every view: keep the historical (V, C, H, W) layout.
            return torch.stack(global_views + local_views)
        # Mixed resolutions (e.g. 224 px global / 98 px local as in the LeJEPA paper):
        # return one stacked tensor per resolution group. The default collate turns this
        # into ``[(N, V_g, C, H_g, W_g), (N, V_l, C, H_l, W_l)]`` and LeJEPAModule.encode
        # accepts that grouped layout (global views still come first).
        return torch.stack(global_views), torch.stack(local_views)
