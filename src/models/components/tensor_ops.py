# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def apply_masks(x: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
    """Gather patch tokens selected by each mask.

    Args:
        x: Tensor shaped ``(B, N, D)``.
        masks: List of integer tensors shaped ``(B, K)``.

    Returns:
        Tensor shaped ``(len(masks) * B, K, D)``.
    """
    masked = []
    for mask in masks:
        index = mask.unsqueeze(-1).expand(-1, -1, x.size(-1))
        masked.append(torch.gather(x, dim=1, index=index))
    return torch.cat(masked, dim=0)


def repeat_interleave_batch(x: torch.Tensor, batch_size: int, repeat: int) -> torch.Tensor:
    """Repeat each logical batch block while preserving I-JEPA mask ordering."""
    num_blocks = len(x) // batch_size
    return torch.cat(
        [
            torch.cat([x[i * batch_size : (i + 1) * batch_size] for _ in range(repeat)], dim=0)
            for i in range(num_blocks)
        ],
        dim=0,
    )
