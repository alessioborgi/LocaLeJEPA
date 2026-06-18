# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from logging import getLogger
from multiprocessing import Value
from typing import Any

import torch
from torch.utils.data import default_collate

logger = getLogger(__name__)


class MultiBlockMaskCollator:
    """Collate image batches and sample I-JEPA encoder/predictor patch masks."""

    def __init__(
        self,
        input_size: int | tuple[int, int] = 224,
        patch_size: int = 16,
        enc_mask_scale: tuple[float, float] = (0.2, 0.8),
        pred_mask_scale: tuple[float, float] = (0.2, 0.8),
        aspect_ratio: tuple[float, float] = (0.3, 3.0),
        nenc: int = 1,
        npred: int = 2,
        min_keep: int = 4,
        allow_overlap: bool = False,
    ) -> None:
        if isinstance(input_size, int):
            input_size = (input_size, input_size)

        self.patch_size = patch_size
        self.height = input_size[0] // patch_size
        self.width = input_size[1] // patch_size
        if self.height < 2 or self.width < 2:
            raise ValueError("I-JEPA masks require at least a 2x2 patch grid.")
        self.enc_mask_scale = tuple(enc_mask_scale)
        self.pred_mask_scale = tuple(pred_mask_scale)
        self.aspect_ratio = tuple(aspect_ratio)
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self._iteration = Value("i", -1)

    def step(self) -> int:
        with self._iteration.get_lock():
            self._iteration.value += 1
            return self._iteration.value

    def _sample_block_size(
        self,
        generator: torch.Generator,
        scale: tuple[float, float],
        aspect_ratio: tuple[float, float],
    ) -> tuple[int, int]:
        min_scale, max_scale = scale
        min_ratio, max_ratio = aspect_ratio

        mask_scale = min_scale + torch.rand((), generator=generator).item() * (max_scale - min_scale)
        ratio = min_ratio + torch.rand((), generator=generator).item() * (max_ratio - min_ratio)
        max_keep = int(self.height * self.width * mask_scale)

        h = max(1, int(round(math.sqrt(max_keep * ratio))))
        w = max(1, int(round(math.sqrt(max_keep / ratio))))
        h = min(h, self.height - 1)
        w = min(w, self.width - 1)
        return h, w

    def _sample_block_mask(
        self,
        block_size: tuple[int, int],
        acceptable_regions: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = block_size
        timeout = 20
        relaxed_regions = 0

        while True:
            top = torch.randint(0, self.height - h + 1, ()).item()
            left = torch.randint(0, self.width - w + 1, ()).item()

            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top : top + h, left : left + w] = 1

            if acceptable_regions is not None:
                for region in acceptable_regions[: max(len(acceptable_regions) - relaxed_regions, 0)]:
                    mask *= region

            indices = torch.nonzero(mask.flatten()).squeeze()
            if indices.numel() > self.min_keep:
                complement = torch.ones((self.height, self.width), dtype=torch.int32)
                complement[top : top + h, left : left + w] = 0
                return indices, complement

            timeout -= 1
            if timeout == 0:
                relaxed_regions += 1
                timeout = 20
                logger.warning("Valid I-JEPA mask not found; relaxing overlap constraints.")

    def __call__(self, batch: list[Any]) -> tuple[Any, list[torch.Tensor], list[torch.Tensor]]:
        batch_size = len(batch)
        collated_batch = default_collate(batch)

        generator = torch.Generator()
        generator.manual_seed(self.step())

        pred_block_size = self._sample_block_size(generator, self.pred_mask_scale, self.aspect_ratio)
        enc_block_size = self._sample_block_size(generator, self.enc_mask_scale, (1.0, 1.0))

        batch_pred_masks: list[list[torch.Tensor]] = []
        batch_enc_masks: list[list[torch.Tensor]] = []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width

        for _ in range(batch_size):
            pred_masks = []
            complements = []
            for _ in range(self.npred):
                mask, complement = self._sample_block_mask(pred_block_size)
                pred_masks.append(mask)
                complements.append(complement)
                min_keep_pred = min(min_keep_pred, mask.numel())
            batch_pred_masks.append(pred_masks)

            acceptable_regions = None if self.allow_overlap else complements
            enc_masks = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(enc_block_size, acceptable_regions=acceptable_regions)
                enc_masks.append(mask)
                min_keep_enc = min(min_keep_enc, mask.numel())
            batch_enc_masks.append(enc_masks)

        batch_pred_masks = [[mask[:min_keep_pred] for mask in masks] for masks in batch_pred_masks]
        batch_enc_masks = [[mask[:min_keep_enc] for mask in masks] for masks in batch_enc_masks]

        return collated_batch, default_collate(batch_enc_masks), default_collate(batch_pred_masks)
