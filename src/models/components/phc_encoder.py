# Copyright (c) 2026 Pavel Snopov
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class PhCEncoder(nn.Module):
    """CNN encoder for 32x32 single-channel photonic crystal unit cells."""

    def __init__(self, emb_dim: int = 1024) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 256, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 256, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.head = nn.Sequential(
            nn.Linear(4 * 4 * 256, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise RuntimeError(f"Expected PhC input with shape (N, 1, H, W), got {tuple(x.shape)}")
        return self.head(self.cnn(x).flatten(1))
