# Copyright (c) 2025 Alessio Borgioli
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torchvision.ops import MLP


class PhCEncoder(nn.Module):
    """CNN encoder for 32×32 single-channel photonic crystal unit cells.

    Architecture:
        3 × (Conv7×7 → BN → ReLU → MaxPool2×2): spatial 32 → 16 → 8 → 4
        Flatten: 4×4×256 = 4096
        2 × Linear: 4096 → 1024 → emb_dim
        MLP projector: emb_dim → 512 → proj_dim

    Interface matches LeJEPA encoders:
        forward(x)
            x    : (N, V, 1, 32, 32)  multi-view pretraining
                   (N, 1, 32, 32)     single-view eval
            emb  : (N*V, emb_dim)
            proj : (V, N, proj_dim)
    """

    def __init__(self, emb_dim: int = 1024, proj_dim: int = 128):
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
        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 256, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, emb_dim),
        )
        self.proj = MLP(emb_dim, [512, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x: torch.Tensor):
        if x.dim() == 4:
            # Single-view eval path: (N, 1, 32, 32)
            emb  = self.fc(self.cnn(x).flatten(1))
            proj = self.proj(emb).unsqueeze(0)   # (1, N, proj_dim)
            return emb, proj

        N, V = x.shape[:2]
        feats = self.cnn(x.flatten(0, 1)).flatten(1)          # (N*V, 4096)
        emb   = self.fc(feats)                                  # (N*V, emb_dim)
        proj  = self.proj(emb).reshape(N, V, -1).transpose(0, 1)  # (V, N, proj_dim)
        return emb, proj


class DOSPredictor(nn.Module):
    """4-layer MLP that regresses density-of-states vectors from embeddings.

    emb_dim → 1024 → 1024 → 512 → 400
    """

    def __init__(self, emb_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024,    1024), nn.ReLU(inplace=True),
            nn.Linear(1024,     512), nn.ReLU(inplace=True),
            nn.Linear(512,      400),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)
