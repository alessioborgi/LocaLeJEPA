# Copyright (c) 2025 Alessio Borgioli
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""SimCLR encoder: ResNet-50 backbone + 2-layer MLP projector.

Forward interface matches LeJEPA:
    x    : (N, V, C, H, W)
    proj : (V, N, proj_dim)
"""

import torch.nn as nn
import torchvision.models as tvmodels
from torchvision.ops import MLP


class SimCLREncoder(nn.Module):

    def __init__(self, proj_dim: int = 128):
        super().__init__()
        backbone = tvmodels.resnet50(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = MLP(2048, [2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        N, V = x.shape[:2]
        emb  = self.backbone(x.flatten(0, 1))       # (N*V, 2048)
        proj = self.proj(emb).reshape(N, V, -1).transpose(0, 1)  # (V, N, proj_dim)
        return emb, proj
