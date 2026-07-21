# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# E2ResNet is vendored from equivision (https://github.com/dmklee/equivision),
# Copyright (c) 2023 David Klee, MIT License ("A Comparison of Equivariant
# Vision Models with ImageNet Pre-training", NeurReps 2023). It implements a
# ResNet that is equivariant to discrete rotations and reflections in the 2D
# plane, based on the equivariant wide ResNet from
#     https://github.com/QUVA-Lab/escnn/blob/master/examples/e2wrn.py
# and the PyTorch implementation of ResNet from
#     https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
#
# The ``build_e2resnet`` factory at the bottom exposes the equivision model zoo
# (group x depth, parameter-matched base widths) with a configurable output
# dimension so the network can be dropped into the JEPA modules as an encoder
# mapping ``(B, C, H, W) -> (B, num_classes)``.

from typing import Callable, List, Optional, Type, Union

import torch
from escnn import gspaces, nn
from escnn.nn import FieldType, GeometricTensor
from torch import Tensor


def conv7x7(
    in_type: FieldType,
    out_type: FieldType,
    stride: int = 1,
    padding: int = 3,
    dilation: int = 1,
    bias: bool = False,
    initialize: bool = True,
) -> nn.EquivariantModule:
    """7x7 group convolution with padding"""
    return nn.R2Conv(
        in_type,
        out_type,
        7,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        initialize=initialize,
    )


def conv3x3(
    in_type: FieldType,
    out_type: FieldType,
    stride: int = 1,
    padding: int = 1,
    dilation: int = 1,
    bias: bool = False,
    initialize: bool = True,
) -> nn.EquivariantModule:
    """3x3 group convolution with padding"""
    return nn.R2Conv(
        in_type,
        out_type,
        3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        initialize=initialize,
    )


def conv1x1(
    in_type: FieldType,
    out_type: FieldType,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    bias: bool = False,
    initialize: bool = True,
) -> nn.EquivariantModule:
    """1x1 group convolution"""
    return nn.R2Conv(
        in_type,
        out_type,
        1,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        initialize=initialize,
    )


class E2BasicBlock(torch.nn.Module):
    expansion: int = 1

    def __init__(
        self,
        gspace: gspaces.GSpace2D,
        inplanes: int,
        planes: int,
        stride: int,
        downsample: Optional[Callable] = None,
        norm_layer: Optional[Callable] = nn.InnerBatchNorm,
        initialize: bool = True,
    ):
        super().__init__()
        in_type = nn.FieldType(gspace, inplanes * [gspace.regular_repr])
        out_type = nn.FieldType(gspace, planes * [gspace.regular_repr])

        self.conv1 = conv3x3(in_type, out_type, stride=stride, initialize=initialize)
        self.bn1 = norm_layer(out_type)
        self.relu = nn.ReLU(out_type, True)
        self.conv2 = conv3x3(out_type, out_type, stride=1, initialize=initialize)
        self.bn2 = norm_layer(out_type)
        self.downsample = downsample

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class E2BottleNeck(torch.nn.Module):
    expansion: int = 4

    def __init__(
        self,
        gspace: gspaces.GSpace2D,
        inplanes: int,
        planes: int,
        stride: int,
        downsample: Optional[Callable] = None,
        norm_layer: Optional[Callable] = nn.InnerBatchNorm,
        initialize: bool = True,
    ):
        super().__init__()
        in_type = FieldType(gspace, inplanes * [gspace.regular_repr])
        out_type = FieldType(gspace, planes * [gspace.regular_repr])
        exp_type = FieldType(gspace, self.expansion * planes * [gspace.regular_repr])

        self.conv1 = conv1x1(in_type, out_type, initialize=initialize)
        self.bn1 = norm_layer(out_type)
        self.relu1 = nn.ReLU(out_type, True)

        self.conv2 = conv3x3(out_type, out_type, stride=stride, initialize=initialize)
        self.bn2 = norm_layer(out_type)
        self.relu2 = nn.ReLU(out_type, True)

        self.conv3 = conv1x1(out_type, exp_type, initialize=initialize)
        self.bn3 = norm_layer(exp_type)
        self.relu3 = nn.ReLU(exp_type, True)

        self.downsample = downsample

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out


class E2ResNet(torch.nn.Module):
    def __init__(
        self,
        gspace: gspaces.GSpace2D,
        block: Type[Union[E2BasicBlock, E2BottleNeck]],
        layers: List[int],
        num_classes: int,
        base_width: int,
        initialize: bool = True,
    ):
        super().__init__()

        self.gspace = gspace
        self.in_type = FieldType(gspace, 3 * [gspace.trivial_repr])

        self.norm_layer = nn.InnerBatchNorm
        self.dilation = 1
        self.base_width = base_width
        self.inplanes = self.base_width

        out_type = FieldType(gspace, self.base_width * [gspace.regular_repr])
        self.conv1 = conv7x7(self.in_type, out_type, stride=2, initialize=initialize)
        self.bn1 = self.norm_layer(out_type)
        self.maxpool = nn.PointwiseMaxPool(out_type, kernel_size=3, stride=2, padding=1)

        self.relu1 = nn.ReLU(out_type, True)
        self.layer1 = self._make_layer(gspace, block, base_width, layers[0])
        self.layer2 = self._make_layer(
            gspace, block, base_width * 2, layers[1], stride=2, initialize=initialize
        )
        self.layer3 = self._make_layer(
            gspace, block, base_width * 4, layers[2], stride=2, initialize=initialize
        )
        self.layer4 = self._make_layer(
            gspace, block, base_width * 8, layers[3], stride=2, initialize=initialize
        )

        fmap_type = FieldType(
            gspace, block.expansion * base_width * 8 * [gspace.regular_repr]
        )
        self.avgpool = nn.PointwiseAdaptiveAvgPool(fmap_type, (1, 1))

        self.gpool = nn.GroupPooling(fmap_type)
        self.fc = torch.nn.Linear(block.expansion * base_width * 8, num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        gspace: gspaces.GSpace,
        block: Type[Union[E2BasicBlock, E2BottleNeck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        initialize: bool = True,
    ) -> torch.nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            in_type = FieldType(gspace, self.inplanes * [gspace.regular_repr])
            out_type = FieldType(
                gspace, planes * block.expansion * [gspace.regular_repr]
            )
            downsample = nn.SequentialModule(
                conv1x1(in_type, out_type, stride=stride, initialize=initialize),
                self.norm_layer(out_type),
            )

        layers = []
        layers.append(
            block(
                gspace,
                self.inplanes,
                planes,
                stride,
                downsample,
                self.norm_layer,
                initialize=initialize,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    gspace,
                    self.inplanes,
                    planes,
                    1,
                    None,
                    self.norm_layer,
                    initialize=initialize,
                )
            )

        return torch.nn.Sequential(*layers)

    def forward_features(self, x: Tensor) -> GeometricTensor:
        """Returns feature map containing regular representation"""
        x = GeometricTensor(x, self.in_type)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        """Returns a G-invariant embedding of shape (B, num_classes)"""
        x = self.forward_features(x)

        x = self.avgpool(x)
        x = self.gpool(x).tensor.squeeze(-2).squeeze(-1)
        return self.fc(x)


# --- equivision model zoo, exposed as a single Hydra-friendly factory ---------

_GSPACES: dict[str, Callable[[], gspaces.GSpace2D]] = {
    "c1": gspaces.trivialOnR2,
    "d1": gspaces.flip2dOnR2,
    "c4": lambda: gspaces.rot2dOnR2(N=4),
    "d4": lambda: gspaces.flipRot2dOnR2(N=4),
    "c8": lambda: gspaces.rot2dOnR2(N=8),
    "d8": lambda: gspaces.flipRot2dOnR2(N=8),
}

_ARCHS: dict[int, tuple[type, list[int]]] = {
    18: (E2BasicBlock, [2, 2, 2, 2]),
    50: (E2BottleNeck, [3, 4, 6, 3]),
    101: (E2BottleNeck, [3, 4, 23, 3]),
}

# (depth, group) -> (base width matching vanilla ResNet param count, fast width
# matching vanilla filter size after basis expansion); values from equivision.
_BASE_WIDTHS: dict[tuple[int, str], tuple[int, int]] = {
    (18, "c1"): (78, 64),
    (18, "d1"): (55, 32),
    (18, "c4"): (39, 16),
    (18, "d4"): (28, 8),
    (18, "c8"): (28, 8),
    (18, "d8"): (20, 3),
    (50, "c1"): (70, 64),
    (50, "d1"): (50, 32),
    (50, "c4"): (35, 16),
    (50, "d4"): (25, 8),
    (50, "c8"): (25, 8),
    (101, "c1"): (70, 64),
    (101, "d1"): (50, 32),
    (101, "c4"): (35, 16),
    (101, "d4"): (25, 8),
}


def build_e2resnet(
    group: str = "d4",
    depth: int = 50,
    num_classes: int = 512,
    fixed_params: bool = True,
    initialize: bool = True,
) -> E2ResNet:
    """Build a group-equivariant ResNet encoder from the equivision zoo.

    The returned model maps images ``(B, 3, H, W)`` to G-invariant embeddings
    ``(B, num_classes)`` (group pooling + spatial pooling + linear head), so it
    can be used directly as a JEPA encoder with ``num_classes=embedding_dim``.

    Args:
        group: symmetry group — one of ``c1, d1, c4, d4, c8, d8`` (cN =
            rotations by 360/N degrees, dN = rotations + horizontal flips).
        depth: ResNet depth — 18, 50, or 101.
        num_classes: output embedding dimension.
        fixed_params: if True, use the base width that matches the vanilla
            ResNet's trainable-parameter count (equivision Table 1); if False,
            use the smaller "fast" width matching the vanilla filter size.
        initialize: run escnn's equivariant weight initialization (slow at
            construction time; disable only for shape tests or before loading
            a checkpoint).
    """
    key = (depth, group)
    if group not in _GSPACES or depth not in _ARCHS or key not in _BASE_WIDTHS:
        raise ValueError(
            f"Unsupported equivariant ResNet: group={group!r}, depth={depth}. "
            f"Available: {sorted(_BASE_WIDTHS)}"
        )

    block, layers = _ARCHS[depth]
    base_width = _BASE_WIDTHS[key][0 if fixed_params else 1]

    model = E2ResNet(
        gspace=_GSPACES[group](),
        block=block,
        layers=layers,
        num_classes=num_classes,
        base_width=base_width,
        initialize=initialize,
    )
    model.name = f"{group}resnet{depth}" + ("" if fixed_params else "-fast")
    return model
