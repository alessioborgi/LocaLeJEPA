# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""timm encoder factory that can turn on activation checkpointing."""

import timm
from torch import nn


def timm_encoder(grad_checkpointing: bool = False, **kwargs) -> nn.Module:
    model = timm.create_model(**kwargs)
    if grad_checkpointing:
        model.set_grad_checkpointing()
    return model
