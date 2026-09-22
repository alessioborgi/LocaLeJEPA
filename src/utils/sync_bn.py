# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Synchronized BatchNorm for models that contain escnn containers.

``torch.nn.SyncBatchNorm.convert_sync_batchnorm`` rebuilds every container by re-adding its
children through ``add_module``. escnn's ``SequentialModule.add_module`` type-checks each
insertion against the container's *current* output type and refuses the rebuild
(``AssertionError: [...{regular (x25)}(200)] != [...{regular (x100)}(800)]``) -- which is how
every D4-RN50 launch under ``trainer.sync_batchnorm: true`` died on the cluster (F48). Lightning
applies that conversion on every multi-GPU launch and never on one GPU, so no single-GPU test
could see it.

This does the same replacement -- every ``_BatchNorm`` becomes a ``SyncBatchNorm`` carrying the
same affine parameters and running statistics -- but writes the new module straight into the
parent's module table, so no container is asked to re-admit a child. Plain torch containers
(the projector's MLP, the online probe) end up exactly as torch's conversion leaves them.
"""

from __future__ import annotations

import torch


def convert_sync_batchnorm_inplace(module: torch.nn.Module, process_group=None) -> torch.nn.Module:
    """Replace every BatchNorm under ``module`` with a ``SyncBatchNorm``, in place; returns ``module``."""
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            new = torch.nn.SyncBatchNorm(
                child.num_features, child.eps, child.momentum, child.affine, child.track_running_stats, process_group,
            )
            if child.affine:
                with torch.no_grad():
                    new.weight = child.weight
                    new.bias = child.bias
            new.running_mean = child.running_mean
            new.running_var = child.running_var
            new.num_batches_tracked = child.num_batches_tracked
            new.training = child.training
            if hasattr(child, "qconfig"):
                new.qconfig = child.qconfig
            module._modules[name] = new          # never parent.add_module: escnn refuses the re-admission
        else:
            convert_sync_batchnorm_inplace(child, process_group)
    return module
