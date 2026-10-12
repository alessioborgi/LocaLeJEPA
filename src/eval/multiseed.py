# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Repeat a frozen-feature linear probe over several seeds.

The probe learning rate is selected once, on the held-out slice of the training
split, exactly as in the single-seed protocol. The extra seeds then refit the
head with that learning rate, so the reported spread measures the variability of
the *readout* (initialisation, data order, held-out split) with the pretrained
encoder and the selected hyperparameter held fixed. It says nothing about the
variability of pretraining, for which each encoder would have to be retrained.
"""

from __future__ import annotations

import logging
from statistics import mean, stdev

import torch

from .linear_probe import run_linear_probe

logger = logging.getLogger(__name__)


def linear_probe_multiseed(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    *,
    num_classes: int,
    seeds: int = 1,
    base_seed: int = 0,
    search_per_seed: bool = False,
    tag: str = "",
    **probe_kwargs,
) -> dict:
    """Run ``run_linear_probe`` over ``seeds`` seeds and summarise the spread.

    Returns the seed-``base_seed`` result dict (so every existing field keeps its
    meaning) with ``top1_seeds``/``top5_seeds`` and the mean/std added. With
    ``seeds <= 1`` the call is identical to ``run_linear_probe``.
    """
    first = run_linear_probe(
        train_features, train_labels, test_features, test_labels,
        num_classes=num_classes, seed=base_seed, **probe_kwargs,
    )
    top1s, top5s, lrs = [first["top1"]], [first["top5"]], [first["best_lr"]]

    extra = dict(probe_kwargs)
    if not search_per_seed:
        # Large training sets select the learning rate stably, so it is chosen once and
        # the extra seeds only refit. On small datasets pass search_per_seed=True: there
        # the held-out slice is a few hundred images and the selected rate itself varies,
        # so each seed repeats the search and the spread includes that variability.
        extra["lr_grid"] = [first["best_lr"]]
    for s in range(base_seed + 1, base_seed + int(seeds)):
        r = run_linear_probe(
            train_features, train_labels, test_features, test_labels,
            num_classes=num_classes, seed=s, **extra,
        )
        top1s.append(r["top1"]); top5s.append(r["top5"]); lrs.append(r["best_lr"])
        logger.info("%s seed %d: linear top-1 = %.2f%%", tag or "probe", s, r["top1"])

    first["lr_seeds"] = lrs
    first["search_per_seed"] = bool(search_per_seed)
    first["top1_seeds"] = top1s
    first["top5_seeds"] = top5s
    first["n_seeds"] = len(top1s)
    first["top1_mean"] = float(mean(top1s))
    first["top5_mean"] = float(mean(top5s))
    first["top1_std"] = float(stdev(top1s)) if len(top1s) > 1 else 0.0
    first["top5_std"] = float(stdev(top5s)) if len(top5s) > 1 else 0.0
    if len(top1s) > 1:
        logger.info("%s: linear top-1 = %.2f +/- %.2f %% over %d seeds",
                    tag or "probe", first["top1_mean"], first["top1_std"], len(top1s))
    return first
