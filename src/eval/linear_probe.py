# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Linear probe on cached frozen features (SPHERE-JEPA Table 3, ``Linear (%)``).

A single ``nn.Linear`` is trained by SGD on features extracted once with the
deterministic eval transform (resize + centre crop), exactly the view the online
probe sees during ``validation_step``.  The learning rate is chosen on a
held-out slice of the train split, then the probe is refit on the full train
split and reported on the pretraining dataset's validation split.

Training on cached features (rather than re-encoding augmented crops each epoch)
keeps the whole sweep at minutes-per-backbone instead of GPU-days; it is the
standard frozen-feature probe and matches the online-probe protocol used during
pretraining, so the numbers are directly comparable to ``val/probe_acc``.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _standardize_stats(features: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    mean = features.float().mean(dim=0)
    std = features.float().std(dim=0).clamp_min(eps)
    return mean, std


@torch.no_grad()
def _evaluate(
    weight: torch.Tensor,
    bias: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
    device: torch.device,
    batch_size: int = 8192,
) -> tuple[float, float]:
    """Top-1 / top-5 accuracy (percent) of a linear head on ``features``."""
    top1 = top5 = total = 0
    num_classes = weight.shape[0]
    for start in range(0, features.shape[0], batch_size):
        x = features[start : start + batch_size].to(device=device, dtype=torch.float32)
        y = labels[start : start + batch_size].to(device=device, dtype=torch.long)
        if mean is not None:
            x = (x - mean) / std
        logits = F.linear(x, weight, bias)
        pred = logits.topk(min(5, num_classes), dim=1).indices
        hits = pred.eq(y.unsqueeze(1))
        top1 += int(hits[:, 0].sum())
        top5 += int(hits.any(dim=1).sum())
        total += x.shape[0]
    return 100.0 * top1 / total, 100.0 * top5 / total


def _train_head(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    num_classes: int,
    lr: float,
    epochs: int,
    batch_size: int,
    weight_decay: float,
    momentum: float,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SGD + cosine schedule on a single linear layer; returns (weight, bias)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dim = train_features.shape[1]
    head = torch.nn.Linear(dim, num_classes).to(device)
    torch.nn.init.trunc_normal_(head.weight, std=0.01)
    torch.nn.init.zeros_(head.bias)

    optimizer = torch.optim.SGD(head.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    num_samples = train_features.shape[0]
    steps_per_epoch = max(1, math.ceil(num_samples / batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)

    for _ in range(epochs):
        perm = torch.randperm(num_samples, generator=generator)
        for start in range(0, num_samples, batch_size):
            idx = perm[start : start + batch_size]
            x = train_features[idx].to(device=device, dtype=torch.float32, non_blocking=True)
            y = train_labels[idx].to(device=device, dtype=torch.long, non_blocking=True)
            if mean is not None:
                x = (x - mean) / std
            loss = F.cross_entropy(head(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

    return head.weight.detach(), head.bias.detach()


def run_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    lr_grid: tuple[float, ...] | list[float] = (0.01, 0.03, 0.1, 0.3, 1.0),
    epochs: int = 50,
    batch_size: int = 4096,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    standardize: bool = True,
    val_frac: float = 0.02,
    device: torch.device | str = "cuda",
    seed: int = 0,
) -> dict:
    """Train a linear probe on frozen features; select the LR on held-out train.

    Args:
        train_features / train_labels: pretraining train split, ``(N, D)`` / ``(N,)``.
        test_features / test_labels: pretraining validation split (the reported set).
        num_classes: size of the label space.
        lr_grid: learning rates to search. Selected on the held-out slice.
        epochs, batch_size, weight_decay, momentum: SGD schedule.
        standardize: per-dimension mean/std normalisation fit on the train split.
        val_frac: fraction of the train split held out for LR selection.
        seed: controls the held-out split and the shuffling.

    Returns:
        ``{"top1", "top5", "best_lr", "search": [...], "epochs", ...}`` with
        accuracies in percent on the *test* (validation) split.
    """
    device = torch.device(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    num_samples = train_features.shape[0]

    mean = std = None
    if standardize:
        mean_cpu, std_cpu = _standardize_stats(train_features)
        mean, std = mean_cpu.to(device), std_cpu.to(device)

    # ── learning-rate search on a held-out slice of the train split ──────────
    search = []
    lr_grid = list(lr_grid)
    if len(lr_grid) == 1:
        best_lr = lr_grid[0]
    else:
        perm = torch.randperm(num_samples, generator=generator)
        n_val = max(1, int(round(val_frac * num_samples)))
        val_idx, fit_idx = perm[:n_val], perm[n_val:]
        fit_x, fit_y = train_features[fit_idx], train_labels[fit_idx]
        val_x, val_y = train_features[val_idx], train_labels[val_idx]

        best_lr, best_acc = lr_grid[0], -1.0
        for lr in lr_grid:
            weight, bias = _train_head(
                fit_x, fit_y, num_classes, lr, epochs, batch_size,
                weight_decay, momentum, mean, std, device, seed,
            )
            acc, _ = _evaluate(weight, bias, val_x, val_y, mean, std, device)
            search.append({"lr": lr, "heldout_top1": acc})
            logger.info("  linear probe | lr=%-6g held-out top-1 = %.2f%%", lr, acc)
            if acc > best_acc:
                best_lr, best_acc = lr, acc

    # ── refit on the full train split with the selected LR ───────────────────
    weight, bias = _train_head(
        train_features, train_labels, num_classes, best_lr, epochs, batch_size,
        weight_decay, momentum, mean, std, device, seed,
    )
    top1, top5 = _evaluate(weight, bias, test_features, test_labels, mean, std, device)
    train_top1, _ = _evaluate(weight, bias, train_features, train_labels, mean, std, device)

    return {
        "top1": top1,
        "top5": top5,
        "train_top1": train_top1,
        "best_lr": best_lr,
        "search": search,
        "epochs": epochs,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "standardize": standardize,
    }
