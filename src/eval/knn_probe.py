# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Weighted k-NN classifier on frozen features (DINO / LeJEPA protocol).

This is the ``k-NN (%)`` column of SPHERE-JEPA Table 3: the frozen backbone
embeds the whole pretraining train split into a feature bank, and every
validation image is classified by a cosine-similarity vote of its ``k`` nearest
neighbours, each weighted by ``exp(sim / T)`` (Wu et al. 2018, as used by DINO
and by LeJEPA/SPHERE-JEPA for their k-NN numbers).

Features are L2-normalised first, so the vote is over cosine similarity — the
natural metric for the hyperspherical embeddings SUSReg targets.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def weighted_knn(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    k_values: tuple[int, ...] | list[int] = (10, 20, 100, 200),
    temperature: float = 0.07,
    device: torch.device | str = "cuda",
    chunk_size: int = 128,
    bank_dtype: torch.dtype = torch.float16,
) -> dict[int, dict[str, float]]:
    """Classify ``test_features`` by a similarity-weighted vote over the bank.

    Args:
        train_features: ``(N, D)`` feature bank (any dtype; normalised here).
        train_labels: ``(N,)`` int64 class indices.
        test_features: ``(M, D)`` query features.
        test_labels: ``(M,)`` int64 ground-truth class indices.
        num_classes: Number of classes in the label space.
        k_values: Neighbourhood sizes to report. All are evaluated in one pass.
        temperature: Softmax temperature on the cosine similarities.
        device: Device the similarity matmul runs on.
        chunk_size: Number of queries per matmul chunk (memory knob).
        bank_dtype: Storage dtype for the bank on device (fp16 halves memory).

    Returns:
        ``{k: {"top1": float, "top5": float}}`` with accuracies in percent.
    """
    device = torch.device(device)
    k_values = sorted({int(k) for k in k_values})
    max_k = max(k_values)
    num_train = train_features.shape[0]
    if max_k > num_train:
        raise ValueError(f"k={max_k} exceeds the {num_train}-sample feature bank.")

    bank = F.normalize(train_features.float(), dim=1).to(device=device, dtype=bank_dtype).t().contiguous()
    bank_labels = train_labels.to(device=device, dtype=torch.long)

    # Accumulate correct counts per k so we never hold an (M, num_classes) score
    # matrix for the whole validation split at once.
    top1 = {k: 0 for k in k_values}
    top5 = {k: 0 for k in k_values}
    total = 0

    for start in range(0, test_features.shape[0], chunk_size):
        query = test_features[start : start + chunk_size].float()
        target = test_labels[start : start + chunk_size].to(device=device, dtype=torch.long)
        query = F.normalize(query, dim=1).to(device=device, dtype=bank_dtype)

        similarity = (query @ bank).float()                      # (B, N)
        sim_k, idx_k = similarity.topk(max_k, dim=1, largest=True)
        labels_k = bank_labels[idx_k]                            # (B, max_k)
        weights_k = (sim_k / temperature).exp()

        batch = query.shape[0]
        for k in k_values:
            scores = torch.zeros(batch, num_classes, device=device)
            scores.scatter_add_(1, labels_k[:, :k], weights_k[:, :k])
            pred = scores.topk(min(5, num_classes), dim=1).indices  # (B, <=5)
            hits = pred.eq(target.unsqueeze(1))
            top1[k] += int(hits[:, 0].sum())
            top5[k] += int(hits.any(dim=1).sum())
        total += batch

    return {
        k: {"top1": 100.0 * top1[k] / total, "top5": 100.0 * top5[k] / total}
        for k in k_values
    }
