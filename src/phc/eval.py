"""DOS regression evaluation for PhC backbones.

Two protocols:
    frozen    — backbone weights frozen; only DOSPredictor trained
    fine_tune — backbone + DOSPredictor trained jointly

Metric: ε_DOS = Σ|pred − dos| / Σ|dos| × 100  (lower is better)

Results are averaged over `num_splits` random train/test splits.
"""

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.phc.models.phc_encoder import DOSPredictor
from src.datasets.phc import PhCDataset

logger = logging.getLogger()


def eps_dos(pred: torch.Tensor, target: torch.Tensor) -> float:
    """ε_DOS = Σ|pred − dos| / Σ|dos| × 100."""
    return (pred - target).abs().sum().item() / target.abs().sum().item() * 100.0


def _fit_dos_head(
    backbone:     nn.Module,
    train_loader: DataLoader,
    device:       torch.device,
    emb_dim:      int   = 1024,
    lr:           float = 1e-3,
    epochs:       int   = 50,
    fine_tune:    bool  = False,
) -> DOSPredictor:
    """Train a DOSPredictor head on a backbone (frozen or fine-tuned)."""
    head   = DOSPredictor(emb_dim=emb_dim).to(device)
    params = list(head.parameters())
    if fine_tune:
        params += list(backbone.parameters())

    opt       = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for _ in range(epochs):
        head.train()
        backbone.train() if fine_tune else backbone.eval()

        for x, dos in train_loader:
            x   = x.to(device, non_blocking=True)
            dos = dos.to(device, non_blocking=True)

            if fine_tune:
                emb, _ = backbone(x)
            else:
                with torch.inference_mode():
                    emb, _ = backbone(x)
                emb = emb.detach()

            loss = F.mse_loss(head(emb), dos)
            opt.zero_grad()
            loss.backward()
            opt.step()

        scheduler.step()

    return head


@torch.inference_mode()
def _eval_split(
    backbone: nn.Module,
    head:     nn.Module,
    loader:   DataLoader,
    device:   torch.device,
) -> float:
    backbone.eval()
    head.eval()
    preds, targets = [], []
    for x, dos in loader:
        x   = x.to(device, non_blocking=True)
        dos = dos.to(device, non_blocking=True)
        emb, _ = backbone(x)
        preds.append(head(emb).cpu())
        targets.append(dos.cpu())
    return eps_dos(torch.cat(preds), torch.cat(targets))


def evaluate(
    backbone:     nn.Module,
    data_path:    str,
    device:       torch.device,
    emb_dim:      int   = 1024,
    batch_size:   int   = 512,
    num_workers:  int   = 4,
    num_splits:   int   = 3,
    train_frac:   float = 0.6,
    fine_tune:    bool  = False,
    head_lr:      float = 1e-3,
    head_epochs:  int   = 50,
) -> float:
    """Evaluate backbone via DOS regression; return mean ε_DOS across splits."""
    seeds  = [42, 123, 999][:num_splits]
    errors = []

    for seed in seeds:
        train_ds = PhCDataset(data_path, split='train', train_frac=train_frac, seed=seed)
        test_ds  = PhCDataset(data_path, split='test',  train_frac=train_frac, seed=seed)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True)

        # Deep-copy so fine-tuning one split doesn't pollute the next
        bb_copy = copy.deepcopy(backbone)
        head    = _fit_dos_head(
            backbone=bb_copy,
            train_loader=train_loader,
            device=device,
            emb_dim=emb_dim,
            lr=head_lr,
            epochs=head_epochs,
            fine_tune=fine_tune,
        )
        err = _eval_split(bb_copy, head, test_loader, device)
        errors.append(err)
        logger.info('seed=%d | ε_DOS = %.2f%%', seed, err)

    mean_err = float(np.mean(errors))
    logger.info(
        'ε_DOS = %.2f ± %.2f%%  (%s)',
        mean_err, float(np.std(errors)),
        'fine-tuned' if fine_tune else 'frozen',
    )
    return mean_err
