# Copyright (c) 2025 Alessio Borgioli
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Supervised PhC baseline — same CNN+FC architecture, 3 000 labeled samples.

No pretraining.  Trains PhCEncoder end-to-end on DOS regression with MSE loss
and evaluates with ε_DOS = Σ|pred − dos| / Σ|dos| × 100 on the held-out split.

Usage (single GPU):
    python main.py --fname configs/phc_supervised_ep500.yaml
"""

import os
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset

from src.phc.models.phc_encoder import PhCEncoder, DOSPredictor
from src.phc.eval import eps_dos
from src.phc.transforms import make_phc_single_transform
from src.datasets.phc import PhCDataset
from src.utils.distributed import init_distributed
from src.utils.logging import CSVLogger, AverageMeter

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

checkpoint_freq = 50
_N_LABELED = 3_000


def main(args, resume_preempt=False):

    # -- optimization
    opt_cfg      = args['optimization']
    use_bfloat16 = opt_cfg['use_bfloat16']
    num_epochs   = opt_cfg['epochs']
    lr           = opt_cfg['lr']
    weight_decay = opt_cfg['weight_decay']

    # -- data
    data_cfg   = args['data']
    batch_size = data_cfg['batch_size']
    num_workers = data_cfg['num_workers']
    pin_mem    = data_cfg['pin_mem']
    n_labeled  = data_cfg.get('n_labeled', _N_LABELED)
    seed       = data_cfg.get('seed', 42)
    train_frac = data_cfg.get('train_frac', 0.6)

    # -- backbone
    backbone_cfg = args['backbone']
    emb_dim      = backbone_cfg.get('emb_dim', 1024)

    # -- logging
    log_cfg    = args['logging']
    folder     = log_cfg['folder']
    tag        = log_cfg['write_tag']
    use_wandb  = log_cfg.get('use_wandb', False)
    load_model = log_cfg.get('load_checkpoint', False) or resume_preempt
    r_file     = log_cfg.get('read_checkpoint')

    os.makedirs(folder, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info('Initialized (rank/world-size) %d/%d', rank, world_size)
    if rank > 0:
        logger.setLevel(logging.ERROR)

    log_file    = os.path.join(folder, f'{tag}_r{rank}.csv')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    save_path   = os.path.join(folder, f'{tag}-ep{{epoch}}.pth.tar')
    load_path   = os.path.join(folder, r_file) if r_file else latest_path

    import yaml
    with open(os.path.join(folder, 'params-phc_supervised.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(args, f)

    csv_logger = CSVLogger(
        log_file,
        ('%d',   'epoch'),
        ('%.5f', 'train_mse'),
        ('%.4f', 'test_eps_dos'),
    )

    if use_wandb and rank == 0:
        import wandb
        wandb.init(project='PhC-Supervised', config=args)

    # -- datasets
    single_transform = make_phc_single_transform(
        use_c4v=data_cfg.get('use_c4v', True),
        use_mirror=data_cfg.get('use_mirror', False),
        use_roll=data_cfg.get('use_roll', True),
        use_refractive=data_cfg.get('use_refractive', True),
        refractive_range=tuple(data_cfg.get('refractive_range', [0.8, 1.2])),
    )

    full_train = PhCDataset(
        path=data_cfg['path'], split='train',
        train_frac=train_frac, seed=seed,
        transform=single_transform,
    )
    test_ds = PhCDataset(
        path=data_cfg['path'], split='test',
        train_frac=train_frac, seed=seed,
        transform=single_transform,
    )

    # subsample to n_labeled examples from the training split
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(full_train), size=min(n_labeled, len(full_train)), replace=False)
    train_ds = Subset(full_train, idx.tolist())

    logger.info('Labeled samples: %d  Test samples: %d', len(train_ds), len(test_ds))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_mem, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_mem,
    )

    # -- model
    backbone = PhCEncoder(emb_dim=emb_dim).to(device)
    dos_head = DOSPredictor(emb_dim=emb_dim).to(device)
    params   = list(backbone.parameters()) + list(dos_head.parameters())
    opt      = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
    scaler    = torch.amp.GradScaler('cuda') if use_bfloat16 else None
    mse_loss  = nn.MSELoss()

    start_epoch = 0
    if load_model and os.path.isfile(load_path):
        ckpt = torch.load(load_path, map_location=device)
        backbone.load_state_dict(ckpt['backbone'])
        dos_head.load_state_dict(ckpt['dos_head'])
        opt.load_state_dict(ckpt['opt'])
        start_epoch = ckpt.get('epoch', 0)
        logger.info('Resumed from epoch %d', start_epoch)

    def save_checkpoint(epoch, test_eps):
        if rank != 0:
            return
        save_dict = {
            'backbone':   backbone.state_dict(),
            'dos_head':   dos_head.state_dict(),
            'opt':        opt.state_dict(),
            'scaler':     None if scaler is None else scaler.state_dict(),
            'epoch':      epoch,
            'test_eps':   test_eps,
        }
        torch.save(save_dict, latest_path)
        if epoch % checkpoint_freq == 0:
            torch.save(save_dict, save_path.format(epoch=epoch))

    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d', epoch + 1)

        backbone.train()
        dos_head.train()
        mse_meter = AverageMeter()

        for x, dos in train_loader:
            x   = x.to(device, non_blocking=True)    # (N, 1, 32, 32)
            dos = dos.to(device, non_blocking=True)   # (N, 400)

            with autocast('cuda', dtype=torch.bfloat16, enabled=use_bfloat16):
                emb, _   = backbone(x)
                pred_dos = dos_head(emb)
                loss     = mse_loss(pred_dos.float(), dos.float())

            opt.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            mse_meter.update(loss.item())

        scheduler.step()

        # -- evaluation
        backbone.eval()
        dos_head.eval()
        all_pred, all_dos = [], []
        with torch.inference_mode():
            for x, dos in test_loader:
                x   = x.to(device, non_blocking=True)
                emb, _ = backbone(x)
                all_pred.append(dos_head(emb).float().cpu())
                all_dos.append(dos.float().cpu())

        pred_cat = torch.cat(all_pred)
        dos_cat  = torch.cat(all_dos)
        test_eps = eps_dos(pred_cat, dos_cat)

        logger.info('Epoch %d | train_mse: %.5f  test_ε_DOS: %.4f%%',
                    epoch + 1, mse_meter.avg, test_eps)

        if use_wandb and rank == 0:
            import wandb
            wandb.log({'train/mse': mse_meter.avg, 'test/eps_dos': test_eps})

        csv_logger.log(epoch + 1, mse_meter.avg, test_eps)
        save_checkpoint(epoch + 1, test_eps)

    if use_wandb and rank == 0:
        import wandb
        wandb.finish()
