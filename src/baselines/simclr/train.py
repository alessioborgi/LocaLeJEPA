# Copyright (c) 2025 Alessio Borgioli
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""SimCLR baseline training — ResNet-50, ImageNet-100, 500 epochs.

Usage (single GPU):
    python main.py --fname configs/simclr_rn50_in100_ep500.yaml

Usage (multi-GPU via submitit):
    python main_distributed.py --fname configs/simclr_rn50_in100_ep500.yaml ...
"""

import os
import logging
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

from src.baselines.simclr.model import SimCLREncoder
from src.losses import nt_xent_loss
from src.transforms import make_lejepa_transforms, make_lejepa_eval_transform
from src.utils.distributed import init_distributed
from src.utils.logging import CSVLogger, AverageMeter

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

checkpoint_freq = 50


def _build_optimizer(net, lr, weight_decay, warmup_steps, total_steps, use_bfloat16):
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps - warmup_steps, eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler('cuda') if use_bfloat16 else None
    return opt, scheduler, scaler


def main(args, resume_preempt=False):

    # -- optimization
    opt_cfg       = args['optimization']
    use_bfloat16  = opt_cfg['use_bfloat16']
    num_epochs    = opt_cfg['epochs']
    warmup_epochs = opt_cfg['warmup']
    lr            = opt_cfg['lr']
    weight_decay  = opt_cfg['weight_decay']
    temperature   = opt_cfg.get('temperature', 0.07)

    # -- data
    data_cfg          = args['data']
    dataset           = data_cfg['dataset']
    batch_size        = data_cfg['batch_size']
    num_workers       = data_cfg['num_workers']
    pin_mem           = data_cfg['pin_mem']
    crop_size         = data_cfg.get('crop_size', 224)
    crop_scale        = data_cfg.get('crop_scale', [0.2, 1.0])
    root_path         = data_cfg.get('root_path')
    image_folder      = data_cfg.get('image_folder')
    copy_data         = data_cfg.get('copy_data', False)
    class_subset_file = data_cfg.get('class_subset_file')

    # -- simclr
    simclr_cfg = args.get('simclr', {})
    proj_dim   = simclr_cfg.get('proj_dim', 128)

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
    with open(os.path.join(folder, 'params-simclr.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(args, f)

    csv_logger = CSVLogger(
        log_file,
        ('%d',   'epoch'),
        ('%.5f', 'nt_xent_loss'),
    )

    if use_wandb and rank == 0:
        import wandb
        wandb.init(project='SimCLR', config=args)

    # SimCLR uses exactly 2 views
    train_transform = make_lejepa_transforms(
        crop_size=crop_size, crop_scale=crop_scale,
        num_views=2,
        gaussian_blur=data_cfg.get('use_gaussian_blur', True),
        horizontal_flip=data_cfg.get('use_horizontal_flip', True),
        color_distortion=data_cfg.get('use_color_distortion', True),
        color_jitter=data_cfg.get('color_jitter_strength', 0.5),
    )
    val_transform = make_lejepa_eval_transform(crop_size=crop_size)

    if dataset in ('imagenet1k', 'imagenet100'):
        from src.datasets.imagenet1k import make_imagenet1k_lejepa
        train_loader, val_loader, train_sampler = make_imagenet1k_lejepa(
            train_transform=train_transform, val_transform=val_transform,
            batch_size=batch_size, pin_mem=pin_mem, num_workers=num_workers,
            world_size=world_size, rank=rank,
            root_path=root_path, image_folder=image_folder, copy_data=copy_data,
            class_subset_file=class_subset_file,
        )
    elif dataset == 'imagenette':
        from src.datasets.imagenette import make_imagenette
        train_loader, val_loader, train_sampler = make_imagenette(
            train_transform=train_transform, val_transform=val_transform,
            batch_size=batch_size, pin_mem=pin_mem, num_workers=num_workers,
            world_size=world_size, rank=rank,
        )
    else:
        raise ValueError(f'Unknown dataset {dataset!r} for SimCLR. '
                         'Choose imagenet1k | imagenet100 | imagenette.')

    ipe          = len(train_loader)
    warmup_steps = ipe * warmup_epochs
    total_steps  = ipe * num_epochs

    net = SimCLREncoder(proj_dim=proj_dim).to(device)
    opt, scheduler, scaler = _build_optimizer(
        net, lr=lr, weight_decay=weight_decay,
        warmup_steps=warmup_steps, total_steps=total_steps,
        use_bfloat16=use_bfloat16,
    )

    start_epoch = 0
    if load_model and os.path.isfile(load_path):
        ckpt = torch.load(load_path, map_location=device)
        net.load_state_dict(ckpt['net'])
        opt.load_state_dict(ckpt['opt'])
        start_epoch = ckpt.get('epoch', 0)
        if scaler is not None and ckpt.get('scaler') is not None:
            scaler.load_state_dict(ckpt['scaler'])
        logger.info('Resumed from epoch %d', start_epoch)

    def save_checkpoint(epoch):
        if rank != 0:
            return
        save_dict = {
            'net':        net.state_dict(),
            'opt':        opt.state_dict(),
            'scaler':     None if scaler is None else scaler.state_dict(),
            'epoch':      epoch,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr':         lr,
        }
        torch.save(save_dict, latest_path)
        if epoch % checkpoint_freq == 0:
            torch.save(save_dict, save_path.format(epoch=epoch))

    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d', epoch + 1)

        if hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

        net.train()
        loss_meter = AverageMeter()

        for itr, (vs, _) in enumerate(train_loader):
            vs = vs.to(device, non_blocking=True)   # (N, 2, C, H, W)

            with autocast('cuda', dtype=torch.bfloat16, enabled=use_bfloat16):
                _, proj = net(vs)                   # proj: (2, N, proj_dim)
                loss    = nt_xent_loss(proj, temperature=temperature)

            opt.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            scheduler.step()

            loss_meter.update(loss.item())

            if itr % 10 == 0:
                logger.info('[%d, %5d] nt_xent: %.4f', epoch + 1, itr, loss_meter.avg)

            if use_wandb and rank == 0:
                import wandb
                wandb.log({'train/nt_xent': loss.item()})

        assert not np.isnan(loss_meter.avg), 'SimCLR loss is NaN'

        csv_logger.log(epoch + 1, loss_meter.avg)
        save_checkpoint(epoch + 1)

    if use_wandb and rank == 0:
        import wandb
        wandb.finish()
