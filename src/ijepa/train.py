# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

try:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.ijepa.masks.multiblock import MaskCollator as MBMaskCollator
from src.ijepa.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch

from src.ijepa.helper import (
    load_checkpoint,
    init_model,
    init_opt)
from src.transforms import make_transforms

log_timings = True
log_freq = 10
checkpoint_freq = 50

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False):

    # -- backbone
    backbone_cfg = args['backbone']
    patch_size   = backbone_cfg['patch_size']

    # -- optimization
    opt_cfg      = args['optimization']
    use_bfloat16 = opt_cfg['use_bfloat16']
    ema          = opt_cfg['ema']
    ipe_scale    = opt_cfg['ipe_scale']
    wd           = float(opt_cfg['weight_decay'])
    final_wd     = float(opt_cfg['final_weight_decay'])
    num_epochs   = opt_cfg['epochs']
    warmup       = opt_cfg['warmup']
    start_lr     = opt_cfg['start_lr']
    lr           = opt_cfg['lr']
    final_lr     = opt_cfg['final_lr']

    # -- data
    data_cfg             = args['data']
    dataset              = data_cfg.get('dataset', 'imagenet1k')
    batch_size           = data_cfg['batch_size']
    pin_mem              = data_cfg['pin_mem']
    num_workers          = data_cfg['num_workers']
    root_path            = data_cfg.get('root_path')
    image_folder         = data_cfg.get('image_folder')
    crop_size            = data_cfg['crop_size']
    crop_scale           = data_cfg.get('crop_scale', [0.2, 1.0])
    copy_data            = data_cfg.get('copy_data', False)
    use_gaussian_blur    = data_cfg.get('use_gaussian_blur', True)
    use_horizontal_flip  = data_cfg.get('use_horizontal_flip', True)
    use_color_distortion = data_cfg.get('use_color_distortion', True)
    color_jitter         = data_cfg.get('color_jitter_strength', 0.8)

    # -- ijepa masking
    ijepa_cfg       = args['ijepa']
    allow_overlap   = ijepa_cfg['allow_overlap']
    num_enc_masks   = ijepa_cfg['num_enc_masks']
    min_keep        = ijepa_cfg['min_keep']
    enc_mask_scale  = ijepa_cfg['enc_mask_scale']
    num_pred_masks  = ijepa_cfg['num_pred_masks']
    pred_mask_scale = ijepa_cfg['pred_mask_scale']
    aspect_ratio    = ijepa_cfg['aspect_ratio']

    # -- logging / checkpointing
    log_cfg    = args['logging']
    folder     = log_cfg['folder']
    tag        = log_cfg['write_tag']
    load_model = log_cfg.get('load_checkpoint', False) or resume_preempt
    r_file     = log_cfg.get('read_checkpoint')

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    os.makedirs(folder, exist_ok=True)

    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    world_size, rank = init_distributed()
    logger.info('Initialized (rank/world-size) %d/%d', rank, world_size)
    if rank > 0:
        logger.setLevel(logging.ERROR)

    log_file    = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path   = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path   = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    csv_logger = CSVLogger(
        log_file,
        ('%d',   'epoch'),
        ('%d',   'itr'),
        ('%.5f', 'loss'),
        ('%.5f', 'mask-A'),
        ('%.5f', 'mask-B'),
        ('%d',   'time (ms)'),
    )

    # -- init model
    encoder, predictor = init_model(
        device=device,
        backbone_cfg=backbone_cfg,
        crop_size=crop_size,
    )
    target_encoder = copy.deepcopy(encoder)

    # -- mask collator + data transforms
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep,
    )

    if dataset in ('phc_blob', 'phc_gpm'):
        from src.phc.transforms import make_phc_single_transform
        transform = make_phc_single_transform(
            use_c4v=data_cfg.get('use_c4v', True),
            use_mirror=data_cfg.get('use_mirror', False),
            use_roll=data_cfg.get('use_roll', True),
            use_refractive=data_cfg.get('use_refractive', True),
            refractive_range=tuple(data_cfg.get('refractive_range', [0.8, 1.2])),
        )
    else:
        transform = make_transforms(
            crop_size=crop_size,
            crop_scale=crop_scale,
            gaussian_blur=use_gaussian_blur,
            horizontal_flip=use_horizontal_flip,
            color_distortion=use_color_distortion,
            color_jitter=color_jitter,
        )

    # -- dataset / data-loader
    if dataset == 'imagenet1k':
        from src.datasets.imagenet1k import make_imagenet1k
        _, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            drop_last=True,
        )
    elif dataset == 'imagenette':
        from src.datasets.imagenette import make_imagenette_ijepa
        _, unsupervised_loader, unsupervised_sampler = make_imagenette_ijepa(
            transform=transform,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
        )
    elif dataset in ('phc_blob', 'phc_gpm'):
        from src.datasets.phc import make_phc_ijepa_loaders
        _, unsupervised_loader, unsupervised_sampler = make_phc_ijepa_loaders(
            path=data_cfg['path'],
            train_transform=transform,
            collator=mask_collator,
            batch_size=batch_size,
            pin_mem=pin_mem,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            train_frac=data_cfg.get('train_frac', 0.6),
            seed=data_cfg.get('seed', 42),
        )
    else:
        raise ValueError(f'Unknown dataset: {dataset!r}. '
                         'Choose "imagenet1k", "imagenette", "phc_blob", or "phc_gpm".')

    ipe = len(unsupervised_loader)

    # -- optimizer + schedulers
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16,
    )
    encoder        = DistributedDataParallel(encoder, static_graph=True)
    predictor      = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch):
        save_dict = {
            'encoder':        encoder.state_dict(),
            'predictor':      predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt':            optimizer.state_dict(),
            'scaler':         None if scaler is None else scaler.state_dict(),
            'epoch':          epoch,
            'loss':           loss_meter.avg,
            'batch_size':     batch_size,
            'world_size':     world_size,
            'lr':             lr,
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d', epoch + 1)

        unsupervised_sampler.set_epoch(epoch)

        loss_meter  = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter  = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):

            def load_imgs():
                imgs    = udata[0].to(device, non_blocking=True)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, masks_1, masks_2)
            imgs, masks_enc, masks_pred = load_imgs()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(imgs)
                        h = F.layer_norm(h, (h.size(-1),))
                        B = len(h)
                        h = apply_masks(h, masks_pred)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                        return h

                def forward_context():
                    z = encoder(imgs, masks_enc)
                    z = predictor(z, masks_enc, masks_pred)
                    return z

                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    loss = AllReduce.apply(loss)
                    return loss

                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = forward_target()
                    z = forward_context()
                    loss = loss_fn(z, h)

                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (float(loss), _new_lr, _new_wd, grad_stats)

            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                'masks: %.1f %.1f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   maskA_meter.avg,
                                   maskB_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()
            assert not np.isnan(loss), 'loss is nan'

        logger.info('avg. loss %.3f', loss_meter.avg)
        save_checkpoint(epoch + 1)


if __name__ == "__main__":
    main()
