# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys

import torch

import src.ijepa.models.vision_transformer as vit
from src.utils.schedulers import WarmupCosineSchedule, CosineWDSchedule
from src.utils.tensors import trunc_normal_

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_checkpoint(device, r_path, encoder, predictor, target_encoder, opt, scaler):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        msg = encoder.load_state_dict(checkpoint['encoder'])
        logger.info('loaded pretrained encoder from epoch %d with msg: %s', epoch, msg)

        msg = predictor.load_state_dict(checkpoint['predictor'])
        logger.info('loaded pretrained predictor from epoch %d with msg: %s', epoch, msg)

        if target_encoder is not None:
            msg = target_encoder.load_state_dict(checkpoint['target_encoder'])
            logger.info('loaded target_encoder from epoch %d with msg: %s', epoch, msg)

        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info('loaded optimizers from epoch %d', epoch)
        del checkpoint

    except Exception as e:
        logger.info('Encountered exception when loading checkpoint: %s', e)
        epoch = 0

    return encoder, predictor, target_encoder, opt, scaler, epoch


def init_model(device, backbone_cfg, crop_size):
    """Initialise I-JEPA encoder + predictor from the ``backbone:`` config section.

    I-JEPA requires a ViT backbone.  Raises ``ValueError`` if a ResNet is
    requested.
    """
    name = backbone_cfg['name']
    if name != 'vit':
        raise ValueError(
            f'I-JEPA requires a ViT backbone (name: vit), got {name!r}. '
            'Use method: lejepa for ResNet backbones.'
        )

    vit_size   = backbone_cfg['vit_size']    # e.g. vit_small, vit_huge
    patch_size = backbone_cfg['patch_size']
    pred_depth = backbone_cfg['pred_depth']
    pred_emb_dim = backbone_cfg['pred_emb_dim']

    in_chans = backbone_cfg.get('in_chans', 3)
    encoder = vit.__dict__[vit_size](img_size=[crop_size], patch_size=patch_size, in_chans=in_chans)
    predictor = vit.__dict__['vit_predictor'](
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_emb_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads,
    )

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    for m in encoder.modules():
        init_weights(m)
    for m in predictor.modules():
        init_weights(m)

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    return encoder, predictor


def init_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25,
):
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0,
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0,
        }
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler
