import logging
import sys

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from src.lejepa.models.lejepa import build_encoder

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def init_model(device, backbone_cfg, crop_size):
    """Build and move the LeJEPA encoder to *device*.

    ``crop_size`` is passed as ``img_size`` for ViT backbones so the model's
    positional embeddings match the training resolution.
    """
    cfg = dict(backbone_cfg)
    cfg.setdefault('img_size', crop_size)
    net = build_encoder(cfg)
    net.to(device)
    logger.info(net)
    return net


def init_probe(device, emb_dim, num_classes):
    probe = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, num_classes))
    probe.to(device)
    return probe


def init_opt(
    net,
    probe,
    lr,
    probe_lr,
    weight_decay,
    probe_weight_decay,
    warmup_steps,
    total_steps,
    use_bfloat16=False,
):
    param_groups = [{"params": net.parameters(), "lr": lr, "weight_decay": weight_decay}]
    if probe is not None:
        param_groups.append(
            {"params": probe.parameters(), "lr": probe_lr, "weight_decay": probe_weight_decay}
        )
    opt = torch.optim.AdamW(param_groups)

    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=max(total_steps - warmup_steps, 1), eta_min=probe_lr)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler = GradScaler() if use_bfloat16 else None
    return opt, scheduler, scaler


def load_checkpoint(device, r_path, net, probe, opt, scaler):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']
        msg = net.load_state_dict(checkpoint['net'])
        logger.info('loaded net from epoch %d with msg: %s', epoch, msg)
        if probe is not None and checkpoint.get('probe') is not None:
            msg = probe.load_state_dict(checkpoint['probe'])
            logger.info('loaded probe from epoch %d with msg: %s', epoch, msg)
        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None and checkpoint.get('scaler') is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info('loaded optimizers from epoch %d', epoch)
        del checkpoint
    except Exception as e:
        logger.info('Encountered exception when loading checkpoint: %s', e)
        epoch = 0
    return net, probe, opt, scaler, epoch
