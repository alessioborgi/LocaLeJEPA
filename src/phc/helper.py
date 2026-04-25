import logging

import torch

from src.phc.models.phc_encoder import PhCEncoder, DOSPredictor

logger = logging.getLogger()


def init_model(device: torch.device, backbone_cfg: dict) -> PhCEncoder:
    emb_dim  = backbone_cfg.get('emb_dim',  1024)
    proj_dim = backbone_cfg.get('proj_dim', 128)
    net = PhCEncoder(emb_dim=emb_dim, proj_dim=proj_dim).to(device)
    logger.info('PhCEncoder — emb_dim=%d  proj_dim=%d', emb_dim, proj_dim)
    return net


def init_dos_head(device: torch.device, backbone_cfg: dict) -> DOSPredictor:
    emb_dim = backbone_cfg.get('emb_dim', 1024)
    return DOSPredictor(emb_dim=emb_dim).to(device)


def init_opt(
    net:          torch.nn.Module,
    lr:           float,
    weight_decay: float,
    momentum:     float,
    total_steps:  int,
    use_bfloat16: bool,
):
    optimizer = torch.optim.SGD(
        net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps,
    )
    scaler = torch.amp.GradScaler('cuda') if use_bfloat16 else None
    return optimizer, scheduler, scaler


def load_checkpoint(
    device:  torch.device,
    r_path:  str,
    net:     torch.nn.Module,
    opt:     torch.optim.Optimizer,
    scaler,
):
    try:
        checkpoint = torch.load(r_path, map_location=device)
    except FileNotFoundError:
        logger.info('No checkpoint at %s — starting from scratch', r_path)
        return net, opt, scaler, 0

    net.load_state_dict(checkpoint['net'])
    opt.load_state_dict(checkpoint['opt'])
    if scaler is not None and checkpoint.get('scaler') is not None:
        scaler.load_state_dict(checkpoint['scaler'])
    epoch = checkpoint.get('epoch', 0)
    logger.info('Loaded checkpoint from %s (epoch %d)', r_path, epoch)
    return net, opt, scaler, epoch
