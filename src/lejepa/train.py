import os
import logging
import sys
import yaml

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from src.lejepa.models.lejepa import SIGReg
from src.lejepa.helper import init_model, init_probe, init_opt, load_checkpoint
from src.transforms import make_lejepa_transforms, make_lejepa_eval_transform
from src.utils.distributed import init_distributed
from src.utils.logging import CSVLogger, AverageMeter

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

checkpoint_freq = 50


def main(args, resume_preempt=False):

    # -- backbone
    backbone_cfg  = args['backbone']
    emb_dim       = backbone_cfg['emb_dim']
    proj_dim      = backbone_cfg['proj_dim']

    # -- optimization
    opt_cfg            = args['optimization']
    use_bfloat16       = opt_cfg['use_bfloat16']
    num_epochs         = opt_cfg['epochs']
    warmup_epochs      = opt_cfg['warmup']
    lr                 = opt_cfg['lr']
    weight_decay       = opt_cfg['weight_decay']
    probe_lr           = opt_cfg['probe_lr']
    probe_weight_decay = opt_cfg['probe_weight_decay']

    # -- data
    data_cfg             = args['data']
    dataset              = data_cfg['dataset']
    batch_size           = data_cfg['batch_size']
    num_views            = data_cfg['num_views']
    num_workers          = data_cfg['num_workers']
    pin_mem              = data_cfg['pin_mem']
    crop_size            = data_cfg['crop_size']
    crop_scale           = data_cfg['crop_scale']
    use_gaussian_blur    = data_cfg['use_gaussian_blur']
    use_horizontal_flip  = data_cfg['use_horizontal_flip']
    use_color_distortion = data_cfg['use_color_distortion']
    color_jitter         = data_cfg['color_jitter_strength']
    root_path            = data_cfg.get('root_path')
    image_folder         = data_cfg.get('image_folder')
    copy_data            = data_cfg.get('copy_data', False)

    # -- lejepa
    lejepa_cfg  = args['lejepa']
    lamb        = lejepa_cfg['lamb']
    num_classes = lejepa_cfg['num_classes']

    # -- logging / checkpointing
    log_cfg         = args['logging']
    folder          = log_cfg['folder']
    tag             = log_cfg['write_tag']
    use_wandb       = log_cfg.get('use_wandb', False)
    load_model      = log_cfg.get('load_checkpoint', False) or resume_preempt
    r_file          = log_cfg.get('read_checkpoint')

    os.makedirs(folder, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info('Initialized (rank/world-size) %d/%d', rank, world_size)
    if rank > 0:
        logger.setLevel(logging.ERROR)

    log_file    = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path   = os.path.join(folder, f'{tag}-ep{{epoch}}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path   = os.path.join(folder, r_file) if r_file else latest_path

    with open(os.path.join(folder, 'params-lejepa.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(args, f)

    csv_logger = CSVLogger(
        log_file,
        ('%d',   'epoch'),
        ('%.5f', 'lejepa_loss'),
        ('%.5f', 'sigreg_loss'),
        ('%.5f', 'inv_loss'),
        ('%.5f', 'probe_loss'),
        ('%.5f', 'test_acc'),
    )

    if use_wandb and rank == 0:
        import wandb
        wandb.init(project='LeJEPA', config=args)

    # -- transforms
    train_transform = make_lejepa_transforms(
        crop_size=crop_size, crop_scale=crop_scale, num_views=num_views,
        gaussian_blur=use_gaussian_blur, horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion, color_jitter=color_jitter,
    )
    val_transform = make_lejepa_eval_transform(crop_size=crop_size)

    # -- dataset
    if dataset == 'imagenet1k':
        from src.datasets.imagenet1k import make_imagenet1k_lejepa
        train_loader, val_loader, train_sampler = make_imagenet1k_lejepa(
            train_transform=train_transform, val_transform=val_transform,
            batch_size=batch_size, pin_mem=pin_mem, num_workers=num_workers,
            world_size=world_size, rank=rank,
            root_path=root_path, image_folder=image_folder, copy_data=copy_data,
        )
    elif dataset == 'imagenette':
        from src.datasets.imagenette import make_imagenette
        train_loader, val_loader, train_sampler = make_imagenette(
            train_transform=train_transform, val_transform=val_transform,
            batch_size=batch_size, pin_mem=pin_mem, num_workers=num_workers,
            world_size=world_size, rank=rank,
        )
    else:
        raise ValueError(f'Unknown dataset: {dataset!r}. Choose "imagenet1k" or "imagenette".')

    ipe          = len(train_loader)
    warmup_steps = ipe * warmup_epochs
    total_steps  = ipe * num_epochs

    # -- model, probe, regularizer
    net    = init_model(device=device, backbone_cfg=backbone_cfg, crop_size=crop_size)
    probe  = init_probe(device=device, emb_dim=emb_dim, num_classes=num_classes)
    sigreg = SIGReg().to(device)

    opt, scheduler, scaler = init_opt(
        net=net, probe=probe,
        lr=lr, probe_lr=probe_lr,
        weight_decay=weight_decay, probe_weight_decay=probe_weight_decay,
        warmup_steps=warmup_steps, total_steps=total_steps,
        use_bfloat16=use_bfloat16,
    )

    start_epoch = 0
    if load_model:
        net, probe, opt, scaler, start_epoch = load_checkpoint(
            device=device, r_path=load_path,
            net=net, probe=probe, opt=opt, scaler=scaler,
        )

    def save_checkpoint(epoch, test_acc):
        if rank != 0:
            return
        save_dict = {
            'net':        net.state_dict(),
            'probe':      probe.state_dict(),
            'opt':        opt.state_dict(),
            'scaler':     None if scaler is None else scaler.state_dict(),
            'epoch':      epoch,
            'test_acc':   test_acc,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr':         lr,
        }
        torch.save(save_dict, latest_path)
        if epoch % checkpoint_freq == 0:
            torch.save(save_dict, save_path.format(epoch=epoch))

    # -- training loop
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d', epoch + 1)

        if hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

        net.train()
        probe.train()
        lejepa_meter = AverageMeter()
        sigreg_meter = AverageMeter()
        inv_meter    = AverageMeter()
        probe_meter  = AverageMeter()

        for itr, (vs, y) in enumerate(train_loader):
            with autocast('cuda', dtype=torch.bfloat16, enabled=use_bfloat16):
                vs = vs.to(device, non_blocking=True)
                y  = y.to(device, non_blocking=True)

                emb, proj   = net(vs)
                inv_loss    = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)
                lejepa_loss = sigreg_loss * lamb + inv_loss * (1 - lamb)
                y_rep       = y.repeat_interleave(num_views)
                probe_loss  = F.cross_entropy(probe(emb.detach()), y_rep)
                loss        = lejepa_loss + probe_loss

            opt.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            scheduler.step()

            lejepa_meter.update(lejepa_loss.item())
            sigreg_meter.update(sigreg_loss.item())
            inv_meter.update(inv_loss.item())
            probe_meter.update(probe_loss.item())

            if itr % 10 == 0:
                logger.info(
                    '[%d, %5d] lejepa: %.4f  sigreg: %.4f  inv: %.4f  probe: %.4f',
                    epoch + 1, itr,
                    lejepa_meter.avg, sigreg_meter.avg,
                    inv_meter.avg, probe_meter.avg,
                )

            if use_wandb and rank == 0:
                import wandb
                wandb.log({
                    'train/lejepa': lejepa_loss.item(),
                    'train/sigreg': sigreg_loss.item(),
                    'train/inv':    inv_loss.item(),
                    'train/probe':  probe_loss.item(),
                })

        assert not np.isnan(lejepa_meter.avg), 'LeJEPA loss is NaN'

        # -- evaluation
        net.eval()
        probe.eval()
        correct, total = 0, 0
        with torch.inference_mode():
            for vs, y in val_loader:
                vs = vs.to(device, non_blocking=True)
                y  = y.to(device, non_blocking=True)
                with autocast('cuda', dtype=torch.bfloat16, enabled=use_bfloat16):
                    correct += (probe(net(vs)[0]).argmax(1) == y).sum().item()
                    total   += y.size(0)
        test_acc = correct / total if total > 0 else 0.0
        logger.info('Epoch %d | test acc: %.4f', epoch + 1, test_acc)

        if use_wandb and rank == 0:
            import wandb
            wandb.log({'test/acc': test_acc, 'test/epoch': epoch})

        csv_logger.log(
            epoch + 1,
            lejepa_meter.avg, sigreg_meter.avg,
            inv_meter.avg, probe_meter.avg,
            test_acc,
        )
        save_checkpoint(epoch + 1, test_acc)

    if use_wandb and rank == 0:
        import wandb
        wandb.finish()
