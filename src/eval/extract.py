"""Backbone feature extraction for linear evaluation.

Supported backbone types
------------------------
I-JEPA ViT (from target_encoder in checkpoint)
    → CLS tokens from the last two transformer blocks, concatenated → LayerNorm
    → feature dim: 2 × embed_dim

LeJEPA ViT (timm backbone inside ViTEncoder)
    → CLS tokens from the last two transformer blocks, concatenated → LayerNorm
    → feature dim: 2 × embed_dim

LeJEPA ResNet (torchvision backbone inside ResNetEncoder)
    → global-average-pool output after the modified fc head → LayerNorm
    → feature dim: emb_dim

LeJEPA PhCEncoder
    → single-view embedding from net.forward(x) → LayerNorm
    → feature dim: emb_dim
"""

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

logger = logging.getLogger()


# ---------------------------------------------------------------------------
# ViT hook: capture CLS token from the last two transformer blocks
# ---------------------------------------------------------------------------

class _CLS2Hook:
    """Register forward hooks on the last two ViT blocks to collect CLS tokens."""

    def __init__(self, blocks):
        self._feats = []
        for blk in list(blocks)[-2:]:
            blk.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        # out: (N, num_patches+1, D) — CLS at position 0
        self._feats.append(out[:, 0])

    def reset(self):
        self._feats.clear()

    def get(self) -> torch.Tensor:
        feat = torch.cat(self._feats, dim=-1)        # (N, 2*D)
        return F.layer_norm(feat, feat.shape[-1:])


# ---------------------------------------------------------------------------
# Backbone loader
# ---------------------------------------------------------------------------

def load_backbone(
    pretrain_config_path: str,
    checkpoint_path:      str,
    device:               torch.device,
    crop_size:            int = 224,
):
    """Load a frozen backbone from a BundLeJEPA checkpoint.

    Reads the training config (params-lejepa.yaml / params-ijepa.yaml) to
    reconstruct the model architecture, then loads the pretrained weights.

    Parameters
    ----------
    pretrain_config_path : path to params-{lejepa,ijepa}.yaml saved during training
    checkpoint_path      : path to the .pth.tar checkpoint
    device               : target device
    crop_size            : spatial resolution expected by the backbone (default 224)

    Returns
    -------
    extract_fn : callable  (x: Tensor[N, C, H, W]) → Tensor[N, feat_dim]
    feat_dim   : int — output feature dimension
    """
    with open(pretrain_config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    method        = cfg.get('method', 'ijepa')
    backbone_cfg  = cfg['backbone']
    backbone_name = backbone_cfg['name']

    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # ------------------------------------------------------------------ I-JEPA
    if method == 'ijepa':
        from src.ijepa.helper import init_model
        encoder, _ = init_model(device='cpu', backbone_cfg=backbone_cfg, crop_size=crop_size)
        state = ckpt.get('target_encoder', ckpt.get('encoder', {}))
        state = {k.replace('module.', ''): v for k, v in state.items()}
        encoder.load_state_dict(state, strict=True)
        encoder.to(device).eval()
        for p in encoder.parameters():
            p.requires_grad_(False)

        hook     = _CLS2Hook(encoder.blocks)
        feat_dim = encoder.embed_dim * 2

        @torch.inference_mode()
        def extract(x: torch.Tensor) -> torch.Tensor:
            hook.reset()
            encoder(x.to(device))
            return hook.get()

        return extract, feat_dim

    # ----------------------------------------------------------------- LeJEPA
    from src.lejepa.helper import init_model
    net = init_model(device='cpu', backbone_cfg=backbone_cfg, crop_size=crop_size)
    state = {k.replace('module.', ''): v for k, v in ckpt['net'].items()}
    net.load_state_dict(state, strict=True)
    net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)

    if backbone_name == 'vit':
        # timm ViT inside ViTEncoder — hook its .blocks
        vit      = net.backbone
        hook     = _CLS2Hook(vit.blocks)
        feat_dim = vit.embed_dim * 2

        @torch.inference_mode()
        def extract(x: torch.Tensor) -> torch.Tensor:
            hook.reset()
            vit(x.to(device))        # hooks collect CLS; return value ignored
            return hook.get()

        return extract, feat_dim

    if backbone_name in ('resnet18', 'resnet50'):
        emb_dim = backbone_cfg['emb_dim']

        @torch.inference_mode()
        def extract(x: torch.Tensor) -> torch.Tensor:
            emb = net.backbone(x.to(device))      # (N, emb_dim) — after modified fc
            return F.layer_norm(emb, emb.shape[-1:])

        return extract, emb_dim

    if backbone_name == 'phc':
        emb_dim = backbone_cfg['emb_dim']

        @torch.inference_mode()
        def extract(x: torch.Tensor) -> torch.Tensor:
            emb, _ = net(x.to(device))             # single-view path (N,1,32,32)
            return F.layer_norm(emb, emb.shape[-1:])

        return extract, emb_dim

    raise ValueError(f'Unknown backbone {backbone_name!r}')


# ---------------------------------------------------------------------------
# Feature extraction loop (with optional disk cache)
# ---------------------------------------------------------------------------

def extract_features(
    extract_fn,
    dataset,
    batch_size:  int   = 256,
    num_workers: int   = 8,
    device:      torch.device = torch.device('cpu'),
    cache_path:  str   = None,
):
    """Forward *dataset* through *extract_fn* and return (features, labels).

    If *cache_path* is set and the file exists, features are loaded from disk.
    Otherwise they are computed and saved for future runs.

    Returns
    -------
    features : np.ndarray  [N, D]
    labels   : np.ndarray  [N]
    """
    if cache_path and os.path.exists(cache_path):
        d = torch.load(cache_path, map_location='cpu')
        logger.info('Loaded cached features from %s', cache_path)
        return d['features'].numpy(), d['labels'].numpy()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    all_feats, all_labels = [], []
    for x, y in loader:
        all_feats.append(extract_fn(x).cpu().float())
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
        all_labels.append(y.cpu())

    F_np = torch.cat(all_feats).numpy()
    y_np = torch.cat(all_labels).numpy()

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        torch.save(
            {'features': torch.from_numpy(F_np), 'labels': torch.from_numpy(y_np)},
            cache_path,
        )
        logger.info('Cached features to %s', cache_path)

    return F_np, y_np
