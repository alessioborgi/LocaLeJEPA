import torch
import torch.nn as nn
import timm
import torchvision.models as tvmodels
from torchvision.ops import MLP


# ---------------------------------------------------------------------------
# SIGReg regularizer (shared across all backbone choices)
# ---------------------------------------------------------------------------

class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization.

    Enforces an isotropic Gaussian distribution on projection embeddings via
    the Epps-Pulley characteristic function test along random 1-D projections.
    Integrates on [0, t_max] and doubles weights to exploit ECF symmetry.
    """

    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        device = proj.device
        A = torch.randn(proj.size(-1), 256, device=device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


# ---------------------------------------------------------------------------
# Backbone encoders — uniform interface: forward(x) -> (emb, proj)
#   x   : (N, V, C, H, W)
#   emb : (N*V, emb_dim)   — used by the online linear probe
#   proj: (V, N, proj_dim) — used by the LeJEPA loss
# ---------------------------------------------------------------------------

class ViTEncoder(nn.Module):
    """timm ViT backbone with MLP projection head."""

    def __init__(self, model_name, emb_dim, proj_dim, img_size, drop_path_rate=0.1):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=emb_dim,
            drop_path_rate=drop_path_rate,
            img_size=img_size,
        )
        self.proj = MLP(emb_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)


class ResNetEncoder(nn.Module):
    """torchvision ResNet (18 or 50) backbone with MLP projection head."""

    _FEAT_DIMS = {'resnet18': 512, 'resnet50': 2048}

    def __init__(self, arch, emb_dim, proj_dim):
        super().__init__()
        if arch not in self._FEAT_DIMS:
            raise ValueError(f'Unknown ResNet arch {arch!r}. Choose resnet18 or resnet50.')
        backbone = getattr(tvmodels, arch)(weights=None)
        feat_dim = self._FEAT_DIMS[arch]
        backbone.fc = nn.Linear(feat_dim, emb_dim)
        self.backbone = backbone
        self.proj = MLP(emb_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)


# ---------------------------------------------------------------------------
# Factory — selects the right encoder from the backbone config section
# ---------------------------------------------------------------------------

def build_encoder(backbone_cfg):
    """Build a LeJEPA encoder from the ``backbone:`` config section.

    backbone.name choices:
      - ``vit``      — timm ViT (backbone.vit_model specifies the exact model)
      - ``resnet18`` — torchvision ResNet-18
      - ``resnet50`` — torchvision ResNet-50
    """
    name     = backbone_cfg['name']
    emb_dim  = backbone_cfg['emb_dim']
    proj_dim = backbone_cfg['proj_dim']

    if name == 'vit':
        return ViTEncoder(
            model_name=backbone_cfg['vit_model'],
            emb_dim=emb_dim,
            proj_dim=proj_dim,
            img_size=backbone_cfg.get('img_size', 224),
            drop_path_rate=backbone_cfg.get('drop_path_rate', 0.1),
        )
    if name in ('resnet18', 'resnet50'):
        return ResNetEncoder(arch=name, emb_dim=emb_dim, proj_dim=proj_dim)
    if name == 'phc':
        from src.phc.models.phc_encoder import PhCEncoder
        return PhCEncoder(emb_dim=emb_dim, proj_dim=proj_dim)

    raise ValueError(
        f'Unknown backbone {name!r}. '
        'Supported choices: vit | resnet18 | resnet50 | phc'
    )
