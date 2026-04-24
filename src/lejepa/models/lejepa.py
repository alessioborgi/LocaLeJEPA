import torch
import torch.nn as nn
import timm
from torchvision.ops import MLP


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization.

    Enforces an isotropic Gaussian distribution on the projection embeddings
    using the Epps-Pulley characteristic function test projected along random
    1-D directions. Integrates on [0, t_max] and doubles weights to exploit
    the symmetry of the ECF for free quadrature improvement.
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


class ViTEncoder(nn.Module):
    """timm ViT backbone with an MLP projection head.

    Forward returns (emb, proj) where:
      - emb  : (N*V, emb_dim)  — used by the online linear probe
      - proj : (V, N, proj_dim) — used by the LeJEPA loss
    """

    def __init__(
        self,
        model_name="vit_small_patch8_224",
        emb_dim=512,
        proj_dim=128,
        img_size=128,
        drop_path_rate=0.1,
    ):
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
