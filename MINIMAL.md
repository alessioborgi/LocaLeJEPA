# LocaLeJEPA: Minimal Example

This page explains a minimal implementation of **LocaLeJEPA**: Vision Transformer (ViT) backbone on the [ImageNette](https://github.com/fastai/imagenette) dataset with PyTorch Lightning.

1. [Method](#method)
2. [Installation](#installation)
3. [Code](#code)
4. [Case study: ViT-S/8 + Imagenette](#case-study-vit-s8--imagenette)

## Method

For each of the $V$ augmented views of image $i$, let $z_i^{(v)}\in\mathbb R^d$ be its projection and $\bar z_i=\frac1V\sum_v z_i^{(v)}$ the mean across views. We want to regularize nearby images together while allowing different groups to occupy different parts of the projection space.

We form those groups with $K$-means on the detached $\bar z_i$. The resulting clusters $C_1,\dots,C_K$ are recomputed at every step, and all views of an image share its assignment. Clusters with $|C_q|<4$ are dropped; $Q$ denotes those retained. If none remain, we treat the whole batch as one cluster.

To focus on a group's shape rather than its position, we subtract its mean separately for each view:

$$r_i^{(v)}=z_i^{(v)}-\mu_q^{(v)},\qquad \mu_q^{(v)}=\frac1{|C_q|}\sum_{j\in C_q}z_j^{(v)}.$$

We match these residuals to $Y\sim\mathrm{Unif}(\mathbb B^d_R)$, a uniform distribution on a $d$-dimensional ball. Choosing $R=\sqrt{d+2}$ gives the target identity covariance. Uniformity is the maximum-entropy choice for this fixed support.

For each retained cluster and view, we use a **sketched Epps–Pulley test** with this uniform-ball target. It compares characteristic functions along random one-dimensional directions; $D_{q,v}$ denotes the resulting discrepancy for cluster $q$ and view $v$.

We average this discrepancy over the retained clusters and views, then combine it with the prediction loss:

$$\mathcal L_{\rm LocaLeReg}=\frac1{|Q|\,V}\sum_{q\in Q}\sum_{v=1}^V D_{q,v},\qquad \mathcal L=(1-\lambda)\,\mathcal L_{\rm pred}+\lambda\,\mathcal L_{\rm LocaLeReg}.$$

Here $\mathcal L_{\rm pred}$ is the mean squared deviation of each view from $\bar z_i$. In code, the test uses 256 random directions and a 17-knot trapezoid rule on $[0,3]$.

## Installation

We prefer [uv](https://docs.astral.sh/uv/):

```bash
uv pip install torch torchvision timm lightning numpy wandb
```

or with pip:

```bash
pip install torch torchvision timm lightning numpy wandb
```

## Code

Uniform distribution matching and locality, as two pure-PyTorch functions. The repository versions are `ballreg_loss` and `local_loss` in [src/losses.py](src/losses.py).

```python
import torch
import torch.nn.functional as F
from numpy.polynomial.legendre import leggauss


def ball_cf(t, d, n_quad=64):
    # a 1-D projection of Unif(ball of radius R) has density proportional to (1 - (s/R)^2)^((d-1)/2) on [-R, R]
    x, w = (torch.tensor(a, dtype=torch.float32, device=t.device) for a in leggauss(n_quad))
    density = (1 - x.square()).pow((d - 1) / 2) * w
    return (torch.cos((d + 2) ** 0.5 * t[:, None] * x) * density).sum(-1) / density.sum()


def ballreg_loss(z, knots=17, num_slices=256, t_max=3.0):
    z = z.float()
    z = z - z.mean(-2, keepdim=True)
    t = torch.linspace(0, t_max, knots, device=z.device)
    weights = torch.full((knots,), 2 * t_max / (knots - 1), device=z.device)
    weights[[0, -1]] /= 2
    weights = weights * torch.exp(-t.square() / 2)
    A = torch.randn(z.size(-1), num_slices, device=z.device)
    A = A / A.norm(dim=0)
    x_t = (z @ A).unsqueeze(-1) * t
    err = (x_t.cos().mean(-3) - ball_cf(t, z.size(-1))).square() + x_t.sin().mean(-3).square()
    return ((err @ weights) * z.size(-2)).mean()


@torch.no_grad()
def kmeans(x, k, iters=10):
    x = F.normalize(x.float(), dim=-1)
    c = x[torch.randperm(len(x), device=x.device)[:k]]
    for _ in range(iters):
        labels = torch.cdist(x, c).argmin(1)
        sums = torch.zeros_like(c).index_add_(0, labels, x)
        counts = torch.bincount(labels, minlength=len(c))
        c = torch.where(counts[:, None] > 0, F.normalize(sums, dim=-1), c)
    return labels


def local_loss(proj, loss=ballreg_loss, k=4, min_size=4):
    labels = kmeans(proj.mean(0).detach(), k)
    sizes = torch.bincount(labels, minlength=k).tolist()
    losses = [loss(proj[:, labels == q]) for q, n in enumerate(sizes) if n >= min_size]
    return torch.stack(losses).mean() if losses else loss(proj)
```

`proj` has shape `(V, N, d)`. `local_loss(proj)` is $\mathcal L_{\rm LocaLeReg}$.

## Case study: ViT-S/8 + Imagenette

```python
import lightning as L
import timm
import torch.nn as nn
from lightning.pytorch.loggers import WandbLogger
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import Imagenette
from torchvision.ops import MLP
```

### Model

```python
class Encoder(nn.Module):
    def __init__(self, embedding_dim=512, proj_dim=16, img_size=128):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            num_classes=embedding_dim,
            img_size=img_size,
            drop_path_rate=0.1,
        )
        self.projector = MLP(embedding_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, views):
        N, V = views.shape[:2]
        emb = self.backbone(views.flatten(0, 1))
        proj = self.projector(emb).reshape(N, V, -1).transpose(0, 1)
        return emb, proj
```

`views` is `(N, V, C, H, W)`; `emb` is `(N·V, 512)`, `proj` is `(V, N, 16)`.

### Dataset

```python
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def train_transform(size):
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.08, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.64, 0.64, 0.64, 0.16)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([T.GaussianBlur(23, sigma=(0.1, 2.0))], p=0.5),
        T.RandomApply([T.RandomSolarize(128, p=0.5)], p=0.2),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def eval_transform(size):
    return T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])


class MultiView:
    def __init__(self, transform, V):
        self.transform, self.V = transform, V

    def __call__(self, img):
        return torch.stack([self.transform(img) for _ in range(self.V)])
```

### Data and Lightning modules

```python
class ImagenetteData(L.LightningDataModule):
    def __init__(self, root, V, crop_size, batch_size, num_workers):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        for split in ("train", "val"):
            Imagenette(self.hparams.root, split, size="160px", download=True)

    def setup(self, stage=None):
        h = self.hparams
        self.train_ds = Imagenette(h.root, "train", size="160px", transform=MultiView(train_transform(h.crop_size), h.V))
        self.val_ds = Imagenette(h.root, "val", size="160px", transform=MultiView(eval_transform(h.crop_size), 1))

    def _loader(self, ds, train):
        h = self.hparams
        return DataLoader(
            ds,
            batch_size=h.batch_size,
            shuffle=train,
            drop_last=train,
            num_workers=h.num_workers,
            pin_memory=True,
            persistent_workers=h.num_workers > 0,
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, train=True)

    def val_dataloader(self):
        return self._loader(self.val_ds, train=False)


class LocaLeJEPA(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.net = Encoder(cfg["embedding_dim"], cfg["proj_dim"], cfg["crop_size"])
        self.probe = nn.Sequential(nn.LayerNorm(cfg["embedding_dim"]), nn.Linear(cfg["embedding_dim"], cfg["num_classes"]))

    def training_step(self, batch, batch_idx):
        views, y = batch
        h = self.hparams
        emb, proj = self.net(views)
        inv_loss = (proj - proj.mean(0)).square().mean()
        reg_loss = local_loss(proj, k=h.k)
        lejepa_loss = (1 - h.lamb) * inv_loss + h.lamb * reg_loss
        # online probe on the first view, detached from the encoder
        emb_first = emb.reshape(views.size(0), views.size(1), -1)[:, 0]
        probe_loss = F.cross_entropy(self.probe(emb_first.detach()), y)
        self.log_dict({
            "train/inv": inv_loss,
            "train/localereg": reg_loss,
            "train/lejepa": lejepa_loss,
            "train/probe": probe_loss,
        })
        return lejepa_loss + probe_loss

    def validation_step(self, batch, batch_idx):
        views, y = batch
        emb = self.net.backbone(views.flatten(0, 1))
        self.log("val/acc", (self.probe(emb).argmax(1) == y).float().mean(), prog_bar=True)

    def configure_optimizers(self):
        h = self.hparams
        opt = torch.optim.AdamW(
            [
                {"params": self.net.parameters()},
                {"params": self.probe.parameters(), "lr": h.probe_lr, "weight_decay": h.probe_wd},
            ],
            lr=h.lr,
            weight_decay=h.wd,
        )
        total = self.trainer.estimated_stepping_batches
        warmup = total // h.epochs * h.warmup_epochs
        scheduler = SequentialLR(
            opt,
            [
                LinearLR(opt, start_factor=0.01, total_iters=warmup),
                CosineAnnealingLR(opt, T_max=total - warmup, eta_min=h.eta_min_ratio * h.lr),
            ],
            milestones=[warmup],
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
```

### Configuration

```python
cfg = dict(
    data_dir="data",
    V=4,
    crop_size=128,
    batch_size=256,
    num_workers=8,
    embedding_dim=512,
    proj_dim=16,
    num_classes=10,
    lamb=0.05,
    k=4,
    lr=1e-3,
    wd=5e-2,
    probe_lr=1e-3,
    probe_wd=1e-7,
    epochs=100,
    warmup_epochs=10,
    eta_min_ratio=1e-3,
    seed=44,
)
```

### Main training loop

```python
if __name__ == "__main__":
    L.seed_everything(cfg["seed"])
    data = ImagenetteData(cfg["data_dir"], cfg["V"], cfg["crop_size"], cfg["batch_size"], cfg["num_workers"])
    trainer = L.Trainer(
        max_epochs=cfg["epochs"],
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        logger=WandbLogger(project="LocaLeJEPA"),
    )
    trainer.fit(LocaLeJEPA(cfg), datamodule=data)
```

The same recipe in the repository pipeline, which uses `fast-pytorch-kmeans` for the clustering:

```bash
uv run python src/train.py experiment=minimal
```

### Training curves

Run `fixed_k_20260922_l0.05_k4_B256x1_seed44`, 100 epochs.

| | |
|:-------------------------:|:-------------------------:|
| **$\mathcal L_{\rm pred}$** | **$\mathcal L_{\rm LocaLeReg}$** |
| <img width="500" alt="Prediction loss" src="assets/minimal/inv.svg"> | <img width="500" alt="LocaLeReg loss" src="assets/minimal/localereg.svg"> |
| **$(1-\lambda)\mathcal L_{\rm pred}+\lambda\mathcal L_{\rm LocaLeReg}$** | **Online probe loss** |
| <img width="500" alt="Combined loss" src="assets/minimal/localejepa.svg"> | <img width="500" alt="Online probe loss" src="assets/minimal/probe.svg"> |

| |
|:-------------------------:|
| **Online probe accuracy, LocaLeJEPA vs LeJEPA** |
| <img width="1604" alt="Online probe accuracy" src="assets/minimal/acc.svg"> |
