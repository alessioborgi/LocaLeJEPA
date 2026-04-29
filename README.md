# BundLeJEPA

**BundLeJEPA** is a research codebase for self-supervised representation learning across two domains — natural images and photonic crystal (PhC) unit cells.  The repository implements an Equivariant extension of **LeJEPA**, called **Bundle LeJEPA**, which incorporates fibre-bundle geometry to learn representations that transform predictably under symmetry operations.  The code that exists today implements:

- **LeJEPA** pretraining (Isotropic Gaussian regularisation + invariance loss) with ViT, ResNet-50, ResNet-18, and a custom PhC CNN backbone.
- **I-JEPA** pretraining (masked-patch prediction in latent space) with ViT and PhC CNN
- **SimCLR** baseline (NT-Xent contrastive loss, ResNet-50)
- **Supervised PhC baseline** (CNN + DOS regression, no pretraining)
- A complete **linear evaluation pipeline** (11 downstream vision datasets + VOC07+12 Faster R-CNN detection)
- **Photonic crystal DOS regression** evaluation (frozen and fine-tuned protocols)

---

## Table of Contents

1. [Background](#1-background)
2. [Repository structure](#2-repository-structure)
3. [Installation](#3-installation)
4. [Methods implemented](#4-methods-implemented)
   - 4.1 [LeJEPA](#41-lejepa)
   - 4.2 [I-JEPA](#42-i-jepa)
   - 4.3 [SimCLR baseline](#43-simclr-baseline)
   - 4.4 [Supervised PhC baseline](#44-supervised-phc-baseline)
5. [Backbones](#5-backbones)
6. [Loss functions](#6-loss-functions)
7. [Datasets and data loaders](#7-datasets-and-data-loaders)
8. [Data augmentation](#8-data-augmentation)
9. [Evaluation pipeline](#9-evaluation-pipeline)
   - 9.1 [Linear probe (vision)](#91-linear-probe-vision)
   - 9.2 [VOC07+12 detection](#92-voc0712-detection)
   - 9.3 [DOS regression (photonics)](#93-dos-regression-photonics)
10. [Configuration system](#10-configuration-system)
11. [Running experiments](#11-running-experiments)
12. [Checkpointing and logging](#12-checkpointing-and-logging)
13. [What is Bundle LeJEPA?](#13-what-is-bundle-lejepa)

---

## 1. Background

### LeJEPA

**LeJEPA** (Learning with Joint-Embedding Predictive Architectures) is a multi-view self-supervised learning method whose loss has two terms:

```
L = (1 - λ) · L_inv  +  λ · L_SIGReg
```

- **L_inv (invariance loss)**: pulls every view's projection toward the mean projection of the *global* views.  The reference and the set of views pulled toward it depend on the backbone:
  - **ViT-S/16**: asymmetry is active.  The reference is the mean of the 2 global crops only (scale 0.3–1.0).  All 8 views (2 global + 6 local) are pulled toward it, but local crops (scale 0.05–0.3) do not contribute to the reference.  This is intentional: a ViT attending over a large crop produces a semantically complete representation; a ViT attending over a tiny patch sees only a fragment and should not vote on the anchor.
  - **ResNet-18 / ResNet-50**: asymmetry is collapsed.  Global average pooling discards spatial structure, so both large-crop and small-crop representations are the same kind of vector.  There is no meaningful distinction between "seeing the whole" and "seeing a part" at the representation level.  All V views contribute to the reference and are pulled toward it equally — equivalent to a standard BYOL/SimCLR-style invariance loss.
  - **PhCEncoder (CNN-based, for photonic crystals)**: same as ResNet — no asymmetry.  The unit cell is always 32×32; there are no crops of different scales.  All V views are peer augmentations (C4v rotations, toroidal rolls, refractive scaling) with no global/local hierarchy in the physics, so all views define the reference jointly.
- **L_SIGReg (Sketched Isotropic Gaussian Regularisation)**: forces the projected embeddings to be distributed like an isotropic Gaussian via the Epps-Pulley characteristic function test along 256 random 1-D projections.  This prevents representation collapse without a momentum encoder or stop-gradient.

λ = 0.02 is the recommended starting point.

### I-JEPA

**I-JEPA** (Image JEPA, He et al. 2022) predicts latent representations of masked target patches from a context (unmasked) view.  It uses a momentum target encoder and requires no data augmentation at the image level, since the masking itself acts as the view generator.

### BundLeJEPA 

The goal of this repository is to implement **Bundle LeJEPA**, which extends LeJEPA with fibre-bundle geometry.  Instead of learning a single invariant representation, Bundle LeJEPA will learn a *section* of a principal bundle: a representation that transforms predictably (equivariantly) under the symmetry group of the data.  The groundwork for this is already laid:

- **record_metadata=True** records per-view crop box and flip state during pretraining (phase 1), so the transformation applied to each view is known.
- **Domain-specific PhC transforms** (RandomC4v, RandomMirror, RandomRoll, RandomRefractiveScaling) are individually composable and labelled, making it possible to later split them into invariant vs. equivariant groups.
- The loss is factored into two clean standalone functions (`invariance_loss`, `sigreg_loss`) that are easy to extend with equivariant terms.

---

## 2. Repository structure

```
BundLeJEPA/
├── main.py                        # Single-machine multi-GPU entry point (mp.spawn)
├── main_distributed.py            # Cluster/submitit entry point
├── eval_linear.py                 # Vision linear evaluation entry point
├── eval_voc.py                    # VOC07+12 Faster R-CNN evaluation entry point
│
├── configs/
│   ├── lejepa_rn18_stl10_ep500.yaml    # LeJEPA ResNet-18 on STL10, 500 epochs
│   ├── lejepa_rn50_in100_ep500.yaml    # LeJEPA ResNet-50 on ImageNet-100, 500 epochs
│   ├── lejepa_vits16_in100_ep500.yaml  # LeJEPA ViT-S/16 on ImageNet-100, 500 epochs
│   ├── phc_blob_ep250.yaml             # LeJEPA PhCEncoder on Blob dataset, 250 epochs
│   ├── phc_gpm_ep250.yaml              # LeJEPA PhCEncoder on Gpm dataset, 250 epochs
│   ├── ijepa_phc_blob_ep250.yaml       # I-JEPA ViT on PhC Blob, 250 epochs
│   ├── ijepa_phc_gpm_ep250.yaml        # I-JEPA ViT on PhC Gpm, 250 epochs
│   ├── simclr_rn50_in100_ep500.yaml    # SimCLR ResNet-50 on ImageNet-100, 500 epochs
│   ├── phc_supervised_ep500.yaml       # Supervised CNN baseline on PhC, 500 epochs
│   ├── eval_linear.yaml                # Linear probe config (11 datasets)
│   └── eval_voc.yaml                   # VOC07+12 detection config
│
└── src/
    ├── losses.py                  # Standalone loss functions (invariance, SIGReg, NT-Xent)
    ├── transforms.py              # Vision augmentation pipelines
    │
    ├── lejepa/
    │   ├── models/lejepa.py       # ViTEncoder, ResNetEncoder, build_encoder()
    │   ├── helper.py              # init_model, init_probe, init_opt, load_checkpoint
    │   └── train.py               # LeJEPA training loop
    │
    ├── ijepa/
    │   ├── models/                # ViT architecture (from I-JEPA original repo)
    │   ├── masks/                 # Block mask collator for I-JEPA
    │   ├── helper.py              # I-JEPA model initialisation
    │   └── train.py               # I-JEPA training loop
    │
    ├── phc/
    │   ├── models/phc_encoder.py  # PhCEncoder (CNN+FC) and DOSPredictor
    │   ├── transforms.py          # RandomC4v, RandomMirror, RandomRoll, RandomRefractiveScaling
    │   ├── helper.py              # PhC model/optimiser init utilities
    │   └── eval.py                # DOS regression evaluation (eps_dos, frozen/fine-tuned)
    │
    ├── baselines/
    │   ├── simclr/
    │   │   ├── model.py           # SimCLREncoder (ResNet-50 + MLP projector)
    │   │   └── train.py           # SimCLR training loop
    │   └── phc_supervised.py      # Supervised PhC baseline training loop
    │
    ├── datasets/
    │   ├── imagenet1k.py          # ImageNet-1k / ImageNet-100 loaders
    │   ├── imagenette.py          # Imagenette loader
    │   ├── stl10.py               # STL10 loader (unlabeled split for LeJEPA)
    │   └── phc.py                 # PhCDataset, make_phc_loaders, make_phc_ijepa_loaders
    │
    ├── eval/
    │   ├── datasets.py            # 11 downstream dataset loaders for linear evaluation
    │   ├── extract.py             # Backbone feature extraction with disk caching
    │   ├── linear_probe.py        # L-BFGS linear probe with val-split + retrain protocol
    │   └── voc_detection.py       # VOC07+12 Faster R-CNN via detectron2
    │
    └── utils/
        ├── distributed.py         # init_distributed (NCCL / single-process fallback)
        ├── logging.py             # CSVLogger, AverageMeter
        ├── schedulers.py          # LR scheduler helpers
        └── tensors.py             # Tensor utility functions
```

---

## 3. Installation

```bash
# Core dependencies
pip install torch torchvision timm

# Downstream evaluation
pip install scikit-learn          # linear probe
pip install pyyaml numpy pillow

# Optional — VOC detection (requires Linux + CUDA)
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Optional — cluster submission
pip install submitit

# Optional — experiment tracking
pip install wandb
```

**Python 3.10+** and **PyTorch 2.0+** are recommended.  All training loops support `bfloat16` mixed precision via `torch.amp.autocast`.

---

## 4. Methods implemented

### 4.1 LeJEPA

LeJEPA is configured with `method: lejepa` in the YAML.

**Training loop** (`src/lejepa/train.py`):

1. Build the encoder (ViT, ResNet, or PhCEncoder) and an optional online linear probe.
2. For each batch, apply the multi-crop transform → tensor `(N, V, C, H, W)`.
3. Forward the batch through the encoder:
   - `emb : (N×V, emb_dim)` — fed to the online probe
   - `proj: (V, N, proj_dim)` — fed to the loss
4. Compute `l_inv = invariance_loss(proj, num_global_crops)` and `l_sig = sigreg_loss(proj)`.
5. `lejepa_loss = l_sig * λ + l_inv * (1 - λ)`.
6. Optionally add the online probe's cross-entropy loss (for in-training accuracy monitoring).
7. AdamW optimiser with linear warmup (1 epoch) followed by cosine annealing.

The **online linear probe** is a single `LayerNorm → Linear` head that runs on `emb.detach()`.  It is not used in evaluation — it only provides a quick signal during pretraining that the representations are semantically meaningful.

**Pretraining configurations:**

| Config | Backbone | Dataset | Epochs | Batch | WD |
|---|---|---|---|---|---|
| `lejepa_rn18_stl10_ep500.yaml` | ResNet-18 | STL10 (unlabeled) | 500 | 256 | 5e-4 |
| `lejepa_rn50_in100_ep500.yaml` | ResNet-50 | ImageNet-100 | 500 | 256 | 5e-4 |
| `lejepa_vits16_in100_ep500.yaml` | ViT-S/16 | ImageNet-100 | 500 | 256 | 5e-2 |
| `phc_blob_ep250.yaml` | PhCEncoder | PhC Blob | 250 | 512 | — |
| `phc_gpm_ep250.yaml` | PhCEncoder | PhC Gpm | 250 | 512 | — |

LR = 5e-4 for all.  Linear warmup 1 epoch, cosine decay to ~0.

### 4.2 I-JEPA

I-JEPA is configured with `method: ijepa`.

**Training loop** (`src/ijepa/train.py`):

1. A single augmented view per image is produced.  A block-mask collator (`src/ijepa/masks/`) splits it into a context region and one or more target regions.
2. The **context encoder** (ViT) processes the visible patches.
3. The **predictor** (narrow ViT) predicts latent representations of target patches from context tokens + mask-position embeddings.
4. The **target encoder** (EMA of context encoder) produces the ground-truth target representations.
5. Loss = mean squared error between predicted and target representations.

For PhC datasets, `backbone.name: vit` is used with `in_chans: 1` (grayscale), `patch_size: 4`, and `crop_size: 32` to match the 32×32 unit cell resolution.

**PhC I-JEPA configurations:**

| Config | Dataset | Backbone | in_chans | patch_size |
|---|---|---|---|---|
| `ijepa_phc_blob_ep250.yaml` | PhC Blob | ViT | 1 | 4 |
| `ijepa_phc_gpm_ep250.yaml` | PhC Gpm | ViT | 1 | 4 |

### 4.3 SimCLR baseline

SimCLR is configured with `method: simclr`.

**Model** (`src/baselines/simclr/model.py`):

- `SimCLREncoder`: torchvision ResNet-50 with `fc` replaced by `nn.Identity()`, followed by a 2-layer MLP projector (2048 → 2048 → proj_dim=128 with BatchNorm).
- Forward: `(N, 2, C, H, W)` → `(2, N, proj_dim)` — matches the LeJEPA interface exactly.

**Training** (`src/baselines/simclr/train.py`):

- Exactly 2 views per image (standard SimCLR).
- Loss: `nt_xent_loss(proj, temperature=0.07)` — NT-Xent contrastive loss.
- SGD + momentum 0.9 + cosine LR decay.
- 500 epochs, batch size 256, ImageNet-100.

**Config:** `configs/simclr_rn50_in100_ep500.yaml`

### 4.4 Supervised PhC baseline

Configured with `method: phc_supervised`.

**What it does** (`src/baselines/phc_supervised.py`):

- Trains `PhCEncoder` + `DOSPredictor` end-to-end from scratch — no pretraining.
- Uses only **3 000** labeled training samples (randomly subsampled from the training split).
- Loss: `nn.MSELoss()` between predicted and ground-truth 400-dim DOS vectors.
- Optimiser: Adam + cosine LR decay, 500 epochs.
- Evaluation: ε_DOS = Σ|pred − DOS| / Σ|DOS| × 100 (%) on the held-out test split.

**Expected results** (from Dangovski et al., ICLR 2022):

| Dataset | ε_DOS (supervised, no pretraining) |
|---|---|
| Blob | ~1.07 % |
| Gpm | ~3.21 % |

LeJEPA/I-JEPA pretraining should improve on these numbers.

**Config:** `configs/phc_supervised_ep500.yaml`

---

## 5. Backbones

All backbones expose the same interface:

```python
emb, proj = encoder(x)
# x    : (N, V, C, H, W)   during pretraining
#         (N, 1, C, H, W)   during eval (single view)
# emb  : (N*V, emb_dim)
# proj : (V, N, proj_dim)
```

### ViTEncoder (`src/lejepa/models/lejepa.py`)

- **timm** ViT with `num_classes=emb_dim` (classification head is repurposed as embedding head).
- MLP projector: `emb_dim → 2048 → 2048 → proj_dim` with BatchNorm1d.
- Configurable: `vit_model`, `emb_dim`, `proj_dim`, `img_size`, `drop_path_rate`.

Default for ImageNet-100: `vit_small_patch16_224`, `emb_dim=384`, `proj_dim=128`.

### ResNetEncoder (`src/lejepa/models/lejepa.py`)

- **torchvision** ResNet-18 or ResNet-50 with `fc` replaced by `nn.Linear(feat_dim, emb_dim)`.
- Same MLP projector as ViT.
- `emb_dim=512` (ResNet-18), `emb_dim=2048` (ResNet-50).
- No global/local crop asymmetry in the loss (all views are treated as global).

### PhCEncoder (`src/phc/models/phc_encoder.py`)

Designed for 32×32 single-channel photonic crystal unit cells.

```
Input: (N, 1, 32, 32)

CNN stack:
  Conv7×7(1→64)   + BN + ReLU + MaxPool2×2  →  (N, 64, 16, 16)
  Conv7×7(64→256) + BN + ReLU + MaxPool2×2  →  (N, 256, 8, 8)
  Conv7×7(256→256)+ BN + ReLU + MaxPool2×2  →  (N, 256, 4, 4)

Flatten:  (N, 4096)

FC stack:
  Linear(4096→1024) + ReLU
  Linear(1024→emb_dim)                       →  (N, emb_dim)

MLP projector:
  emb_dim → 512 → proj_dim   (BN between layers)
```

`emb_dim=1024`, `proj_dim=128` by default.

### DOSPredictor (`src/phc/models/phc_encoder.py`)

A 4-layer MLP head attached to PhCEncoder for DOS regression:

```
emb_dim → 1024 → 1024 → 512 → 400
```

Outputs a 400-dimensional density-of-states vector.

---

## 6. Loss functions

All three losses live in `src/losses.py` and operate on projections in **(V, N, D)** layout — V views, N samples, D embedding dimension.

### `invariance_loss(proj, num_global=None)`

Pulls every view's projection toward the mean of the reference (global) views:

```
L_inv = mean_{v,n} || proj[v,n] - mean_g(proj[:num_global,:,:])[n] ||^2
```

When `num_global=None`, the reference is the mean over all views.  For ResNet and PhC backbones `num_global = num_views`, so there is no asymmetry.

### `sigreg_loss(proj, knots=17)`

Sketched Isotropic Gaussian Regularisation (SIGReg).  Enforces an isotropic Gaussian distribution on the projected embeddings via the Epps-Pulley characteristic function test:

1. Sample 256 random unit-norm directions A ∈ R^(D×256).
2. Project: `x_t = (proj @ A) * t`  where `t` is 17 log-spaced knots on [0, 3].
3. Compare the empirical characteristic function (ECF) of the projections to the Gaussian characteristic function `φ(t) = exp(−t²/2)`.
4. Integrate the squared error with trapezoid weights (ECF symmetry exploited to double inner weights).

Always casts to float32 internally so it is safe under bfloat16 autocast.

### `nt_xent_loss(proj, temperature=0.07)`

Normalised Temperature-scaled Cross Entropy (SimCLR, Chen et al. 2020).  Requires exactly 2 views:

1. L2-normalise each view: `z1, z2 ∈ R^(N, D)`.
2. Concatenate into `z ∈ R^(2N, D)`.
3. Compute similarity matrix `sim = (z @ z.T) / τ`, mask diagonal.
4. Positive pair for sample `i ∈ [0, N)` is `i + N` and vice versa.
5. Return `F.cross_entropy(sim, labels)`.

---

## 7. Datasets and data loaders

### Vision pretraining datasets

| Dataset | Module | Notes |
|---|---|---|
| ImageNet-1k | `src/data/imagenet1k_datamodule.py` | Full ImageNet; `class_subset_file` enables ImageNet-100 |
| ImageNet-100 | same | 100-class subset from Tian et al. (CMC, ECCV 2020) |
| STL10 | `src/data/stl10_datamodule.py` | Unlabeled split (105 000 images) used for LeJEPA |
| Imagenette | `src/data/imagenette_datamodule.py` | 10-class ImageNet subset; auto-downloaded |

Each Lightning datamodule can be selected through `configs/data/*.yaml`. Compatibility factories such as `make_<dataset>_lejepa(train_transform, val_transform, ...)` remain in the datamodule files for the legacy manual training loops and return `(train_loader, val_loader, train_sampler)`.

### Photonic crystal datasets (`src/data/phc_datamodule.py`)

**PhCDataset**:

- Loads a `.npz` file containing `images: (N, 32, 32)` and `dos: (N, 400)`.
- Normalises pixel values to [0, 1] and adds a channel dim → `(N, 1, 32, 32)`.
- Splits by seeded random permutation: `train_frac=0.6` train, remainder test.
- Accepts any transform compatible with `(1, 32, 32)` float tensors.

Two dataset variants:

| Dataset | Structure | Invariances |
|---|---|---|
| **Blob** | Centred shape variation | C4v (4 rotations × 2 mirror states) |
| **Gpm** | Horizontal mirror symmetry | Rolling translations + mirrors |

Loaders:

- `make_phc_loaders(...)` — standard LeJEPA loaders (returns `(train_loader, val_loader, train_sampler)`)
- `make_phc_ijepa_loaders(...)` — I-JEPA loaders with mask collator support; always uses `DistributedSampler` (required by `ijepa/train.py` which calls `sampler.set_epoch()` unconditionally)

### Linear evaluation datasets (`src/eval/datasets.py`)

11 downstream classification datasets used in the linear probe evaluation:

| Dataset | Classes | Metric | Source |
|---|---|---|---|
| CIFAR-10 | 10 | Top-1 | torchvision |
| CIFAR-100 | 100 | Top-1 | torchvision |
| Food-101 | 101 | Top-1 | torchvision |
| MIT-67 | 67 | Top-1 | manual download |
| Oxford Pets | 37 | Mean per-class | torchvision |
| Oxford Flowers-102 | 102 | Mean per-class | torchvision |
| Caltech-101 | 101 | Mean per-class | torchvision |
| Stanford Cars | 196 | Top-1 | torchvision |
| FGVC Aircraft | 100 | Mean per-class | torchvision |
| DTD | 47 | Top-1 | torchvision |
| SUN397 | 397 | Top-1 | torchvision |

All use the same evaluation transform: `Resize(256, BICUBIC) → CenterCrop(crop_size) → ToTensor → Normalize(ImageNet stats)`.

---

## 8. Data augmentation

### Vision: DINO-style multi-crop (`src/transforms.py`)

`make_multicrop_transforms(...)` implements the DINO augmentation strategy:

- **Global crops** (2 views, scale 0.3–1.0, output 224×224)
- **Local crops** (6 views, scale 0.05–0.3, output 224×224 — same size, tighter crop for zoom-in semantics)
- Photometric augmentations applied to all views: ColorJitter, RandomGrayscale, GaussianBlur (p=0.5), RandomSolarize (p=0.2), RandomHorizontalFlip
- All views in a batch stack to `(N, V=8, C, H, W)` — single contiguous tensor

**Transformation metadata recording** (for future equivariance): when `record_metadata=True`, each call returns `(views, meta)` where `meta: (V, 6)` stores per-view `[top, left, height, width, scale, flipped]` in [0, 1].  The DataLoader then yields `((batch_views, batch_meta), batch_labels)`.

### Vision: Legacy uniform multi-view (`src/transforms.py`)

`make_lejepa_transforms(...)` applies the same single-resolution transform V times independently.  Used when `num_global_crops` is set without `num_local_crops` in the config (backward-compatible path).

### Photonic crystal: domain-specific transforms (`src/phc/transforms.py`)

All transforms operate on `(1, 32, 32)` float tensors in pixel space (not PIL images):

| Class | What it does |
|---|---|
| `RandomC4v` | Uniformly samples one of 8 C4v operations: 4 rotations (0°, 90°, 180°, 270°) × (identity + horizontal flip) |
| `RandomMirror` | Uniformly samples horizontal flip, vertical flip, or identity |
| `RandomRoll` | Toroidal (periodic) translation by a uniformly random (dy, dx) offset |
| `RandomRefractiveScaling` | Scales pixel values by a uniform factor in [lo, hi] — simulates refractive index variation |

`make_phc_transforms(num_views, ...)` → `PhCMultiViewTransform` — generates V independently augmented views per unit cell, returning `(V, 1, 32, 32)`.

`make_phc_single_transform(...)` → single-view transform for I-JEPA (returns one `(1, 32, 32)` tensor).

Each transform class is individually accessible, making it straightforward to later annotate which operations belong to the invariant group vs. the equivariant group for Bundle LeJEPA.

---

## 9. Evaluation pipeline

### 9.1 Linear probe (vision)

Entry point: `python eval_linear.py --config configs/eval_linear.yaml`

**Protocol** (matching Yu et al., NeurIPS 2024):

1. **Feature extraction** (`src/eval/extract.py`): freeze the backbone, forward center-cropped 224×224 images.
   - *ViT* (LeJEPA or I-JEPA): register hooks on the last 2 transformer blocks, collect the CLS token from each, concatenate → `(N, 2×embed_dim)`, then LayerNorm.
   - *ResNet*: use the output of the modified `fc` head → `(N, emb_dim)`, then LayerNorm.
   - *PhCEncoder*: use the single-view `emb` output → `(N, emb_dim)`, then LayerNorm.
   - Features can be cached to disk (keyed by checkpoint name + dataset + split) to avoid re-extraction when sweeping probe hyper-parameters.

2. **L-BFGS linear probe** (`src/eval/linear_probe.py`):
   - Carve a validation split (20%) from the training features by seeded random selection.
   - Sweep 45 log-spaced λ values in [1e-6, 1e5] with `LogisticRegression(solver='lbfgs', C=1/λ, max_iter=5000)`, selecting the best λ on the validation split.
   - Retrain with the best λ on the **full** training set (train + val combined).
   - Report top-1 accuracy on the held-out test set.

3. **Output**: a summary table printed to stdout + a CSV written to `output_folder/eval_<checkpoint>.csv`.

**Key config options** (`configs/eval_linear.yaml`):

```yaml
checkpoint:      logs/lejepa_rn50_in100/lejepa-latest.pth.tar
pretrain_config: logs/lejepa_rn50_in100/params-lejepa.yaml
datasets_root:   /path/to/datasets
cache_dir:       eval_cache/
use_cache:       true
n_reg:           45       # number of λ values
max_iter:        5000     # L-BFGS iterations per fit
val_frac:        0.2      # fraction of train held out for λ selection
```

### 9.2 VOC07+12 detection

Entry point: `python eval_voc.py --config configs/eval_voc.yaml`

**Protocol** (§3.4 of implementation plan):

- **Backbone**: ResNet-50 only (C4 architecture); ViT requires ViTDet (not included).
- **Train**: VOC2007 trainval + VOC2012 trainval (≈16 551 images).
- **Test**: VOC2007 test.
- **Architecture**: Faster R-CNN with C4 backbone via detectron2.
- **Weight initialisation**: LeJEPA ResNet-50 weights are remapped from LeJEPA key names (`backbone.layerN`) to detectron2 key names (`backbone.bottom_up.resN+1`).
- **Training schedule**:
  - 24 000 iterations, batch size 16
  - Linear warmup for 1 000 steps (start factor 1/1000)
  - LR = 0.1, step decay at 18 000 and 22 000 iterations
- **Metrics**: AP_all, AP50, AP75 (reported by PascalVOCDetectionEvaluator).

**Key config options** (`configs/eval_voc.yaml`):

```yaml
voc_root:     /path/to/VOCdevkit
base_lr:      0.1
max_iter:     24000
warmup_iters: 1000
batch_size:   16
output_dir:   logs/voc_detection/
```

### 9.3 DOS regression (photonics)

Entry point: internal, called from `src/phc/eval.py`

**Metric**: ε_DOS = Σ|DOS_pred − DOS| / Σ|DOS| × 100  (lower is better)

**Two evaluation protocols**:

| Protocol | Backbone | What it tests |
|---|---|---|
| Frozen | Weights locked | Quality of pretrained representations directly |
| Fine-tuned | Weights updated | Quality of the initialisation for the full model |

**Procedure** (both protocols):
1. Attach a `DOSPredictor` head to the pretrained backbone.
2. Fine-tune the head (frozen protocol) or both head + backbone (fine-tuned protocol) for 100 epochs with Adam, batch size 64, LR grid {1e-3, 1e-4, 1e-5}.  No augmentations during fine-tuning.
3. Report mean ± std of ε_DOS over 3 independent random splits of the labeled data.

---

## 10. Configuration system

All experiments are driven by YAML files.  The top-level `method` key selects the training loop:

```yaml
method: lejepa         # → src/lejepa/train.py
method: ijepa          # → src/ijepa/train.py
method: simclr         # → src/baselines/simclr/train.py
method: phc_supervised # → src/baselines/phc_supervised.py
```

A typical LeJEPA config has five sections:

```yaml
method: lejepa

backbone:
  name:     resnet50          # vit | resnet18 | resnet50 | phc
  emb_dim:  2048
  proj_dim: 128

data:
  dataset:          imagenet100
  root_path:        /path/to/imagenet
  batch_size:       256
  num_global_crops: 2
  num_local_crops:  6
  record_metadata:  true      # store crop/flip metadata for equivariance

lejepa:
  lamb:        0.02           # SIGReg weight
  num_classes: 100
  use_probe:   true           # online linear probe during training

optimization:
  epochs:       500
  warmup:       1             # epochs of linear warmup
  lr:           5.0e-4
  weight_decay: 5.0e-4
  use_bfloat16: true

logging:
  folder:   logs/run_name/
  write_tag: lejepa
  use_wandb: false
```

For PhC datasets, add a `data.path` key pointing to the `.npz` file and remove image-specific keys (`crop_size`, `use_gaussian_blur`, etc.) or simply leave them — they are accessed with `.get()` and have safe defaults.

---

## 11. Running experiments

### Single machine (multi-GPU)

```bash
python main.py --fname configs/lejepa_rn50_in100_ep500.yaml \
               --devices cuda:0 cuda:1 cuda:2 cuda:3
```

### Cluster (submitit / SLURM)

```bash
python main_distributed.py --fname configs/lejepa_rn50_in100_ep500.yaml \
    --nodes 2 --tasks-per-node 8 --partition gpu
```

### PhC pretraining with LeJEPA

```bash
# Edit configs/phc_blob_ep250.yaml: set data.path to your .npz file
python main.py --fname configs/phc_blob_ep250.yaml
```

### PhC pretraining with I-JEPA

```bash
python main.py --fname configs/ijepa_phc_blob_ep250.yaml
```

### SimCLR baseline

```bash
python main.py --fname configs/simclr_rn50_in100_ep500.yaml
```

### Supervised PhC baseline

```bash
python main.py --fname configs/phc_supervised_ep500.yaml
```

### Linear evaluation (11 datasets)

```bash
# 1. Edit configs/eval_linear.yaml: set checkpoint, pretrain_config, datasets_root
python eval_linear.py --config configs/eval_linear.yaml
```

### VOC07+12 detection

```bash
# 1. Edit configs/eval_voc.yaml: set voc_root, checkpoint, pretrain_config
# 2. Requires a ResNet-50 LeJEPA checkpoint
python eval_voc.py --config configs/eval_voc.yaml
```

### DOS regression evaluation

```python
from src.phc.eval import evaluate

eps_frozen, eps_finetuned = evaluate(
    backbone,          # pretrained PhCEncoder
    data_path='phc.npz',
    device='cuda',
    emb_dim=1024,
    num_splits=3,
    fine_tune=True,    # also run fine-tuned protocol
)
```

---

## 12. Checkpointing and logging

**Checkpoints** are saved to `logging.folder/` as:

- `<write_tag>-latest.pth.tar` — always overwritten with the most recent epoch
- `<write_tag>-ep<N>.pth.tar` — saved every `checkpoint_freq=50` epochs

Each checkpoint contains:
```python
{
    'net':        encoder.state_dict(),
    'probe':      probe.state_dict(),   # None if use_probe=False
    'opt':        optimizer.state_dict(),
    'scaler':     gradscaler.state_dict(),
    'epoch':      int,
    'lr':         float,
    'batch_size': int,
    'world_size': int,
}
```

**The training config** is saved as `params-lejepa.yaml` (or `params-ijepa.yaml`, `params-simclr.yaml`) inside the log folder.  This file is read back by `src/eval/extract.py` to reconstruct the architecture automatically, without requiring the original command-line arguments.

**CSV logs** are written by `CSVLogger` (`src/utils/logging.py`) to `<write_tag>_r<rank>.csv` with one row per epoch:

| Column | Description |
|---|---|
| `epoch` | Epoch number |
| `lejepa_loss` | Combined LeJEPA loss |
| `sigreg_loss` | SIGReg term alone |
| `inv_loss` | Invariance term alone |
| `probe_loss` | Online probe cross-entropy |
| `test_acc` | Online probe top-1 accuracy |

**WandB** logging is optional — set `logging.use_wandb: true` and call `wandb login` before training.

---

## 13. What is Bundle LeJEPA?

The name **BundLeJEPA** is short for **Bundle LeJEPA**.  The planned extension introduces fibre-bundle geometry into the joint-embedding framework:

- A **principal bundle** `(E, B, G, π)` consists of a total space `E`, base space `B`, structure group `G` (the symmetry group of the data), and a projection `π: E → B`.
- A **section** of the bundle assigns to each base point a point in its fibre — this is exactly what an equivariant representation should do: each input maps to an embedding that transforms predictably under the action of `G`.

In the photonic crystal domain, the relevant groups are C4v (Blob dataset) and the Gpm wallpaper group (Gpm dataset).  In the vision domain, the relevant group includes 2D similarity transformations (crops, flips, scales).

**Bundle LeJEPA will:**

1. Decompose the projection head into *invariant* and *equivariant* branches.
2. Modify the invariance loss into a group-equivariance consistency loss: instead of collapsing all views to the same point, require `f(g·x) = ρ(g) · f(x)` where `ρ` is a known linear representation of `g`.
3. Use the **transformation metadata** already recorded during pretraining (the `(V, 6)` crop/flip tensors) to pair views with their known transformations.
4. Use the **composable PhC transforms** to annotate which symmetry operation was applied to each view.

The existing codebase is designed to make this extension minimally invasive:

- Loss functions are standalone callables in `src/losses.py` — add the equivariance term there.
- Transformation metadata is already collated into the batch under `record_metadata=True`.
- PhC transforms are individually named classes that can be split into invariant vs. equivariant groups by inspection.
- All encoders share the same `(emb, proj)` forward interface — projector heads can be replaced without touching the backbone.

---

*Built on [I-JEPA](https://github.com/facebookresearch/ijepa) (Meta AI) and LeJEPA (Galilei Group).  Photonic crystal setup follows [Dangovski et al., ICLR 2022 (E-SSL)](https://arxiv.org/abs/2111.00899).*
