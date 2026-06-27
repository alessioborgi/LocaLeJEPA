# BundLeJEPA – Two Symmetries, One Bundle: Invariance and Equivariance Packaged Together

A implementation of the paper "BundLeJEPA: .."

**Copyright © 2026, _BundLeJEPA_.**
Authors: Alessio Borgi, Paul Snopov, Kartik Tandon, Ines Garcia, Sebastiano Cultrera di Montesano, Fabrizio Silvestri, Pietro Liò, Randall Balestriero, Claudio Battiloro.

---

## What is BundLeJEPA?

**BundLeJEPA** (Bundle LeJEPA) is a self-supervised representation learning framework that extends **LeJEPA** with fibre-bundle geometry.  Instead of learning a single invariant representation, it learns a section of a principal bundle — an embedding that transforms predictably (equivariantly) under the symmetry group of the data.

The repository currently implements:

- **LeJEPA** — invariance loss + Sketched Isotropic Gaussian Regularisation (SIGReg), with ViT, ResNet-18/50, and a custom photonic crystal (PhC) CNN backbone
- **I-JEPA** — masked-patch prediction in latent space, with ViT and PhC CNN
- **SimCLR** — NT-Xent contrastive baseline with ResNet-50
- **Supervised PhC baseline** — CNN + DOS regression, no pretraining
- **Linear evaluation** — 11 downstream vision datasets + VOC07+12 Faster R-CNN detection
- **DOS regression** — frozen and fine-tuned evaluation on PhC datasets

---

## Installation

```bash
pip install torch torchvision timm
pip install scikit-learn pyyaml numpy pillow

# Optional
pip install 'git+https://github.com/facebookresearch/detectron2.git'  # VOC detection
pip install submitit                                                    # SLURM submission
pip install wandb                                                       # experiment tracking
```

**Python 3.10+** and **PyTorch 2.0+** required.

---

## Repository structure

```
BundLeJEPA/
├── main.py / main_distributed.py   # single-machine and SLURM entry points
├── eval_linear.py / eval_voc.py    # downstream evaluation entry points
├── configs/                        # YAML experiment configs
└── src/
    ├── losses.py                   # invariance_loss, sigreg_loss, nt_xent_loss
    ├── transforms.py               # DINO-style multi-crop augmentation
    ├── lejepa/                     # LeJEPA training loop and encoders
    ├── ijepa/                      # I-JEPA training loop, masks, and ViT
    ├── phc/                        # PhC encoder, domain transforms, DOS eval
    ├── baselines/                  # SimCLR and supervised PhC baseline
    ├── datasets/                   # ImageNet, STL10, Imagenette, PhC loaders
    ├── eval/                       # linear probe, feature extraction, VOC detection
    └── utils/                      # distributed, logging, schedulers
```

---

## Running experiments

```bash
# LeJEPA — single machine
python main.py --fname configs/lejepa_rn50_in100_ep500.yaml \
               --devices cuda:0 cuda:1

# LeJEPA — SLURM cluster
python main_distributed.py --fname configs/lejepa_rn50_in100_ep500.yaml \
    --nodes 2 --tasks-per-node 8 --partition gpu

# I-JEPA on PhC data
python main.py --fname configs/ijepa_phc_blob_ep250.yaml

# SimCLR baseline
python main.py --fname configs/simclr_rn50_in100_ep500.yaml

# Linear evaluation (11 datasets)
# → edit configs/eval_linear.yaml first (checkpoint, datasets_root)
python eval_linear.py --config configs/eval_linear.yaml

# VOC07+12 detection
# → edit configs/eval_voc.yaml first (voc_root, checkpoint)
python eval_voc.py --config configs/eval_voc.yaml
```

---

## Conformal-Autoencoder dimension detection (bobblehead)

This evaluation reproduces the bobblehead experiment of Kevrekidis et al.,
*"Thinner Latent Spaces: Detecting Dimension and Imposing Invariance with
Conformal Autoencoders"* (arXiv:2408.16138, §3.3) — but replaces the paper's
PCA-60 front-end with a **frozen pretrained JEPA backbone** as the fixed feature
extractor:

```
images --(frozen backbone)--> 512-d features --(Conformal Autoencoder)--> latent
```

The data set (`data/yoda_dog_images_and_labels.zip`) holds 8097 RGB frames of two
figures (Yoda + bulldog) rotating at independent fixed frequencies, so the image
manifold is a **2-torus**.  The Conformal Autoencoder over-specifies its latent
layer and penalises the orthogonality of latent input-gradients (Algorithm 1 /
Eq. 6); redundant coordinates collapse, so the number of *active* latent
coordinates reveals the intrinsic dimension.  `labels.npy` (the two rotation
angles per frame) is used only for an optional circular-recovery sanity check —
the dimension-detection task itself is unsupervised.

```bash
# Compares three frozen baselines (SphereJEPA+SUSReg, nViT+SIGReg, nViT+SubReg).
# Edit configs/eval_conformal.yaml for backbones / CAE hyper-parameters.
python eval_conformal.py --config configs/eval_conformal.yaml
```

Outputs (under `logs/eval_conformal/`): `results.json` / `results.csv` with the
inferred dimension and angle-recovery metrics per backbone, plus per-backbone
diagnostic plots (latent-gradient-norm trajectories, latent-variance bars, and
the recovered torus scatter).  Code: `src/data/bobblehead_datamodule.py`,
`src/eval/extract.py`, `src/eval/conformal_autoencoder.py`.

---

## Photonic Crystal (PhC) Pipeline

The PhC pipeline runs in two sequential stages: pre-training (self-supervised) followed by fine-tuning (DOS regression).

### 1. Download the dataset (once, on the login node)

```bash
python scripts/download_phc.py --output-dir data/phc
```

This downloads five HDF5 files (~few GB total) from Dropbox into `data/phc/`.

### 2. Pre-train

```bash
# Blob
sbatch src/slurm_files/slurm_harvard_phc_pretrain.sh

# GPM
sbatch --export=ALL,DATASET=gpm src/slurm_files/slurm_harvard_phc_pretrain.sh
```

Checkpoint is saved to `logs/pretrain/phc_{blob|gpm}_ep250/checkpoints/last.ckpt`.

### 3. Fine-tune (after pre-training finishes)

```bash
# Blob
sbatch src/slurm_files/slurm_harvard_phc_finetune.sh

# GPM
sbatch --export=ALL,DATASET=gpm src/slurm_files/slurm_harvard_phc_finetune.sh
```

The fine-tune script automatically picks up the checkpoint from step 2.  To use a custom checkpoint path:

```bash
sbatch --export=ALL,DATASET=blob,CKPT_PATH=logs/pretrain/phc_blob_ep250/checkpoints/last.ckpt \
       src/slurm_files/slurm_harvard_phc_finetune.sh
```

### Submit all four jobs at once (with SLURM dependencies)

Pre-training and fine-tuning for both datasets can be submitted together; SLURM holds each fine-tune job until its pre-train job succeeds.

```bash
# Blob
BLOB_PRETRAIN=$(sbatch --parsable src/slurm_files/slurm_harvard_phc_pretrain.sh)
sbatch --dependency=afterok:$BLOB_PRETRAIN src/slurm_files/slurm_harvard_phc_finetune.sh

# GPM
GPM_PRETRAIN=$(sbatch --parsable --export=ALL,DATASET=gpm src/slurm_files/slurm_harvard_phc_pretrain.sh)
sbatch --dependency=afterok:$GPM_PRETRAIN --export=ALL,DATASET=gpm src/slurm_files/slurm_harvard_phc_finetune.sh
```

---

## Configuration

All experiments are driven by YAML files.  The `method` key selects the training loop (`lejepa`, `ijepa`, `simclr`, `phc_supervised`).  A minimal LeJEPA config:

```yaml
method: lejepa

backbone:
  name:     resnet50
  emb_dim:  2048
  proj_dim: 128

data:
  dataset:          imagenet100
  root_path:        /path/to/imagenet
  batch_size:       256
  num_global_crops: 2
  num_local_crops:  6

lejepa:
  lamb: 0.02        # SIGReg weight

optimization:
  epochs:       500
  lr:           5.0e-4
  weight_decay: 5.0e-4
  use_bfloat16: true

logging:
  folder:    logs/run_name/
  use_wandb: false
```

---

*Built on [I-JEPA](https://github.com/facebookresearch/ijepa) (Meta AI) and LeJEPA (Galilei Group).  Photonic crystal setup follows [Dangovski et al., ICLR 2022](https://arxiv.org/abs/2111.00899).*
