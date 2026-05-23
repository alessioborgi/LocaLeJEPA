# Sheaf Neural Networks – PyTorch Implementation

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
