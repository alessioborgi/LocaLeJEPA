# BundLeJEPA

Self-supervised pretraining in the [LeJEPA](https://arxiv.org/abs/2511.08544) family
(invariance loss + distribution-matching regularizer, no teacher/EMA, no predictor),
with **local regularizers**: the regularizer is applied inside k-means neighbourhoods of the
projection space instead of on the whole batch. The proposed variant is
**LeJEPA + local BallReg** — a sketched uniform-ball characteristic-function test on each
neighbourhood — which on ImageNet-1K (ViT-S/8, 100 epochs) lifts LeJEPA's frozen-feature
k-NN / linear top-1 from 24.3 / 40.7 to 34.1 / 49.5 with the same backbone and schedule.

Authors: Alessio Borgi, Kartik Tandon, Pavel Snopov, Inés García-Redondo, Davide D'Ascenzo,
Sebastiano Cultrera di Montesano, Fabrizio Silvestri, Pietro Lio, Francesca Dominici,
Randall Balestriero, Claudio Battiloro.

## Overview

LeJEPA trains an encoder `f` and projector `g` on `V` augmented views of each image with

```
loss = (1 - λ) · L_inv(z) + λ · L_reg(z),      z = g(f(view))  ∈ R^{V × N × D}
```

`L_inv` pulls every view towards the mean of the views of the same image; `L_reg` matches the
distribution of the projections to a fixed target through the Epps–Pulley characteristic-
function statistic along random 1-D sketches (SIGReg: isotropic Gaussian target).

This repository keeps that scaffold ([`src/models/jepa_module.py`](src/models/jepa_module.py),
[`src/models/lejepa_module.py`](src/models/lejepa_module.py)) and swaps the regularizer through
`model.loss_type` ([`src/losses.py`](src/losses.py)):

| `loss_type`     | regularizer                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `sigreg`        | SIGReg, isotropic-Gaussian target (LeJEPA baseline)                          |
| `susreg`        | SUSReg, uniform-on-the-sphere target (SPHERE-JEPA)                          |
| `local_sigreg`  | SIGReg computed independently in `k` k-means neighbourhoods of the batch     |
| `local_ballreg` | **ours**: uniform-ball target in `k` k-means neighbourhoods (centred per cluster) |

Everything else — multi-view augmentation, projector, online linear probe used for checkpoint
selection, AdamW + linear warm-up + cosine schedule — is shared by all variants, so a
comparison between `loss_type` values isolates the regularizer.

Backbones used in the reported experiments: plain timm ViT-S/8 and ViT-L/14
(`configs/model/encoder/timm_vit.yaml`), a hyperbolic (Poincaré-ball head) ViT-S/8
(`hvit_small_p8.yaml`) and a D4-equivariant ResNet-50 (`d4resnet50.yaml`, needs the
`equivariant` extra). Any `nn.Module` mapping `(B, C, H, W) -> (B, embedding_dim)` works.

## Repository structure

```
src/
  train.py                     Hydra + Lightning entry point (pretraining)
  losses.py                    invariance, SIGReg, SUSReg, BallReg, local variants, ...
  models/jepa_module.py        shared LightningModule scaffold (encode, probe, optimizer, schedule)
  models/lejepa_module.py      LeJEPAModule: loss_type ∈ {sigreg, susreg, local_sigreg, local_ballreg}
  models/components/           hyperbolic ViT head, D4-equivariant ResNet (escnn)
  data/                        Imagenette / ImageNet-1K datamodules, multi-view transforms
  eval/                        frozen-backbone loading, feature extraction, k-NN and linear probes
  utils/                       Hydra/Lightning helpers (from lightning-hydra-template)
configs/
  train.yaml                   root config; `experiment=<name>` selects a full experiment
  experiment/lejepa/           the experiments (see configs/experiment/README.md)
  model/, data/, optimizer/,   Hydra config groups
  trainer/, callbacks/, ...
  eval/                        YAML configs for eval_indomain.py
eval_indomain.py               frozen-backbone k-NN + linear-probe evaluation
scripts/
  stage_imagenet1k_hf.py       stage ImageNet-1K from HuggingFace into the expected layout
```

The `experiments` branch adds the baselines (Rectified LpJEPA, Sub-JEPA, SPHERE-JEPA, I-JEPA,
nGPT-style ViTs, photonic-crystal experiments), the ablation configs, the transfer / RankMe /
rotation-angle evaluations, the report-generating scripts, the figures and one SLURM launcher.

## Installation

Requires Python 3.12 (managed automatically by `uv`) and Linux x86_64 or macOS; a CUDA GPU for
anything beyond smoke tests. Install [`uv`](https://docs.astral.sh/uv/) once
(`curl -LsSf https://astral.sh/uv/install.sh | sh`), then:

```bash
git clone https://github.com/alessioborgi/BundLeJEPA.git
cd BundLeJEPA
uv sync
```

`uv sync` creates `.venv/` from the committed `uv.lock` (torch 2.14 with CUDA 13 wheels,
lightning 2.6, timm 1.0). Optional extras:

```bash
uv sync --extra equivariant   # D4-equivariant ResNet-50 (escnn; builds py3nj from source -> needs gfortran + Python headers)
uv sync --extra imagenet      # scripts/stage_imagenet1k_hf.py (HuggingFace datasets)
```

On macOS the `equivariant` extra builds `lie-learn` and `py3nj` from source and needs `gcc`
and `libomp` from Homebrew:

```bash
brew install gcc libomp
export LDFLAGS="-L$(brew --prefix libomp)/lib"
export CPPFLAGS="-I$(brew --prefix libomp)/include"
uv sync --extra equivariant
```

If the venv should not live in the repository (e.g. home-directory quotas), set
`UV_PROJECT_ENVIRONMENT=/path/to/venv` before `uv sync` / `uv run`.

The experiments reported in the paper were run in a conda environment with Python 3.10,
torch 2.8.0+cu128, lightning 2.2.4, timm 1.0.27, escnn 1.0.11 and numpy 1.26; the code is
unchanged, but bit-exact reproduction across these library versions is not guaranteed.

## Quick start

Imagenette (160 px) is downloaded automatically into `data/imagenette/` on first use.

```bash
# LeJEPA baseline: ViT-S/8 + SIGReg, Imagenette, 800 epochs, 1 GPU
uv run python src/train.py experiment=lejepa/vit_imagenette_sigreg logger=csv

# Ours: ViT-S/8 + local BallReg (λ=0.1, k=4), same data / schedule
uv run python src/train.py experiment=lejepa/vit_imagenette_localballreg_l01_k4 logger=csv
```

A 30-second CPU smoke test of the whole pipeline (3 train batches, 1 epoch):

```bash
uv run python src/train.py experiment=lejepa/vit_imagenette_localballreg_l01_k4 \
  trainer=cpu trainer.max_epochs=1 +trainer.limit_train_batches=3 +trainer.limit_val_batches=2 \
  +trainer.limit_test_batches=2 data.batch_size=8 data.num_workers=0 logger=csv
```

Logging defaults to Weights & Biases (`configs/logger/wandb.yaml`, entity `sheaf_hypergraphs`);
pass `logger=csv` for a local CSV log. The W&B entity, project and offline mode are read from
`WANDB_ENTITY`, `WANDB_PROJECT` and `WANDB_OFFLINE` (copy `.env.example` to `.env`), or override
per run with `logger.wandb.entity=<yours>`.

## Training

```bash
uv run python src/train.py experiment=<group>/<name> [hydra overrides ...]
```

An experiment config (`configs/experiment/<group>/<name>.yaml`) fixes the dataset, encoder,
regularizer, hyper-parameters and schedule. Anything can be overridden on the command line,
e.g. `trainer.max_epochs=100 model.lamb=0.05 optimizer.lr=1e-3 data.batch_size=128`.

ImageNet-1K runs read `data/imagenet/imagenet_full_size/061417/{train,val}/<class>/*.JPEG`
(`configs/data/imagenet1k.yaml`); `uv run python scripts/stage_imagenet1k_hf.py --dest data/imagenet`
creates that layout from the gated HuggingFace dataset. `paths.data_dir=/elsewhere` moves the
data root. The principal ImageNet-1K configs are

```bash
uv run python src/train.py experiment=lejepa/vit_in1k_sigreg                 # LeJEPA (SIGReg), 100 ep.
uv run python src/train.py experiment=lejepa/vit_in1k_localballreg_l01_k4    # ours, 100 ep.
```

Multi-GPU (the ViT-L/14 "paper setting" configs use 4 GPUs, one process per GPU):

```bash
uv run python src/train.py experiment=lejepa/vitl14_in1k_localballreg_l01_k4_paper trainer=ddp trainer.devices=4
```

Training resumes from a Lightning checkpoint with `ckpt_path=<path>/last.ckpt`.

## Evaluation

`eval_indomain.py` freezes a pretrained encoder, extracts features with the deterministic
eval transform, and reports **weighted k-NN** (cosine, `exp(sim/T)` votes, T = 0.07, k = 20)
and a **linear probe** on standardized features (SGD, learning rate picked on a held-out slice),
following the SPHERE-JEPA Table 3 protocol:

```bash
uv run python eval_indomain.py --config configs/eval/indomain_in1k_plainvit.yaml            # all backbones in the config
uv run python eval_indomain.py --config configs/eval/indomain_in1k_plainvit.yaml --only vit_sigreg
uv run python eval_indomain.py --config configs/eval/indomain_in1k_plainvit.yaml --merge-only  # rebuild results.{json,csv}
```

Each `backbones:` entry names the Hydra experiment the checkpoint was trained with (the encoder
is rebuilt from that config) and the checkpoint path; per-backbone results are written to
`<output_folder>/per_backbone/<name>.json` and merged into `results.json` / `results.csv`.
`configs/eval/indomain_imagenette.yaml` is the same protocol on Imagenette.

## Configurations

* `configs/train.yaml` composes `data`, `model` (with `model/encoder`), `optimizer`, `trainer`,
  `callbacks`, `logger`, `paths`, `extras`; `experiment=...` overrides a full set of them.
* `configs/model/lejepa.yaml` documents every `LeJEPAModule` hyper-parameter (`lamb`,
  `loss_type`, `local_sigreg_num_clusters` = k, `local_ballreg_*`, the dynamic-k schedule, the
  online probe).
* `configs/experiment/README.md` lists every experiment config, which one is the LeJEPA baseline
  and which one is ours for each backbone / dataset, and what differs between them.
* The experiment configs contain everything that was passed to the runs, including the per-step
  batch (64 × `accumulate_grad_batches: 4` = 256 effective) and the gradient clipping used by the
  local regularizers, so `experiment=<name>` alone reproduces a run.

## Outputs

* Hydra run directory: `logs/train/runs/<date>_<time>/` (config dump, `train.log`, logger files).
* Checkpoints: `<run dir>/checkpoints/best.ckpt` (best `val/probe_acc` of the online linear
  probe) and `last.ckpt`; `callbacks.model_checkpoint.dirpath=checkpoints/<name>` pins them to a
  stable path.
* Metrics: W&B (default) or `<run dir>/csv/` with `logger=csv`.
* Evaluation: `logs/eval_indomain_*/` (`results.json`, `results.csv`, `per_backbone/`, feature cache).

## Reproducing results

The full experimental infrastructure lives on the `experiments` branch:

```bash
git checkout experiments
uv sync --all-extras
```

It contains all ImageNet-1K / Imagenette configs (LeJEPA, ours, ablations over λ and k, dynamic k,
local SIGReg, Rectified LpJEPA, SUSReg/SubReg, three backbones), the ViT-L/14 paper-setting
configs, the evaluations behind every reported table (in-domain k-NN/linear, 7-dataset transfer,
RankMe, Yoda rotation-angle probe), the scripts that regenerate the result tables in `results/`,
the plotting scripts and the SLURM launcher `scripts/slurm_submit.sh`. See `EXPERIMENTS.md`
there. In short:

| setting | config | command |
|---|---|---|
| LeJEPA (SIGReg), ViT-S/8, IN1K 100 ep. | `lejepa/vit_in1k_sigreg` | `uv run python src/train.py experiment=lejepa/vit_in1k_sigreg` |
| **Ours**, local BallReg λ0.1 k4, ViT-S/8, IN1K 100 ep. | `lejepa/vit_in1k_localballreg_l01_k4` | `uv run python src/train.py experiment=lejepa/vit_in1k_localballreg_l01_k4` |
| LeJEPA paper setting, ViT-L/14, 2×224 + 8×98 views | `lejepa/vitl14_in1k_sigreg_paper` | `... experiment=lejepa/vitl14_in1k_sigreg_paper trainer=ddp trainer.devices=4` |
| Ours, paper setting | `lejepa/vitl14_in1k_localballreg_l01_k4_paper` | `... experiment=lejepa/vitl14_in1k_localballreg_l01_k4_paper trainer=ddp trainer.devices=4` |

## Reproducibility notes

* **Seeds.** `seed` is `null` by default (no seeding); pass `seed=<int>` to call
  `lightning.seed_everything`. The sketch directions of SIGReg/BallReg, k-means initialisation
  and data augmentation are random per step, and `trainer.deterministic` is `False`, so runs
  with the same seed are close but not bit-identical on GPU.
* **Data.** Imagenette downloads itself. ImageNet-1K must be staged (see Training). All datasets
  are read from `paths.data_dir` (default `<repo>/data/`).
* **Effective batch.** The single-GPU ImageNet-1K / Imagenette runs use `data.batch_size: 64`
  with `trainer.accumulate_grad_batches: 4`. For the local regularizers the k-means clustering
  sees the per-step batch (64 samples), so changing `batch_size` changes the method, not only
  the memory footprint.
* **Precision.** `trainer=gpu` / `trainer=ddp` use `bf16-mixed`; the losses upcast to fp32 internally.
* **Hardware.** The single-GPU configs were run on one A100/H200 (per-step batch 64 × 4 views
  at 128 px, bf16). The ViT-L/14 paper-setting configs use 4 GPUs (`trainer=ddp`, 64 samples per
  GPU); their checkpoints are ~3.7 GB each.
* **Checkpoint selection.** Reported "final" numbers use `last.ckpt` (epoch 99 of 100);
  `best.ckpt` is the online-probe-selected checkpoint and is reported alongside.

## Citation and acknowledgements

Built on [LeJEPA](https://github.com/galilai-group/lejepa) (Balestriero & LeCun),
[lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template), the
[hyp_metric](https://github.com/htdt/hyp_metric) hyperbolic ViT head and the
[equivision](https://github.com/dmklee/equivision) D4-ResNet; third-party notices in `src/NOTICE`. Licensed under the
terms in `LICENSE`.
