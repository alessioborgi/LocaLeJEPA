# LocaLeJEPA

Self-supervised pretraining in the [LeJEPA](https://arxiv.org/abs/2511.08544) family
(invariance loss + distribution-matching regularizer, no teacher/EMA, no predictor),
with **local regularizers**: the regularizer is applied inside k-means neighbourhoods of the
projection space instead of on the whole batch. The proposed variant is
**LocaLe-Reg** — a sketched uniform-ball characteristic-function test on each centred
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

This repository keeps that scaffold ([`src/models/lejepa_module.py`](src/models/lejepa_module.py))
and swaps the regularizer through the `model/reg_loss` config group
([`configs/model/reg_loss/`](configs/model/reg_loss/), functions in [`src/losses.py`](src/losses.py)):

| `model/reg_loss` | regularizer                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `sigreg`         | SIGReg, isotropic-Gaussian target (LeJEPA baseline)                          |
| `susreg`         | SUSReg, uniform-on-the-sphere target (SPHERE-JEPA)                          |
| `uniform_reg`    | uniform-ball target (`ballreg_loss`)                                          |
| `locale_sigreg`  | SIGReg computed independently in `k` k-means neighbourhoods of the batch     |
| `localereg`      | **LocaLe-Reg (ours)**: uniform-ball target in `k` k-means neighbourhoods (centred per cluster) |

The `locale_*` configs compose `local.yaml` (`local_loss`: k-means over the batch, then a test
per neighbourhood) with the global test placed at `loss`, so the test's parameters are set once
and are the same inside every neighbourhood. The neighbourhood count is set directly
(`model.reg_loss.num_clusters=4`); the inner test's parameters live one level down, under
`model.reg_loss.loss.*` — `model.reg_loss.num_slices` is **not** a valid override on a local
config, because `local_loss` has no such parameter.

The inner test inherits `uniform_reg.yaml`: `num_slices: 256`, `t_max: 3.0`, `scale_grid: true`.
Note `scale_grid` is a parameter of the uniform-ball test only — `sigreg_loss` does not take it,
so it does not apply to `locale_sigreg`.

**One pair does not share a statistic resolution.** `vitl14_in1k_sigreg_paper` sets
`num_slices: 1024` over `t_max: 5.0`, while its local counterpart uses the inherited 256 / 3.0.
It is the only config in the repo that sets 1024, and it is the *baseline* arm. The 100-epoch
ImageNet-1K comparison is unaffected — both arms use 256 / 3.0 there.

Everything else — multi-view augmentation, projector, online linear probe used for checkpoint
selection, AdamW + linear warm-up + cosine schedule — is shared by all variants, so a
comparison between `reg_loss` choices isolates the regularizer.

Backbones used in the reported experiments: plain timm ViT-S/8 and ViT-L/14
(`configs/model/encoder/timm_vit.yaml`), a hyperbolic (Poincaré-ball head) ViT-S/8
(`hvit_small_p8.yaml`) and a D4-equivariant ResNet-50 (`d4resnet50.yaml`, needs the
`equivariant` extra). Any `nn.Module` mapping `(B, C, H, W) -> (B, embedding_dim)` works.

## Repository structure

```
src/
  train.py                     Hydra + Lightning entry point (pretraining)
  src/eval_indomain.py             frozen-backbone k-NN + linear-probe evaluation
  losses.py                    invariance, SIGReg, SUSReg, uniform-ball test, local_loss (k-means + any test), ...
  models/lejepa_module.py      LeJEPAModule: encode, reg_loss + invariance, online probe, optimizer, schedule
  models/components/           hyperbolic ViT head, D4-equivariant ResNet (escnn)
  data/                        Imagenette / ImageNet-1K datamodules, multi-view transforms
  eval/                        frozen-backbone loading, feature extraction, crash-safe IO, and the
                               probes: k-NN, linear, transfer, RankMe, angle (Yoda), patch-PCA,
                               conformal autoencoder
  utils/                       Hydra/Lightning helpers (from lightning-hydra-template)
configs/
  train.yaml                   root config; `experiment=<name>` selects a full experiment
  experiment/lejepa/           the experiments (see configs/experiment/README.md)
  model/, data/, optimizer/,   Hydra config groups
  model/reg_loss/              sigreg, susreg, uniform_reg; local -> locale_sigreg, localereg
  trainer/, callbacks/, ...
  eval/                        YAML configs for src/eval_indomain.py
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
git clone https://github.com/alessioborgi/LocaLeJEPA.git
cd LocaLeJEPA
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

# Ours: ViT-S/8 + LocaLe-Reg (λ=0.1, k=4), same data / schedule
uv run python src/train.py experiment=lejepa/vit_imagenette_localballreg_l01_k4 logger=csv
```

A 30-second CPU smoke test of the whole pipeline (3 train batches, 1 epoch):

```bash
uv run python src/train.py experiment=lejepa/vit_imagenette_localballreg_l01_k4 \
  trainer=cpu trainer.max_epochs=1 +trainer.limit_train_batches=3 +trainer.limit_val_batches=2 \
  +trainer.limit_test_batches=2 data.batch_size=8 data.num_workers=0 logger=csv
```

It checks that the pipeline runs, and nothing else. The `val/probe_acc` and `test/probe_acc` it
prints are **not metrics**: `limit_val_batches=2` at `batch_size=8` is 16 images, and because the
val loader is unshuffled over a class-sorted `ImageFolder` all 16 come from the first class of
ten. Two smoke runs differing 0.359 vs 0.031 look like a dramatic result and mean nothing.
(`val/probe_acc` is also the checkpoint-selection monitor, so the `trainer.test` table it prints
at the end carries the same degenerate number.)

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

`src/eval_indomain.py` freezes a pretrained encoder, extracts features with the deterministic
eval transform, and reports **weighted k-NN** (cosine, `exp(sim/T)` votes, T = 0.07, k = 20)
and a **linear probe** on standardized features (SGD, learning rate picked on a held-out slice),
following the SPHERE-JEPA Table 3 protocol:

```bash
uv run python src/eval_indomain.py --config configs/eval/indomain_in1k_plainvit.yaml            # all backbones in the config
uv run python src/eval_indomain.py --config configs/eval/indomain_in1k_plainvit.yaml --only vit_sigreg
uv run python src/eval_indomain.py --config configs/eval/indomain_in1k_plainvit.yaml --merge-only  # rebuild results.{json,csv}
```

Each `backbones:` entry names the Hydra experiment the checkpoint was trained with (the encoder
is rebuilt from that config) and the checkpoint path; per-backbone results are written to
`<output_folder>/per_backbone/<name>.json` and merged into `results.json` / `results.csv`.
`configs/eval/indomain_imagenette.yaml` is the same protocol on Imagenette.

## Configurations

* `configs/train.yaml` composes `data`, `model` (with `model/encoder` and `model/reg_loss`),
  `optimizer`, `trainer`, `callbacks`, `logger`, `paths`, `extras`; `experiment=...` overrides a
  full set of them.
* `configs/model/lejepa.yaml` documents every `LeJEPAModule` hyper-parameter (`lamb`, the
  invariance loss, the online probe); `configs/model/reg_loss/<name>.yaml` the regularizer's
  (`local.yaml`: `num_clusters` = k and the k-means; `uniform_reg.yaml`: `radius`, `center`,
  `scale_grid`, ...); `callbacks=dynamic_num_clusters` schedules k over epochs.
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
| **Ours**, LocaLe-Reg λ0.1 k4, ViT-S/8, IN1K 100 ep. | `lejepa/vit_in1k_localballreg_l01_k4` | `uv run python src/train.py experiment=lejepa/vit_in1k_localballreg_l01_k4` |
| LeJEPA paper setting, ViT-L/14, 2×224 + 8×98 views | `lejepa/vitl14_in1k_sigreg_paper` | `... experiment=lejepa/vitl14_in1k_sigreg_paper trainer=ddp trainer.devices=4` |
| Ours, paper setting | `lejepa/vitl14_in1k_localballreg_l01_k4_paper` | `... experiment=lejepa/vitl14_in1k_localballreg_l01_k4_paper trainer=ddp trainer.devices=4` |

## Reproducibility notes

* **Seeds.** `seed` is `null` by default (no seeding); pass `seed=<int>` to call
  `lightning.seed_everything`. The sketch directions of SIGReg/LocaLe-Reg, k-means initialisation
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
