<div align="center">

# LocaLeJEPA: Local Regularizers for Joint-Embedding Predictive Architectures

[Kartik Tandon](https://scholar.google.com/citations?user=Jxj7usMAAAAJ)<sup>1,‡</sup>,
[Alessio Borgi](https://scholar.google.com/citations?hl=it&user=Ds4ktdkAAAAJ)<sup>2,‡</sup>,
[Pavel Snopov](https://scholar.google.com/citations?user=CAmErvEAAAAJ)<sup>3</sup>,
[Inés García-Redondo](https://scholar.google.com/citations?user=QJugIzgAAAAJ)<sup>4</sup>,
[Davide D'Ascenzo](https://scholar.google.com/citations?user=E_rNRYwAAAAJ)<sup>5</sup>,
[Sebastiano Cultrera di Montesano](https://scholar.google.com/citations?user=NFX7NNYAAAAJ)<sup>6</sup>,
[Fabrizio Silvestri](https://scholar.google.com/citations?user=pi985dQAAAAJ)<sup>2</sup>,
[Pietro Liò](https://scholar.google.com/citations?user=4YhNJBEAAAAJ)<sup>7</sup>,
[Francesca Dominici](https://scholar.google.com/citations?user=RSwElNMAAAAJ)<sup>8</sup>,
[Randall Balestriero](https://scholar.google.com/citations?user=S1x_xqcAAAAJ)<sup>9</sup>,
[Claudio Battiloro](https://scholar.google.com/citations?user=_J11o_IAAAAJ)<sup>8</sup>

<sup>1</sup>University of Pennsylvania · <sup>2</sup>Sapienza University of Rome · <sup>3</sup>University of Texas Rio Grande Valley ·
<sup>4</sup>University of Fribourg · <sup>5</sup>University of Milan · <sup>6</sup>Broad Institute ·
<sup>7</sup>University of Cambridge · <sup>8</sup>Harvard University · <sup>9</sup>Brown University

<p><sup>‡</sup>Equal contribution</p>

</div>

<p align="center">
  <a href="https://arxiv.org/abs/2511.08544">
    <img src="https://img.shields.io/badge/Builds%20on-LeJEPA-orange" alt="LeJEPA" style="height: 25px; margin-right: 5px;">
  </a>
  <a href="https://github.com/alessioborgi/LocaLeJEPA">
    <img src="https://img.shields.io/badge/Code-GitHub-black?logo=github" alt="Code" style="height: 25px; margin-right: 5px;">
  </a>
  <a href="https://www.python.org/downloads/release/python-3120/">
    <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12" style="height: 25px; margin-right: 5px;">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.14-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.14" style="height: 25px; margin-right: 5px;">
  </a>
  <a href="https://lightning.ai/">
    <img src="https://img.shields.io/badge/Lightning-2.6-792EE5?logo=lightning&logoColor=white" alt="Lightning 2.6" style="height: 25px; margin-right: 5px;">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License" style="height: 25px;">
  </a>
  <!-- Add once available:
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-orange" alt="arXiv" style="height: 25px;"></a>
  <a href="https://openreview.net/forum?id=XXXX"><img src="https://img.shields.io/badge/OpenReview-Paper-8C1B13" alt="OpenReview" style="height: 25px;"></a>
  -->
</p>

---

<div align="justify">

## Overview 🚀

> **TL;DR.** LocaLeJEPA is a self-supervised JEPA framework that matches the embedding distribution
> locally. Its regularizer, LocaLeReg, partitions each batch into k-means neighborhoods, re-centers each
> one, and matches it to a uniform ball, so the objective respects the intrinsic geometry of the data and
> extends to non-Euclidean embedding spaces. It yields markedly stronger frozen features on ImageNet-1K
> across Euclidean, hyperbolic and rotation-equivariant backbones: 6.6 to 9.4 points higher linear-probe
> accuracy and nearly 10 points higher k-NN accuracy than global distribution matching, with consistent
> gains of about 6 points on downstream transfer.

> **Quick start.** [`MINIMAL.md`](MINIMAL.md) presents the whole method on a single page, from the
> formulation to a compact PyTorch Lightning implementation, and trains a ViT-S/8 with LocaLeReg on
> Imagenette on one GPU:
>
> ```bash
> uv sync --locked
> uv run python src/train.py experiment=minimal
> ```

Joint-Embedding Predictive Architectures (JEPAs) are an appealing approach to self-supervised learning,
but they are prone to representational collapse, in which all inputs are mapped to the same embedding.
[LeJEPA](https://arxiv.org/abs/2511.08544) resolves this with a theoretically grounded distribution-matching
objective: the embeddings of a batch are driven towards an isotropic Gaussian, enforced through a
statistical test along random one-dimensional projections. The method requires neither a teacher network
nor a stop-gradient, and has a single trade-off weight, λ.

This objective is, however, tied to Euclidean assumptions. Its target is a single global Gaussian, which
presumes that the embedding space is Euclidean and that the data fill it as one isotropic region. Neither
holds in general. Natural images are organized into distinct regions and lower-dimensional structures, and
many encoders impose a non-Euclidean embedding geometry by design, such as hyperbolic ViTs or
rotation-equivariant CNNs. A global Euclidean target does not reflect these geometric priors.

LocaLeJEPA is motivated by a manifold-theoretic reformulation of LeJEPA's entropy-maximization property. A
manifold is locally Euclidean even when it is not Euclidean globally, so the distribution-matching
objective is applied locally rather than to the batch as a whole. Its regularizer, LocaLeReg, implements
this in a stable and scalable way: at each training step, the batch is partitioned into K neighborhoods by
k-means, and each neighborhood is re-centered on its own mean and matched to a uniform distribution on a
ball. The position of each neighborhood is left unconstrained, and only its internal spread is
regularized. All other components of LeJEPA are unchanged.

<p align="center"><img src="assets/LocaLeJEPA_Main_Figure.svg" width="100%" alt="LocaLeJEPA main figure"/></p>

On ImageNet-1K, LocaLeJEPA outperforms LeJEPA with hyperbolic ViT-S/8, D4-equivariant ResNet-50 and
Euclidean ViT-S/8 backbones, in linear and k-NN classification and on downstream transfer. For a frozen
ViT-S/8 pretrained for 100 epochs, linear-probe accuracy increases from 40.7% to 49.6% and k-NN accuracy
from 24.3% to 34.1%. These results use λ = 0.1 and K = 4 for LocaLeJEPA and λ = 0.02 for LeJEPA, and each
comes from a single pretraining run.

The two methods are not trained with identical recipes, and part of each difference may therefore be
attributable to optimization. LeJEPA runs use a learning rate of 2e-3 with one warm-up epoch (1e-3 for the
D4-ResNet-50). Local matching collapses under this setting, so LocaLeJEPA runs use a learning rate of 1e-3,
ten warm-up epochs and gradient clipping at 1.0. The `*_in1k_sigreg_localrecipe` configurations train
LeJEPA with the LocaLeJEPA recipe and isolate this effect.

LocaLeJEPA features also encode orientation more accurately. On a video of two independently rotating
figurines, whose frames lie on a torus, a linear probe on LocaLeJEPA features recovers the two rotation
angles with 23% and 17% lower error than on LeJEPA features (Imagenette pretraining, K = 2, the best
setting evaluated). After ImageNet-1K pretraining the improvement is smaller: 3.68° against 3.94° for one
figurine, and none for the other.

The regularizer can also be added to an existing training loop in
[five lines](#use-the-regularizer-in-your-own-code), and the encoders are described
[under the hood](#under-the-hood).

## Try it

**1. Install.** You need Python 3.12; real training needs a CUDA GPU.

```bash
git clone https://github.com/alessioborgi/LocaLeJEPA.git
cd LocaLeJEPA
uv sync                     # exact versions from uv.lock: torch 2.14, lightning 2.6, timm 1.0
```

* **No [uv](https://docs.astral.sh/uv/getting-started/installation/)?** `conda env create -f environment-aicr.yml`,
  activate it, and drop `uv run` from every command below.
* **D4-equivariant ResNet-50 only:** it needs escnn, which pip cannot install cleanly on Python 3.12. On
  Linux, run `PY=$(which python) bash scripts/aicr/install_escnn.sh` inside the environment.
* **Logging:** runs go to Weights & Biases. Run `wandb login` once, or add `logger=csv` to log locally.

**2. Check that everything runs** (30 seconds on a CPU; downloads Imagenette on first use):

```bash
uv run python src/train.py experiment=lejepa/vit_imagenette_localballreg_l01_k4 \
  trainer=cpu trainer.max_epochs=1 +trainer.limit_train_batches=3 +trainer.limit_val_batches=2 \
  +trainer.limit_test_batches=2 data.batch_size=8 data.num_workers=0 logger=csv
```

This only checks the plumbing. The accuracy it prints is **not a metric**: its 16 validation images all
come from one class.

**3. Train LeJEPA and LocaLeJEPA side by side** on Imagenette (one GPU each, 800 epochs):

```bash
uv run python src/train.py experiment=lejepa/vit_imagenette_sigreg logger=csv              # LeJEPA
uv run python src/train.py experiment=lejepa/vit_imagenette_localballreg_l01_k4 logger=csv  # LocaLeJEPA
```

Config names spell out the setting (`l01_k4`: λ = 0.1, K = 4). To follow a run, watch `val/probe_acc`, an
online linear probe; with `logger=csv` it is in `logs/train/runs/<date>_<time>/csv/version_0/metrics.csv`.

## Reproduce the paper

Every number comes from four steps: get the data, pretrain, evaluate the frozen encoder, read off the tables.

### Step 1: get the data

Everything lives under `data/`.

| Dataset | Used for | How |
|---|---|---|
| Imagenette | rotation study, quick experiments | downloaded automatically |
| ImageNet-1K | main results | gated on Hugging Face: accept its terms, then the commands below |
| 7 transfer datasets | transfer probing | the commands below |
| Bobblehead video | rotation-angle probe | put `yoda_dog_images_and_labels.zip` in `data/` (see below) |

```bash
huggingface-cli login
uv run python scripts/stage_imagenet1k_hf.py --dest data/imagenet
uv run python scripts/verify_imagenet_staging.py --root data/imagenet/imagenet_full_size/061417
uv run python scripts/download_transfer_datasets.py --root data/transfer_datasets --verify
```

The bobblehead video is the sequence of Lederman & Talmon (2014), distributed with
[Kevrekidis et al.](https://arxiv.org/abs/2408.16138). If your data lives elsewhere, symlink it into
`data/`: the evaluation configs read these folders by name.

### Step 2: pretrain

Each run is one file in [`configs/experiment/`](configs/experiment/) with every hyperparameter written down.
Save its checkpoint under the config's name with `/` replaced by `_`; that is where the evaluation configs
look for it.

```bash
EXP=lejepa/vit_in1k_localballreg_l01_k4      # ViT-S/8, ImageNet-1K, 100 epochs, LocaLeJEPA
uv run python src/train.py experiment=$EXP paths.ckpt_dir=checkpoints/${EXP//\//_}
```

`paths.ckpt_dir` holds `best.ckpt`, `last.ckpt` and `on_exception.ckpt`; add
`ckpt_path=checkpoints/<name>/last.ckpt` to the same command to resume. An ImageNet-1K run takes days on
one GPU. [`configs/experiment/README.md`](configs/experiment/README.md) lists every config and what differs
between each LeJEPA baseline and its LocaLeJEPA counterpart.

> [!IMPORTANT]
> The configs now decay the learning rate with a cosine. The paper's LocaLeJEPA runs and its D4-ResNet-50
> LeJEPA run held it constant after warm-up; its other LeJEPA runs decayed to half the peak. To match the
> paper exactly, add `model.lr_eta_min_ratio=1.0` to any LocaLeJEPA config, and
> `optimizer.lr=1e-3 model.lr_eta_min_ratio=1.0` to `lejepa/d4rn50_in1k_sigreg` (whose config now uses 2e-3).

<details>
<summary><b>On a SLURM cluster</b></summary>

[`scripts/slurm_submit.sh`](scripts/slurm_submit.sh) runs the same command under the same checkpoint name.
Set the partition, account and time limit in its `#SBATCH` header once, then submit from the repository
root:

```bash
mkdir -p logs/slurm
echo 'eval "$(conda shell.bash hook)"; conda activate localejepa' > env.sh   # with uv: skip ENV_SETUP
sbatch --export=ALL,EXPERIMENT=lejepa/vit_in1k_localballreg_l01_k4,REQUEUE=1,ENV_SETUP=env.sh \
  scripts/slurm_submit.sh
```

With `REQUEUE=1` the job queues its own successor, so a run longer than your time limit chains itself to the
end, and a run that crashes without progress stops. All configs need one GPU except ViT-L/14, which needs
eight on one node:

```bash
sbatch --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=16 --mem=512G \
  --export=ALL,EXPERIMENT=lejepa/vitl14_in1k_localballreg_paper,GPUS=8,REQUEUE=1,ENV_SETUP=env.sh \
  scripts/slurm_submit.sh
```

The script's header documents its other options, including `CMD=...` to run any command, such as an
evaluation.

</details>

### Step 3: evaluate

Every evaluation freezes the encoder and, except patch-PCA (which reads the patch tokens), uses its 512-d
output before the projector. A config lists the
checkpoints to evaluate; the script writes one JSON per checkpoint and merges them into `results.csv`.
Add `--only <name>` to evaluate one entry.

| What it measures | Script | Configs |
|---|---|---|
| ImageNet-1K k-NN and linear probe | `src/eval_indomain.py` | `indomain_in1k_plainvit.yaml` (ViT-S/8), `indomain_in1k_ep100.yaml` (other encoders) |
| Linear probing on 7 transfer datasets | `src/eval_transfer.py` | `transfer_in1k_ep100.yaml` |
| Effective rank of the features (RankMe) | `src/eval_rankme.py` | `rankme_in1k_ep100.yaml` |
| Rotation angles on the bobblehead video | `src/eval_angle_probe.py`, then `src/eval_angle_probe_multiseed.py` | one file per checkpoint in `yoda/` |
| Patch-PCA maps of object parts | `src/visualize_patch_pca.py` | your copy of `patch_pca_template.yaml` |

All configs are in [`configs/eval/`](configs/eval/). For example:

```bash
uv run python src/eval_indomain.py --config configs/eval/indomain_in1k_plainvit.yaml \
  --only vit_localballreg_l01_k4
uv run python src/eval_transfer.py --config configs/eval/transfer_in1k_ep100.yaml
uv run python src/eval_rankme.py --config configs/eval/rankme_in1k_ep100.yaml
uv run python src/eval_angle_probe.py --config configs/eval/yoda/vit_localballreg_l01_k4_in1k.yaml
uv run python src/eval_angle_probe_multiseed.py --config configs/eval/yoda/vit_localballreg_l01_k4_in1k.yaml \
  --n-seeds 10
```

The first command reproduces the ViT-S/8 LocaLeJEPA result. It writes `linear.top1` ≈ 49.5 to
`logs/eval_indomain_in1k_plainvit/per_backbone/vit_localballreg_l01_k4.json` with one probe seed (paper:
49.6, five seeds). The probe tunes its learning rate on a held-out slice of the training set, never on the
validation images.

### Step 4: read off the tables

Each evaluation's `results.csv` holds one table row per checkpoint. Two scripts format the appendix tables:

| Script | Builds |
|---|---|
| `uv run python scripts/make_transfer_results.py` | the transfer table and its 7-dataset average |
| `uv run python scripts/make_rankme_results.py` | the RankMe table |

The ImageNet-1K numbers need no formatter: read `linear_top1` and `knn20_top1` straight from `results.csv`.

<details>
<summary><b>Seeds, ablations and the rotation study: campaigns</b></summary>

Studies with many runs are *campaigns*: a YAML in [`configs/campaign/`](configs/campaign/) lists settings
and seeds, and each (setting, seed) pair becomes one run with a fixed name. Expand a campaign into a numbered
list, then train it as one SLURM job array:

```bash
uv run python scripts/aicr/campaign.py expand configs/campaign/torus.yaml --out torus.json  # lists 11 runs
uv run python scripts/aicr/campaign.py show torus.json --index 3                            # what run 3 is
sbatch --array=0-10 --export=ALL,GRID=torus.json,REQUEUE=1,ENV_SETUP=env.sh scripts/slurm_submit.sh
```

Campaign runs are evaluated with the matching `configs/eval/*_campaign.yaml`, for example
`uv run python src/eval_angle_probe.py --config configs/eval/yoda/torus_campaign.yaml`, and tabulated by:

| Script | Builds |
|---|---|
| `scripts/make_equivariance_report.py` | the rotation-study tables |
| `scripts/make_imagenet1k_results.py` | the multi-seed ImageNet-1K table |
| `scripts/make_ablation_tables.py` | the Imagenette λ × K × batch grid |
| `scripts/audit_tables.py` | re-derives every number those tables quote |

To re-check an ImageNet-1K row with independent k-NN and logistic-regression code and a shuffled-label
control:

```bash
uv run python scripts/verify_indomain_result.py --config configs/eval/indomain_in1k_plainvit.yaml \
  --only vit_localballreg_l01_k4
```

</details>

## Under the hood

### Use the regularizer in your own code

[`src/losses.py`](src/losses.py) depends only on torch and numpy, so you can import it or copy it into your
project (MIT license). Stack your projector outputs as `(views, batch, dim)`:

```python
from src.losses import invariance_loss, local_ballreg_loss

lamb = 0.1
inv = invariance_loss(proj)                      # pull each view to the mean of its image's views
reg = local_ballreg_loss(proj, num_clusters=4)   # LocaLeReg: k-means, re-center, uniform-ball test
loss = (1 - lamb) * inv + lamb * reg
```

Each step costs one k-means on the batch plus K small tests. k-means uses `fast-pytorch-kmeans` by default;
pass `kmeans_backend="torch"` to avoid that dependency. `num_clusters` is a maximum: groups smaller than
`min_cluster_size` are skipped, and if all of them are, the whole-batch test runs instead. Pass `stats={}`
to see how many groups were actually used.

### Configuration

Settings are [Hydra](https://hydra.cc) configs in [`configs/`](configs/). Change any key on the command line,
for example `trainer.max_epochs=100 model.lamb=0.05`, and add `--cfg job` to print the final config
without training.

**The regularizer** is one config group, `model/reg_loss`, so switching methods is one word:

| `model/reg_loss=` | Matches the embeddings to |
|---|---|
| `sigreg` | an isotropic Gaussian, whole batch (LeJEPA) |
| `susreg` | the uniform law on a sphere, whole batch (SPHERE-JEPA) |
| `subreg` | a Gaussian, in fixed random subspaces (Sub-JEPA) |
| `rdmreg` | a rectified Laplace law (Rectified LpJEPA; use with `model=rectifiedlpjepa`) |
| `locale_sigreg` | a Gaussian in each of K neighborhoods, without re-centering (a control) |
| `localereg` | **a uniform ball in each of K re-centered neighborhoods (LocaLeReg, LocaLeJEPA)** |

The `locale_sigreg` control helps k-NN but leaves the linear probe at LeJEPA's level, and on Imagenette it
collapses for every setting but λ = 0.1, K = 4; our experiments do not separate re-centering from the ball
target. Set the number of neighborhoods with `model.reg_loss.num_clusters=4`, tune the per-neighborhood test
through `model.reg_loss.loss.*` (for example `model.reg_loss.loss.radius=1.0`), and grow K during training
with `callbacks=dynamic_num_clusters`.

**The encoder** is the `model/encoder` group: `timm_vit` (ViT-S/8, ViT-L/14), `hvit_small_p8` (ViT with a
hyperbolic output space), `d4resnet50` (ResNet-50 equivariant to 90° rotations and flips), and
`nvit_small_p8` / `spherical_vit_small_p8` (normalized and spherical ViTs, rotation study only). Any module
that maps images to a vector works.

## Reproducibility notes

* **The batch is part of the method.** Single-GPU runs take 64 images per step and accumulate 4 steps, so
  256 images make an update. The k-means sees 64, so changing `data.batch_size` changes the neighborhoods,
  not only memory use.
* **Reported checkpoints are the final ones** (`last.ckpt`). `best.ckpt`, chosen by the online probe,
  appears only in a separate "eval-selected" column.
* **ViT-L/14.** A ViT-L/14 pair (`vitl14_*`, 8 GPUs) runs LocaLeJEPA at LeJEPA's own ImageNet-1K settings.
  The ViT-L/14 pair shares its statistic resolution: both runs use 1024 random projections instead of 256.
* **Randomness.** `seed=<int>` seeds everything, but projections, k-means starts and augmentations are
  redrawn at every step, and GPU kernels are not deterministic. Seeded runs agree closely, not bit-for-bit.
* **Software.** The paper's runs used Python 3.10, torch 2.8, lightning 2.2.4, timm 1.0.27 and escnn
  1.0.11; `uv.lock` pins newer versions. Expect agreement within run-to-run noise.
* **Tests.** `uv run pytest tests/unit tests/integration` checks the losses, probes, configs and campaigns,
  and that the overrides quoted in this README exist.

<details>
<summary><b>Where things live</b></summary>

```text
src/train.py                  pretraining entry point
src/losses.py                 every regularizer; local_ballreg_loss is LocaLeJEPA
src/models/                   the LeJEPA module and the encoders
src/data/                     Imagenette, ImageNet-1K and bobblehead data, multi-view augmentation
src/eval_*.py, src/eval/      the evaluations and their probes
configs/experiment/           one YAML per reported run
configs/campaign/             multi-run studies (settings × seeds)
configs/eval/                 which checkpoints each evaluation reads
scripts/slurm_submit.sh       example SLURM launcher
scripts/make_*.py             format the result tables
visualisations/               rotation-study figures
MINIMAL.md                    the ten-minute walkthrough
```

</details>

## Citation

```bibtex
@article{tandon2026localejepa,
  title={LocaLeJEPA: Local Regularizers for Joint-Embedding Predictive Architectures},
  author={Kartik Tandon and Alessio Borgi and Pavel Snopov and In\'{e}s Garc\'{i}a-Redondo and
          Davide D'Ascenzo and Sebastiano Cultrera di Montesano and Fabrizio Silvestri and
          Pietro Li\`{o} and Francesca Dominici and Randall Balestriero and Claudio Battiloro},
  year={2026}
}
```

Built on [LeJEPA](https://github.com/galilai-group/lejepa) (Balestriero & LeCun, 2025),
[lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template),
[hyp_metric](https://github.com/htdt/hyp_metric) (hyperbolic ViT head) and
[equivision](https://github.com/dmklee/equivision) (D4-ResNet-50); third-party notices are in `src/NOTICE`.
Released under the [MIT License](LICENSE).

</div>
