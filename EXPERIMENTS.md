# Reproducing the experiments

This branch (`experiments`) = `main` + everything needed to rerun and re-analyse the reported
experiments. Install with

```bash
uv sync --all-extras          # equivariant (escnn, needs gfortran), imagenet (HF staging), experiments (plots, scipy, sklearn)
```

and run every command below from the repository root (`uv run python ...`; the `eval_*.py`
scripts import `src` relative to the current directory).

## Layout added on top of `main`

```
configs/experiment/         all pretraining configs (see configs/experiment/README.md for the map)
configs/eval/               in-domain (k-NN + linear), transfer, RankMe configs
configs/eval/yoda/          rotation-angle ("Yoda" bobblehead) probe configs, one per backbone
eval_transfer.py            7-dataset transfer linear probing (SPHERE-JEPA Table 1 protocol)
eval_rankme.py              RankMe effective rank of the frozen features
eval_angle_probe*.py        ridge probe of the two figurine rotation angles (+ multi-seed, low-data)
eval_conformal.py           Conformal-Autoencoder intrinsic-dimension detection on frozen features
scripts/make_*.py           regenerate the result tables in results/ (+ LaTeX in paper/)
scripts/download_transfer_datasets.py   torchvision transfer datasets
scripts/verify_plainvit_result.py   independent audit of the headline plain-ViT numbers
scripts/slurm_submit.sh     the single SLURM launcher
scripts/benchmark_regularizers.py   forward+backward time / memory of the regularizers on synthetic batches
visualisations/             torus / low-data figures of the Yoda probe (write to visualisations/figures/)
results/                    the generated result tables (Markdown) and the k-schedule / radius / complexity write-ups
src/models/                 rectifiedlpjepa (thin LeJEPAModule subclass); other baselines are reg_loss configs
src/models/components/      nvit, spherical_vit, vision_transformer + tensor_ops (I-JEPA code they use)
src/data/                   stl10, bobblehead datamodules
src/callbacks/              dynamic k schedule
```

## 1. Pretraining

| what | config | launch |
|---|---|---|
| LeJEPA (SIGReg), ViT-S/8, IN1K, 100 ep. | `lejepa/vit_in1k_sigreg` | `uv run python src/train.py experiment=lejepa/vit_in1k_sigreg` |
| **Ours**, local BallReg λ0.1 k4, ViT-S/8 | `lejepa/vit_in1k_localballreg_l01_k4` | `uv run python src/train.py experiment=lejepa/vit_in1k_localballreg_l01_k4` |
| same pair, hyperbolic ViT-S/8 | `lejepa/hvit_in1k_{sigreg,localballreg_l01_k4}` | idem |
| same pair, D4-ResNet-50 (`--extra equivariant`) | `lejepa/d4rn50_in1k_{sigreg,localballreg_l01_k4}` | idem |
| paper setting, ViT-L/14, 4 GPUs | `lejepa/vitl14_in1k_{sigreg,localballreg_l01_k4}_paper` | `... trainer=ddp trainer.devices=4` |
| ablations λ / k / dynamic k / local SIGReg / RDMReg | see `configs/experiment/README.md` | idem |

Sweeps are Hydra multiruns: `-m` with comma-separated values runs the cartesian product, e.g.
`uv run python src/train.py -m experiment=lejepa/vit_imagenette_localballreg_l01_k4 model.reg_loss.num_clusters=2,4,8`.
W&B names each run after its CLI overrides (`logger.wandb.name` in `configs/logger/wandb.yaml`).
Multirun is sequential on one GPU; on SLURM submit one `scripts/slurm_submit.sh` job per config.

Pin checkpoints to a stable directory so the eval configs find them:
`callbacks.model_checkpoint.dirpath=checkpoints/lejepa_vit_in1k_sigreg` (the eval configs
expect `checkpoints/<model>_<experiment name>/{best,last}.ckpt`, e.g.
`checkpoints/lejepa_vit_in1k_localballreg_l01_k4/last.ckpt`). Resume with `ckpt_path=<dir>/last.ckpt`.

ImageNet-1K: `uv run python scripts/stage_imagenet1k_hf.py --dest data/imagenet` (gated HF
dataset; accept the licence and `hf auth login` first). Imagenette downloads itself.

## 2. Evaluation (the reported tables)

All evaluators take a YAML listing `backbones: [{name, experiment, checkpoint}]`, write one
`per_backbone/<name>.json` each (safe to run one backbone per job with `--only <name>`), and
`--merge-only` rebuilds `results.{json,csv}` from those files.

| table | command | then |
|---|---|---|
| IN1K k-NN + linear, HViT / D4-RN50 (`results/Imagenet1K_results.md`) | `uv run python eval_indomain.py --config configs/eval/indomain_in1k_ep100.yaml` | `uv run python scripts/make_imagenet1k_results.py` |
| IN1K k-NN + linear, plain ViT (`results/report_plainViT_LastExp.md`) | `uv run python eval_indomain.py --config configs/eval/indomain_in1k_plainvit.yaml` | `uv run python scripts/make_plainvit_report.py` |
| transfer, 7 datasets (`results/Transfer_results.md`) | `uv run python scripts/download_transfer_datasets.py --root <root>` once, then `uv run python eval_transfer.py --config configs/eval/transfer_in1k_ep100.yaml` | `uv run python scripts/make_transfer_results.py` |
| RankMe (`results/RankMe_results.md`) | `uv run python eval_rankme.py --config configs/eval/rankme_in1k_ep100.yaml` | `uv run python scripts/make_rankme_results.py` |
| Yoda rotation-angle probe (`results/report_equivariance.md`) | per backbone: `uv run python eval_angle_probe.py --config configs/eval/yoda/<name>.yaml` then `uv run python eval_angle_probe_multiseed.py --config configs/eval/yoda/<name>.yaml --n-seeds 10` | `uv run python scripts/make_equivariance_report.py` |

The bobblehead data (`data/yoda_dog_images_and_labels.zip`, 8097 frames + `labels.npy`) is the
dataset of Kevrekidis et al., arXiv:2408.16138 and is not redistributed here.
`eval_conformal.py --config configs/eval/conformal.yaml` runs the Conformal-Autoencoder
dimension-detection experiment on the same frames.

Machine-specific paths in the eval configs are `cache_dir` (feature caches, ~1.4 GB per IN1K
backbone) and `data_root` for the transfer datasets — change them to your storage.

## 3. Figures

`visualisations/*.py` read the Yoda feature caches / metrics under `logs/` and write PNG + PDF
into `visualisations/figures/` (git-ignored; see `visualisations/README.md`).
`scripts/make_rankme_results.py` also writes the RankMe figures.

## 4. SLURM

`scripts/slurm_submit.sh` is the only submission script. Its `#SBATCH` header carries the
defaults (partition `gpu`, 1 GPU, 8 CPUs, 64 GB, 3 days, logs in `logs/slurm/`); override any of
them on the `sbatch` line. The job runs through `uv run` and is selected with environment variables:

```bash
# pretraining (resumes from checkpoints/<run>/last.ckpt if present; REQUEUE=1 chains a successor)
sbatch --export=ALL,EXPERIMENT=lejepa/vit_in1k_sigreg scripts/slurm_submit.sh
sbatch --export=ALL,EXPERIMENT=lejepa/vit_in1k_localballreg_l01_k4,REQUEUE=1 scripts/slurm_submit.sh
sbatch --gres=gpu:4 --ntasks-per-node=4 --mem=256G \
       --export=ALL,EXPERIMENT=lejepa/vitl14_in1k_localballreg_l01_k4_paper,GPUS=4,REQUEUE=1 scripts/slurm_submit.sh

# evaluation / reports: any command
sbatch --time=6:00:00 --mem=128G --cpus-per-task=16 \
       --export=ALL,CMD="python eval_indomain.py --config configs/eval/indomain_in1k_ep100.yaml --only vit_sigreg" scripts/slurm_submit.sh
```

Variables: `EXPERIMENT`, `GPUS`, `RUN_NAME`, `CKPT_DIR`, `REQUEUE`, `OVERRIDES="logger=csv ..."`,
`DATA_DIR` (→ `paths.data_dir`), `ENV_SETUP` (a file to `source` first, e.g. module loads or
`export PATH=$HOME/.local/bin:$PATH` for `uv`). Run `uv sync` on a login node before submitting
(compute nodes are assumed offline; the job uses `uv run --no-sync`).

## 5. Environment used for the reported numbers

Harvard FASRC cluster, conda env with Python 3.10.20, torch 2.8.0+cu128, torchvision 0.23,
lightning 2.2.4, timm 1.0.27, escnn 1.0.11, numpy 1.26.4, fast-pytorch-kmeans 0.2.2,
wandb 0.18.5. `uv.lock` pins newer versions (torch 2.14 / lightning 2.6.6 / timm 1.0.29) that
run the same code; small numerical differences across library versions are possible.
