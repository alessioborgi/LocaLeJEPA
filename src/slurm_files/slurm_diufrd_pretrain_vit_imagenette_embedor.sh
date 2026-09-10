#!/bin/bash
#SBATCH --job-name=blj-embedor-pretrain
#SBATCH --output=/HOME/garciare/BundLeJEPA/logs/embedor/output_%x_%A_%a.log
#SBATCH --error=/HOME/garciare/BundLeJEPA/logs/embedor/error_%x_%A_%a.log
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --partition=GPU
#SBATCH --nodelist=diufrd205
#SBATCH --chdir=/HOME/garciare/BundLeJEPA

# Pretrain the four ViT-S/8 / Imagenette backbones for the EmbedOR class-
# separation study, one SLURM array task per backbone. diufrd205 has 8x RTX PRO
# 6000, so all four run at once (1 GPU each). 800 epochs, global batch 256,
# online linear probe, logged to W&B (project BundLeJEPA, group embedor_vit_imagenette).
#
# Checkpoints -> checkpoints/<name>/{best,last}.ckpt  (best = max val/probe_acc).
# Re-submitting the array resumes any task that already has a last.ckpt.
#
# PREREQUISITES (run once on the LOGIN node, before sbatch):
#   1. mkdir -p /HOME/garciare/BundLeJEPA/logs/embedor
#   2. wandb login                       # writes ~/.netrc (read by the compute node)
#   3. uv sync                           # build .venv/ from pyproject
#   4. stage Imagenette so the 4 tasks don't race on the download:
#        uv run python -c "from torchvision.datasets.utils import download_and_extract_archive as d; \
#          d('https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz', 'data/imagenette', filename='imagenette2-160.tgz')"
#
# Submit:
#   sbatch src/slurm_files/slurm_diufrd_pretrain_vit_imagenette_embedor.sh
#   sbatch --array=2-3 src/slurm_files/slurm_diufrd_pretrain_vit_imagenette_embedor.sh   # subset
#
# Verify when done:
#   ls checkpoints/{lejepa,rectlpjepa,ball_l01_k2,ball_l01_k4}/best.ckpt

set -euo pipefail

# --- environment ------------------------------------------------------------
# uv-managed venv (see PREREQUISITES). If yours is conda, swap for
# `source activate <env>` and it still works as long as `python` has the deps.
source "$HOME/.local/bin/env" 2>/dev/null || true   # put uv on PATH
source .venv/bin/activate
RUN="python"

export RICH_NO_COLOR=1
export NO_COLOR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB__SERVICE_WAIT=300
export WANDB_START_METHOD=thread
# export WANDB_MODE=offline        # uncomment if diufrd205 has no outbound network,
                                   # then `wandb sync wandb/offline-run-*` afterwards

# --- preflight -------------------------------------------------------------
mkdir -p logs/embedor checkpoints
test -d data/imagenette/imagenette2-160/train || {
  echo "ERROR: Imagenette not staged at data/imagenette/imagenette2-160/."
  echo "       Run the staging one-liner from PREREQUISITES on the login node first."
  exit 1
}
$RUN -c "import wandb, lightning, timm, hydra" || {
  echo "ERROR: env missing packages -- run 'uv sync' on the login node."
  exit 1
}

# --- task selection -------------------------------------------------------
NAMES=(lejepa rectlpjepa ball_l01_k2 ball_l01_k4)
EXPERIMENTS=(
  lejepa/vit_imagenette_sigreg
  rectifiedlpjepa/vit_imagenette_rdmreg
  lejepa/vit_imagenette_localballreg_l01_k2
  lejepa/vit_imagenette_localballreg_l01_k4
)

i="${SLURM_ARRAY_TASK_ID:?run as an array job: sbatch --array=0-3 ...}"
NAME="${NAMES[$i]}"
EXP="${EXPERIMENTS[$i]}"
CKPT_DIR="checkpoints/${NAME}"

RESUME=""
[[ -f "${CKPT_DIR}/last.ckpt" ]] && RESUME="ckpt_path=${CKPT_DIR}/last.ckpt"

echo "=== [$i] ${NAME}  (${EXP})  start $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# W&B: one run per backbone per submission, grouped so the four collapse into
# one comparison panel. The run id is tied to the SLURM job id (not the backbone
# name), so a within-submission requeue continues the same run, but a manual
# kill-and-resubmit starts a fresh run instead of appending to the old one.
# WANDB_ENTITY: your W&B team/user. configs/logger/wandb.yaml hardcodes
# "sheaf_hypergraphs"; override here. Empty string => your default entity.
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_ID="blj_embedor_${NAME}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"

$RUN src/train.py \
  experiment="${EXP}" \
  +trainer.gradient_clip_val=1.0 \
  logger=many_loggers \
  "logger.wandb.name=blj_embedor_${NAME}" \
  "logger.wandb.id=${WANDB_ID}" \
  "logger.wandb.group=embedor_vit_imagenette" \
  "logger.wandb.entity=${WANDB_ENTITY:-null}" \
  "+logger.wandb.resume=allow" \
  "logger.wandb.tags=[embedor,imagenette,vit-s8,${NAME}]" \
  "callbacks.model_checkpoint.dirpath=${CKPT_DIR}" \
  ${RESUME}

echo "=== [$i] ${NAME}  end $(date) ==="
