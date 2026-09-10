#!/bin/bash
#SBATCH --job-name=blj-ball-ksweep
#SBATCH --output=/HOME/garciare/BundLeJEPA/logs/embedor/output_%x_%A_%a.log
#SBATCH --error=/HOME/garciare/BundLeJEPA/logs/embedor/error_%x_%A_%a.log
#SBATCH --array=0-2
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --partition=GPU
#SBATCH --nodelist=diufrd205
#SBATCH --chdir=/HOME/garciare/BundLeJEPA

# Second BallJEPA sweep for the EmbedOR study: crank locality (k) and the
# regularizer weight (lambda) past the l0.1/k2-4 point that came out weak.
#
#   task 0  ball_l01_k8   lambda 0.1  k 8   -- locality arm  (k2 -> k4 -> k8)
#   task 1  ball_l03_k4   lambda 0.3  k 4   -- lambda arm     (l0.1 -> l0.3)
#   task 2  ball_l03_k8   lambda 0.3  k 8   -- strong + local (the hypothesis corner)
#
# 800 epochs, batch 256, ViT-S/8, online probe. Checkpoints ->
# checkpoints/<name>/{best,last}.ckpt. Re-submitting resumes from last.ckpt.
#
#   sbatch src/slurm_files/slurm_diufrd_pretrain_balljepa_ksweep.sh
#
# Then add the 3 names to configs/eval_embedor_imagenette.yaml and re-run
# src/slurm_files/slurm_diufrd_eval_embedor.sh.

set -euo pipefail

source "$HOME/.local/bin/env" 2>/dev/null || true
source .venv/bin/activate

export RICH_NO_COLOR=1
export NO_COLOR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB__SERVICE_WAIT=300
export WANDB_START_METHOD=thread

mkdir -p logs/embedor checkpoints
test -d data/imagenette/imagenette2-160/train || {
  echo "ERROR: Imagenette not staged at data/imagenette/imagenette2-160/."; exit 1;
}
python -c "import wandb, lightning, timm, hydra" || {
  echo "ERROR: env missing packages -- run 'uv sync' on the login node."; exit 1;
}

NAMES=(ball_l01_k8 ball_l03_k4 ball_l03_k8)
EXPERIMENTS=(
  lejepa/vit_imagenette_localballreg_l01_k8
  lejepa/vit_imagenette_localballreg_l03_k4
  lejepa/vit_imagenette_localballreg_l03_k8
)

i="${SLURM_ARRAY_TASK_ID:?run as an array job: sbatch --array=0-2 ...}"
NAME="${NAMES[$i]}"
EXP="${EXPERIMENTS[$i]}"
CKPT_DIR="checkpoints/${NAME}"

RESUME=""
[[ -f "${CKPT_DIR}/last.ckpt" ]] && RESUME="ckpt_path=${CKPT_DIR}/last.ckpt"

WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_ID="blj_embedor_${NAME}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"

echo "=== [$i] ${NAME}  (${EXP})  start $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python src/train.py \
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
