#!/bin/bash
#SBATCH --job-name=blj-ksweep-seeded
#SBATCH --output=/HOME/garciare/BundLeJEPA/logs/embedor/output_%x_%A_%a.log
#SBATCH --error=/HOME/garciare/BundLeJEPA/logs/embedor/error_%x_%A_%a.log
#SBATCH --array=0-8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --partition=GPU
#SBATCH --nodelist=diufrd205
#SBATCH --chdir=/HOME/garciare/BundLeJEPA

# Seeded k-sweep for the "probe learns faster as k grows" finding.
#   tasks 0-8  : BallJEPA local BallReg, lambda=0.1, k in {2,4,8} x seed in {0,1,2}
#   tasks 9-11 : LeJEPA (SIGReg) x seed in {0,1,2}   -- run with --array=0-11 to
#                include the cross-method speed baseline
#
# Each run keeps periodic weight-only snapshots (callbacks=with_snapshots,
# every 25 epochs -> checkpoints/<name>/snapshots/) for eval_snapshots.py, plus
# the usual best.ckpt / last.ckpt. ~2.7 GB of snapshots per run.
#
#   sbatch src/slurm_files/slurm_diufrd_pretrain_ksweep_seeded.sh            # k-sweep only
#   sbatch --array=0-11 src/slurm_files/slurm_diufrd_pretrain_ksweep_seeded.sh
#   sbatch --array=6-8  src/slurm_files/slurm_diufrd_pretrain_ksweep_seeded.sh   # just k=8

set -euo pipefail

source "$HOME/.local/bin/env" 2>/dev/null || true
source .venv/bin/activate

export RICH_NO_COLOR=1
export NO_COLOR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB__SERVICE_WAIT=300
export WANDB_START_METHOD=thread

mkdir -p logs/embedor checkpoints
test -d data/imagenette/imagenette2-160/train || { echo "ERROR: Imagenette not staged"; exit 1; }
python -c "import wandb, lightning, timm, hydra" || { echo "ERROR: run 'uv sync'"; exit 1; }

NAMES=(
  ball_l01_k2_s0 ball_l01_k2_s1 ball_l01_k2_s2
  ball_l01_k4_s0 ball_l01_k4_s1 ball_l01_k4_s2
  ball_l01_k8_s0 ball_l01_k8_s1 ball_l01_k8_s2
  lejepa_s0      lejepa_s1      lejepa_s2
)
EXPERIMENTS=(
  lejepa/vit_imagenette_localballreg_l01_k2 lejepa/vit_imagenette_localballreg_l01_k2 lejepa/vit_imagenette_localballreg_l01_k2
  lejepa/vit_imagenette_localballreg_l01_k4 lejepa/vit_imagenette_localballreg_l01_k4 lejepa/vit_imagenette_localballreg_l01_k4
  lejepa/vit_imagenette_localballreg_l01_k8 lejepa/vit_imagenette_localballreg_l01_k8 lejepa/vit_imagenette_localballreg_l01_k8
  lejepa/vit_imagenette_sigreg              lejepa/vit_imagenette_sigreg              lejepa/vit_imagenette_sigreg
)
SEEDS=(0 1 2 0 1 2 0 1 2 0 1 2)

i="${SLURM_ARRAY_TASK_ID:?run as an array job}"
NAME="${NAMES[$i]}"
EXP="${EXPERIMENTS[$i]}"
SEED="${SEEDS[$i]}"
CKPT_DIR="checkpoints/${NAME}"

RESUME=""
[[ -f "${CKPT_DIR}/last.ckpt" ]] && RESUME="ckpt_path=${CKPT_DIR}/last.ckpt"

WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_ID="blj_lc_${NAME}_${SLURM_ARRAY_JOB_ID:-manual}"

echo "=== [$i] ${NAME}  (${EXP}, seed ${SEED})  start $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python src/train.py \
  experiment="${EXP}" \
  seed="${SEED}" \
  callbacks=with_snapshots \
  +trainer.gradient_clip_val=1.0 \
  logger=many_loggers \
  "logger.wandb.name=blj_lc_${NAME}" \
  "logger.wandb.id=${WANDB_ID}" \
  "logger.wandb.group=embedor_lc_ksweep" \
  "logger.wandb.entity=${WANDB_ENTITY:-null}" \
  "+logger.wandb.resume=allow" \
  "logger.wandb.tags=[embedor,imagenette,vit-s8,learning-curve,${NAME}]" \
  "callbacks.model_checkpoint.dirpath=${CKPT_DIR}" \
  ${RESUME}

echo "=== [$i] ${NAME}  end $(date) ==="
