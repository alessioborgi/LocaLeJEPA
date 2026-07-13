#!/bin/bash
#SBATCH -J BLJ_sweep_local
#SBATCH -p gpu,gpu_h200,gpu_requeue
#SBATCH -c 8
#SBATCH -t 3-00:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -o logs/BLJ_sweep_%j.out
#SBATCH -e logs/BLJ_sweep_%j.err

# Stabilized hyperparameter sweep for the local-regularizer backbones, which
# collapsed at the default lamb=0.02 (the local per-cluster anti-collapse signal
# is too weak). Each job stays in a UNIQUE checkpoint dir so runs never collide
# (the -vN stale-checkpoint bug), and saves best.ckpt (by val/probe_acc) + last.
#
# Submit one job per config, e.g.:
#   sbatch --export=ALL,LOSS=local_sigreg,LAMB=0.1,KCL=4 slurm_files/slurm_harvard_sweep_local.sh
#
# Required env: LOSS (local_sigreg|local_ballreg), LAMB (model.lamb), KCL (num_clusters).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

: "${LOSS:?set LOSS=local_sigreg|local_ballreg}"
: "${LAMB:?set LAMB, e.g. 0.1}"
: "${KCL:?set KCL (num_clusters), e.g. 4}"

case "$LOSS" in
  local_sigreg)  EXP=lejepa/vit_imagenette_local_sigreg ;;
  local_ballreg) EXP=lejepa/vit_imagenette_local_ballreg ;;
  *) echo "bad LOSS=$LOSS"; exit 1 ;;
esac

# Stabilizers (fixed across the sweep): lower LR, longer warmup, gradient clipping.
LR="${LR:-1e-3}"
WARMUP="${WARMUP:-10}"
CLIP="${CLIP:-1.0}"

TAG="${LOSS}_lamb${LAMB}_k${KCL}"
CKPT_DIR="${SLURM_SUBMIT_DIR}/checkpoints/sweep_local/${TAG}"
RUN_NAME="PreTrain_sweep_${TAG}"

module purge
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa
wandb login
export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Sweep job start $(date) ==="
echo "LOSS=$LOSS  LAMB=$LAMB  KCL=$KCL  LR=$LR  WARMUP=$WARMUP  CLIP=$CLIP"
echo "ckpt dir: $CKPT_DIR"

python src/train.py \
  experiment="$EXP" \
  logger=wandb \
  "logger.wandb.name=${RUN_NAME}" \
  "logger.wandb.tags=[lejepa,imagenette,vit,sweep,${LOSS}]" \
  "data.data_dir=${SLURM_SUBMIT_DIR}/data/imagenette" \
  data.batch_size=64 \
  +trainer.accumulate_grad_batches=4 \
  "+trainer.gradient_clip_val=${CLIP}" \
  "optimizer.lr=${LR}" \
  "model.lamb=${LAMB}" \
  "model.warmup_epochs=${WARMUP}" \
  "model.local_sigreg_num_clusters=${KCL}" \
  "callbacks.model_checkpoint.dirpath=${CKPT_DIR}"

echo "=== Sweep job end $(date) ==="
