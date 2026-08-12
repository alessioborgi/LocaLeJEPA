#!/bin/bash
#SBATCH -J BLJ_d4rn50_localballreg_l03_k2
#SBATCH -p gpu,gpu_h200,gpu_requeue
#SBATCH -c 8
#SBATCH -t 3-00:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h200"
#SBATCH -o logs/BLJ_d4rn50_localballreg_l03_k2_%j.out
#SBATCH -e logs/BLJ_d4rn50_localballreg_l03_k2_%j.err

# D4-ResNet-50 (equiv.) + local BallReg (λ0.3, k2) LeJEPA pretraining on ImageNet-1K (100 epochs).
# Self-requeues across allocations, then auto-launches its Yoda eval.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module purge
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa

wandb login
export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP=lejepa/d4rn50_in1k_localballreg_l03_k2
CKPT_DIR="${SLURM_SUBMIT_DIR}/checkpoints/lejepa_d4rn50_in1k_localballreg_l03_k2"
RUN_NAME="PreTrain_LeJEPA_d4rn50_in1k_localballreg_l03_k2"
echo "=== Pretrain start $(date) :: ${EXP} ==="

python -c "import hydra, lightning, omegaconf, wandb, escnn" || { echo "Missing pkgs"; exit 1; }
test -d "${SLURM_SUBMIT_DIR}/data/imagenet/imagenet_full_size/061417/train" || { echo "IN1K not staged"; exit 1; }

RESUME=""
[ -f "${CKPT_DIR}/last.ckpt" ] && RESUME="ckpt_path=${CKPT_DIR}/last.ckpt"

SELF="src/slurm_files/slurm_harvard_pretrain_lejepa_d4rn50_in1k_localballreg_l03_k2.sh"
EVAL_SCRIPT="src/slurm_files/slurm_harvard_eval_yoda_d4rn50_localballreg_l03_k2_in1k.sh"
SUCCESSOR=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} "$SELF")
echo "Queued successor ${SUCCESSOR} (afterany:${SLURM_JOB_ID})"
set +e

python src/train.py \
  experiment="$EXP" \
  +trainer.gradient_clip_val=1.0 \
  logger=wandb \
  "logger.wandb.name=${RUN_NAME}" \
  "logger.wandb.tags=[lejepa,imagenet1k,d4resnet50,equivariant,local-ballreg]" \
  "data.data_dir=${SLURM_SUBMIT_DIR}/data/imagenet" \
  data.batch_size=64 \
  +trainer.accumulate_grad_batches=4 \
  "callbacks.model_checkpoint.dirpath=${CKPT_DIR}" \
  ${RESUME}
STATUS=$?
set -e
if [ $STATUS -eq 0 ]; then
  echo "=== Reached max_epochs $(date): cancel successor, launch eval ==="
  scancel "${SUCCESSOR}" 2>/dev/null || true
  sbatch "$EVAL_SCRIPT"
  bash scripts/pretrain_queue.sh refill --yes || true   # free this slot for the next held run
else
  ATTEMPTS_FILE="${CKPT_DIR}/.crash_attempts"
  n=$(cat "$ATTEMPTS_FILE" 2>/dev/null || echo 0); n=$((n+1)); echo "$n" > "$ATTEMPTS_FILE"
  if [ "$n" -le 8 ]; then
    echo "=== crash #${n}/8 (likely transient): successor ${SUCCESSOR} resumes ==="
  else
    echo "=== too many crashes; stopping ==="; scancel "${SUCCESSOR}" 2>/dev/null || true; bash scripts/pretrain_queue.sh refill --yes || true; exit ${STATUS}
  fi
fi
echo "=== Pretrain end $(date) ==="
