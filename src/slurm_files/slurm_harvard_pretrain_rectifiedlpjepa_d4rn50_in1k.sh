#!/bin/bash
#SBATCH -J BLJ_rectifiedlpjepa_d4rn50_in1k
#SBATCH -p gpu,gpu_h200,gpu_requeue
#SBATCH -c 8
#SBATCH -t 3-00:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h200"
#SBATCH -o logs/BLJ_rectifiedlpjepa_d4rn50_in1k_%j.out
#SBATCH -e logs/BLJ_rectifiedlpjepa_d4rn50_in1k_%j.err

# Rectified LpJEPA (RDMReg, Rectified Laplace target) pretraining on
# ImageNet-1K with the equivision D4-equivariant ResNet-50 (escnn group
# convolutions, ~24.8M params) as the encoder instead of the ViT-S/8.
# Requires ImageNet staged under
#   data/imagenet/imagenet_full_size/061417/{train,val}
#
# Saves best.ckpt (by val/probe_acc) + last.ckpt under
#   checkpoints/rectifiedlpjepa_d4rn50_in1k_rdmreg/
#
# Submit with:
#   sbatch src/slurm_files/slurm_harvard_pretrain_rectifiedlpjepa_d4rn50_in1k.sh

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

EXP=rectifiedlpjepa/d4rn50_in1k_rdmreg
CKPT_DIR="${SLURM_SUBMIT_DIR}/checkpoints/rectifiedlpjepa_d4rn50_in1k_rdmreg"
RUN_NAME="PreTrain_RectifiedLpJEPA_d4rn50_in1k_rdmreg"

echo "=== Pretrain start $(date) ==="
echo "Experiment: ${EXP}"
echo "Ckpt dir  : ${CKPT_DIR}"
echo "Python    : $(which python)"

python -c "import hydra, lightning, omegaconf, wandb, escnn" || {
  echo "Missing required Python packages in the active Conda environment."
  exit 1
}
test -d "${SLURM_SUBMIT_DIR}/data/imagenet/imagenet_full_size/061417/train" || {
  echo "ImageNet-1K not staged — run slurm_harvard_stage_imagenet1k.sh first."
  exit 1
}

# encode() flattens all views into one tensor and the escnn group convs carry
# |D4|=8 regular-representation channels per field, so keep the per-step batch
# small and accumulate to an effective global batch of 256. Resume from
# last.ckpt if a previous allocation timed out.
RESUME=""
[ -f "${CKPT_DIR}/last.ckpt" ] && RESUME="ckpt_path=${CKPT_DIR}/last.ckpt"

SELF="src/slurm_files/slurm_harvard_pretrain_rectifiedlpjepa_d4rn50_in1k.sh"
EVAL_SCRIPT="src/slurm_files/slurm_harvard_eval_yoda_d4rn50_rdmreg_in1k.sh"
# Self-requeue: queue a successor that runs after this job ENDS for any
# reason (afterany). If this allocation times out before reaching
# max_epochs, the script is killed here and the successor resumes from
# last.ckpt. On a clean finish or a crash we cancel it below.
SUCCESSOR=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} "$SELF")
echo "Queued successor ${SUCCESSOR} (afterany:${SLURM_JOB_ID}) to continue toward max_epochs"
set +e

python src/train.py \
  experiment="$EXP" \
  trainer.max_epochs=100 \
  logger=wandb \
  "logger.wandb.name=${RUN_NAME}" \
  "logger.wandb.tags=[rectifiedlpjepa,imagenet1k,d4resnet50,equivariant,rdmreg]" \
  "data.data_dir=${SLURM_SUBMIT_DIR}/data/imagenet" \
  data.batch_size=64 \
  +trainer.accumulate_grad_batches=4 \
  "callbacks.model_checkpoint.dirpath=${CKPT_DIR}" \
  ${RESUME}
STATUS=$?
set -e
if [ $STATUS -eq 0 ]; then
  echo "=== Reached max_epochs $(date): cancel successor ${SUCCESSOR}, launch eval ==="
  scancel "${SUCCESSOR}" 2>/dev/null || true
  sbatch "$EVAL_SCRIPT"
  bash scripts/pretrain_queue.sh refill --yes || true   # free this slot for the next held run
else
  # A nonzero exit here is a real crash (a timeout would have killed the
  # script before this point). Transient infra failures happen (e.g. "CUDA
  # device busy/unavailable" at setup on a bad node), so let the successor
  # retry up to a cap instead of halting the whole run.
  ATTEMPTS_FILE="${CKPT_DIR}/.crash_attempts"
  n=$(cat "$ATTEMPTS_FILE" 2>/dev/null || echo 0); n=$((n+1)); echo "$n" > "$ATTEMPTS_FILE"
  if [ "$n" -le 8 ]; then
    echo "=== Training exited ${STATUS} (crash #${n}/8, likely transient): letting successor ${SUCCESSOR} resume ==="
  else
    echo "=== Too many crashes (${n}); stopping retry chain and cancelling ${SUCCESSOR} ==="
    scancel "${SUCCESSOR}" 2>/dev/null || true
    exit ${STATUS}
  fi
fi
echo "=== Pretrain end $(date) ==="
