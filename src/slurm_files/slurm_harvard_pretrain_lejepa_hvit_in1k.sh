#!/bin/bash
#SBATCH -J BLJ_lejepa_hvit_in1k
#SBATCH -p gpu,gpu_h200,gpu_requeue
#SBATCH -c 8
#SBATCH -t 3-00:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h200"
#SBATCH -o logs/BLJ_lejepa_hvit_in1k_%j.out
#SBATCH -e logs/BLJ_lejepa_hvit_in1k_%j.err

# LeJEPA (SIGReg) pretraining on ImageNet-1K with the Hyperbolic ViT-S/8
# (Poincare-ball embedding, hyp_c=0.1, clip_r=2.3). Requires ImageNet staged
# via slurm_harvard_stage_imagenet1k.sh (data/imagenet symlink).
#
# Saves best.ckpt (by val/probe_acc) + last.ckpt under
#   checkpoints/lejepa_hvit_in1k_sigreg/
#
# Submit with:
#   sbatch src/slurm_files/slurm_harvard_pretrain_lejepa_hvit_in1k.sh

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

EXP=lejepa/hvit_in1k_sigreg
CKPT_DIR="${SLURM_SUBMIT_DIR}/checkpoints/lejepa_hvit_in1k_sigreg"
RUN_NAME="PreTrain_LeJEPA_hvit_in1k_sigreg"

echo "=== Pretrain start $(date) ==="
echo "Experiment: ${EXP}"
echo "Ckpt dir  : ${CKPT_DIR}"
echo "Python    : $(which python)"

python -c "import hydra, lightning, omegaconf, wandb, timm" || {
  echo "Missing required Python packages in the active Conda environment."
  exit 1
}
test -d "${SLURM_SUBMIT_DIR}/data/imagenet/imagenet_full_size/061417/train" || {
  echo "ImageNet-1K not staged — run slurm_harvard_stage_imagenet1k.sh first."
  exit 1
}

# encode() flattens all views into one tensor (batch * num_views images at
# once), so use batch_size=64 + accumulate 4 to keep the effective global batch
# at 256. Resume from last.ckpt if a previous allocation timed out.
RESUME=""
[ -f "${CKPT_DIR}/last.ckpt" ] && RESUME="ckpt_path=${CKPT_DIR}/last.ckpt"

SELF="src/slurm_files/slurm_harvard_pretrain_lejepa_hvit_in1k.sh"
EVAL_SCRIPT="src/slurm_files/slurm_harvard_eval_yoda_hvit_sigreg_in1k.sh"
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
  "logger.wandb.tags=[lejepa,imagenet1k,hvit,hyperbolic,sigreg]" \
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
