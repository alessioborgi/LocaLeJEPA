#!/bin/bash
#SBATCH -J BLJ_lejepa_vitl14_in1k_paper
#SBATCH -p gpu,gpu_h200,gpu_requeue
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 8
#SBATCH -t 3-00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --constraint="a100|h200"
#SBATCH -o logs/BLJ_lejepa_vitl14_in1k_paper_%j.out
#SBATCH -e logs/BLJ_lejepa_vitl14_in1k_paper_%j.err

# LeJEPA (SIGReg) with the timm ViT-L/14 on ImageNet-1K, 100 epochs.
# Paper setting (LeJEPA arXiv:2511.08544, Sec. 6.1): ViT-L/14 (304M), 2x224 + 8x98 px
# views, 100 epochs, global batch 256, 4-GPU DDP (64 samples / GPU, one node).
# Self-requeues across 3-day allocations (resumes from last.ckpt), then auto-launches
# its Yoda eval.
#
# Saves best.ckpt (by val/probe_acc) + last.ckpt under (netscratch; ~3.7 GB each)
#   /n/netscratch/dominici_lab/Lab/aborgi/checkpoints/lejepa_vitl14_in1k_sigreg_paper  (symlinked from checkpoints/lejepa_vitl14_in1k_sigreg_paper/)
#
# Submit with:
#   sbatch src/slurm_files/slurm_harvard_pretrain_lejepa_vitl14_in1k_paper.sh

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

EXP=lejepa/vitl14_in1k_sigreg_paper
CKPT_DIR="/n/netscratch/dominici_lab/Lab/aborgi/checkpoints/lejepa_vitl14_in1k_sigreg_paper"
RUN_NAME="PreTrain_LeJEPA_vitl14_in1k_sigreg_paper"
mkdir -p "${CKPT_DIR}"
ln -sfn "${CKPT_DIR}" "${SLURM_SUBMIT_DIR}/checkpoints/lejepa_vitl14_in1k_sigreg_paper"
echo "=== Pretrain start $(date) :: ${EXP} ==="
echo "Ckpt dir  : ${CKPT_DIR}"
echo "Python    : $(which python)"

python -c "import hydra, lightning, omegaconf, wandb, timm" || { echo "Missing pkgs"; exit 1; }
test -d "${SLURM_SUBMIT_DIR}/data/imagenet/imagenet_full_size/061417/train" || { echo "IN1K not staged"; exit 1; }

# data.batch_size is the GLOBAL batch (256); the datamodule divides it by the
# DDP world size, so each of the 4 ranks sees 64 samples x (2 global + 8 local)
# views per step -- the same per-rank N as the single-GPU ViT-S/8 runs.
# Resume from last.ckpt if a previous allocation timed out.
RESUME=""
[ -f "${CKPT_DIR}/last.ckpt" ] && RESUME="ckpt_path=${CKPT_DIR}/last.ckpt"

SELF="src/slurm_files/slurm_harvard_pretrain_lejepa_vitl14_in1k_paper.sh"
EVAL_SCRIPT="src/slurm_files/slurm_harvard_eval_yoda_vitl14_sigreg_in1k_paper.sh"

# Self-requeue: queue a successor that runs after this job ENDS for any reason
# (afterany). If this allocation times out before max_epochs, the successor
# resumes from last.ckpt. On a clean finish we cancel it below.
#
# Drop any stale successor before queueing a new one. A preempted job is requeued
# under the SAME job id and re-runs this line, so without this every preemption
# leaves another successor behind -- they then all fire together and race on one
# checkpoint dirpath (the last-v1/last-v2 symptom). Held jobs are left alone:
# those are experiments deliberately deferred by scripts/pretrain_queue.sh.
for _old in $(squeue -u "$USER" -h -t PENDING -o "%i|%j|%r" 2>/dev/null | grep "|${SLURM_JOB_NAME}|" | grep -vi held | cut -d"|" -f1); do
  [ "$_old" != "${SLURM_JOB_ID}" ] && scancel "$_old" 2>/dev/null || true
done
SUCCESSOR=$(sbatch --parsable --dependency=afterany:${SLURM_JOB_ID} "$SELF")
echo "Queued successor ${SUCCESSOR} (afterany:${SLURM_JOB_ID})"
set +e

# One process per GPU via srun; Lightning's SLURMEnvironment picks up
# SLURM_NTASKS / SLURM_PROCID / SLURM_LOCALID and initialises NCCL DDP.
srun --ntasks=4 --ntasks-per-node=4 --cpus-per-task=8 \
  python src/train.py \
  experiment="$EXP" \
  trainer=ddp trainer.devices=4 trainer.num_nodes=1 \
  logger=wandb \
  "logger.wandb.name=${RUN_NAME}" \
  "logger.wandb.tags=[lejepa,imagenet1k,vitl14,sigreg,paper-setting]" \
  "data.data_dir=${SLURM_SUBMIT_DIR}/data/imagenet" \
  data.batch_size=256 \
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
  # A nonzero exit here is a real crash (a timeout would have killed the script
  # before this point). Transient infra failures happen, so let the successor
  # retry up to a cap instead of halting the whole run.
  ATTEMPTS_FILE="${CKPT_DIR}/.crash_attempts"
  n=$(cat "$ATTEMPTS_FILE" 2>/dev/null || echo 0); n=$((n+1)); echo "$n" > "$ATTEMPTS_FILE"
  if [ "$n" -le 8 ]; then
    echo "=== crash #${n}/8 (likely transient): successor ${SUCCESSOR} resumes ==="
  else
    echo "=== too many crashes; stopping ==="; scancel "${SUCCESSOR}" 2>/dev/null || true; bash scripts/pretrain_queue.sh refill --yes || true; exit ${STATUS}
  fi
fi
echo "=== Pretrain end $(date) ==="
