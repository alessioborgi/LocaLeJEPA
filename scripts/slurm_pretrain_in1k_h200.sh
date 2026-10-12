#!/bin/bash
#SBATCH --job-name=BLJ_in1k
#SBATCH --partition=gpu_h200,gpu,gpu_requeue
#SBATCH --exclude=holygpu8a12204
#SBATCH --requeue
#SBATCH --constraint=h200|h100|a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# Generic ImageNet-1K pretrain on 4x H200 DDP: per-GPU batch 64 x 4 ranks x accum 1
# = effective 256, i.e. the same effective batch AND the same per-rank regulariser
# batch as the single-GPU rows of Table 3, so the numbers stay comparable.
# Generalises scripts/slurm_pretrain_vit_dynk_h200.sh to any experiment.
# gpu_requeue: idle lab GPUs, pre-emptible; a pre-empted job is requeued and resumes
# from last.ckpt. holygpu8a12204 is excluded: its GPUs fail with "CUDA-capable
# device(s) is/are busy or unavailable" (2026-09-29, ate 10 job starts).
#
#   sbatch -J BLJ_vit_rdmreg --export=ALL,EXP=rectifiedlpjepa/vit_in1k_rdmreg,\
#CKPT=checkpoints/rectifiedlpjepa_vit_in1k_rdmreg scripts/slurm_pretrain_in1k_h200.sh
#
# EXP  : hydra experiment= value
# CKPT : checkpoint directory (resumed from last.ckpt if present)
# EXTRA: optional extra hydra overrides (default: the stabilised local recipe)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa
export PROJECT_ROOT=$PWD WANDB_START_METHOD=thread PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_ASYNC_ERROR_HANDLING=1 OMP_NUM_THREADS=8
: "${EXP:?set EXP=<hydra experiment>}"
: "${CKPT:?set CKPT=<checkpoint dir>}"
CKPT_DIR="$PWD/$CKPT"
SELF=scripts/slurm_pretrain_in1k_h200.sh
mkdir -p "$CKPT_DIR"

# drop stale pending copies of this job name, then queue ONE successor (afterany) that
# resumes from last.ckpt if this allocation dies early; cancelled on clean completion.
for o in $(squeue -u "$(whoami)" -h -t PENDING -o "%i|%j" | awk -F'|' -v n="$SLURM_JOB_NAME" '$2==n{print $1}'); do
  [ "$o" != "$SLURM_JOB_ID" ] && scancel "$o" 2>/dev/null || true
done
SUCC=$(sbatch --parsable -J "$SLURM_JOB_NAME" --dependency=afterany:$SLURM_JOB_ID "$SELF" 2>/dev/null || echo "")
echo "=== $SLURM_JOB_NAME $(date) on $(hostname) | exp=$EXP | ckpt=$CKPT_DIR | successor ${SUCC:-none} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

run_training () {
  RESUME=""; [ -f "$CKPT_DIR/last.ckpt" ] && RESUME="ckpt_path=$CKPT_DIR/last.ckpt"
  # --kill-on-bad-exit=1: if one rank dies at startup the whole step is killed at once,
  # otherwise the survivors block forever in the NCCL barrier and idle the allocation.
  srun --kill-on-bad-exit=1 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=10 \
    python src/train.py experiment="$EXP" \
    trainer=ddp trainer.devices=4 trainer.num_nodes=1 \
    trainer.max_epochs=100 trainer.accumulate_grad_batches=1 \
    logger=csv \
    "data.data_dir=$PWD/data/imagenet" data.batch_size=64 \
    data.num_workers=8 data.persistent_workers=true \
    "callbacks.model_checkpoint.dirpath=$CKPT_DIR" ${EXTRA:-} $RESUME
}

wait_for_gpus () {   # a GPU can still be held by the previous job for a few seconds
  for _ in $(seq 1 18); do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    [ "$n" -eq 0 ] && return 0
    echo "  $n process(es) still on the GPUs, waiting 10s..."; sleep 10
  done
  echo "  GPUs still busy after 3 min, trying anyway"
}

STATUS=1
for attempt in 1 2 3; do
  echo "--- training attempt $attempt/3 $(date '+%F %T') ---"
  wait_for_gpus
  set +e; run_training; STATUS=$?; set -e
  [ $STATUS -eq 0 ] && break
  echo "--- attempt $attempt exited $STATUS; retrying in this allocation in 60s ---"
  sleep 60
done
if [ $STATUS -eq 0 ]; then
  echo "=== reached max_epochs: cancelling successor ==="
  [ -n "$SUCC" ] && scancel "$SUCC" 2>/dev/null || true
else
  A="$CKPT_DIR/.crash_attempts"; n=$(( $(cat "$A" 2>/dev/null || echo 0) + 1 )); echo $n > "$A"
  if [ $n -gt 6 ]; then echo "too many crashes; stopping"; [ -n "$SUCC" ] && scancel "$SUCC" 2>/dev/null || true; exit $STATUS; fi
  echo "=== exit $STATUS (crash #$n); successor $SUCC resumes from last.ckpt ==="
fi
echo "=== end $(date) | status $STATUS ==="
