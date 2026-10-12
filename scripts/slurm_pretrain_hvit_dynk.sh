#!/bin/bash
#SBATCH --job-name=BLJ_hvit_dynk_resume
#SBATCH --partition=gpu_test
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa
export PROJECT_ROOT=$PWD WANDB_START_METHOD=thread PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CKPT_DIR="$PWD/checkpoints/lejepa_hvit_in1k_localballreg_l01_dynk"
SELF=scripts/slurm_pretrain_hvit_dynk.sh
RESUME=""; [ -f "$CKPT_DIR/last.ckpt" ] && RESUME="ckpt_path=$CKPT_DIR/last.ckpt"
# drop stale pending copies of this job, then queue ONE successor (afterany) to
# continue past the 12h gpu_test limit; cancelled below on clean completion.
for o in $(squeue -u "$(whoami)" -h -t PENDING -o "%i|%j" | awk -F'|' -v n="$SLURM_JOB_NAME" '$2==n{print $1}'); do
  [ "$o" != "$SLURM_JOB_ID" ] && scancel "$o" 2>/dev/null || true
done
SUCC=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID "$SELF" 2>/dev/null || echo "")
echo "=== dynk resume $(date) on $(hostname); successor ${SUCC:-none (submit limit)} ==="
set +e
python src/train.py experiment=lejepa/hvit_in1k_localballreg_l01_dynk \
  trainer.max_epochs=100 logger=csv \
  "data.data_dir=$PWD/data/imagenet" data.batch_size=32 \
  trainer.accumulate_grad_batches=8 \
  "callbacks.model_checkpoint.dirpath=$CKPT_DIR" $RESUME
STATUS=$?
set -e
if [ $STATUS -eq 0 ]; then
  echo "=== reached max_epochs: cancel successor, launch Yoda eval ==="
  [ -n "$SUCC" ] && scancel "$SUCC" 2>/dev/null || true
  sbatch scripts/slurm_yoda_hvit_dynk.sh
else
  A="$CKPT_DIR/.crash_attempts"; n=$(( $(cat "$A" 2>/dev/null || echo 0) + 1 )); echo $n > "$A"
  if [ $n -gt 6 ]; then echo "too many crashes; stopping"; [ -n "$SUCC" ] && scancel "$SUCC" 2>/dev/null || true; exit $STATUS; fi
  echo "=== exit $STATUS (crash #$n); successor $SUCC will resume from last.ckpt ==="
fi
echo "=== end $(date) ==="
