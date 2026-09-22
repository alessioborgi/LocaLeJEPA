#!/usr/bin/env bash
#SBATCH --job-name=fixed-k
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-179%1
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# Prefer: bash scripts/submit_fixed_k_20260922.sh
# It splits B64 (40G) and B256 (80G), preserving these global task indices.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
index="${SLURM_ARRAY_TASK_ID:-${1:-}}"
[[ "$index" =~ ^[0-9]+$ ]] && (( index < 180 )) || { echo 'Index must be 0..179' >&2; exit 1; }
lambdas=(0.02 0.05 0.1 0.2 0.3)
clusters=(1 2 4 8 16 32)
batches=(64 256)
seeds=(42 43 44)
seed=${seeds[$((index % 3))]}
batch=${batches[$((index / 3 % 2))]}
k=${clusters[$((index / 6 % 6))]}
lamb=${lambdas[$((index / 36))]}
accum=$((256 / batch))
unset CMD
export EXPERIMENT=ablations/fixed_k_20260922
export RUN_NAME="${RUN_PREFIX:-fixed_k_20260922}_l${lamb}_k${k}_B${batch}x${accum}_seed${seed}"
export CKPT_DIR="$PWD/checkpoints/$RUN_NAME"
export GPUS=1 REQUEUE=0
export OVERRIDES="model.lamb=$lamb model.reg_loss.num_clusters=$k data.batch_size=$batch trainer.accumulate_grad_batches=$accum seed=$seed trainer.devices=1 hydra.run.dir=$PWD/logs/ablations/$RUN_NAME"
echo "$index $RUN_NAME $OVERRIDES"
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0
exec bash scripts/slurm_submit.sh
