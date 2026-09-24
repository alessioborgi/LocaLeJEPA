#!/usr/bin/env bash
#SBATCH --job-name=ramp-B256
#SBATCH --partition=GPU
#SBATCH --nodelist=diufrd203
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --array=0%1
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

# Submit from repo root after: mkdir -p logs/slurm
# Default PILOT=1: 2 full epochs, K=2 then K=16; memory test only.
# Full 30-run campaign (after pilot succeeds):
# sbatch --array=0-29%2 --time=1-00:00:00 --export=ALL,PILOT=0 scripts/slurm_ramp_b256_20260922.sh
# Index: seed fastest, then direction, then lambda. Original fixed-K jobs untouched.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
index="${SLURM_ARRAY_TASK_ID:-${1:-0}}"
[[ "$index" =~ ^[0-9]+$ ]] && (( index < 30 )) || { echo 'Index must be 0..29' >&2; exit 1; }
pilot="${PILOT:-1}"
[[ "$pilot" == 0 || "$pilot" == 1 ]] || { echo 'PILOT must be 0 or 1' >&2; exit 1; }
lambdas=(0.02 0.05 0.1 0.2 0.3)
seeds=(42 43 44)
lamb=${lambdas[$((index / 6))]}
seed=${seeds[$((index % 3))]}
if (( index / 3 % 2 == 0 )); then direction=fwd; start=2; end=16; else direction=rev; start=16; end=2; fi
unset CMD
export EXPERIMENT=ablations/ramp_b256_20260922
export GPUS=1 REQUEUE=0
prefix="${RUN_PREFIX:-ramps_b256_20260922}"
extra=""
if [[ "$pilot" == 1 ]]; then
  # Keep full training/validation batches and accumulation; shorten only schedule/duration.
  # Isolate pilot outputs so full runs never resume these altered-schedule checkpoints.
  prefix="${prefix}_pilot_${SLURM_JOB_ID:-local}"
  extra="trainer.max_epochs=2 callbacks.dynamic_num_clusters.step_epochs=1 callbacks.dynamic_num_clusters.step=14 logger.wandb.group=ramps_b256_20260922_pilot"
fi
export RUN_NAME="${prefix}_l${lamb}_${direction}_B256x1_seed${seed}"
export CKPT_DIR="$PWD/checkpoints/$RUN_NAME"
export OVERRIDES="model.lamb=$lamb seed=$seed model.reg_loss.num_clusters=$start callbacks.dynamic_num_clusters.start=$start callbacks.dynamic_num_clusters.end=$end trainer.devices=1 hydra.run.dir=$PWD/logs/ablations/$RUN_NAME $extra"
echo "index=$index pilot=$pilot run=$RUN_NAME $OVERRIDES"
[[ "${DRY_RUN:-0}" == 1 ]] && exit 0
exec bash scripts/slurm_submit.sh
