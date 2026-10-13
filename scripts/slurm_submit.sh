#!/bin/bash
#SBATCH --job-name=localejepa
#SBATCH --partition=gpu                 # EDIT: your GPU partition (add --account=... if your site needs one)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=23:50:00                 # EDIT: at most your site's limit; REQUEUE=1 chains jobs past it
#SBATCH --no-requeue                    # a site requeue would start a second copy next to the successor
#SBATCH --output=logs/slurm/%x_%j.out   # submit from the repo root, after `mkdir -p logs/slurm`
#SBATCH --error=logs/slurm/%x_%j.err
#
# Example SLURM launcher for LocaLeJEPA. Submit it from the repository root. Commands run through
# `uv run` when uv is on PATH, otherwise with the `python` of the environment set up by ENV_SETUP.
#
# Three modes, selected by environment variables (pass them with --export=ALL,...):
#
#   EXPERIMENT=<config>   pretrain one config of configs/experiment/, e.g. lejepa/vit_in1k_localballreg_l01_k4
#   GRID=<grid.json>      pretrain one run of a campaign grid (INDEX defaults to the array task id), made with
#                         python scripts/aicr/campaign.py expand configs/campaign/<name>.yaml --out grid.json
#   CMD="<command>"       run anything else, e.g. an evaluation
#
# Optional variables:
#   GPUS=N          GPUs for this run (default 1; in GRID mode, the run's own count). N>1 launches one process
#                   per GPU with srun and trainer=ddp; request the same on the sbatch line:
#                   sbatch --gres=gpu:4 --ntasks-per-node=4 --mem=256G --export=ALL,...,GPUS=4 ...
#   RUN_NAME=...    W&B run name and checkpoint directory name (default: EXPERIMENT with '/' -> '_')
#   CKPT_DIR=...    where best.ckpt, last.ckpt and on_exception.ckpt go (default: checkpoints/$RUN_NAME)
#   REQUEUE=1       queue a successor (same resources, afterany) that resumes from last.ckpt when this job
#                   ends before training does; it is cancelled when training finishes, or when this job
#                   failed without writing a newer last.ckpt (so a crashing run does not loop).
#   OVERRIDES="..." extra Hydra overrides, e.g. OVERRIDES="logger=csv trainer.max_epochs=10"
#   DATA_DIR=...    dataset root (default: <repo>/data/), passed as paths.data_dir
#   ENV_SETUP=...   shell file sourced first, e.g. one containing
#                   eval "$(conda shell.bash hook)"; conda activate localejepa
#   USE_UV=0        ignore uv even if it is on PATH and use the active environment
#
# A job always resumes from $CKPT_DIR/last.ckpt when it exists, so a run that timed out or was
# preempted can simply be resubmitted with the same EXPERIMENT (or RUN_NAME).
#
# Examples:
#   sbatch --export=ALL,EXPERIMENT=lejepa/vit_in1k_localballreg_l01_k4,REQUEUE=1 scripts/slurm_submit.sh
#   sbatch --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=16 --mem=512G \
#          --export=ALL,EXPERIMENT=lejepa/vitl14_in1k_localballreg_paper,GPUS=8,REQUEUE=1 scripts/slurm_submit.sh
#   sbatch --array=0-10 --export=ALL,GRID=torus.json,REQUEUE=1 scripts/slurm_submit.sh
#   sbatch --time=6:00:00 --mem=128G --cpus-per-task=16 --export=ALL,CMD="python src/eval_indomain.py \
#          --config configs/eval/indomain_in1k_plainvit.yaml --only vit_localballreg_l01_k4" scripts/slurm_submit.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
mkdir -p logs/slurm

if [ -n "${ENV_SETUP:-}" ]; then set +u; source "$ENV_SETUP"; set -u; fi   # conda's hooks read unset variables
if [ "${USE_UV:-1}" != "0" ] && command -v uv >/dev/null 2>&1; then RUN=(uv run --no-sync); else RUN=(); fi
py() { ${RUN[@]+"${RUN[@]}"} python "$@"; }

export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== $(date) | host $(hostname) | job ${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+[$SLURM_ARRAY_TASK_ID]} ==="
echo "runner: ${RUN[*]:-active env} | python $(py -c 'import sys; print(sys.version.split()[0])')"

# ---------------------------------------------------------------- any command
if [ -n "${CMD:-}" ]; then
  echo "CMD: $CMD"
  # shellcheck disable=SC2086
  ${RUN[@]+"${RUN[@]}"} $CMD
  echo "=== end $(date) ==="
  exit 0
fi

# ---------------------------------------------------------------- campaign grid
# The run's name, experiment, seed, epochs and GPU count come from the grid; OVERRIDES is appended.
# Nothing exported is rewritten, so a successor job re-derives exactly the same run.
RUN_OVERRIDES=()
if [ -n "${GRID:-}" ]; then
  export INDEX="${INDEX:-${SLURM_ARRAY_TASK_ID:?set INDEX or submit with --array}}"
  CELL="$(py scripts/aicr/campaign.py show "$GRID" --index "$INDEX")"
  field() { py -c 'import json, sys; print(json.loads(sys.argv[1])[sys.argv[2]])' "$CELL" "$1"; }
  EXPERIMENT="$(field experiment)"
  RUN_NAME="${RUN_NAME:-$(field run_name)}"
  GPUS="${GPUS:-$(field gpus)}"
  USER_OVERRIDES="${OVERRIDES:-}"
  eval "$(py scripts/aicr/campaign.py args "$GRID" --index "$INDEX")"   # exports OVERRIDES, MAX_EPOCHS, RUN_NAME
  set -f
  # shellcheck disable=SC2206
  for item in $OVERRIDES; do [[ "$item" == experiment=* ]] || RUN_OVERRIDES+=("$item"); done
  set +f
  RUN_OVERRIDES+=("trainer.max_epochs=${MAX_EPOCHS}")
  export OVERRIDES="$USER_OVERRIDES"       # restored before the successor inherits the environment
fi

# ---------------------------------------------------------------- pretraining
: "${EXPERIMENT:?set EXPERIMENT=<config>, GRID=<grid.json> or CMD=<command>}"
GPUS="${GPUS:-1}"
RUN_NAME="${RUN_NAME:-${EXPERIMENT//\//_}}"
CKPT_DIR="${CKPT_DIR:-$PWD/checkpoints/${RUN_NAME}}"
mkdir -p "$CKPT_DIR"
LAST="${CKPT_DIR}/last.ckpt"
last_mtime() { stat -c %Y "$LAST" 2>/dev/null || echo 0; }
BEFORE="$(last_mtime)"

RESUME=()
[ -f "$LAST" ] && RESUME=("ckpt_path=${LAST}")

SUCCESSOR=""
if [ "${REQUEUE:-0}" = "1" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  # Same resources as this job, which may have been raised on the sbatch command line.
  SUCCESSOR=$(sbatch --parsable --dependency=afterany:"${SLURM_JOB_ID}" --job-name="${SLURM_JOB_NAME}" \
    --partition="${SLURM_JOB_PARTITION}" --nodes=1 --ntasks-per-node="${GPUS}" --gres=gpu:"${GPUS}" \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" ${SLURM_MEM_PER_NODE:+--mem=${SLURM_MEM_PER_NODE}M} \
    --time="$(squeue -h -j "${SLURM_JOB_ID}" -o %l)" --export=ALL "$(readlink -f "$0")")
  echo "queued successor ${SUCCESSOR} (afterany:${SLURM_JOB_ID})"
fi

ARGS=(src/train.py "experiment=${EXPERIMENT}" "paths.ckpt_dir=${CKPT_DIR}")
[ -n "${DATA_DIR:-}" ] && ARGS+=("paths.data_dir=${DATA_DIR}")
ARGS+=(${RUN_OVERRIDES[@]+"${RUN_OVERRIDES[@]}"})
set -f
# shellcheck disable=SC2206
ARGS+=(${OVERRIDES:-})
set +f
# Name the W&B run after RUN_NAME unless another logger was chosen (e.g. logger=csv).
[[ " ${ARGS[*]} " == *" logger="* ]] || ARGS+=("logger.wandb.name=${RUN_NAME}")
ARGS+=(${RESUME[@]+"${RESUME[@]}"})

echo "experiment: ${EXPERIMENT} | run: ${RUN_NAME} | gpus: ${GPUS} | ckpt: ${CKPT_DIR}"
set +e
if [ "$GPUS" -gt 1 ]; then
  # One process per GPU; Lightning's SLURMEnvironment reads SLURM_NTASKS/PROCID/LOCALID.
  srun --ntasks="$GPUS" --ntasks-per-node="$GPUS" --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    ${RUN[@]+"${RUN[@]}"} python "${ARGS[@]}" trainer=ddp "trainer.devices=${GPUS}" trainer.num_nodes=1
else
  py "${ARGS[@]}"
fi
STATUS=$?
set -e

if [ -n "$SUCCESSOR" ]; then
  if [ $STATUS -eq 0 ]; then
    echo "finished: cancelling successor ${SUCCESSOR}"; scancel "$SUCCESSOR" 2>/dev/null || true
  elif [ "$(last_mtime)" = "$BEFORE" ]; then
    echo "failed without a newer last.ckpt: cancelling successor ${SUCCESSOR} instead of retrying"
    scancel "$SUCCESSOR" 2>/dev/null || true
  fi
fi
echo "=== end $(date) | status $STATUS ==="
exit $STATUS
