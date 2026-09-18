#!/bin/bash
#SBATCH --job-name=localejepa
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# Canonical SLURM launcher for LocaLeJEPA. Everything runs through `uv run`.
#
# Two modes, selected by environment variables (pass them with --export):
#
#   EXPERIMENT=<hydra experiment>   pretraining via src/train.py, e.g.
#                                   EXPERIMENT=lejepa/vit_in1k_localballreg_l01_k4
#   CMD="<shell command>"           anything else (evaluations, report scripts), e.g.
#                                   CMD="python src/eval_indomain.py --config configs/eval/indomain_in1k_ep100.yaml --only vit_sigreg"
#
# Optional variables (pretraining mode):
#   GPUS=N          number of GPUs (default 1). N>1 launches one process per GPU with
#                   srun and trainer=ddp; request matching resources on the sbatch line:
#                   sbatch --gres=gpu:4 --ntasks-per-node=4 --mem=256G --export=ALL,EXPERIMENT=...,GPUS=4 ...
#   RUN_NAME=...    W&B run name + checkpoint dir name (default: EXPERIMENT with '/' -> '_')
#   CKPT_DIR=...    where best.ckpt / last.ckpt are written (default: checkpoints/$RUN_NAME)
#   REQUEUE=1       queue a successor job (afterany) that resumes from last.ckpt if this
#                   allocation ends before max_epochs; cancelled automatically on success.
#   OVERRIDES="..." extra Hydra overrides, e.g. OVERRIDES="logger=csv trainer.max_epochs=10"
#
# Optional variables:
#   DATA_DIR=...    dataset root (default: <repo>/data). Passed as paths.data_dir (pretraining mode).
#   ENV_SETUP=...   shell file to source first (module loads, `export PATH=$HOME/.local/bin:$PATH`, ...)
#
# The job always resumes from $CKPT_DIR/last.ckpt when it exists, so a timed-out or
# preempted run can simply be resubmitted with the same EXPERIMENT / RUN_NAME.
#
# Examples:
#   sbatch --export=ALL,EXPERIMENT=lejepa/vit_in1k_sigreg scripts/slurm_submit.sh
#   sbatch --export=ALL,EXPERIMENT=lejepa/vit_in1k_localballreg_l01_k4,REQUEUE=1 scripts/slurm_submit.sh
#   sbatch --gres=gpu:4 --ntasks-per-node=4 --mem=256G \
#          --export=ALL,EXPERIMENT=lejepa/vitl14_in1k_localballreg_l01_k4_paper,GPUS=4,REQUEUE=1 scripts/slurm_submit.sh
#   sbatch --time=6:00:00 --mem=128G --cpus-per-task=16 \
#          --export=ALL,CMD="python src/eval_indomain.py --config configs/eval/indomain_in1k_ep100.yaml --only vit_sigreg" scripts/slurm_submit.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
mkdir -p logs/slurm

[ -n "${ENV_SETUP:-}" ] && source "$ENV_SETUP"
command -v uv >/dev/null 2>&1 || { echo "uv not found on PATH (install: curl -LsSf https://astral.sh/uv/install.sh | sh)"; exit 1; }

export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_OVERRIDE=()
[ -n "${DATA_DIR:-}" ] && DATA_OVERRIDE=("paths.data_dir=${DATA_DIR}")

echo "=== $(date) | host $(hostname) | job ${SLURM_JOB_ID:-local} ==="
echo "uv $(uv --version) | python: $(uv run --no-sync python -c 'import sys; print(sys.version.split()[0])')"

# ---------------------------------------------------------------- generic command
if [ -n "${CMD:-}" ]; then
  echo "CMD: $CMD"
  # shellcheck disable=SC2086
  uv run --no-sync $CMD
  echo "=== end $(date) ==="
  exit 0
fi

# ---------------------------------------------------------------- pretraining
: "${EXPERIMENT:?set EXPERIMENT=<hydra experiment> (e.g. lejepa/vit_in1k_sigreg) or CMD=...}"
GPUS="${GPUS:-1}"
RUN_NAME="${RUN_NAME:-${EXPERIMENT//\//_}}"
CKPT_DIR="${CKPT_DIR:-$PWD/checkpoints/${RUN_NAME}}"
mkdir -p "$CKPT_DIR"

RESUME=()
[ -f "${CKPT_DIR}/last.ckpt" ] && RESUME=("ckpt_path=${CKPT_DIR}/last.ckpt")

# Self-requeue: a successor with afterany resumes from last.ckpt if this allocation
# ends early; it is cancelled below when training reaches max_epochs.
SUCCESSOR=""
if [ "${REQUEUE:-0}" = "1" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  SUCCESSOR=$(sbatch --parsable --dependency=afterany:"${SLURM_JOB_ID}" \
    --job-name="${SLURM_JOB_NAME}" --export=ALL "$(readlink -f "$0")")
  echo "queued successor ${SUCCESSOR} (afterany:${SLURM_JOB_ID})"
fi

ARGS=(src/train.py "experiment=${EXPERIMENT}" "callbacks.model_checkpoint.dirpath=${CKPT_DIR}")
# Name the W&B run after RUN_NAME unless the logger is switched (e.g. OVERRIDES="logger=csv").
[[ "${OVERRIDES:-}" == *"logger="* ]] || ARGS+=("logger.wandb.name=${RUN_NAME}")
# shellcheck disable=SC2206
ARGS+=(${OVERRIDES:-})
ARGS+=(${DATA_OVERRIDE[@]+"${DATA_OVERRIDE[@]}"} ${RESUME[@]+"${RESUME[@]}"})

echo "experiment: ${EXPERIMENT} | run: ${RUN_NAME} | gpus: ${GPUS} | ckpt: ${CKPT_DIR}"
set +e
if [ "$GPUS" -gt 1 ]; then
  # One process per GPU; Lightning's SLURMEnvironment reads SLURM_NTASKS/PROCID/LOCALID.
  srun --ntasks="$GPUS" --ntasks-per-node="$GPUS" --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    uv run --no-sync python "${ARGS[@]}" trainer=ddp "trainer.devices=${GPUS}" trainer.num_nodes=1
else
  uv run --no-sync python "${ARGS[@]}"
fi
STATUS=$?
set -e

if [ $STATUS -eq 0 ] && [ -n "$SUCCESSOR" ]; then
  echo "finished: cancelling successor ${SUCCESSOR}"
  scancel "$SUCCESSOR" 2>/dev/null || true
fi
echo "=== end $(date) | status $STATUS ==="
exit $STATUS
