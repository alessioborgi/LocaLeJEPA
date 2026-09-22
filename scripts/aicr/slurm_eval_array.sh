#!/bin/bash
# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# One evaluation per array task, over a GENERATED eval config and its names file
# (scripts/aicr/make_ablation_eval_config.py). Task i evaluates line i+1 of NAMES, so the two
# orders are the same by construction and a test pins them together. Used to re-read the ablation
# checkpoints with the draft's probes (2026-09-22, state/10-PAPER-STATE.md); slurm_eval.sh stays
# the single-backbone path, whose config lives inside the clone.
#
#   sbatch -A <account> -p <measured> --array=0-N%32 --nodes=1 --ntasks-per-node=1 -c 8 \
#          --mem=60G --gpus-per-node=1 -t 00:30:00 -o <logs>/%x-%A_%a.out \
#          --export=NONE,REPO=<clone>,EVAL_CONFIG=<abs cfg>,NAMES=<abs names>,EVAL_OUT=<abs out dir> \
#          <clone>/scripts/aicr/slurm_eval_array.sh
set -euo pipefail
# One check per line, no quotes inside the ${VAR:?message}: bash parses that word with its own
# quoting rules and an apostrophe in the message is an unterminated string.
: "${REPO:?REPO must be exported}"
: "${EVAL_CONFIG:?EVAL_CONFIG must be exported: an absolute path to the generated eval config}"
: "${NAMES:?NAMES must be exported: an absolute path to the backbone names file}"
: "${EVAL_OUT:?EVAL_OUT must be exported: the output_folder the config names}"
INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set: this script is an array task}"
[ -f "$REPO/scripts/aicr/roots.env" ] || { echo "missing $REPO/scripts/aicr/roots.env" >&2; exit 2; }
# shellcheck disable=SC1091
source "$REPO/scripts/aicr/roots.env"
[ "${PY:-CHANGE_ME}" != "CHANGE_ME" ] || { echo "roots.env still has PY=CHANGE_ME" >&2; exit 2; }
export SLURM_EXPORT_ENV=ALL WANDB_OFFLINE=true HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

NAME="$(sed -n "$((INDEX + 1))p" "$NAMES")"
[ -n "$NAME" ] || { echo "no backbone on line $((INDEX + 1)) of $NAMES: the array is wider than the names file" >&2; exit 2; }
echo "=== $(date -Is) | eval task $INDEX -> $NAME | config $EVAL_CONFIG | host $(hostname) ==="

set +e
srun --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
    env PYTHONPATH="$REPO" "$PY" "$REPO/src/eval_indomain.py" --config "$EVAL_CONFIG" --only "$NAME"
RC=$?
set -e
echo "eval exit code $RC (informational: the artifact decides)"
# A backbone whose checkpoint is missing is SKIPPED by the eval with a logged error and exit 0.
# The per-backbone JSON is the only evidence the work happened (research_rules/05).
RESULT="$EVAL_OUT/per_backbone/$NAME.json"
[ -f "$RESULT" ] || { echo "=== EVAL FAILED $NAME: no $RESULT ===" >&2; exit 1; }
echo "=== EVAL DONE $NAME $(date -Is) ==="
