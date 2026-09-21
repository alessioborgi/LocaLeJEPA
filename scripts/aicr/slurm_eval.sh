#!/bin/bash
# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# One evaluation of one backbone, on AICR. Pre-flights the checkpoint the eval config names for
# `--only` -- it must exist AND carry the launcher's `.verified` marker (plan G5/V4) -- before a
# single feature is extracted, so an eval can never quietly run on a half-trained or
# never-verified checkpoint. Chained to its pretrain with --dependency=aftercorr (runbook §7).
#
#   sbatch -A <account> -p <measured> --gpus-per-node=1 --ntasks-per-node=1 -c 16 --mem=60G \
#          -t 03:00:00 -o <logs>/eval-%j.out \
#          --export=NONE,REPO=<clone>,EVAL_SCRIPT=src/eval_indomain.py,EVAL_CONFIG=configs/eval/<cfg>.yaml,ONLY=<backbone>,EVAL_ROOT=<run root> \
#          <clone>/scripts/aicr/slurm_eval.sh
set -euo pipefail
# One check per line, no quotes inside the ${VAR:?message}: bash parses that word with its own
# quoting rules and an apostrophe in the message is an unterminated string.
: "${REPO:?REPO must be exported}"
: "${EVAL_SCRIPT:?EVAL_SCRIPT must be exported, e.g. src/eval_indomain.py}"
: "${EVAL_CONFIG:?EVAL_CONFIG must be exported, e.g. configs/eval/indomain_in1k_ep100.yaml}"
: "${ONLY:?ONLY must be exported: the backbone name from the eval config}"
: "${EVAL_ROOT:?EVAL_ROOT must be exported: the directory the eval config resolves its relative paths against}"
[ -f "$REPO/scripts/aicr/roots.env" ] || { echo "missing $REPO/scripts/aicr/roots.env" >&2; exit 2; }
# shellcheck disable=SC1091
source "$REPO/scripts/aicr/roots.env"
[ "${PY:-CHANGE_ME}" != "CHANGE_ME" ] || { echo "roots.env still has PY=CHANGE_ME" >&2; exit 2; }
export SLURM_EXPORT_ENV=ALL WANDB_OFFLINE=true HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
CAMPAIGN="$PY $REPO/scripts/aicr/campaign.py"

echo "=== $(date -Is) | eval $EVAL_SCRIPT $ONLY | config $EVAL_CONFIG | root $EVAL_ROOT | host $(hostname) ==="
$CAMPAIGN eval-preflight --config "$REPO/$EVAL_CONFIG" --only "$ONLY" --root "$EVAL_ROOT"
cd "$EVAL_ROOT"
set +e
srun --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
    env PYTHONPATH="$REPO" "$PY" "$REPO/$EVAL_SCRIPT" --config "$REPO/$EVAL_CONFIG" --only "$ONLY"
RC=$?
set -e
echo "eval exit code $RC (informational)"
$CAMPAIGN eval-postflight --config "$REPO/$EVAL_CONFIG" --only "$ONLY" --root "$EVAL_ROOT"
echo "=== EVAL DONE $ONLY $(date -Is) ==="
