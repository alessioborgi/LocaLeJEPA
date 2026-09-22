#!/bin/bash
# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# One pretraining run of a campaign grid, on AICR. Thin on purpose: every decision is made by
# scripts/aicr/campaign.py, which is unit-tested, and this file only sequences them.
#
# NO #SBATCH RESOURCE DIRECTIVES. Resources come from `campaign.py submit` on the command line
# and a successor re-reads them from the running job's SLURM_* variables. The FASRC launcher kept
# them in its header, and its successor reverted a 4-GPU run to 1 GPU that way (F32, F33).
#
# Submitted by the line `campaign.py submit` prints, which carries --export=NONE plus:
#   GRID          the expanded grid (JSON) this array reads
#   REPO          the clone to run from (BASH_SOURCE is the spool copy under sbatch, so it is
#                 not usable for self-location)
#   REQUEUE_LEFT  how many successors may still be submitted (default 2)
#   TIME_LIMIT    the --time this job was given, so a successor asks for the same
#   GRID_INDEX    only on a successor: the array task it continues
#   CKPT_ROOT_OVERRIDE  optional: this submission's checkpoint root (ablation waves -> /scratch)
# Site roots come from $REPO/scripts/aicr/roots.env (git-ignored; template beside it).
#
# Sequence, and what each step refuses on:
#   preflight   roots present -> tree clean and at the grid's commit -> env hash == lock ->
#               geometry printed and per-GPU micro-batch asserted -> resume decision from the
#               step INSIDE the checkpoints (done => exit 0 without training)
#   staging     ImageNet-1K: /tmp free space >= 1.2x the tar, tar -xf, structural verification
#               with a decode sample; Imagenette: $DATA_ROOT directly
#   training    srun, one task per GPU (Lightning's SLURMEnvironment requires
#               --ntasks-per-node == devices), SIGTERM delivered to every rank by --signal
#   postflight  artifact epoch == max_epochs-1, .hydra config == manifest, checkpoint hparams
#               == config, metrics.csv complete, tree, env  ->  .verified marker, or
#               RESUMABLE -> a successor with THIS job's resources, or FAILED
set -euo pipefail

: "${GRID:?GRID=<grid.json> must be exported by campaign.py submit}"
: "${REPO:?REPO=<clone> must be exported by campaign.py submit}"
REQUEUE_LEFT="${REQUEUE_LEFT:-2}"
INDEX="${SLURM_ARRAY_TASK_ID:-${GRID_INDEX:?neither SLURM_ARRAY_TASK_ID nor GRID_INDEX is set}}"

[ -f "$REPO/scripts/aicr/campaign.py" ] || { echo "REPO=$REPO does not contain scripts/aicr/campaign.py" >&2; exit 2; }
[ -f "$REPO/scripts/aicr/roots.env" ] || { echo "missing $REPO/scripts/aicr/roots.env -- copy roots.env.example and fill it in" >&2; exit 2; }
# shellcheck disable=SC1091
source "$REPO/scripts/aicr/roots.env"
for name in PY CKPT_ROOT LOG_ROOT DATA_ROOT ENV_LOCK; do
    [ "${!name:-CHANGE_ME}" != "CHANGE_ME" ] || { echo "roots.env still has $name=CHANGE_ME" >&2; exit 2; }
done
# Per-submission checkpoint root: `campaign.py submit --export CKPT_ROOT_OVERRIDE=<dir>` sends an
# ablation wave's ~150 GB to /scratch (decision 14) WITHOUT editing roots.env between submissions:
# a task reads roots.env when it STARTS, so a file edited after submission would route the
# late-starting tasks of the same array to the other root. The successor carries it (below).
CKPT_ROOT="${CKPT_ROOT_OVERRIDE:-$CKPT_ROOT}"

cd "$REPO"
export SLURM_EXPORT_ENV=ALL          # steps inherit THIS script's environment, not sbatch's --export=NONE
export WANDB_OFFLINE=true HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
GPUS="${SLURM_NTASKS_PER_NODE:-1}"
CAMPAIGN="$PY $REPO/scripts/aicr/campaign.py"

echo "=== $(date -Is) | job ${SLURM_JOB_ID:-?}${SLURM_ARRAY_TASK_ID:+[$SLURM_ARRAY_TASK_ID]} | host $(hostname) | gpus $GPUS | requeue_left $REQUEUE_LEFT ==="
$CAMPAIGN show "$GRID" --index "$INDEX"

# ----------------------------------------------------------------------------- preflight
eval "$($CAMPAIGN args "$GRID" --index "$INDEX")"           # RUN_NAME, MAX_EPOCHS, OVERRIDES
if [ "${MODE:-train}" = "calibrate" ]; then
    # Calibration artefacts are disposable, and one geometry is calibrated more than once at a
    # time (two partitions; later the loader lever): key their run and checkpoint dirs by
    # partition and job id, or two jobs share one run dir and calibrate-verdict sees four
    # launches -- research_rules/05's live-duplicate trap, the same one as the double tar (F44).
    RUN_NAME="${RUN_NAME}-calib-${SLURM_JOB_PARTITION:-local}-${SLURM_JOB_ID:-$$}"
fi
RUN_DIR="$LOG_ROOT/train/runs/$RUN_NAME"
CKPT_DIR="$CKPT_ROOT/$RUN_NAME"
mkdir -p "$RUN_DIR" "$CKPT_DIR"

$CAMPAIGN preflight "$GRID" --index "$INDEX" --lock "$ENV_LOCK" --python "$PY"
eval "$($CAMPAIGN geometry "$GRID" --index "$INDEX" --world-size "$GPUS" --shell)"   # GEO_*
$CAMPAIGN geometry "$GRID" --index "$INDEX" --world-size "$GPUS"
eval "$($CAMPAIGN resume --ckpt-dir "$CKPT_DIR" --max-epochs "$MAX_EPOCHS" --shell)"   # DECISION, RESUME, STALE
# an unreadable checkpoint left in place would be auto-loaded by Lightning on a fresh start (F50)
for f in ${STALE:-}; do mv "$f" "$f.unreadable.$(date +%s)"; echo "moved unreadable $f aside"; done
if [ "$DECISION" = "done" ]; then
    echo "already complete: $CKPT_DIR/last.ckpt is at epoch $((MAX_EPOCHS - 1)); nothing to do"
    exit 0
fi

# ----------------------------------------------------------------------------- staging
TMP="/tmp/localejepa_${SLURM_JOB_ID:-$$}"
cleanup() { rm -rf "$TMP"; }
trap cleanup EXIT
# A SIGTERM that reaches THIS script is a cancel (scancel signals the batch step too) or a
# preemption (which Slurm requeues by itself); the wall's --signal=TERM@120 goes to the srun
# tasks only. So a signalled script must not submit a successor -- the first wave's cancel spawned
# eighteen of them (F50). Bash runs the trap after the foreground srun returns.
SIGNALLED=0; TIME_LEFT=unknown
on_term() { SIGNALLED=1; TIME_LEFT=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null || echo unknown); }
trap on_term TERM
find /tmp -maxdepth 1 -name 'localejepa_*' -user "$USER" -mmin +2880 -exec rm -rf {} + 2>/dev/null || true

DATA_OVERRIDE=""
if [ "${GEO_DATASET:-}" = "imagenet1k" ]; then
    : "${DATA_TAR:?this run needs ImageNet-1K but roots.env has no DATA_TAR}"
    [ -f "$DATA_TAR" ] || { echo "DATA_TAR=$DATA_TAR does not exist" >&2; exit 2; }
    $CAMPAIGN stage-check --free "$(df -B1 --output=avail /tmp | tail -1)" --tar-bytes "$(stat -c %s "$DATA_TAR")"
    mkdir -p "$TMP"
    echo "staging $DATA_TAR -> $TMP"
    time tar -xf "$DATA_TAR" -C "$TMP"
    time "$PY" scripts/verify_imagenet_staging.py --root "$TMP/imagenet_full_size/061417" --decode 64
    DATA_OVERRIDE="data.data_dir=$TMP"
else
    DATA_OVERRIDE="paths.data_dir=$DATA_ROOT"
fi

# ----------------------------------------------------------------------------- training
# The trainer, the CSV logger and the loader workers come from campaign.py (launch-overrides), the
# calibrate set below from calibrate-overrides: every override this launcher adds is composed by a
# test against every campaign cell (F47: a shell-composed override without its `+` killed nine
# calibration tasks at launch and no test could see it).
LAUNCH=($($CAMPAIGN launch-overrides --gpus "$GPUS" --cpus "${SLURM_CPUS_PER_TASK:-16}"))
RESUME_OVERRIDE=""
[ -n "${RESUME:-}" ] && RESUME_OVERRIDE="ckpt_path=$RESUME"
COMMON=("$PY" src/train.py $OVERRIDES "${LAUNCH[@]}" "hydra.run.dir=$RUN_DIR"
        "callbacks.model_checkpoint.dirpath=$CKPT_DIR" "callbacks.last_checkpoint.dirpath=$CKPT_DIR"
        "callbacks.on_exception.dirpath=$CKPT_DIR" "$DATA_OVERRIDE")
SRUN=(srun --ntasks-per-node="$GPUS" --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" --kill-on-bad-exit=1)

if [ "${MODE:-train}" = "calibrate" ]; then
    # Plan Phase 2 / V2: two launches under one run dir. The first prints the geometry (above),
    # measures throughput and the data-wait fraction, and SIGTERMs itself at CALIB_INTERRUPT --
    # the real path: Lightning's handler, SIGTERMException, OnExceptionCheckpoint. The second
    # must resume from that checkpoint at the very next step and reach CALIB_STEPS. No artifact
    # postflight: no epoch boundary is reached, so there is no last.ckpt to verify.
    CALIB_STEPS="${CALIB_STEPS:-200}"; CALIB_INTERRUPT="${CALIB_INTERRUPT:-150}"
    # the calibrate set (per-step LR rows and CSV rows, max_steps, the rehearsed SIGTERM) is
    # campaign.py's, composed by the manifests test against every cell (F46, F47)
    FIRST=($($CAMPAIGN calibrate-overrides --max-steps "$CALIB_STEPS" --interrupt "$CALIB_INTERRUPT"))
    CALIB=($($CAMPAIGN calibrate-overrides --max-steps "$CALIB_STEPS"))
    echo "=== calibrating $RUN_NAME: $CALIB_STEPS steps, interruption at $CALIB_INTERRUPT ==="
    set +e
    "${SRUN[@]}" "${COMMON[@]}" "${FIRST[@]}"
    RC1=$?
    set -e
    eval "$($CAMPAIGN resume --ckpt-dir "$CKPT_DIR" --max-epochs "$MAX_EPOCHS" --shell)"
    if [ "$DECISION" != "resume" ]; then
        echo "CALIBRATION FAILED: no resumable checkpoint after the interruption (first launch exit $RC1, decision $DECISION)" >&2
        exit 1
    fi
    set +e
    "${SRUN[@]}" "${COMMON[@]}" "${CALIB[@]}" "ckpt_path=$RESUME"
    RC2=$?
    set -e
    echo "launch exit codes: $RC1 (interrupted), $RC2 (resumed) -- informational"
    $CAMPAIGN calibrate-verdict --run-dir "$RUN_DIR" --interrupt-step "$CALIB_INTERRUPT" --max-steps "$CALIB_STEPS"
    exit $?
fi

echo "=== training $RUN_NAME ($DECISION${RESUME:+ from $RESUME}) ==="
set +e
# shellcheck disable=SC2086
"${SRUN[@]}" "${COMMON[@]}" $RESUME_OVERRIDE
RC=$?
set -e
echo "train.py exit code $RC (informational: the artifact decides -- research_rules/01)"

# ----------------------------------------------------------------------------- postflight
set +e
$CAMPAIGN postflight "$GRID" --index "$INDEX" --run-dir "$RUN_DIR" --ckpt-dir "$CKPT_DIR" \
    --lock "$ENV_LOCK" --python "$PY" --train-rc "$RC"
VERDICT=$?
set -e
case "$VERDICT" in
    0) echo "=== VERIFIED $RUN_NAME $(date -Is) ==="; exit 0 ;;
    3) echo "=== RESUMABLE $RUN_NAME; requeue_left=$REQUEUE_LEFT ==="
       if [ "$REQUEUE_LEFT" -gt 0 ]; then
           SIG=(); [ "$SIGNALLED" = 1 ] && SIG=(--signalled --time-left "$TIME_LEFT")
           if CMD="$($CAMPAIGN successor --requeue-left "$REQUEUE_LEFT" --time "${TIME_LIMIT:?TIME_LIMIT was not exported}" \
                  --script "$REPO/scripts/aicr/slurm_train.sh" --log-dir "$LOG_ROOT/slurm" "${SIG[@]}" \
                  --export "GRID=$GRID" --export "REPO=$REPO" --export "GRID_INDEX=$INDEX" --export "TIME_LIMIT=$TIME_LIMIT" \
                  ${CKPT_ROOT_OVERRIDE:+--export "CKPT_ROOT_OVERRIDE=$CKPT_ROOT_OVERRIDE"})"; then
               echo "successor: $CMD"
               eval "$CMD"
               exit 0
           fi
           echo "no successor submitted (see the reason above); resume by hand with GRID_INDEX=$INDEX" >&2; exit 3
       fi
       echo "requeue budget exhausted: resubmit by hand with GRID_INDEX=$INDEX" >&2; exit 3 ;;
    *) echo "=== FAILED $RUN_NAME (see .failed in $CKPT_DIR) ==="; exit 1 ;;
esac
