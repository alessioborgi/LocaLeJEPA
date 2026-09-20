#!/bin/bash
#SBATCH --job-name=stage_in1k
#SBATCH --account=dominici_lab_hu
#SBATCH --partition=cpu
# -c 2, not 4. stage_imagenet1k_hf.py is a single-threaded loop writing files:
# I/O-bound, not CPU-bound. Asking for 4 makes the job queue behind
# QOSMaxCpuPerUserLimit waiting for four free slots instead of two, which is
# what happened to job 1004159. (The -c 4 advice belongs to the TEST SUITE,
# where torch spawns a thread per core and 4 matches OMP_NUM_THREADS.)
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/clabat9_hu/localejepa/slurm/stage_in1k-%j.out
#SBATCH --error=/scratch/clabat9_hu/localejepa/slurm/stage_in1k-%j.err
#
# Stage ImageNet-1K from the HuggingFace cache into a PERSISTENT ImageFolder tree.
#
# WHY THIS FILE EXISTS. Job 997480 did exactly this and wrote to
# `/tmp/in1k_997480/`, which is node-local and was wiped when the allocation
# ended. 1,281,167 train and 50,000 val images, staged perfectly, gone. The
# job reported COMPLETED 0:0, because the staging script calls os._exit(0) and
# because TIMEOUT also reports 0:0 on this cluster (finding F33).
#
# `/tmp` is the right target for a TRAINING job, which copies the dataset onto
# node-local NVMe at startup (F8: ~110x faster than NFS). It is the wrong target
# for the master copy, because then there is nothing left to copy from.
#
# Destination is /scratch, not /work: `state/05-DECISIONS.md` puts the dataset
# on scratch and checkpoints on work, and /work has 260G free on a quota shared
# with the rest of the lab. /scratch is subject to the 30-day purge -- the purge
# canary is planted and must be read before relying on this tree long-term.
#
#   sbatch scripts/slurm_stage_in1k.sh
#
# Override the destination with IN1K_DEST=/some/path sbatch --export=ALL,...
set -euo pipefail

DEST="${IN1K_DEST:-/scratch/clabat9_hu/localejepa/data}"
REPO="${REPO:-/work/hu/dominici_lab_hu/cbattiloro_localejepa/LocaLeJEPA}"
PY="${PY:-$HOME/conda/envs/localejepa-aicr/bin/python}"

export HF_HOME=/scratch/clabat9_hu/hf
export HF_HUB_DISABLE_XET=1          # F27: Xet is a separate backend; 68 KB/s vs 7 MB/s
export HF_HUB_ENABLE_HF_TRANSFER=0   # hangs on a stalled link, no read timeout
export HF_HUB_DOWNLOAD_TIMEOUT=30

echo "host      : $(hostname)"
echo "dest      : $DEST"
echo "HF_HOME   : $HF_HOME"
df -h "$(dirname "$DEST")" | tail -1
echo

mkdir -p "$DEST"
time "$PY" "$REPO/scripts/stage_imagenet1k_hf.py" --dest "$DEST"

# The job does not get to claim success on the staging script's own word. That
# script checks one thing -- 1000 class dirs per split -- which a partial
# download that touched every class passes. The verifier checks the counts that
# actually pin the dataset, and `set -e` makes a non-zero exit fail the job.
echo
echo "=== verifying the staged tree ==="
time "$PY" "$REPO/scripts/verify_imagenet_staging.py" \
     --root "$DEST/imagenet_full_size/061417" --decode 200

echo
echo "STAGED AND VERIFIED: $DEST/imagenet_full_size/061417"
echo "point the datamodule at it with: data.data_dir=$DEST"
du -sh "$DEST"
