#!/bin/bash
#SBATCH -J BLJ_stage_in1k
#SBATCH -p shared
#SBATCH -c 8
#SBATCH -t 1-00:00
#SBATCH --mem=32000
#SBATCH -o logs/BLJ_stage_in1k_%j.out
#SBATCH -e logs/BLJ_stage_in1k_%j.err

# Stage ImageNet-1K (~160 GB) from the gated HuggingFace dataset
# ILSVRC/imagenet-1k into the ImageFolder layout the ImageNet1KDataModule
# expects, on netscratch (home quota is far too small), then symlink
# data/imagenet -> netscratch so every *_in1k experiment works unchanged.
#
# PREREQUISITE (one-time, must be done by the user):
#   1. Accept the license at https://huggingface.co/datasets/ILSVRC/imagenet-1k
#   2. hf auth login   (or the older `huggingface-cli login`; stores the token
#      under ~/.cache/huggingface/token)
#
# NOTE: netscratch retention deletes untouched files after ~90 days — re-run
# this job if the dataset ever turns into empty class dirs again.
#
# Submit with:
#   sbatch src/slurm_files/slurm_harvard_stage_imagenet1k.sh

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa

DEST=/n/netscratch/dominici_lab/Lab/aborgi/imagenet
HF_CACHE=/n/netscratch/dominici_lab/Lab/aborgi/hf_cache
mkdir -p "$DEST" "$HF_CACHE"

# Keep the HuggingFace parquet cache on netscratch (home has only ~23 GB free);
# the auth token still resolves from the default ~/.cache/huggingface/token.
export HF_DATASETS_CACHE="$HF_CACHE"
export HF_HUB_CACHE="$HF_CACHE"
export HF_XET_CACHE="$HF_CACHE/xet"
# High-performance Xet transfer (huggingface_hub deprecated HF_HUB_ENABLE_HF_TRANSFER).
export HF_XET_HIGH_PERFORMANCE=1

python - <<'PYEOF' || { echo "No usable HuggingFace token — see PREREQUISITE in this script."; exit 1; }
from huggingface_hub import get_token
import sys
sys.exit(0 if get_token() else 1)
PYEOF

python scripts/stage_imagenet1k_hf.py --dest "$DEST" --cache-dir "$HF_CACHE"

# Make the default config paths work without overrides, then drop the parquet
# download cache (the extracted ImageFolder under $DEST is what training reads).
ln -sfn "$DEST" "${SLURM_SUBMIT_DIR}/data/imagenet"
rm -rf "$HF_CACHE"

echo "=== Staging done $(date): data/imagenet -> $DEST ==="
