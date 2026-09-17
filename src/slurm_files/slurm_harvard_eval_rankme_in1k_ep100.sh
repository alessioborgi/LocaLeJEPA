#!/bin/bash
#SBATCH -J BLJ_eval_rankme_in1k_ep100
#SBATCH -p gpu,gpu_h200,gpu_requeue
#SBATCH -c 16
#SBATCH -t 0-01:30
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h200|a40|rtxa6000|rtx6000pro"
#SBATCH --requeue
#SBATCH -o logs/BLJ_eval_rankme_in1k_ep100_%A_%a.out
#SBATCH -e logs/BLJ_eval_rankme_in1k_ep100_%A_%a.err
#SBATCH --array=0-15

# RankMe (Garrido et al., ICML 2023) of the 100-epoch IN1K pretrains: embed
# 25,600 ImageNet-1K train images (+ the transfer train splits when not already
# cached) with the frozen encoder and measure the entropic rank of the 512-d
# representations and 16-d embeddings. Backbones whose features are already in
# the in-domain / transfer caches need no GPU; run those on a login node with
#   PYTHONPATH=. python eval_rankme.py --config configs/eval_rankme_in1k_ep100.yaml --cached-only
# and submit this array only for the remaining ones, e.g.
#   sbatch --array=1,5,6,9,12 src/slurm_files/slurm_harvard_eval_rankme_in1k_ep100.sh
#
# One array task per backbone entry of configs/eval_rankme_in1k_ep100.yaml (in
# order). A task whose checkpoint has not reached FINAL_EPOCH exits without
# evaluating (override with FINAL_EPOCH=0 to rank an unfinished run).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module purge
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa

CFG=configs/eval_rankme_in1k_ep100.yaml
NAME=$(python -c "import yaml; c=yaml.safe_load(open('$CFG')); print(c['backbones'][${SLURM_ARRAY_TASK_ID:-0}]['name'])")
CKPT=$(python -c "import yaml; c=yaml.safe_load(open('$CFG')); print(c['backbones'][${SLURM_ARRAY_TASK_ID:-0}]['checkpoint'])")
FINAL_EPOCH=${FINAL_EPOCH:-99}

EPOCH=$(python -c "import torch; print(torch.load('$CKPT', map_location='cpu', weights_only=False, mmap=True).get('epoch', -1))")
if [ "$EPOCH" -lt "$FINAL_EPOCH" ]; then
  echo "=== $NAME: $CKPT is at epoch $EPOCH < $FINAL_EPOCH — still pretraining, skipping ==="
  exit 0
fi

echo "=== RankMe IN1K (100 ep.): $NAME (epoch $EPOCH) | start $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
PYTHONPATH=. python eval_rankme.py --config "$CFG" --only "$NAME" ${EXTRA_ARGS:-}
echo "=== RankMe IN1K (100 ep.): $NAME | end $(date) ==="
