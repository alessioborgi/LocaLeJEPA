#!/bin/bash
#SBATCH -J BLJ_eval_transfer_in1k_ep100_gt
#SBATCH -p gpu_test
#SBATCH -c 8
#SBATCH -t 0-04:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH -o logs/BLJ_eval_transfer_in1k_ep100_gt_%A_%a.out
#SBATCH -e logs/BLJ_eval_transfer_in1k_ep100_gt_%A_%a.err
#SBATCH --array=0-6

# Transfer linear-probing evaluation (SPHERE-JEPA Table 1 protocol) of the
# finished 100-epoch IN1K pretrains: freeze the encoder, embed the train/test
# splits of DTD, Aircraft, CIFAR-10/100, Flowers, Food and Pets, train a linear
# probe per dataset on the frozen features, report top-1 on each test split.
#
# One array task per backbone entry of configs/eval_transfer_in1k_ep100.yaml
# (indices 0-13 = D4-RN50 / HViT runs, 14-15 = plain ViT-S/8 baselines; a task whose
# checkpoint has not reached the final epoch exits without evaluating). Each
# task writes logs/eval_transfer_in1k_ep100/per_backbone/<name>.json
# incrementally (one dataset at a time), so a requeued task resumes.
#
#   sbatch src/slurm_files/slurm_harvard_eval_transfer_in1k_ep100.sh
#   sbatch --array=7 src/slurm_files/slurm_harvard_eval_transfer_in1k_ep100.sh   # one backbone
#   sbatch --export=ALL,EXTRA_ARGS=--force src/slurm_files/slurm_harvard_eval_transfer_in1k_ep100.sh  # re-probe cached features
#   python eval_transfer.py --config configs/eval_transfer_in1k_ep100.yaml --merge-only

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module purge
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa

CFG=configs/eval_transfer_in1k_ep100.yaml
BACKBONES=(
  d4rn50_sigreg hvit_sigreg hvit_rdmreg
  d4rn50_localballreg_l01_k2 d4rn50_localballreg_l01_k4 d4rn50_localballreg_l03_k2
  hvit_localballreg_l03_k2
  d4rn50_rdmreg d4rn50_localsigreg_l01_k4 d4rn50_localballreg_l01_dynk
  hvit_localballreg_l01_k2 hvit_localballreg_l01_k4 hvit_localsigreg_l01_k4 hvit_localballreg_l01_dynk
  vit_sigreg vit_localballreg_l01_k4
)
NAME=${BACKBONES[${SLURM_ARRAY_TASK_ID:-0}]}
FINAL_EPOCH=${FINAL_EPOCH:-99}

# Only evaluate checkpoints that reached the final epoch (the paper's "100 ep.").
CKPT=$(python -c "import yaml,sys; c=yaml.safe_load(open('$CFG')); print(next(b['checkpoint'] for b in c['backbones'] if b['name']=='$NAME'))")
EPOCH=$(python -c "import torch; print(torch.load('$CKPT', map_location='cpu', weights_only=False, mmap=True).get('epoch', -1))")
# WAIT_FOR_FINAL=<minutes>: instead of skipping, poll the checkpoint until it reaches
# the final epoch (for runs that are one epoch away), then evaluate.
if [ "$EPOCH" -lt "$FINAL_EPOCH" ] && [ -n "${WAIT_FOR_FINAL:-}" ]; then
  for _ in $(seq 1 "$WAIT_FOR_FINAL"); do
    EPOCH=$(python -c "import torch; print(torch.load('$CKPT', map_location='cpu', weights_only=False, mmap=True).get('epoch', -1))" 2>/dev/null || echo -1)
    [ "$EPOCH" -ge "$FINAL_EPOCH" ] && break
    echo "waiting for $NAME to reach epoch $FINAL_EPOCH (now $EPOCH, $(date))"; sleep 60
  done
fi
if [ "$EPOCH" -lt "$FINAL_EPOCH" ]; then
  echo "=== $NAME: $CKPT is at epoch $EPOCH < $FINAL_EPOCH — still pretraining, skipping ==="
  exit 0
fi

# The datasets are downloaded on a login node; wait (up to 3 h) for the marker the
# download script writes once every archive passed its checksum.
DATA_ROOT=$(python -c "import yaml; print(yaml.safe_load(open('$CFG'))['data_root'])")
for _ in $(seq 1 180); do [ -f "$DATA_ROOT/.ready" ] && break; echo "waiting for $DATA_ROOT/.ready ($(date))"; sleep 60; done
[ -f "$DATA_ROOT/.ready" ] || { echo "datasets not ready after 3 h — aborting"; exit 1; }

echo "=== Transfer IN1K (100 ep.) eval: $NAME (epoch $EPOCH) | start $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python eval_transfer.py --config "$CFG" --only "$NAME" ${EXTRA_ARGS:-}
echo "=== Transfer IN1K (100 ep.) eval: $NAME | end $(date) ==="
