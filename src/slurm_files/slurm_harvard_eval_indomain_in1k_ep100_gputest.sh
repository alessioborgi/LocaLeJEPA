#!/bin/bash
#SBATCH -J BLJ_eval_indomain_in1k_ep100_gt
#SBATCH -p gpu_test
#SBATCH -c 8
#SBATCH -t 0-06:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH -o logs/BLJ_eval_indomain_in1k_ep100_gt_%A_%a.out
#SBATCH -e logs/BLJ_eval_indomain_in1k_ep100_gt_%A_%a.err
#SBATCH --array=0-10

# In-domain ImageNet-1K evaluation (SPHERE-JEPA Table 3 protocol) of the
# finished 100-epoch IN1K pretrains: freeze the encoder, embed the full
# 1.28M-image train split as k-NN bank / linear-probe train set, report top-1
# on the 50k val split with a weighted k-NN vote and a linear probe.
#
# One array task per backbone entry of configs/eval_indomain_in1k_ep100.yaml;
# each writes its own logs/eval_indomain_in1k_ep100/per_backbone/<name>.json.
#
#   sbatch src/slurm_files/slurm_harvard_eval_indomain_in1k_ep100.sh
#   sbatch --array=0 src/slurm_files/slurm_harvard_eval_indomain_in1k_ep100.sh   # single backbone
#   python eval_indomain.py --config configs/eval_indomain_in1k_ep100.yaml --merge-only

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module purge
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa

CFG=configs/eval_indomain_in1k_ep100.yaml
BACKBONES=(
  d4rn50_sigreg hvit_sigreg hvit_rdmreg
  d4rn50_localballreg_l01_k2 d4rn50_localballreg_l01_k4 d4rn50_localballreg_l03_k2
  hvit_localballreg_l03_k2
  d4rn50_localballreg_l01_k2_best d4rn50_localballreg_l01_k4_best d4rn50_localballreg_l03_k2_best
  hvit_localballreg_l03_k2_best
)
NAME=${BACKBONES[${SLURM_ARRAY_TASK_ID:-0}]}

if [ -f "logs/eval_indomain_in1k_ep100/per_backbone/${NAME}.json" ]; then
  echo "=== $NAME already evaluated (per_backbone/${NAME}.json exists) — skipping ==="
  exit 0
fi
echo "=== In-domain IN1K (100 ep.) eval: $NAME | start $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python eval_indomain.py --config "$CFG" --only "$NAME"
echo "=== In-domain IN1K (100 ep.) eval: $NAME | end $(date) ==="
