#!/bin/bash
#SBATCH -J BLJ_eval_indomain_in1k
#SBATCH -p gpu,gpu_h200
#SBATCH -c 16
#SBATCH -t 0-12:00
#SBATCH --mem=128000
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h200"
#SBATCH -o logs/BLJ_eval_indomain_in1k_%A_%a.out
#SBATCH -e logs/BLJ_eval_indomain_in1k_%A_%a.err
#SBATCH --array=0-3

# In-domain ImageNet-1K evaluation (SPHERE-JEPA Table 3 protocol) of the
# IN1K-pretrained backbones: freeze the encoder, embed the full 1.28M-image
# train split as the k-NN feature bank / linear-probe training set, report
# top-1 on the 50k val split with a weighted k-NN vote and a linear probe.
#
# One array task per backbone, each writing logs/eval_indomain_in1k/per_backbone/
# so the four jobs never clobber each other's results.
#
#   sbatch src/slurm_files/slurm_harvard_eval_indomain_in1k.sh
#   sbatch --array=0 src/slurm_files/slurm_harvard_eval_indomain_in1k.sh   # single backbone

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module purge
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa

CFG=configs/eval_indomain_in1k.yaml
BACKBONES=(hvit_sigreg d4rn50_sigreg hvit_rdmreg d4rn50_rdmreg)
NAME=${BACKBONES[${SLURM_ARRAY_TASK_ID:-0}]}

echo "=== In-domain IN1K eval: $NAME | start $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python eval_indomain.py --config "$CFG" --only "$NAME"
echo "=== In-domain IN1K eval: $NAME | end $(date) ==="
