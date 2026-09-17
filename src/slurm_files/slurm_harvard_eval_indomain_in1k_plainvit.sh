#!/bin/bash
#SBATCH -J BLJ_eval_indomain_in1k_plainvit
#SBATCH -p gpu,gpu_h200,gpu_requeue
#SBATCH -c 16
#SBATCH -t 0-06:00
#SBATCH --mem=128000
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h200|a40|rtxa6000|rtx6000pro"
#SBATCH --requeue
#SBATCH -o logs/BLJ_eval_indomain_in1k_plainvit_%A_%a.out
#SBATCH -e logs/BLJ_eval_indomain_in1k_plainvit_%A_%a.err
#SBATCH --array=0-3

# In-domain ImageNet-1K k-NN + linear-probe evaluation of the plain ViT-S/8
# 100-epoch pretrains. One array task per backbone entry of
# configs/eval_indomain_in1k_plainvit.yaml; each writes
# logs/eval_indomain_in1k_plainvit/per_backbone/<name>.json.
#
#   sbatch src/slurm_files/slurm_harvard_eval_indomain_in1k_plainvit.sh
#   sbatch --array=0,2 src/slurm_files/slurm_harvard_eval_indomain_in1k_plainvit.sh  # SIGReg only

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module purge
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa

CFG=configs/eval_indomain_in1k_plainvit.yaml
BACKBONES=(vit_sigreg vit_localballreg_l01_k4 vit_sigreg_best vit_localballreg_l01_k4_best)
NAME=${BACKBONES[${SLURM_ARRAY_TASK_ID:-0}]}

if [ -f "logs/eval_indomain_in1k_plainvit/per_backbone/${NAME}.json" ]; then
  echo "=== $NAME already evaluated — skipping ==="; exit 0
fi
echo "=== In-domain IN1K plain-ViT eval: $NAME | start $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python eval_indomain.py --config "$CFG" --only "$NAME"
echo "=== In-domain IN1K plain-ViT eval: $NAME | end $(date) ==="
