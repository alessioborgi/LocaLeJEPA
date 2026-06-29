#!/bin/bash
#SBATCH -J BundLeJEPA_eval_yoda
#SBATCH -p gpu,gpu_h200,gpu_requeue
#SBATCH -c 8
#SBATCH -t 0-02:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -o logs/BundLeJEPA_eval_yoda_%j.out
#SBATCH -e logs/BundLeJEPA_eval_yoda_%j.err

# Yoda (bobblehead) linear-probe eval of the two newly-trained plain ViT-S/8
# backbones. Intended to be submitted with a dependency on the two training
# jobs so it fires automatically once they finish:
#   sbatch --dependency=afterok:<jobA>:<jobB> slurm_files/slurm_harvard_eval_yoda_vit.sh

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa

echo "=== Eval Job Start: $(date) ==="
echo "Python: $(which python)"

python eval_angle_probe.py --config configs/eval_angle_probe_yoda_vit.yaml

echo "=== Eval Job End: $(date) ==="
