#!/bin/bash
#SBATCH -J BundLeJEPA_eval_yoda_local
#SBATCH -p gpu,gpu_h200,gpu_requeue
#SBATCH -c 8
#SBATCH -t 0-02:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -o logs/BundLeJEPA_eval_yoda_local_%j.out
#SBATCH -e logs/BundLeJEPA_eval_yoda_local_%j.err

# Yoda (bobblehead) linear-probe eval of the two local-regularizer ViT-S/8
# backbones. Extracts + caches frozen features (single-seed eval), then runs the
# 10-split multi-seed probe. Submit with a dependency on the two training jobs:
#   sbatch --dependency=afterok:<jobA>:<jobB> slurm_files/slurm_harvard_eval_yoda_vit_local.sh

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa

CFG=configs/eval_angle_probe_yoda_vit_local.yaml

echo "=== Eval Job Start: $(date) ==="
echo "Python: $(which python)"

# 1) Extract + cache frozen features and single-seed metrics.
python eval_angle_probe.py --config "$CFG"

# 2) Multi-seed probe over 10 splits (reuses the cache just written).
python eval_angle_probe_multiseed.py --config "$CFG" --n-seeds 10

echo "=== Eval Job End: $(date) ==="
