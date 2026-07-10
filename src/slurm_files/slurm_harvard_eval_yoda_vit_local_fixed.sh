#!/bin/bash
#SBATCH -J BundLeJEPA_eval_yoda_local_fix
#SBATCH -p gpu,gpu_h200,gpu_requeue
#SBATCH -c 8
#SBATCH -t 0-01:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -o logs/BundLeJEPA_eval_yoda_local_fix_%j.out
#SBATCH -e logs/BundLeJEPA_eval_yoda_local_fix_%j.err

# CORRECTED Yoda eval — pins the true epoch-799 checkpoints (last-vN.ckpt) plus
# sigreg's pre-divergence epoch_016. Fresh feature extraction, then 10-split probe.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa

CFG=configs/eval_angle_probe_yoda_vit_local_fixed.yaml

echo "=== Eval Job Start: $(date) ==="
python eval_angle_probe.py --config "$CFG"
python eval_angle_probe_multiseed.py --config "$CFG" --n-seeds 10
echo "=== Eval Job End: $(date) ==="
