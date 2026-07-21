#!/bin/bash
#SBATCH -J BLJ_eval_yoda_d4rn50_sigreg
#SBATCH -p gpu,gpu_h200,gpu_requeue
#SBATCH -c 8
#SBATCH -t 0-02:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h200"
#SBATCH -o logs/BLJ_eval_yoda_d4rn50_sigreg_%j.out
#SBATCH -e logs/BLJ_eval_yoda_d4rn50_sigreg_%j.err

# Yoda eval of the D4-equivariant ResNet-50 LeJEPA (SIGReg) backbone: extract +
# cache frozen features (single-seed), then the 10-split multi-seed probe.
# Intended to be submitted with an afterok dependency on the pretraining job.
# (2h limit: escnn model construction + inference is slower than the ViT.)

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module purge
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa

CFG=configs/eval_angle_probe_yoda_d4rn50_sigreg.yaml
echo "=== Eval start $(date) ==="
python eval_angle_probe.py --config "$CFG"
python eval_angle_probe_multiseed.py --config "$CFG" --n-seeds 10
echo "=== Eval end $(date) ==="
