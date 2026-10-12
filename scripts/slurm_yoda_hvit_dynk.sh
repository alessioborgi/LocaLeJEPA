#!/bin/bash
#SBATCH --job-name=BLJ_yoda_hvit_dynk
#SBATCH --partition=gpu_test
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa
export PROJECT_ROOT=$PWD
echo "=== Yoda eval start $(date) on $(hostname) ==="
python src/eval_angle_probe.py --config configs/eval/yoda/hvit_localballreg_l01_dynk_in1k.yaml
python src/eval_angle_probe_multiseed.py --config configs/eval/yoda/hvit_localballreg_l01_dynk_in1k.yaml --n-seeds 10
echo "=== Yoda eval end $(date) ==="
