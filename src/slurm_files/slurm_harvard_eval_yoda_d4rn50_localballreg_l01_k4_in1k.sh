#!/bin/bash
#SBATCH -J BLJ_ev_d4rn50_localballreg_l01_k4
#SBATCH -p gpu,gpu_h200,gpu_requeue
#SBATCH -c 8
#SBATCH -t 0-02:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h200"
#SBATCH -o logs/BLJ_ev_d4rn50_localballreg_l01_k4_%j.out
#SBATCH -e logs/BLJ_ev_d4rn50_localballreg_l01_k4_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module purge
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa

CFG=configs/eval_angle_probe_yoda_d4rn50_localballreg_l01_k4_in1k.yaml
echo "=== Eval start $(date) ==="
python eval_angle_probe.py --config "$CFG"
python eval_angle_probe_multiseed.py --config "$CFG" --n-seeds 10
python scripts/make_equivariance_report.py --out report_equivariance.md || true
echo "=== Eval end $(date) ==="
