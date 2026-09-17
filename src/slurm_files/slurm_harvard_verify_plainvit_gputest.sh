#!/bin/bash
#SBATCH -J BLJ_verify_plainvit
#SBATCH -p gpu_test
#SBATCH -c 8
#SBATCH -t 0-04:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH -o logs/BLJ_verify_plainvit_%j.out
#SBATCH -e logs/BLJ_verify_plainvit_%j.err
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
module purge
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa
echo "=== verify start $(date) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/verify_plainvit_result.py --out logs/verify_plainvit/verify.md
echo "=== verify end $(date) ==="
