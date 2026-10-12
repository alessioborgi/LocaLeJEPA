#!/bin/bash
#SBATCH --job-name=BLJ_seeds_rest
#SBATCH --partition=gpu_test
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa
export PROJECT_ROOT=$PWD
# in-domain is already complete for all 22 rows; only these four transfer backbones remain
for NAME in hvit_localsigreg_l01_k4 hvit_localballreg_l01_dynk vit_localballreg_l01_k4 vit_sigreg; do
  echo "=== $(date) transfer 5-seed: $NAME ==="
  python src/eval_transfer.py --config configs/eval/transfer_in1k_ep100.yaml --only "$NAME" --force
done
python src/eval_transfer.py --config configs/eval/transfer_in1k_ep100.yaml --merge-only
python paper/make_paper_tables.py
echo "=== end $(date) ==="
