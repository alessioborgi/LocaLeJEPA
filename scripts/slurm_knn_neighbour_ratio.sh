#!/bin/bash
#SBATCH --job-name=BLJ_knnratio
#SBATCH --partition=gpu_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# Same-class share of the k-NN neighbourhood (scripts/knn_neighbour_ratio.py):
# every IN1K val image against the full cached 1.28M-image train bank.
#
#   sbatch scripts/slurm_knn_neighbour_ratio.sh                  # vit, hvit, d4rn50
#   sbatch --export=ALL,FAMILIES=vit scripts/slurm_knn_neighbour_ratio.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa
export PROJECT_ROOT=$PWD
echo "=== $(date) | host $(hostname) | job ${SLURM_JOB_ID:-local} ==="
for F in ${FAMILIES:-vit hvit d4rn50}; do python scripts/knn_neighbour_ratio.py --family "$F"; done
echo "=== end $(date) ==="
