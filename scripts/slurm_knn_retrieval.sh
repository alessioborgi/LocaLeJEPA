#!/bin/bash
#SBATCH --job-name=BLJ_knnvis
#SBATCH --partition=gpu_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# k-NN retrieval figures (visualisations/plot_knn_retrieval.py). The image-level
# panel reads the cached k-NN bank on CPU; the patch panels need patch tokens of
# a 10k-image validation bank, extracted here and cached in logs/knn_retrieval/
# so re-styling afterwards runs on a login node (--stage patches).
#
#   sbatch scripts/slurm_knn_retrieval.sh                       # plain ViT-S/8
#   sbatch --export=ALL,FAMILY=hvit scripts/slurm_knn_retrieval.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa
export PROJECT_ROOT=$PWD
echo "=== $(date) | host $(hostname) | job ${SLURM_JOB_ID:-local} ==="
python visualisations/plot_knn_retrieval.py --family "${FAMILY:-vit}" --stage all
echo "=== end $(date) ==="
