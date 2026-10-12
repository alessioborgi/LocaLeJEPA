#!/bin/bash
#SBATCH --job-name=BLJ_geom
#SBATCH --partition=gpu_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# Figure "three embedding geometries" (visualisations/plot_encoder_geometries.py).
# Only the D4 panel needs a GPU: it embeds the eight group views of a few
# ImageNet val images with both D4-ResNet-50 checkpoints and caches them as
# paper/figs/encoder_geometries_orbits.npz. The two other columns read the
# cached val features of src/eval_indomain.py, so re-styling afterwards runs on
# a login node with `python visualisations/plot_encoder_geometries.py`.
#
#   sbatch scripts/slurm_encoder_geometries.sh            # cache + figure
#   sbatch --export=ALL,EXTRACT_ONLY=1 scripts/slurm_encoder_geometries.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa
export PROJECT_ROOT=$PWD
ARGS=(--extract)
[ -n "${EXTRACT_ONLY:-}" ] && ARGS=(--extract-only)
echo "=== $(date) | host $(hostname) | job ${SLURM_JOB_ID:-local} ==="
python visualisations/plot_encoder_geometries.py "${ARGS[@]}"
echo "=== end $(date) ==="
