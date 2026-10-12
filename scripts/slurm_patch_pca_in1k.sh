#!/bin/bash
#SBATCH --job-name=BLJ_pca
#SBATCH --partition=gpu_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# Patch-PCA RGB maps (src/visualize_patch_pca.py) for the IN1K pretrains.
# Runs group-scope PCA on 5 val images for each requested class, one folder per
# backbone under logs/patch_pca_in1k/<scope>/. Overridable via --export:
#   CLASSES  space-separated quoted class names or folder ids (default below)
#   SCOPE    group | image (default group)
#   ONLY     comma-separated backbone names (default: all in the config)
#   CONFIG   eval YAML (default configs/eval/patch_pca_in1k.yaml)
#   sbatch scripts/slurm_patch_pca_in1k.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa
export PROJECT_ROOT=$PWD
SCOPE="${SCOPE:-group}"
OUT="${OUT:-logs/patch_pca_in1k/${SCOPE}}"
ONLY_ARG=()
[ -n "${ONLY:-}" ] && ONLY_ARG=(--only "$ONLY")
eval "CLASS_ARR=(${CLASSES:-\"English springer\" \"golden retriever\" \"tabby\"})"
echo "=== $(date) | host $(hostname) | job ${SLURM_JOB_ID:-local} | scope $SCOPE | classes ${CLASS_ARR[*]} ==="
python src/visualize_patch_pca.py --config "${CONFIG:-configs/eval/patch_pca_in1k.yaml}" \
  "${ONLY_ARG[@]}" --scope "$SCOPE" --classes "${CLASS_ARR[@]}" \
  --per-class "${PER_CLASS:-5}" --seed "${SEED:-42}" --out "$OUT" --overwrite
echo "=== end $(date) ==="
