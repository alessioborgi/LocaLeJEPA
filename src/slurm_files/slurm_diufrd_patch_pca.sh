#!/bin/bash
#SBATCH --job-name=blj-patch-pca
#SBATCH --output=logs/embedor/%x_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --partition=GPU
#SBATCH --exclude=diufrd202,diufrd204
#SBATCH --chdir=/HOME/garciare/BundLeJEPA
set -euo pipefail
source .venv/bin/activate
export MPLCONFIGDIR="${TMPDIR:-/tmp}/blj-pca-mpl-${SLURM_JOB_ID}"
python visualize_patch_pca.py "$@"
