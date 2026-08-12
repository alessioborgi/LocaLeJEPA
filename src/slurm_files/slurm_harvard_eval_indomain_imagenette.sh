#!/bin/bash
#SBATCH -J BLJ_eval_indomain_imagenette
#SBATCH -p gpu,gpu_h200
#SBATCH -c 8
#SBATCH -t 0-03:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h200"
#SBATCH -o logs/BLJ_eval_indomain_imagenette_%j.out
#SBATCH -e logs/BLJ_eval_indomain_imagenette_%j.err

# In-domain Imagenette evaluation (same k-NN + linear protocol as the IN1K
# job) over every Imagenette-pretrained backbone, including the SUSReg
# (SPHERE-JEPA) checkpoint. Small enough to run all backbones in one job.
#
#   sbatch src/slurm_files/slurm_harvard_eval_indomain_imagenette.sh

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module purge
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa

CFG=configs/eval_indomain_imagenette.yaml
echo "=== In-domain Imagenette eval start $(date) ==="
python eval_indomain.py --config "$CFG"
echo "=== In-domain Imagenette eval end $(date) ==="
