#!/bin/bash
#SBATCH -J BundLeJEPA_PhC_Pretrain
#SBATCH -p gpu,gpu_h200,gpu_requeue,hsph_gpu
#SBATCH -c 4
#SBATCH -t 0-08:00
#SBATCH --mem=32000
#SBATCH --gres=gpu:1
#SBATCH -o logs/PhC_Pretrain_%j.out
#SBATCH -e logs/PhC_Pretrain_%j.err

##################### Experiment configuration ##################

##### PhC LeJEPA Pre-training #####
# Available datasets:
#   blob   → experiment=lejepa/phc_blob_ep250
#   gpm    → experiment=lejepa/phc_gpm_ep250
#
# Override at submission time with:
#   sbatch --export=ALL,DATASET=gpm src/slurm_files/slurm_harvard_phc_pretrain.sh

DEFAULT_DATASET="blob"   # blob | gpm

###############################################################
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

module purge

source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa

wandb login
export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Python: $(which python)"
python -c "import sys; print('Python version:', sys.version)"

python -c "import hydra, lightning, omegaconf, wandb" || {
  echo "Missing required Python packages in the active Conda environment."
  exit 1
}

DATASET="${DATASET:-${DEFAULT_DATASET}}"

case "$DATASET" in
  blob|gpm) ;;
  *)
    echo "Unsupported DATASET: $DATASET. Valid options: blob | gpm"
    exit 1
    ;;
esac

RUN_DIR="logs/pretrain/phc_${DATASET}_ep250"

echo "Dataset   : ${DATASET}"
echo "Experiment: lejepa/phc_${DATASET}_ep250"
echo "Output dir: ${RUN_DIR}"

python src/train.py \
  experiment="lejepa/phc_${DATASET}_ep250" \
  hydra.run.dir="${RUN_DIR}" \
  data.data_dir="${SLURM_SUBMIT_DIR}/data/phc" \
  logger=wandb \
  "logger.wandb.name=PreTrain_LeJEPA_phc_${DATASET}" \
  "logger.wandb.tags=[pretraining,lejepa,phc,${DATASET}]"

echo "=== Job End: $(date) ==="
echo "Checkpoint saved to: ${RUN_DIR}/checkpoints/last.ckpt"
