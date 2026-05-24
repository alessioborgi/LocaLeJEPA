#!/bin/bash
#SBATCH -J BundLeJEPA_PhC_Finetune
#SBATCH -p gpu,gpu_h200,gpu_requeue,hsph_gpu
#SBATCH -c 4
#SBATCH -t 0-04:00
#SBATCH --mem=32000
#SBATCH --gres=gpu:1
#SBATCH -o logs/PhC_Finetune_%j.out
#SBATCH -e logs/PhC_Finetune_%j.err

##################### Experiment configuration ##################

##### PhC DOS Regression Fine-tuning #####
# Requires a pre-trained checkpoint produced by slurm_harvard_phc_pretrain.sh.
# The checkpoint path defaults to the output of the pre-training script for the
# same dataset; override CKPT_PATH at submission time if needed.
#
# Available datasets:
#   blob   → experiment=phc_regression/blob_supervised_ep500
#   gpm    → experiment=phc_regression/gpm_supervised_ep500
#
# Override at submission time:
#   sbatch --export=ALL,DATASET=gpm src/slurm_files/slurm_harvard_phc_finetune.sh
#   sbatch --export=ALL,DATASET=blob,CKPT_PATH=logs/pretrain/phc_blob_ep250/checkpoints/last.ckpt \
#          src/slurm_files/slurm_harvard_phc_finetune.sh

DEFAULT_DATASET="blob"    # blob | gpm
DEFAULT_FREEZE_ENCODER="true"   # true = frozen encoder (linear probing); false = full fine-tuning

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

# Default checkpoint path: matches the output of slurm_harvard_phc_pretrain.sh
DEFAULT_CKPT_PATH="logs/pretrain/phc_${DATASET}_ep250/checkpoints/last.ckpt"
CKPT_PATH="${CKPT_PATH:-${DEFAULT_CKPT_PATH}}"
FREEZE_ENCODER="${FREEZE_ENCODER:-${DEFAULT_FREEZE_ENCODER}}"

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Checkpoint not found: ${CKPT_PATH}"
  echo "Run the pre-training script first:"
  echo "  sbatch --export=ALL,DATASET=${DATASET} src/slurm_files/slurm_harvard_phc_pretrain.sh"
  exit 1
fi

RUN_DIR="logs/finetune/phc_${DATASET}_ep500"

echo "Dataset        : ${DATASET}"
echo "Checkpoint     : ${CKPT_PATH}"
echo "Freeze encoder : ${FREEZE_ENCODER}"
echo "Experiment     : phc_regression/${DATASET}_supervised_ep500"
echo "Output dir     : ${RUN_DIR}"

python src/train.py \
  experiment="phc_regression/${DATASET}_supervised_ep500" \
  model.pretrained_ckpt_path="${CKPT_PATH}" \
  model.freeze_encoder="${FREEZE_ENCODER}" \
  hydra.run.dir="${RUN_DIR}" \
  data.data_dir="${SLURM_SUBMIT_DIR}/data/phc" \
  logger=wandb \
  "logger.wandb.name=Finetune_PhC_${DATASET}" \
  "logger.wandb.tags=[finetuning,phc_regression,phc,${DATASET}]"

echo "=== Job End: $(date) ==="
