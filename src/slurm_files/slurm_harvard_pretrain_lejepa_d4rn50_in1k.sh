#!/bin/bash
#SBATCH -J BLJ_lejepa_d4rn50_in1k
#SBATCH -p gpu,gpu_h200,gpu_requeue
#SBATCH -c 8
#SBATCH -t 3-00:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h200"
#SBATCH -o logs/BLJ_lejepa_d4rn50_in1k_%j.out
#SBATCH -e logs/BLJ_lejepa_d4rn50_in1k_%j.err

# LeJEPA (SIGReg) pretraining on ImageNet-1K with the equivision D4-equivariant
# ResNet-50 (escnn group convolutions, ~24.8M params) as the encoder instead of
# the ViT-S/8. Requires ImageNet staged under
#   data/imagenet/imagenet_full_size/061417/{train,val}
#
# Saves best.ckpt (by val/probe_acc) + last.ckpt under
#   checkpoints/lejepa_d4rn50_in1k_sigreg/
#
# Submit with:
#   sbatch src/slurm_files/slurm_harvard_pretrain_lejepa_d4rn50_in1k.sh

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

EXP=lejepa/d4rn50_in1k_sigreg
CKPT_DIR="${SLURM_SUBMIT_DIR}/checkpoints/lejepa_d4rn50_in1k_sigreg"
RUN_NAME="PreTrain_LeJEPA_d4rn50_in1k_sigreg"

echo "=== Pretrain start $(date) ==="
echo "Experiment: ${EXP}"
echo "Ckpt dir  : ${CKPT_DIR}"
echo "Python    : $(which python)"

python -c "import hydra, lightning, omegaconf, wandb, escnn" || {
  echo "Missing required Python packages in the active Conda environment."
  exit 1
}

# encode() flattens all views into one tensor and the escnn group convs carry
# |D4|=8 regular-representation channels per field, so keep the per-step batch
# small and accumulate to an effective global batch of 256.
python src/train.py \
  experiment="$EXP" \
  logger=wandb \
  "logger.wandb.name=${RUN_NAME}" \
  "logger.wandb.tags=[lejepa,imagenet1k,d4resnet50,equivariant,sigreg]" \
  "data.data_dir=${SLURM_SUBMIT_DIR}/data/imagenet" \
  data.batch_size=64 \
  +trainer.accumulate_grad_batches=4 \
  "callbacks.model_checkpoint.dirpath=${CKPT_DIR}"

echo "=== Pretrain end $(date) ==="
