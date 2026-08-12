#!/bin/bash
#SBATCH -J BLJ_rlpjepa_hvit_imnette
#SBATCH -p gpu,gpu_h200,gpu_requeue
#SBATCH -c 8
#SBATCH -t 3-00:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h200"
#SBATCH -o logs/BLJ_rlpjepa_hvit_imnette_%j.out
#SBATCH -e logs/BLJ_rlpjepa_hvit_imnette_%j.err

# Rectified LpJEPA (RDMReg, Rectified Laplace target) pretraining on Imagenette
# with the Hyperbolic ViT-S/8 (Poincare-ball embedding, hyp_metric-style,
# hyp_c=0.1, clip_r=2.3) as the encoder — identical protocol to
# rectifiedlpjepa/vit_imagenette_rdmreg (800 epochs, global batch 256), so the
# frozen backbone drops into the Yoda probe table of paper/yoda_probe_paper.tex.
#
# Saves best.ckpt (by val/probe_acc) + last.ckpt under
#   checkpoints/rectifiedlpjepa_hvit_imagenette_rdmreg/
#
# Submit with:
#   sbatch src/slurm_files/slurm_harvard_pretrain_rectifiedlpjepa_hvit_imagenette.sh

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

EXP=rectifiedlpjepa/hvit_imagenette_rdmreg
CKPT_DIR="${SLURM_SUBMIT_DIR}/checkpoints/rectifiedlpjepa_hvit_imagenette_rdmreg"
RUN_NAME="PreTrain_RectifiedLpJEPA_hvit_imagenette_rdmreg"

echo "=== Pretrain start $(date) ==="
echo "Experiment: ${EXP}"
echo "Ckpt dir  : ${CKPT_DIR}"
echo "Python    : $(which python)"

python -c "import hydra, lightning, omegaconf, wandb, timm" || {
  echo "Missing required Python packages in the active Conda environment."
  exit 1
}

# encode() flattens all views into one tensor (batch * num_views images at
# once), so use batch_size=64 + accumulate 4 to keep the effective global batch
# at 256 (matches every other Imagenette ViT-S/8 run).
python src/train.py \
  experiment="$EXP" \
  logger=wandb \
  "logger.wandb.name=${RUN_NAME}" \
  "logger.wandb.tags=[rectifiedlpjepa,imagenette,hvit,hyperbolic,rdmreg]" \
  "data.data_dir=${SLURM_SUBMIT_DIR}/data/imagenette" \
  data.batch_size=64 \
  +trainer.accumulate_grad_batches=4 \
  "callbacks.model_checkpoint.dirpath=${CKPT_DIR}"

echo "=== Pretrain end $(date) ==="
