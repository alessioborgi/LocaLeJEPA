#!/bin/bash
#SBATCH -J BLJ_localballreg_dynk
#SBATCH -p gpu,gpu_h200,gpu_requeue
#SBATCH -c 8
#SBATCH -t 3-00:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH -o logs/BLJ_localballreg_dynk_%j.out
#SBATCH -e logs/BLJ_localballreg_dynk_%j.err

# Local BallReg ViT-S/8 pretraining on Imagenette with a DYNAMIC k schedule:
# the local k-means neighborhood count starts at k=1 and steps up by one every
# 100 epochs, capping at k=8 from epoch 700 on (k=1 @0, k=2 @100, ..., k=8 @700).
# Fixed lambda=0.1, stabilized recipe (LR 1e-3, 10-epoch warmup, grad clip 1.0)
# -- the recipe proven healthy for the local regularizers. 800 epochs, global
# batch 256. Its frozen backbone is then linear-probed on the Yoda rotation-angle
# task and added to paper/yoda_probe.tex.
#
# Saves best.ckpt (by val/probe_acc) + last.ckpt under
#   checkpoints/local_ballreg_dynk_l0.1/
#
# Submit with:
#   sbatch src/slurm_files/slurm_harvard_pretrain_localballreg_dynk.sh

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

EXP=lejepa/vit_imagenette_local_ballreg_dynk
CKPT_DIR="${SLURM_SUBMIT_DIR}/checkpoints/local_ballreg_dynk_l0.1"
RUN_NAME="PreTrain_localballreg_dynk_l0.1"

echo "=== Pretrain start $(date) ==="
echo "Experiment: ${EXP}"
echo "Ckpt dir  : ${CKPT_DIR}"
echo "Python    : $(which python)"

python -c "import hydra, lightning, omegaconf, wandb" || {
  echo "Missing required Python packages in the active Conda environment."
  exit 1
}

# encode() flattens all views into one tensor (batch * num_views images at once),
# so 256*4 views overflows an 80 GB A100. Use batch_size=64 + accumulate 4 to keep
# the effective global batch at 256 (matches every other Imagenette ViT-S/8 run).
python src/train.py \
  experiment="$EXP" \
  logger=wandb \
  "logger.wandb.name=${RUN_NAME}" \
  "logger.wandb.tags=[lejepa,imagenette,vit,local_ballreg,dynamic_k]" \
  "data.data_dir=${SLURM_SUBMIT_DIR}/data/imagenette" \
  data.batch_size=64 \
  +trainer.accumulate_grad_batches=4 \
  +trainer.gradient_clip_val=1.0 \
  "callbacks.model_checkpoint.dirpath=${CKPT_DIR}"

echo "=== Pretrain end $(date) ==="
