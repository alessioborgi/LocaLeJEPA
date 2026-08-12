#!/bin/bash
# Launch the full ImageNet-1K backbone-comparison suite with Slurm dependency
# chaining:
#
#   stage ImageNet-1K (HF download -> netscratch ImageFolder)
#     -> 4 pretrains  (LeJEPA + RectifiedLpJEPA) x (D4-ResNet-50 + Hyperbolic ViT)
#       -> 4 chained Yoda probe evals
#
# PREREQUISITE (one-time, user action): accept the license at
# https://huggingface.co/datasets/ILSVRC/imagenet-1k and run
# `huggingface-cli login`. Without a token the staging job exits immediately
# and every downstream job stays pending, then dies on the failed dependency.
#
# If ImageNet is already staged (data/imagenet symlink valid), staging is
# skipped and pretrains start without a dependency.
#
# Run from the repo root:
#   bash src/slurm_files/launch_in1k_suite.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

DEP=""
if [ -d data/imagenet/imagenet_full_size/061417/train ]; then
  echo "ImageNet-1K already staged — skipping staging job."
else
  STAGE=$(sbatch --parsable src/slurm_files/slurm_harvard_stage_imagenet1k.sh)
  echo "staging: $STAGE"
  DEP="--dependency=afterok:$STAGE"
fi

declare -A EVAL_OF=(
  [slurm_harvard_pretrain_lejepa_d4rn50_in1k.sh]=slurm_harvard_eval_yoda_d4rn50_sigreg_in1k.sh
  [slurm_harvard_pretrain_rectifiedlpjepa_d4rn50_in1k.sh]=slurm_harvard_eval_yoda_d4rn50_rdmreg_in1k.sh
  [slurm_harvard_pretrain_lejepa_hvit_in1k.sh]=slurm_harvard_eval_yoda_hvit_sigreg_in1k.sh
  [slurm_harvard_pretrain_rectifiedlpjepa_hvit_in1k.sh]=slurm_harvard_eval_yoda_hvit_rdmreg_in1k.sh
)

for pretrain in "${!EVAL_OF[@]}"; do
  P=$(sbatch --parsable ${DEP} "src/slurm_files/${pretrain}")
  E=$(sbatch --parsable --dependency=afterok:"$P" "src/slurm_files/${EVAL_OF[$pretrain]}")
  echo "pretrain ${pretrain}: $P -> eval ${EVAL_OF[$pretrain]}: $E"
done

squeue -u "$USER" -o "%.10i %.30j %.2t %.12E %R"
