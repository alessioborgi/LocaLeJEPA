#!/usr/bin/env bash
set -euo pipefail

# Pretrain the four ViT-S/8 / Imagenette backbones compared in the EmbedOR
# class-separation study (configs/eval_embedor_imagenette.yaml):
#
#   lejepa            lejepa/vit_imagenette_sigreg               (SIGReg)
#   rectlpjepa        rectifiedlpjepa/vit_imagenette_rdmreg      (RDMReg)
#   ball_l01_k2       lejepa/vit_imagenette_localballreg_l01_k2  (local BallReg, k=2)
#   ball_l01_k4       lejepa/vit_imagenette_localballreg_l01_k4  (local BallReg, k=4)
#
# Each run: 800 epochs, global batch 256, ViT-S/8 @ 128px, online linear probe.
# Checkpoints -> checkpoints/<name>/{best,last}.ckpt (best = max val/probe_acc).
# Auto-resumes from last.ckpt if present.
#
# Usage
# -----
#   # one GPU per run, in parallel, backgrounded (plain multi-GPU box):
#   scripts/pretrain_vit_imagenette_embedor.sh --gpus 0,1,2,3
#
#   # single GPU, runs sequentially:
#   scripts/pretrain_vit_imagenette_embedor.sh --gpus 0
#
#   # just print the commands:
#   scripts/pretrain_vit_imagenette_embedor.sh --dry-run
#
#   # extra Hydra overrides pass straight through (e.g. smaller batch on a 3080):
#   scripts/pretrain_vit_imagenette_embedor.sh --gpus 0,1 data.batch_size=128 +trainer.accumulate_grad_batches=2
#
# SLURM: submit this once per run instead, e.g.
#   srun --gres=gpu:1 uv run python src/train.py experiment=lejepa/vit_imagenette_sigreg \
#     +trainer.gradient_clip_val=1.0 logger=csv \
#     callbacks.model_checkpoint.dirpath=checkpoints/lejepa

NAMES=(lejepa rectlpjepa ball_l01_k2 ball_l01_k4)
EXPERIMENTS=(
  lejepa/vit_imagenette_sigreg
  rectifiedlpjepa/vit_imagenette_rdmreg
  lejepa/vit_imagenette_localballreg_l01_k2
  lejepa/vit_imagenette_localballreg_l01_k4
)

GPUS=""
DRY_RUN=0
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

cd "$(dirname "$0")/.."
mkdir -p logs checkpoints

IFS=',' read -r -a GPU_ARR <<< "${GPUS:-}"
[[ ${#GPU_ARR[@]} -eq 0 ]] && GPU_ARR=("")   # no --gpus: let the framework pick

run_one() {
  local name="$1" exp="$2" gpu="$3"
  local ckpt_dir="checkpoints/${name}"
  local resume=""
  [[ -f "${ckpt_dir}/last.ckpt" ]] && resume="ckpt_path=${ckpt_dir}/last.ckpt"

  local -a cmd=(
    uv run python src/train.py
    "experiment=${exp}"
    +trainer.gradient_clip_val=1.0
    logger=csv
    "callbacks.model_checkpoint.dirpath=${ckpt_dir}"
    ${resume:+$resume}
    ${EXTRA[@]+"${EXTRA[@]}"}
  )

  local env_prefix=""
  [[ -n "$gpu" ]] && env_prefix="CUDA_VISIBLE_DEVICES=${gpu} "

  echo "=== ${name} (${exp}) ${gpu:+on GPU ${gpu}} ==="
  echo "${env_prefix}${cmd[*]}"
  [[ $DRY_RUN -eq 1 ]] && return 0

  if [[ -n "$gpu" && ${#GPU_ARR[@]} -gt 1 ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" > "logs/pretrain_${name}.log" 2>&1 &
  else
    ${env_prefix:+env CUDA_VISIBLE_DEVICES=$gpu} "${cmd[@]}" 2>&1 | tee "logs/pretrain_${name}.log"
  fi
}

for i in "${!EXPERIMENTS[@]}"; do
  gpu="${GPU_ARR[$(( i % ${#GPU_ARR[@]} ))]}"
  run_one "${NAMES[$i]}" "${EXPERIMENTS[$i]}" "$gpu"
done

if [[ $DRY_RUN -eq 0 && ${#GPU_ARR[@]} -gt 1 ]]; then
  echo "=== all runs launched in background; tail logs/pretrain_*.log ==="
  wait
  echo "=== all runs finished ==="
fi
