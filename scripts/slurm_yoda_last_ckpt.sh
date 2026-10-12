#!/bin/bash
#SBATCH --job-name=BLJ_yoda_last
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# Re-run the bobblehead (Yoda) rotation probe of every finished ImageNet-1K
# pretrain on its FINAL checkpoint (last.ckpt, epoch 99) instead of the
# online-probe-selected best.ckpt, so that every IN1K readout in the paper
# uses the same epoch-100 weights. The previous best.ckpt results were moved
# to logs/eval_angle_probe_yoda_<name>_in1k_bestckpt/ for the sensitivity check.
#
#   sbatch --partition=gpu_test --time=06:00:00 scripts/slurm_yoda_last_ckpt.sh   # one job, all 15 sequentially
#   sbatch --array=0-14%5 scripts/slurm_yoda_last_ckpt.sh                        # or one task per config on gpu
#   sbatch --dependency=afterok:<array job id> --wrap "python paper/make_paper_tables.py"  (done by the launcher below)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa
export PROJECT_ROOT=$PWD

CONFIGS=(
  configs/eval/yoda/d4rn50_sigreg_in1k.yaml
  configs/eval/yoda/d4rn50_localballreg_l01_k2_in1k.yaml
  configs/eval/yoda/d4rn50_localballreg_l01_k4_in1k.yaml
  configs/eval/yoda/d4rn50_localballreg_l03_k2_in1k.yaml
  configs/eval/yoda/d4rn50_localsigreg_l01_k4_in1k.yaml
  configs/eval/yoda/d4rn50_localballreg_l01_dynk_in1k.yaml
  configs/eval/yoda/hvit_sigreg_in1k.yaml
  configs/eval/yoda/hvit_rdmreg_in1k.yaml
  configs/eval/yoda/hvit_localballreg_l01_k2_in1k.yaml
  configs/eval/yoda/hvit_localballreg_l01_k4_in1k.yaml
  configs/eval/yoda/hvit_localballreg_l03_k2_in1k.yaml
  configs/eval/yoda/hvit_localsigreg_l01_k4_in1k.yaml
  configs/eval/yoda/hvit_localballreg_l01_dynk_in1k.yaml
  configs/eval/yoda/vit_sigreg_in1k.yaml
  configs/eval/yoda/vit_localballreg_l01_k4_in1k.yaml
)
run_one() {
  CFG=$1
  CKPT=$(grep -o 'checkpoints/[^ ]*' "$CFG")
  EPOCH=$(python -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False, mmap=True)['epoch'])" "$CKPT")
  echo "=== $(date) | $CFG | $CKPT epoch $EPOCH | $(hostname) ==="
  case "$CKPT" in *last.ckpt) ;; *) echo "refusing: config does not point at last.ckpt"; return 1;; esac
  if [ "$EPOCH" -lt "${MIN_EPOCH:-99}" ]; then echo "checkpoint below epoch ${MIN_EPOCH:-99}; not evaluating"; return 1; fi
  OUT=$(grep '^output_folder:' "$CFG" | awk '{print $2}')
  if ls "$OUT"/cache/*_features.pt >/dev/null 2>&1; then echo "stale feature cache in $OUT/cache — remove it first"; return 1; fi
  python src/eval_angle_probe.py --config "$CFG"
  python src/eval_angle_probe_multiseed.py --config "$CFG" --n-seeds 10
  echo "=== done $CFG $(date) ==="
}
# As a Slurm array each task evaluates one config; submitted without --array
# (e.g. on gpu_test, whose QOS caps submitted jobs at 2) one job runs them all.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  run_one "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
else
  FAILED=()
  for CFG in "${CONFIGS[@]}"; do
    if [ -f "$(grep '^output_folder:' "$CFG" | awk '{print $2}')/results_multiseed.csv" ]; then echo "skip (done): $CFG"; continue; fi
    run_one "$CFG" || FAILED+=("$CFG")
  done
  if [ ${#FAILED[@]} -gt 0 ]; then printf 'FAILED: %s\n' "${FAILED[@]}"; exit 1; fi
fi
echo "=== end $(date) ==="
