#!/bin/bash
#SBATCH -J BLJ_finish_plainvit
#SBATCH -p gpu_test
#SBATCH -c 8
#SBATCH -t 0-12:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH -o logs/BLJ_finish_plainvit_%j.out
#SBATCH -e logs/BLJ_finish_plainvit_%j.err

# One-shot "finish" job for the plain ViT-S/8 IN1K experiment on gpu_test (starts
# immediately, 12 h, but QOS allows only 2 submitted jobs -> everything runs
# sequentially in this single job): for every run whose last.ckpt reached epoch
# 99, run the in-domain k-NN + linear classification eval (final + best ckpt),
# the Yoda ridge-regression probe (10 seeds), then merge and write
# report_plainViT_LastExp.md. Runs still pretraining are skipped and the job
# exits 1 so the hourly watchdog re-submits the missing pieces later.
#
#   sbatch --dependency=afterany:<pretrain jobs> src/slurm_files/slurm_harvard_finish_plainvit_gputest.sh

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module purge
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa

CFG=configs/eval_indomain_in1k_plainvit.yaml
OUT=logs/eval_indomain_in1k_plainvit/per_backbone
_epoch() { python -c "import torch; print(torch.load('$1', map_location='cpu', weights_only=False, mmap=True).get('epoch', -1))" 2>/dev/null || echo -1; }

echo "=== finish plainvit start $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
incomplete=0
for v in sigreg localballreg_l01_k4; do
  CKDIR=checkpoints/lejepa_vit_in1k_${v}
  ep=$(_epoch "$CKDIR/last.ckpt")
  if [ "$ep" -lt 99 ]; then echo "--- vit_${v}: last.ckpt at epoch ${ep} < 99, skipping"; incomplete=1; continue; fi
  for name in vit_${v} vit_${v}_best; do
    if [ -f "$OUT/${name}.json" ]; then echo "--- $name already evaluated"; continue; fi
    echo "--- in-domain eval $name $(date)"
    python eval_indomain.py --config "$CFG" --only "$name" || { echo "!!! $name failed"; incomplete=1; }
  done
  RES=logs/eval_angle_probe_yoda_vit_${v}_in1k/results_multiseed.json
  if [ ! -f "$RES" ]; then
    YCFG=configs/eval_angle_probe_yoda_vit_${v}_in1k.yaml
    echo "--- Yoda regression probe vit_${v} $(date)"
    python eval_angle_probe.py --config "$YCFG" && python eval_angle_probe_multiseed.py --config "$YCFG" --n-seeds 10 \
      || { echo "!!! Yoda probe vit_${v} failed"; incomplete=1; }
  fi
done

python eval_indomain.py --config "$CFG" --merge-only || true
python scripts/make_plainvit_report.py
python scripts/make_equivariance_report.py --out report_equivariance.md || true
echo "=== finish plainvit end $(date) (incomplete=${incomplete}) ==="
exit $incomplete
