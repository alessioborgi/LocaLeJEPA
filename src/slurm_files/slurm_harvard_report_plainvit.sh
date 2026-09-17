#!/bin/bash
#SBATCH -J BLJ_report_plainvit
#SBATCH -p gpu,gpu_h200,gpu_requeue
#SBATCH -c 8
#SBATCH -t 0-03:00
#SBATCH --mem=64000
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h200|a40|rtxa6000|rtx6000pro"
#SBATCH --requeue
#SBATCH -o logs/BLJ_report_plainvit_%j.out
#SBATCH -e logs/BLJ_report_plainvit_%j.err

# Final step of the plain-ViT IN1K pipeline: make sure the Yoda linear-regression
# probe (ridge regression of the bobblehead angles, 10 seeds) exists for every
# finished run, merge the in-domain k-NN + linear results, and write
# report_plainViT_LastExp.md. Submitted by slurm_harvard_plainvit_watchdog.sh
# with an afterany dependency on the in-domain eval array; safe to re-run by hand:
#
#   sbatch src/slurm_files/slurm_harvard_report_plainvit.sh

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module purge
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa

for v in sigreg localballreg_l01_k4; do
  CK=checkpoints/lejepa_vit_in1k_${v}/best.ckpt
  RES=logs/eval_angle_probe_yoda_vit_${v}_in1k/results_multiseed.json
  if [ -f "$CK" ] && [ ! -f "$RES" ]; then
    CFG=configs/eval_angle_probe_yoda_vit_${v}_in1k.yaml
    echo "=== Yoda regression probe missing for vit_${v}: running $(date) ==="
    python eval_angle_probe.py --config "$CFG" && \
    python eval_angle_probe_multiseed.py --config "$CFG" --n-seeds 10
  fi
done

python eval_indomain.py --config configs/eval_indomain_in1k_plainvit.yaml --merge-only || true
python scripts/make_plainvit_report.py
echo "=== report written $(date) ==="
