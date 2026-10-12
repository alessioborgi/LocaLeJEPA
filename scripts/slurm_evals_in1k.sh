#!/bin/bash
#SBATCH --job-name=BLJ_evals
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# ImageNet-1K evaluation chain (in-domain k-NN/linear, 7-dataset transfer, RankMe,
# and the bobblehead rotation probe when configs/eval/yoda/<NAME>_in1k.yaml exists)
# for one finished pretrain, with the cluster's conda env (no uv here).
#   sbatch --export=ALL,NAME=hvit_localballreg_l01_k4 scripts/slurm_evals_in1k.sh
# Optional: MIN_EPOCH=99 (default) refuses to evaluate a last.ckpt below that epoch,
# so the job can be queued with --dependency on a still-running pretrain.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa
export PROJECT_ROOT=$PWD
: "${NAME:?set NAME=<backbone name as in configs/eval/*_in1k_ep100.yaml>}"
CKPT=$(python - "$NAME" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open("configs/eval/indomain_in1k_ep100.yaml"))
print(next(b["checkpoint"] for b in cfg["backbones"] if b["name"] == sys.argv[1]))
PY
)
EPOCH=$(python -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch'])" "$CKPT")
echo "=== $(date) | $NAME | $CKPT epoch $EPOCH ==="
if [ "$EPOCH" -lt "${MIN_EPOCH:-99}" ]; then echo "checkpoint below epoch ${MIN_EPOCH:-99}; not evaluating"; exit 1; fi
python src/eval_indomain.py --config configs/eval/indomain_in1k_ep100.yaml --only "$NAME"
python src/eval_transfer.py --config configs/eval/transfer_in1k_ep100.yaml --only "$NAME"
python src/eval_rankme.py --config configs/eval/rankme_in1k_ep100.yaml --only "$NAME"
python src/eval_indomain.py --config configs/eval/indomain_in1k_ep100.yaml --merge-only
python src/eval_transfer.py --config configs/eval/transfer_in1k_ep100.yaml --merge-only
python src/eval_rankme.py --config configs/eval/rankme_in1k_ep100.yaml --merge-only
if [ -f "configs/eval/yoda/${NAME}_in1k.yaml" ] && [ ! -f "logs/eval_angle_probe_yoda_${NAME}_in1k/results_multiseed.csv" ]; then
  python src/eval_angle_probe.py --config "configs/eval/yoda/${NAME}_in1k.yaml"
  python src/eval_angle_probe_multiseed.py --config "configs/eval/yoda/${NAME}_in1k.yaml" --n-seeds 10
fi
python paper/make_paper_tables.py
echo "=== end $(date) ==="
