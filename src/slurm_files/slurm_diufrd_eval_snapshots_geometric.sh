#!/bin/bash
#SBATCH --job-name=blj-eval-snap-geom
#SBATCH --output=/HOME/garciare/BundLeJEPA/logs/embedor/eval_%x_%A_%a.log
#SBATCH --error=/HOME/garciare/BundLeJEPA/logs/embedor/eval_%x_%A_%a.log
#SBATCH --array=0-6
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=48GB
#SBATCH --partition=GPU
#SBATCH --nodelist=diufrd202
#SBATCH --chdir=/HOME/garciare/BundLeJEPA

# Learning curves over the Hyperbolic-ViT snapshots -- one array task per run
# (~32 snapshots each: GPU feature extraction + CPU curvature, ~15-25 min).
#
# The point of this job is the 100-EPOCH reading. The manuscript's Table 2
# reports its BallReg gains at 100 epochs; our Imagenette runs go to 800 and
# LocaLeReg ties SIGReg there. Reading both horizons off the SAME run decides
# whether the local advantage is a convergence-speed effect that has closed by
# 800, or was never present on this dataset at all.
#
#   sbatch src/slurm_files/slurm_diufrd_eval_snapshots_geometric.sh
#   sbatch --array=0,2,3,4 ...    # just the matched-recipe arms
#
# Then on the login node (or locally after rsync):
#   python scripts/snapshot_horizon_table.py --data logs/eval_snapshots_geometric
#
# NODE: diufrd202 (RTX 3080, 10GB) at batch_size 128 -- inference only, so the
# small card is enough and 203/205 stay free for pretraining. Override with
#   sbatch --nodelist=diufrd203 src/slurm_files/slurm_diufrd_eval_snapshots_geometric.sh
# Not 204: this rebuilds the encoder through the same torch build that has no
# sm_70 kernels.

set -euo pipefail
source "$HOME/.local/bin/env" 2>/dev/null || true
source .venv/bin/activate
export RICH_NO_COLOR=1 NO_COLOR=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/embedor

# Index map matches configs/eval_snapshots_geometric.yaml and the NAMES order of
# slurm_diufrd_pretrain_geometric_imagenette.sh is NOT the same -- this one
# follows the config file, so --array ranges here refer to the config's order.
RUNS=(hvit_sigreg_matched hvit_rdmreg_matched
      hvit_locale_k2 hvit_locale_k4 hvit_locale_k8
      hvit_sigreg_paper hvit_rdmreg_paper)
RUN="${RUNS[${SLURM_ARRAY_TASK_ID:?run as an array job}]}"

# eval_snapshots.py only warns and skips when a run has no snapshots, which
# would leave a silently empty learning curve. Fail here instead.
n_snap=$(ls "checkpoints/$RUN/snapshots/"snap_epoch*.ckpt 2>/dev/null | wc -l)
[[ $n_snap -gt 0 ]] || {
  echo "ERROR: no snapshots under checkpoints/$RUN/snapshots -- was this run"
  echo "       trained with callbacks=with_snapshots?"; exit 1;
}

# The three LocaLeReg arms were killed at epoch ~50 and resumed, so their
# snapshot series should be continuous across the restart. A gap means the
# resume did not pick up last.ckpt and the early epochs came from a different
# weight trajectory than the late ones -- which would silently corrupt exactly
# the 100-vs-800 comparison this job exists to make.
python - "$RUN" <<'PY'
import glob, os, re, sys
run = sys.argv[1]
eps = sorted(int(re.search(r"epoch(\d+)", os.path.basename(p)).group(1))
             for p in glob.glob(f"checkpoints/{run}/snapshots/snap_epoch*.ckpt"))
gaps = [(a, b) for a, b in zip(eps, eps[1:]) if b - a > 25]
print(f"{run}: {len(eps)} snapshots, epochs {eps[0]}..{eps[-1]}")
if gaps:
    print(f"  WARNING: gaps larger than the 25-epoch interval at {gaps} -- the "
          f"100-vs-800 comparison assumes one continuous trajectory.")
PY

echo "=== eval_snapshots $RUN start $(date) on $(hostname) -- $n_snap snapshots ==="
python eval_snapshots.py --config configs/eval_snapshots_geometric.yaml --only "$RUN"
echo "=== eval_snapshots $RUN end $(date) ==="
