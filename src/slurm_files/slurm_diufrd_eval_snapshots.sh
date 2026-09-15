#!/bin/bash
#SBATCH --job-name=blj-eval-snapshots
#SBATCH --output=/HOME/garciare/BundLeJEPA/logs/embedor/eval_%x_%A_%a.log
#SBATCH --error=/HOME/garciare/BundLeJEPA/logs/embedor/eval_%x_%A_%a.log
#SBATCH --array=0-8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=48GB
#SBATCH --partition=GPU
#SBATCH --nodelist=diufrd202
#SBATCH --chdir=/HOME/garciare/BundLeJEPA

# Learning-curve eval over the seeded k-sweep snapshots -- one array task per run
# (~32 snapshots each, GPU feature extraction + POT curvature gap, ~15-25 min).
#
#   sbatch src/slurm_files/slurm_diufrd_eval_snapshots.sh
#
# Then on the login node:
#   uv run python visualisations/plot_embedor_learning_curves.py
#
# NODE: diufrd202 (RTX 3080, 10GB). This job is inference-only feature
# extraction plus CPU curvature, so it does not need a big card -- 205 and 203
# stay free for pretraining. Override without editing this file:
#   sbatch --nodelist=diufrd203 src/slurm_files/slurm_diufrd_eval_snapshots.sh
# On a V100 node (204) nothing changes: no bf16 is used here, features are fp32.
#
# Array is 0-8: the nine BallJEPA runs at k in {2,4,8}. Tasks 9-11
# (lejepa_s0/s1/s2) stay in the config but are NOT scheduled -- the seeded LeJEPA
# arms of slurm_diufrd_pretrain_ksweep_seeded.sh were never run, so those
# checkpoint directories do not exist. Raise to 0-11 once they do.
#
# Tasks 12-17 are the k=16/32 extension. Run them once those pretrainings finish:
#   sbatch --array=12-17 src/slurm_files/slurm_diufrd_eval_snapshots.sh
# They feed the AULC(0-200) endpoint, which is the learning-SPEED half of the
# ablation; the final-accuracy half comes from eval_indomain.py. Both then go
# into scripts/ksweep_trend_test.py for the Jonckheere-Terpstra trend over k.

set -euo pipefail
source "$HOME/.local/bin/env" 2>/dev/null || true
source .venv/bin/activate
export RICH_NO_COLOR=1 NO_COLOR=1 OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/embedor

# Index map is IDENTICAL to slurm_diufrd_pretrain_ksweep_seeded.sh, so an array
# range means the same runs in both scripts.
RUNS=(ball_l01_k2_s0 ball_l01_k2_s1 ball_l01_k2_s2
      ball_l01_k4_s0 ball_l01_k4_s1 ball_l01_k4_s2
      ball_l01_k8_s0 ball_l01_k8_s1 ball_l01_k8_s2
      lejepa_s0 lejepa_s1 lejepa_s2
      ball_l01_k16_s0 ball_l01_k16_s1 ball_l01_k16_s2
      ball_l01_k32_s0 ball_l01_k32_s1 ball_l01_k32_s2)
RUN="${RUNS[${SLURM_ARRAY_TASK_ID:?run as an array job}]}"

# eval_snapshots.py only warns and skips when a run has no snapshots, which would
# leave a silently empty learning curve. Fail here instead.
n_snap=$(ls "checkpoints/$RUN/snapshots/"snap_epoch*.ckpt 2>/dev/null | wc -l)
[[ $n_snap -gt 0 ]] || {
  echo "ERROR: no snapshots under checkpoints/$RUN/snapshots -- was this run"
  echo "       trained with callbacks=with_snapshots?"; exit 1;
}
echo "=== eval_snapshots $RUN start $(date) on $(hostname) -- $n_snap snapshots ==="
python eval_snapshots.py --config configs/eval_snapshots_ksweep.yaml --only "$RUN"
echo "=== eval_snapshots $RUN end $(date) ==="
