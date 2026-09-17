#!/bin/bash
#SBATCH --job-name=blj-indomain-ksweep
#SBATCH --output=/HOME/garciare/BundLeJEPA/logs/embedor/eval_%x_%j.log
#SBATCH --error=/HOME/garciare/BundLeJEPA/logs/embedor/eval_%x_%j.log
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --mem=64GB
#SBATCH --partition=GPU
#SBATCH --exclude=diufrd202,diufrd204
#SBATCH --chdir=/HOME/garciare/BundLeJEPA

# SPHERE-JEPA Table-4 evaluation of the CURRENT Imagenette checkpoints, and the
# only GPU step in the pipeline. Extracts TRAIN (9469) and VAL (3925) frozen
# ENCODER features for every backbone and runs the two probes:
#
#     k-NN (%)    similarity-weighted vote over the train bank, T = 0.07
#     Linear (%)  linear head on frozen features, lr picked on a 10% split
#
#   -> logs/eval_indomain_imagenette_ksweep/{results.csv,results.json}
#   -> logs/eval_indomain_imagenette_ksweep/cache/<name>/imagenette_{train,val}_128_ep<N>.pt
#
# Those caches are also the input to eval_structure.py, which computes the
# layouts and cluster metrics on CPU -- so this is the only job that needs a GPU.
#
# Caches are keyed by checkpoint EPOCH, so a re-evaluated backbone can never
# silently reuse stale features -- the failure mode that produced the bogus
# "collapsed" ball_l01_k8 row in the old metrics.csv.
#
# PREREQUISITES (login node):
#   uv sync
#   ls checkpoints/{lejepa,rectlpjepa}/best.ckpt
#   ls checkpoints/ball_l01_k{2,4,8,16,32}_s{0,1,2}/best.ckpt
#
# NODE: was pinned to diufrd205; now excludes 202 and 204, NEITHER of which can
# run this torch build -- 204 is the V100 (no sm_70 kernels in a cu13 wheel) and
# 202's driver reports CUDA 12080 against a build needing >= 580, where torch
# silently falls back to CPU (measured 2026-09-16, job 23298).
#
#   sbatch src/slurm_files/slurm_diufrd_eval_indomain_ksweep.sh
#
# AFTERWARDS, pull the results + caches and everything else runs offline:
#   rsync -avz garciare@diufrd200.unifr.ch:BundLeJEPA/logs/eval_indomain_imagenette_ksweep/ \
#              ~/Documents/Code/BundLeJEPA/logs/eval_indomain_imagenette_ksweep/
#   python eval_structure.py
#   python visualisations/plot_embedding_structure.py

set -euo pipefail

source "$HOME/.local/bin/env" 2>/dev/null || true
source .venv/bin/activate

export RICH_NO_COLOR=1
export NO_COLOR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs/embedor
test -d data/imagenette/imagenette2-160/train || {
  echo "ERROR: Imagenette not staged at data/imagenette/imagenette2-160/."; exit 1;
}
python -c "import torch, sklearn, timm, hydra" || {
  echo "ERROR: env incomplete -- run 'uv sync' on the login node."; exit 1;
}

# Fail loudly and early on a missing checkpoint rather than silently producing a
# short table: eval_indomain.py skips unreadable backbones and carries on.
missing=0
for c in checkpoints/lejepa/best.ckpt checkpoints/rectlpjepa/best.ckpt \
         checkpoints/ball_l01_k2_s{0,1,2}/best.ckpt \
         checkpoints/ball_l01_k4_s{0,1,2}/best.ckpt \
         checkpoints/ball_l01_k8_s{0,1,2}/best.ckpt \
         checkpoints/ball_l01_k16_s{0,1,2}/best.ckpt \
         checkpoints/ball_l01_k32_s{0,1,2}/best.ckpt; do
  [[ -f "$c" ]] || { echo "MISSING: $c"; missing=1; }
done
[[ $missing -eq 0 ]] || { echo "ERROR: checkpoints above are missing; fix the config first."; exit 1; }

echo "=== Table 4: k-NN + linear, train bank -> val  $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# ONLY lets you evaluate just the new arms. eval_indomain.py rebuilds
# results.csv from every per_backbone/*.json on disk, not only the ones it just
# ran, so a restricted run still writes the COMPLETE table -- and the 11 banked
# backbones keep their existing numbers instead of having their linear probe
# (a 7-point lr grid x 100 epochs) recomputed. Extraction would be skipped
# anyway, since the feature cache is keyed by checkpoint epoch; the probe is
# what costs the time.
#   ONLY=ball_k16_s0,ball_k16_s1,ball_k16_s2,ball_k32_s0,ball_k32_s1,ball_k32_s2 \
#     sbatch src/slurm_files/slurm_diufrd_eval_indomain_ksweep.sh
ONLY_ARG=()
[[ -n "${ONLY:-}" ]] && ONLY_ARG=(--only "$ONLY")
python eval_indomain.py --config configs/eval_indomain_imagenette_ksweep.yaml "${ONLY_ARG[@]}"

echo "=== done $(date) ==="
echo "results: logs/eval_indomain_imagenette_ksweep/results.csv"
