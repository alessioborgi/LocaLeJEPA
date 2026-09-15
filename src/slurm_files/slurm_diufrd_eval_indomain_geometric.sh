#!/bin/bash
#SBATCH --job-name=blj-indomain-geom
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

# The ONLY GPU step for the geometric-backbone geometry study. Extracts frozen
# TRAIN (9469) and VAL (3925) encoder features for the 7 Hyperbolic ViT runs of
# slurm_diufrd_pretrain_geometric_imagenette.sh and runs both probes:
#
#     k-NN (%)    similarity-weighted vote over the train bank, T = 0.07
#     Linear (%)  linear head on frozen features, lr picked on a 10% split
#
#   -> logs/eval_indomain_imagenette_geometric/{results.csv,results.json}
#   -> logs/eval_indomain_imagenette_geometric/cache/<name>/imagenette_{train,val}_128_ep<N>.pt
#
# The cache is the input to eval_structure.py (t-SNE / UMAP / EmbedOR layouts,
# k-means-vs-spectral ARI, Ollivier-Ricci Delta-kappa and kappa-AUROC,
# fragmentation, purity) -- all CPU, all offline.
#
# 202 (10GB 3080) and 204 (V100, no kernels in this torch build) are excluded
# for the same reasons as the pretraining array.
#
#   sbatch src/slurm_files/slurm_diufrd_eval_indomain_geometric.sh
#
# AFTERWARDS, pull results + caches; everything below runs on the laptop:
#   rsync -avz garciare@diufrd200.unifr.ch:BundLeJEPA/logs/eval_indomain_imagenette_geometric/ \
#              ~/Documents/Code/BundLeJEPA/logs/eval_indomain_imagenette_geometric/
#
# Then run the structure eval TWICE -- once in each geometry. The encoder emits
# points inside the Poincare ball, so cosine reads only their direction and
# throws away the norm, which is where a hyperbolic model puts hierarchy depth.
# The difference between the two tables is the measurement of what the imposed
# geometry actually bought:
#   python eval_structure.py --cache logs/eval_indomain_imagenette_geometric/cache \
#                            --out   logs/eval_structure_geometric_cosine
#   python eval_structure.py --cache logs/eval_indomain_imagenette_geometric/cache \
#                            --out   logs/eval_structure_geometric_poincare \
#                            --metric poincare --hyp-c 0.1
#   python visualisations/plot_embedding_structure.py \
#       --data logs/eval_structure_geometric_poincare \
#       --fig-dir visualisations/figures/geometric_poincare
#
# Compare the two tables on loo_knn20, kmeans/spectral ARI, Delta-kappa and
# kappa-AUROC. NOT on the Table-4 knn*/linear columns: the hyperbolic path uses
# a majority vote and a tangent-space head (no cosine temperature has a
# hyperbolic equivalent), so those are comparable across backbones but not
# across metrics. results.csv above is the cosine Table 4 either way.

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

# Fail loudly on a missing checkpoint: eval_indomain.py skips unreadable
# backbones and carries on, which silently produces a short table.
missing=0
for c in checkpoints/hvit_sigreg_matched/best.ckpt checkpoints/hvit_sigreg_paper/best.ckpt \
         checkpoints/hvit_rdmreg_matched/best.ckpt checkpoints/hvit_rdmreg_paper/best.ckpt \
         checkpoints/hvit_locale_k{2,4,8}/best.ckpt; do
  [[ -f "$c" ]] || { echo "MISSING: $c"; missing=1; }
done
[[ $missing -eq 0 ]] || {
  echo "ERROR: the checkpoints above are missing. Either those runs have not"
  echo "       finished, or they were named differently -- check 'ls checkpoints/'"
  echo "       and edit configs/eval_indomain_imagenette_geometric.yaml."
  echo "       To evaluate only what exists: add --only <name>,<name> below."
  exit 1
}

echo "=== Table 4 (geometric): k-NN + linear, train bank -> val  $(date) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python eval_indomain.py --config configs/eval_indomain_imagenette_geometric.yaml

echo "=== done $(date) ==="
echo "results: logs/eval_indomain_imagenette_geometric/results.csv"
