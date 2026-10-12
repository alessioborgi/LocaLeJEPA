#!/bin/bash
#SBATCH --job-name=BLJ_seeds
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# Re-run the frozen-feature LINEAR probes of the ImageNet-1K study over five
# seeds, reusing the cached epoch-99 features (nothing is re-extracted and no
# encoder is retrained). The probe learning rate is selected once as before;
# the five seeds vary the head initialisation, the data order and the held-out
# LR-selection slice, so the reported spread is readout variability.
# k-NN is deterministic given the feature bank and is unchanged.
#
#   sbatch scripts/slurm_eval_seeds_in1k.sh              # everything
#   sbatch --export=ALL,ONLY=vit_sigreg scripts/slurm_eval_seeds_in1k.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa
export PROJECT_ROOT=$PWD
ONLY_ARG=${ONLY:+--only $ONLY}
echo "=== $(date) | five-seed linear probes | $(hostname) ==="
python src/eval_indomain.py --config configs/eval/indomain_in1k_ep100.yaml   ${ONLY_ARG}
python src/eval_indomain.py --config configs/eval/indomain_in1k_plainvit.yaml ${ONLY_ARG}
python src/eval_transfer.py  --config configs/eval/transfer_in1k_ep100.yaml   ${ONLY_ARG} --force
python paper/make_paper_tables.py
echo "=== end $(date) ==="
