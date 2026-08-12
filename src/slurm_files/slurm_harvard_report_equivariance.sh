#!/bin/bash
#SBATCH -J BLJ_report_equiv
#SBATCH -p shared
#SBATCH -c 2
#SBATCH -t 0-00:20
#SBATCH --mem=8000
#SBATCH -o logs/BLJ_report_equiv_%j.out
#SBATCH -e logs/BLJ_report_equiv_%j.err

# Aggregate the four ImageNet-1K Yoda-probe evals into report_equivariance.md.
# Submitted with afterany on the four eval jobs so it runs once they have all
# finished (any exit state) and reports whatever results exist.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module purge
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa

echo "=== Report build $(date) ==="
python scripts/make_equivariance_report.py --out report_equivariance.md
echo "=== Done; report_equivariance.md ==="
cat report_equivariance.md
