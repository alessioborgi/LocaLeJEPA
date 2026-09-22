#!/usr/bin/env bash
# Run with bash, not sbatch. Submit two disjoint 90-task arrays.
# DRY_RUN=1 prints commands; MAX_CONCURRENT sets the limit PER array (default 1).
set -euo pipefail
cd "$(dirname "$0")/.."
limit="${MAX_CONCURRENT:-1}"
[[ "$limit" =~ ^[1-9][0-9]*$ ]] || { echo 'MAX_CONCURRENT must be a positive integer' >&2; exit 1; }
mkdir -p logs/slurm
for geometry in 0 1; do
  indices=""
  for ((cell=0; cell<30; cell++)); do
    start=$((cell * 6 + geometry * 3))
    indices+="${indices:+,}${start}-$((start + 2))"
  done
  if (( geometry == 0 )); then batch=64; memory=40G; else batch=256; memory=80G; fi
  command=(sbatch --job-name="fixed-k-B${batch}" --mem="$memory"
    --array="${indices}%${limit}" scripts/slurm_fixed_k_20260922.sh)
  printf '%q ' "${command[@]}"
  printf '\n'
  if [[ "${DRY_RUN:-0}" != 1 ]]; then "${command[@]}"; fi
done
