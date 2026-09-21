#!/bin/bash
# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Download the seven transfer datasets as a batch job. Compute nodes have outbound internet
# here (runbook §8) and the login node SIGKILLs sustained downloads, so "run on a login node"
# was the wrong instruction on this cluster. The job ends with the structural check
# (`--verify`): every dataset re-opened offline, its class count and split sizes recorded in
# `.verified` -- the existence-only `.ready` marker was the kind of gate research_rules/05 warns
# about.
#
#   sbatch -A <account> -p cpu -c 2 --mem=16G -t 06:00:00 -o <logs>/transfer-%j.out \
#          --export=NONE,REPO=<clone>,TRANSFER_ROOT=<dir> <clone>/scripts/aicr/slurm_download_transfer.sh
set -euo pipefail
: "${REPO:?REPO=<clone> must be exported}" "${TRANSFER_ROOT:?TRANSFER_ROOT=<dir> must be exported}"
[ -f "$REPO/scripts/aicr/roots.env" ] || { echo "missing $REPO/scripts/aicr/roots.env" >&2; exit 2; }
# shellcheck disable=SC1091
source "$REPO/scripts/aicr/roots.env"
[ "${PY:-CHANGE_ME}" != "CHANGE_ME" ] || { echo "roots.env still has PY=CHANGE_ME" >&2; exit 2; }
case "$TRANSFER_ROOT" in /tmp/*) echo "REFUSING: TRANSFER_ROOT under /tmp is node-local (F39)" >&2; exit 2 ;; esac
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DOWNLOAD_TIMEOUT=30
cd "$REPO"
echo "=== $(date -Is) | transfer datasets -> $TRANSFER_ROOT | host $(hostname) ==="
mkdir -p "$TRANSFER_ROOT"
time env PYTHONPATH="$REPO" "$PY" scripts/download_transfer_datasets.py --root "$TRANSFER_ROOT"
echo "=== verifying offline ==="
env PYTHONPATH="$REPO" "$PY" scripts/download_transfer_datasets.py --root "$TRANSFER_ROOT" --verify
echo "TRANSFER DATASETS VERIFIED: $TRANSFER_ROOT $(date -Is)"
