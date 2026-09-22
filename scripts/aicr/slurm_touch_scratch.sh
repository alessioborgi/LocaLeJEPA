#!/bin/bash
# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Keep the campaign's datasets alive on /scratch: reset every file's mtime under TOUCH_ROOTS and
# resubmit this job 25 days out (F49: the purge deletes files whose mtime is older than 30 days and
# sweeps daily; extracted datasets carry their archives' old mtimes and vanish at the next sweep).
# Companion of slurm_refresh_tar.sh, which keeps the campaign tar on a fresh inode.
#
#   sbatch -A <account> -p cpu -c 1 --mem=2G -t 06:00:00 -o <logs>/touch_scratch-%j.out \
#          --export=NONE,TOUCH_ROOTS=<dir>:<dir>,ACCOUNT=<account>,LOGS=<logs>,REPO=<clone> \
#          <clone>/scripts/aicr/slurm_touch_scratch.sh
set -euo pipefail
: "${TOUCH_ROOTS:?TOUCH_ROOTS=<dir>:<dir>... must be exported}"
: "${ACCOUNT:?ACCOUNT must be exported}" "${LOGS:?LOGS must be exported}" "${REPO:?REPO must be exported}"
IFS=':' read -r -a ROOTS <<< "$TOUCH_ROOTS"
echo "=== $(date -Is) | touch ${ROOTS[*]} | host $(hostname) ==="
for root in "${ROOTS[@]}"; do
    case "$root" in /tmp/*) echo "REFUSING: $root is node-local (F39)" >&2; exit 2 ;; esac
    [ -d "$root" ] || { echo "REFUSING: $root is not a directory -- was it purged already?" >&2; exit 3; }
    n=$(find "$root" -exec touch -h {} + -print | wc -l)
    echo "touched $n entries under $root"
done
echo "TOUCHED $(date -Is)"
NEXT=$(sbatch --parsable -A "$ACCOUNT" -p cpu -c 1 --mem=2G -t 06:00:00 -J touch_scratch -o "$LOGS/touch_scratch-%j.out" \
       --begin=now+25days --export=NONE,TOUCH_ROOTS="$TOUCH_ROOTS",ACCOUNT="$ACCOUNT",LOGS="$LOGS",REPO="$REPO" \
       "$REPO/scripts/aicr/slurm_touch_scratch.sh")
echo "next touch queued as job $NEXT (begins in 25 days)"
