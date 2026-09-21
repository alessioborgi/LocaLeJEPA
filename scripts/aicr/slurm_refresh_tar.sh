#!/bin/bash
# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Keep the ImageNet-1K tar alive across the 30-day scratch purge whose clock (atime/mtime/ctime)
# is undocumented (research_rules/01; canary answers 2026-10-20). A `touch` protects against
# none of the clocks a site might use; a COPY to a new inode is fresh under every one of them.
# The copy is verified against the recorded sha256 before it replaces the original, and the job
# re-submits itself for 25 days later, so the calendar is in Slurm, not in someone's memory.
#
#   sbatch -A <account> -p cpu -c 1 --mem=4G -t 04:00:00 -o <logs>/refresh_tar-%j.out \
#          --export=NONE,TAR=<path>.tar,REPO=<clone>,ACCOUNT=<account>,LOGS=<logs> \
#          <clone>/scripts/aicr/slurm_refresh_tar.sh
set -euo pipefail
: "${TAR:?TAR=<the tar> must be exported}" "${REPO:?REPO=<clone> must be exported}"
: "${ACCOUNT:?ACCOUNT must be exported}" "${LOGS:?LOGS=<log dir> must be exported}"
[ -f "$TAR" ] || { echo "REFUSING: $TAR does not exist -- was it purged? re-create it with slurm_tar_in1k.sh" >&2; exit 2; }
[ -f "$TAR.sha256" ] || { echo "REFUSING: $TAR.sha256 missing; the copy could not be verified" >&2; exit 2; }

echo "=== $(date -Is) | refresh $TAR | host $(hostname) ==="
ls -la --time-style=full-iso "$TAR"
time cp "$TAR" "$TAR.refresh"                  # a NEW inode with a fresh atime/mtime/ctime
( cd "$(dirname "$TAR")" && sed "s#$(basename "$TAR")\$#$(basename "$TAR.refresh")#" "$TAR.sha256" | sha256sum -c - )
mv "$TAR.refresh" "$TAR"                        # atomic replace; the old inode is released
ls -la --time-style=full-iso "$TAR"
echo "REFRESHED: $TAR $(date -Is)"

NEXT=$(sbatch --parsable -A "$ACCOUNT" -p cpu -c 1 --mem=4G -t 04:00:00 --begin=now+25days \
       -J refresh_tar -o "$LOGS/refresh_tar-%j.out" \
       --export=NONE,TAR="$TAR",REPO="$REPO",ACCOUNT="$ACCOUNT",LOGS="$LOGS" \
       "$REPO/scripts/aicr/slurm_refresh_tar.sh")
echo "next refresh queued as job $NEXT (begins in 25 days; cancel it if the tar is retired)"
