#!/bin/bash
# Copyright (c) 2026 Alessio Borgi, Pavel Snopov, Kartik Tendon, Ines Garcia Redondo, Randall Balestriero, Claudio Battiloro
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Pack the verified ImageNet-1K ImageFolder tree into ONE uncompressed tar beside it, so that
# every training job stages the dataset with a single sequential read instead of 1.28 M opens
# on NFS (F8, F39). Uncompressed on purpose: JPEGs do not compress and the point is one stream.
#
# The tar is not trusted on the exit code of `tar`: its entry count must equal the tree's file
# count and its sha256 is written beside it, which the refresh job re-checks (plan G0).
#
# Resources go on the sbatch line (no header defaults in this directory):
#   sbatch -A <account> -p cpu -c 2 --mem=16G -t 08:00:00 -o <logs>/tar_in1k-%j.out \
#          --export=NONE,SRC=<dir holding imagenet_full_size>,TAR=<dir>/imagenet1k_061417.tar \
#          <clone>/scripts/aicr/slurm_tar_in1k.sh
set -euo pipefail
: "${SRC:?SRC=<directory that contains imagenet_full_size/> must be exported}"
: "${TAR:?TAR=<path of the tar to write> must be exported}"
case "$TAR" in /tmp/*) echo "REFUSING: TAR under /tmp is node-local and vanishes with the job (F39)" >&2; exit 2 ;; esac
[ -d "$SRC/imagenet_full_size/061417/train" ] || { echo "no $SRC/imagenet_full_size/061417/train -- is the tree verified?" >&2; exit 2; }

# One writer. F44: the same block was submitted twice and two `tar -cf` streams wrote into
# ONE `$TAR.tmp` (O_TRUNC, then interleaved) -- research_rules/05's live-duplicate trap. `mkdir`
# is atomic on NFS where `flock` is not (research_rules/01).
LOCK="$TAR.lock"
mkdir "$LOCK" 2>/dev/null || { echo "REFUSING: another tar job holds $LOCK ($(cat "$LOCK/owner" 2>/dev/null))" >&2; exit 2; }
echo "job=${SLURM_JOB_ID:-none} host=$(hostname) $(date -Is)" > "$LOCK/owner"
trap 'rm -rf "$LOCK"' EXIT
[ -e "$TAR.tmp" ] && { echo "REFUSING: $TAR.tmp exists -- a writer is running or died; remove it by hand if the latter" >&2; exit 2; }
echo "=== $(date -Is) | tar $SRC/imagenet_full_size -> $TAR | host $(hostname) ==="
df -h "$(dirname "$TAR")" | tail -1
cd "$SRC"
echo "counting the tree (1.28 M files on NFS: minutes) ..."
FILES=$(find imagenet_full_size | wc -l)
echo "tree entries: $FILES"
time tar -cf "$TAR.tmp" imagenet_full_size
ENTRIES=$(tar -tf "$TAR.tmp" | wc -l)
echo "tar entries : $ENTRIES"
[ "$ENTRIES" -eq "$FILES" ] || { echo "MISMATCH: the tar has $ENTRIES entries, the tree $FILES; leaving $TAR.tmp for inspection" >&2; exit 1; }
( cd "$(dirname "$TAR")" && sha256sum "$(basename "$TAR.tmp")" | sed "s/$(basename "$TAR.tmp")/$(basename "$TAR")/" > "$TAR.sha256" )
mv "$TAR.tmp" "$TAR"
ls -la "$TAR" "$TAR.sha256"
echo "TAR READY: $TAR entries=$ENTRIES sha256=$(cut -d' ' -f1 "$TAR.sha256") $(date -Is)"
