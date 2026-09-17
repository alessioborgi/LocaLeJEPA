#!/bin/bash
#SBATCH -J BLJ_plainvit_watchdog
#SBATCH -p shared
#SBATCH -c 1
#SBATCH -t 0-00:20
#SBATCH --mem=16000
#SBATCH -o logs/BLJ_plainvit_watchdog_%j.out
#SBATCH -e logs/BLJ_plainvit_watchdog_%j.err

# Waits for the two plain ViT-S/8 IN1K pretrains (BLJ_lejepa_vit_in1k and
# BLJ_vit_localballreg_l01_k4_in1k) to reach 100 epochs, then launches the
# in-domain k-NN + linear eval for each finished run and a report job that
# (re)writes report_plainViT_LastExp.md. Re-arms itself every hour; retires
# when all four in-domain rows and both Yoda regression probes exist and the
# report has been written after them.
#
#   sbatch src/slurm_files/slurm_harvard_plainvit_watchdog.sh

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module purge
source /n/sw/Mambaforge-22.11.1-4/etc/profile.d/conda.sh
conda activate topo-jepa

ME=$(whoami)
SELF=src/slurm_files/slurm_harvard_plainvit_watchdog.sh
EVAL=src/slurm_files/slurm_harvard_eval_indomain_in1k_plainvit.sh
REPORT=src/slurm_files/slurm_harvard_report_plainvit.sh
OUT=logs/eval_indomain_in1k_plainvit/per_backbone
MAX_EPOCH=99
declare -A IDX=([sigreg]="0,2" [localballreg_l01_k4]="1,3")

echo "=== plainvit watchdog $(date) ==="
# A queued/running BLJ_finish_plainvit (gpu_test one-shot: evals + probes + report,
# see slurm_harvard_finish_plainvit_gputest.sh) covers both the eval array and the
# report job, so treat it as "already queued" for either.
_queued() { squeue -u "$ME" -h -o "%j" | grep -qE "^($1|BLJ_finish_plainvit)$"; }
_epoch() {  # last.ckpt epoch, or -1
  python - "$1" <<'PY' 2>/dev/null || echo -1
import sys, torch
try:
    print(torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True).get("epoch", -1))
except Exception:
    print(-1)
PY
}

# ---- slot priority (user request 2026-09-09): after RDMReg, these two runs come
# first among OUR jobs. Implemented with Slurm nice values, not by killing runs:
# every pending BLJ pretrain job that is not in .pretrain_priority gets nice=$NICE,
# so whenever a GPU becomes available to this account the plain-ViT jobs are the
# ones Slurm picks. (Killing a running run to "free a slot" was tried on
# 2026-09-09 and does not work here: the freed GPU goes to the ~300 higher-
# fairshare jobs of other users, and the killed run just loses its slot.)
# Also: release a plain-ViT job if someone held it, and resubmit the pretrain if
# its job chain vanished from the queue.
declare -A JOBNAME=([sigreg]=BLJ_lejepa_vit_in1k [localballreg_l01_k4]=BLJ_vit_localballreg_l01_k4_in1k)
declare -A PRETRAIN=([sigreg]=src/slurm_files/slurm_harvard_pretrain_lejepa_vit_in1k.sh
                     [localballreg_l01_k4]=src/slurm_files/slurm_harvard_pretrain_lejepa_vit_in1k_localballreg_l01_k4.sh)
PROTECT=$(tr '\n' ' ' < .pretrain_priority 2>/dev/null)
NICE=1000
squeue -u "$ME" -h -t PENDING -o "%i|%j|%n" 2>/dev/null | grep "|BLJ_" | grep -vE "\|BLJ_(eval|ev_|report_)|watchdog" \
  | awk -F'|' -v p="$PROTECT" 'BEGIN{n=split(p,a," "); for(i=1;i<=n;i++) prot[a[i]]=1} !($2 in prot) {print $1"|"$2}' \
  | while IFS='|' read -r id nm; do
      cur=$(sprio -j "$id" -h -o "%N" 2>/dev/null | tr -d ' ')
      if [ "${cur:-0}" != "$NICE" ]; then
        scontrol update JobId="$id" Nice=$NICE 2>/dev/null && echo "demoted $nm ($id) to nice=$NICE"
      fi
    done
for v in sigreg localballreg_l01_k4; do
  nm=${JOBNAME[$v]}
  ep=$(_epoch "checkpoints/lejepa_vit_in1k_${v}/last.ckpt")
  [ "$ep" -ge "$MAX_EPOCH" ] && continue
  if squeue -u "$ME" -h -t R -o "%j" | grep -qx "$nm"; then echo "$nm running"; continue; fi
  pend=$(squeue -u "$ME" -h -t PENDING -o "%i|%j|%r" | awk -F'|' -v n="$nm" '$2==n')
  if [ -z "$pend" ]; then
    echo "$nm has NO job in the queue (chain died?) -> resubmitting ${PRETRAIN[$v]}"
    sbatch "${PRETRAIN[$v]}" && continue
  fi
  for id in $(echo "$pend" | grep -i held | cut -d'|' -f1); do
    scontrol release "$id" 2>/dev/null && echo "  released held $nm job $id"
  done
  echo "$nm pending: $(echo "$pend" | cut -d'|' -f1,3 | tr '\n' ' ')"
done

need_report=0; all_done=1
for v in sigreg localballreg_l01_k4; do
  CKDIR=checkpoints/lejepa_vit_in1k_${v}
  ep=$(_epoch "$CKDIR/last.ckpt")
  have_final=$([ -f "$OUT/vit_${v}.json" ] && [ -f "$OUT/vit_${v}_best.json" ] && echo 1 || echo 0)
  have_yoda=$([ -f "logs/eval_angle_probe_yoda_vit_${v}_in1k/results_multiseed.json" ] && echo 1 || echo 0)
  echo "vit_${v}: last.ckpt epoch=${ep} indomain_done=${have_final} yoda_done=${have_yoda}"
  if [ "$ep" -lt "$MAX_EPOCH" ]; then all_done=0; continue; fi
  if [ "$have_final" -eq 0 ]; then
    all_done=0
    if _queued BLJ_eval_indomain_in1k_plainvit; then
      echo "  in-domain eval already queued/running"
    else
      J=$(sbatch --parsable --array="${IDX[$v]}" "$EVAL") && echo "  submitted in-domain eval array $J (tasks ${IDX[$v]})"
      _queued BLJ_report_plainvit || { R=$(sbatch --parsable --dependency=afterany:"$J" "$REPORT") && echo "  submitted report job $R (afterany:$J)"; }
    fi
  fi
  [ "$have_yoda" -eq 0 ] && { all_done=0; need_report=1; }
done

if [ "$all_done" -eq 1 ]; then
  if [ -f report_plainViT_LastExp.md ] && [ report_plainViT_LastExp.md -nt "$OUT/vit_localballreg_l01_k4_best.json" ] \
     && [ report_plainViT_LastExp.md -nt "$OUT/vit_sigreg_best.json" ]; then
    echo "All evals done and report is current; watchdog retiring."; exit 0
  fi
  _queued BLJ_report_plainvit || { R=$(sbatch --parsable "$REPORT") && echo "submitted final report job $R"; }
elif [ "$need_report" -eq 1 ] && ! _queued BLJ_report_plainvit && ! _queued BLJ_eval_indomain_in1k_plainvit; then
  # In-domain rows exist but a Yoda probe is missing (e.g. its auto-launched eval
  # failed): the report job runs the probe itself.
  R=$(sbatch --parsable "$REPORT") && echo "submitted report job $R to fill the missing Yoda probe"
fi

for i in 1 2 3; do
  sbatch --begin=now+1hour "$SELF" >/dev/null 2>&1 && { echo "re-armed (+1 h)"; exit 0; }
  sleep 30
done
echo "WARNING: could not re-arm watchdog"; exit 1
