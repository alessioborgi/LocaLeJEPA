#!/bin/bash
#SBATCH -J BLJ_rdmreg_watchdog
#SBATCH -p shared
#SBATCH -c 1
#SBATCH -t 0-00:10
#SBATCH --mem=2000
#SBATCH -o logs/BLJ_rdmreg_watchdog_%j.out
#SBATCH -e logs/BLJ_rdmreg_watchdog_%j.err

# Guarantees the D4 RDMReg run always holds a running slot until it finishes.
# Every ~20 min: if RDMReg is neither finished nor running, free one slot by
# stopping another run (its successor is HELD first so the afterany successor
# does not immediately start), letting RDMReg's pending job claim the GPU.
# Stopped runs lose nothing -- they resume from last.ckpt when released.
#
# Self-re-arms until RDMReg is done, then exits without rescheduling.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
ME=$(whoami)
RDM_NAME=BLJ_rectifiedlpjepa_d4rn50_in1k
EVAL_RESULT=logs/eval_angle_probe_yoda_d4rn50_rdmreg_in1k/results_multiseed.json
SELF=src/slurm_files/slurm_harvard_rdmreg_watchdog.sh
STAMP=.rdmreg_watchdog_last_free
COOLDOWN=2700          # >=45 min between slot-frees, so we never mass-kill runs

echo "=== watchdog $(date) ==="

# 1) RDMReg finished (eval produced) -> stop watching, do not re-arm.
if [ -f "$EVAL_RESULT" ]; then
  echo "RDMReg finished and evaluated; watchdog retiring."
  exit 0
fi

if squeue -u "$ME" -h -t R -o "%j" | grep -qx "$RDM_NAME"; then
  echo "RDMReg is running -- nothing to do."
else
  echo "RDMReg NOT running; ensuring it is eligible."
  for id in $(squeue -u "$ME" -h -t PENDING -o "%i|%j" | awk -F'|' -v n="$RDM_NAME" '$2==n{print $1}'); do
    scontrol release "$id" 2>/dev/null && echo "  released RDMReg job $id"
  done

  now=$(date +%s); last=$(cat "$STAMP" 2>/dev/null || echo 0)
  if [ $((now - last)) -lt "$COOLDOWN" ]; then
    echo "  cooldown active ($(( (COOLDOWN - (now-last)) / 60 )) min left); not freeing another slot."
  else
    # Victim = a running job that is NOT in the priority list (.pretrain_priority:
    # RDMReg first, then the two plain-ViT IN1K runs); prefer the hvit runs.
    PROTECT="$(tr '\n' ' ' < .pretrain_priority 2>/dev/null) $RDM_NAME"
    _pick() { squeue -u "$ME" -h -t R -o "%i|%j" | awk -F'|' -v p="$PROTECT" -v re="$1" 'BEGIN{n=split(p,a," "); for(i=1;i<=n;i++) prot[a[i]]=1} !($2 in prot) && $2 ~ re {print; exit}'; }
    victim=$(_pick hvit); [ -z "$victim" ] && victim=$(_pick .)
    if [ -n "$victim" ]; then
      vid=${victim%%|*}; vnm=${victim##*|}
      # Hold successors FIRST: cancelling the parent satisfies afterany otherwise.
      for sid in $(squeue -u "$ME" -h -t PENDING -o "%i|%j" | awk -F'|' -v n="$vnm" '$2==n{print $1}'); do
        scontrol hold "$sid" 2>/dev/null && echo "  held successor $sid ($vnm)"
      done
      scancel "$vid" && echo "  freed slot: stopped $vid ($vnm) for RDMReg"
      echo "$now" > "$STAMP"
    else
      echo "  no other running job to stop."
    fi
  fi
fi

# 2) Re-arm for the next check.
sbatch --begin=now+20minutes "$SELF" >/dev/null 2>&1 && echo "re-armed (+20 min)"
