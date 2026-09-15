#!/bin/bash
# Did the LocaLeReg arms CRASH or COLLAPSE?
#
# eval_indomain.py reports the epoch of best.ckpt, which is selected on
# val/probe_acc -- so "ep 51" only says the online probe peaked at 51, not that
# training stopped there. This script answers the actual question, per run:
#
#   best  = epoch of best.ckpt        (where the probe peaked)
#   last  = epoch of last.ckpt        (how far training ACTUALLY got)
#   snap  = highest 25-epoch snapshot (independent witness of the same thing)
#
#   best << last  -> COLLAPSE: the run kept going and got worse. Fix the recipe.
#   best ~= last  -> CRASH: training stopped. Read the error log section below.
#
# Read-only: inspects checkpoints, sacct and the job logs, writes nothing.
#
#   bash scripts/diagnose_geometric_runs.sh
#   bash scripts/diagnose_geometric_runs.sh hvit_locale_k4     # just one run

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

RUNS=("$@")
if [[ ${#RUNS[@]} -eq 0 ]]; then
  RUNS=(hvit_sigreg_matched hvit_rdmreg_matched hvit_locale_k2 hvit_locale_k4
        hvit_locale_k8 hvit_sigreg_paper hvit_rdmreg_paper)
fi

source "$HOME/.local/bin/env" 2>/dev/null || true
source .venv/bin/activate 2>/dev/null || true

echo "=============================================================================="
echo " CHECKPOINT PROGRESS   (best << last means collapse, best ~= last means crash)"
echo "=============================================================================="
printf '%-22s %6s %6s %6s %7s  %s\n' run best last snap snaps last-write
for name in "${RUNS[@]}"; do
  dir="checkpoints/${name}"
  [[ -d "$dir" ]] || { printf '%-22s  (no checkpoint directory)\n' "$name"; continue; }
  python - "$dir" <<'PY'
import glob, os, re, sys, torch

d = sys.argv[1]

def epoch_of(path):
    if not os.path.exists(path):
        return "-"
    try:
        return torch.load(path, map_location="cpu", weights_only=False).get("epoch", "?")
    except Exception as exc:
        return f"ERR({type(exc).__name__})"

snaps = sorted(glob.glob(os.path.join(d, "snapshots", "snap_epoch*.ckpt")))
top = max((int(re.search(r"(\d+)", os.path.basename(s)).group(1)) for s in snaps), default="-")
last = os.path.join(d, "last.ckpt")
stamp = ""
if os.path.exists(last):
    import datetime
    stamp = datetime.datetime.fromtimestamp(os.path.getmtime(last)).strftime("%Y-%m-%d %H:%M")
print(f"{os.path.basename(d):<22} {str(epoch_of(os.path.join(d,'best.ckpt'))):>6} "
      f"{str(epoch_of(last)):>6} {str(top):>6} {len(snaps):>7}  {stamp}")
PY
done

echo
echo "=============================================================================="
echo " SLURM HISTORY   (job state + exit code; OUT_OF_MEMORY / TIMEOUT / FAILED)"
echo "=============================================================================="
sacct --name=blj-geom-imagenette --starttime=now-14days \
      --format=JobID%18,JobName%22,State%18,ExitCode%8,Elapsed%12,MaxRSS%10,NodeList%12 2>/dev/null \
  | grep -v '\.batch\|\.extern' || echo "sacct unavailable or no matching jobs"

echo
echo "=============================================================================="
echo " ERROR LOGS   (last failure signature per array task)"
echo "=============================================================================="
shopt -s nullglob
logs=(logs/embedor/error_blj-geom-imagenette_*.log)
if [[ ${#logs[@]} -eq 0 ]]; then
  echo "no logs/embedor/error_blj-geom-imagenette_*.log found"
else
  for f in "${logs[@]}"; do
    hit=$(grep -n -i -m3 -E 'nan|inf\b|traceback|out of memory|CUDA error|CANCELLED|DUE TO TIME|assert' "$f" 2>/dev/null)
    [[ -z "$hit" ]] && continue
    echo "--- $f"
    echo "$hit"
    echo "    ...tail:"
    tail -n 6 "$f" | sed 's/^/    /'
    echo
  done
  echo "(logs with no matching signature are omitted; ${#logs[@]} scanned)"
fi

echo "=============================================================================="
echo "NEXT: if best << last it collapsed -- plot val/probe_acc and the loss terms"
echo "      in wandb (group geometric_imagenette) and find the epoch they diverge."
echo "      if best ~= last it crashed -- the signature is above; resubmit with"
echo "      sbatch --array=4-6%4 src/slurm_files/slurm_diufrd_pretrain_geometric_imagenette.sh"
echo "      (last.ckpt is picked up automatically, so it resumes where it stopped)."
echo "=============================================================================="
