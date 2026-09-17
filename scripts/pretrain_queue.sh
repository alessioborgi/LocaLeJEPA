#!/bin/bash
# Concurrency manager for the self-requeueing BundLeJEPA pretraining jobs.
#
# The pretrain scripts (src/slurm_files/slurm_harvard_pretrain_*.sh) each queue a
# successor with `--dependency=afterany:$SLURM_JOB_ID` before training, and resume
# from `${CKPT_DIR}/last.ckpt`. That gives free crash/preemption recovery, but two
# failure modes follow from it:
#
#   1. DUPLICATES. Every requeue of a job runs the "queue successor" line again
#      without cancelling the previous successor, so successors accumulate. Several
#      copies of one experiment then run at once, all writing the SAME checkpoint
#      dirpath -- which is where `last-v1.ckpt` / `last-v2.ckpt` come from, and
#      which makes `ckpt_path=last.ckpt` resume an arbitrary one of the racers.
#
#   2. NO CAP. Cancelling a running job SATISFIES its successor's `afterany`
#      dependency, so the successor starts immediately. Cancelling alone can never
#      reduce concurrency; the successor must be HELD first.
#
# Hence the ordering rule used throughout: hold/cancel the successor FIRST, then
# cancel the parent. State is never lost -- it lives in last.ckpt, and a held
# successor resumes from there whenever it is released.
#
# Usage:
#   scripts/pretrain_queue.sh status          # what is running/pending, duplicates
#   scripts/pretrain_queue.sh dedupe [--yes]  # keep one running + one successor per experiment
#   scripts/pretrain_queue.sh cap N [--yes]   # keep N experiments running, hold the rest
#   scripts/pretrain_queue.sh refill [--yes]  # release held jobs up to the cap
#   scripts/pretrain_queue.sh release JOBID   # release one held job by hand
#
# Without --yes every subcommand is a DRY RUN: it prints the exact scancel /
# scontrol commands it would run and changes nothing.

set -uo pipefail

USER_ID="${USER}"
PREFIX="BLJ_"                      # only manage BundLeJEPA jobs
CAP_FILE="${SLURM_SUBMIT_DIR:-$(pwd)}/.pretrain_cap"
DRY=1

_maybe() {                          # echo in dry-run, execute with --yes
  if [ "$DRY" -eq 1 ]; then echo "  [dry-run] $*"; else echo "  + $*"; eval "$@"; fi
}

# ONE squeue call for the whole run: id|name|state|reason|dependency. Per-job
# squeue/scontrol lookups cost a Slurm RPC each and made this take minutes on a
# 20-job queue. Pretrain jobs only -- eval jobs are short and not throttled.
SNAP=""
_snapshot() {
  [ -n "$SNAP" ] && return
  # Exclude evals (BLJ_eval_*, BLJ_ev_*), report jobs and the watchdogs: none of
  # them is a pretrain experiment, and counting them makes cap/refill think the
  # slots are full.
  SNAP=$(squeue -u "$USER_ID" -h -o "%i|%j|%T|%r|%E" 2>/dev/null | grep "|${PREFIX}" | grep -vE "\|${PREFIX}(eval|ev_|report_)|watchdog")
}

_jobs() {  # $1 = state
  _snapshot; echo "$SNAP" | awk -F'|' -v s="$1" '$3==s'
}

_parent_of() { _snapshot; echo "$SNAP" | awk -F'|' -v i="$1" '$1==i {print $5}' | grep -oE "afterany:[0-9]+" | head -1 | cut -d: -f2; }

# Pending jobs that are NOT user-held. These start on their own as soon as
# resources appear, so they occupy a slot just as surely as a RUNNING job -- the
# cap has to count them, or refill over-releases.
_pending_active() { _jobs PENDING | awk -F'|' 'tolower($4) !~ /held/'; }

# Unique experiment names currently holding a slot (running or queued-and-eligible).
_active_names() { { _jobs RUNNING; _pending_active; } | cut -d'|' -f2 | sort -u; }

cmd_status() {
  echo "=== running ==="
  _jobs RUNNING | awk -F'|' '{printf "  %-12s %s\n", $1, $2}'
  echo "=== pending (successors / held) ==="
  for row in $(_jobs PENDING); do
    id=${row%%|*}; nm=$(echo "$row" | cut -d'|' -f2)
    held=$(squeue -h -j "$id" -o "%r" 2>/dev/null | grep -qi "held" && echo "HELD" || echo "queued")
    printf "  %-12s %-40s %-6s parent=%s\n" "$id" "$nm" "$held" "$(_parent_of "$id")"
  done
  echo "=== duplicate experiment names ==="
  _jobs RUNNING | cut -d'|' -f2 | sort | uniq -c | awk '$1>1 {printf "  RUNNING x%s  %s\n", $1, $2}'
  _jobs PENDING | cut -d'|' -f2 | sort | uniq -c | awk '$1>1 {printf "  PENDING x%s  %s\n", $1, $2}'
  local nr; nr=$(_jobs RUNNING | wc -l)
  local nu; nu=$(_jobs RUNNING | cut -d'|' -f2 | sort -u | wc -l)
  echo "=== totals: $nr running ($nu unique experiments), $(_jobs PENDING | wc -l) pending; cap=$(cat "$CAP_FILE" 2>/dev/null || echo unset) ==="
}

# Keep exactly one RUNNING copy and one successor per experiment name.
cmd_dedupe() {
  echo "== de-duplicating (successors cancelled before parents) =="
  local kill_succ="" kill_run=""
  # Iterate over every experiment seen in EITHER state: an experiment whose parent
  # is merely queued-for-resources still accumulates one successor per preemption,
  # and looking only at RUNNING would miss it entirely.
  for nm in $( { _jobs RUNNING; _pending_active; } | cut -d'|' -f2 | sort -u); do
    local run_ids keep
    run_ids=$(_jobs RUNNING | awk -F'|' -v n="$nm" '$2==n {print $1}' | sort -n)
    if [ -n "$run_ids" ]; then
      keep=$(echo "$run_ids" | head -1)                     # oldest running copy wins
      for id in $run_ids; do [ "$id" != "$keep" ] && kill_run="$kill_run $id"; done
    else
      # No running copy: the lowest-id eligible pending job is the live one.
      keep=$(_pending_active | awk -F'|' -v n="$nm" '$2==n {print $1}' | sort -n | head -1)
    fi
    # Candidate successors: eligible pending jobs of this name other than `keep`.
    # HELD jobs are deliberately excluded -- those are the deferred experiments.
    local succ skeep=""
    succ=$(_pending_active | awk -F'|' -v n="$nm" -v k="$keep" '$2==n && $1!=k {print $1}' | sort -n)
    for s in $succ; do [ "$(_parent_of "$s")" = "$keep" ] && skeep="$s" && break; done
    [ -z "$skeep" ] && skeep=$(echo "$succ" | head -1)
    for s in $succ; do [ "$s" != "$skeep" ] && kill_succ="$kill_succ $s"; done
    if [ -n "$(echo "$run_ids" | sed -n 2p)" ] || [ "$(echo "$succ" | wc -w)" -gt 1 ]; then
      printf "  %-40s keep=%s succ=%s\n" "$nm" "$keep" "${skeep:-none}"
    fi
  done
  [ -n "$kill_succ" ] && _maybe "scancel$kill_succ" || echo "  no duplicate successors"
  [ -n "$kill_run" ]  && _maybe "scancel$kill_run"  || echo "  no duplicate running copies"
}

# Keep N experiments running; hold the successor and cancel the parent for the rest.
cmd_cap() {
  local n="$1"
  # Refuse to run while duplicates exist: with several successors per experiment
  # the "hold the successor" step would pick an arbitrary one and the others would
  # still fire when the parent is cancelled, silently breaking the cap.
  # Count within each state separately: one RUNNING job plus its own PENDING
  # successor share a name and are the normal, correct steady state.
  local dups
  dups=$( { _jobs RUNNING | cut -d'|' -f2 | sort | uniq -c | awk '$1>1 {print "RUNNING x"$1"  "$2}'
            _jobs PENDING | cut -d'|' -f2 | sort | uniq -c | awk '$1>1 {print "PENDING x"$1"  "$2}'; } )
  if [ -n "$dups" ]; then
    echo "REFUSING: duplicate jobs still present for:"
    echo "$dups" | sed 's/^/    /'
    echo "  Run '$0 dedupe --yes' first, then re-run cap."
    return 1
  fi
  echo "$n" > "$CAP_FILE"
  echo "== capping concurrency at $n (cap recorded in $CAP_FILE) =="
  local running; running=$(_active_names)
  local total; total=$(echo "$running" | wc -l)
  [ "$total" -le "$n" ] && { echo "  $total running <= cap $n, nothing to defer"; return; }
  # Defer the LAST (total-n) names in KEEP_ORDER; anything unlisted is deferred first.
  # KEEP_ORDER env var, else .pretrain_priority (one job name per line, highest
  # priority first: RDMReg, then the two plain-ViT IN1K runs).
  local keep_list="${KEEP_ORDER:-$(tr '\n' ' ' < "${SLURM_SUBMIT_DIR:-$(pwd)}/.pretrain_priority" 2>/dev/null)}"
  local keep="" defer=""
  for nm in $keep_list; do
    echo "$running" | grep -qx "$nm" && [ "$(echo "$keep" | wc -w)" -lt "$n" ] && keep="$keep $nm"
  done
  for nm in $running; do echo "$keep" | grep -qw "$nm" || defer="$defer $nm"; done
  echo "  KEEP  :$keep"
  echo "  DEFER :$defer"
  for nm in $defer; do
    local rid sid
    rid=$(_jobs RUNNING | awk -F'|' -v x="$nm" '$2==x {print $1}' | head -1)
    sid=$(_jobs PENDING | awk -F'|' -v x="$nm" '$2==x {print $1}' | head -1)
    echo "  -- $nm (running=$rid successor=${sid:-none})"
    # HOLD the successor BEFORE cancelling the parent, or afterany fires instantly.
    [ -n "$sid" ] && _maybe "scontrol hold $sid"
    [ -n "$rid" ] && _maybe "scancel $rid"
  done
  echo "  Deferred runs keep their state in last.ckpt; 'refill' releases them."
}

# Release held jobs while under the cap.
cmd_refill() {
  local n; n=$(cat "$CAP_FILE" 2>/dev/null || echo 0)
  [ "$n" -le 0 ] && { echo "no cap set ($CAP_FILE); nothing to do"; return; }
  local cur; cur=$(_active_names | wc -l)   # running OR queued-and-eligible both occupy a slot
  echo "== refill: $cur running, cap $n =="
  local slots=$((n - cur))
  [ "$slots" -le 0 ] && { echo "  no free slots"; return; }
  for id in $(squeue -u "$USER_ID" -h -t PENDING -o "%i|%j|%r" | grep "|${PREFIX}" | grep -i "held" | cut -d'|' -f1); do
    [ "$slots" -le 0 ] && break
    _maybe "scontrol release $id"
    slots=$((slots - 1))
  done
}

main() {
  local sub="${1:-status}"; shift || true
  local args=()
  for a in "$@"; do [ "$a" = "--yes" ] && DRY=0 || args+=("$a"); done
  case "$sub" in
    status)  cmd_status ;;
    dedupe)  cmd_dedupe ;;
    cap)     cmd_cap "${args[0]:?usage: cap N [--yes]}" ;;
    refill)  cmd_refill ;;
    release) DRY=0; scontrol release "${args[0]:?usage: release JOBID}" ;;
    *) echo "usage: $0 {status|dedupe|cap N|refill|release JOBID} [--yes]"; exit 1 ;;
  esac
  [ "$DRY" -eq 1 ] && [ "$sub" != "status" ] && echo "
DRY RUN -- nothing changed. Re-run with --yes to apply."
  return 0
}
main "$@"
