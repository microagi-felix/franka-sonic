#!/usr/bin/env bash
# harness/driver.sh — the autonomous phase loop.
#
#   tmux new-window -t bakeoff -n driver -c ~/code/franka-sonic \
#       'bash harness/driver.sh 2>&1 | tee -a ~/runs/franka-sonic/driver/driver.log'
#   tmux kill-window -t bakeoff:driver        # stop it
#
# For each phase P0..P4 it:
#   1. skips the phase if plan/STATUS.md already says `GATE PN: PASS`;
#   2. exits 2 if the last stop marker after the previous phase's PASS is a
#      `BLOCKED:` line (a human then appends `DRIVER: resume` to clear it);
#   3. for P0 only: waits (60 s poll, up to 4 h) for the interactive session in
#      tmux window `bakeoff:claude` to write `GATE P0: PASS` or `BLOCKED:`
#      before attempting P0 itself;
#   4. otherwise runs `claude -p` on plan/prompts/PN.md, up to 3 attempts,
#      `timeout 6h` each, fresh context per attempt, re-reading STATUS.md after
#      each; if an attempt leaves no PASS marker but harness/gates/pN.sh exits
#      0, the driver records the PASS itself (gates decide phases, not prose).
#
# After P4: `DRIVER: all phases passed` into STATUS.md, push, exit 0.
# After 3 failed attempts on a phase: `BLOCKED: driver gave up on PN`, push,
# exit 2.
#
# Idempotent: re-running it after a crash skips everything already marked PASS.
# Only one driver at a time (flock on ~/runs/franka-sonic/driver/driver.lock).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

STATUS="$REPO_ROOT/plan/STATUS.md"
PROMPTS="$REPO_ROOT/plan/prompts"
GATES="$REPO_ROOT/harness/gates"
DRIVER_RUNS="$HOME/runs/franka-sonic/driver"
LOGDIR="$DRIVER_RUNS/$(date +%F)"

CLAUDE="${DRIVER_CLAUDE:-$HOME/.local/bin/claude}"
EFFORT="${DRIVER_EFFORT:-xhigh}"          # Felix: xhigh explicitly, never inherited
ATTEMPTS="${DRIVER_ATTEMPTS:-3}"
PHASE_TIMEOUT="${DRIVER_PHASE_TIMEOUT:-6h}"
P0_WAIT_POLLS="${DRIVER_P0_WAIT_POLLS:-240}"   # 240 × 60 s = 4 h
P0_POLL_SECONDS="${DRIVER_P0_POLL_SECONDS:-60}"
GR00T_PY="$HOME/Isaac-GR00T/.venv/bin/python"
PHASES="P0 P1 P2 P3 P4"

mkdir -p "$LOGDIR" || exit 1

log() { printf '[driver %s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

# --------------------------------------------------------------------- locking
exec 9>"$DRIVER_RUNS/driver.lock"
if command -v flock >/dev/null 2>&1; then
  if ! flock -n 9; then
    log "another driver already holds $DRIVER_RUNS/driver.lock — exiting"
    exit 3
  fi
fi

# --------------------------------------------------------------------- status
status_pass() {           # $1 = phase
  grep -qE "GATE $1: PASS" "$STATUS"
}

# The last stop marker in STATUS.md after the previous phase's PASS.
# Markers: "GATE Pn: PASS", "BLOCKED:", "DRIVER: resume" (the human's clear).
last_marker() {           # $1 = previous phase or ""
  local start=1 n
  if [ -n "${1:-}" ]; then
    n=$(grep -nE "GATE $1: PASS" "$STATUS" | tail -1 | cut -d: -f1)
    [ -n "$n" ] && start=$((n + 1))
  fi
  tail -n "+$start" "$STATUS" \
    | grep -oE "GATE P[0-4]: PASS|BLOCKED:|DRIVER: resume" | tail -1
}

is_blocked() {            # $1 = previous phase or ""
  [ "$(last_marker "${1:-}")" = "BLOCKED:" ]
}

status_append() {         # $1 = text
  printf -- '- %s  %s\n' "$(date '+%Y-%m-%d %H:%M')" "$1" >> "$STATUS"
}

push_status() {           # $1 = commit message
  git -C "$REPO_ROOT" add -- plan/STATUS.md >/dev/null 2>&1
  if git -C "$REPO_ROOT" commit -m "$1" -- plan/STATUS.md >/dev/null 2>&1; then
    log "committed: $1"
  else
    log "nothing to commit for: $1"
  fi
  # Another agent pushes to main concurrently: rebase, then push.
  git -C "$REPO_ROOT" pull --rebase --autostash >/dev/null 2>&1 \
    || log "WARN: git pull --rebase failed before push"
  if git -C "$REPO_ROOT" push >/dev/null 2>&1; then
    log "pushed"
  else
    log "WARN: git push failed (STATUS.md is committed locally)"
  fi
}

# --------------------------------------------------------------------- phases
wait_for_p0() {
  # The interactive session in tmux `bakeoff:claude` owns P0. Poll STATUS.md.
  local i marker
  log "P0 is owned by the interactive session — polling STATUS.md every ${P0_POLL_SECONDS}s (max $P0_WAIT_POLLS polls)"
  for i in $(seq 1 "$P0_WAIT_POLLS"); do
    if status_pass P0; then
      log "P0 PASS seen in STATUS.md after $i poll(s)"
      return 0
    fi
    if is_blocked ""; then
      log "STATUS.md carries a BLOCKED marker while waiting for P0 — stopping"
      return 1
    fi
    sleep "$P0_POLL_SECONDS"
  done
  log "P0 wait timed out after $((P0_WAIT_POLLS * P0_POLL_SECONDS))s — the driver will attempt P0 itself"
  return 0
}

run_attempt() {           # $1 = phase, $2 = attempt no, $3 = log file
  local phase="$1" k="$2" logf="$3" prompt_file="$PROMPTS/$1.md" rc

  {
    echo "=============================================================="
    echo "=== driver attempt $k/$ATTEMPTS for $phase"
    echo "=== when:   $(date -Is)  host: $(hostname)"
    echo "=== repo:   $REPO_ROOT @ $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>&1)"
    echo "=== prompt: $prompt_file ($(wc -c < "$prompt_file" 2>/dev/null) bytes)"
  } >> "$logf"

  if git -C "$REPO_ROOT" pull --ff-only >> "$logf" 2>&1; then
    echo "=== git pull --ff-only ok" >> "$logf"
  else
    echo "=== git pull --ff-only FAILED (continuing — the tree may be dirty)" >> "$logf"
  fi

  echo "=== GPU claims before the attempt:" >> "$logf"
  "$GR00T_PY" "$REPO_ROOT/harness/gpus.py" list >> "$logf" 2>&1 \
    || echo "=== gpus.py list failed" >> "$logf"

  # The effective command line, verbatim, so the effort level can be verified
  # later (Felix: xhigh must be explicit, not inherited from settings.json).
  echo "=== command: timeout -k 120 $PHASE_TIMEOUT $CLAUDE -p --dangerously-skip-permissions --effort $EFFORT --output-format text \"\$(cat $prompt_file)\"" >> "$logf"
  echo "==============================================================" >> "$logf"

  timeout -k 120 "$PHASE_TIMEOUT" \
    "$CLAUDE" -p \
      --dangerously-skip-permissions \
      --effort "$EFFORT" \
      --output-format text \
      "$(cat "$prompt_file")" >> "$logf" 2>&1
  rc=$?

  echo "=== claude exit rc=$rc at $(date -Is)" >> "$logf"
  [ "$rc" -eq 124 ] && echo "=== (rc 124 = hit the $PHASE_TIMEOUT timeout)" >> "$logf"
  echo "=== GPU claims after the attempt:" >> "$logf"
  "$GR00T_PY" "$REPO_ROOT/harness/gpus.py" list >> "$logf" 2>&1 \
    || echo "=== gpus.py list failed" >> "$logf"

  return "$rc"
}

verify_gate() {           # $1 = phase; gates decide phases, not prose
  local phase="$1" gate="$GATES/$(echo "$phase" | tr 'A-Z' 'a-z').sh" out rc
  [ -x "$gate" ] || [ -f "$gate" ] || { log "no gate script at $gate"; return 1; }
  out=$(bash "$gate" 2>&1); rc=$?
  printf '%s\n' "$out" | tail -20
  return "$rc"
}

run_phase() {             # $1 = phase, $2 = previous phase or ""
  local phase="$1" prev="${2:-}" k logf rc
  local prompt_file="$PROMPTS/$phase.md"

  if [ ! -f "$prompt_file" ]; then
    log "missing prompt $prompt_file"
    status_append "BLOCKED: driver found no prompt file plan/prompts/$phase.md"
    push_status "driver: BLOCKED on $phase (no prompt)"
    return 1
  fi

  for k in $(seq 1 "$ATTEMPTS"); do
    logf="$LOGDIR/$phase-attempt-$k.log"
    log "$phase attempt $k/$ATTEMPTS (timeout $PHASE_TIMEOUT, effort $EFFORT) -> $logf"
    run_attempt "$phase" "$k" "$logf"
    rc=$?
    log "$phase attempt $k finished rc=$rc"

    if status_pass "$phase"; then
      log "$phase: GATE PASS found in STATUS.md"
      return 0
    fi
    if is_blocked "$prev"; then
      log "$phase: the phase agent wrote BLOCKED into STATUS.md — stopping"
      return 1
    fi

    # No marker either way: ask the gate directly.
    log "$phase: no marker in STATUS.md — running the gate myself"
    if verify_gate "$phase"; then
      status_append "GATE $phase: PASS (verified by harness/driver.sh after attempt $k; the phase agent left no marker)"
      push_status "driver: gate $phase PASS (driver-verified)"
      return 0
    fi
    log "$phase: gate still failing after attempt $k"
    # Backoff, so a `claude -p` that fails in seconds (auth, quota) does not
    # burn all three attempts in a minute.
    [ "$k" -lt "$ATTEMPTS" ] && sleep "${DRIVER_BACKOFF_SECONDS:-60}"
  done

  status_append "BLOCKED: driver gave up on $phase after $ATTEMPTS attempts (logs: $LOGDIR/$phase-attempt-*.log)"
  push_status "driver: BLOCKED on $phase after $ATTEMPTS attempts"
  return 1
}

# --------------------------------------------------------------------- main
trap 'log "driver exiting"' EXIT

log "start pid=$$ repo=$REPO_ROOT claude=$CLAUDE effort=$EFFORT attempts=$ATTEMPTS timeout=$PHASE_TIMEOUT"
log "logs in $LOGDIR"
if [ ! -x "$CLAUDE" ]; then
  log "FATAL: $CLAUDE is not executable"
  exit 1
fi
status_append "DRIVER: start (pid $$, effort $EFFORT, logs $LOGDIR)"
push_status "driver: start"

prev=""
for PHASE in $PHASES; do
  if status_pass "$PHASE"; then
    log "$PHASE already PASS in STATUS.md — skipping"
    prev="$PHASE"
    continue
  fi

  if is_blocked "$prev"; then
    log "STATUS.md is BLOCKED at $PHASE — stopping. To resume: fix the cause, append a"
    log "  '- <date>  DRIVER: resume' line to plan/STATUS.md, then restart this window."
    exit 2
  fi

  if [ "$PHASE" = "P0" ]; then
    if ! wait_for_p0; then
      exit 2
    fi
    if status_pass P0; then
      prev="P0"
      continue
    fi
  fi

  if ! run_phase "$PHASE" "$prev"; then
    log "$PHASE did not pass — driver stops here"
    exit 2
  fi
  prev="$PHASE"
done

log "all phases passed"
status_append "DRIVER: all phases passed"
push_status "driver: all phases passed"
exit 0
