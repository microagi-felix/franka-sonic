#!/usr/bin/env bash
# GATE P2 — lane B-1: the dual-FR3 SONIC embodiment. Prints PASS / FAIL / WARN
# per check and exits non-zero if any check FAILed. WARN never fails the gate.
#
#   bash harness/gates/p2.sh
#
# Artifact-based: the MJCF and the robot config exist, the motion library has
# enough clips, the encoder/decoder ONNX were exported, and the decoder replay
# wrote a mean joint error. The replay THRESHOLD is a WARN (a decoder that
# cannot replay the demos is a finding, not a broken pipeline); a missing
# replay file is a FAIL. Conventions: plan/PLAN.md "Artifact conventions".
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS="$HOME/runs/franka-sonic"
LANE_B="$RUNS/lane_b"
WBC="$HOME/GR00T-WholeBodyControl"
MIN_CLIPS=100
MAX_JOINT_ERR=0.1

FAILED=0
WARNED=0

pass() { printf 'PASS  %-34s %s\n' "$1" "${2:-}"; }
fail() { printf 'FAIL  %-34s %s\n' "$1" "${2:-}"; FAILED=$((FAILED + 1)); }
warn() { printf 'WARN  %-34s %s\n' "$1" "${2:-}"; WARNED=$((WARNED + 1)); }

echo "GATE P2 — $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)"
echo "----------------------------------------------------------------------"

# 1 -------------------------------------------------------------- MJCF
xml=$(find "$REPO_ROOT/harness" "$WBC" -maxdepth 6 -type f -name 'dual_fr3*.xml' \
        2>/dev/null | head -1)
if [ -n "$xml" ]; then
  pass "dual_fr3.xml" "$xml ($(wc -l < "$xml") lines)"
else
  fail "dual_fr3.xml" "expected harness/lane_b/dual_fr3.xml (P2 WP 2.2)"
fi

# 2 -------------------------------------------------------------- robot config
cfg=$(find "$REPO_ROOT/harness" "$WBC" -maxdepth 8 -type f \
        \( -name 'robots_dual_fr3.py' -o -name 'dual_fr3.py' \) 2>/dev/null | head -1)
if [ -n "$cfg" ]; then
  pass "dual-FR3 robot config" "$cfg"
else
  fail "dual-FR3 robot config" \
       "expected harness/lane_b/robots_dual_fr3.py (P2 WP 2.3)"
fi

# 3 -------------------------------------------------------------- exp yaml (WARN)
yml=$(find "$REPO_ROOT/harness" "$WBC/gear_sonic/config" -maxdepth 6 -type f \
        -name '*dual_fr3*.yaml' 2>/dev/null | head -1)
if [ -n "$yml" ]; then
  pass "sonic_dual_fr3.yaml" "$yml"
else
  warn "sonic_dual_fr3.yaml" "no *dual_fr3*.yaml found — training config not in the repo?"
fi

# 4 -------------------------------------------------------------- motion library
pkl_dir=""
pkl_n=0
if [ -d "$LANE_B" ]; then
  while read -r d; do
    [ -n "$d" ] || continue
    n=$(find "$d" -maxdepth 2 -type f -name '*.pkl' 2>/dev/null | wc -l)
    if [ "$n" -gt "$pkl_n" ]; then pkl_n=$n; pkl_dir=$d; fi
  done < <(find "$LANE_B" -maxdepth 3 -type d -name 'motions' 2>/dev/null)
  if [ -z "$pkl_dir" ]; then
    # fall back to any run folder that simply holds pkls
    while read -r d; do
      [ -n "$d" ] || continue
      n=$(find "$d" -maxdepth 3 -type f -name '*.pkl' 2>/dev/null | wc -l)
      if [ "$n" -gt "$pkl_n" ]; then pkl_n=$n; pkl_dir=$d; fi
    done < <(find "$LANE_B" -maxdepth 1 -type d -name '*motion*' 2>/dev/null)
  fi
fi
if [ "$pkl_n" -ge "$MIN_CLIPS" ]; then
  pass "motion library >= $MIN_CLIPS pkl" "$pkl_n clips in $pkl_dir"
elif [ "$pkl_n" -gt 0 ]; then
  fail "motion library >= $MIN_CLIPS pkl" "only $pkl_n clip(s) in $pkl_dir"
else
  fail "motion library >= $MIN_CLIPS pkl" \
       "expected lane_b/*_motion_lib/out/motions/*.pkl (P2 WP 2.5)"
fi

# 5 -------------------------------------------------------------- ONNX pair
enc=$(find "$LANE_B" -maxdepth 4 -type f -name '*encoder*.onnx' 2>/dev/null | head -1)
dec=$(find "$LANE_B" -maxdepth 4 -type f -name '*decoder*.onnx' 2>/dev/null | head -1)
if [ -n "$enc" ] && [ -n "$dec" ]; then
  pass "encoder + decoder ONNX" "$(basename "$enc") + $(basename "$dec") in $(dirname "$dec")"
else
  fail "encoder + decoder ONNX" \
       "encoder='${enc:-missing}' decoder='${dec:-missing}' — expected lane_b/*_export_onnx/out/ (P2 WP 2.7)"
fi

# 6 -------------------------------------------------------------- decoder replay
rep=$(find "$LANE_B" -maxdepth 4 -type f -name 'replay*.json' 2>/dev/null | head -1)
if [ -n "$rep" ]; then
  err=$(python3 - "$rep" <<'PY' 2>/dev/null
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
for k in ("mean_joint_error_rad", "mean_joint_error", "joint_error_rad"):
    if isinstance(d.get(k), (int, float)):
        print(d[k]); break
PY
)
  if [ -n "$err" ]; then
    ok=$(awk -v e="$err" -v t="$MAX_JOINT_ERR" 'BEGIN { print (e < t) ? 1 : 0 }')
    if [ "$ok" = "1" ]; then
      pass "decoder replay < $MAX_JOINT_ERR rad" "mean joint error $err rad  $rep"
    else
      warn "decoder replay < $MAX_JOINT_ERR rad" \
           "mean joint error $err rad (above threshold — a finding, not a gate failure)  $rep"
    fi
  else
    warn "decoder replay < $MAX_JOINT_ERR rad" \
         "$rep has no mean_joint_error_rad key — cannot judge the threshold"
  fi
else
  fail "decoder replay json" \
       "expected lane_b/*_decoder_replay/out/replay.json with mean_joint_error_rad (P2 WP 2.7)"
fi

echo "----------------------------------------------------------------------"
if [ "$FAILED" -eq 0 ]; then
  echo "GATE P2: PASS ($WARNED warning(s))"
  exit 0
fi
echo "GATE P2: FAIL ($FAILED failing check(s), $WARNED warning(s))"
exit 1
