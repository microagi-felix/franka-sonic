#!/usr/bin/env bash
# GATE P5 — lane B ceiling fix: a retrained SONIC decoder that executes its own
# tokens. Prints PASS / FAIL / WARN per check, exits non-zero on any FAIL.
#
#   bash harness/gates/p5.sh
#   P5_MIN_SUCCESS=15 bash harness/gates/p5.sh     # threshold override
#
# Artifact-based. Everything P5 produces is NEWER than P5_EPOCH (all P0–P4
# artifacts, including the 10:20 UTC copy-back into lane_b/final/, are older),
# so "newest artifact newer than the epoch" identifies P5's own outputs.
set -uo pipefail

RUNS="$HOME/runs/franka-sonic"
RUNS_TMP="/tmp/franka-sonic"
P5_EPOCH="2026-09-03 13:00:00 UTC"
MIN_EPISODES=20
MIN_SUCCESS="${P5_MIN_SUCCESS:-15}"
MAX_REPLAY_ERR=0.2

present() { local d out=""; for d in $1; do [ -d "$d" ] && out="$out $d"; done; echo "${out# }"; }
LANE_B_ALL=$(present "$RUNS/lane_b $RUNS_TMP/lane_b")
find_runs() { local roots="$1"; shift; [ -n "$roots" ] || return 0; find $roots "$@" 2>/dev/null; }
newest() { xargs -r ls -t 2>/dev/null | head -1; }

FAILED=0; WARNED=0
pass() { printf 'PASS  %-34s %s\n' "$1" "${2:-}"; }
fail() { printf 'FAIL  %-34s %s\n' "$1" "${2:-}"; FAILED=$((FAILED + 1)); }
warn() { printf 'WARN  %-34s %s\n' "$1" "${2:-}"; WARNED=$((WARNED + 1)); }

echo "GATE P5 — $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)  (epoch $P5_EPOCH)"
echo "----------------------------------------------------------------------"

# 1 ------------------------------------------------- a new decoder was trained
ts=$(find_runs "$LANE_B_ALL" -maxdepth 4 -type f -name train_summary.json -path '*sonic_rl*' -newermt "$P5_EPOCH" | newest)
if [ -n "$ts" ]; then pass "P5 SONIC RL run" "$ts"; else fail "P5 SONIC RL run" "no lane_b/*_sonic_rl*/out/train_summary.json newer than the epoch (WP 5.0)"; fi

# 2 ------------------------------------------------- ONNX pair from it
dec=$(find_runs "$LANE_B_ALL" -maxdepth 4 -type f -name '*decoder*.onnx' -newermt "$P5_EPOCH" | newest)
enc=$(find_runs "$LANE_B_ALL" -maxdepth 4 -type f -name '*encoder*.onnx' -newermt "$P5_EPOCH" | newest)
if [ -n "$dec" ] && [ -n "$enc" ]; then pass "P5 encoder + decoder ONNX" "$(dirname "$dec")"; else fail "P5 encoder + decoder ONNX" "no export newer than the epoch (WP 5.2)"; fi

# 3 ------------------------------------------------- decoder replay (WARN only)
rep=$(find_runs "$LANE_B_ALL" -maxdepth 4 -type f -name 'replay*.json' -newermt "$P5_EPOCH" | newest)
if [ -n "$rep" ]; then
  err=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); v=next((d[k] for k in ("mean_joint_error_rad","mean_joint_error","joint_error_rad") if isinstance(d.get(k),(int,float))),None); print("" if v is None else v)' "$rep" 2>/dev/null)
  if [ -n "$err" ]; then
    if awk -v e="$err" -v t="$MAX_REPLAY_ERR" 'BEGIN { exit !(e < t) }'; then pass "decoder replay < $MAX_REPLAY_ERR rad" "$err rad  $rep"; else warn "decoder replay < $MAX_REPLAY_ERR rad" "$err rad (finding, not a failure)  $rep"; fi
  else warn "decoder replay < $MAX_REPLAY_ERR rad" "$rep has no mean_joint_error_rad key"; fi
else fail "P5 decoder replay json" "no lane_b/*_decoder_replay*/out/replay*.json newer than the epoch (WP 5.2)"; fi

# 4 ------------------------------------------------- new token labels + dataset
tok=$(find_runs "$LANE_B_ALL" -maxdepth 5 -type f -name index.json -path '*label_tokens*/out/tokens/*' -newermt "$P5_EPOCH" | newest)
if [ -n "$tok" ]; then
  pass "P5 token labels" "$tok"
  ds="$(dirname "$(dirname "$tok")")/gr00t_v2_sonic/meta/modality.json"
  if [ -f "$ds" ]; then pass "P5 token dataset (for P6)" "$ds"; else warn "P5 token dataset (for P6)" "no $ds — run label_tokens with the dataset step"; fi
else fail "P5 token labels" "no lane_b/*_label_tokens*/out/tokens/index.json newer than the epoch (WP 5.3)"; fi

# 5 ------------------------------------------------- THE gate: B-oracle >= MIN_SUCCESS/20
ocsv=$(find_runs "$LANE_B_ALL" -maxdepth 5 -type f -path '*oracle_b*/out/eval/eval_results.csv' -newermt "$P5_EPOCH" | newest)
if [ -n "$ocsv" ]; then
  read -r rows succ <<<"$(python3 -c 'import csv,sys; r=list(csv.DictReader(open(sys.argv[1]))); print(len(r), sum(1 for x in r if str(x.get("success","")).strip().lower()=="true"))' "$ocsv" 2>/dev/null)"
  rows=${rows:-0}; succ=${succ:-0}
  if [ "$rows" -lt "$MIN_EPISODES" ]; then fail "B-oracle >= $MIN_EPISODES episodes" "only $rows rows in $ocsv"; else pass "B-oracle >= $MIN_EPISODES episodes" "$rows rows  $ocsv"; fi
  if [ "$succ" -ge "$MIN_SUCCESS" ]; then pass "B-oracle success >= $MIN_SUCCESS/$MIN_EPISODES" "$succ/$rows"; else fail "B-oracle success >= $MIN_SUCCESS/$MIN_EPISODES" "$succ/$rows — the ceiling is still closed (WP 5.3)"; fi
else fail "B-oracle success >= $MIN_SUCCESS/$MIN_EPISODES" "no lane_b/*_oracle_b*/out/eval/eval_results.csv newer than the epoch (WP 5.3)"; fi

echo "----------------------------------------------------------------------"
if [ "$FAILED" -eq 0 ]; then echo "GATE P5: PASS ($WARNED warning(s))"; exit 0; fi
echo "GATE P5: FAIL ($FAILED failing check(s), $WARNED warning(s))"; exit 1
