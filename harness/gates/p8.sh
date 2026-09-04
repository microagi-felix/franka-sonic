#!/usr/bin/env bash
# GATE P8 — SONIC decoder rebuilt on the round-2 data; whole set relabelled
#
#   bash harness/gates/p8.sh
#
# Round-2 gate. Every check is scoped to artifacts newer than P8_EPOCH, so
# round-1 runs (2026-09-03) can never satisfy it.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS="$HOME/runs/franka-sonic"
RUNS_TMP="/tmp/franka-sonic"
EPOCH="${P8_EPOCH:-2026-09-04 02:00:00 UTC}"
STATUS="$REPO_ROOT/plan/STATUS.md"

present() { local d out=""; for d in $1; do [ -d "$d" ] && out="$out $d"; done; echo "${out# }"; }
SHARED_ALL=$(present "$RUNS/shared $RUNS_TMP/shared")
LANE_A_ALL=$(present "$RUNS/lane_a $RUNS_TMP/lane_a")
LANE_B_ALL=$(present "$RUNS/lane_b $RUNS_TMP/lane_b")
find_runs() { local roots="$1"; shift; [ -n "$roots" ] || return 0; find $roots "$@" 2>/dev/null; }
newest() { xargs -r ls -t 2>/dev/null | head -1; }
rows() { python3 -c 'import csv,sys; print(len(list(csv.DictReader(open(sys.argv[1])))))' "$1" 2>/dev/null || echo 0; }
successes() { python3 -c 'import csv,sys; print(sum(1 for r in csv.DictReader(open(sys.argv[1])) if str(r.get("success","")).strip().lower()=="true"))' "$1" 2>/dev/null || echo 0; }
episodes() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("total_episodes",0))' "$1" 2>/dev/null || echo 0; }

FAILED=0; WARNED=0
pass() { printf 'PASS  %-38s %s\n' "$1" "${2:-}"; }
fail() { printf 'FAIL  %-38s %s\n' "$1" "${2:-}"; FAILED=$((FAILED + 1)); }
warn() { printf 'WARN  %-38s %s\n' "$1" "${2:-}"; WARNED=$((WARNED + 1)); }

echo "GATE P8 — $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)  (epoch $EPOCH)"
echo "----------------------------------------------------------------------"

MIN_SUCCESS="${P8_MIN_SUCCESS:-15}"
MIN_EPISODES=20

# 1 --------------------------------------------------- B-oracle on the new episodes
ocsv=$(find_runs "$LANE_B_ALL" -maxdepth 5 -type f -path '*oracle_b*/out/eval/eval_results.csv' -newermt "$EPOCH" | newest)
if [ -n "$ocsv" ]; then
  r=$(rows "$ocsv"); s=$(successes "$ocsv")
  if [ "$r" -ge "$MIN_EPISODES" ]; then pass "P8 B-oracle >= $MIN_EPISODES episodes" "$r rows  $ocsv"
  else fail "P8 B-oracle >= $MIN_EPISODES episodes" "only $r rows in $ocsv"; fi
  if [ "$s" -ge "$MIN_SUCCESS" ]; then pass "P8 B-oracle success >= $MIN_SUCCESS/20" "$s/$r"
  else fail "P8 B-oracle success >= $MIN_SUCCESS/20" "$s/$r — the decoder cannot execute its own labels (WP 8.2)"; fi
else fail "P8 B-oracle success >= $MIN_SUCCESS/20" "no lane_b/*oracle_b*/out/eval/eval_results.csv newer than the epoch"; fi

# 2 --------------------------------------------------- token dataset, same episode set as lane A's
sinfo=$(find_runs "$LANE_B_ALL" -maxdepth 5 -type f -path '*_label_tokens*/out/gr00t_v2_sonic/meta/info.json' -newermt "$EPOCH" | newest)
ainfo=$(find_runs "$SHARED_ALL" -maxdepth 5 -type f -path '*_dataset*/out/gr00t_v2/meta/info.json' -newermt "$EPOCH" | newest)
if [ -n "$sinfo" ] && [ -n "$ainfo" ]; then
  ns=$(episodes "$sinfo"); na=$(episodes "$ainfo")
  if [ "$ns" -eq "$na" ] && [ "$ns" -gt 0 ]; then pass "P8 token dataset == lane A episodes" "$ns episodes  $sinfo"
  else fail "P8 token dataset == lane A episodes" "lane B $ns vs lane A $na — the lanes would train on different data (WP 8.3)"; fi
elif [ -z "$sinfo" ]; then fail "P8 token dataset == lane A episodes" "no lane_b/*_label_tokens*/out/gr00t_v2_sonic/meta/info.json newer than the epoch"
else fail "P8 token dataset == lane A episodes" "no P7 gr00t_v2 dataset to compare against"; fi

# 3 --------------------------------------------------- winner's ONNX pair present
onx=$(find_runs "$LANE_B_ALL" -maxdepth 4 -type f -path '*_export_onnx*/out/model_decoder.onnx' -newermt "$EPOCH" | newest)
if [ -n "$onx" ]; then pass "P8 decoder ONNX exported" "$onx"; else fail "P8 decoder ONNX exported" "no lane_b/*_export_onnx*/out/model_decoder.onnx newer than the epoch"; fi

# 4 --------------------------------------------------- finals kept (advisory)
fin=$(find_runs "$LANE_B_ALL" -maxdepth 3 -type d -path '*final/p8/*' -newermt "$EPOCH" | newest)
if [ -n "$fin" ]; then pass "P8 finals copied" "$fin"; else warn "P8 finals copied" "no lane_b/final/p8/<name>/ — advisory (WP 8.2)"; fi

echo "----------------------------------------------------------------------"
if [ "$FAILED" -eq 0 ]; then echo "GATE P8: PASS ($WARNED warning(s))"; exit 0; fi
echo "GATE P8: FAIL ($FAILED failing check(s), $WARNED warning(s))"; exit 1
