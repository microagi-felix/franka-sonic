#!/usr/bin/env bash
# GATE P6 — lane B redone on the P5 decoder: new token dataset -> GR00T
# fine-tune (P1's exact command) -> 20-rollout eval -> REPORT.md regenerated.
#
#   bash harness/gates/p6.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS="$HOME/runs/franka-sonic"
RUNS_TMP="/tmp/franka-sonic"
P6_EPOCH="2026-09-03 13:00:00 UTC"
MIN_EPISODES=20

present() { local d out=""; for d in $1; do [ -d "$d" ] && out="$out $d"; done; echo "${out# }"; }
LANE_B_ALL=$(present "$RUNS/lane_b $RUNS_TMP/lane_b")
find_runs() { local roots="$1"; shift; [ -n "$roots" ] || return 0; find $roots "$@" 2>/dev/null; }
newest() { xargs -r ls -t 2>/dev/null | head -1; }

FAILED=0; WARNED=0
pass() { printf 'PASS  %-34s %s\n' "$1" "${2:-}"; }
fail() { printf 'FAIL  %-34s %s\n' "$1" "${2:-}"; FAILED=$((FAILED + 1)); }
warn() { printf 'WARN  %-34s %s\n' "$1" "${2:-}"; WARNED=$((WARNED + 1)); }

echo "GATE P6 — $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)  (epoch $P6_EPOCH)"
echo "----------------------------------------------------------------------"

# 1 ------------------------------------------------- P5 passed (its oracle csv exists and is newer than the epoch)
ocsv=$(find_runs "$LANE_B_ALL" -maxdepth 5 -type f -path '*oracle_b*/out/eval/eval_results.csv' -newermt "$P6_EPOCH" | newest)
if [ -n "$ocsv" ]; then pass "P5 B-oracle present" "$ocsv"; else fail "P5 B-oracle present" "P6 needs P5's oracle_b run"; fi

# 2 ------------------------------------------------- new checkpoint-2000 for lane B
ckpt=$(find_runs "$LANE_B_ALL" -maxdepth 6 -type d -name 'checkpoint-2000' -newermt "$P6_EPOCH" | newest)
if [ -n "$ckpt" ]; then pass "lane B checkpoint-2000 (P6)" "$ckpt"; else fail "lane B checkpoint-2000 (P6)" "no lane_b/*_finetune*/out/checkpoints/checkpoint-2000 newer than the epoch"; fi

# 3 ------------------------------------------------- new lane B eval, >= 20 episodes, newer than the oracle
csv=$(find_runs "$LANE_B_ALL" -maxdepth 5 -type f -path '*_eval*/out/eval/eval_results.csv' -newermt "$P6_EPOCH" | grep -v oracle | newest)
if [ -n "$csv" ]; then
  rows=$(python3 -c 'import csv,sys; print(len(list(csv.DictReader(open(sys.argv[1])))))' "$csv" 2>/dev/null); rows=${rows:-0}
  if [ "$rows" -ge "$MIN_EPISODES" ]; then pass "lane B eval >= $MIN_EPISODES episodes (P6)" "$rows rows  $csv"; else fail "lane B eval >= $MIN_EPISODES episodes (P6)" "only $rows rows in $csv"; fi
  if [ -n "$ocsv" ] && [ "$csv" -nt "$ocsv" ]; then pass "lane B eval newer than the P5 oracle"; else warn "lane B eval newer than the P5 oracle" "eval csv is not newer than $ocsv"; fi
else fail "lane B eval >= $MIN_EPISODES episodes (P6)" "no lane_b/*_eval*/out/eval/eval_results.csv newer than the epoch"; fi

# 4 ------------------------------------------------- REPORT.md regenerated after the eval
rep="$REPO_ROOT/plan/REPORT.md"
if [ -f "$rep" ] && [ -n "${csv:-}" ] && [ "$rep" -nt "$csv" ]; then pass "plan/REPORT.md regenerated" "$(stat -c %y "$rep" 2>/dev/null | cut -c1-16)"; else fail "plan/REPORT.md regenerated" "plan/REPORT.md must be newer than the P6 eval csv (python3 harness/report/aggregate.py)"; fi

echo "----------------------------------------------------------------------"
if [ "$FAILED" -eq 0 ]; then echo "GATE P6: PASS ($WARNED warning(s))"; exit 0; fi
echo "GATE P6: FAIL ($FAILED failing check(s), $WARNED warning(s))"; exit 1
