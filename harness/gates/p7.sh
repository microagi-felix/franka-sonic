#!/usr/bin/env bash
# GATE P7 — round-2 demo set: bigger, wider, exported and validated
#
#   bash harness/gates/p7.sh
#
# Round-2 gate. Every check is scoped to artifacts newer than P7_EPOCH, so
# round-1 runs (2026-09-03) can never satisfy it.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS="$HOME/runs/franka-sonic"
RUNS_TMP="/tmp/franka-sonic"
EPOCH="${P7_EPOCH:-2026-09-04 02:00:00 UTC}"
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

echo "GATE P7 — $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)  (epoch $EPOCH)"
echo "----------------------------------------------------------------------"

MIN_EPISODES="${P7_MIN_EPISODES:-600}"

# 1 --------------------------------------------------- new demo set, exported
exp=$(find_runs "$SHARED_ALL" -maxdepth 3 -type f -path '*_demos*/out/export.done' -newermt "$EPOCH" | newest)
if [ -n "$exp" ]; then pass "P7 demos exported" "$(cat "$exp" | head -c 60)  ${exp%/out/export.done}"
else fail "P7 demos exported" "no shared/*_demos*/out/export.done newer than the epoch (WP 7.2/7.3)"; fi

# 2 --------------------------------------------------- replay check on a generated episode
rep=$(find_runs "$SHARED_ALL" -maxdepth 3 -type f -path '*_demos*/out/replay.done' -newermt "$EPOCH" | newest)
if [ -n "$rep" ]; then pass "P7 replay check" "$(cat "$rep" | head -c 60)"
else fail "P7 replay check" "no shared/*_demos*/out/replay.done newer than the epoch (WP 7.4)"; fi

# 3 --------------------------------------------------- dataset with enough episodes
info=$(find_runs "$SHARED_ALL" -maxdepth 5 -type f -path '*_dataset*/out/gr00t_v2/meta/info.json' -newermt "$EPOCH" | newest)
if [ -n "$info" ]; then
  n=$(episodes "$info")
  if [ "$n" -ge "$MIN_EPISODES" ]; then pass "P7 dataset >= $MIN_EPISODES episodes" "$n episodes  $info"
  else fail "P7 dataset >= $MIN_EPISODES episodes" "only $n episodes in $info"; fi
else fail "P7 dataset >= $MIN_EPISODES episodes" "no shared/*_dataset*/out/gr00t_v2/meta/info.json newer than the epoch"; fi

# 4 --------------------------------------------------- spawn coverage recorded
cov=$(find_runs "$SHARED_ALL" -maxdepth 3 -type f -path '*_demos*/out/coverage.json' -newermt "$EPOCH" | newest)
if [ -n "$cov" ]; then
  ok=$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1])).get("covers_eval","")).lower())' "$cov" 2>/dev/null)
  if [ "$ok" = "true" ]; then pass "P7 spawn coverage vs eval range" "$cov"
  else warn "P7 spawn coverage vs eval range" "covers_eval=$ok in $cov — finding, not a failure (WP 7.1)"; fi
else fail "P7 spawn coverage vs eval range" "no shared/*_demos*/out/coverage.json newer than the epoch (WP 7.1)"; fi

echo "----------------------------------------------------------------------"
if [ "$FAILED" -eq 0 ]; then echo "GATE P7: PASS ($WARNED warning(s))"; exit 0; fi
echo "GATE P7: FAIL ($FAILED failing check(s), $WARNED warning(s))"; exit 1
