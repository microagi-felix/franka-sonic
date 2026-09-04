#!/usr/bin/env bash
# GATE P9 — both policies trained long, every checkpoint screened, a winner named per lane
#
#   bash harness/gates/p9.sh
#
# Round-2 gate. Every check is scoped to artifacts newer than P9_EPOCH, so
# round-1 runs (2026-09-03) can never satisfy it.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS="$HOME/runs/franka-sonic"
RUNS_TMP="/tmp/franka-sonic"
EPOCH="${P9_EPOCH:-2026-09-04 02:00:00 UTC}"
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

echo "GATE P9 — $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)  (epoch $EPOCH)"
echo "----------------------------------------------------------------------"

MIN_SCREENED="${P9_MIN_SCREENED:-4}"

# 1 --------------------------------------------------- screening evals per lane
for lane in lane_a lane_b; do
  case "$lane" in lane_a) roots="$LANE_A_ALL";; *) roots="$LANE_B_ALL";; esac
  n=$(find_runs "$roots" -maxdepth 5 -type f -path '*_eval*/out/eval/eval_results.csv' -newermt "$EPOCH" | grep -vc oracle)
  n=${n:-0}
  if [ "$n" -ge "$MIN_SCREENED" ]; then pass "P9 $lane screened >= $MIN_SCREENED ckpts" "$n eval runs newer than the epoch"
  else fail "P9 $lane screened >= $MIN_SCREENED ckpts" "only $n eval runs newer than the epoch (WP 9.2)"; fi
done

# 2 --------------------------------------------------- checkpoints beyond round 1's 2000 steps
for lane in lane_a lane_b; do
  case "$lane" in lane_a) roots="$LANE_A_ALL";; *) roots="$LANE_B_ALL";; esac
  n=$(find_runs "$roots" -maxdepth 6 -type d -name 'checkpoint-*' -newermt "$EPOCH" | wc -l)
  n=${n:-0}
  if [ "$n" -ge 4 ]; then pass "P9 $lane >= 4 checkpoints" "$n checkpoint dirs newer than the epoch"
  else fail "P9 $lane >= 4 checkpoints" "only $n checkpoint dirs newer than the epoch (WP 9.1)"; fi
done

# 3 --------------------------------------------------- the stopping rule named a winner per lane
for lane in lane_a lane_b; do
  line=$(grep -oE "P9 BEST $lane=[^ ]+" "$STATUS" 2>/dev/null | tail -1)
  ckpt=${line#P9 BEST $lane=}
  if [ -n "$line" ] && [ -d "$ckpt" ]; then pass "P9 best checkpoint ($lane)" "$ckpt"
  elif [ -n "$line" ]; then fail "P9 best checkpoint ($lane)" "STATUS names $ckpt but it is not a directory"
  else fail "P9 best checkpoint ($lane)" "no 'P9 BEST $lane=<path>' line in plan/STATUS.md (WP 9.3)"; fi
done

echo "----------------------------------------------------------------------"
if [ "$FAILED" -eq 0 ]; then echo "GATE P9: PASS ($WARNED warning(s))"; exit 0; fi
echo "GATE P9: FAIL ($FAILED failing check(s), $WARNED warning(s))"; exit 1
