#!/usr/bin/env bash
# GATE P10 — the real comparison: four rows at 100 rollouts + the report
#
#   bash harness/gates/p10.sh
#
# Round-2 gate. Every check is scoped to artifacts newer than P10_EPOCH, so
# round-1 runs (2026-09-03) can never satisfy it.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS="$HOME/runs/franka-sonic"
RUNS_TMP="/tmp/franka-sonic"
EPOCH="${P10_EPOCH:-2026-09-04 02:00:00 UTC}"
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

echo "GATE P10 — $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)  (epoch $EPOCH)"
echo "----------------------------------------------------------------------"

MIN_ROLLOUTS="${P10_MIN_ROLLOUTS:-100}"
MIN_ORACLE="${P10_MIN_ORACLE:-100}"

newest_eval() {  # $1 = roots, $2 = path glob, $3 = "oracle" | "policy"
  if [ "$3" = "oracle" ]; then find_runs "$1" -maxdepth 5 -type f -path "$2" -newermt "$EPOCH" | newest
  else find_runs "$1" -maxdepth 5 -type f -path "$2" -newermt "$EPOCH" | grep -v oracle | newest; fi
}

# Explicit run folders win over "newest". P9 recorded the harness debt that
# makes this necessary: any tool that resolves "the run I just made" by recency
# is wrong as soon as two runs of the same lane and stage overlap, which is what
# P10 does by design (eight 200-rollout rows into the run roots the 20-rollout
# screens were still using). Set P10_LANE_A_EVAL / P10_LANE_B_EVAL /
# P10_ORACLE_A / P10_ORACLE_B to a run folder to pin a row; unset falls back to
# the newest folder as before. A pinned folder must still pass every check.
pin_or_newest() {  # $1 = pinned run folder ("" for none), $2..$4 = newest_eval args
  if [ -n "$1" ]; then
    local c="$1/out/eval/eval_results.csv"
    [ -f "$c" ] && printf '%s\n' "$c"
    return
  fi
  newest_eval "$2" "$3" "$4"
}

acsv=$(pin_or_newest "${P10_LANE_A_EVAL:-}" "$LANE_A_ALL" '*_eval*/out/eval/eval_results.csv' policy)
bcsv=$(pin_or_newest "${P10_LANE_B_EVAL:-}" "$LANE_B_ALL" '*_eval*/out/eval/eval_results.csv' policy)
oa=$(pin_or_newest "${P10_ORACLE_A:-}" "$LANE_A_ALL" '*oracle_a*/out/eval/eval_results.csv' oracle)
ob=$(pin_or_newest "${P10_ORACLE_B:-}" "$LANE_B_ALL" '*oracle_b*/out/eval/eval_results.csv' oracle)

check_rows() {  # $1 = label, $2 = csv, $3 = minimum
  if [ -z "$2" ]; then fail "$1 >= $3 rollouts" "no eval_results.csv newer than the epoch"; return; fi
  local r; r=$(rows "$2")
  if [ "$r" -ge "$3" ]; then pass "$1 >= $3 rollouts" "$r rows  $2"
  else fail "$1 >= $3 rollouts" "only $r rows in $2"; fi
}

check_rows "P10 lane A eval"  "$acsv" "$MIN_ROLLOUTS"
check_rows "P10 lane B eval"  "$bcsv" "$MIN_ROLLOUTS"
check_rows "P10 A-oracle"     "$oa"   "$MIN_ORACLE"
check_rows "P10 B-oracle"     "$ob"   "$MIN_ORACLE"

# same n for both policies, or the comparison is not like for like
if [ -n "$acsv" ] && [ -n "$bcsv" ]; then
  ra=$(rows "$acsv"); rb=$(rows "$bcsv")
  if [ "$ra" -eq "$rb" ]; then pass "P10 both policies, same n" "$ra rollouts each"
  else warn "P10 both policies, same n" "lane A $ra vs lane B $rb rollouts"; fi
fi

# the two policy rows must be the checkpoints the P9 screens pre-registered
checkpoint_of() { python3 -c 'import json,sys; print((json.load(open(sys.argv[1])).get("args") or {}).get("checkpoint") or "")' "$1" 2>/dev/null; }
for spec in "lane A:$acsv:${P10_CKPT_A:-checkpoint-20000}" "lane B:$bcsv:${P10_CKPT_B:-checkpoint-17500}"; do
  lbl=${spec%%:*}; rest=${spec#*:}; csvp=${rest%%:*}; want=${rest#*:}
  if [ -z "$csvp" ]; then continue; fi
  runf=$(dirname "$(dirname "$(dirname "$csvp")")")
  got=$(checkpoint_of "$runf/config.json")
  case "$got" in
    */"$want") pass "P10 $lbl row is the pre-registered ckpt" "$want" ;;
    "")        warn "P10 $lbl row is the pre-registered ckpt" "no checkpoint stamped in $runf/config.json" ;;
    *)         warn "P10 $lbl row is the pre-registered ckpt" "row measures ${got##*/}, screens chose $want" ;;
  esac
done

# report regenerated after the last eval
rep="$REPO_ROOT/plan/REPORT.md"
last=$(printf '%s\n' "$acsv" "$bcsv" "$oa" "$ob" | grep -v '^$' | xargs -r ls -t 2>/dev/null | head -1)
if [ -f "$rep" ] && [ -n "$last" ] && [ "$rep" -nt "$last" ]; then pass "P10 plan/REPORT.md regenerated" "$(stat -c %y "$rep" 2>/dev/null | cut -c1-16)"
else fail "P10 plan/REPORT.md regenerated" "plan/REPORT.md must be newer than the newest P10 eval csv (python3 harness/report/aggregate.py)"; fi

echo "----------------------------------------------------------------------"
if [ "$FAILED" -eq 0 ]; then echo "GATE P10: PASS ($WARNED warning(s))"; exit 0; fi
echo "GATE P10: FAIL ($FAILED failing check(s), $WARNED warning(s))"; exit 1
