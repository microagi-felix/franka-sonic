#!/usr/bin/env bash
# GATE P12 — round 3b: the concurrent seeded re-measurement of both round-3 headline
# checkpoints, lane A's budget extension (screened twice per checkpoint, best and last at
# 200 rollouts), the determinism probe of the lane-B policy function, the report.
#
#   bash harness/gates/p12.sh
#
# Every check is scoped to artifacts newer than P12_EPOCH. A row counts as a re-measurement
# only if its own cmd.sh carries a seed (recorded as --seed N, the server's flag) and names a checkpoint-20000 (lane A) or
# checkpoint-10000/-17500 (lane B) of a 2026-09-05 fine-tune or its final/p11 copy; a row of
# the lane-A extension counts only if the checkpoint named in its cmd.sh is newer than the
# epoch. Round-3 rows can therefore never satisfy a round-3b check.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS="$HOME/runs/franka-sonic"
RUNS_TMP="/tmp/franka-sonic"
EPOCH="${P12_EPOCH:-2026-09-09 00:00:00 UTC}"
STATUS="$REPO_ROOT/plan/STATUS.md"

present() { local d out=""; for d in $1; do [ -d "$d" ] && out="$out $d"; done; echo "${out# }"; }
LANE_A_ALL=$(present "$RUNS/lane_a $RUNS_TMP/lane_a")
LANE_B_ALL=$(present "$RUNS/lane_b $RUNS_TMP/lane_b")
find_runs() { local roots="$1"; shift; [ -n "$roots" ] || return 0; find $roots "$@" 2>/dev/null; }
newest() { xargs -r ls -t 2>/dev/null | head -1; }
rows() { python3 -c 'import csv,sys; print(len(list(csv.DictReader(open(sys.argv[1])))))' "$1" 2>/dev/null || echo 0; }
newer() { [ -n "${1:-}" ] && [ -e "$1" ] && [ -n "$(find "$1" -maxdepth 0 -newermt "$EPOCH" 2>/dev/null)" ]; }
run_of_csv() { dirname "$(dirname "$(dirname "$1")")"; }

FAILED=0; WARNED=0
pass() { printf 'PASS  %-44s %s\n' "$1" "${2:-}"; }
fail() { printf 'FAIL  %-44s %s\n' "$1" "${2:-}"; FAILED=$((FAILED + 1)); }
warn() { printf 'WARN  %-44s %s\n' "$1" "${2:-}"; WARNED=$((WARNED + 1)); }

echo "GATE P12 — $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)  (epoch $EPOCH)"
echo "----------------------------------------------------------------------"

MIN_ROLLOUTS="${P12_MIN_ROLLOUTS:-200}"
MIN_REMEASURE_PER_LANE="${P12_MIN_REMEASURE_PER_LANE:-2}"
MIN_CKPTS="${P12_MIN_CKPTS:-8}"
MIN_SCREENS="${P12_MIN_SCREENS:-16}"
MIN_EXT_ROWS="${P12_MIN_EXT_ROWS:-1}"

# 1 --------------------------------------------------- WP 12.0: the rescue line (a path or MISSING)
line=$(grep -oE "P12 RESCUE lane_b=[^ ]+" "$STATUS" 2>/dev/null | tail -1)
val=${line#P12 RESCUE lane_b=}
if [ -n "$line" ] && [ "$val" = "MISSING" ]; then warn "P12 rescue of lane B checkpoint-10000" "STATUS says MISSING (pod restart) — fallback rows on checkpoint-17500 expected"
elif [ -n "$line" ] && [ -f "$val/config.json" ]; then pass "P12 rescue of lane B checkpoint-10000" "$val"
elif [ -n "$line" ]; then fail "P12 rescue of lane B checkpoint-10000" "STATUS names $val but it has no config.json"
else fail "P12 rescue of lane B checkpoint-10000" "no 'P12 RESCUE lane_b=<path|MISSING>' line in plan/STATUS.md (WP 12.0)"; fi

# 2 --------------------------------------------------- WP 12.1: seeded re-measurement rows of the round-3 headline checkpoints
count_remeasure() {  # $1 = roots, $2 = checkpoint regex -> number of qualifying seeded rows launched after the epoch
  local csv run ck c=0
  for csv in $(find_runs "$1" -maxdepth 5 -type f -path '*_eval*/out/eval/eval_results.csv' -newermt "$EPOCH" | grep -v oracle | grep -v probe); do
    run=$(run_of_csv "$csv")
    [ -n "$(find "$run/cmd.sh" -maxdepth 0 -newermt "$EPOCH" 2>/dev/null)" ] || continue
    grep -qE -- "--(server-)?seed[= ][0-9]+" "$run/cmd.sh" 2>/dev/null || continue   # cmd.sh records the server flag as --seed N
    ck=$(grep -oE '[^ "]*/checkpoint-[0-9]+' "$run/cmd.sh" 2>/dev/null | head -1)
    echo "$ck" | grep -qE "$2" || continue
    [ "$(rows "$csv")" -ge "$MIN_ROLLOUTS" ] && c=$((c + 1))
  done
  echo "$c"
}
na=$(count_remeasure "$LANE_A_ALL" '(2026-09-05_finetune-2/out/checkpoints|final/p11)/checkpoint-20000$')
nb=$(count_remeasure "$LANE_B_ALL" '(2026-09-05_finetune-2/out/checkpoints|final/p11)/checkpoint-(10000|17500)$')
if [ "$na" -ge "$MIN_REMEASURE_PER_LANE" ]; then pass "P12 lane_a seeded re-measurement rows >= $MIN_REMEASURE_PER_LANE" "$na rows at >= $MIN_ROLLOUTS"
else fail "P12 lane_a seeded re-measurement rows >= $MIN_REMEASURE_PER_LANE" "only $na seeded post-epoch rows of the round-3 checkpoint-20000 at >= $MIN_ROLLOUTS (WP 12.1)"; fi
if [ "$nb" -ge "$MIN_REMEASURE_PER_LANE" ]; then pass "P12 lane_b seeded re-measurement rows >= $MIN_REMEASURE_PER_LANE" "$nb rows at >= $MIN_ROLLOUTS"
else fail "P12 lane_b seeded re-measurement rows >= $MIN_REMEASURE_PER_LANE" "only $nb seeded post-epoch rows of the round-3 checkpoint-10000/-17500 at >= $MIN_ROLLOUTS (WP 12.1)"; fi

# 3 --------------------------------------------------- WP 12.2: the lane-A extension
line=$(grep -oE "P12 INIT lane_a=[^ ]+" "$STATUS" 2>/dev/null | tail -1)
ck=${line#P12 INIT lane_a=}
if [ -n "$line" ] && [ -f "$ck/config.json" ]; then pass "P12 lane_a warm start" "$ck"
elif [ -n "$line" ]; then fail "P12 lane_a warm start" "STATUS names $ck but it has no config.json"
else fail "P12 lane_a warm start" "no 'P12 INIT lane_a=<checkpoint dir>' line in plan/STATUS.md (WP 12.2)"; fi

n=$(find_runs "$LANE_A_ALL" -maxdepth 6 -type d -name 'checkpoint-*' -newermt "$EPOCH" | grep -v final | wc -l); n=${n:-0}
if [ "$n" -ge "$MIN_CKPTS" ]; then pass "P12 lane_a >= $MIN_CKPTS extension checkpoints" "$n checkpoint dirs newer than the epoch"
else fail "P12 lane_a >= $MIN_CKPTS extension checkpoints" "only $n checkpoint dirs newer than the epoch (WP 12.2)"; fi

count_ext() {  # $1 = min rows -> evals of post-epoch lane-A checkpoints with >= $1 rows
  local csv run ck c=0
  for csv in $(find_runs "$LANE_A_ALL" -maxdepth 5 -type f -path '*_eval*/out/eval/eval_results.csv' -newermt "$EPOCH" | grep -v oracle); do
    run=$(run_of_csv "$csv")
    ck=$(grep -oE '[^ "]*/checkpoint-[0-9]+' "$run/cmd.sh" 2>/dev/null | head -1)
    newer "$ck" || continue
    [ "$(rows "$csv")" -ge "$1" ] && c=$((c + 1))
  done
  echo "$c"
}
n=$(count_ext 20)
if [ "$n" -ge "$MIN_SCREENS" ]; then pass "P12 lane_a extension screens >= $MIN_SCREENS" "$n screens of post-epoch checkpoints (two per checkpoint)"
else fail "P12 lane_a extension screens >= $MIN_SCREENS" "only $n screens of post-epoch lane-A checkpoints (WP 12.2 wants two per checkpoint)"; fi
n=$(count_ext "$MIN_ROLLOUTS")
if [ "$n" -ge "$MIN_EXT_ROWS" ]; then pass "P12 lane_a extension rows >= $MIN_EXT_ROWS at >= $MIN_ROLLOUTS" "$n rows"
else fail "P12 lane_a extension rows >= $MIN_EXT_ROWS at >= $MIN_ROLLOUTS" "only $n rows of post-epoch lane-A checkpoints at >= $MIN_ROLLOUTS (WP 12.2)"; fi

line=$(grep -oE "P12 BEST lane_a=[^ ]+" "$STATUS" 2>/dev/null | tail -1)
ck=${line#P12 BEST lane_a=}
if [ -n "$line" ] && [ -d "$ck" ] && newer "$ck"; then pass "P12 best extension checkpoint (lane_a)" "$ck"
elif [ -n "$line" ] && [ -d "$ck" ]; then fail "P12 best extension checkpoint (lane_a)" "STATUS names $ck but it predates the epoch"
elif [ -n "$line" ]; then fail "P12 best extension checkpoint (lane_a)" "STATUS names $ck but it is not a directory"
else fail "P12 best extension checkpoint (lane_a)" "no 'P12 BEST lane_a=<path>' line in plan/STATUS.md (WP 12.2)"; fi

# 4 --------------------------------------------------- WP 12.3: the determinism probe
line=$(grep -oE "P12 PROBE=[^ ]+" "$STATUS" 2>/dev/null | tail -1)
pf=${line#P12 PROBE=}
if [ -n "$line" ] && [ -f "$pf" ] && grep -q "^VERDICT:" "$pf"; then pass "P12 determinism probe" "$(grep -m1 '^VERDICT:' "$pf" | cut -c1-90)"
elif [ -n "$line" ] && [ -f "$pf" ]; then fail "P12 determinism probe" "$pf has no line starting with VERDICT:"
elif [ -n "$line" ]; then fail "P12 determinism probe" "STATUS names $pf but it does not exist"
else fail "P12 determinism probe" "no 'P12 PROBE=<markdown path>' line in plan/STATUS.md (WP 12.3)"; fi

# 5 --------------------------------------------------- WP 12.5: report regenerated after the last post-epoch eval
rep="$REPO_ROOT/plan/REPORT.md"
last=$(find_runs "$LANE_A_ALL $LANE_B_ALL" -maxdepth 5 -type f -path '*/out/eval/eval_results.csv' -newermt "$EPOCH" | newest)
if [ -f "$rep" ] && [ -n "$last" ] && [ "$rep" -nt "$last" ]; then pass "P12 plan/REPORT.md regenerated" "$(stat -c %y "$rep" 2>/dev/null | cut -c1-16)"
elif [ -f "$rep" ] && [ -z "$last" ]; then fail "P12 plan/REPORT.md regenerated" "no post-epoch eval csv found at all"
else fail "P12 plan/REPORT.md regenerated" "plan/REPORT.md must be newer than the newest post-epoch eval csv (WP 12.5)"; fi
grep -qiE "round 3b" "$rep" 2>/dev/null && pass "P12 report has a Round 3b section" "" || fail "P12 report has a Round 3b section" "no 'Round 3b' heading in plan/REPORT.md (WP 12.5)"

echo "----------------------------------------------------------------------"
if [ "$FAILED" -eq 0 ]; then echo "GATE P12: PASS ($WARNED warning(s))"; exit 0; fi
echo "GATE P12: FAIL ($FAILED failing check(s), $WARNED warning(s))"; exit 1
