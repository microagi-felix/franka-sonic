#!/usr/bin/env bash
# GATE P11 — round 3: both lanes warm-restarted on the wide + eval-matched union, every
# checkpoint screened, a winner named per lane, the pre-registered rows at 200 rollouts,
# the eval-box B-oracle, the report.
#
#   bash harness/gates/p11.sh
#
# Every check is scoped to artifacts newer than P11_EPOCH. An evaluation counts only when
# the checkpoint (or token set) named in its own cmd.sh is itself newer than the epoch, so a
# round-2 row that happens to run after the epoch can never satisfy a round-3 check.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS="$HOME/runs/franka-sonic"
RUNS_TMP="/tmp/franka-sonic"
EPOCH="${P11_EPOCH:-2026-09-05 08:00:00 UTC}"
STATUS="$REPO_ROOT/plan/STATUS.md"

present() { local d out=""; for d in $1; do [ -d "$d" ] && out="$out $d"; done; echo "${out# }"; }
SHARED_ALL=$(present "$RUNS/shared $RUNS_TMP/shared")
LANE_A_ALL=$(present "$RUNS/lane_a $RUNS_TMP/lane_a")
LANE_B_ALL=$(present "$RUNS/lane_b $RUNS_TMP/lane_b")
find_runs() { local roots="$1"; shift; [ -n "$roots" ] || return 0; find $roots "$@" 2>/dev/null; }
newest() { xargs -r ls -t 2>/dev/null | head -1; }
rows() { python3 -c 'import csv,sys; print(len(list(csv.DictReader(open(sys.argv[1])))))' "$1" 2>/dev/null || echo 0; }
episodes() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("total_episodes",0))' "$1" 2>/dev/null || echo 0; }
newer() { [ -n "${1:-}" ] && [ -e "$1" ] && [ -n "$(find "$1" -maxdepth 0 -newermt "$EPOCH" 2>/dev/null)" ]; }
run_of_csv() { dirname "$(dirname "$(dirname "$1")")"; }

FAILED=0; WARNED=0
pass() { printf 'PASS  %-40s %s\n' "$1" "${2:-}"; }
fail() { printf 'FAIL  %-40s %s\n' "$1" "${2:-}"; FAILED=$((FAILED + 1)); }
warn() { printf 'WARN  %-40s %s\n' "$1" "${2:-}"; WARNED=$((WARNED + 1)); }

echo "GATE P11 — $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)  (epoch $EPOCH)"
echo "----------------------------------------------------------------------"

MIN_EPISODES="${P11_MIN_EPISODES:-1900}"
MIN_CKPTS="${P11_MIN_CKPTS:-8}"
MIN_SCREENED="${P11_MIN_SCREENED:-8}"
MIN_ROLLOUTS="${P11_MIN_ROLLOUTS:-200}"
MIN_ROWS="${P11_MIN_ROWS:-3}"
MIN_ORACLE="${P11_MIN_ORACLE:-200}"

# 1 --------------------------------------------------- the union datasets (WP 11.1 / 11.3)
ainfo=$(find_runs "$SHARED_ALL" -maxdepth 5 -type f -path '*_dataset*/out/gr00t_v2/meta/info.json' -newermt "$EPOCH" | newest)
binfo=$(find_runs "$LANE_B_ALL" -maxdepth 5 -type f -path '*_label_tokens*/out/gr00t_v2_sonic/meta/info.json' -newermt "$EPOCH" | newest)
na=$(episodes "${ainfo:-/dev/null}"); nb=$(episodes "${binfo:-/dev/null}")
if [ -n "$ainfo" ] && [ "$na" -ge "$MIN_EPISODES" ]; then pass "P11 lane A union dataset >= $MIN_EPISODES eps" "$na episodes  $ainfo"
else fail "P11 lane A union dataset >= $MIN_EPISODES eps" "newest gr00t_v2 after the epoch has $na episodes (WP 11.1)"; fi
if [ -n "$binfo" ] && [ "$nb" -ge "$MIN_EPISODES" ]; then pass "P11 lane B union dataset >= $MIN_EPISODES eps" "$nb episodes  $binfo"
else fail "P11 lane B union dataset >= $MIN_EPISODES eps" "newest gr00t_v2_sonic after the epoch has $nb episodes (WP 11.3)"; fi
if [ -n "$ainfo" ] && [ -n "$binfo" ] && [ "$na" -eq "$nb" ]; then pass "P11 both lanes, same episode count" "$na"
else fail "P11 both lanes, same episode count" "lane A $na vs lane B $nb"; fi

# 2 --------------------------------------------------- the warm start named per lane (WP 11.4)
for lane in lane_a lane_b; do
  line=$(grep -oE "P11 INIT $lane=[^ ]+" "$STATUS" 2>/dev/null | tail -1)
  ck=${line#P11 INIT $lane=}
  if [ -n "$line" ] && [ -f "$ck/config.json" ]; then pass "P11 warm start ($lane)" "$ck"
  elif [ -n "$line" ]; then fail "P11 warm start ($lane)" "STATUS names $ck but it has no config.json"
  else fail "P11 warm start ($lane)" "no 'P11 INIT $lane=<checkpoint dir>' line in plan/STATUS.md (WP 11.4)"; fi
done

# 3 --------------------------------------------------- checkpoints per lane, all after the epoch
for lane in lane_a lane_b; do
  case "$lane" in lane_a) roots="$LANE_A_ALL";; *) roots="$LANE_B_ALL";; esac
  n=$(find_runs "$roots" -maxdepth 6 -type d -name 'checkpoint-*' -newermt "$EPOCH" | wc -l); n=${n:-0}
  if [ "$n" -ge "$MIN_CKPTS" ]; then pass "P11 $lane >= $MIN_CKPTS checkpoints" "$n checkpoint dirs newer than the epoch"
  else fail "P11 $lane >= $MIN_CKPTS checkpoints" "only $n checkpoint dirs newer than the epoch (WP 11.4)"; fi
done

# 4 --------------------------------------------------- screens and rows: evals of P11 checkpoints only
count_evals() {  # $1 = roots, $2 = min rows -> number of qualifying policy evals
  local csv run ck c=0
  for csv in $(find_runs "$1" -maxdepth 5 -type f -path '*_eval*/out/eval/eval_results.csv' -newermt "$EPOCH" | grep -v oracle); do
    run=$(run_of_csv "$csv")
    ck=$(grep -oE '[^ "]*/checkpoint-[0-9]+' "$run/cmd.sh" 2>/dev/null | head -1)
    newer "$ck" || continue
    [ "$(rows "$csv")" -ge "$2" ] && c=$((c + 1))
  done
  echo "$c"
}
for lane in lane_a lane_b; do
  case "$lane" in lane_a) roots="$LANE_A_ALL";; *) roots="$LANE_B_ALL";; esac
  n=$(count_evals "$roots" 20)
  if [ "$n" -ge "$MIN_SCREENED" ]; then pass "P11 $lane screened >= $MIN_SCREENED ckpts" "$n evals of post-epoch checkpoints"
  else fail "P11 $lane screened >= $MIN_SCREENED ckpts" "only $n evals of post-epoch checkpoints (WP 11.5)"; fi
  n=$(count_evals "$roots" "$MIN_ROLLOUTS")
  if [ "$n" -ge "$MIN_ROWS" ]; then pass "P11 $lane >= $MIN_ROWS rows at >= $MIN_ROLLOUTS" "$n rows"
  else fail "P11 $lane >= $MIN_ROWS rows at >= $MIN_ROLLOUTS" "only $n rows of post-epoch checkpoints at >= $MIN_ROLLOUTS rollouts (WP 11.6)"; fi
done

# 5 --------------------------------------------------- the rule named a winner per lane, and it is a P11 checkpoint
for lane in lane_a lane_b; do
  line=$(grep -oE "P11 BEST $lane=[^ ]+" "$STATUS" 2>/dev/null | tail -1)
  ck=${line#P11 BEST $lane=}
  if [ -n "$line" ] && [ -d "$ck" ] && newer "$ck"; then pass "P11 best checkpoint ($lane)" "$ck"
  elif [ -n "$line" ] && [ -d "$ck" ]; then fail "P11 best checkpoint ($lane)" "STATUS names $ck but it predates the epoch (a round-2 checkpoint?)"
  elif [ -n "$line" ]; then fail "P11 best checkpoint ($lane)" "STATUS names $ck but it is not a directory"
  else fail "P11 best checkpoint ($lane)" "no 'P11 BEST $lane=<path>' line in plan/STATUS.md (WP 11.5)"; fi
done

# 6 --------------------------------------------------- the eval-box B-oracle (tokens labelled after the epoch)
ob=""
for csv in $(find_runs "$LANE_B_ALL" -maxdepth 5 -type f -path '*oracle_b*/out/eval/eval_results.csv' -newermt "$EPOCH"); do
  run=$(run_of_csv "$csv")
  tok=$(grep -oE -- '--tokens [^ ]+' "$run/cmd.sh" 2>/dev/null | head -1 | cut -d' ' -f2)
  newer "$tok" || continue
  [ "$(rows "$csv")" -ge "$MIN_ORACLE" ] && ob="$csv"
done
if [ -n "$ob" ]; then pass "P11 eval-box B-oracle >= $MIN_ORACLE rollouts" "$(rows "$ob") rows  $ob"
else fail "P11 eval-box B-oracle >= $MIN_ORACLE rollouts" "no oracle_b run after the epoch with post-epoch tokens and >= $MIN_ORACLE rows (WP 11.6)"; fi

# 7 --------------------------------------------------- report regenerated after the last P11 eval
rep="$REPO_ROOT/plan/REPORT.md"
last=$(find_runs "$LANE_A_ALL $LANE_B_ALL" -maxdepth 5 -type f -path '*/out/eval/eval_results.csv' -newermt "$EPOCH" | newest)
if [ -f "$rep" ] && [ -n "$last" ] && [ "$rep" -nt "$last" ]; then pass "P11 plan/REPORT.md regenerated" "$(stat -c %y "$rep" 2>/dev/null | cut -c1-16)"
else fail "P11 plan/REPORT.md regenerated" "plan/REPORT.md must be newer than the newest post-epoch eval csv (WP 11.7)"; fi

echo "----------------------------------------------------------------------"
if [ "$FAILED" -eq 0 ]; then echo "GATE P11: PASS ($WARNED warning(s))"; exit 0; fi
echo "GATE P11: FAIL ($FAILED failing check(s), $WARNED warning(s))"; exit 1
