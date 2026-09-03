#!/usr/bin/env bash
# GATE P1 — lane A: GR00T direct. Prints PASS / FAIL / WARN per check and exits
# non-zero if any check FAILed. WARN never fails the gate but is reported.
#
#   bash harness/gates/p1.sh
#
# Checks are artifact-based: a dataset with a modality.json, a checkpoint-2000,
# an open-loop eval log, an eval run folder with >= 20 episodes, and the
# A-oracle replay. A gate reads run folders and exit codes, never prose
# (AGENTS.md rule f). Run-folder conventions: plan/PLAN.md "Artifact
# conventions".
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS="$HOME/runs/franka-sonic"
RUNS_TMP="/tmp/franka-sonic"          # bakeoff.py's instance-local fallback root
MIN_EPISODES=20

# Run folders live under $HOME normally and under /tmp when bakeoff.py routed
# them there because home was below the storage floor (AGENTS.md rule k), so
# every lookup searches BOTH roots. present() drops roots that do not exist —
# a `find` with no path argument would silently walk $PWD instead.
present() { local d out=""; for d in $1; do [ -d "$d" ] && out="$out $d"; done; echo "${out# }"; }
LANE_A_ALL=$(present "$RUNS/lane_a $RUNS_TMP/lane_a")
DATASET_ALL=$(present "$RUNS $RUNS_TMP $HOME/datasets/franka-sonic")

# find(1) over a space-separated root list; a no-op when the list is empty.
find_runs() { local roots="$1"; shift; [ -n "$roots" ] || return 0; find $roots "$@" 2>/dev/null; }

FAILED=0
WARNED=0

pass() { printf 'PASS  %-34s %s\n' "$1" "${2:-}"; }
fail() { printf 'FAIL  %-34s %s\n' "$1" "${2:-}"; FAILED=$((FAILED + 1)); }
warn() { printf 'WARN  %-34s %s\n' "$1" "${2:-}"; WARNED=$((WARNED + 1)); }

# rows in an eval_results.csv, header excluded
csv_rows() { awk 'NF { n++ } END { print (n > 0 ? n - 1 : 0) }' "$1" 2>/dev/null; }

# newest eval_results.csv under the roots $1 whose path matches ($2) / does not match ($3)
find_eval_csv() {
  local roots="$1" want="$2" skip="$3" p
  find_runs "$roots" -maxdepth 5 -type f -path '*/out/eval/eval_results.csv' \
       -printf '%T@ %p\n' | sort -rn | cut -d' ' -f2- |
  while read -r p; do
    if [ -n "$want" ]; then case "$p" in *"$want"*) ;; *) continue ;; esac; fi
    if [ -n "$skip" ]; then case "$p" in *"$skip"*) continue ;; esac; fi
    echo "$p"
  done | head -1
}

echo "GATE P1 — $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)"
echo "----------------------------------------------------------------------"

# 1 -------------------------------------------------------------- GR00T v2 dataset
# The lane-A dataset is the one whose path does NOT mention sonic (that is P3's).
ds=$(find_runs "$DATASET_ALL" -maxdepth 7 -type f -name modality.json \
       -path '*/meta/*' | grep -vi sonic | head -1)
if [ -n "$ds" ]; then
  keys=$(grep -c '"' "$ds" 2>/dev/null)
  pass "GR00T v2 dataset modality.json" "$ds ($keys quoted lines)"
  if grep -qi 'joint_pos\|action' "$ds"; then
    pass "modality.json has action keys"
  else
    warn "modality.json has action keys" "no action/joint_pos key found in $ds"
  fi
else
  fail "GR00T v2 dataset modality.json" \
       "expected <run>/out/gr00t_v2/meta/modality.json (P1 WP 1.1)"
fi

# 2 -------------------------------------------------------------- checkpoint-2000
ckpt=$(find_runs "$LANE_A_ALL" -maxdepth 6 -type d -name 'checkpoint-2000' | head -1)
if [ -n "$ckpt" ]; then
  pass "lane A checkpoint-2000" "$ckpt ($(du -sh "$ckpt" 2>/dev/null | cut -f1))"
else
  fail "lane A checkpoint-2000" "expected <run>/out/checkpoints/checkpoint-2000 (P1 WP 1.3)"
fi

# 3 -------------------------------------------------------------- open-loop eval
ol_dir=$(find_runs "$LANE_A_ALL" -maxdepth 1 -type d -name '*open_loop*' | sort | tail -1)
if [ -n "$ol_dir" ]; then
  ol_log=$(find "$ol_dir" -maxdepth 2 -type f \( -name 'run.log' -o -name '*.json' \) \
             -size +0 2>/dev/null | head -1)
  if [ -n "$ol_log" ]; then
    pass "open-loop eval log" "$ol_log"
  else
    fail "open-loop eval log" "$ol_dir exists but holds no non-empty log/json"
  fi
else
  fail "open-loop eval log" "expected a lane_a/*open_loop* run folder (P1 WP 1.4)"
fi

# 4 -------------------------------------------------------------- policy eval >= 20
csv=$(find_eval_csv "$LANE_A_ALL" "" "oracle")
if [ -n "$csv" ]; then
  rows=$(csv_rows "$csv")
  if [ "${rows:-0}" -ge "$MIN_EPISODES" ]; then
    pass "lane A eval >= $MIN_EPISODES episodes" "$rows episodes  $csv"
  else
    fail "lane A eval >= $MIN_EPISODES episodes" "only $rows episode row(s) in $csv"
  fi
else
  fail "lane A eval >= $MIN_EPISODES episodes" \
       "expected lane_a/*_eval/out/eval/eval_results.csv (P1 WP 1.6)"
fi

# 5 -------------------------------------------------------------- A-oracle
ocsv=$(find_eval_csv "$LANE_A_ALL" "oracle" "")
if [ -n "$ocsv" ]; then
  orows=$(csv_rows "$ocsv")
  if [ "${orows:-0}" -ge 1 ]; then
    pass "A-oracle replay csv" "$orows episode(s)  $ocsv"
  else
    fail "A-oracle replay csv" "$ocsv has no episode rows"
  fi
else
  fail "A-oracle replay csv" \
       "expected lane_a/*_oracle_a/out/eval/eval_results.csv (P1 WP 1.7)"
fi

# 6 -------------------------------------------------------------- repo artifacts (WARN)
if [ -f "$REPO_ROOT/harness/lane_a/modality_config_dual_fr3.py" ]; then
  pass "modality config in repo" "harness/lane_a/modality_config_dual_fr3.py"
else
  warn "modality config in repo" "not at harness/lane_a/modality_config_dual_fr3.py — code belongs in the repo"
fi

echo "----------------------------------------------------------------------"
if [ "$FAILED" -eq 0 ]; then
  echo "GATE P1: PASS ($WARNED warning(s))"
  exit 0
fi
echo "GATE P1: FAIL ($FAILED failing check(s), $WARNED warning(s))"
exit 1
