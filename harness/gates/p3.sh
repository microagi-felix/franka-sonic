#!/usr/bin/env bash
# GATE P3 — lane B-2: GR00T over SONIC. Prints PASS / FAIL / WARN per check and
# exits non-zero if any check FAILed. WARN never fails the gate.
#
#   bash harness/gates/p3.sh
#
# Artifact-based: the token dataset, a checkpoint-2000 for lane B, an eval run
# folder with >= 20 episodes, and the B-oracle replay. Conventions:
# plan/PLAN.md "Artifact conventions".
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
LANE_B_ALL=$(present "$RUNS/lane_b $RUNS_TMP/lane_b")
DATASET_ALL=$(present "$RUNS $RUNS_TMP $HOME/datasets/franka-sonic")

# find(1) over a space-separated root list; a no-op when the list is empty.
find_runs() { local roots="$1"; shift; [ -n "$roots" ] || return 0; find $roots "$@" 2>/dev/null; }

FAILED=0
WARNED=0

pass() { printf 'PASS  %-34s %s\n' "$1" "${2:-}"; }
fail() { printf 'FAIL  %-34s %s\n' "$1" "${2:-}"; FAILED=$((FAILED + 1)); }
warn() { printf 'WARN  %-34s %s\n' "$1" "${2:-}"; WARNED=$((WARNED + 1)); }

csv_rows() { awk 'NF { n++ } END { print (n > 0 ? n - 1 : 0) }' "$1" 2>/dev/null; }

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

echo "GATE P3 — $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)"
echo "----------------------------------------------------------------------"

# 1 -------------------------------------------------------------- token dataset
# Lane B's dataset is the one whose path DOES mention sonic/token.
ds=$(find_runs "$DATASET_ALL" -maxdepth 7 -type f -name modality.json \
       -path '*/meta/*' | grep -iE 'sonic|token' | head -1)
if [ -n "$ds" ]; then
  pass "token dataset modality.json" "$ds"
  if grep -qiE 'token' "$ds"; then
    pass "modality.json mentions the token action"
  else
    warn "modality.json mentions the token action" \
         "no 'token' key in $ds — is this really the lane-B dataset?"
  fi
else
  fail "token dataset modality.json" \
       "expected lane_b/*_label_tokens/out/gr00t_v2_sonic/meta/modality.json (P3 WP 3.2)"
fi

# 2 -------------------------------------------------------------- checkpoint-2000
ckpt=$(find_runs "$LANE_B_ALL" -maxdepth 6 -type d -name 'checkpoint-2000' | head -1)
if [ -n "$ckpt" ]; then
  pass "lane B checkpoint-2000" "$ckpt ($(du -sh "$ckpt" 2>/dev/null | cut -f1))"
else
  fail "lane B checkpoint-2000" \
       "expected lane_b/*_finetune/out/checkpoints/checkpoint-2000 (P3 WP 3.3)"
fi

# 3 -------------------------------------------------------------- policy eval >= 20
csv=$(find_eval_csv "$LANE_B_ALL" "" "oracle")
if [ -n "$csv" ]; then
  rows=$(csv_rows "$csv")
  if [ "${rows:-0}" -ge "$MIN_EPISODES" ]; then
    pass "lane B eval >= $MIN_EPISODES episodes" "$rows episodes  $csv"
  else
    fail "lane B eval >= $MIN_EPISODES episodes" "only $rows episode row(s) in $csv"
  fi
else
  fail "lane B eval >= $MIN_EPISODES episodes" \
       "expected lane_b/*_eval/out/eval/eval_results.csv (P3 WP 3.4)"
fi

# 4 -------------------------------------------------------------- B-oracle
ocsv=$(find_eval_csv "$LANE_B_ALL" "oracle" "")
if [ -n "$ocsv" ]; then
  orows=$(csv_rows "$ocsv")
  if [ "${orows:-0}" -ge 1 ]; then
    pass "B-oracle replay csv" "$orows episode(s)  $ocsv"
  else
    fail "B-oracle replay csv" "$ocsv has no episode rows"
  fi
else
  fail "B-oracle replay csv" \
       "expected lane_b/*_oracle_b/out/eval/eval_results.csv (P3 WP 3.5)"
fi

# 5 -------------------------------------------------------------- repo artifacts (WARN)
srv=$(find "$REPO_ROOT/harness/lane_b" -maxdepth 1 -type f -name '*sonic*joint*.py' 2>/dev/null | head -1)
if [ -n "$srv" ]; then
  pass "lane B policy server in repo" "$srv"
else
  warn "lane B policy server in repo" \
       "no harness/lane_b/*sonic*joint*.py — the decoder-in-the-server code belongs in the repo"
fi

echo "----------------------------------------------------------------------"
if [ "$FAILED" -eq 0 ]; then
  echo "GATE P3: PASS ($WARNED warning(s))"
  exit 0
fi
echo "GATE P3: FAIL ($FAILED failing check(s), $WARNED warning(s))"
exit 1
