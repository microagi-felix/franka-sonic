#!/usr/bin/env bash
# GATE P4 — compare + decide. Prints PASS / FAIL / WARN per check and exits
# non-zero if any check FAILed. WARN never fails the gate.
#
#   bash harness/gates/p4.sh
#
# Artifact-based: plan/REPORT.md exists and holds a markdown table that names
# both lanes. Conventions: plan/PLAN.md "Artifact conventions".
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORT="$REPO_ROOT/plan/REPORT.md"
RUNS="$HOME/runs/franka-sonic"

FAILED=0
WARNED=0

pass() { printf 'PASS  %-34s %s\n' "$1" "${2:-}"; }
fail() { printf 'FAIL  %-34s %s\n' "$1" "${2:-}"; FAILED=$((FAILED + 1)); }
warn() { printf 'WARN  %-34s %s\n' "$1" "${2:-}"; WARNED=$((WARNED + 1)); }

echo "GATE P4 — $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)"
echo "----------------------------------------------------------------------"

# 1 -------------------------------------------------------------- report exists
if [ -s "$REPORT" ]; then
  pass "plan/REPORT.md" "$(wc -l < "$REPORT") lines"
else
  fail "plan/REPORT.md" "missing or empty — P4 WP 4.2"
  echo "----------------------------------------------------------------------"
  echo "GATE P4: FAIL ($FAILED failing check(s), $WARNED warning(s))"
  exit 1
fi

# 2 -------------------------------------------------------------- markdown table
rows=$(grep -c '^[[:space:]]*|' "$REPORT")
if [ "$rows" -ge 3 ]; then
  pass "REPORT.md has a table" "$rows table row(s)"
else
  fail "REPORT.md has a table" "only $rows line(s) start with '|'"
fi

# 3 -------------------------------------------------------------- both lanes named
a=$(grep -icE 'lane[ _-]?a' "$REPORT")
b=$(grep -icE 'lane[ _-]?b' "$REPORT")
if [ "$a" -ge 1 ] && [ "$b" -ge 1 ]; then
  pass "both lanes in the report" "lane A mentioned $a×, lane B $b×"
else
  fail "both lanes in the report" "lane A $a×, lane B $b× — the table must carry both"
fi

# 4 -------------------------------------------------------------- table names both lanes
if grep -E '^[[:space:]]*\|' "$REPORT" | grep -qiE 'lane[ _-]?a' && \
   grep -E '^[[:space:]]*\|' "$REPORT" | grep -qiE 'lane[ _-]?b'; then
  pass "table rows for both lanes"
else
  fail "table rows for both lanes" "no '|' row mentions lane A and none/one mentions lane B"
fi

# 5 -------------------------------------------------------------- oracles (WARN)
if grep -qiE 'oracle' "$REPORT"; then
  pass "oracles reported" "$(grep -ciE 'oracle' "$REPORT") mention(s)"
else
  warn "oracles reported" "A-oracle / B-oracle calibrate both lanes — report them"
fi

# 6 -------------------------------------------------------------- aggregation script (WARN)
if [ -f "$REPO_ROOT/harness/report/aggregate.py" ]; then
  pass "harness/report/aggregate.py"
else
  warn "harness/report/aggregate.py" "report written by hand? the aggregation belongs in the repo"
fi

# 7 -------------------------------------------------------------- source run folders (WARN)
n=$(find "$RUNS" -maxdepth 5 -type f -path '*/out/eval/eval_results.csv' 2>/dev/null | wc -l)
if [ "$n" -ge 4 ]; then
  pass "eval run folders on disk" "$n eval_results.csv (2 policies + 2 oracles expected)"
else
  warn "eval run folders on disk" "$n eval_results.csv found — expected at least 4"
fi

echo "----------------------------------------------------------------------"
if [ "$FAILED" -eq 0 ]; then
  echo "GATE P4: PASS ($WARNED warning(s))"
  exit 0
fi
echo "GATE P4: FAIL ($FAILED failing check(s), $WARNED warning(s))"
exit 1
