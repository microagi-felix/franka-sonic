#!/usr/bin/env bash
# GATE P0 — environments. Prints PASS / FAIL / WARN per check and exits
# non-zero if any check FAILed. WARN never fails the gate but is reported.
#
#   bash harness/gates/p0.sh
#
# Checks are deliberately the cheap, factual ones: does the interpreter exist,
# does the import resolve, is there a run folder with a results CSV. A gate
# reads run folders and exit codes, never prose (AGENTS.md rule f).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KIT_PY=/isaac-sim/kit/python/bin/python3
PYSH=/isaac-sim/python.sh
USERBASE_FR3="$HOME/env/pyuser-fr3"
USERBASE_SONIC="$HOME/env/pyuser-sonic"
GR00T_PY="$HOME/Isaac-GR00T/.venv/bin/python"
FR3_REPO="$HOME/code/franka-bimanual-isaac-sim"
HF_HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"

FAILED=0
WARNED=0

pass() { printf 'PASS  %-34s %s\n' "$1" "${2:-}"; }
fail() { printf 'FAIL  %-34s %s\n' "$1" "${2:-}"; FAILED=$((FAILED + 1)); }
warn() { printf 'WARN  %-34s %s\n' "$1" "${2:-}"; WARNED=$((WARNED + 1)); }

echo "GATE P0 — $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname)"
echo "----------------------------------------------------------------------"

# 1 -------------------------------------------------------------- group
if id -nG | grep -qw isaac-sim; then
  pass "isaac-sim group" "$(id -nG)"
else
  fail "isaac-sim group" "run env/bootstrap.sh, then open a NEW login shell"
fi

# 2 -------------------------------------------------------------- python.sh
if [ -x "$PYSH" ]; then
  pass "/isaac-sim/python.sh executable"
else
  fail "/isaac-sim/python.sh executable" "not visible — group not active in this shell?"
fi

# 3 -------------------------------------------------------------- sim imports
# From $HOME, not the repo: from the repo `import evaluation` resolves via cwd
# and a broken editable install would pass.
if out=$(cd "$HOME" && PYTHONUSERBASE="$USERBASE_FR3" "$PYSH" -c \
      "import isaacsim, isaaclab; import evaluation, tasks; print('sim ok')" 2>&1); then
  pass "sim stack imports" "$(echo "$out" | tail -1)"
else
  fail "sim stack imports" "$(echo "$out" | tail -2 | tr '\n' ' ')"
fi

# 4 -------------------------------------------------------------- eval CLI
if out=$(cd "$FR3_REPO" && PYTHONUSERBASE="$USERBASE_FR3" "$PYSH" -m evaluation.eval --help 2>&1); then
  pass "evaluation.eval --help" "$(echo "$out" | grep -c -- '--' ) flags"
else
  fail "evaluation.eval --help" "$(echo "$out" | tail -2 | tr '\n' ' ')"
fi

# 5 -------------------------------------------------------------- smoke run folder
csv=$(ls -1t "$HOME"/runs/franka-sonic/shared/*p0.smoke*/out/eval/eval_results.csv 2>/dev/null | head -1)
if [ -n "$csv" ]; then
  pass "p0.smoke eval_results.csv" "$csv ($(wc -l < "$csv") lines)"
else
  fail "p0.smoke eval_results.csv" "run: python3 harness/bakeoff.py run shared p0.smoke"
fi

# 6 -------------------------------------------------------------- gr00t
if out=$("$GR00T_PY" -c "import gr00t; print('gr00t ok')" 2>&1); then
  pass "gr00t import" "$(echo "$out" | tail -1)"
else
  fail "gr00t import" "$(echo "$out" | tail -2 | tr '\n' ' ')"
fi

# 7 -------------------------------------------------------------- N1.7-3B weights
snap="$HF_HUB/models--nvidia--GR00T-N1.7-3B/snapshots"
if [ -d "$snap" ] && [ -n "$(ls -A "$snap" 2>/dev/null)" ]; then
  pass "GR00T-N1.7-3B in HF cache" "$(du -sh "$HF_HUB/models--nvidia--GR00T-N1.7-3B" 2>/dev/null | cut -f1)"
else
  fail "GR00T-N1.7-3B in HF cache" "expected $snap"
fi

# 8 -------------------------------------------------------------- gear_sonic (WARN only)
# Lane B needs it, but P0 is allowed to proceed without it: the SONIC install
# is P2's problem and blocking P0 on it would stall lane A for nothing.
if out=$(cd "$HOME" && PYTHONUSERBASE="$USERBASE_SONIC" "$PYSH" -c \
      "import gear_sonic; print('gear_sonic ok')" 2>&1); then
  pass "gear_sonic import" "$(echo "$out" | tail -1)"
else
  warn "gear_sonic import" "$(echo "$out" | tail -1) — needed for lane B (P2), not for P0"
fi

# 9 -------------------------------------------------------------- tmux
if command -v tmux >/dev/null 2>&1; then
  pass "tmux" "$(tmux -V)"
else
  fail "tmux" "sudo apt-get install -y tmux (env/bootstrap.sh step 2)"
fi

# 10 ------------------------------------------------------------- allocator
if out=$("$GR00T_PY" "$REPO_ROOT/harness/gpus.py" probe 2>&1); then
  eligible=$(echo "$out" | awk '$4=="yes"{n++} END{print n+0}')
  pass "harness/gpus.py probe" "$eligible eligible device(s)"
  echo "$out" | sed 's/^/      /'
else
  fail "harness/gpus.py probe" "$(echo "$out" | tail -2 | tr '\n' ' ')"
fi

echo "----------------------------------------------------------------------"
if [ "$FAILED" -eq 0 ]; then
  echo "GATE P0: PASS ($WARNED warning(s))"
  exit 0
fi
echo "GATE P0: FAIL ($FAILED failing check(s), $WARNED warning(s))"
exit 1
