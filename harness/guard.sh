#!/usr/bin/env bash
# PreToolUse hook for the Bash tool (wired in .claude/settings.json).
#
# Claude Code sends the hook payload as JSON on stdin; exit 0 = allow, exit 2 =
# block and show stderr to the model. All the logic lives in harness/guard.py —
# this wrapper only hands stdin through. (It used to inline the Python in a
# heredoc; quoting a program inside `python3 -c "$(cat <<PY ... )"` breaks on
# apostrophes and escaped quotes, and a guard that fails to parse exits 2 and
# blocks *every* command. Never inline it again.)
#
# The rules are the AGENTS.md ones a bad reflex can trip in one keystroke:
# no deletions of any kind, no writes outside our roots, no kill-by-pattern,
# no docker, no kubectl delete. Read AGENTS.md — this is a backstop, not the
# policy.
exec python3 "$(dirname "${BASH_SOURCE[0]}")/guard.py"
