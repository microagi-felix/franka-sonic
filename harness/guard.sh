#!/usr/bin/env bash
# PreToolUse hook for the Bash tool (wired in .claude/settings.json).
#
# Claude Code sends the hook payload as JSON on stdin. Exit 0 = allow, exit 2 =
# block and show stderr to the model. No jq on this pod, so the JSON is parsed
# by python3 from stdin; the heredoc goes through `python3 -c "$(cat <<'PY' …)"`
# precisely so that stdin stays the payload rather than the script.
#
# These are the rules from AGENTS.md that a bad reflex can trip in one keystroke.
# It is a backstop, not the policy: read AGENTS.md.
python3 -c "$(cat <<'PY'
import json, re, sys

try:
    payload = json.load(sys.stdin)
except Exception:
    sys.exit(0)  # not a payload we understand: never block on our own bug

cmd = ((payload.get("tool_input") or {}).get("command")) or ""
if not isinstance(cmd, str) or not cmd.strip():
    sys.exit(0)

SHARED = r"/data/lustre/shared"
WRITERS = r"(?:tee|cp|mv|rm|mkdir|touch|rsync|dd|install|chmod|chown|ln)"

RULES = [
    # (regex, reason)
    (r"\bpkill\b\s+(?:[^|;&]*\s)?-\w*f",
     "pkill -f kills by pattern across the whole node, which is shared. "
     "Kill by PID from your own run folder (AGENTS.md rule b)."),
    (r"\bkillall\b",
     "killall kills by name across the whole node, which is shared. "
     "Kill by PID from your own run folder (AGENTS.md rule b)."),
    (r"\bkubectl\b[^|;&]*\bdelete\b",
     "kubectl delete is out of scope for this campaign — STOP and write "
     "BLOCKED in plan/STATUS.md (AGENTS.md rule j)."),
    (r"\bdocker\b(?!\s+--version\b)",
     "this pod is unprivileged: docker is a shim that cannot build or run "
     "anything. Use the Kit python user-sites instead (AGENTS.md rule d)."),
    (r">>?\s*" + SHARED,
     "/data/lustre/shared is shared, multi-user, READ-ONLY storage "
     "(AGENTS.md rule c)."),
    (r"\b" + WRITERS + r"\b[^|;&]*" + SHARED,
     "/data/lustre/shared is shared, multi-user, READ-ONLY storage "
     "(AGENTS.md rule c)."),
    (r"\brm\b\s+(?:-\w+\s+)*-\w*[rR]\w*f\w*\s+(?:~|\\\$HOME|/home/\w+)\s*/?\s*(?:$|[;&|])",
     "rm -rf of the home tree would destroy the Lustre subtree that survives "
     "pod restarts — everything this campaign has."),
    (r"\brm\b\s+(?:-\w+\s+)*-\w*[rR]\w*f\w*\s+/\s*(?:$|[;&|*])",
     "rm -rf / — no."),
    (r"\bsudo\b[^|;&]*\brm\b",
     "sudo rm: nothing this campaign owns needs root to delete "
     "(AGENTS.md rule j)."),
    (r"\bchown\b[^|;&]*\s-\w*R",
     "recursive chown can rewrite ownership of another user's files "
     "(AGENTS.md rule j)."),
    (r"\bnvidia-smi\b[^|;&]*\s-r\b",
     "nvidia-smi -r resets a GPU other tenants are using."),
]

for pattern, reason in RULES:
    if re.search(pattern, cmd):
        print(f"BLOCKED by harness/guard.sh: {reason}", file=sys.stderr)
        print(f"  command: {cmd[:400]}", file=sys.stderr)
        sys.exit(2)

sys.exit(0)
PY
)"
exit $?
