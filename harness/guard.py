#!/usr/bin/env python3
"""PreToolUse guard for the Bash tool — the executable half of harness/guard.sh.

Claude Code sends the hook payload as JSON on stdin. Exit 0 = allow, exit 2 =
block and show stderr to the model. This lives in its own file (rather than a
heredoc inside guard.sh) because quoting a Python program inside
`python3 -c "$(cat <<'PY' … )"` breaks on apostrophes and escaped quotes — and a
guard that fails to parse exits 2 and blocks *every* command.

These are the AGENTS.md rules a bad reflex can trip in one keystroke. It is a
backstop, not the policy. The hook is evaluated per tool call, so edits here
take effect on already-running sessions.

False positives are expected and acceptable — rewrite the command (e.g. write a
script into the run folder and run the file) rather than fighting the hook.
"""

import json
import re
import sys

USER = "felixminzenmay"
SHARED = r"/data/lustre/shared"
WRITERS = r"(?:tee|cp|mv|mkdir|touch|rsync|dd|install|chmod|chown|ln)"

# Paths nothing in this campaign may write to. Other people's homes included;
# our own /home/felixminzenmay is deliberately NOT protected.
PROTECTED = (
    r"(?<![\w~.:/-])"
    r"(?:/data|/opt|/usr|/etc|/root|/srv|/mnt|/isaac-sim"
    r"|/home/(?!" + USER + r"(?:/|\b))[A-Za-z0-9._-]+)"
)
PROTECTED_TOKEN = re.compile(PROTECTED + r"[^\s;|&\x27\x22$]*")

# The one authorised exception (AGENTS.md rule j): Kit's portable-mode dirs.
ALLOWED_PREFIXES = (
    "/isaac-sim/kit",        # cache, logs, data — widening is authorised
    "/isaac-sim/extscache",
)

RULE_O = (
    "AGENTS.md rule o: nothing on this pod is ever deleted, and nothing "
    "outside our own roots is ever written. Stale artifacts stay in place "
    "with a NEEDS-CLEANUP line in plan/STATUS.md."
)

# Felix, 2026-09-03: "tell the agent also not to delete anything and be
# careful! there are shared folders which should not be touched."
DELETE_RULES = [
    (r"\brm\b", "rm"),
    (r"\brmdir\b", "rmdir"),
    (r"\bunlink\b", "unlink"),
    (r"\bshred\b", "shred"),
    (r"\btruncate\b", "truncate"),
    (r"\bgit\s+clean\b", "git clean"),
    (r"\bfind\b[^|;&]*\s-delete\b", "find -delete"),
    (r"\brsync\b[^|;&]*--delete", "rsync --delete"),
    (r"shutil\.rmtree", "shutil.rmtree"),
    (r"\bos\.(?:remove|unlink|rmdir|removedirs)\s*\(", "os.remove/unlink/rmdir"),
    (r"\.unlink\s*\(", "Path.unlink"),
    (r"\brmtree\s*\(", "rmtree("),
]

# The older, sharp edges.
RULES = [
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
    (r"\bsudo\b[^|;&]*\brm\b",
     "sudo rm: nothing this campaign owns needs root to delete "
     "(AGENTS.md rules j and o)."),
    (r"\bnvidia-smi\b[^|;&]*\s-r\b",
     "nvidia-smi -r resets a GPU other tenants are using."),
]

# For cp/mv/rsync/install/ln only the LAST argument is a target — reading FROM
# /data is fine and P2 needs it. For the others every argument is a target.
TARGET_LAST = {"cp", "mv", "rsync", "install", "ln"}
TARGET_ALL = {"tee", "mkdir", "touch", "dd", "chmod", "chown"}


def unallowed(token: str) -> bool:
    return not token.startswith(ALLOWED_PREFIXES)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0  # not a payload we understand: never block on our own bug

    cmd = ((payload.get("tool_input") or {}).get("command")) or ""
    if not isinstance(cmd, str) or not cmd.strip():
        return 0

    def block(reason: str) -> int:
        print("BLOCKED by harness/guard.sh: " + reason, file=sys.stderr)
        print("  command: " + cmd[:400], file=sys.stderr)
        return 2

    # ---------------------------------------------------------- deletions
    for pattern, what in DELETE_RULES:
        if re.search(pattern, cmd):
            return block("`" + what + "` deletes data. " + RULE_O)

    # ------------------------------------------- writes outside our roots
    m = re.search(r">>?\s*(" + PROTECTED + r"[^\s;|&\x27\x22]*)", cmd)
    if m and unallowed(m.group(1)):
        return block(
            "redirecting output into " + m.group(1)
            + " writes outside our roots. " + RULE_O
        )

    if re.search(r"\bsed\b[^|;&]*\s-i\b", cmd):
        bad = [t for t in PROTECTED_TOKEN.findall(cmd) if unallowed(t)]
        if bad:
            return block("`sed -i` would edit " + bad[0] + " in place. " + RULE_O)

    for segment in re.split(r"[;|&\n]+", cmd):
        words = segment.split()
        verb = None
        idx = -1
        for i, word in enumerate(words):
            base = word.rsplit("/", 1)[-1]
            if base in TARGET_LAST or base in TARGET_ALL:
                verb, idx = base, i
                break
        if verb is None:
            continue
        args = words[idx + 1:]
        targets = ([args[-1]] if args else []) if verb in TARGET_LAST else args
        for arg in targets:
            for token in PROTECTED_TOKEN.findall(arg):
                if unallowed(token):
                    return block(
                        "`" + verb + "` would write to " + token
                        + ", which is shared or system storage (/data in full, "
                        "/isaac-sim except the authorised Kit cache, logs, data "
                        "and extscache, /opt /usr /etc /root /srv /mnt, and "
                        "other people's homes). " + RULE_O
                    )

    # ------------------------------------------------------ sharp edges
    for pattern, reason in RULES:
        if re.search(pattern, cmd):
            return block(reason)

    return 0


if __name__ == "__main__":
    sys.exit(main())
