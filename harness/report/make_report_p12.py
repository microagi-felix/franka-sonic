#!/usr/bin/env python3
"""P12 WP 12.5 — regenerate plan/REPORT.md with round 3b appended.

Repo copy of `/tmp/franka-sonic/p12/make_report.py` with P11's flag list embedded:
the working copy read it out of `/tmp/franka-sonic/p11/make_report.sh`, which does not
survive a pod restart, and a report nobody can regenerate is not a reproducible report
(harness debt (e) in its reporting form).

Every P11 input is kept verbatim from `/tmp/franka-sonic/p11/make_report.sh` (so
sections 1-13 stay the report they already are) and every round-3b input is
resolved from a RECORD, never from recency:

  rows      the `P12 ROW <label>=<abs run folder>` lines of plan/STATUS.md
  screens   the watcher's own state file, whose `run` field came from each
            launcher's `[bakeoff] OK rc=0  log: <run>/logs/run.log` line
  extension the `P12 INIT lane_a=` checkpoint's own fine-tune run folder
  probe     the path in the `P12 PROBE=` line

  usage: python3 make_report.py [--out PATH] [--dry-run]

Harness debt (c) is the reason this file exists: two runs of one lane and stage
overlap in P12 by design, so "newest wins" would mis-file half of them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

REPO = Path("/home/felixminzenmay/code/franka-sonic")
STATUS = REPO / "plan/STATUS.md"
WATCHER_STATE = Path("/tmp/franka-sonic/p12/watcher_state.json")


P11_FLAGS = [
    '--held-out-from',
    '20',
    '--r3-screens-until',
    '2026-09-05T21:00:00+00:00',
    '--r3-rescreen',
    'lane B ck17500 — orchestrator control, unseeded=/tmp/franka-sonic/lane_b/2026-09-05_eval-28',
    '--r3-rescreen',
    'lane B ck17500 — orchestrator control, seed 12345=/tmp/franka-sonic/lane_b/2026-09-05_eval-29',
    '--r3-rescreen',
    'lane B ck17500 — orchestrator control, unseeded=/tmp/franka-sonic/lane_b/2026-09-05_eval-30',
    '--r3-rescreen',
    'lane B ck17500 — orchestrator control, unseeded=/tmp/franka-sonic/lane_b/2026-09-05_eval-31',
    '--r3-rescreen',
    'lane B ck17500 — watcher launch path (23:09)=/tmp/franka-sonic/lane_b/2026-09-05_eval-32',
    '--r3-rescreen',
    'lane B ck20000 — watcher launch path (23:09)=/tmp/franka-sonic/lane_b/2026-09-05_eval-33',
    '--row',
    'lane_a_eval=/home/felixminzenmay/runs/franka-sonic/lane_a/2026-09-05_eval-10',
    '--row',
    'lane_b_eval=/home/felixminzenmay/runs/franka-sonic/lane_b/2026-09-05_eval-7',
    '--oracle-a',
    '/home/felixminzenmay/runs/franka-sonic/lane_a/2026-09-05_oracle_a',
    '--oracle-b',
    '/home/felixminzenmay/runs/franka-sonic/lane_b/2026-09-05_oracle_b',
    '--r3-row',
    'lane_a=/tmp/franka-sonic/lane_a/2026-09-05_eval-9',
    '--r3-row',
    'lane_a=/tmp/franka-sonic/lane_a/2026-09-05_eval-13',
    '--r3-row',
    'lane_a=/tmp/franka-sonic/lane_a/2026-09-05_eval-14',
    '--r3-row',
    'lane_b=/tmp/franka-sonic/lane_b/2026-09-05_eval-6',
    '--r3-row',
    'lane_b=/tmp/franka-sonic/lane_b/2026-09-05_eval-24',
    '--r3-row',
    'lane_b=/tmp/franka-sonic/lane_b/2026-09-05_eval-4',
    '--r3-mixture',
    '/tmp/franka-sonic/lane_b/2026-09-05_eval-24',
    '--r3-segment',
    'eval-24 regime 1, episodes 0-81 (stalling)=/tmp/franka-sonic/lane_b/2026-09-05_eval-24:0-81',
    '--r3-segment',
    'eval-24 regime 2, episodes 82-187 (completing; flip at 22:59 UTC)=/tmp/franka-sonic/lane_b/2026-09-05_eval-24:82-187',
    '--r3-segment',
    'eval-24 regime 3, episodes 188-199 (stalling; flip at 00:22 UTC)=/tmp/franka-sonic/lane_b/2026-09-05_eval-24:188-199',
    '--r3-segment',
    'eval-32 re-screen, episodes 0-11 (completing)=/tmp/franka-sonic/lane_b/2026-09-05_eval-32:0-11',
    '--r3-segment',
    'eval-32 re-screen, episodes 12-19 (stalling; flip at 23:21 UTC)=/tmp/franka-sonic/lane_b/2026-09-05_eval-32:12-19',
    '--r3-oracle-b',
    '/tmp/franka-sonic/lane_b/2026-09-05_oracle_b',
    '--r3-stopped',
    'lane A round 3 `checkpoint-17500` — stopped at episode 26=/tmp/franka-sonic/lane_a/2026-09-05_eval-11',
    '--r3-stopped',
    'lane A round 3 `checkpoint-15000` — stopped at episode 23=/tmp/franka-sonic/lane_a/2026-09-05_eval-12',
    '--r3-stopped',
    'lane B round 3 `checkpoint-20000` — stopped at episode 22=/tmp/franka-sonic/lane_b/2026-09-05_eval-26',
    '--r3-stopped',
    'lane B round 3 `checkpoint-10000` — stopped at episode 30=/tmp/franka-sonic/lane_b/2026-09-05_eval-27',
    '--r3-stopped',
    'lane B round 3 `checkpoint-12500` — still running at report time=/tmp/franka-sonic/lane_b/2026-09-06_eval',
    '--r2-rerun',
    'lane_a=/tmp/franka-sonic/lane_a/2026-09-05_eval-10',
    '--r2-rerun',
    'lane_b=/tmp/franka-sonic/lane_b/2026-09-05_eval-25',
    '--artefact-run',
    'with the flag, run 1=/tmp/franka-sonic/lane_b/2026-09-05_eval-17',
    '--artefact-run',
    'with the flag, run 2=/tmp/franka-sonic/lane_b/2026-09-05_eval-18',
    '--artefact-run',
    'with the flag, run 3=/tmp/franka-sonic/lane_b/2026-09-05_eval-19',
    '--artefact-run',
    'without it, run 1=/tmp/franka-sonic/lane_b/2026-09-05_eval-20',
    '--artefact-run',
    'without it, run 2=/tmp/franka-sonic/lane_b/2026-09-05_eval-21',
    '--artefact-run',
    'without it, run 3=/tmp/franka-sonic/lane_b/2026-09-05_eval-22',
]


def p11_flags(out: Path) -> list[str]:
    """P11's own aggregate.py invocation, verbatim, plus --out."""
    return list(P11_FLAGS) + ["--out", str(out)]


def status_lines(prefix: str) -> dict[str, str]:
    """Every `<prefix> <label>=<path>` line of STATUS.md, last write winning."""
    found: dict[str, str] = {}
    for line in STATUS.read_text(errors="replace").splitlines():
        m = re.search(rf"{prefix} (\S+)=(\S+)", line)
        if m:
            found[m.group(1)] = m.group(2)
    return found


def status_last(prefix: str) -> str | None:
    hits = re.findall(rf"{prefix}=(\S+)", STATUS.read_text(errors="replace"))
    return hits[-1] if hits else None


def r3b_flags() -> tuple[list[str], list[str]]:
    """(flags, notes-for-the-log). Anything absent is simply not passed."""
    flags: list[str] = []
    log: list[str] = []

    rows = status_lines("P12 ROW")
    for label in sorted(rows):
        run = Path(rows[label])
        csv = run / "out/eval/eval_results.csv"
        if not csv.is_file():
            log.append(f"SKIP row {label}: {csv} missing")
            continue
        n = sum(1 for _ in csv.open()) - 1
        flags += ["--r3b-row", f"{label}={run}"]
        log.append(f"row  {label:16} {n:3d} episodes  {run}")

    if WATCHER_STATE.is_file():
        st = json.loads(WATCHER_STATE.read_text())
        for lane, entries in sorted(st.items()):
            for key, e in sorted(entries.items(), key=lambda kv: (kv[1]["step"], kv[1]["screen"])):
                run = e.get("run")
                if not run or e.get("status") != "done":
                    log.append(f"SKIP screen {lane} {key}: status={e.get('status')} run={run}")
                    continue
                # no backticks in the label: the table quotes it itself
                label = f"{'lane A' if lane == 'lane_a' else 'lane B'} extension "
                label += f"checkpoint-{e['step']} screen {e['screen']}"
                flags += ["--r3b-screen", f"{label}={run}"]
                log.append(f"scr  step {e['step']:>5} #{e['screen']}  {run}")

    init = status_last("P12 INIT lane_a")
    if init:
        ck = Path(init)
        # <run folder>/out/checkpoints/checkpoint-N -> <run folder> is ROUND 3's;
        # the EXTENSION is the fine-tune that *started* from it.
        flags += ["--r3b-r3-finetune", str(ck.parents[2])]
        log.append(f"r3 finetune      {ck.parents[2]}")
    ext = sorted(Path("/tmp/franka-sonic/lane_a").glob("*_finetune*"))
    ext = [d for d in ext if (d / "cmd.sh").is_file()
           and str(init or "@none@") in (d / "cmd.sh").read_text(errors="replace")]
    if len(ext) == 1:
        flags += ["--r3b-ext-finetune", str(ext[0])]
        log.append(f"extension        {ext[0]}  (the only fine-tune whose cmd.sh inits from {init})")
    else:
        log.append(f"SKIP extension: {len(ext)} candidate folder(s) init from {init}: {ext}")

    probe = status_last("P12 PROBE")
    if probe and Path(probe).is_file():
        flags += ["--r3b-probe", probe]
        log.append(f"probe            {probe}")
    else:
        log.append(f"SKIP probe: {probe!r} is not a file")
    return flags, log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "plan/REPORT.md")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--note", action="append", default=[])
    a = ap.parse_args()

    flags = p11_flags(a.out)
    extra, log = r3b_flags()
    for n in a.note:
        extra += ["--r3b-note", n]
    print("\n".join(log), file=sys.stderr)
    cmd = [sys.executable, "harness/report/aggregate.py"] + flags + extra
    print("\n[make_report] " + " ".join(shlex.quote(c) for c in cmd) + "\n", file=sys.stderr)
    if a.dry_run:
        return 0
    return subprocess.call(cmd, cwd=REPO)


if __name__ == "__main__":
    raise SystemExit(main())
