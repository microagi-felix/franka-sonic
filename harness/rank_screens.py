#!/usr/bin/env python3
"""Apply the campaign's pre-registered checkpoint-selection rule to a screen series.

    (successes summed over ALL screens of the checkpoint, milestone-6 rate, milestone-5 rate,
     step)   — descending; never mean progress.

Reads `harness/screen_watcher.py`'s state file, so every screen it counts was attributed to its
run folder by that screen's own launcher log — not by recency (harness debt (c)). Read-only: it
prints the series and the `<PREFIX> BEST <lane>=<path>` line for the phase agent to paste into
plan/STATUS.md, and changes nothing.

    python3 harness/rank_screens.py --state-dir /tmp/franka-sonic/p12 \\
        --checkpoints /tmp/franka-sonic/lane_a/2026-09-10_finetune/out/checkpoints \\
        --lane lane_a --prefix "P12"

Two screens per checkpoint is the P11 lesson (a single 20-rollout screen measures the regime a
run happened to be in, not the checkpoint), so a checkpoint with fewer than
`--screens-per-checkpoint` finished screens is ranked on what it has AND named in a NOTE: the
ranking key's first term is a SUM, so an under-screened checkpoint is ranked low by construction
and can only lose. Never quote this ranking as a measurement — a screen selects, it does not
measure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-dir", required=True, type=Path)
    ap.add_argument("--checkpoints", required=True, type=Path,
                    help="the fine-tune's checkpoints directory (the BEST line names a dir in it)")
    ap.add_argument("--lane", default="lane_a")
    ap.add_argument("--prefix", default="P12", help="the STATUS line prefix, e.g. P12")
    ap.add_argument("--screens-per-checkpoint", type=int, default=2)
    a = ap.parse_args(argv)

    state = json.loads((a.state_dir / "watcher_state.json").read_text())
    by_step: dict[int, list[dict]] = {}
    for _key, ent in sorted(state.get(a.lane, {}).items(),
                            key=lambda kv: (int(kv[0].split(":")[0]), kv[0])):
        if ent.get("status") != "done" or not ent.get("result"):
            print(f"  (skipping {ent.get('tag')}: status={ent.get('status')})")
            continue
        by_step.setdefault(int(ent["step"]), []).append(ent)

    rows = []
    for step, ents in sorted(by_step.items()):
        m = [e["result"]["milestone_rates"] for e in ents]
        rows.append({
            "step": step,
            "screens": len(ents),
            "succ": sum(e["result"]["n_success"] for e in ents),
            "n": sum(e["result"]["n"] for e in ents),
            "m6": sum(x[5] for x in m) / len(m),
            "m5": sum(x[4] for x in m) / len(m),
            "each": "+".join(f"{e['result']['n_success']}/{e['result']['n']}" for e in ents),
            "runs": [e["run"] for e in ents],
        })
    if not rows:
        print("no completed screens yet")
        return 1
    rows.sort(key=lambda r: (r["succ"], r["m6"], r["m5"], r["step"]), reverse=True)

    print(f"\n{'step':>7} {'screens':>7} {'successes':>10} {'each':>14} {'m6':>6} {'m5':>6}")
    for r in rows:
        print(f"{r['step']:>7} {r['screens']:>7} {r['succ']:>4}/{r['n']:<5} {r['each']:>14} "
              f"{r['m6']:>6.2f} {r['m5']:>6.2f}")

    short = [r for r in rows if r["screens"] < a.screens_per_checkpoint]
    if short:
        print(f"\nNOTE: {len(short)} checkpoint(s) have fewer than {a.screens_per_checkpoint} "
              "finished screens: " + ", ".join(str(r["step"]) for r in short))

    print("\nseries (by step):")
    for r in sorted(rows, key=lambda r: r["step"]):
        print(f"  {r['step']:>5}: {r['each']} = {r['succ']}/{r['n']}")

    best = rows[0]
    print(f"\n{a.prefix} BEST {a.lane}={a.checkpoints / ('checkpoint-%d' % best['step'])}")
    if len(rows) > 1:
        print(f"(runner-up: checkpoint-{rows[1]['step']}, {rows[1]['each']})")
    for r in rows:
        for run in r["runs"]:
            print(f"  screen run  step {r['step']:>5}  {run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
