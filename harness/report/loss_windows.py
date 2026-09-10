#!/usr/bin/env python3
"""Training loss as the MEAN over each 2500-step window, for two fine-tune runs side by side.

WP 12.2 of P12 asks for "the training-loss curve of the extension beside round 3's
(`trainer_state.json` of both runs; mean over each 2500-step window)". The report's
`aggregate.py:loss_series` samples a single logged value at each 2500-step boundary,
which is one `logging_steps` average and therefore noisy; a window mean over all ~250
logs in the window is the reading the prompt asks for and the one that can answer
"was the loss still falling when the cosine hit zero".

Reads the newest `trainer_state.json` under a run's `out/checkpoints/*` (HuggingFace's
Trainer writes the whole `log_history` into every checkpoint, so the last checkpoint
carries the complete curve). Read-only; interprets nothing beyond the arithmetic.

  usage: loss_windows.py [--every 2500] [--markdown] LABEL=RUN_FOLDER [LABEL=RUN_FOLDER ...]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys


def last_trainer_state(run: Path) -> Path | None:
    """The checkpoint with the largest step under run/out/checkpoints — not the newest
    by mtime (harness debt (c): recency is never the way to resolve a run's artifacts)."""
    ckpts = []
    for d in (run / "out" / "checkpoints").glob("checkpoint-*"):
        try:
            ckpts.append((int(d.name.split("-")[-1]), d))
        except ValueError:
            continue
    for _step, d in sorted(ckpts, reverse=True):
        ts = d / "trainer_state.json"
        if ts.exists():
            return ts
    return None


def windows(ts: Path, every: int) -> list[dict]:
    hist = [
        e
        for e in json.loads(ts.read_text()).get("log_history", [])
        if isinstance(e, dict) and "loss" in e and "step" in e
    ]
    if not hist:
        return []
    out = []
    top = max(int(e["step"]) for e in hist)
    for lo in range(0, top, every):
        hi = lo + every
        vals = [float(e["loss"]) for e in hist if lo < int(e["step"]) <= hi]
        if not vals:
            continue
        out.append(
            {
                "window": f"{lo + 1}-{hi}",
                "hi": hi,
                "n": len(vals),
                "mean": statistics.fmean(vals),
                "min": min(vals),
                "max": max(vals),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", metavar="LABEL=RUN")
    ap.add_argument("--every", type=int, default=2500)
    ap.add_argument("--markdown", action="store_true", help="emit one paragraph for --r3b-note")
    a = ap.parse_args()

    series: list[tuple[str, Path, list[dict]]] = []
    for p in a.pairs:
        if "=" not in p:
            print(f"expected LABEL=RUN, got {p!r}", file=sys.stderr)
            return 2
        label, run = p.split("=", 1)
        rp = Path(run).expanduser()
        ts = last_trainer_state(rp)
        if ts is None:
            print(f"  {label:<14} {run}  no trainer_state.json under out/checkpoints yet")
            continue
        series.append((label, ts, windows(ts, a.every)))

    if not a.markdown:
        for label, ts, ws in series:
            print(f"{label}  ({ts})")
            for w in ws:
                print(
                    f"    steps {w['window']:>11}  mean {w['mean']:.4f}  "
                    f"(min {w['min']:.4f}, max {w['max']:.4f}, n={w['n']})"
                )
            if len(ws) >= 2:
                d = 100 * (1 - ws[-1]["mean"] / ws[0]["mean"]) if ws[0]["mean"] else 0.0
                rises = sum(1 for x, y in zip(ws, ws[1:]) if y["mean"] > x["mean"])
                print(
                    f"    first->last {ws[0]['mean']:.4f} -> {ws[-1]['mean']:.4f} "
                    f"({d:+.1f} % drop), {rises} of {len(ws) - 1} windows rose"
                )
            print()
        return 0

    # one paragraph, for --r3b-note
    bits = []
    for label, _ts, ws in series:
        if not ws:
            continue
        curve = ", ".join(f"{w['hi']}: {w['mean']:.4f}" for w in ws)
        tail = ""
        if len(ws) >= 2:
            d = 100 * (1 - ws[-1]["mean"] / ws[0]["mean"]) if ws[0]["mean"] else 0.0
            rises = sum(1 for x, y in zip(ws, ws[1:]) if y["mean"] > x["mean"])
            tail = f" ({d:+.1f} % from first window to last; {rises} of {len(ws) - 1} windows rose)"
        bits.append(f"**{label}** — {curve}{tail}")
    if not bits:
        return 0
    print(
        "**The loss curve as a window mean, not a point sample.** Each number is the mean of "
        f"every logged loss in a {a.every}-step window of that run's own `trainer_state.json` "
        "(the last checkpoint carries the whole `log_history`), which is what WP 12.2 asked for; "
        "the point-sampled row in the table above is one `logging_steps` average per boundary and "
        "is correspondingly noisier. "
        + "  ".join(bits)
        + "  The two runs are one recipe with two optimizer restarts, not one descent: the "
        "schedule and the optimizer state reset at the join, so the extension's first window is "
        "high by construction and a fall across the join is not evidence of a single curve."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
