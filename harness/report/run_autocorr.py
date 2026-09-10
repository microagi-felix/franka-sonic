#!/usr/bin/env python3
"""Are the episodes of one evaluation run exchangeable? A permutation runs-test per run.

Every 95 % interval in plan/REPORT.md is a binomial interval on the live episodes of a run,
which assumes those episodes are independent draws from one rate. P12 showed that assumption
is false for the DEAD marks — dead starts come in runs far longer than independence allows
(`/tmp/franka-sonic/p12/dead_clustering.py`: p = 0.0001 with a longest run of 19). This script
asks the same question of the marks the intervals are actually computed on: the live outcomes,
success against live failure, with dead episodes dropped exactly as the reporting rule drops
them.

Method: the Wald-Wolfowitz runs test done by permutation, so there is no normal approximation
and no assumption about the rate. The statistic is the number of maximal runs of one kind; the
null shuffles the run's own marks, which fixes the number of successes and asks only whether
their ORDER is exchangeable. Fewer runs than the null = clustering = positive autocorrelation =
the printed interval is too narrow. More runs = alternation.

Two-sided p, reported as min(p_fewer, p_more) * 2 clipped at 1.

    python3 run_autocorr.py [--iters 20000] LABEL=RUN_FOLDER [...]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys


def marks(run: Path, held_out_from: int = 20) -> str:
    """'1' success, '0' live failure, over the held-out slice, dead episodes dropped."""
    rows = []
    with (run / "out/eval/eval_results.csv").open() as fh:
        for r in csv.DictReader(fh):
            ep = int(float(r["episode"]))
            if ep < held_out_from:
                continue
            ok = str(r["success"]).strip().lower() in ("true", "1")
            prog = float(r["progress"] or 0.0)
            if not ok and prog == 0.0:
                continue  # dead: the artefact, excluded from every rate in the report
            rows.append((ep, "1" if ok else "0"))
    return "".join(m for _e, m in sorted(rows))


def n_runs(s: str) -> int:
    return 1 + sum(1 for a, b in zip(s, s[1:]) if a != b) if s else 0


def longest(s: str, ch: str) -> int:
    best = cur = 0
    for c in s:
        cur = cur + 1 if c == ch else 0
        best = max(best, cur)
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", metavar="LABEL=RUN")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--held-out-from", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260910)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)

    print(
        f"{'row':<18} {'live':>5} {'succ':>5} {'runs':>5} {'longest 0':>10} "
        f"{'E[runs]':>8} {'p(two-sided)':>13}  verdict"
    )
    for pair in a.pairs:
        if "=" not in pair:
            print(f"expected LABEL=RUN, got {pair!r}", file=sys.stderr)
            return 2
        label, run = pair.split("=", 1)
        s = marks(Path(run).expanduser(), a.held_out_from)
        if len(s) < 10 or "0" not in s or "1" not in s:
            print(f"{label:<18} {len(s):>5}  too few / one-sided to test")
            continue
        obs = n_runs(s)
        chars = list(s)
        perm = []
        for _ in range(a.iters):
            rng.shuffle(chars)
            perm.append(n_runs("".join(chars)))
        exp = sum(perm) / len(perm)
        fewer = sum(1 for r in perm if r <= obs) / len(perm)
        more = sum(1 for r in perm if r >= obs) / len(perm)
        p = min(1.0, 2 * min(fewer, more))
        if p >= 0.05:
            verdict = "exchangeable (no evidence against)"
        elif obs < exp:
            verdict = "CLUSTERED — printed interval too narrow"
        else:
            verdict = "ALTERNATING — printed interval conservative"
        print(
            f"{label:<18} {len(s):>5} {s.count('1'):>5} {obs:>5} {longest(s, '0'):>10} "
            f"{exp:>8.1f} {p:>13.4f}  {verdict}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
