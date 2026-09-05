#!/usr/bin/env python3
"""P4/P6 aggregation: the four eval run folders -> plan/REPORT.md.

    python3 harness/report/aggregate.py             # writes plan/REPORT.md
    python3 harness/report/aggregate.py --stdout    # prints the report instead

Stdlib only, so a plain `python3` runs it. For each of the four categories it
takes the NEWEST run folder (by the bakeoff finalisation stamp) under either
run root -- ~/runs/franka-sonic, or /tmp/franka-sonic when harness/bakeoff.py
routed the run to instance-local storage (AGENTS.md rule k):

    lane A policy   lane_a/<date>_eval[-N]        out/eval/eval_results.csv
    lane A oracle   lane_a/<date>_oracle_a[-N]    out/eval/eval_results.csv
    lane B policy   lane_b/<date>_eval[-N]        out/eval/eval_results.csv
    lane B oracle   lane_b/<date>_oracle_b[-N]    out/eval/eval_results.csv

and reads, per folder: eval_results.csv (success, length, progress per
episode), episode_<k>_result.json (rubric milestones: `criteria_reached` out
of `criteria_total`; the handover rubric's six milestones are ordered by
dependency, so milestone k is reached iff criteria_reached >= k), config.json
(bakeoff stamp: args, devices, repo SHAs, storage root) and out/eval/config.json
(evaluation.eval's own arguments: task, seed, rate, replan, horizon).

Every sentence that depends on a count (the headline verdict, the milestone
reading, the two oracle verdicts) is computed here from those csv rows, never
written into the template: the report has to stay honest when the numbers
change. The fine-tune blocks additionally read, per lane, the newest
<lane>/<date>_finetune folder's cmd.sh (parity) and its
out/checkpoints/checkpoint-2000/trainer_state.json (last logged training loss).

GPU-hours come from run-folder timestamps only: the earliest of the README's
"Launched" stamp and the first `=== <ISO> $ <cmd>` header in logs/run.log,
to the config.json `date` (stamped when bakeoff finalises the folder), times
the number of devices in `cuda_visible_devices`. Folders whose device list is
not a plain list of indices, or that carry no config.json, are listed as
"not recorded" and excluded from the sums. Nothing is estimated.

The prose lives in REPORT_template.md next to this file; this script fills
its {{PLACEHOLDERS}} and prints which run folders it used.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import math
import os
import re
import shlex
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN_ROOTS = (Path(os.path.expanduser("~/runs/franka-sonic")), Path("/tmp/franka-sonic"))
TEMPLATE = Path(__file__).with_name("REPORT_template.md")
REPORT = REPO / "plan" / "REPORT.md"

MILESTONES = [
    "left reaches",
    "left lifts",
    "placed at centre",
    "right reaches",
    "right lifts",
    "placed at end",
]

# key, table label, lane, stage substring, what the GPU-hours cell means
CATEGORIES = [
    ("lane_a_policy", "lane A — GR00T direct (joint targets)", "lane_a", "eval", "lane"),
    ("lane_a_oracle", "A-oracle — recorded joint targets replayed, no VLA", "lane_a", "oracle_a", "run"),
    ("lane_b_policy", "lane B — GR00T over SONIC (token -> decoder)", "lane_b", "eval", "lane"),
    ("lane_b_oracle", "B-oracle — encoder tokens -> decoder, no VLA", "lane_b", "oracle_b", "run"),
]

FOLDER_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(.+?)(?:-(\d+))?$")
HEADER_RE = re.compile(r"^=== (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\S*) \$ ")
NOT_RECORDED = "not recorded"
# A config.json stamp this close to the launch is the launch stamp: bakeoff
# writes it twice and its `finally` does not wait on a detached job.
FINALIZER_SLOP = 180.0

# gate P5's B-oracle criterion (harness/gates/p5.sh): >= 15 successes of 20
# rollouts. Scaled to the oracle run's own episode count so the verdict text
# survives a run with a different number of rollouts.
OB_GATE_K, OB_GATE_N = 15, 20


# --------------------------------------------------------------------------- small helpers
def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def parse_ts(text: str | None) -> dt.datetime | None:
    if not text:
        return None
    try:
        t = dt.datetime.fromisoformat(text.strip())
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t


def parse_devices(value) -> list[int] | None:
    """[] for a CPU stage (no devices stamped), a list of indices, or None when
    the stamp is not a plain device list (then GPU time is not recoverable)."""
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        value = ",".join(str(v) for v in value)
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if parts and all(p.isdigit() for p in parts):
        return [int(p) for p in parts]
    return None


def fmt_minutes(seconds: float | None) -> str:
    if seconds is None:
        return NOT_RECORDED
    return f"{seconds / 60:.1f} min"


def fmt_hours(hours: float | None) -> str:
    return NOT_RECORDED if hours is None else f"{hours:.2f}"


def fmt_ts(t: dt.datetime | None) -> str:
    return NOT_RECORDED if t is None else t.astimezone(dt.timezone.utc).strftime("%H:%M:%S")


def short_path(p: Path) -> str:
    home = Path(os.path.expanduser("~"))
    try:
        return "~/" + str(p.relative_to(home))
    except ValueError:
        return str(p)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided binomial interval, closed form for the two extremes we
    actually hit (k = 0 and k = n); a bisection on the binomial CDF otherwise."""
    if n <= 0:
        return (0.0, 1.0)
    if k == 0:
        return (0.0, 1.0 - (alpha / 2) ** (1.0 / n))
    if k == n:
        return ((alpha / 2) ** (1.0 / n), 1.0)

    def cdf(p: float, upto: int) -> float:
        from math import comb

        return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(upto + 1))

    def solve(target, upto, lo=0.0, hi=1.0):
        for _ in range(60):
            mid = (lo + hi) / 2
            if cdf(mid, upto) > target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    lower = solve(1 - alpha / 2, k - 1)
    upper = solve(alpha / 2, k)
    return (lower, upper)


def ci_text(r: dict) -> str:
    lo, hi = r["ci95"]
    return f"{100 * lo:.0f}–{100 * hi:.0f} %"


def overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def count_text(r: dict) -> str:
    return f"{r['n_success']}/{r['n']}"


def gate_needed(n: int) -> int:
    """The B-oracle gate count for an n-rollout run (15/20 of the rollouts)."""
    return math.ceil(OB_GATE_K / OB_GATE_N * n)


def ceiling_state(ob: dict) -> str:
    """"open" | "borderline" | "closed".

    Gate P5's B-oracle criterion is a 15-of-20 threshold, i.e. 75 %. At n = 20 a
    point estimate is all there is, but at larger n the criterion has to be
    applied to the interval or a run that misses it by two episodes gets called
    a failed ceiling on evidence that cannot tell 72 % from 76 %."""
    if ob["n_success"] >= gate_needed(ob["n"]):
        return "open"
    return "borderline" if ob["ci95"][1] >= OB_GATE_K / OB_GATE_N else "closed"


def gate_phrase(ob: dict) -> str:
    need = gate_needed(ob["n"])
    if ob["n"] == OB_GATE_N:
        return f"the {need}/{OB_GATE_N} threshold gate P5 requires"
    return (
        f"the {need}/{ob['n']} that gate P5's {OB_GATE_K}-of-{OB_GATE_N} criterion "
        "scales to at this rollout count"
    )


# --------------------------------------------------------------------------- run folders
def run_folders(lane: str) -> list[Path]:
    out = []
    for root in RUN_ROOTS:
        d = root / lane
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.is_dir() and FOLDER_RE.match(p.name):
                out.append(p)
    return out


def folder_stage(p: Path) -> str:
    m = FOLDER_RE.match(p.name)
    return m.group(2) if m else p.name


def artifact_end(d: Path) -> dt.datetime | None:
    """Newest mtime among a run folder's own log and output entries, at most two
    levels deep.

    `bakeoff.py` stamps config.json in a `finally` that does not wait on a
    detached trainer (the finalizer debt recorded in P9), so for round 2's
    fine-tunes the stamp lands seconds after launch while the run still has
    hours to go; and the two folders the storage switch created by hand have no
    stamp at all. The artifacts themselves say when writing stopped, which is
    a measurement rather than an estimate."""
    best = 0.0
    for pattern in ("logs/*", "out/*", "out/*/*"):
        for p in d.glob(pattern):
            try:
                best = max(best, p.lstat().st_mtime)
            except OSError:
                pass
    return dt.datetime.fromtimestamp(best, dt.timezone.utc) if best else None


def folder_devices(d: Path, cfg: dict) -> tuple[list[int] | None, str]:
    """The device list a run held, and where it was read from. Folders written
    by hand (round 2's storage switch) carry no bakeoff stamp; their cmd.sh
    acquires inside the script and echoes the claim into logs/run.log."""
    if cfg and "cuda_visible_devices" in cfg:
        return parse_devices(cfg.get("cuda_visible_devices")), "config.json"
    for probe in (d / "cmd.sh", d / "logs" / "run.log"):
        if not probe.is_file():
            continue
        with open(probe, errors="replace") as fh:
            head = fh.read(20000)
        m = re.search(r"CUDA_VISIBLE_DEVICES=([0-9]+(?:,[0-9]+)*)", head)
        if m:
            return parse_devices(m.group(1)), probe.name
    return None, "not recorded"


def folder_timing(d: Path) -> dict:
    cfg = read_json(d / "config.json") or {}
    launched = None
    readme = d / "README.md"
    if readme.exists():
        for line in readme.read_text(errors="replace").splitlines():
            if line.startswith("**Launched.**"):
                stamp = line.split("**Launched.**", 1)[1].split(" on ")[0].strip()
                # A date-only stamp (hand-written README) carries no time of
                # day; fall back to the launch files' mtimes below instead of
                # reading it as midnight.
                launched = parse_ts(stamp) if "T" in stamp else None
    if launched is None:
        launch_files = [p for p in (readme, d / "cmd.sh") if p.exists()]
        if launch_files:
            launched = dt.datetime.fromtimestamp(
                min(p.stat().st_mtime for p in launch_files), dt.timezone.utc
            )
    headers: list[dt.datetime] = []
    runlog = d / "logs" / "run.log"
    if runlog.exists():
        with open(runlog, errors="replace") as f:
            for line in f:
                m = HEADER_RE.match(line)
                if m:
                    t = parse_ts(m.group(1))
                    if t:
                        headers.append(t)
    starts = [t for t in [launched, *headers] if t]
    start = min(starts) if starts else None
    end = parse_ts(cfg.get("date")) if cfg else None
    # The finalizer debt: bakeoff stamps config.json once at launch and again in
    # a `finally` that does not wait on a detached trainer, so a stamp landing
    # within FINALIZER_SLOP of the launch is the launch stamp and not the end.
    # The two folders the storage switch created by hand carry no stamp at all.
    # Only in those two cases do the artifacts' mtimes decide -- otherwise the
    # stamp is authoritative, because a later mtime can simply mean something
    # else touched the folder afterwards (the lane-B token dataset's `meta/`
    # was written into hours later by the fine-tune's dataloader).
    if end is None or (start is not None and (end - start).total_seconds() < FINALIZER_SLOP):
        art = artifact_end(d)
        if art and (end is None or art > end):
            end = art
    devices, dev_source = folder_devices(d, cfg)
    span = (end - start).total_seconds() if (start and end and end >= start) else None
    gpu_hours = None
    if span is not None and devices is not None:
        gpu_hours = span / 3600.0 * len(devices)
    return {
        "dir": d,
        "cfg": cfg,
        "has_cfg": bool(cfg),
        "start": start,
        "end": end,
        "span_s": span,
        "attempts": max(1, len(headers)),
        "devices": devices,
        "dev_source": dev_source,
        "gpu_hours": gpu_hours,
        "fallback": bool(cfg.get("run_root_fallback")) if cfg else str(d).startswith("/tmp/"),
    }


def newest_eval_folder(lane: str, stage: str) -> tuple[Path | None, list[Path]]:
    cands = [
        p
        for p in run_folders(lane)
        if folder_stage(p) == stage and (p / "out" / "eval" / "eval_results.csv").is_file()
    ]

    def key(p: Path):
        t = folder_timing(p)
        return (t["end"] or dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc), p.name)

    cands.sort(key=key)
    return (cands[-1] if cands else None), cands


# --------------------------------------------------------------------------- eval folders
def load_eval(d: Path) -> dict:
    eval_dir = d / "out" / "eval"
    with open(eval_dir / "eval_results.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    episodes = []
    for r in rows:
        k = int(r["episode"])
        res = read_json(eval_dir / f"episode_{k}_result.json") or {}
        metrics = res.get("metrics") or {}
        reached = metrics.get("criteria_reached")
        total = metrics.get("criteria_total")
        progress = float(r["progress"])
        if reached is None:  # fall back to the csv's progress (= reached / total)
            reached = round(progress * len(MILESTONES))
            total = float(len(MILESTONES))
        episodes.append(
            {
                "episode": k,
                "length": int(r["episode_length"]),
                "success": r["success"].strip().lower() == "true",
                "progress": progress,
                "reached": int(reached),
                "total": int(total),
                "termination": res.get("termination_reason"),
            }
        )
    n = len(episodes)
    successes = [e for e in episodes if e["success"]]
    milestone_rates = [
        (sum(1 for e in episodes if e["reached"] >= k) / n if n else 0.0)
        for k in range(1, len(MILESTONES) + 1)
    ]
    timing = folder_timing(d)
    return {
        "dir": d,
        "episodes": episodes,
        "n": n,
        "n_success": len(successes),
        "success_rate": (len(successes) / n) if n else 0.0,
        "ci95": clopper_pearson(len(successes), n),
        "milestone_rates": milestone_rates,
        "progress_mean": (sum(e["progress"] for e in episodes) / n) if n else 0.0,
        "steps_to_success": [e["length"] for e in successes],
        "median_steps": statistics.median([e["length"] for e in successes]) if successes else None,
        "criteria_total": max((e["total"] for e in episodes), default=len(MILESTONES)),
        "eval_cfg": read_json(eval_dir / "config.json") or {},
        "run_summary": read_json(eval_dir / "run_summary.json") or {},
        "timing": timing,
        "cfg": timing["cfg"],
    }


def lane_gpu_rows(lane: str) -> list[dict]:
    return [folder_timing(p) for p in run_folders(lane)]


# --------------------------------------------------------------------------- round 2 discovery
# Nothing below resolves "the run I want" by recency. P6's aggregate and P9's
# screening watcher both picked the newest folder by mtime and both misread a
# run once two runs of the same lane and stage overlapped, which is exactly
# what P10 does (eight 200-rollout rows into run roots the screens were still
# using). Every run here is identified by what `harness/bakeoff.py` stamped
# into its own config.json: `--rollouts` says screen or row, `--checkpoint`
# says which checkpoint, and the checkpoint's fine-tune folder says which round.
ROUND2_TRAIN_STEPS = 20000
SCREEN_ROLLOUTS = 20
ROW_MIN_ROLLOUTS = 100


def folder_end(p: Path) -> dt.datetime:
    return folder_timing(p)["end"] or dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc)


def finetune_dirs(lane: str, max_steps: int = ROUND2_TRAIN_STEPS) -> list[Path]:
    """Fine-tune run folders of `lane` whose cmd.sh asks for `max_steps` steps.
    Round 2 ran each lane across two folders — home, then instance-local /tmp
    after the 2026-09-05 00:30 storage switch, resumed into the same optimizer
    state — so this returns both, and both count as round 2."""
    out = []
    for p in run_folders(lane):
        if folder_stage(p) != "finetune" or not (p / "cmd.sh").is_file():
            continue
        m = re.search(r"--max-steps\s+(\d+)", (p / "cmd.sh").read_text(errors="replace"))
        if m and int(m.group(1)) == max_steps:
            out.append(p)
    return out


def eval_runs(lane: str) -> list[dict]:
    """Every eval run folder of `lane` that has a csv, with the bakeoff stamp
    parsed: requested rollouts, the checkpoint it measured and that
    checkpoint's fine-tune run folder."""
    out = []
    for p in run_folders(lane):
        if folder_stage(p) != "eval" or not (p / "out" / "eval" / "eval_results.csv").is_file():
            continue
        args = (read_json(p / "config.json") or {}).get("args") or {}
        ck = str(args.get("checkpoint") or "")
        m = re.search(r"checkpoint-(\d+)", ck)
        # <ft run>/out/checkpoints/checkpoint-N  ->  <ft run>
        ft = Path(ck).parents[2] if ck else None
        out.append(
            {
                "dir": p,
                "requested": int(args.get("rollouts") or 0),
                "checkpoint": ck,
                "ft_dir": ft,
                "step": int(m.group(1)) if m else None,
            }
        )
    return out


def round2_evals(lane: str) -> tuple[dict[int, dict], dict[int, dict]]:
    """(screens, rows) of round 2 keyed by fine-tune step.

    Screens are the 20-rollout runs — the last one per step wins, which is what
    WP 9.3's stopping rule does with its series lines. Rows are the runs of at
    least 100 rollouts; if a step somehow has two, the last one wins as well."""
    ft = {str(p) for p in finetune_dirs(lane)}
    screens: dict[int, dict] = {}
    rows: dict[int, dict] = {}
    for e in eval_runs(lane):
        if e["step"] is None or e["ft_dir"] is None or str(e["ft_dir"]) not in ft:
            continue
        bucket = screens if e["requested"] == SCREEN_ROLLOUTS else (
            rows if e["requested"] >= ROW_MIN_ROLLOUTS else None
        )
        if bucket is None:
            continue
        prev = bucket.get(e["step"])
        if prev is None or folder_end(e["dir"]) >= folder_end(prev["dir"]):
            bucket[e["step"]] = e
    return screens, rows


def rank_key(r: dict) -> tuple:
    """P10's pre-registered ranking rule: (successes, milestone-6 rate,
    milestone-5 rate, step). Never mean progress — round 2's lane-A
    checkpoint-5000 had the highest mean progress of its lane with zero
    successes (the place-at-centre-and-stall mode)."""
    return (r["n_success"], r["milestone_rates"][5], r["milestone_rates"][4], r["step"])


def restrict(r: dict, first: int, last: int | None = None) -> dict:
    """The same row recomputed over episodes with index >= `first`.

    The 20-rollout screens that chose each lane's checkpoint used seeds 0-19 of
    the same sequence the 200-rollout rows use, so episodes 0-19 of a row are
    the selection set and 20-199 are genuinely held out from it."""
    eps = [e for e in r["episodes"] if e["episode"] >= first and (last is None or e["episode"] < last)]
    n = len(eps)
    succ = [e for e in eps if e["success"]]
    out = dict(r)
    out.update(
        {
            "episodes": eps,
            "n": n,
            "n_success": len(succ),
            "success_rate": (len(succ) / n) if n else 0.0,
            "ci95": clopper_pearson(len(succ), n),
            "milestone_rates": [
                (sum(1 for e in eps if e["reached"] >= k) / n if n else 0.0)
                for k in range(1, len(MILESTONES) + 1)
            ],
            "progress_mean": (sum(e["progress"] for e in eps) / n) if n else 0.0,
            "steps_to_success": [e["length"] for e in succ],
            "median_steps": statistics.median([e["length"] for e in succ]) if succ else None,
            "first_episode": first,
        }
    )
    return out


# --------------------------------------------------------------------------- parity
def finetune_command(lane: str) -> tuple[Path | None, dict | None, str | None]:
    """Newest <lane>/<date>_finetune folder -> (folder, parsed torchrun args, raw line)."""
    cands = [p for p in run_folders(lane) if folder_stage(p) == "finetune" and (p / "cmd.sh").is_file()]
    if not cands:
        return None, None, None
    cands.sort(key=lambda p: (folder_timing(p)["end"] or dt.datetime.min.replace(tzinfo=dt.timezone.utc), p.name))
    d = cands[-1]
    # Join backslash continuations first: round 2's cmd.sh wraps the torchrun
    # invocation over ten lines, so matching raw lines would hand shlex a
    # dangling backslash (and only a fragment of the command).
    text = re.sub(r"\\\n\s*", " ", (d / "cmd.sh").read_text(errors="replace"))
    line = None
    cuda = None
    for raw in text.splitlines():
        if "launch_finetune" in raw:
            line = raw.strip()
        if raw.startswith("export CUDA_VISIBLE_DEVICES="):
            cuda = raw.split("=", 1)[1]
    if cuda is None:  # round 2 acquires inside cmd.sh; the stamp has the devices
        cuda = (read_json(d / "config.json") or {}).get("cuda_visible_devices") or None
    if line is None:
        return d, None, None
    toks = shlex.split(line)
    args: dict[str, str] = {}
    prog = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            if "=" in t:
                k, v = t.split("=", 1)
                args[k] = v
                i += 1
                continue
            vals = []
            j = i + 1
            while j < len(toks) and not toks[j].startswith("--"):
                vals.append(toks[j])
                j += 1
            args[t] = " ".join(vals) if vals else "(flag)"
            i = j
        else:
            prog.append(t)
            i += 1
    args["(program)"] = " ".join(prog)
    args["(CUDA_VISIBLE_DEVICES)"] = cuda or NOT_RECORDED
    return d, args, line


def last_checkpoint(d: Path | None) -> Path | None:
    """The highest-numbered checkpoint directory of a fine-tune run folder."""
    if d is None:
        return None
    ck = d / "out" / "checkpoints"
    if not ck.is_dir():
        return None
    cands = []
    for p in ck.iterdir():
        m = re.fullmatch(r"checkpoint-(\d+)", p.name)
        if m and p.is_dir():
            cands.append((int(m.group(1)), p))
    return max(cands)[1] if cands else None


def finetune_loss(d: Path | None, step: int | None = None) -> str:
    """Last logged training loss of a fine-tune run.

    HuggingFace's Trainer writes its whole `log_history` into every checkpoint's
    `trainer_state.json`; the last entry carrying a `loss` key is the loss at
    the final logged step. With no `step` the highest-numbered checkpoint is
    read, so this follows a run whatever its budget. Anything missing (no
    folder, no checkpoint, no loss entry -- e.g. a fine-tune still running)
    reads as "not recorded"."""
    ckpt = (d / "out" / "checkpoints" / f"checkpoint-{step}") if (d and step) else last_checkpoint(d)
    state = read_json(ckpt / "trainer_state.json") if ckpt else None
    if not state:
        return NOT_RECORDED
    losses = [e["loss"] for e in state.get("log_history", []) if isinstance(e, dict) and "loss" in e]
    if not losses:
        return NOT_RECORDED
    return f"{float(losses[-1]):g}"


def loss_series(d: Path | None, every: int = 2500) -> str:
    """Training loss sampled every `every` steps from the last checkpoint's
    log_history — the curve behind 'both lanes were still improving'."""
    ckpt = last_checkpoint(d)
    state = read_json(ckpt / "trainer_state.json") if ckpt else None
    if not state:
        return NOT_RECORDED
    hist = [e for e in state.get("log_history", []) if isinstance(e, dict) and "loss" in e]
    if not hist:
        return NOT_RECORDED
    picked = []
    for target in range(every, 10**9, every):
        near = [e for e in hist if abs(int(e.get("step", 0)) - target) <= every // 10]
        if not near:
            break
        e = min(near, key=lambda e: abs(int(e.get("step", 0)) - target))
        picked.append(f"{target}: {float(e['loss']):.4f}")
    return ", ".join(picked) if picked else NOT_RECORDED


def modality_summary(path: Path) -> dict:
    text = path.read_text() if path.exists() else ""
    m = re.search(r"delta_indices\s*=\s*list\(range\((\d+)\)\)", text)
    keys = re.findall(r'modality_keys=\[([^\]]*)\]', text)
    action_keys = None
    for block in re.findall(r'"action":\s*ModalityConfig\((.*?)\n\s*\),', text, flags=re.S):
        km = re.search(r"modality_keys=\[([^\]]*)\]", block)
        if km:
            action_keys = [k.strip().strip('"') for k in km.group(1).split(",") if k.strip()]
    return {
        "file": path,
        "horizon": int(m.group(1)) if m else None,
        "action_keys": action_keys,
        "absolute": text.count("ActionRepresentation.ABSOLUTE"),
        "relative": text.count("ActionRepresentation.RELATIVE"),
        "non_eef": text.count("ActionType.NON_EEF"),
        "eef": len(re.findall(r"ActionType\.EEF\b", text)),
    }


EVAL_PARITY_KEYS = [
    ("task", ("task",)),
    ("seed", ("seed",)),
    ("rollouts", ("rollouts",)),
    ("rate (Hz)", ("rate",)),
    ("replan every (steps)", ("policy", "replan_every")),
    ("max steps (horizon)", ("max_steps",)),
    ("terminate on success", ("terminate_on_success",)),
    ("client", ("policy", "client")),
    ("embodiment / binding", ("policy", "embodiment")),
    ("grip threshold", ("policy", "grip_threshold")),
    ("splat", ("splat",)),
]


def dig(d: dict, path: tuple) -> str:
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return NOT_RECORDED
        cur = cur[k]
    return str(cur)


# --------------------------------------------------------------------------- tables
def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main_table(results: dict, lane_hours: dict) -> str:
    header = [
        "lane",
        "full-success rate",
        "95 % CI",
        "milestone 1…6 reach rates",
        "mean progress",
        "median steps-to-success",
        "action jerk",
        "eval wall-clock",
        "GPU-hours",
    ]
    rows = []
    for key, label, lane, _stage, hours_kind in CATEGORIES:
        r = results.get(key)
        if r is None:
            rows.append([label, "MISSING"] + ["—"] * (len(header) - 2))
            continue
        lo, hi = r["ci95"]
        if hours_kind == "lane":
            hrs = lane_hours[lane]["total"]
            hours_cell = f"{fmt_hours(hrs)} (lane total)"
        else:
            hours_cell = f"{fmt_hours(r['timing']['gpu_hours'])} (this run)"
        rows.append(
            [
                label,
                f"{r['n_success']}/{r['n']} ({100 * r['success_rate']:.0f} %)",
                f"{100 * lo:.0f}–{100 * hi:.0f} %",
                " / ".join(f"{x:.2f}" for x in r["milestone_rates"]),
                f"{r['progress_mean']:.3f}",
                (f"{r['median_steps']:.0f} steps ({r['median_steps'] / 50:.1f} s at 50 Hz)"
                 if r["median_steps"] is not None else "n/a (no success)"),
                NOT_RECORDED,
                fmt_minutes(r["timing"]["span_s"]),
                hours_cell,
            ]
        )
    return md_table(header, rows)


def milestone_table(results: dict) -> str:
    header = ["milestone (ordered)"] + [label.split(" — ")[0] for _k, label, *_ in CATEGORIES]
    rows = []
    for i, name in enumerate(MILESTONES):
        row = [f"{i + 1}. {name}"]
        for key, *_ in CATEGORIES:
            r = results.get(key)
            row.append("—" if r is None else f"{100 * r['milestone_rates'][i]:.0f} %")
        rows.append(row)
    return md_table(header, rows)


def sources_table(results: dict, candidates: dict) -> str:
    header = ["category", "run folder", "storage", "devices", "finalised (UTC)", "franka-sonic SHA", "other candidates"]
    rows = []
    for key, label, _lane, _stage, _kind in CATEGORIES:
        r = results.get(key)
        if r is None:
            rows.append([label, "MISSING", "—", "—", "—", "—", "—"])
            continue
        cfg = r["cfg"]
        sha = (cfg.get("repo_shas") or {}).get("franka-sonic", NOT_RECORDED)
        others = [p.name for p in candidates[key] if p != r["dir"]]
        rows.append(
            [
                label.split(" — ")[0],
                f"`{short_path(r['dir'])}`",
                "instance-local /tmp (not persistent)" if r["timing"]["fallback"] else "~/runs (Lustre)",
                cfg.get("cuda_visible_devices", NOT_RECORDED),
                (r["timing"]["end"].strftime("%Y-%m-%d %H:%M") if r["timing"]["end"] else NOT_RECORDED),
                sha[:12] if isinstance(sha, str) else NOT_RECORDED,
                ", ".join(others) if others else "none",
            ]
        )
    return md_table(header, rows)


def gpu_table(lane: str, rows: list[dict]) -> tuple[str, dict]:
    header = ["run folder", "storage", "devices", "start (UTC)", "end (UTC)", "wall-clock", "attempts in run.log", "GPU-hours"]
    out = []
    total = 0.0
    unrecorded = []
    for t in rows:
        d = t["dir"]
        dev = t["devices"]
        if dev is None and t["span_s"] is None:
            out.append([f"`{short_path(d)}`", "—", NOT_RECORDED, "—", "—", "—", "—", "not counted (no stamp)"])
            unrecorded.append(d)
            continue
        if dev is None:
            dev_cell = f"{NOT_RECORDED} ({(t['cfg'] or {}).get('cuda_visible_devices')!r})"
        elif not dev:
            dev_cell = "none (CPU)"
        else:
            dev_cell = ",".join(str(x) for x in dev)
            if t["dev_source"] != "config.json":
                dev_cell += f" (from `{t['dev_source']}`)"
        if t["gpu_hours"] is None:
            cell = NOT_RECORDED
            if dev is not None and not dev and t["span_s"] is not None:
                cell = "0 (CPU)"
            else:
                unrecorded.append(d)
        else:
            cell = fmt_hours(t["gpu_hours"])
            total += t["gpu_hours"]
        out.append(
            [
                f"`{short_path(d)}`",
                "/tmp" if t["fallback"] else "~/runs",
                dev_cell,
                fmt_ts(t["start"]),
                fmt_ts(t["end"]),
                fmt_minutes(t["span_s"]),
                str(t["attempts"]),
                cell,
            ]
        )
    out.append(["**total**", "", "", "", "", "", "", f"**{total:.2f}**"])
    return md_table(header, out), {"total": total, "unrecorded": unrecorded}


def parity_table(a: dict | None, b: dict | None) -> tuple[str, list[str]]:
    if not a or not b:
        return "fine-tune command(s) not found — parity cannot be checked", ["missing"]
    keys = sorted(set(a) | set(b), key=lambda k: (not k.startswith("("), k))
    rows = []
    diffs = []
    for k in keys:
        va, vb = a.get(k, "(absent)"), b.get(k, "(absent)")
        same = va == vb
        if not same:
            diffs.append(k)
        rows.append([f"`{k}`", f"`{va}`", f"`{vb}`", "same" if same else "**differs**"])
    return md_table(["argument", "lane A", "lane B", ""], rows), diffs


def eval_parity_table(results: dict) -> tuple[str, list[str]]:
    header = ["evaluation.eval argument"] + [label.split(" — ")[0] for _k, label, *_ in CATEGORIES]
    rows = []
    policy_diffs = []
    for name, path in EVAL_PARITY_KEYS:
        vals = []
        for key, *_ in CATEGORIES:
            r = results.get(key)
            vals.append("—" if r is None else dig(r["eval_cfg"], path))
        a, b = vals[0], vals[2]
        mark = "" if a == b else " (policies differ)"
        if a != b:
            policy_diffs.append(name)
        rows.append([name + mark] + [f"`{v}`" for v in vals])
    return md_table(header, rows), policy_diffs


def mvec(r: dict) -> str:
    """The six-milestone reach vector in per cent, leading — P10's rule, because
    for a staged task a falling mean progress can accompany a policy that gets
    strictly further (lane A at 7500: progress 0.467 -> 0.392 while milestone 4
    went 0 % -> 40 %)."""
    return " / ".join(f"{100 * x:.0f}" for x in r["milestone_rates"])


def row_table(entries: list[tuple[str, dict]]) -> str:
    header = [
        "row",
        "milestones 1…6 (% reached)",
        "successes",
        "success rate",
        "exact 95 % CI",
        "mean progress",
        "median steps-to-success",
    ]
    rows = []
    for label, r in entries:
        rows.append(
            [
                label,
                mvec(r),
                f"**{r['n_success']}/{r['n']}**",
                f"{100 * r['success_rate']:.1f} %",
                ci_text(r),
                f"{r['progress_mean']:.3f}",
                (
                    f"{r['median_steps']:.0f} ({r['median_steps'] / 50:.1f} s)"
                    if r["median_steps"] is not None
                    else "n/a (no success)"
                ),
            ]
        )
    return md_table(header, rows)


def ceiling_table(entries: list[tuple[str, dict, dict | None]]) -> str:
    """Two numbers per policy row: absolute success, and success divided by its
    own lane's oracle ceiling. A lane's oracle bounds its interface, not its
    policy, so the ratio says how much of the reachable headroom the VLA took."""
    header = ["row", "absolute", "own oracle ceiling", "÷ ceiling"]
    rows = []
    for label, r, ceil in entries:
        if ceil is None:
            rows.append([label, f"{100 * r['success_rate']:.1f} %", "—", "—"])
            continue
        ratio = (r["success_rate"] / ceil["success_rate"]) if ceil["success_rate"] else None
        rows.append(
            [
                label,
                f"{r['n_success']}/{r['n']} = {100 * r['success_rate']:.1f} %",
                f"{ceil['n_success']}/{ceil['n']} = {100 * ceil['success_rate']:.1f} %",
                "n/a" if ratio is None else f"**{100 * ratio:.0f} %**",
            ]
        )
    return md_table(header, rows)


def screen_table(screens: dict[str, dict[int, dict]]) -> str:
    """The P9 screening series — the two learning curves that chose the
    checkpoints. 20 rollouts each, one evaluation per checkpoint, the same
    binding as every row below."""
    steps = sorted(set(screens["lane_a"]) | set(screens["lane_b"]))
    header = ["fine-tune step"] + [f"**{s}**" for s in steps]
    rows = []
    for lane, short in (("lane_a", "lane A"), ("lane_b", "lane B")):
        succ, prog, m4 = [f"{short} successes / 20"], [f"{short} mean progress"], [f"{short} milestone 4 (%)"]
        for s in steps:
            r = screens[lane].get(s)
            if r is None:
                succ.append("—"); prog.append("—"); m4.append("—")
                continue
            succ.append(f"**{r['n_success']}**")
            prog.append(f"{r['progress_mean']:.3f}")
            m4.append(f"{100 * r['milestone_rates'][3]:.0f}")
        rows += [succ, m4, prog]
    return md_table(header, rows)


def stage_hours(lane: str) -> tuple[str, float]:
    """GPU-hours of a lane grouped by stage, so 'lane B's SONIC RL is counted
    in' is a number a reader can check rather than a claim."""
    per: dict[str, list[float]] = collections.defaultdict(list)
    for p in run_folders(lane):
        t = folder_timing(p)
        if t["gpu_hours"] is None:
            continue
        per[folder_stage(p)].append(t["gpu_hours"])
    total = sum(sum(v) for v in per.values())
    rows = []
    for stage in sorted(per, key=lambda s: -sum(per[s])):
        h = sum(per[stage])
        rows.append(
            [f"`{stage}`", str(len(per[stage])), f"{h:.2f}", f"{100 * h / total:.0f} %" if total else "—"]
        )
    rows.append(["**total**", f"**{sum(len(v) for v in per.values())}**", f"**{total:.2f}**", "100 %"])
    return md_table(["stage", "run folders", "GPU-hours", "share"], rows), total


# --------------------------------------------------------------------------- computed prose
# Everything below turns counts into sentences. No sentence here may assume an
# outcome: each branch is chosen by the numbers in the run folders, so the
# report stays true whatever the newest eval and oracle runs say.
def headline_verdict(ra: dict, rb: dict, oa: dict, ob: dict) -> str:
    """The verdict sentence. Every branch is chosen by the counts, so the
    paragraph stays true whatever the rows say — including the branch where the
    two lanes cannot be told apart, which is a result and not a failure."""
    state = ceiling_state(ob)
    common = (
        "Replaying the encoder-labelled SONIC tokens through the decoder with no VLA in the "
        f"loop succeeds {count_text(ob)} ({100 * ob['success_rate']:.1f} %, mean progress "
        f"{ob['progress_mean']:.3f}, exact 95 % CI {ci_text(ob)})"
    )
    if state == "open":
        head = (
            f"**The B-oracle ceiling is open.** {common}, at or above {gate_phrase(ob)}. Lane "
            f"B's {count_text(rb)} therefore measures the VLA over the token space rather "
            "than a controller that cannot execute its own labels, and can be read next to "
            f"lane A's {count_text(ra)} — though it is bounded by that ceiling and lane A's "
            f"is bounded by the A-oracle's {count_text(oa)}, which are not the same height."
        )
    elif state == "borderline":
        head = (
            f"**The B-oracle ceiling is open, but it is not the A-oracle's.** {common} — below "
            f"{gate_phrase(ob)}, though its interval covers that criterion, so the ceiling is "
            + ("It is below the A-oracle's " if overlap(oa["ci95"], ob["ci95"]) else
               "It is measurably below the A-oracle's ") + f"{count_text(oa)}"
            + ("" if overlap(oa["ci95"], ob["ci95"]) else " — those two intervals do not overlap") + ". "
            f"Lane B's {count_text(rb)} is therefore a VLA number, read against a ceiling "
            f"roughly {100 * (oa['success_rate'] - ob['success_rate']):.0f} points lower than "
            "lane A's: the decoder executes most of the labels its own encoder wrote, not all "
            "of them, and every lane-B episode is played through that handicap."
        )
    else:
        head = (
            f"**The B-oracle ceiling is closed.** {common}, below {gate_phrase(ob)} and below "
            f"the A-oracle's {count_text(oa)}. Lane B's {count_text(rb)} therefore cannot be "
            "read as a VLA number: the controller under it misses the task on the recorded "
            "spawns with no VLA in the loop at all."
        )
    pts = abs(100 * ra["success_rate"] - 100 * rb["success_rate"])
    cmp_head = (
        f"The two headline rows are lane A {count_text(ra)} ({100 * ra['success_rate']:.1f} %) "
        f"and lane B {count_text(rb)} ({100 * rb['success_rate']:.1f} %)"
    )
    if ra["n_success"] == rb["n_success"] and ra["n"] == rb["n"]:
        winner = "Neither lane won: the two counts are identical"
    elif abs(ra["success_rate"] - rb["success_rate"]) < 1e-12:
        winner = "Neither lane won: the two rates are identical"
    else:
        higher = "Lane A" if ra["success_rate"] > rb["success_rate"] else "Lane B"
        winner = f"{higher} scored higher, by {pts:.1f} points"
        if ra["n"] == rb["n"]:
            winner += f" = {abs(ra['n_success'] - rb['n_success'])} episodes of {ra['n']}"
    if overlap(ra["ci95"], rb["ci95"]):
        sep = (
            f"**do overlap**, so n = {ra['n']} does not separate the two lanes: the gap is "
            "inside what this many rollouts produce by sampling alone. That is the result — "
            "not a failure of the experiment, and not evidence that the lanes are equal "
            "either; it is the resolution this n buys."
        )
    else:
        sep = (
            f"**do not overlap**, so n = {ra['n']} does separate these two rows: the gap is "
            "larger than sampling at this n explains. It separates *these two runs* — one "
            "training seed per lane, one checkpoint chosen per lane by a 20-rollout screen, "
            "one task, one architecture pair — not the two control stacks in general."
        )
    return (
        f"{head} {cmp_head}; {winner}. Exact 95 % Clopper–Pearson intervals are lane A "
        f"{ci_text(ra)} and lane B {ci_text(rb)}, and they {sep}"
    )


def spread_sentence(label: str, entries: list[tuple[str, dict]]) -> str:
    """How much of 'which checkpoint won' is luck: the range across a lane's
    three 200-rollout rows, each of which a 20-rollout screen ranked."""
    if len(entries) < 2:
        return f"{label}: only one checkpoint was measured at this n, so there is no spread to report."
    best = max(entries, key=lambda kv: kv[1]["n_success"])
    worst = min(entries, key=lambda kv: kv[1]["n_success"])
    span = best[1]["n_success"] - worst[1]["n_success"]
    def short(k: str) -> str:
        return k.split("`")[1] if "`" in k else k

    parts = ", ".join(f"{short(k)} {count_text(v)}" for k, v in entries)
    ns = {v["n"] for _k, v in entries}
    where = f"all measured at n = {ns.pop()}" if len(ns) == 1 else "measured at the n shown"
    return (
        f"{label}: {parts} — a spread of "
        f"{100 * (best[1]['success_rate'] - worst[1]['success_rate']):.1f} points across "
        f"{len(entries)} checkpoints of one training run, {where}."
    )


def repro_sentence(lane_label: str, screen: dict | None, row: dict | None, n_screen: int = SCREEN_ROLLOUTS) -> str:
    """The same checkpoint on the same seeds, evaluated twice: once as the
    20-rollout screen, once as the first 20 episodes of the 200-rollout row.
    Both use seed 0 + episode index and the identical binding, so any gap is
    the evaluation's own run-to-run variance -- GR00T N1.7 samples its actions,
    so a seed does not pin a rollout."""
    if screen is None or row is None:
        return f"{lane_label}: not measurable (a screen or a row is missing)."
    head = restrict(row, 0, n_screen)
    if head["n"] == 0:
        return f"{lane_label}: the row has no episodes below {n_screen}."
    gap = screen["n_success"] - head["n_success"]
    if gap == 0:
        tail = "the two agree exactly"
    else:
        tail = (
            f"they differ by {abs(gap)} episode{'s' if abs(gap) != 1 else ''} "
            f"({abs(100 * (screen['success_rate'] - head['success_rate'])):.0f} points) "
            "on identical seeds"
        )
    return (
        f"{lane_label} `checkpoint-{screen['step']}` on episodes 0-{head['n'] - 1}: the screen "
        f"scored {count_text(screen)}, the 200-rollout row scores {count_text(head)} on those "
        f"same episodes — {tail}."
    )


def selection_sentence(lane_label: str, screens: dict[int, dict], picked: int, row: dict | None) -> str:
    """What the 20-rollout screen said about the chosen checkpoint versus what
    200 rollouts said about it. The screens select; they do not measure."""
    s = screens.get(picked)
    if s is None:
        return f"{lane_label}: no screen recorded for checkpoint-{picked}."
    txt = (
        f"{lane_label} chose **checkpoint-{picked}** on the screen's {count_text(s)} "
        f"(milestones {mvec(s)})"
    )
    if row is None:
        return txt + " — no 200-rollout row for it."
    drift = 100 * (row["success_rate"] - s["success_rate"])
    return (
        txt + f"; at 200 rollouts that same checkpoint scores {count_text(row)} "
        f"({100 * row['success_rate']:.1f} %), {drift:+.1f} points against its own screen."
    )


def ci_sentence(results: dict) -> str:
    parts = []
    for key, label, *_ in CATEGORIES:
        r = results.get(key)
        if r is None:
            continue
        parts.append(f"{label.split(' — ')[0]} {count_text(r)} is compatible with {ci_text(r)}")
    return "; ".join(parts) + "."


def m1_note(results: dict) -> str:
    counts = []
    for key in ("lane_a_policy", "lane_b_policy", "lane_b_oracle"):
        r = results.get(key)
        if r is not None:
            counts.append(sum(1 for e in r["episodes"] if e["reached"] >= 1))
    if not counts:
        return "are not recorded."
    joined = (
        " and ".join([", ".join(str(c) for c in counts[:-1]), str(counts[-1])])
        if len(counts) > 1
        else str(counts[0])
    )
    spread = max(counts) - min(counts)
    tail = (
        f"a spread of {spread} episode{'s' if spread != 1 else ''}"
        + (", inside the sampling noise at this n." if spread <= 1 else ".")
    )
    return f"are {joined} episodes respectively — {tail}"


def milestone_reading(results: dict) -> str:
    lines = []
    for key, label, *_ in CATEGORIES:
        r = results.get(key)
        short = label.split(" — ")[0]
        if r is None:
            lines.append(f"- **{short}**: MISSING")
            continue
        n = r["n"]
        counts = [
            sum(1 for e in r["episodes"] if e["reached"] >= k) for k in range(1, len(MILESTONES) + 1)
        ]
        drop = next((i for i, c in enumerate(counts) if c < n), None)
        if drop is None:
            drop_txt = f"no milestone falls below {n}/{n}"
        else:
            drop_txt = (
                f"the first milestone not reached by every episode is {drop + 1} "
                f"({MILESTONES[drop]}) at {counts[drop]}/{n}"
            )
        tally = collections.Counter(round(e["progress"], 3) for e in r["episodes"])
        mode_val, mode_cnt = max(tally.items(), key=lambda kv: (kv[1], -kv[0]))
        lines.append(
            f"- **{short}**: "
            + ", ".join(f"{c}/{n}" for c in counts)
            + f" episodes reach milestones 1…{len(MILESTONES)}; {drop_txt}; most common "
            f"progress {mode_val:.3f} in {mode_cnt} of {n} episodes."
        )
    return "\n".join(lines)


def ob_verdict(ob: dict, oa: dict) -> str:
    state = ceiling_state(ob)
    gap = 100 * ((oa["success_rate"] if oa["n"] else 0.0) - ob["success_rate"])
    stem = (
        f"{count_text(ob)} ({100 * ob['success_rate']:.1f} %, mean progress "
        f"{ob['progress_mean']:.3f}, 95 % CI {ci_text(ob)})"
    )
    if state == "open":
        return (
            f"It is open: {stem}, at or above {gate_phrase(ob)} and {gap:.0f} points from the "
            f"A-oracle's {count_text(oa)}. The decoder that lane B's policy server runs "
            "executes the labels its own encoder wrote."
        )
    if state == "borderline":
        return (
            f"It is open but lower than lane A's: {stem}. That is below {gate_phrase(ob)}, "
            "though the interval covers the criterion, so the ceiling is not measurably below "
            f"it; it is {gap:.0f} points below the A-oracle's {count_text(oa)}"
            + (", a gap this n does not resolve. " if overlap(oa["ci95"], ob["ci95"])
               else ", and that gap is larger than sampling explains (the intervals do not "
                    "overlap). ")
            + "Lane B's policy plays every episode through a decoder that drops roughly "
            f"one episode in {1 / max(1e-9, 1 - ob['success_rate']):.0f} on its own labels."
        )
    return (
        f"It is closed: {stem}, below {gate_phrase(ob)} and {gap:.0f} points from the "
        f"A-oracle's {count_text(oa)}. The decoder does not execute the labels its own encoder "
        "wrote, which caps lane B before the VLA is asked anything."
    )


def b_verdict(ra: dict, rb: dict, ob: dict) -> str:
    if ceiling_state(ob) in ("open", "borderline"):
        return (
            f"**Lane B's {count_text(rb)} is therefore readable as a VLA number.** With the "
            f"ceiling at {count_text(ob)}, lane B's row measures GR00T over the token space "
            "driving a decoder that can execute those tokens — the comparison the bake-off "
            f"asked for. It remains one checkpoint of one training run over {rb['n']} "
            "rollouts, with the interval quoted in section 1, and it is bounded above by the "
            "ceiling, not by the A-oracle."
        )
    return (
        f"**Consequently lane B's {count_text(rb)} is uninformative about the VLA.** Lane B's "
        "fine-tune emitted in-range tokens (|t| ≤ 1.0, none clipped) at the same latency as "
        "lane A; whether GR00T learned the token stream cannot be judged behind a decoder that "
        "cannot execute the labels. A B-oracle near the A-oracle is a precondition for reading "
        "lane B's row at all — which is why P5 gated on it and P6 re-ran it."
    )


def storage_note(r: dict) -> str:
    return (
        "instance-local `/tmp`, not persistent, `NEEDS-COPY` in STATUS.md"
        if r["timing"]["fallback"]
        else "Lustre home, persistent"
    )


def instance_local_note(results: dict) -> str:
    if any(r["timing"]["fallback"] for r in results.values()):
        return (
            "Those folders do not survive a pod restart; their `eval_results.csv` and episode "
            "JSONs are small and are tagged `NEEDS-COPY` in `plan/STATUS.md`."
        )
    return (
        "Every row above is on Lustre home, so nothing in this section needs copying off "
        "instance-local storage."
    )


# --------------------------------------------------------------------------- main
def build(verbose: bool = True, rows: dict[str, Path] | None = None, held_out_from: int = 20) -> str:
    rows = rows or {}
    results: dict[str, dict] = {}
    candidates: dict[str, list[Path]] = {}
    missing = []
    for key, label, lane, stage, _kind in CATEGORIES:
        if key in rows:
            d, cands, how = rows[key], [rows[key]], "explicit --row"
        else:
            d, cands = newest_eval_folder(lane, stage)
            how = "newest by finalisation stamp"
        candidates[key] = cands
        if d is None:
            missing.append(label)
            continue
        results[key] = load_eval(d)
        if verbose:
            r = results[key]
            print(
                f"[aggregate] {key:14s} <- {d}  ({r['n']} episodes, {r['n_success']} successes; "
                f"{how}, {len(cands)} candidate folder(s))",
                file=sys.stderr,
            )
    if missing:
        raise SystemExit("[aggregate] MISSING eval run folder(s): " + "; ".join(missing))

    lane_hours = {}
    gpu_tables = {}
    for lane in ("shared", "lane_a", "lane_b"):
        table, info = gpu_table(lane, lane_gpu_rows(lane))
        gpu_tables[lane] = table
        lane_hours[lane] = info

    ft_a_dir, ft_a, ft_a_line = finetune_command("lane_a")
    ft_b_dir, ft_b, ft_b_line = finetune_command("lane_b")
    parity, diffs = parity_table(ft_a, ft_b)
    mod_a = modality_summary(REPO / "harness" / "lane_a" / "modality_config_dual_fr3.py")
    mod_b = modality_summary(REPO / "harness" / "lane_b" / "modality_config_dual_fr3_sonic.py")
    modality_rows = []
    for name, m in (("lane A", mod_a), ("lane B", mod_b)):
        modality_rows.append(
            [
                name,
                f"`{m['file'].relative_to(REPO)}`",
                str(m["horizon"]) if m["horizon"] is not None else NOT_RECORDED,
                ", ".join(m["action_keys"]) if m["action_keys"] else NOT_RECORDED,
                f"{m['absolute']} ABSOLUTE / {m['relative']} RELATIVE",
                f"{m['non_eef']} NON_EEF / {m['eef']} EEF",
            ]
        )
    modality_table = md_table(
        ["lane", "modality config", "action horizon", "action keys", "representation", "type"], modality_rows
    )
    eval_parity, eval_diffs = eval_parity_table(results)

    ra, rb = results["lane_a_policy"], results["lane_b_policy"]
    oa, ob = results["lane_a_oracle"], results["lane_b_oracle"]
    n = ra["n"]

    def succ(r):
        return f"{r['n_success']}/{r['n']}"

    # ---- round 2: the screening series, the eight rows, the two ceilings ----
    screens: dict[str, dict[int, dict]] = {}
    lane_rows: dict[str, dict[int, dict]] = {}
    for lane in ("lane_a", "lane_b"):
        sc, rw = round2_evals(lane)
        screens[lane] = {s: dict(load_eval(e["dir"]), step=s) for s, e in sorted(sc.items())}
        lane_rows[lane] = {s: dict(load_eval(e["dir"]), step=s) for s, e in sorted(rw.items())}
        if verbose:
            print(
                f"[aggregate] {lane}: {len(screens[lane])} round-2 screens "
                f"{sorted(screens[lane])}, {len(lane_rows[lane])} 200-rollout rows "
                f"{sorted(lane_rows[lane])}",
                file=sys.stderr,
            )
    picked = {
        lane: (max(screens[lane].values(), key=rank_key)["step"] if screens[lane] else None)
        for lane in screens
    }

    def row_step(r: dict) -> int | None:
        m = re.search(r"checkpoint-(\d+)", str(((r["cfg"] or {}).get("args") or {}).get("checkpoint") or ""))
        return int(m.group(1)) if m else None

    lane_entries: dict[str, list[tuple[str, dict]]] = {}
    for lane, short in (("lane_a", "lane A"), ("lane_b", "lane B")):
        ordered = sorted(lane_rows[lane].values(), key=lambda r: (r["step"] != picked[lane], -r["step"]))
        lane_entries[lane] = [
            (
                f"{short} `checkpoint-{r['step']}`" + (" — **headline**" if r["step"] == picked[lane] else ""),
                r,
            )
            for r in ordered
        ]
    row_entries = (
        lane_entries["lane_a"]
        + [("**A-oracle** — recorded joint targets, no VLA", oa)]
        + lane_entries["lane_b"]
        + [("**B-oracle** — encoder tokens → decoder, no VLA", ob)]
    )
    held = [(label, restrict(r, held_out_from)) for label, r in row_entries]
    ra_h, rb_h = restrict(ra, held_out_from), restrict(rb, held_out_from)

    def overlap_sentence(x: dict, y: dict, tag: str) -> str:
        ov = overlap(x["ci95"], y["ci95"])
        return (
            f"{tag} — lane A {count_text(x)} ({ci_text(x)}) against lane B {count_text(y)} "
            f"({ci_text(y)}): the two intervals "
            + (
                "**overlap**, so this comparison does not separate the lanes."
                if ov
                else "**do not overlap**, so this comparison separates these two rows."
            )
        )

    def headline_check(r: dict, lane: str, short: str) -> str:
        step, want = row_step(r), picked[lane]
        if step is None or want is None:
            return f"{short}: the pre-registered pick could not be recovered from the stamps."
        if step == want:
            return (
                f"{short}: the row reported above is `checkpoint-{step}`, which is the "
                "checkpoint the screens had already chosen before any 200-rollout row was "
                "seen — the pre-registration holds."
            )
        return (
            f"**{short}: the row reported above is `checkpoint-{step}`, but the screening rule "
            f"picked `checkpoint-{want}`. The headline is not the pre-registered row — say so "
            "when quoting it.**"
        )

    stage_a, hours_a = stage_hours("lane_a")
    stage_b, hours_b = stage_hours("lane_b")
    stage_sh, hours_sh = stage_hours("shared")

    subs = {
        "GENERATED": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "SOURCES_TABLE": sources_table(results, candidates),
        "MAIN_TABLE": main_table(results, lane_hours),
        "MILESTONE_TABLE": milestone_table(results),
        "PARITY_TABLE": parity,
        "PARITY_DIFFS": ", ".join(f"`{d}`" for d in diffs) if diffs else "none",
        "PARITY_VERDICT": (
            "The two commands are identical except for the arguments listed as differing above; "
            "every one of those is the data path, the modality config, the output folder or "
            "torchrun's rendezvous port (the two fine-tunes ran concurrently and cannot share "
            "one), so the training budget and every training hyper-parameter is the same."
            if set(diffs) <= {"--dataset-path", "--modality-config-path", "--output-dir", "--master_port"}
            else "**The two fine-tunes differ in more than the data/config/output paths — the "
            "comparison is not at parity. See the differing rows above.**"
        ),
        "FT_A_DIR": short_path(ft_a_dir) if ft_a_dir else NOT_RECORDED,
        "FT_B_DIR": short_path(ft_b_dir) if ft_b_dir else NOT_RECORDED,
        "FT_A_CMD": ft_a_line or NOT_RECORDED,
        "FT_B_CMD": ft_b_line or NOT_RECORDED,
        "FT_A_LOSS": finetune_loss(ft_a_dir),
        "FT_B_LOSS": finetune_loss(ft_b_dir),
        "MODALITY_TABLE": modality_table,
        "EVAL_PARITY_TABLE": eval_parity,
        "EVAL_PARITY_VERDICT": (
            "The two policy evals share every argument above."
            if not eval_diffs
            else "**The two policy evals differ in: " + ", ".join(eval_diffs) + ".**"
        ),
        "A_SUCC": succ(ra),
        "B_SUCC": succ(rb),
        "OA_SUCC": succ(oa),
        "OB_SUCC": succ(ob),
        "A_PROGRESS": f"{ra['progress_mean']:.3f}",
        "B_PROGRESS": f"{rb['progress_mean']:.3f}",
        "OA_PROGRESS": f"{oa['progress_mean']:.3f}",
        "OB_PROGRESS": f"{ob['progress_mean']:.3f}",
        "A_M1": f"{100 * ra['milestone_rates'][0]:.0f} %",
        "B_M1": f"{100 * rb['milestone_rates'][0]:.0f} %",
        "OB_M1": f"{100 * ob['milestone_rates'][0]:.0f} %",
        "OA_MEDIAN_STEPS": f"{oa['median_steps']:.0f}" if oa["median_steps"] is not None else "n/a",
        "OA_MIN_STEPS": str(min(oa["steps_to_success"])) if oa["steps_to_success"] else "n/a",
        "OA_MAX_STEPS": str(max(oa["steps_to_success"])) if oa["steps_to_success"] else "n/a",
        "N_ROLLOUTS": str(n),
        "N_LAST": str(max(0, n - 1)),
        "HEADLINE_VERDICT": headline_verdict(ra, rb, oa, ob),
        "CI_SENTENCE": ci_sentence(results),
        "M1_NOTE": m1_note(results),
        "MILESTONE_READING": milestone_reading(results),
        "OB_VERDICT": ob_verdict(ob, oa),
        "B_VERDICT": b_verdict(ra, rb, ob),
        "GPU_TABLE_SHARED": gpu_tables["shared"],
        "GPU_TABLE_A": gpu_tables["lane_a"],
        "GPU_TABLE_B": gpu_tables["lane_b"],
        "GPU_TOTAL_SHARED": f"{lane_hours['shared']['total']:.2f}",
        "GPU_TOTAL_A": f"{lane_hours['lane_a']['total']:.2f}",
        "GPU_TOTAL_B": f"{lane_hours['lane_b']['total']:.2f}",
        "GPU_UNRECORDED": (
            ", ".join(f"`{short_path(p)}`" for lane in lane_hours for p in lane_hours[lane]["unrecorded"])
            or "none"
        ),
        "A_DIR": short_path(ra["dir"]),
        "B_DIR": short_path(rb["dir"]),
        "OA_DIR": short_path(oa["dir"]),
        "OB_DIR": short_path(ob["dir"]),
        "A_STORAGE": storage_note(ra),
        "B_STORAGE": storage_note(rb),
        "OA_STORAGE": storage_note(oa),
        "OB_STORAGE": storage_note(ob),
        "INSTANCE_LOCAL": (
            ", ".join(f"`{short_path(r['dir'])}`" for r in results.values() if r["timing"]["fallback"]) or "none"
        ),
        "INSTANCE_LOCAL_NOTE": instance_local_note(results),
        # ---- round 2 ----
        "ROWS_TABLE": row_table(row_entries),
        "ROWS_TABLE_HELDOUT": row_table(held),
        "HELDOUT_FROM": str(held_out_from),
        "HELDOUT_N": str(ra_h["n"]),
        "CEILING_TABLE": ceiling_table(
            [(lbl, r, oa) for lbl, r in lane_entries["lane_a"]]
            + [(lbl, r, ob) for lbl, r in lane_entries["lane_b"]]
        ),
        "SCREEN_TABLE": screen_table(screens),
        "SELECTION_A": selection_sentence(
            "**Lane A**", screens["lane_a"], picked["lane_a"], lane_rows["lane_a"].get(picked["lane_a"])
        ),
        "SELECTION_B": selection_sentence(
            "**Lane B**", screens["lane_b"], picked["lane_b"], lane_rows["lane_b"].get(picked["lane_b"])
        ),
        "REPRO_A": repro_sentence(
            "**Lane A**", screens["lane_a"].get(picked["lane_a"]), lane_rows["lane_a"].get(picked["lane_a"])
        ),
        "REPRO_B": repro_sentence(
            "**Lane B**", screens["lane_b"].get(picked["lane_b"]), lane_rows["lane_b"].get(picked["lane_b"])
        ),
        "SPREAD_A": spread_sentence("**Lane A**", lane_entries["lane_a"]),
        "SPREAD_B": spread_sentence("**Lane B**", lane_entries["lane_b"]),
        "OVERLAP_ALL": overlap_sentence(ra, rb, f"All {ra['n']} episodes"),
        "OVERLAP_HELDOUT": overlap_sentence(
            ra_h, rb_h, f"Held-out episodes {held_out_from}–{held_out_from + ra_h['n'] - 1}"
        ),
        "HEADLINE_CHECK_A": headline_check(ra, "lane_a", "Lane A"),
        "HEADLINE_CHECK_B": headline_check(rb, "lane_b", "Lane B"),
        "STAGE_HOURS_A": stage_a,
        "STAGE_HOURS_B": stage_b,
        "STAGE_HOURS_SHARED": stage_sh,
        "STAGE_TOTAL_A": f"{hours_a:.2f}",
        "STAGE_TOTAL_B": f"{hours_b:.2f}",
        "STAGE_TOTAL_SHARED": f"{hours_sh:.2f}",
        "LOSS_SERIES_A": loss_series(ft_a_dir),
        "LOSS_SERIES_B": loss_series(ft_b_dir),
        "A_CKPT": f"checkpoint-{row_step(ra)}" if row_step(ra) else NOT_RECORDED,
        "B_CKPT": f"checkpoint-{row_step(rb)}" if row_step(rb) else NOT_RECORDED,
        "A_RATIO": (
            f"{100 * ra['success_rate'] / oa['success_rate']:.0f} %" if oa["success_rate"] else "n/a"
        ),
        "B_RATIO": (
            f"{100 * rb['success_rate'] / ob['success_rate']:.0f} %" if ob["success_rate"] else "n/a"
        ),
        "A_RATE": f"{100 * ra['success_rate']:.1f} %",
        "B_RATE": f"{100 * rb['success_rate']:.1f} %",
        "OA_RATE": f"{100 * oa['success_rate']:.1f} %",
        "OB_RATE": f"{100 * ob['success_rate']:.1f} %",
        "A_MVEC": mvec(ra),
        "B_MVEC": mvec(rb),
        "OA_MVEC": mvec(oa),
        "OB_MVEC": mvec(ob),
        "A_CI": ci_text(ra),
        "B_CI": ci_text(rb),
        "OA_CI": ci_text(oa),
        "OB_CI": ci_text(ob),
    }

    text = TEMPLATE.read_text()
    unused = set(subs)
    for key, value in subs.items():
        token = "{{" + key + "}}"
        if token in text:
            unused.discard(key)
        text = text.replace(token, value)
    leftover = sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", text)))
    if leftover:
        raise SystemExit(f"[aggregate] template placeholders without a value: {leftover}")
    if verbose and unused:
        print(f"[aggregate] note: unused substitutions {sorted(unused)}", file=sys.stderr)
    return text


ROW_KEYS = [c[0] for c in CATEGORIES]
ROW_ALIASES = {
    "lane_a_eval": "lane_a_policy",
    "lane_b_eval": "lane_b_policy",
    "oracle_a": "lane_a_oracle",
    "oracle_b": "lane_b_oracle",
}


def parse_rows(pairs: list[str], flags: dict[str, Path | None]) -> dict[str, Path]:
    """`--row <label>=<run folder>` (repeatable) plus the four named flags.

    Explicit paths exist because "newest wins" is wrong the moment two runs of
    the same lane and stage overlap, which P10 does by design: eight rows and
    the screens that chose them share the two run roots. A label may be a
    category key (`lane_a_policy`) or its friendlier alias (`lane_a_eval`)."""
    out: dict[str, Path] = {}
    for name, path in flags.items():
        if path is not None:
            out[ROW_ALIASES.get(name, name)] = Path(path).expanduser().resolve()
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"[aggregate] --row needs <label>=<run folder>, got {pair!r}")
        label, path = pair.split("=", 1)
        key = ROW_ALIASES.get(label.strip(), label.strip())
        if key not in ROW_KEYS:
            raise SystemExit(
                f"[aggregate] --row label {label!r} is not one of "
                f"{', '.join(ROW_KEYS)} (aliases: {', '.join(ROW_ALIASES)})"
            )
        out[key] = Path(path.strip()).expanduser().resolve()
    for key, p in out.items():
        if not (p / "out" / "eval" / "eval_results.csv").is_file():
            raise SystemExit(f"[aggregate] {key}: {p} has no out/eval/eval_results.csv")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stdout", action="store_true", help="print the report instead of writing plan/REPORT.md")
    ap.add_argument("--out", type=Path, default=REPORT, help=f"output path (default {REPORT})")
    ap.add_argument(
        "--row", action="append", default=[], metavar="LABEL=RUN",
        help="pin a row to a run folder instead of taking the newest; repeatable. "
             f"LABEL is one of {', '.join(ROW_KEYS)} or {', '.join(ROW_ALIASES)}",
    )
    ap.add_argument("--lane-a-eval", type=Path, help="run folder for lane A's policy row")
    ap.add_argument("--lane-b-eval", type=Path, help="run folder for lane B's policy row")
    ap.add_argument("--oracle-a", type=Path, help="run folder for the A-oracle row")
    ap.add_argument("--oracle-b", type=Path, help="run folder for the B-oracle row")
    ap.add_argument(
        "--held-out-from", type=int, default=20, metavar="K",
        help="first episode index of the held-out slice reported alongside every row "
             "(default 20: the 20-rollout screens that selected the checkpoints used "
             "episodes 0-19 of the same seed sequence)",
    )
    args = ap.parse_args(argv)
    rows = parse_rows(
        args.row,
        {
            "lane_a_eval": args.lane_a_eval,
            "lane_b_eval": args.lane_b_eval,
            "oracle_a": args.oracle_a,
            "oracle_b": args.oracle_b,
        },
    )
    text = build(rows=rows, held_out_from=args.held_out_from)
    if args.stdout:
        sys.stdout.write(text)
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"[aggregate] wrote {args.out} ({len(text.splitlines())} lines)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
