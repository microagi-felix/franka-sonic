#!/usr/bin/env python3
"""P4 aggregation: the four eval run folders -> plan/REPORT.md.

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
import csv
import datetime as dt
import json
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
    if end is None:
        logs = list((d / "logs").glob("*")) if (d / "logs").is_dir() else []
        if logs:
            end = dt.datetime.fromtimestamp(max(p.stat().st_mtime for p in logs), dt.timezone.utc)
    devices = parse_devices(cfg.get("cuda_visible_devices")) if cfg else None
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


# --------------------------------------------------------------------------- parity
def finetune_command(lane: str) -> tuple[Path | None, dict | None, str | None]:
    """Newest <lane>/<date>_finetune folder -> (folder, parsed torchrun args, raw line)."""
    cands = [p for p in run_folders(lane) if folder_stage(p) == "finetune" and (p / "cmd.sh").is_file()]
    if not cands:
        return None, None, None
    cands.sort(key=lambda p: (folder_timing(p)["end"] or dt.datetime.min.replace(tzinfo=dt.timezone.utc), p.name))
    d = cands[-1]
    line = None
    cuda = None
    for raw in (d / "cmd.sh").read_text().splitlines():
        if "launch_finetune" in raw:
            line = raw.strip()
        if raw.startswith("export CUDA_VISIBLE_DEVICES="):
            cuda = raw.split("=", 1)[1]
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
        if not t["has_cfg"]:
            out.append([f"`{short_path(d)}`", "—", NOT_RECORDED, "—", "—", "—", "—", "not counted (no config.json)"])
            unrecorded.append(d)
            continue
        dev = t["devices"]
        if dev is None:
            dev_cell = f"{NOT_RECORDED} ({t['cfg'].get('cuda_visible_devices')!r})"
        elif not dev:
            dev_cell = "none (CPU)"
        else:
            dev_cell = ",".join(str(x) for x in dev)
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


# --------------------------------------------------------------------------- main
def build(verbose: bool = True) -> str:
    results: dict[str, dict] = {}
    candidates: dict[str, list[Path]] = {}
    missing = []
    for key, label, lane, stage, _kind in CATEGORIES:
        d, cands = newest_eval_folder(lane, stage)
        candidates[key] = cands
        if d is None:
            missing.append(label)
            continue
        results[key] = load_eval(d)
        if verbose:
            r = results[key]
            print(
                f"[aggregate] {key:14s} <- {d}  ({r['n']} episodes, {r['n_success']} successes; "
                f"{len(cands)} candidate folder(s))",
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
    ci0 = clopper_pearson(0, n)[1]
    ci1 = clopper_pearson(n, n)[0]

    def succ(r):
        return f"{r['n_success']}/{r['n']}"

    subs = {
        "GENERATED": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "SOURCES_TABLE": sources_table(results, candidates),
        "MAIN_TABLE": main_table(results, lane_hours),
        "MILESTONE_TABLE": milestone_table(results),
        "PARITY_TABLE": parity,
        "PARITY_DIFFS": ", ".join(f"`{d}`" for d in diffs) if diffs else "none",
        "PARITY_VERDICT": (
            "The two commands are identical except for the arguments listed as differing above; "
            "every one of those is the data path, the modality config or the output folder, so the "
            "training budget is the same."
            if set(diffs) <= {"--dataset-path", "--modality-config-path", "--output-dir"}
            else "**The two fine-tunes differ in more than the data/config/output paths — the "
            "comparison is not at parity. See the differing rows above.**"
        ),
        "FT_A_DIR": short_path(ft_a_dir) if ft_a_dir else NOT_RECORDED,
        "FT_B_DIR": short_path(ft_b_dir) if ft_b_dir else NOT_RECORDED,
        "FT_A_CMD": ft_a_line or NOT_RECORDED,
        "FT_B_CMD": ft_b_line or NOT_RECORDED,
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
        "CI_ZERO_UPPER": f"{100 * ci0:.0f} %",
        "CI_FULL_LOWER": f"{100 * ci1:.0f} %",
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
        "INSTANCE_LOCAL": (
            ", ".join(f"`{short_path(r['dir'])}`" for r in results.values() if r["timing"]["fallback"]) or "none"
        ),
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stdout", action="store_true", help="print the report instead of writing plan/REPORT.md")
    ap.add_argument("--out", type=Path, default=REPORT, help=f"output path (default {REPORT})")
    args = ap.parse_args(argv)
    text = build()
    if args.stdout:
        sys.stdout.write(text)
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"[aggregate] wrote {args.out} ({len(text.splitlines())} lines)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
