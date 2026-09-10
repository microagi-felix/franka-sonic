#!/usr/bin/env python3
"""Screen every checkpoint of a fine-tune, N times each, as the checkpoints appear.

Descendant of the P11/P12 run-folder watchers, made a repo file so its two hard-won fixes
stop being rewritten every campaign (harness debt (e)):

  * **Attribution.** A screen's run folder comes from that screen's OWN launcher log -- the
    `[bakeoff] run folder <path>` line, cross-checked against the closing
    `[bakeoff] OK rc=0  log: <run>/logs/run.log` line. Never a directory listing, never
    newest-by-mtime. That trap (harness debt (c)) has already recorded a 16/20 screen as
    1/7 in P9 and mis-filed a GPU-hour sum in P11.
  * **Zombie reap.** `alive()` waits the pid before reading /proc, so a finished screen
    whose launcher has not been reaped is not mistaken for a running one.

  * **Two screens per checkpoint, >= --min-gap-s apart** is P11's lesson: a single
    20-rollout screen measures the regime a run happened to be in, not the checkpoint.

Mechanical only. It detects a settled `checkpoint-<n>`, runs the screen binding on it through
harness/bakeoff.py (so the GPU claim goes through the allocator), and appends one line per
finished screen to `<state-dir>/series.txt`. Ranking is the caller's: this file never picks a
winner.

Throttle: `<state-dir>/max_screens` holds one integer, read live. **0 holds every launch** --
that is how a phase agent pauses screening (e.g. while a row's first episodes are in flight)
without stopping or deleting anything. Stop it by creating `<state-dir>/watcher.stop`.

NOTHING IS EVER DELETED: the state file is written to a sibling and renamed over, a refused
launch leaves its run folder in place, and the stop path only stops.

    python3 harness/screen_watcher.py --state-dir /tmp/franka-sonic/p12 \
        --checkpoints lane_a=/tmp/franka-sonic/lane_a/2026-09-10_finetune/out/checkpoints \
        --base-port lane_a=8800 --screens-per-checkpoint 2 --rollouts 20
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Set by main() from the command line; module-level so every function below reads them exactly
# as the /tmp originals did (this file is a transformation of those, not a rewrite).
STATE_DIR: Path = Path(".")
STOP: Path = Path(".")
STATE: Path = Path(".")
SERIES: Path = Path(".")
MAXFILE: Path = Path(".")
CKPTS: dict[str, Path] = {}
BASE_PORT: dict[str, int] = {}
RUN_ROOT = "/tmp/franka-sonic"
ROLLOUTS = 20
SCREENS_PER_CKPT = 2
MIN_GAP_S = 600         # the screens of one checkpoint are at least this far apart
SETTLE_S = 150          # a checkpoint dir must be this old before we trust the write finished
POLL_S = 60
NEEDED = ("config.json", "model-00002-of-00002.safetensors", "trainer_state.json")
DEFAULT_MAX_SCREENS = 2


def log(msg):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def load_state():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            log("state file unreadable; starting a fresh in-memory state (the file stays)")
    return {lane: {} for lane in CKPTS}


def save_state(state):
    tmp = STATE.with_suffix(".json.new")   # written, then renamed over: no deletion
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    os.replace(tmp, STATE)


def max_screens():
    try:
        return max(0, int(MAXFILE.read_text().strip()))
    except Exception:
        return DEFAULT_MAX_SCREENS


def settled(ck: Path) -> bool:
    if not ck.is_dir():
        return False
    for n in NEEDED:
        if not (ck / n).is_file():
            return False
    newest = max((ck / n).stat().st_mtime for n in NEEDED)
    return (time.time() - newest) > SETTLE_S


def launch(lane: str, step: int, screen: int, attempt: int):
    tag = f"scr_{lane[-1]}_{step}_s{screen}" + ("" if attempt == 1 else f"_retry{attempt}")
    logf = STATE_DIR / f"{tag}.log"
    port = BASE_PORT[lane] + (step // 2500) % 20 + 20 * (screen - 1) + 60 * (attempt - 1)
    cmd = ["python3", "harness/bakeoff.py", "run", lane, "eval", "--gpus", "1",
           "--checkpoint", str(CKPTS[lane] / f"checkpoint-{step}"),
           "--rollouts", str(ROLLOUTS), "--port", str(port)]
    env = dict(os.environ, BAKEOFF_RUN_ROOT=RUN_ROOT)
    with logf.open("a") as fh:
        fh.write(f"\n===== {tag} launched {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                 f"=====\n{' '.join(cmd)}\n")
        p = subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=fh,
                             stderr=subprocess.STDOUT, start_new_session=True)
    log(f"launched {tag} pid={p.pid} port={port} log={logf}")
    return {"tag": tag, "log": str(logf), "pid": p.pid, "port": port, "attempt": attempt,
            "screen": screen, "step": step, "status": "running", "launched_at": time.time(),
            "launched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run": None, "rc": None}


def alive(pid: int) -> bool:
    """True only if the pid is a *running* process (reap first, then read /proc state)."""
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
        if done == pid:
            return False
    except ChildProcessError:
        pass          # not our child — normal after a watcher restart
    except OSError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as fh:
            state = fh.read().rsplit(") ", 1)[1].split(" ", 1)[0]
    except (FileNotFoundError, IndexError, PermissionError):
        return False
    return state != "Z"


RUN_RE = re.compile(r"^\[bakeoff\] run folder (\S+)", re.M)
END_RE = re.compile(r"^\[bakeoff\] (OK|FAILED) rc=(-?\d+)\s+log: (\S+)/logs/run\.log", re.M)
ACQ_RE = re.compile(r"idle device\(s\), found|not enough idle|no idle device|could not acquire", re.I)


def read_launcher(entry):
    txt = Path(entry["log"]).read_text(errors="replace")
    marker = f"===== {entry['tag']} launched"
    if marker in txt:
        txt = txt[txt.rindex(marker):]
    m = RUN_RE.search(txt)
    if m:
        entry["run"] = m.group(1)
    e = END_RE.search(txt)
    if e:
        entry["rc"] = int(e.group(2))
        if entry["run"] and entry["run"] != e.group(3):
            log(f"ATTRIBUTION MISMATCH {entry['tag']}: start says {entry['run']}, "
                f"end says {e.group(3)} — keeping the end line")
        entry["run"] = e.group(3)
    entry["acquire_failed"] = bool(ACQ_RE.search(txt)) and entry.get("rc") not in (0,)
    return entry


def summarise(lane, step, screen, run):
    sys.path.insert(0, str(REPO / "harness" / "report"))
    from aggregate import load_eval  # noqa: E402
    r = load_eval(Path(run))          # load_eval takes the RUN folder; it appends out/eval itself
    ms = "/".join(f"{100 * x:.0f}" for x in r["milestone_rates"])
    line = (f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}  {lane}  step {step:>5}  "
            f"screen {screen}  {r['n_success']}/{r['n']}  progress {r['progress_mean']:.3f}  "
            f"milestones {ms}  run {run}")
    with SERIES.open("a") as fh:
        fh.write(line + "\n")
    log("SERIES " + line)
    return {"n": r["n"], "n_success": r["n_success"], "progress": r["progress_mean"],
            "milestone_rates": r["milestone_rates"]}


def configure(argv=None) -> None:
    """Populate the module-level config from the command line."""
    global STATE_DIR, STOP, STATE, SERIES, MAXFILE, CKPTS, BASE_PORT, RUN_ROOT
    global ROLLOUTS, SCREENS_PER_CKPT, MIN_GAP_S, SETTLE_S, POLL_S, DEFAULT_MAX_SCREENS
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--state-dir", required=True, type=Path,
                    help="where watcher_state.json, series.txt, max_screens, watcher.stop and "
                         "the per-screen launcher logs live")
    ap.add_argument("--checkpoints", action="append", required=True, metavar="LANE=DIR",
                    help="a fine-tune's checkpoints directory to watch; repeatable")
    ap.add_argument("--base-port", action="append", default=[], metavar="LANE=PORT",
                    help="first policy-server port for a lane (default 8800); repeatable")
    ap.add_argument("--run-root", default="/tmp/franka-sonic",
                    help="BAKEOFF_RUN_ROOT for the screens it launches (default /tmp/franka-sonic)")
    ap.add_argument("--rollouts", type=int, default=20, help="rollouts per screen (default 20)")
    ap.add_argument("--screens-per-checkpoint", type=int, default=2)
    ap.add_argument("--min-gap-s", type=int, default=600)
    ap.add_argument("--settle-s", type=int, default=150)
    ap.add_argument("--poll-s", type=int, default=60)
    ap.add_argument("--max-screens-default", type=int, default=2,
                    help="used only when <state-dir>/max_screens is absent or unreadable")
    a = ap.parse_args(argv)

    def pairs(items, cast):
        out = {}
        for it in items:
            if "=" not in it:
                raise SystemExit(f"[watcher] want LANE=VALUE, got {it!r}")
            k, _, v = it.partition("=")
            out[k.strip()] = cast(v.strip())
        return out

    STATE_DIR = a.state_dir.expanduser().resolve()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STOP = STATE_DIR / "watcher.stop"
    STATE = STATE_DIR / "watcher_state.json"
    SERIES = STATE_DIR / "series.txt"
    MAXFILE = STATE_DIR / "max_screens"
    CKPTS = pairs(a.checkpoints, lambda v: Path(v).expanduser())
    BASE_PORT = {lane: 8800 for lane in CKPTS} | pairs(a.base_port, int)
    RUN_ROOT = a.run_root
    ROLLOUTS = a.rollouts
    SCREENS_PER_CKPT = a.screens_per_checkpoint
    MIN_GAP_S = a.min_gap_s
    SETTLE_S = a.settle_s
    POLL_S = a.poll_s
    DEFAULT_MAX_SCREENS = a.max_screens_default
    missing = [lane for lane, d in CKPTS.items() if not d.parent.parent.is_dir()]
    if missing:
        log(f"note: no run folder yet for {missing} — waiting for it to appear")


def main():
    log(f"screen watcher up. roots={ {k: str(v) for k, v in CKPTS.items()} } "
        f"screens/ckpt={SCREENS_PER_CKPT} rollouts={ROLLOUTS} gap={MIN_GAP_S}s stop-file={STOP}")
    state = load_state()
    for lane in state:
        for key, ent in state[lane].items():
            if ent.get("status") == "done_unsummarised" and ent.get("run"):
                try:
                    ent["result"] = summarise(lane, ent["step"], ent["screen"], ent["run"])
                    ent["status"] = "done"
                except Exception as exc:                           # noqa: BLE001
                    log(f"re-summarise {ent['tag']} still failing: {exc!r}")
    save_state(state)
    while not STOP.exists():
        changed = False
        running = sum(1 for lane in state for s in state[lane]
                      if state[lane][s]["status"] == "running")
        for lane, root in CKPTS.items():
            if not root.is_dir():
                continue
            for ck in sorted(root.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])):
                step = int(ck.name.split("-")[1])
                for screen in range(1, SCREENS_PER_CKPT + 1):
                    key = f"{step}:{screen}"
                    ent = state[lane].get(key)
                    if ent is None:
                        if not settled(ck) or running >= max_screens():
                            continue
                        if screen > 1:
                            prev = state[lane].get(f"{step}:{screen - 1}")
                            if prev is None or (time.time() - prev["launched_at"]) < MIN_GAP_S:
                                continue          # the two screens must be >= 10 min apart
                        state[lane][key] = launch(lane, step, screen, 1)
                        running += 1
                        changed = True
                        continue
                    if ent["status"] != "running":
                        continue
                    ent = read_launcher(ent)
                    if alive(ent["pid"]):
                        continue
                    if ent.get("rc") == 0 and ent.get("run"):
                        try:
                            ent["result"] = summarise(lane, step, screen, ent["run"])
                            ent["status"] = "done"
                        except Exception as exc:                   # noqa: BLE001
                            log(f"{ent['tag']}: rc=0 but summarise failed: {exc!r}")
                            ent["status"] = "done_unsummarised"
                    elif ent.get("acquire_failed"):
                        log(f"{ent['tag']}: allocator refused (run folder {ent.get('run')} left "
                            f"in place), will retry when a device frees")
                        state[lane].pop(key)      # in-memory bookkeeping only; no file removed
                        running -= 1
                        changed = True
                        continue
                    elif ent["attempt"] < 2:
                        log(f"{ent['tag']}: rc={ent.get('rc')} — one retry")
                        state[lane][key] = launch(lane, step, screen, ent["attempt"] + 1)
                        changed = True
                        continue
                    else:
                        ent["status"] = "failed"
                        log(f"{ent['tag']}: failed twice (rc={ent.get('rc')}); left to the agent")
                    running -= 1
                    changed = True
        if changed:
            save_state(state)
        time.sleep(POLL_S)
    log("stop file present — watcher exiting (nothing was deleted)")
    save_state(state)
    return 0


if __name__ == "__main__":
    configure()
    sys.exit(main())
