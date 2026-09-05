#!/usr/bin/env python3
"""ONE entry point for every bake-off stage: it makes the run folder, stamps
provenance, holds the GPU claim, tees the log, and appends to plan/STATUS.md.

    python3 harness/bakeoff.py run shared p0.smoke
    python3 harness/bakeoff.py run lane_a finetune --gpus 2 --tiny
    python3 harness/bakeoff.py root          # run root, free GB, storage floor

Stdlib only, so plain `python3` runs it — the stages themselves shell out to
the right interpreter (`/isaac-sim/python.sh` for the sim,
`~/Isaac-GR00T/.venv/bin/python` for GR00T and the allocator).

Layout it creates (AGENTS.md rule e):

    ~/runs/franka-sonic/<lane>/<YYYY-MM-DD>_<stage>/
        README.md      what and why, written at launch
        cmd.sh         the exact commands, re-runnable by hand
        config.json    args, host, date, repo SHAs, CUDA_VISIBLE_DEVICES
        logs/run.log   everything the stage printed
        out/           artifacts (out/eval/ is evaluation.eval's own folder)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import shlex
import signal
import socket
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = Path(os.path.expanduser("~/runs/franka-sonic"))
# Felix, 2026-09-03: "if out of storage use instance local or stop". Lustre
# /home sits at 99 % on a shared 70 TB filesystem; the node overlay (/tmp) has
# ~11 TB but does NOT survive a pod restart. Never free space by deleting
# (AGENTS.md rule o) — fall back, and say so everywhere.
RUNS_FALLBACK = Path("/tmp/franka-sonic")
# Both roots, newest-last, for every helper that enumerates run folders.
RUN_ROOTS = (RUNS, RUNS_FALLBACK)
# 2026-09-03 03:30 UTC: home free fell 1200 -> 717 GB in 3.5 h (~140 GB ours).
# 300 GB was less than one hour of headroom, and an ENOSPC mid-checkpoint kills
# a 6 h attempt — so the floor is 600 GB. Precedence, highest first:
#   env DRIVER_MIN_HOME_FREE_GB  >  ~/runs/franka-sonic/min_home_free_gb  >  default
# The file exists so the floor can be retuned on the pod without a push.
MIN_HOME_FREE_GB_DEFAULT = 600.0
MIN_HOME_FREE_FILE = RUNS / "min_home_free_gb"
STATUS = REPO / "plan" / "STATUS.md"
GPUS_PY = REPO / "harness" / "gpus.py"

GR00T_PY = Path(os.path.expanduser("~/Isaac-GR00T/.venv/bin/python"))
PYSH = Path("/isaac-sim/python.sh")
USERBASE_FR3 = os.path.expanduser("~/env/pyuser-fr3")
FR3_REPO = Path(os.path.expanduser("~/code/franka-bimanual-isaac-sim"))


# --------------------------------------------------------------------------- pins
def load_pins(path: Path | None = None) -> dict[str, dict[str, str]]:
    """Minimal parser for env/pins.yaml (repos: / <name>: / key: value).

    Hand-rolled on purpose: this file must run under a bare `python3`, and the
    pod's system python has no PyYAML. The format is fixed by env/pins.yaml's
    own header comment.
    """
    path = path or REPO / "env" / "pins.yaml"
    repos: dict[str, dict[str, str]] = {}
    section = None
    name = None
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        key, _, value = line.strip().partition(":")
        value = value.strip()
        if indent == 0:
            section = key
            name = None
        elif indent == 2 and section == "repos":
            name = key
            repos[name] = {}
        elif indent >= 4 and name is not None:
            repos[name][key] = value
    return repos


def git_sha(path: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def repo_shas() -> dict[str, str | None]:
    shas: dict[str, str | None] = {"franka-sonic": git_sha(REPO)}
    for name, meta in load_pins().items():
        p = Path(os.path.expanduser(meta.get("path", "")))
        if p and p.exists():
            shas[name] = git_sha(p)
    return shas


# --------------------------------------------------------------------------- run root
def free_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(path).free / (1024 ** 3)
    except OSError:
        return 0.0


def _min_home_free_gb() -> tuple[float, str]:
    """Resolved floor and where it came from (env > override file > default)."""
    raw = os.environ.get("DRIVER_MIN_HOME_FREE_GB")
    if raw:
        try:
            return float(raw), "env DRIVER_MIN_HOME_FREE_GB"
        except ValueError:
            print(f"[bakeoff] ignoring unparsable DRIVER_MIN_HOME_FREE_GB={raw!r}",
                  file=sys.stderr, flush=True)
    try:
        return float(MIN_HOME_FREE_FILE.read_text().split()[0]), f"file {MIN_HOME_FREE_FILE}"
    except (OSError, ValueError, IndexError):
        pass
    return MIN_HOME_FREE_GB_DEFAULT, "default"


def min_home_free_gb() -> float:
    return _min_home_free_gb()[0]


def under_root(path: Path, root: Path) -> bool:
    """True when `path` lives inside `root` (used to spot instance-local runs)."""
    try:
        Path(os.path.abspath(str(path))).relative_to(Path(os.path.abspath(str(root))))
    except ValueError:
        return False
    return True


def run_root() -> tuple[Path, str | None]:
    """Where run folders go: ~ normally, instance-local /tmp when home is full.

    Returns (root, fallback_note). The note is stamped into config.json so a
    reader always knows an artifact is on non-persistent storage.
    """
    # P11 (2026-09-05): BAKEOFF_RUN_ROOT forces a root regardless of free space. Round 2's
    # trainers watched the shared home volume drain from OUTSIDE our control (2495 -> 899 GB
    # in one night) and had to be moved mid-run; a 20 000-step fine-tune keeping 8 x 34 GB
    # checkpoints is ~270 GB per lane, so P11 puts both trainers on instance-local /tmp from
    # step 0 instead of waiting for the floor to trip.
    forced = os.environ.get("BAKEOFF_RUN_ROOT")
    if forced:
        root = Path(os.path.expanduser(forced))
        root.mkdir(parents=True, exist_ok=True)
        note = None if root == RUNS else (
            f"BAKEOFF_RUN_ROOT={forced} (orchestrator-forced; instance-local /tmp is NOT "
            "persistent across pod restarts)")
        return root, note
    RUNS.mkdir(parents=True, exist_ok=True)
    free = free_gb(RUNS)
    floor = min_home_free_gb()
    if free >= floor:
        return RUNS, None
    RUNS_FALLBACK.mkdir(parents=True, exist_ok=True)
    note = (
        f"instance-local /tmp (NOT persistent across pod restarts) — "
        f"$HOME had {free:.0f} GB free, below the {floor:.0f} GB floor"
    )
    print(
        "\n"
        "[bakeoff] ############################################################\n"
        f"[bakeoff] # HOME IS FULL: {free:.0f} GB free (< {floor:.0f} GB).\n"
        f"[bakeoff] # Run folder goes to {RUNS_FALLBACK} — instance-local, NOT\n"
        "[bakeoff] # persistent across pod restarts. Record the real path in\n"
        "[bakeoff] # plan/STATUS.md and the WORKLOG. Never delete to free space\n"
        "[bakeoff] # (AGENTS.md rules k and o).\n"
        "[bakeoff] ############################################################\n",
        file=sys.stderr, flush=True,
    )
    return RUNS_FALLBACK, note


# --------------------------------------------------------------------------- run folder
class Run:
    def __init__(self, lane: str, stage: str, args: argparse.Namespace):
        self.lane, self.stage, self.args = lane, stage, args
        resume = getattr(args, "resume", None)
        if resume:
            # Continue an existing run folder (rule o: a partial run is progress, not clutter).
            self.dir = Path(os.path.expanduser(resume)).resolve()
            if not (self.dir / "out").is_dir():
                sys.exit(f"[bakeoff] --resume {self.dir} is not a run folder (no out/)")
            self.fallback = None
            if under_root(self.dir, RUNS_FALLBACK):
                self.fallback = "instance-local /tmp (NOT persistent across pod restarts)"
        else:
            day = _dt.date.today().isoformat()
            root, self.fallback = run_root()
            base = root / lane / f"{day}_{stage}"
            # mkdir is the arbiter, not exists(): two stages launched in the same second used to
            # compute the same suffix and one died on FileExistsError (P8, two ceiling-test lanes
            # in parallel, 2026-09-04 16:26 UTC). The loser simply takes the next suffix.
            d, n = base, 1
            while True:
                try:
                    d.mkdir(parents=True)
                    break
                except FileExistsError:
                    n += 1
                    d = Path(f"{base}-{n}")
            self.dir = d
            (self.dir / "logs").mkdir()
            (self.dir / "out").mkdir()
        self.log = self.dir / "logs" / "run.log"
        self.devices: str | None = None
        wanted = getattr(args, "steps", "all") or "all"
        self.steps: set[str] | None = None if wanted == "all" else set(wanted.split(","))

    # --- sub-step bookkeeping (a 6 h driver kill lands mid-stage; the next attempt resumes) ---
    def wants(self, step: str) -> bool:
        return self.steps is None or step in self.steps

    def done(self, step: str) -> bool:
        return (self.dir / "out" / f"{step}.done").exists()

    def mark_done(self, step: str, note: str = "") -> None:
        (self.dir / "out" / f"{step}.done").write_text(
            f"{_dt.datetime.now().astimezone().isoformat(timespec='seconds')} {note}\n"
        )

    def log_tail_has(self, pattern: str, path: Path | None = None, max_bytes: int = 4_000_000):
        """Regex-search the end of a log; scripts here exit 0 on failure (os._exit in a finally),
        so the DONE marker in the log is the only truthful exit status."""
        import re
        p = path or self.log
        if not p.exists():
            return None
        with p.open("rb") as fh:
            fh.seek(max(0, p.stat().st_size - max_bytes))
            text = fh.read().decode("utf-8", "replace")
        return re.search(pattern, text)

    def wait_pids(self, procs: dict, poll_s: float = 15.0) -> dict:
        """Foreground, bounded-by-process-lifetime wait for Popen children we started."""
        rcs = {}
        while len(rcs) < len(procs):
            for name, p in procs.items():
                if name not in rcs and p.poll() is not None:
                    rcs[name] = p.returncode
                    print(f"[bakeoff] {name} pid {p.pid} exited rc={p.returncode}", flush=True)
            if len(rcs) < len(procs):
                time.sleep(poll_s)
        return rcs

    def write_readme(self, what: str, why: str) -> None:
        (self.dir / "README.md").write_text(
            f"# {self.lane}/{self.stage} — {self.dir.name}\n\n"
            f"**What.** {what}\n\n"
            f"**Why.** {why}\n\n"
            f"**Launched.** {_dt.datetime.now().astimezone().isoformat(timespec='seconds')} "
            f"on {socket.gethostname()}\n\n"
            f"**Re-run.** `bash {self.dir / 'cmd.sh'}`\n\n"
            f"**Result.** (filled in by the run; see logs/run.log and config.json)\n"
        )

    def write_cmd(self, lines: list[str]) -> None:
        (self.dir / "cmd.sh").write_text(
            "#!/usr/bin/env bash\n"
            f"# {self.lane}/{self.stage} — exactly what harness/bakeoff.py ran.\n"
            "set -euo pipefail\n\n" + "\n".join(lines) + "\n"
        )
        (self.dir / "cmd.sh").chmod(0o755)

    def stamp(self) -> None:
        (self.dir / "config.json").write_text(
            json.dumps(
                {
                    "lane": self.lane,
                    "stage": self.stage,
                    # argparse's namespace carries the `fn=cmd_run` callback from
                    # set_defaults; drop it or json.dumps raises on the stamp.
                    "args": {k: v for k, v in vars(self.args).items() if k != "fn"},
                    "hostname": socket.gethostname(),
                    "date": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                    "cuda_visible_devices": self.devices,
                    "run_dir": str(self.dir),
                    "run_root_fallback": self.fallback,
                    "repo_shas": repo_shas(),
                },
                indent=2, sort_keys=True, default=str,
            ) + "\n"
        )

    def tee(self, cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> int:
        """Run cmd, streaming combined output to stdout and logs/run.log."""
        printable = " ".join(shlex.quote(c) for c in cmd)
        with self.log.open("a") as fh:
            header = f"\n=== {_dt.datetime.now().isoformat(timespec='seconds')} $ {printable}\n"
            fh.write(header)
            fh.flush()
            print(header.rstrip(), flush=True)
            p = subprocess.Popen(
                cmd, cwd=str(cwd) if cwd else None, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            assert p.stdout is not None
            for line in p.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                fh.write(line)
                fh.flush()  # a 2 h fine-tune otherwise shows nothing until 8 KB accumulate
            rc = p.wait()
            fh.write(f"=== exit {rc}\n")
        return rc


def sim_env(extra: dict | None = None) -> dict:
    env = dict(os.environ)
    env["PYTHONUSERBASE"] = USERBASE_FR3
    env.update(extra or {})
    return env


def wait_for_port(host: str, port: int, timeout: float = 180.0) -> bool:
    """Block until something accepts on host:port. Foreground, bounded (rule i)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(2.0)
            try:
                s.connect((host, port))
                return True
            except OSError:
                time.sleep(1.0)
    return False


def stop_proc(proc: subprocess.Popen, run: Run, label: str) -> None:
    """SIGTERM then SIGKILL a child WE started. Never by pattern (rule b).

    Uses Popen.wait, not `kill -0`: a killed child stays a zombie until its
    parent reaps it, and `kill -0` on a zombie succeeds — the first version
    of this helper reported "survived SIGKILL" for exactly that reason.
    """
    if proc.poll() is not None:
        print(f"[bakeoff] {label} pid {proc.pid} already exited (rc={proc.returncode})")
        (run.dir / "out" / f"{label}.pid").unlink(missing_ok=True)
        return
    for sig, wait in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            proc.send_signal(sig)
            proc.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            continue
        except ProcessLookupError:
            pass
        print(f"[bakeoff] {label} pid {proc.pid} stopped ({sig.name})")
        (run.dir / "out" / f"{label}.pid").unlink(missing_ok=True)
        return
    print(f"[bakeoff] WARNING: {label} pid {proc.pid} survived SIGKILL", file=sys.stderr)


# --------------------------------------------------------------------------- stages
def stage_p0_smoke(run: Run) -> int:
    """Stub openpi policy server + one 100-step Cosmos-client rollout.

    The whole sim-client -> wire -> action chunk -> env.step -> rubric path with
    no model anywhere: the half of gate P0 that proves the harness, not a policy.
    """
    port = run.args.port
    run.write_readme(
        what=(
            "P0 harness smoke: `evaluation.stub_policy_server --n-arms 2` (CPU hold/echo "
            f"policy on the openpi wire, port {port}) driven by `evaluation.eval --client "
            "Cosmos --embodiment franka_dual --rate 15 --replan-every 8 --rollouts 1 "
            "--max-steps 100 --no-splat --headless`."
        ),
        why=(
            "Proves the sim stack installed into ~/env/pyuser-fr3 boots Isaac, renders, "
            "speaks the policy wire and writes a scored run folder — before any model "
            "exists. Gate P0 checks for out/eval/eval_results.csv."
        ),
    )
    eval_out = run.dir / "out" / "eval"
    server_log = run.dir / "logs" / "stub_server.log"
    pidfile = run.dir / "out" / "stub_server.pid"

    server_cmd = [
        str(PYSH), "-m", "evaluation.stub_policy_server",
        "--n-arms", "2", "--port", str(port),
    ]
    eval_cmd = [
        str(PYSH), "-m", "evaluation.eval",
        "--run-folder", str(eval_out),
        "--client", "Cosmos",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--embodiment", "franka_dual",
        "--rate", "15",
        "--replan-every", "8",
        "--rollouts", "1",
        "--max-steps", "100",
        "--no-splat",
        "--headless",
    ]
    run.write_cmd([
        f"cd {FR3_REPO}",
        f"export PYTHONUSERBASE={USERBASE_FR3}",
        f"export CUDA_VISIBLE_DEVICES={run.devices or ''}",
        "",
        "# 1. stub policy server in the background, PID recorded for a by-PID kill",
        " ".join(shlex.quote(c) for c in server_cmd) + f" > {server_log} 2>&1 &",
        f"echo $! > {pidfile}",
        "",
        "# 2. the evaluator",
        " ".join(shlex.quote(c) for c in eval_cmd),
        "",
        "# 3. stop the server by the PID we recorded (never pkill -f)",
        f"kill $(cat {pidfile})",
    ])

    env = sim_env({"CUDA_VISIBLE_DEVICES": run.devices} if run.devices else None)
    print(f"[bakeoff] starting stub policy server on 127.0.0.1:{port}")
    with server_log.open("w") as fh:
        server = subprocess.Popen(
            server_cmd, cwd=str(FR3_REPO), env=env,
            stdout=fh, stderr=subprocess.STDOUT, text=True,
        )
    pidfile.write_text(f"{server.pid}\n")
    try:
        if not wait_for_port("127.0.0.1", port, timeout=180):
            print(
                f"[bakeoff] stub server never listened on {port} — see {server_log}",
                file=sys.stderr,
            )
            return 1
        print(f"[bakeoff] stub server up (pid {server.pid})")
        return run.tee(eval_cmd, cwd=FR3_REPO, env=env)
    finally:
        stop_proc(server, run, "stub_server")


RATE_HZ = 50  # rev 3c: recorder, datasets and eval all at 50 Hz (100 Hz physics, decimation 2)
MIMIC_TASK = "Isaac-Stack-Cube-DualFranka-IK-Abs-Mimic-v0"
# mimic/env_cfg.py's GEN_SPAWN_XY / GEN_SPAWN_YAW — the round-1 generation spawn range, and the
# defaults of --spawn-xy / --spawn-yaw so round-1 commands reproduce (P7 WP 7.0/7.1).
GEN_SPAWN_XY_DEFAULT = 0.06
GEN_SPAWN_YAW_DEFAULT = 0.5
ISAACLAB_ROOT = "/workspace/isaaclab"  # Isaac Lab 2.3.2 checkout the sim user-site was built from
DATA = REPO / "harness" / "data"
LANE_A = REPO / "harness" / "lane_a"


def _demo_env(run: Run) -> dict:
    extra = {
        "CUDA_VISIBLE_DEVICES": run.devices or "",
        "MIMIC_RATE_HZ": str(RATE_HZ),
        "ISAACLAB_ROOT": ISAACLAB_ROOT,
    }
    # P7 WP 7.1: generation-time spawn widening. Only harness/data/generate_handover.py's
    # Mimic-env patch reads these; the export/replay envs and the EVALUATION env ignore them.
    # The defaults equal mimic/env_cfg.py's own GEN_SPAWN_XY / GEN_SPAWN_YAW, so a round-1
    # command reproduces bit-for-bit.
    extra["MIMIC_SPAWN_XY"] = f"{getattr(run.args, 'spawn_xy', GEN_SPAWN_XY_DEFAULT):g}"
    extra["MIMIC_SPAWN_YAW"] = f"{getattr(run.args, 'spawn_yaw', GEN_SPAWN_YAW_DEFAULT):g}"
    arm_noise = getattr(run.args, "arm_noise_std", -1.0)
    if arm_noise is not None and arm_noise >= 0.0:
        extra["MIMIC_ARM_NOISE_STD"] = f"{arm_noise:g}"
    return sim_env(extra)


def stage_demos(run: Run) -> int:
    """WP 1.0 (P0's demo half): scripted 50 Hz sources -> auto-annotate -> MimicGen x80 ->
    video-backed export with the IK joint targets -> one-episode JointPos replay check.

    Sub-steps (--steps): sources, annotate, generate, export, replay. Each leaves an
    out/<step>.done marker so a killed run resumes with --resume <run>.
    """
    out = run.dir / "out"
    sources, annotated = out / "sources.hdf5", out / "sources_annotated.hdf5"
    fixed, export_dir = out / "sources_annotated_fixed.hdf5", out / "export"
    # Generation output: a fresh name per attempt (a failed attempt's empty file stays — rule o);
    # later steps read the path this step recorded in out/generated.path.
    gen_path_file = out / "generated.path"
    if gen_path_file.exists():
        generated = Path(gen_path_file.read_text().strip())
    else:
        generated, k = out / "generated.hdf5", 1
        while generated.exists() or generated.with_name(f"{generated.stem}_w0.hdf5").exists():
            k += 1
            generated = out / f"generated-{k}.hdf5"
    a = run.args
    n_sources, n_generated = a.n_sources, a.n_generated
    n_procs, n_shards, n_replay = a.n_procs, a.n_shards, a.n_replay
    spawn_xy, spawn_yaw = a.spawn_xy, a.spawn_yaw
    coverage_json = out / "coverage.json"
    # A re-drawn replay check gets its own pick file and its own eval folder: evaluation.eval
    # resumes a run folder that already holds rows, and a second draw is a different experiment.
    # Seed 0 keeps round 1's names.
    tag = "" if a.replay_seed == 0 else f"_s{a.replay_seed}"
    replay_pick = out / f"replay_episodes{tag}.json"
    replay_dir = out / f"replay_check{tag}"
    env = _demo_env(run)
    run.write_readme(
        what=(
            f"The demo set both lanes train on: {n_sources} scripted source demos at {RATE_HZ} Hz "
            f"(harness/data/scripted_source_demos.py, mirrored right-arm grasp for the angled rig), "
            f"auto-annotated, MimicGen-expanded to {n_generated} episodes ({n_procs} single-env workers "
            f"at {RATE_HZ} Hz via MIMIC_RATE_HZ) with the cube spawned in "
            f"+/-{spawn_xy:g} m / +/-{spawn_yaw:g} rad around the start slot"
            + (f" and arm reset noise std {a.arm_noise_std:g} rad" if a.arm_noise_std >= 0 else "")
            + f", exported in {n_shards} shards as video-backed "
            f"training-schema HDF5 with the differential-IK joint targets recorded per step, then "
            f"out/coverage.json (measured spawn distribution vs the evaluation range) and a "
            f"{n_replay}-episode replay through the JointPos env ({replay_dir.name})."
        ),
        why=(
            "No demo HDF5 existed when P1 started (gate p0 covers the environment half only). "
            "Rev 3c: one 50 Hz demo set, two action tables; lane A trains on absolute joint targets."
        ),
    )
    src_cmd = [str(PYSH), str(DATA / "scripted_source_demos.py"), "--headless",
               "--num_demos", str(n_sources), "--rate", str(RATE_HZ), "--dataset", str(sources)]
    ann_cmd = [str(PYSH), str(DATA / "annotate_sources.py"), "--task", MIMIC_TASK,
               "--input_file", str(sources), "--output_file", str(annotated), "--auto", "--headless"]
    fix_cmd = [str(GR00T_PY), str(DATA / "fix_subtask_signals.py"),
               "--input", str(annotated), "--output", str(fixed)]
    gen_cmd = [str(PYSH), str(DATA / "generate_parallel.py"), "--task", MIMIC_TASK,
               "--input", str(fixed), "--output", str(generated),
               "--total", str(n_generated), "--procs", str(n_procs),
               "--deadline_min", f"{a.gen_deadline_min:g}"]
    exp_cmds = {
        f"export_shard{i}": [str(PYSH), str(DATA / "export_generated_50hz.py"),
                             "--input", str(generated),
                             "--output", str(export_dir / f"demos_shard{i}.hdf5"),
                             "--rate", str(RATE_HZ), "--shard", str(i), "--num_shards", str(n_shards),
                             "--headless"]
        for i in range(n_shards)
    }
    cov_cmd = [str(GR00T_PY), str(DATA / "spawn_coverage.py"),
               "--export", str(export_dir / "*.hdf5"), "--output", str(coverage_json)]
    replay_cmd = [str(PYSH), str(LANE_A / "eval_oracle_a.py"),
                  "--demos", str(export_dir), "--run-folder", str(replay_dir),
                  "--rate", str(RATE_HZ), "--rollouts", str(n_replay),
                  "--max-steps", str(run.args.max_steps), "--no-splat", "--headless"]
    q = lambda c: " ".join(shlex.quote(x) for x in c)  # noqa: E731
    spawn_exports = (f"export MIMIC_SPAWN_XY={spawn_xy:g} MIMIC_SPAWN_YAW={spawn_yaw:g}"
                     + (f" MIMIC_ARM_NOISE_STD={a.arm_noise_std:g}" if a.arm_noise_std >= 0 else ""))
    run.write_cmd([
        f"cd {FR3_REPO}", f"export PYTHONUSERBASE={USERBASE_FR3}",
        f"export CUDA_VISIBLE_DEVICES={run.devices or ''}",
        f"export MIMIC_RATE_HZ={RATE_HZ} ISAACLAB_ROOT={ISAACLAB_ROOT}",
        f"{spawn_exports}   # P7 WP 7.1: generation-time spawn range (generate_handover.py only)", "",
        "# 1. sources (Isaac recorder format, success-only)", q(src_cmd), "",
        "# 2. auto-annotate subtask signals, then make left_placed MimicGen-parsable (0.20 m rig)",
        q(ann_cmd), q(fix_cmd), "",
        "# 3. MimicGen, process-level parallel", q(gen_cmd), "",
        "# 4. export (video-backed, half-res frames, IK joint targets) in parallel shards",
        *[q(c) + f" > {run.dir / 'logs' / (n + '.log')} 2>&1 &" for n, c in exp_cmds.items()],
        "wait", "",
        "# 5. measured spawn distribution vs the evaluation range (WP 7.1)", q(cov_cmd), "",
        f"# 6. replay check: {n_replay} generated episode(s) through the JointPos env. The indices"
        f" are drawn once (seed {a.replay_seed}) when the export finishes and recorded in"
        f" {replay_pick}.",
        q(replay_cmd) + f" --episode-indices \"$(python3 -c 'import json,sys;"
                        f' print(",".join(str(i) for i in json.load(open(sys.argv[1]))["indices"]))\''
                        f" {shlex.quote(str(replay_pick))})\"",
    ])

    if run.wants("sources") and not run.done("sources"):
        run.tee(src_cmd, cwd=FR3_REPO, env=env)
        m = run.log_tail_has(r"EXPERT_DONE: (\d+)/(\d+) source demos")
        if not m or int(m.group(1)) == 0:
            print("[bakeoff] sources: no EXPERT_DONE with saved > 0 in the log", file=sys.stderr)
            return 1
        run.mark_done("sources", f"{m.group(1)}/{m.group(2)} saved")
        print(f"[bakeoff] sources: {m.group(1)}/{m.group(2)} saved -> {sources}", flush=True)

    if run.wants("annotate") and not run.done("annotate"):
        run.tee(ann_cmd, cwd=FR3_REPO, env=env)
        if not annotated.exists():
            print(f"[bakeoff] annotate: {annotated} was not written", file=sys.stderr)
            return 1
        run.mark_done("annotate")

    if run.wants("fixsignals") and not run.done("fixsignals"):
        rc = run.tee(fix_cmd, cwd=REPO, env=dict(os.environ))
        if rc != 0 or not run.log_tail_has(r"FIX_SIGNALS_DONE"):
            print("[bakeoff] fixsignals failed", file=sys.stderr)
            return 1
        run.mark_done("fixsignals")

    if run.wants("generate") and not run.done("generate"):
        run.tee(gen_cmd, cwd=FR3_REPO, env=env)
        # The rate is -1 when no worker log carried a progress line (round 1 printed `nan` there,
        # which this regex must also survive — a 3 h generation must not fail on its own report).
        m = run.log_tail_has(r"PARALLEL_GEN_DONE: (\d+) episodes merged .*aggregate "
                             r"(\d+)/(\d+) = (-?[\d.]+|nan)% success")
        if not m or int(m.group(1)) == 0:
            print("[bakeoff] generate: no PARALLEL_GEN_DONE with episodes > 0", file=sys.stderr)
            return 1
        keep = {"episodes": int(m.group(1)), "successes": int(m.group(2)),
                "trials": int(m.group(3)), "keep_rate_pct": float(m.group(4))}
        (out / "keep_rate.json").write_text(json.dumps(keep, indent=2) + "\n")
        gen_path_file.write_text(f"{generated}\n")
        run.mark_done("generate", json.dumps(keep))
        print(f"[bakeoff] generate: {keep}", flush=True)

    if run.wants("export") and not run.done("export"):
        export_dir.mkdir(exist_ok=True)
        # One renderer per claimed device, round-robin. With one device (round 1: 4 shards on
        # `--gpus 1`) every shard gets that same device, exactly as before.
        devs = [d for d in (run.devices or "").split(",") if d.strip()]
        procs = {}
        for i, (name, cmd) in enumerate(exp_cmds.items()):
            log = run.dir / "logs" / f"{name}.log"
            shard_env = dict(env)
            if devs:
                shard_env["CUDA_VISIBLE_DEVICES"] = devs[i % len(devs)]
            with log.open("w") as fh:
                procs[name] = subprocess.Popen(cmd, cwd=str(FR3_REPO), env=shard_env,
                                               stdout=fh, stderr=subprocess.STDOUT, text=True)
            (out / f"{name}.pid").write_text(f"{procs[name].pid}\n")
            print(f"[bakeoff] {name} pid {procs[name].pid} gpu "
                  f"{shard_env.get('CUDA_VISIBLE_DEVICES', 'inherited')} -> {log}", flush=True)
        run.wait_pids(procs)
        total = 0
        for name in exp_cmds:
            m = run.log_tail_has(r"EXPORT_DONE: (\d+) episodes", run.dir / "logs" / f"{name}.log")
            if not m:
                print(f"[bakeoff] {name}: no EXPORT_DONE in its log", file=sys.stderr)
                return 1
            total += int(m.group(1))
        run.mark_done("export", f"{total} episodes in {n_shards} shards")
        print(f"[bakeoff] export: {total} episodes -> {export_dir}", flush=True)

    if run.wants("coverage") and not run.done("coverage"):
        # `env` (not os.environ) so coverage.json records the MIMIC_SPAWN_* the generation ran with.
        rc = run.tee(cov_cmd, cwd=REPO, env=env)
        if rc != 0 or not run.log_tail_has(r"COVERAGE_DONE"):
            print("[bakeoff] coverage failed", file=sys.stderr)
            return 1
        cov = json.loads(coverage_json.read_text())
        run.mark_done("coverage", f"covers_eval={cov['covers_eval']} over {cov['episodes']} episodes")

    if run.wants("replay") and not run.done("replay"):
        # WP 7.4: n_replay RANDOM episodes, not the first n. eval_oracle_a.py's table holds the
        # replay_success episodes only, so the draw is over that count (coverage.json measured it).
        if not replay_pick.exists():
            if coverage_json.exists():
                n_pool = json.loads(coverage_json.read_text())["episodes_replay_success"]
            else:
                m = run.log_tail_has(r"EXPORT_DONE: (\d+) episodes",
                                     run.dir / "logs" / "export_shard0.log")
                n_pool = int(m.group(1)) if m else n_replay
            k = min(n_replay, max(1, n_pool))
            idx = sorted(random.Random(a.replay_seed).sample(range(max(1, n_pool)), k))
            replay_pick.write_text(json.dumps({"seed": a.replay_seed, "pool": n_pool,
                                               "indices": idx}, indent=2) + "\n")
        idx = json.loads(replay_pick.read_text())["indices"]
        rc = run.tee(replay_cmd + ["--episode-indices", ",".join(str(i) for i in idx)],
                     cwd=FR3_REPO, env=env)
        csv = replay_dir / "eval_results.csv"
        rows = [l for l in csv.read_text().splitlines()[1:] if l.strip()] if csv.exists() else []
        ok = len(rows) >= len(idx) and all(",True," in l for l in rows[:len(idx)])
        if rc != 0 or not ok:
            print(f"[bakeoff] replay check failed (rc={rc}, {sum(',True,' in l for l in rows)}/"
                  f"{len(idx)} successes on episodes {idx}) — see {csv}", file=sys.stderr)
            return 1
        run.mark_done("replay", f"handover_success=True on {len(idx)}/{len(idx)} random episodes "
                                f"{idx} (seed {a.replay_seed}) on the JointPos env")
    return 0


def _need(run: Run, attr: str, hint: str) -> Path:
    v = getattr(run.args, attr, None)
    if not v:
        sys.exit(f"[bakeoff] --{attr} is required for {run.lane}/{run.stage} ({hint})")
    p = Path(os.path.expanduser(v)).resolve()
    if not p.exists():
        sys.exit(f"[bakeoff] --{attr} {p} does not exist")
    return p


def _find(pattern: str) -> list[Path]:
    """Every run folder matching `pattern`, across BOTH roots (home + /tmp)."""
    hits: list[Path] = []
    for root in RUN_ROOTS:
        hits += sorted(root.glob(pattern))
    return hits


def _latest(pattern: str) -> Path | None:
    hits = _find(pattern)
    return hits[-1] if hits else None


def stage_dataset(run: Run) -> int:
    """WP 1.1: export HDF5 shards -> GR00T v2 (LeRobot v2.1 layout) + stats + loader validation.
    CPU only (no GPU claim). Reads --demos (default: newest shared/*_demos*)."""
    demos = Path(run.args.demos) if run.args.demos else _latest("shared/*_demos*")
    demos = _need(run, "demos", "the shared/*_demos run folder") if run.args.demos else demos
    if demos is None or not (demos / "out" / "export").is_dir():
        sys.exit(f"[bakeoff] no export shards under {demos}/out/export")
    # P11 (2026-09-05): a union demo folder keeps each source set in its own subdirectory
    # (out/export/<set>/demos_shard*.hdf5 with the video dirs beside them) so shard basenames
    # stay distinct — label_tokens keys clips by (shard basename, demo name). Hand the
    # converter every level that has shards; it refuses a pattern that matches nothing.
    export_dir = demos / "out" / "export"
    patterns = [str(export_dir / pat) for pat in ("*.hdf5", "*/*.hdf5") if any(export_dir.glob(pat))]
    if not patterns:
        sys.exit(f"[bakeoff] no *.hdf5 or */*.hdf5 under {export_dir}")
    out = run.dir / "out" / "gr00t_v2"
    cmd = [str(GR00T_PY), str(DATA / "convert_hdf5_to_gr00t_v2.py"),
           "--input", *patterns, "--output", str(out),
           "--fps", str(RATE_HZ), "--task", "hand the block from the left arm to the right",
           "--joint-label", "ik_target_clamped", "--video-size", "640x360",
           "--validate-max-episodes", "5",
           "--modality-config-path", str(LANE_A / "modality_config_dual_fr3.py")]
    run.write_readme(
        what=(f"HDF5 -> GR00T v2 conversion of {demos} (harness/data/convert_hdf5_to_gr00t_v2.py, "
              f"written here because lerobot is not on the pod and NVIDIA's convert_v3_to_v2.py needs "
              f"it), then gr00t/data/stats.py and a LeRobotEpisodeLoader validation."),
        why="P1 WP 1.1 — the dataset both the fine-tune and the open-loop eval read; gate p1 check 1.",
    )
    run.write_cmd([f"cd {REPO}", " ".join(shlex.quote(c) for c in cmd)])
    rc = run.tee(cmd, cwd=REPO, env=dict(os.environ))
    if rc != 0 or not (out / "meta" / "modality.json").exists():
        return rc or 1
    return 0


def _modality_config(lane: str) -> Path:
    """Lane A: joint targets; lane B: SONIC token + grippers. Same file everywhere else (rev 3c)."""
    if lane == "lane_b":
        return REPO / "harness" / "lane_b" / "modality_config_dual_fr3_sonic.py"
    return LANE_A / "modality_config_dual_fr3.py"


def _finetune_cmd(dataset: Path, out: Path, num_gpus: int, lane: str = "lane_a",
                  train_steps: int = 2000, save_steps: int = 500,
                  save_total_limit: int = 5, master_port: int = 29517,
                  base_model: str = "nvidia/GR00T-N1.7-3B") -> list[str]:
    """The fine-tune command. Round 1's numbers are the defaults, so every round-1
    invocation reproduces token-for-token; P9 raises the budget through --train-steps
    /--save-steps/--save-total-limit (WP 9.0). NOTE the CLI's own --max-steps means the
    *evaluation* horizon, which is why the trainer's step budget is a different flag here.
    Only --dataset-path, --modality-config-path and --output-dir may ever differ between
    the two lanes. P11 (2026-09-05) adds --base-model-path: each lane warm-starts from its
    own round-2 checkpoint-20000 (--init-from) — the same treatment for both lanes."""
    launch = [str(GR00T_PY), "-m", "gr00t.experiment.launch_finetune"]
    if num_gpus > 1:
        launch = [str(Path(GR00T_PY).parent / "torchrun"), f"--nproc_per_node={num_gpus}",
                  f"--master_port={master_port}", "-m", "gr00t.experiment.launch_finetune"]
    return launch + [
        "--base-model-path", base_model,
        "--dataset-path", str(dataset),
        "--embodiment-tag", "NEW_EMBODIMENT",
        "--modality-config-path", str(_modality_config(lane)),
        "--num-gpus", str(num_gpus),
        "--max-steps", str(train_steps), "--save-steps", str(save_steps), "--global-batch-size", "32",
        "--color-jitter-params", "brightness", "0.3", "contrast", "0.4", "saturation", "0.5", "hue", "0.08",
        "--dataloader-num-workers", "4",
        "--save-total-limit", str(save_total_limit),
        "--no-use-wandb",
        "--output-dir", str(out),
    ]


def _dataset_dir(run: Run) -> Path:
    """Lane A reads */*_dataset*/out/gr00t_v2; lane B reads lane_b/*_label_tokens*/out/gr00t_v2_sonic."""
    if run.lane == "lane_b":
        ds_run = Path(run.args.dataset).expanduser() if run.args.dataset else \
            latest_run("lane_b", "label_tokens", "out/gr00t_v2_sonic/meta/modality.json")
        sub = "gr00t_v2_sonic"
    else:
        ds_run = Path(run.args.dataset).expanduser() if run.args.dataset else _latest("*/*_dataset*")
        sub = "gr00t_v2"
    if ds_run is None or not (ds_run / "out" / sub / "meta" / "modality.json").exists():
        sys.exit(f"[bakeoff] no {sub} dataset under {ds_run}")
    return ds_run / "out" / sub


def stage_finetune(run: Run) -> int:
    """WP 1.3 / WP 3.3: GR00T N1.7 fine-tune as NEW_EMBODIMENT (rev 3c hyperparameters, identical
    for both lanes — only --dataset-path and --modality-config-path differ)."""
    dataset = _dataset_dir(run)
    out = run.dir / "out" / "checkpoints"
    n_gpus = len((run.devices or "").split(",")) if run.devices else 1
    train_steps = int(getattr(run.args, "train_steps", 2000))
    save_steps = int(getattr(run.args, "save_steps", 500))
    save_total_limit = int(getattr(run.args, "save_total_limit", 5))
    # torchrun's rendezvous port. P9 trains both lanes concurrently on one node, and two
    # torchrun jobs cannot bind the same port; it is a launcher detail, not a hyperparameter.
    master_port = 29517 if run.lane != "lane_b" else 29518
    # P11: warm start from a GR00T checkpoint dir (weights + processor; the optimizer and the
    # LR schedule start fresh — this is NOT --resume-from-checkpoint, which is a bool that
    # continues the run inside --output-dir).
    init_from = getattr(run.args, "init_from", None)
    base_model = os.path.expanduser(init_from) if init_from else "nvidia/GR00T-N1.7-3B"
    if init_from and not (Path(base_model) / "config.json").is_file():
        sys.exit(f"[bakeoff] --init-from {base_model} has no config.json (need a GR00T checkpoint dir)")
    cmd = _finetune_cmd(dataset, out, n_gpus, run.lane, train_steps, save_steps,
                        save_total_limit, master_port, base_model)
    wp = "P3 WP 3.3 — lane B's policy; gate p3" if run.lane == "lane_b" else "P1 WP 1.3 — lane A's policy; gate p1"
    run.write_readme(
        what=(f"gr00t.experiment.launch_finetune on {dataset} with {n_gpus} GPU(s): "
              f"--max-steps {train_steps} --save-steps {save_steps} --global-batch-size 32, "
              "SONIC-tutorial colour jitter, "
              f"4 dataloader workers, default LR/optimizer, no LoRA; modality config {_modality_config(run.lane)}; base model {base_model}."),
        why=f"{wp} reads out/checkpoints/checkpoint-{train_steps}.",
    )
    run.write_cmd([f"cd {os.path.expanduser('~/Isaac-GR00T')}",
                   f"export CUDA_VISIBLE_DEVICES={run.devices or ''}",
                   " ".join(shlex.quote(c) for c in cmd)])
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = run.devices or ""
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    # No CUDA toolkit on the pod: DeepSpeed's import-time `nvcc -V` probe needs a CUDA_HOME.
    # harness/env/cuda_home_stub/bin/nvcc answers it; nothing gets compiled (adamw_torch, ZeRO-2).
    env["CUDA_HOME"] = str(REPO / "harness" / "env" / "cuda_home_stub")
    # NCCL between two GPUs in this container faults with "illegal memory access" over both the
    # P2P and the shared-memory transport (measured 2026-09-03 with a 2-rank all_reduce on 5,7);
    # the socket transport works. Slower per step, but the only 2-rank path that runs here.
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_SHM_DISABLE", "1")
    # P9 2026-09-04: 2 ranks is the only rank count that works here. At 4 ranks this fine-tune
    # dies with "illegal memory access" inside the NCCL watchdog at the first collective after
    # the config save, ~40 s in, in both lanes and in two independent attempts. It is NOT the
    # transport: a 4-rank 256 MiB all_reduce on the same four devices completes in all four
    # configurations swept (sockets baseline, +NVLS/CollNet off, +Ring/Simple, +lo only) at
    # ~100 ms/iter. The difference against the probe is that gr00t's
    # _init_distributed_process_group binds eagerly (init_process_group(device_id=cuda:LOCAL_RANK)),
    # which is the next thing to test if a later round wants >2 ranks. Not pursued in P9 because
    # 2 ranks per lane also leaves four devices free for the concurrent screening WP 9.2 needs.
    rc = run.tee(cmd, cwd=Path(os.path.expanduser("~/Isaac-GR00T")), env=env)
    if rc != 0:
        return rc
    if not (out / f"checkpoint-{train_steps}").is_dir():
        print(f"[bakeoff] finetune exited 0 but {out}/checkpoint-{train_steps} is missing", file=sys.stderr)
        return 1
    return 0


def stage_open_loop(run: Run) -> int:
    """WP 1.4: open_loop_eval.py on checkpoints 500/1000/1500/2000 -> MSE trend."""
    ds_run = Path(run.args.dataset) if run.args.dataset else _latest("*/*_dataset*")
    dataset = ds_run / "out" / "gr00t_v2"
    ckpt_root = Path(run.args.checkpoint).resolve() if run.args.checkpoint else None
    if ckpt_root is None:
        ft = _latest("lane_a/*_finetune*")
        ckpt_root = ft / "out" / "checkpoints" if ft else None
    if ckpt_root is None or not ckpt_root.is_dir():
        sys.exit("[bakeoff] --checkpoint <…/out/checkpoints> is required")
    if ckpt_root.name.startswith("checkpoint-"):
        ckpt_root = ckpt_root.parent
    steps = [500, 1000, 1500, 2000]
    run.write_readme(
        what=f"gr00t/eval/open_loop_eval.py on {ckpt_root}/checkpoint-{{{','.join(map(str, steps))}}} "
             f"against {dataset} (traj 0 1 2, horizon 40, 600 steps).",
        why="P1 WP 1.4 — a falling MSE trend is the sanity check that the fine-tune learned anything.",
    )
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = run.devices or ""
    results = {}
    lines = []
    for s in steps:
        ck = ckpt_root / f"checkpoint-{s}"
        if not ck.is_dir():
            print(f"[bakeoff] {ck} missing — skipped", flush=True)
            continue
        cmd = [str(GR00T_PY), "gr00t/eval/open_loop_eval.py", "--dataset-path", str(dataset),
               "--embodiment-tag", "NEW_EMBODIMENT", "--model-path", str(ck),
               "--traj-ids", "0", "1", "2", "--action-horizon", "40", "--steps", "600",
               "--save-plot-path", str(run.dir / "out" / f"open_loop_ckpt{s}.jpeg")]
        lines.append(" ".join(shlex.quote(c) for c in cmd))
        before = run.log.stat().st_size if run.log.exists() else 0
        run.tee(cmd, cwd=Path(os.path.expanduser("~/Isaac-GR00T")), env=env)
        # parse only THIS checkpoint's section of the log (the first match would be ckpt-500's)
        import re
        with run.log.open("rb") as fh:
            fh.seek(before)
            section = fh.read().decode("utf-8", "replace")
        mse = re.findall(r"Average MSE across all trajs: ([\d.eE+-]+)", section)
        mae = re.findall(r"Average MAE across all trajs: ([\d.eE+-]+)", section)
        per = re.findall(r"MSE for trajectory (\d+): ([\d.eE+-]+), MAE: ([\d.eE+-]+)", section)
        results[str(s)] = {"mse": float(mse[-1]) if mse else None,
                           "mae": float(mae[-1]) if mae else None,
                           "per_traj": {t: {"mse": float(a), "mae": float(b)} for t, a, b in per}}
        print(f"[bakeoff] checkpoint-{s}: {results[str(s)]}", flush=True)
    run.write_cmd([f"cd {os.path.expanduser('~/Isaac-GR00T')}", f"export CUDA_VISIBLE_DEVICES={run.devices or ''}", *lines])
    (run.dir / "out" / "open_loop_eval.json").write_text(json.dumps(
        {"dataset": str(dataset), "checkpoints": str(ckpt_root), "traj_ids": [0, 1, 2],
         "action_horizon": 40, "steps": 600, "results": results}, indent=2) + "\n")
    return 0 if results and all(v["mse"] is not None for v in results.values()) else 1


def _eval_cmd(run: Run, eval_out: Path, client: str, extra: list[str]) -> list[str]:
    return [str(PYSH), "-m", "evaluation.eval", "--run-folder", str(eval_out), "--client", client,
            "--task", "Isaac-Stack-Cube-DualFranka-JointPos-v0", "--embodiment", "franka_dual",
            "--rate", str(RATE_HZ), "--replan-every", "20", "--rollouts", str(run.args.rollouts),
            "--max-steps", str(run.args.max_steps), "--no-splat", "--headless", *extra]


def stage_eval(run: Run) -> int:
    """WP 1.5/1.6 (lane A) / WP 3.4 (lane B): policy server (ZmqAct wire, chunk 40, replan 20) +
    evaluation.eval, 20 rollouts on the JointPos env. The sim side is byte-identical for both
    lanes; lane B's server carries the SONIC decoder ONNX inside. 2 GPUs: server on the first,
    Isaac on the second (or both on one)."""
    ckpt = _need(run, "checkpoint", "…/out/checkpoints/checkpoint-2000")
    devs = (run.devices or "").split(",")
    server_dev, sim_dev = (devs[0], devs[1]) if len(devs) > 1 else (devs[0], devs[0])
    port = run.args.port
    eval_out = run.dir / "out" / "eval"
    server_log = run.dir / "logs" / "server.log"
    pidfile = run.dir / "out" / "server.pid"
    if run.lane == "lane_b":
        exp = _export_run(run)
        server_py = LANE_B / "serve_gr00t_sonic_joint.py"
        extra = ["--decoder-onnx", str(exp / "out" / "model_decoder.onnx"),
                 "--encoder-onnx", str(exp / "out" / "model_encoder.onnx")]
        wp = "P3 WP 3.4 — lane B's success rate; gate p3 wants >= 20 episodes in out/eval/eval_results.csv."
    else:
        server_py = LANE_A / "serve_gr00t_joint.py"
        extra = []
        wp = "P1 WP 1.6 — lane A's success rate; gate p1 wants >= 20 episodes in out/eval/eval_results.csv."
    server_cmd = [str(GR00T_PY), str(server_py), "--model-path", str(ckpt),
                  "--embodiment-tag", "NEW_EMBODIMENT",
                  "--modality-config-path", str(_modality_config(run.lane)),
                  "--host", "127.0.0.1", "--port", str(port), "--replan-every", "20",
                  "--image-size", "640x360", "--device", "cuda", *extra]
    eval_cmd = _eval_cmd(run, eval_out, "ZmqAct",
                         ["--endpoint", f"tcp://127.0.0.1:{port}", "--grip-threshold", "0.5"])
    run.write_readme(
        what=(f"{server_py.relative_to(REPO)} on {ckpt} (GPU {server_dev}, port {port}, 40-step "
              f"chunk replanned every 20 steps{'; SONIC decoder ONNX inside the server at 50 Hz' if run.lane == 'lane_b' else ''}) "
              f"driven by evaluation.eval --client ZmqAct on the "
              f"JointPos env (GPU {sim_dev}), {run.args.rollouts} rollouts at {RATE_HZ} Hz, "
              f"horizon {run.args.max_steps}."),
        why=wp,
    )
    q = lambda c: " ".join(shlex.quote(x) for x in c)  # noqa: E731
    run.write_cmd([f"cd {FR3_REPO}", f"export PYTHONUSERBASE={USERBASE_FR3}", "",
                   f"CUDA_VISIBLE_DEVICES={server_dev} {q(server_cmd)} > {server_log} 2>&1 &",
                   f"echo $! > {pidfile}", "",
                   f"CUDA_VISIBLE_DEVICES={sim_dev} {q(eval_cmd)}", "", f"kill $(cat {pidfile})"])
    senv = dict(os.environ)
    senv["CUDA_VISIBLE_DEVICES"] = server_dev
    with server_log.open("w") as fh:
        server = subprocess.Popen(server_cmd, cwd=str(REPO), env=senv,
                                  stdout=fh, stderr=subprocess.STDOUT, text=True)
    pidfile.write_text(f"{server.pid}\n")
    try:
        deadline = time.time() + 900  # model load; bounded (rule i)
        while time.time() < deadline and server.poll() is None:
            if "SERVER_READY" in server_log.read_text(errors="replace"):
                break
            time.sleep(5)
        if server.poll() is not None or not wait_for_port("127.0.0.1", port, timeout=60):
            print(f"[bakeoff] policy server did not come up — see {server_log}", file=sys.stderr)
            return 1
        print(f"[bakeoff] policy server up (pid {server.pid}) on {port}", flush=True)
        env = sim_env({"CUDA_VISIBLE_DEVICES": sim_dev})
        return run.tee(eval_cmd, cwd=FR3_REPO, env=env)
    finally:
        stop_proc(server, run, "server")


def stage_oracle_a(run: Run) -> int:
    """WP 1.7: the recorded joint targets on the recorded cube spawns through evaluation.eval."""
    demos = Path(run.args.demos) if run.args.demos else _latest("shared/*_demos*")
    export_dir = demos / "out" / "export" if demos else None
    if export_dir is None or not export_dir.is_dir():
        sys.exit(f"[bakeoff] no export shards under {demos}")
    eval_out = run.dir / "out" / "eval"
    cmd = [str(PYSH), str(LANE_A / "eval_oracle_a.py"), "--demos", str(export_dir),
           "--run-folder", str(eval_out), "--rate", str(RATE_HZ), "--replan-every", "20",
           "--rollouts", str(run.args.rollouts), "--max-steps", str(run.args.max_steps),
           "--no-splat", "--headless"]
    offset = float(getattr(run.args, "left_j1_offset_rad", 0.0) or 0.0)
    tol = ""
    if offset:
        # P5 tolerance test (lane_b/oracle_a_tol): the same replay with the left joint-1 targets
        # shifted by a constant; the offset at which the A-oracle stops succeeding is the grasp
        # precision the task demands of lane B's decoder.
        cmd += ["--left-j1-offset-rad", str(offset)]
        tol = f" TOLERANCE TEST: left joint-1 targets offset by {offset:+.3f} rad on every row."
    run.write_readme(
        what=(f"harness/lane_a/eval_oracle_a.py over {export_dir}: episode k's recorded IK joint targets "
              f"+ binary gripper replayed on episode k's recorded cube spawn through evaluation.eval "
              f"(JointPos env variant with a table-driven spawn), {run.args.rollouts} rollouts at {RATE_HZ} Hz.{tol}"),
        why=("P5 WP 5.3 — how much lateral hand error the handover grasp tolerates (orchestrator note 19:45 2b)."
             if offset else
             "P1 WP 1.7 — the A-oracle calibrates lane A: ≈100 % by construction, else the dataset/replay path is wrong."),
    )
    run.write_cmd([f"cd {FR3_REPO}", f"export PYTHONUSERBASE={USERBASE_FR3}",
                   f"export CUDA_VISIBLE_DEVICES={run.devices or ''}",
                   " ".join(shlex.quote(c) for c in cmd)])
    return run.tee(cmd, cwd=FR3_REPO, env=sim_env({"CUDA_VISIBLE_DEVICES": run.devices or ""}))


# --------------------------------------------------------------------------- lane B (P2)
WBC_REPO = Path(os.path.expanduser("~/GR00T-WholeBodyControl"))
USERBASE_SONIC = os.path.expanduser("~/env/pyuser-sonic")
LANE_B = REPO / "harness" / "lane_b"


def sonic_env(devices: str | None) -> dict:
    """Environment for gear_sonic under /isaac-sim/python.sh (AGENTS.md rule d)."""
    env = dict(os.environ)
    env["PYTHONUSERBASE"] = USERBASE_SONIC
    env["CUDA_VISIBLE_DEVICES"] = devices or ""
    # gear_sonic's opt/wandb default is use_wandb=True + online; never publish from the pod.
    env["WANDB_MODE"] = "offline"
    # harness/lane_b holds reward terms (sonic_rewards.py) and the replay callback that the
    # experiment configs reference by module name; eval/export re-instantiate them too.
    env["PYTHONPATH"] = f"{LANE_B}:{env.get('PYTHONPATH', '')}".rstrip(":")
    return env


def latest_run(lane: str, stage_substr: str, must_have: str) -> Path | None:
    """Newest run folder (both roots) whose name contains `stage_substr` and holds `must_have`."""
    cands = []
    for root in RUN_ROOTS:
        d = root / lane
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if stage_substr in p.name and (p / must_have).exists():
                cands.append(p)
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def _q(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def stage_sonic_rl(run: Run) -> int:
    """WP 2.4 / 2.6: SONIC PPO (train_agent_trl.py +exp=sonic_dual_fr3) on the dual-FR3
    embodiment. `--tiny` is the num_envs=1 smoke (3 PPO iterations). The full run is bounded
    by --hours of wall-clock: the trainer is stopped by its recorded PID and the run counts as
    OK when a last.pt exists (the model-save callback writes it every 50 iterations)."""
    motions = Path(run.args.motions).expanduser() if run.args.motions else None
    if motions is None:
        ml = latest_run("lane_b", "motion_lib", "out/motions")
        if ml is None:
            print("[bakeoff] no lane_b/*_motion_lib/out/motions found; pass --motions",
                  file=sys.stderr)
            return 1
        motions = ml / "out" / "motions"
    tiny = bool(run.args.tiny)
    num_envs = 1 if tiny else run.args.num_envs
    iters = 3 if tiny else run.args.iters
    hours = 0.5 if tiny else run.args.hours
    base_dir = run.dir / "out"
    exp = run.args.exp
    cmd = [
        str(PYSH), "gear_sonic/train_agent_trl.py", f"+exp={exp}",
        f"num_envs={num_envs}", "headless=True", f"base_dir={base_dir}",
        f"exp_var={'smoke' if tiny else 'rl'}",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file={motions}",
        f"algo.config.num_learning_iterations={iters}",
        "++callbacks.model_save.save_frequency=500",
        "use_wandb=false",
    ] + shlex.split(getattr(run.args, "sonic_overrides", "") or "")
    # P5 (2026-09-03): --checkpoint <model_step_*.pt|last.pt> warm-starts the policy from an
    # earlier run (gear_sonic's resume_checkpoint: weights only, fresh optimizer, iteration 0)
    # so a reward/termination variant does not pay the cold-start iterations again.
    init_ckpt = Path(run.args.checkpoint).expanduser() if run.args.checkpoint else None
    if init_ckpt is not None:
        if not init_ckpt.exists():
            print(f"[bakeoff] --checkpoint {init_ckpt} does not exist", file=sys.stderr)
            return 1
        cmd.append(f"checkpoint={init_ckpt}")
    run.write_readme(
        what=(f"SONIC PPO on the dual-FR3 embodiment: {_q(cmd)} — num_envs={num_envs}, "
              f"{iters} PPO iterations max, wall-clock cap {hours} h, motion library {motions}."),
        why=("P2 WP 2.6: the encoder/decoder pair lane B's GR00T fine-tune emits tokens for. "
             "Fixed-base dual arm, 14 DoF, robot-motion encoder only; one GPU (2 allocatable, "
             "NCCL P2P/SHM faults on this pod)." if not tiny else
             "P2 WP 2.4: num_envs=1 smoke — body names, overrides and the motion library must "
             "load and step before the long run."),
    )
    log = run.dir / "logs" / "train.log"
    pidfile = run.dir / "out" / "train.pid"
    run.write_cmd([
        f"cd {WBC_REPO}", f"export PYTHONUSERBASE={USERBASE_SONIC}",
        f"export CUDA_VISIBLE_DEVICES={run.devices or ''}", "",
        f"{_q(cmd)} > {log} 2>&1 &", f"echo $! > {pidfile}",
        f"# wall-clock cap {hours} h, then: kill $(cat {pidfile})",
    ])
    t0 = time.time()
    # /isaac-sim/python.sh is a bash wrapper that does NOT exec python: a signal to p.pid
    # alone would orphan the real trainer on the GPU. Own session -> kill the whole group.
    with log.open("w") as fh:
        p = subprocess.Popen(cmd, cwd=str(WBC_REPO), env=sonic_env(run.devices),
                             stdout=fh, stderr=subprocess.STDOUT, text=True,
                             start_new_session=True)
    pidfile.write_text(f"{p.pid}\n")
    (run.dir / "out" / "train.pgid").write_text(f"{os.getpgid(p.pid)}\n")
    print(f"[bakeoff] trainer wrapper pid {p.pid} (own process group), log {log}", flush=True)
    killed = False
    deadline = t0 + hours * 3600.0
    while p.poll() is None:
        if time.time() > deadline:
            print("[bakeoff] wall-clock cap reached — stopping the trainer's process group",
                  flush=True)
            pgid = os.getpgid(p.pid)
            for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
                try:
                    os.killpg(pgid, sig)
                    p.wait(timeout=wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
                except ProcessLookupError:
                    break
            killed = True
            break
        time.sleep(15)
    rc = p.returncode if p.returncode is not None else -1
    elapsed = time.time() - t0
    lasts = sorted(base_dir.glob("**/last.pt"), key=lambda q: q.stat().st_mtime)
    steps = sorted(base_dir.glob("**/model_step_*.pt"))
    last_pt = lasts[-1] if lasts else None
    exp_dir = last_pt.parent if last_pt else None
    ok = (rc == 0) if tiny else bool(last_pt) and (rc == 0 or killed)
    summary = {
        "rc": rc, "killed_by_wallclock": killed, "elapsed_s": round(elapsed, 1),
        "num_envs": num_envs, "iters_requested": iters, "hours_cap": hours,
        "motions": str(motions), "last_pt": str(last_pt) if last_pt else None,
        "experiment_dir": str(exp_dir) if exp_dir else None,
        "model_step_checkpoints": [str(s) for s in steps], "ok": ok,
    }
    (run.dir / "out" / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[bakeoff] sonic_rl summary: {json.dumps(summary)}", flush=True)
    return 0 if ok else (rc or 1)


def stage_export_onnx(run: Run) -> int:
    """WP 2.7a: eval_agent_trl.py +export_onnx_only=True on the RL checkpoint -> the
    encoder/decoder ONNX pair copied to out/model_encoder.onnx + out/model_decoder.onnx."""
    ckpt = Path(run.args.checkpoint).expanduser() if run.args.checkpoint else None
    if ckpt is None:
        rl = latest_run("lane_b", "sonic_rl", "out/train_summary.json")
        if rl is not None:
            s = json.loads((rl / "out" / "train_summary.json").read_text())
            ckpt = Path(s["last_pt"]) if s.get("last_pt") else None
    if ckpt is None or not ckpt.exists():
        print("[bakeoff] no checkpoint (pass --checkpoint <…/last.pt>)", file=sys.stderr)
        return 1
    # base_eval.yaml only knows `checkpoint`; every other key must be added with `++`
    # (eval_agent_trl merges the CLI config over the checkpoint's training config last).
    cmd = [str(PYSH), "gear_sonic/eval_agent_trl.py", f"checkpoint={ckpt}",
           "++export_onnx_only=True", "++num_envs=1", "++headless=True", "++use_wandb=false"
           ] + shlex.split(getattr(run.args, "sonic_overrides", "") or "")
    run.write_readme(
        what=f"ONNX export of the SONIC universal-token module from {ckpt}: {_q(cmd)}",
        why="P2 WP 2.7: gate p2 wants out/model_encoder.onnx + out/model_decoder.onnx; "
            "P3 labels the demos with the encoder and serves the decoder.",
    )
    run.write_cmd([f"cd {WBC_REPO}", f"export PYTHONUSERBASE={USERBASE_SONIC}",
                   f"export CUDA_VISIBLE_DEVICES={run.devices or ''}", "", _q(cmd)])
    rc = run.tee(cmd, cwd=WBC_REPO, env=sonic_env(run.devices))
    exported = ckpt.parent / "exported"
    found = {}
    for tag, dst in (("_encoder.onnx", "model_encoder.onnx"), ("_decoder.onnx", "model_decoder.onnx"),
                     ("_g1.onnx", "model_g1_pair.onnx")):
        srcs = sorted(exported.glob(f"model_step_*{tag}"))
        if srcs:
            shutil.copy2(srcs[-1], run.dir / "out" / dst)
            found[dst] = str(srcs[-1])
    for extra in ("model_config.yaml",):
        if (ckpt.parent / extra).exists():
            shutil.copy2(ckpt.parent / extra, run.dir / "out" / extra)
    (run.dir / "out" / "export_summary.json").write_text(
        json.dumps({"checkpoint": str(ckpt), "rc": rc, "copied": found}, indent=2) + "\n")
    ok = "model_encoder.onnx" in found and "model_decoder.onnx" in found
    print(f"[bakeoff] export: rc={rc} copied={found}", flush=True)
    return 0 if ok else (rc or 1)


def stage_decoder_replay(run: Run) -> int:
    """WP 2.7b: closed-loop replay of ONE demo clip through the trained encoder/decoder in
    the SONIC env (eval_agent_trl.py + harness/lane_b/replay_callback.py) ->
    out/replay.json with mean_joint_error_rad (gate p2, WARN above 0.1 rad)."""
    ckpt = Path(run.args.checkpoint).expanduser() if run.args.checkpoint else None
    if ckpt is None:
        rl = latest_run("lane_b", "sonic_rl", "out/train_summary.json")
        if rl is not None:
            s = json.loads((rl / "out" / "train_summary.json").read_text())
            ckpt = Path(s["last_pt"]) if s.get("last_pt") else None
    if ckpt is None or not ckpt.exists():
        print("[bakeoff] no checkpoint (pass --checkpoint <…/last.pt>)", file=sys.stderr)
        return 1
    motions = Path(run.args.motions).expanduser() if run.args.motions else None
    if motions is None:
        ml = latest_run("lane_b", "motion_lib", "out/motions")
        if ml is None:
            print("[bakeoff] no motion library found; pass --motions", file=sys.stderr)
            return 1
        motions = ml / "out" / "motions"
    # one ORIGINAL demo clip (no _M, no augmentation tag): the first by name
    clips = sorted(p for p in motions.glob("*.pkl")
                   if p.stem.count("_") == 2 and not p.stem.endswith("_M"))
    if not clips:
        clips = sorted(motions.glob("*.pkl"))
    clip = clips[0]
    clip_dir = run.dir / "out" / "clip"
    clip_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(clip, clip_dir / clip.name)
    out_json = run.dir / "out" / "replay.json"
    cmd = [
        str(PYSH), "gear_sonic/eval_agent_trl.py", f"checkpoint={ckpt}", "++headless=True",
        "++num_envs=1", "++use_wandb=false", "++use_encoder=g1", "++run_once=True",
        "++eval_callbacks=[replay]",
        "++callbacks.replay._target_=replay_callback.ReplayCallback",
        f"++callbacks.replay.out_json={out_json}", f"++callbacks.replay.clip={clip.stem}",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file={clip_dir}",
        "++manager_env.commands.motion.start_from_first_frame=True",
    ] + shlex.split(getattr(run.args, "sonic_overrides", "") or "")
    run.write_readme(
        what=f"Decoder replay of demo clip {clip.stem} through {ckpt} in the SONIC env: {_q(cmd)}",
        why="P2 WP 2.7: can the learned token space + decoder reproduce a handover demo? "
            "mean_joint_error_rad < 0.1 is the target; above it is a finding (risk 2), not a stop.",
    )
    env = sonic_env(run.devices)
    env["PYTHONPATH"] = f"{LANE_B}:{env.get('PYTHONPATH', '')}".rstrip(":")
    run.write_cmd([f"cd {WBC_REPO}", f"export PYTHONUSERBASE={USERBASE_SONIC}",
                   f"export PYTHONPATH={LANE_B}", f"export CUDA_VISIBLE_DEVICES={run.devices or ''}",
                   "", _q(cmd)])
    rc = run.tee(cmd, cwd=WBC_REPO, env=env)
    if out_json.exists():
        d = json.loads(out_json.read_text())
        print(f"[bakeoff] replay: mean_joint_error_rad={d.get('mean_joint_error_rad')} "
              f"measured={d.get('mean_measured_joint_error_rad')} frames={d.get('n_frames')} "
              f"time_out={d.get('ended_by_time_out')}", flush=True)
        return 0
    print(f"[bakeoff] replay wrote no replay.json (rc={rc})", file=sys.stderr)
    return rc or 1


# --------------------------------------------------------------------------- lane B (P3)
def _export_run(run: Run) -> Path:
    """The lane_b/*_export_onnx run folder holding model_encoder.onnx + model_decoder.onnx
    (--onnx, else the newest one — the P2 gate's newest-wins rule)."""
    p = Path(run.args.onnx).expanduser() if getattr(run.args, "onnx", None) else \
        latest_run("lane_b", "export_onnx", "out/model_decoder.onnx")
    if p is None or not (p / "out" / "model_encoder.onnx").exists() \
            or not (p / "out" / "model_decoder.onnx").exists():
        sys.exit(f"[bakeoff] no encoder/decoder ONNX pair under {p} (pass --onnx <lane_b/*_export_onnx run>)")
    return p


def _tokens_run(run: Run) -> Path:
    p = Path(run.args.tokens).expanduser() if getattr(run.args, "tokens", None) else \
        latest_run("lane_b", "label_tokens", "out/tokens/index.json")
    if p is None or not (p / "out" / "tokens" / "index.json").exists():
        sys.exit(f"[bakeoff] no token labels under {p} (pass --tokens <lane_b/*_label_tokens run>)")
    return p


def stage_label_tokens(run: Run) -> int:
    """WP 3.1 + 3.2: offline SONIC token labels for the demos and the lane-B dataset variant.

    Sub-steps (out/<step>.done; --steps selects, --resume continues):
      validate  GPU   one demo clip through the trained policy in the SONIC env with
                      harness/lane_b/dump_obs_callback.py -> out/validation/env_dump.npz
      obs       CPU   label_tokens.py obs   (gear_sonic's own command term, offline)
      encode    CPU   label_tokens.py encode (encoder ONNX -> out/tokens/episode_*.npz)
      check     CPU   label_tokens.py check  (offline vs env, ONNX vs env policy, proprio layout)
      dataset   CPU   make_sonic_dataset.py  (lane A frames + token action table -> out/gr00t_v2_sonic)
    """
    exp = _export_run(run)
    enc_onnx, dec_onnx = exp / "out" / "model_encoder.onnx", exp / "out" / "model_decoder.onnx"
    ckpt = None
    if (exp / "out" / "export_summary.json").exists():
        ckpt = json.loads((exp / "out" / "export_summary.json").read_text()).get("checkpoint")
    if run.args.checkpoint:
        ckpt = os.path.expanduser(run.args.checkpoint)
    motions = Path(run.args.motions).expanduser() if run.args.motions else None
    if motions is None:
        ml = latest_run("lane_b", "motion_lib", "out/motions")
        if ml is None:
            sys.exit("[bakeoff] no motion library found; pass --motions")
        motions = ml / "out" / "motions"
    manifest = motions.parent / "manifest.json"
    ds_run = Path(run.args.dataset).expanduser() if run.args.dataset else _latest("*/*_dataset*")
    if ds_run is None or not (ds_run / "out" / "gr00t_v2" / "meta" / "provenance.json").exists():
        sys.exit(f"[bakeoff] no lane A gr00t_v2 dataset (with provenance.json) under {ds_run}")
    dataset_a = ds_run / "out" / "gr00t_v2"
    out = run.dir / "out"
    # the ORIGINAL clips (no _M, no augmentation tag) in their own directory: the motion lib
    # loads a whole directory
    clips_dir = out / "clips_orig"
    clips_dir.mkdir(exist_ok=True)
    originals = sorted(p for p in motions.glob("*.pkl")
                       if p.stem.count("_") == 2 and not p.stem.endswith("_M"))
    # P8 WP 8.2: --max-episodes N labels only the FIRST N dataset episodes, so a per-checkpoint
    # ceiling test costs ~20 clips instead of the whole set. The obs step is the expensive one,
    # so the clip directory is narrowed to exactly the clips those N episodes resolve to
    # (dataset provenance -> motion-lib manifest, the same mapping label_tokens.py encode uses);
    # encode gets the flag it already has. N <= 0 (the default) labels everything, i.e. round 1.
    max_eps = int(getattr(run.args, "max_episodes", 0) or 0)
    if max_eps > 0:
        prov = json.loads((dataset_a / "meta" / "provenance.json").read_text())
        eps = sorted(int(k) for k in prov if k.isdigit() and isinstance(prov[k], dict))[:max_eps]
        man = json.loads(manifest.read_text())
        entries = man["clips"] if isinstance(man, dict) and "clips" in man else man
        clip_of = {(os.path.basename(e["source_shard"]), e["source_demo"]): e["name"]
                   for e in entries if e.get("aug") == "orig" and not e.get("mirrored")}
        want, missing = [], []
        for ep in eps:
            key = (os.path.basename(prov[str(ep)]["source_file"]), prov[str(ep)]["demo_name"])
            (want if key in clip_of else missing).append(clip_of.get(key, key))
        if missing:
            sys.exit(f"[bakeoff] --max-episodes {max_eps}: {len(missing)} of the first {max_eps} "
                     f"dataset episodes have no original clip in {manifest}: {missing[:5]}")
        keep = set(want)
        originals = [q for q in originals if q.stem in keep]
        print(f"[bakeoff] --max-episodes {max_eps}: labelling {len(originals)} clips "
              f"(episodes {eps[0]}..{eps[-1]}) of {len(keep)} requested", flush=True)
    for p in originals:
        if not (clips_dir / p.name).exists():
            shutil.copy2(p, clips_dir / p.name)
    val_dir = out / "validation"
    val_dir.mkdir(exist_ok=True)
    val_clip_dir = val_dir / "clip"
    val_clip_dir.mkdir(exist_ok=True)
    clip = originals[0]
    if not (val_clip_dir / clip.name).exists():
        shutil.copy2(clip, val_clip_dir / clip.name)
    dump_npz = val_dir / "env_dump.npz"

    validate_cmd = [
        str(PYSH), "gear_sonic/eval_agent_trl.py", f"checkpoint={ckpt}", "++headless=True",
        "++num_envs=1", "++use_wandb=false", "++use_encoder=g1", "++run_once=True",
        "++eval_callbacks=[dump]",
        "++callbacks.dump._target_=dump_obs_callback.DumpObsCallback",
        f"++callbacks.dump.out_npz={dump_npz}", f"++callbacks.dump.clip={clip.stem}",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file={val_clip_dir}",
        "++manager_env.commands.motion.start_from_first_frame=True",
    ]
    obs_cmd = [str(PYSH), str(LANE_B / "label_tokens.py"), "obs", "--clips", str(clips_dir),
               "--out", str(out / "encoder_obs")]
    encode_cmd = [str(GR00T_PY), str(LANE_B / "label_tokens.py"), "encode",
                  "--obs", str(out / "encoder_obs"), "--encoder", str(enc_onnx),
                  "--dataset", str(dataset_a), "--manifest", str(manifest), "--out", str(out / "tokens")] \
        + (["--max-episodes", str(max_eps)] if max_eps > 0 else [])
    check_cmd = [str(GR00T_PY), str(LANE_B / "label_tokens.py"), "check", "--dump", str(dump_npz),
                 "--obs", str(out / "encoder_obs"), "--encoder", str(enc_onnx), "--decoder", str(dec_onnx),
                 "--out", str(val_dir / "validation.json")]
    dataset_cmd = [str(GR00T_PY), str(LANE_B / "make_sonic_dataset.py"), "--source", str(dataset_a),
                   "--tokens", str(out / "tokens"), "--output", str(out / "gr00t_v2_sonic"),
                   "--modality-config-path", str(_modality_config("lane_b"))]
    run.write_readme(
        what=(f"Token labels for the {len(originals)} original demo clips of {motions} with the encoder "
              f"{enc_onnx} (checkpoint {ckpt}); validation replay of {clip.stem} in the SONIC env; "
              f"lane-B dataset variant of {dataset_a} (same frames, action = [token 64 | grips 2])."),
        why="P3 WP 3.1/3.2 — gate p3 reads out/gr00t_v2_sonic/meta/modality.json; the fine-tune reads the dataset.",
    )
    senv = sonic_env(run.devices)
    run.write_cmd([f"cd {WBC_REPO}", f"export PYTHONUSERBASE={USERBASE_SONIC}",
                   f"export PYTHONPATH={LANE_B}", f"export CUDA_VISIBLE_DEVICES={run.devices or ''}", "",
                   "# validate (GPU)", _q(validate_cmd), "", "# obs (CPU, sonic env)", _q(obs_cmd), "",
                   f"cd {REPO}", "# encode / check / dataset (CPU, GR00T venv)", _q(encode_cmd),
                   _q(check_cmd), _q(dataset_cmd)])
    steps = [
        ("validate", validate_cmd, WBC_REPO, senv, dump_npz),
        ("obs", obs_cmd, WBC_REPO, senv, out / "encoder_obs" / "obs_index.json"),
        ("encode", encode_cmd, REPO, dict(os.environ), out / "tokens" / "index.json"),
        ("check", check_cmd, REPO, dict(os.environ), val_dir / "validation.json"),
        ("dataset", dataset_cmd, REPO, dict(os.environ), out / "gr00t_v2_sonic" / "meta" / "modality.json"),
    ]
    for name, cmd, cwd, env, marker in steps:
        if not run.wants(name):
            continue
        if run.done(name):
            print(f"[bakeoff] step {name} already done", flush=True)
            continue
        if name == "validate" and not ckpt:
            print("[bakeoff] validate: no RL checkpoint known (pass --checkpoint); skipping", flush=True)
            continue
        rc = run.tee(cmd, cwd=cwd, env=env)
        if not marker.exists():
            print(f"[bakeoff] step {name} produced no {marker} (rc={rc})", file=sys.stderr)
            return rc or 1
        if name == "check" and rc != 0:
            print("[bakeoff] check reported MISMATCH — see validation.json; not marking done", file=sys.stderr)
            return rc
        run.mark_done(name, f"rc={rc}")
    return 0


def stage_oracle_b(run: Run) -> int:
    """WP 3.5: the encoder-labelled token stream of each recorded episode through the decoder
    (no VLA), on the episode's recorded cube spawn, through evaluation.eval -> lane B's ceiling."""
    exp = _export_run(run)
    tok = _tokens_run(run)
    demos = Path(run.args.demos) if run.args.demos else _latest("shared/*_demos*")
    export_dir = demos / "out" / "export" if demos else None
    if export_dir is None or not export_dir.is_dir():
        sys.exit(f"[bakeoff] no export shards under {demos}")
    eval_out = run.dir / "out" / "eval"
    cmd = [str(PYSH), str(LANE_B / "eval_oracle_b.py"), "--demos", str(export_dir),
           "--tokens", str(tok / "out" / "tokens"),
           "--decoder-onnx", str(exp / "out" / "model_decoder.onnx"),
           "--encoder-onnx", str(exp / "out" / "model_encoder.onnx"),
           "--run-folder", str(eval_out), "--rate", str(RATE_HZ), "--replan-every", "20",
           "--rollouts", str(run.args.rollouts), "--max-steps", str(run.args.max_steps),
           "--no-splat", "--headless"]
    run.write_readme(
        what=(f"harness/lane_b/eval_oracle_b.py: episode k's offline token labels ({tok}) streamed through "
              f"the SONIC decoder ONNX ({exp}) at {RATE_HZ} Hz, no VLA, on episode k's recorded cube spawn "
              f"({export_dir}), through evaluation.eval on the JointPos env variant with a table-driven "
              f"spawn; {run.args.rollouts} rollouts, horizon {run.args.max_steps}."),
        why="P3 WP 3.5 — the B-oracle is lane B's ceiling: low means the controller lost, not the VLA.",
    )
    run.write_cmd([f"cd {FR3_REPO}", f"export PYTHONUSERBASE={USERBASE_FR3}",
                   f"export CUDA_VISIBLE_DEVICES={run.devices or ''}", _q(cmd)])
    return run.tee(cmd, cwd=FR3_REPO, env=sim_env({"CUDA_VISIBLE_DEVICES": run.devices or ""}))


def stage_not_implemented(run: Run) -> int:
    print(
        f"[bakeoff] stage {run.lane}/{run.stage} is not implemented yet — "
        f"it belongs to a later phase (see plan/PLAN.md).",
        file=sys.stderr,
    )
    return 1


REGISTRY = {
    ("shared", "p0.smoke"): (stage_p0_smoke, 1),
    # Later phases register here as they are built; until then they are stubs
    # so a typo fails loudly instead of silently doing nothing.
    ("shared", "demos"): (stage_demos, 1),
    ("shared", "dataset"): (stage_dataset, 0),
    ("lane_a", "finetune"): (stage_finetune, 2),
    ("lane_a", "open_loop"): (stage_open_loop, 1),
    ("lane_a", "eval"): (stage_eval, 2),
    ("lane_a", "oracle_a"): (stage_oracle_a, 1),
    # P2 (2026-09-03): one GPU each — only 2 of 8 devices are allocatable and NCCL P2P/SHM
    # faults between them (P1 finding), so the RL run is single-GPU by decision.
    ("lane_b", "sonic_rl"): (stage_sonic_rl, 1),
    ("lane_b", "export_onnx"): (stage_export_onnx, 1),
    ("lane_b", "decoder_replay"): (stage_decoder_replay, 1),
    # P3 (2026-09-03): label_tokens needs a GPU only for its validation replay; finetune matches
    # lane A (2 GPUs); eval = server + Isaac (1 GPU when the other is busy); oracle_b = Isaac.
    ("lane_b", "label_tokens"): (stage_label_tokens, 1),
    ("lane_b", "finetune"): (stage_finetune, 2),
    ("lane_b", "eval"): (stage_eval, 1),
    ("lane_b", "oracle_b"): (stage_oracle_b, 1),
    # P5 diagnostic: the A-oracle with --left-j1-offset-rad, filed under lane_b so P1's readers
    # (newest lane_a/*oracle_a*) never see a perturbed run.
    ("lane_b", "oracle_a_tol"): (stage_oracle_a, 1),
}


# --------------------------------------------------------------------------- gpu claim
def acquire(job: str, n: int) -> str | None:
    if n <= 0:
        return None
    r = subprocess.run(
        [str(GR00T_PY), str(GPUS_PY), "acquire", "--n", str(n), "--job", job,
         "--pid", str(os.getpid())],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout, end="")
        print(r.stderr, end="", file=sys.stderr)
        sys.exit(f"[bakeoff] could not acquire {n} GPU(s) for {job} (AGENTS.md rule a)")
    for line in r.stdout.splitlines():
        if line.startswith("CUDA_VISIBLE_DEVICES="):
            return line.split("=", 1)[1].strip()
    return None


def release(job: str) -> None:
    subprocess.run(
        [str(GR00T_PY), str(GPUS_PY), "release", "--job", job],
        capture_output=True, text=True,
    )


def append_status(line: str) -> None:
    with STATUS.open("a") as fh:
        fh.write(line if line.endswith("\n") else line + "\n")


# --------------------------------------------------------------------------- cli
def cmd_root(args) -> int:
    """Read-only probe: which root a NEW run folder would get, and why."""
    floor, src = _min_home_free_gb()
    root, note = run_root()
    print(f"run_root            {root}")
    print(f"floor               {floor:.0f} GB  (from {src})")
    print(f"home free           {free_gb(RUNS):.0f} GB  ({RUNS})")
    print(f"instance-local free {free_gb(RUNS_FALLBACK):.0f} GB  ({RUNS_FALLBACK})")
    print(f"fallback            {note or 'no — home is at or above the floor'}")
    return 0


def cmd_run(args) -> int:
    key = (args.lane, args.stage)
    if key not in REGISTRY:
        known = ", ".join(f"{a}/{b}" for a, b in sorted(REGISTRY))
        sys.exit(f"[bakeoff] unknown stage {args.lane}/{args.stage}. Known: {known}")
    fn, default_gpus = REGISTRY[key]
    n_gpus = args.gpus if args.gpus is not None else default_gpus

    run = Run(args.lane, args.stage, args)
    job = f"{args.lane}-{args.stage}-{run.dir.name}"
    print(f"[bakeoff] run folder {run.dir}")
    try:
        run.devices = acquire(job, n_gpus)
        if run.devices:
            print(f"[bakeoff] CUDA_VISIBLE_DEVICES={run.devices} (job {job})")
        run.stamp()
        rc = fn(run)
    finally:
        release(job)
        run.stamp()
    stamp = _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    append_status(
        f"- {stamp}  bakeoff {args.lane}/{args.stage} "
        f"{'OK' if rc == 0 else 'FAILED'}  rc={rc}  gpus={run.devices or 'none'}  "
        f"run={run.dir}"
    )
    print(f"[bakeoff] {'OK' if rc == 0 else 'FAILED'} rc={rc}  log: {run.log}")
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="FR3 bake-off stage runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one stage")
    r.add_argument("lane", choices=sorted({l for l, _ in REGISTRY}))
    r.add_argument("stage")
    r.add_argument("--gpus", type=int, default=None,
                   help="devices to claim (default: the stage's own number)")
    r.add_argument("--tiny", action="store_true",
                   help="prototype numbers (10 sources, 2000 steps, 20 rollouts)")
    r.add_argument("--port", type=int, default=8000,
                   help="policy-server port for stages that start one")
    r.add_argument("--steps", default="all",
                   help="comma-separated sub-steps to run (stage-specific; default all)")
    r.add_argument("--resume", default=None,
                   help="existing run folder to continue instead of creating a new one")
    r.add_argument("--demos", default=None,
                   help="demo run folder (shared/*_demos) for stages that read the demo set")
    r.add_argument("--dataset", default=None,
                   help="dataset run folder (*_dataset) for stages that read gr00t_v2")
    r.add_argument("--checkpoint", default=None,
                   help="checkpoint dir (…/checkpoints/checkpoint-2000) for eval stages")
    r.add_argument("--motions", default=None,
                   help="SONIC motion-library dir (lane_b/*_motion_lib/out/motions) for lane_b stages")
    r.add_argument("--onnx", default=None,
                   help="lane_b/*_export_onnx run folder (encoder+decoder ONNX); default newest")
    r.add_argument("--tokens", default=None,
                   help="lane_b/*_label_tokens run folder (out/tokens) for oracle_b; default newest")
    r.add_argument("--init-from", default=None,
                   help="finetune: GR00T checkpoint dir to warm-start from (P11: each lane's own "
                        "round-2 checkpoint-20000); default the NVIDIA base model")
    r.add_argument("--max-episodes", type=int, default=0,
                   help="lane_b/label_tokens: label only the first N dataset episodes "
                        "(P8 WP 8.2 ceiling tests; 0 = all)")
    # shared/demos sizing (P7 WP 7.0). Defaults are round 1's hard-coded numbers, so every
    # round-1 command reproduces unchanged.
    r.add_argument("--n-sources", type=int, default=10,
                   help="shared/demos: scripted source demos to record")
    r.add_argument("--n-generated", type=int, default=80,
                   help="shared/demos: MimicGen successes wanted (a target, not a guarantee)")
    r.add_argument("--n-procs", type=int, default=4,
                   help="shared/demos: parallel single-env MimicGen workers")
    r.add_argument("--n-shards", type=int, default=4,
                   help="shared/demos: parallel export renderer shards")
    r.add_argument("--n-replay", type=int, default=1,
                   help="shared/demos: generated episodes replayed on the JointPos env (WP 7.4)")
    r.add_argument("--replay-seed", type=int, default=0,
                   help="shared/demos: RNG seed for which episodes the replay check draws")
    r.add_argument("--gen-deadline-min", type=float, default=0.0,
                   help="shared/demos: wall-clock budget for MimicGen generation in minutes; at "
                        "the deadline the workers are stopped and what they flushed is merged "
                        "(0 = no cap)")
    r.add_argument("--spawn-xy", type=float, default=GEN_SPAWN_XY_DEFAULT,
                   help="shared/demos: GENERATION cube spawn half-range in x and y (m). The "
                        "evaluation env is untouched by this (P7 WP 7.1)")
    r.add_argument("--spawn-yaw", type=float, default=GEN_SPAWN_YAW_DEFAULT,
                   help="shared/demos: GENERATION cube spawn yaw half-range (rad)")
    r.add_argument("--arm-noise-std", type=float, default=-1.0,
                   help="shared/demos: GENERATION arm reset-pose noise std (rad); "
                        "< 0 leaves the env cfg's own value (0.02)")
    r.add_argument("--num-envs", type=int, default=2048, help="lane_b/sonic_rl: parallel envs")
    r.add_argument("--exp", default="sonic_dual_fr3",
                   help="lane_b/sonic_rl: gear_sonic experiment config (+exp=…)")
    r.add_argument("--iters", type=int, default=100000,
                   help="lane_b/sonic_rl: max PPO iterations (the --hours cap usually wins)")
    r.add_argument("--hours", type=float, default=1.5,
                   help="lane_b/sonic_rl: wall-clock cap; the trainer is stopped by PID after it")
    r.add_argument("--rollouts", type=int, default=20)
    r.add_argument("--sonic-overrides", default="",
                   help="lane_b sonic_rl/export_onnx/decoder_replay: extra Hydra overrides appended "
                        "to the gear_sonic command, e.g. '++manager_env.config.robot.type=dual_fr3_stiff'")
    r.add_argument("--left-j1-offset-rad", type=float, default=0.0,
                   help="lane_b/oracle_a_tol: constant offset on the left joint-1 targets (P5 tolerance test)")
    r.add_argument("--max-steps", type=int, default=1500,
                   help="evaluation.eval horizon at 50 Hz (30 s = the env's own episode length)")
    # P9 WP 9.0: the fine-tune's step budget. --max-steps above is the EVALUATION horizon in
    # this CLI, so the trainer's budget needs its own name. Defaults are round 1's numbers.
    r.add_argument("--train-steps", type=int, default=2000,
                   help="finetune: gradient steps (round 1 ran 2000; P9 runs 20000)")
    r.add_argument("--save-steps", type=int, default=500,
                   help="finetune: checkpoint interval in steps")
    r.add_argument("--save-total-limit", type=int, default=5,
                   help="finetune: checkpoints kept. AGENTS.md rule o: set it high enough that "
                        "the trainer never rotates one away (P9 keeps all 8)")
    r.set_defaults(fn=cmd_run)
    q = sub.add_parser("root", help="print the run root a new run would use, free space and floor")
    q.set_defaults(fn=cmd_root)
    args = ap.parse_args(argv)
    # Orchestrator -> phase-agent channel that cannot conflict with STATUS.md pushes:
    # anything in plan/ORCHESTRATOR_NOTES.md is echoed on every bakeoff invocation.
    notes = REPO / "plan" / "ORCHESTRATOR_NOTES.md"
    if notes.is_file():
        print("[bakeoff] ===== plan/ORCHESTRATOR_NOTES.md (read it; it may change your plan) =====", flush=True)
        print(notes.read_text().rstrip(), flush=True)
        print("[bakeoff] ===== end of orchestrator notes =====", flush=True)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
