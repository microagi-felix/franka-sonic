#!/usr/bin/env python3
"""ONE entry point for every bake-off stage: it makes the run folder, stamps
provenance, holds the GPU claim, tees the log, and appends to plan/STATUS.md.

    python3 harness/bakeoff.py run shared p0.smoke
    python3 harness/bakeoff.py run lane_a finetune --gpus 2 --tiny

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
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = Path(os.path.expanduser("~/runs/franka-sonic"))
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


# --------------------------------------------------------------------------- run folder
class Run:
    def __init__(self, lane: str, stage: str, args: argparse.Namespace):
        self.lane, self.stage, self.args = lane, stage, args
        day = _dt.date.today().isoformat()
        base = RUNS / lane / f"{day}_{stage}"
        d, n = base, 1
        while d.exists():
            n += 1
            d = Path(f"{base}-{n}")
        self.dir = d
        (self.dir / "logs").mkdir(parents=True)
        (self.dir / "out").mkdir()
        self.log = self.dir / "logs" / "run.log"
        self.devices: str | None = None

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
    ("shared", "demos"): (stage_not_implemented, 1),
    ("lane_a", "finetune"): (stage_not_implemented, 2),
    ("lane_a", "eval"): (stage_not_implemented, 1),
    ("lane_b", "sonic_rl"): (stage_not_implemented, 4),
    ("lane_b", "export_onnx"): (stage_not_implemented, 1),
    ("lane_b", "label_tokens"): (stage_not_implemented, 1),
    ("lane_b", "finetune"): (stage_not_implemented, 2),
    ("lane_b", "eval"): (stage_not_implemented, 1),
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
    r.set_defaults(fn=cmd_run)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
