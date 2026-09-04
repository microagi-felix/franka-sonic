"""franka-sonic copy of `mimic/scripts/generate_parallel.py` (franka-bimanual-isaac-sim @14f0d8a).

Parallel MimicGen: N single-env generation processes -> merged dataset. Upstream found that
in-process --num_envs degraded the success rate (~24% -> ~9% at 4 envs), so it parallelises at
the PROCESS level: N independent single-env `generate_handover.py` runs with distinct seeds and
output files, merged into one HDF5 afterwards.

Changes to upstream: workers run THIS directory's `generate_handover.py` (ISAACLAB_ROOT + 50 Hz
aware), and nothing is ever deleted — an existing worker/output file is an error, not something to
overwrite (AGENTS.md rule o). `MIMIC_RATE_HZ` and `ISAACLAB_ROOT` are inherited by the workers.

    MIMIC_RATE_HZ=50 PYTHONUSERBASE=~/env/pyuser-fr3 /isaac-sim/python.sh harness/data/generate_parallel.py \
        --task Isaac-Stack-Cube-DualFranka-IK-Abs-Mimic-v0 --input <annotated> --output <run>/out/generated.hdf5 \
        --total 80 --procs 4

Each worker gets ceil(total/procs) trials (generation_guarantee retries failures until it has its
quota of SUCCESSES). Reports the aggregate success (keep) rate as PARALLEL_GEN_DONE.
"""

import argparse
import math
import os
import re
import subprocess
import sys
import time

import h5py
from frankas_assets.specs.task import add_task_argument

HERE = os.path.dirname(os.path.abspath(__file__))


def merge(worker_files: list[str], output: str) -> int:
    """Concatenate data/demo_* groups from worker files into one HDF5 (never overwrites)."""
    if os.path.exists(output):
        raise SystemExit(f"[parallel] refusing to overwrite existing {output} (rule o: no deletes)")
    n = 0
    with h5py.File(output, "w") as out:
        data = out.create_group("data")
        for wf in worker_files:
            if not os.path.exists(wf):
                print(f"[parallel] WARNING: worker file missing: {wf}", flush=True)
                continue
            try:
                h5py.File(wf, "r").close()
            except OSError as exc:  # a deadline-stopped worker can leave an unreadable tail
                print(f"[parallel] WARNING: worker file unreadable, skipped: {wf} ({exc})", flush=True)
                continue
            with h5py.File(wf, "r") as f:
                # copy file-level attrs once (env args etc.)
                for k, v in f.attrs.items():
                    if k not in out.attrs:
                        out.attrs[k] = v
                if "data" in f:
                    for k, v in f["data"].attrs.items():
                        if k not in data.attrs:
                            data.attrs[k] = v
                    for name in f["data"]:
                        f.copy(f["data"][name], data, name=f"demo_{n}")
                        n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_task_argument(p, default="Isaac-Stack-Cube-DualFranka-IK-Abs-Mimic-v0")
    p.add_argument("--input", required=True, help="annotated source dataset")
    p.add_argument("--output", required=True, help="merged output HDF5")
    p.add_argument("--total", type=int, default=80, help="total successful demos wanted")
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--seed_base", type=int, default=100)
    p.add_argument("--deadline_min", type=float, default=0.0,
                   help="P7: wall-clock budget in minutes; at the deadline the workers this "
                        "process started are stopped (SIGTERM, then SIGKILL after a grace "
                        "period) and whatever they already flushed is merged. 0 = no cap.")
    args = p.parse_args()

    per = math.ceil(args.total / args.procs)
    workdir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(workdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.output))[0]
    if os.path.exists(args.output):
        raise SystemExit(f"[parallel] {args.output} already exists; pick a fresh name (rule o)")

    # Spread the workers round-robin over the devices the allocator claimed. With one device
    # (round 1: `--gpus 1`, 4 procs) every worker gets that same device, exactly as before.
    devices = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]

    procs, files, logs = [], [], []
    for i in range(args.procs):
        wf = os.path.join(workdir, f"{stem}_w{i}.hdf5")
        lg = os.path.join(workdir, f"{stem}_w{i}.log")
        if os.path.exists(wf):
            raise SystemExit(f"[parallel] worker output {wf} already exists; pick a fresh name (rule o)")
        files.append(wf)
        logs.append(lg)
        cmd = [
            sys.executable, os.path.join(HERE, "generate_handover.py"),
            "--task", args.task,
            "--input_file", os.path.abspath(args.input),
            "--output_file", wf,
            "--generation_num_trials", str(per),
            "--headless",
        ]
        env = os.environ.copy()
        env["MIMIC_DATAGEN_SEED"] = str(args.seed_base + i)  # distinct RNG per worker
        dev = devices[i % len(devices)] if devices else ""
        if dev:
            env["CUDA_VISIBLE_DEVICES"] = dev
        procs.append(subprocess.Popen(cmd, stdout=open(lg, "w"), stderr=subprocess.STDOUT, env=env))
        print(f"[parallel] worker {i} pid {procs[-1].pid}: {per} successes -> {wf} "
              f"(gpu {dev or 'inherited'}, log {lg})", flush=True)

    t0 = time.monotonic()
    stopped = False
    if args.deadline_min > 0:
        # Bounded foreground poll (AGENTS.md rule i). Only our own children are signalled
        # (rule b: never by pattern); each flushes its HDF5 after every recorded episode, so
        # what is on disk at the deadline is complete up to the last success.
        budget = args.deadline_min * 60.0
        while time.monotonic() - t0 < budget and any(q.poll() is None for q in procs):
            time.sleep(10.0)
        alive = [q for q in procs if q.poll() is None]
        if alive:
            stopped = True
            print(f"[parallel] DEADLINE {args.deadline_min:g} min reached with {len(alive)} "
                  f"worker(s) still running: stopping pids {[q.pid for q in alive]}", flush=True)
            for q in alive:
                q.terminate()
            grace0 = time.monotonic()
            while time.monotonic() - grace0 < 120.0 and any(q.poll() is None for q in alive):
                time.sleep(5.0)
            for q in alive:
                if q.poll() is None:
                    print(f"[parallel] worker pid {q.pid} ignored SIGTERM; SIGKILL", flush=True)
                    q.kill()
    codes = [q.wait() for q in procs]
    dt = time.monotonic() - t0
    print(f"[parallel] workers done in {dt/60:.1f} min, exit codes {codes}"
          f"{' (deadline-stopped)' if stopped else ''}", flush=True)

    # aggregate success rate from worker logs ("S/T (P%) successful demos")
    succ = trials = 0
    pat = re.compile(r"(\d+)/(\d+) \([\d.]+%\) successful demos")
    for lg in logs:
        last = None
        with open(lg, errors="ignore") as fh:
            for line in fh:
                m = pat.search(line)
                if m:
                    last = m
        if last:
            succ += int(last.group(1))
            trials += int(last.group(2))
    n = merge(files, os.path.abspath(args.output))
    rate = (100.0 * succ / trials) if trials else float("nan")
    print(f"PARALLEL_GEN_DONE: {n} episodes merged -> {args.output} "
          f"(aggregate {succ}/{trials} = {rate:.1f}% success, {dt/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
