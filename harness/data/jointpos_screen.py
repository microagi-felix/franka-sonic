#!/usr/bin/env python3
"""P7 WP 7.4: replay EVERY exported episode open-loop on the JointPos env and record the verdict.

The export's own `replay_success` attr comes from the IK-Abs env replaying the generated
*Cartesian* actions with the differential-IK controller closing the loop on the arm state. Lane
A's labels, the A-oracle and both lanes' evaluation instead play the recorded ABSOLUTE joint
targets open-loop on `Isaac-Stack-Cube-DualFranka-JointPos-v0`, which is a strictly harder ask:
P7 measured 17/20 there on a set that was 978/1024 replay_success. An episode that does not
reproduce the handover under the evaluation's own controller teaches lane A a trajectory that
does not work and drags the A-oracle's "100 % by construction" row down, so it is screened out.

This runs `harness/lane_a/eval_oracle_a.py` once per device over a stride of the episode table
(exactly the protocol the demo stage's replay check uses), merges the per-device
`eval_results.csv`, writes `<demos run>/out/jointpos_screen.json`, and stamps a
`jointpos_replay_success` attr onto every screened demo group in the export shards. Consumers
(`convert_hdf5_to_gr00t_v2.py`, `eval_oracle_a.py`) treat a MISSING attr as True, so round-1
exports and any set that was never screened behave exactly as before.

Nothing is ever deleted: the attr is added next to `replay_success`, which keeps its own meaning.

    ~/Isaac-GR00T/.venv/bin/python harness/data/jointpos_screen.py \
        --demos ~/runs/franka-sonic/shared/2026-09-04_demos --devices 0,1,2,3,4,5,6,7
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FR3_REPO = os.path.expanduser("~/code/franka-bimanual-isaac-sim")
PYSH = "/isaac-sim/python.sh"


def episode_table(export_dir: str) -> list[str]:
    """The episode names `eval_oracle_a.load_episodes` yields, in its order (successes only)."""
    import h5py

    names = []
    for path in sorted(glob.glob(os.path.join(export_dir, "*.hdf5"))):
        with h5py.File(path, "r") as f:
            for name in sorted(f["data"].keys(), key=lambda n: int(n.split("_")[1])):
                g = f["data"][name]
                if not bool(g.attrs.get("replay_success", True)):
                    continue
                if not bool(g.attrs.get("jointpos_replay_success", True)):
                    continue
                names.append(f"{os.path.basename(path)}:{name}")
    return names


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demos", required=True, help="shared/*_demos run folder")
    ap.add_argument("--devices", required=True, help="comma-separated CUDA devices (from the allocator)")
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--limit", type=int, default=0, help="0 = every episode")
    args = ap.parse_args()

    demos = os.path.abspath(os.path.expanduser(args.demos))
    export_dir = os.path.join(demos, "out", "export")
    out_dir = os.path.join(demos, "out", "jointpos_screen")
    os.makedirs(out_dir, exist_ok=True)
    names = episode_table(export_dir)
    if args.limit:
        names = names[: args.limit]
    devices = [d for d in args.devices.split(",") if d.strip()]
    print(f"[screen] {len(names)} episodes over {len(devices)} device(s)", flush=True)

    env0 = dict(os.environ)
    env0["PYTHONUSERBASE"] = os.path.expanduser("~/env/pyuser-fr3")
    env0["PYTHONUNBUFFERED"] = "1"
    procs, groups = [], []
    for k, dev in enumerate(devices):
        idx = list(range(k, len(names), len(devices)))
        if not idx:
            continue
        groups.append(idx)
        folder = os.path.join(out_dir, f"eval_{k}")
        cmd = [PYSH, os.path.join(REPO, "harness", "lane_a", "eval_oracle_a.py"),
               "--demos", export_dir, "--run-folder", folder,
               "--episode-indices", ",".join(str(i) for i in idx),
               "--start-episode", "0", "--rate", f"{args.rate:g}",
               "--rollouts", str(len(idx)), "--max-steps", str(args.max_steps),
               "--no-splat", "--headless", "--no-save-video"]
        env = dict(env0)
        env["CUDA_VISIBLE_DEVICES"] = dev
        log = os.path.join(out_dir, f"eval_{k}.log")
        procs.append(subprocess.Popen(cmd, cwd=FR3_REPO, env=env,
                                      stdout=open(log, "w"), stderr=subprocess.STDOUT))
        print(f"[screen] worker {k} pid {procs[-1].pid} gpu {dev}: {len(idx)} episodes -> {folder}",
              flush=True)
    with open(os.path.join(out_dir, "workers.pid"), "w") as fh:
        fh.write("\n".join(str(p.pid) for p in procs) + "\n")

    t0 = time.monotonic()
    while any(p.poll() is None for p in procs):
        time.sleep(20.0)
    print(f"[screen] workers done in {(time.monotonic() - t0) / 60:.1f} min, "
          f"exit codes {[p.returncode for p in procs]}", flush=True)

    verdict: dict[str, bool] = {}
    for k, idx in enumerate(groups):
        csv_path = os.path.join(out_dir, f"eval_{k}", "eval_results.csv")
        if not os.path.exists(csv_path):
            print(f"[screen] WARNING: no eval_results.csv for worker {k}", flush=True)
            continue
        with open(csv_path) as fh:
            for row in csv.DictReader(fh):
                pos = int(row["episode"])
                if pos < len(idx):
                    verdict[names[idx[pos]]] = str(row["success"]).strip().lower() == "true"
    passed = sum(1 for v in verdict.values() if v)
    doc = {"episodes_screened": len(verdict), "episodes_in_table": len(names),
           "passed": passed, "failed": len(verdict) - passed,
           "pass_rate_pct": round(100.0 * passed / len(verdict), 2) if verdict else None,
           "protocol": f"eval_oracle_a.py on Isaac-Stack-Cube-DualFranka-JointPos-v0, "
                       f"--rate {args.rate:g} --max-steps {args.max_steps} --no-splat, "
                       f"joint label ik_target_clamped, {len(devices)} devices",
           "verdict": verdict}
    doc_path = os.path.join(demos, "out", "jointpos_screen.json")
    with open(doc_path, "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")

    import h5py

    stamped = 0
    for path in sorted(glob.glob(os.path.join(export_dir, "*.hdf5"))):
        base = os.path.basename(path)
        mine = {n.split(":", 1)[1]: v for n, v in verdict.items() if n.startswith(base + ":")}
        if not mine:
            continue
        with h5py.File(path, "r+") as f:
            for name, ok in mine.items():
                f["data"][name].attrs["jointpos_replay_success"] = bool(ok)
                stamped += 1
    print(f"[screen] stamped jointpos_replay_success on {stamped} demo group(s)", flush=True)
    print(f"SCREEN_DONE: {passed}/{len(verdict)} episodes replay to handover_success on the "
          f"JointPos env ({doc['pass_rate_pct']}%) -> {doc_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
