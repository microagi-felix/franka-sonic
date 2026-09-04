#!/usr/bin/env python3
"""P7 WP 7.1: measured spawn distribution of an exported demo set vs the EVALUATION range.

Reads the per-demo `initial_cube_pose (7,)` (env-frame pos + quat wxyz) that
`export_generated_50hz.py` writes into every export shard and reports min/max/mean/std of
cube x, y and yaw, next to the range the evaluation env actually draws from, plus a boolean
`covers_eval`. Written to `<demos run>/out/coverage.json`; gate p7 check 4 reads it.

The evaluation range is NOT the generation range and neither is hard-coded here:

* generation = `Isaac-Stack-Cube-DualFranka-IK-Abs-Mimic-v0` -> `mimic/env_cfg.py` replaces
  `events.init_cube`'s pose_range with GEN_SPAWN_XY / GEN_SPAWN_YAW (round 1: 0.06 m / 0.5 rad),
  which P7 widens at runtime through MIMIC_SPAWN_XY / MIMIC_SPAWN_YAW (harness/data/generate_handover.py);
* evaluation = `Isaac-Stack-Cube-DualFranka-JointPos-v0` -> `DualFrankaJointPosEnvCfg` inherits
  `DualFrankaCubeStackEnvCfg`, whose `_block_spawn_event` keeps the tight default and which no
  eval-side code overrides (`evaluation/eval.py` touches only the donut task's events).

Both are read out of the franka repo's sources (no Isaac import — this must run under the GR00T
venv while the sim GPUs are busy); `--eval-xy/--eval-yaw/--eval-centre` override the parse.

    ~/Isaac-GR00T/.venv/bin/python harness/data/spawn_coverage.py \
        --export '<run>/out/export/*.hdf5' --output <run>/out/coverage.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

import h5py
import numpy as np

FR3_REPO = os.path.expanduser("~/code/franka-bimanual-isaac-sim")
TASK_CFG = os.path.join(FR3_REPO, "tasks", "stack_fr3", "dual_stack_env_cfg.py")
MIMIC_CFG = os.path.join(FR3_REPO, "mimic", "env_cfg.py")


def _eval_range() -> dict:
    """(half-range x/y in m, half-range yaw in rad, centre xy) of the EVALUATION spawn event."""
    src = open(TASK_CFG).read()
    mx = re.search(r'"x":\s*\(x\s*-\s*([0-9.]+),\s*x\s*\+\s*([0-9.]+)\)', src)
    my = re.search(r'"y":\s*\(y\s*-\s*([0-9.]+),\s*y\s*\+\s*([0-9.]+)\)', src)
    mw = re.search(r'"yaw":\s*\(-([0-9.]+),\s*([0-9.]+)\)', src)
    ms = re.search(r"^SLOT_Y\s*=\s*(-?[0-9.]+)", src, re.M)
    if not (mx and my and mw and ms):
        sys.exit(f"[coverage] could not parse _block_spawn_event out of {TASK_CFG}; "
                 "pass --eval-xy/--eval-yaw/--eval-centre explicitly")
    if float(mx.group(1)) != float(mx.group(2)) or float(my.group(1)) != float(my.group(2)):
        sys.exit("[coverage] the evaluation spawn event is not symmetric; pass --eval-xy explicitly")
    sys.path.insert(0, FR3_REPO)
    from frankas_assets.rig import ROBOT_POS  # Isaac-free single source of truth for the rig
    return {
        "xy_half_range_m": float(mx.group(1)),
        "y_half_range_m": float(my.group(1)),
        "yaw_half_range_rad": float(mw.group(2)),
        "centre_xy_m": [float(ROBOT_POS[0]), float(ms.group(1))],
        "source": f"{TASK_CFG}:_block_spawn_event + frankas_assets.rig.ROBOT_POS",
        "env": "Isaac-Stack-Cube-DualFranka-JointPos-v0 (DualFrankaJointPosEnvCfg -> "
               "DualFrankaCubeStackEnvCfg.events.init_cube; evaluation/eval.py does not override it)",
    }


def _gen_range_cfg() -> dict:
    src = open(MIMIC_CFG).read()
    xy = re.search(r"^GEN_SPAWN_XY\s*=\s*([0-9.]+)", src, re.M)
    yaw = re.search(r"^GEN_SPAWN_YAW\s*=\s*([0-9.]+)", src, re.M)
    return {
        "cfg_xy_half_range_m": float(xy.group(1)) if xy else None,
        "cfg_yaw_half_range_rad": float(yaw.group(1)) if yaw else None,
        "env_MIMIC_SPAWN_XY": os.environ.get("MIMIC_SPAWN_XY"),
        "env_MIMIC_SPAWN_YAW": os.environ.get("MIMIC_SPAWN_YAW"),
        "env_MIMIC_ARM_NOISE_STD": os.environ.get("MIMIC_ARM_NOISE_STD"),
        "env": "Isaac-Stack-Cube-DualFranka-IK-Abs-Mimic-v0 (DualFrankaHandoverMimicEnvCfg)",
    }


def _yaw(q: np.ndarray) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True, help="glob of export shards (…/out/export/*.hdf5)")
    ap.add_argument("--output", required=True, help="coverage.json to write")
    ap.add_argument("--eval-xy", type=float, default=None, help="override the parsed eval half-range (m)")
    ap.add_argument("--eval-yaw", type=float, default=None, help="override the parsed eval half-range (rad)")
    ap.add_argument("--eval-centre", default=None, help="override the parsed eval centre, 'x,y'")
    ap.add_argument("--only-successful", action="store_true",
                    help="restrict the statistics to the episodes a dataset build would keep: "
                         "replay_success AND jointpos_replay_success (a missing attr means True)")
    args = ap.parse_args()

    ev = _eval_range()
    if args.eval_xy is not None:
        ev["xy_half_range_m"] = ev["y_half_range_m"] = args.eval_xy
        ev["source"] += " (--eval-xy override)"
    if args.eval_yaw is not None:
        ev["yaw_half_range_rad"] = args.eval_yaw
        ev["source"] += " (--eval-yaw override)"
    if args.eval_centre:
        ev["centre_xy_m"] = [float(v) for v in args.eval_centre.split(",")]
        ev["source"] += " (--eval-centre override)"

    files = sorted(glob.glob(args.export))
    if not files:
        sys.exit(f"[coverage] no export shards match {args.export}")
    xs, ys, yaws, n_all, n_ok = [], [], [], 0, 0
    for f in files:
        with h5py.File(f, "r") as h:
            for name in h["data"]:
                g = h["data"][name]
                if "initial_cube_pose" not in g:
                    continue
                ok = (bool(g.attrs.get("replay_success", False))
                      and bool(g.attrs.get("jointpos_replay_success", True)))
                n_all += 1
                n_ok += int(ok)
                if args.only_successful and not ok:
                    continue
                p = np.asarray(g["initial_cube_pose"])
                xs.append(float(p[0]))
                ys.append(float(p[1]))
                yaws.append(_yaw(p[3:7]))
    if not xs:
        sys.exit(f"[coverage] no initial_cube_pose found in {len(files)} shard(s)")

    def stats(v: list[float]) -> dict:
        a = np.asarray(v)
        return {"min": float(a.min()), "max": float(a.max()),
                "mean": float(a.mean()), "std": float(a.std()),
                "half_range": float((a.max() - a.min()) / 2.0)}

    m = {"x": stats(xs), "y": stats(ys), "yaw": stats(yaws)}
    cx, cy = ev["centre_xy_m"]
    covers = {
        "x": m["x"]["min"] <= cx - ev["xy_half_range_m"] and m["x"]["max"] >= cx + ev["xy_half_range_m"],
        "y": m["y"]["min"] <= cy - ev["y_half_range_m"] and m["y"]["max"] >= cy + ev["y_half_range_m"],
        "yaw": m["yaw"]["min"] <= -ev["yaw_half_range_rad"] and m["yaw"]["max"] >= ev["yaw_half_range_rad"],
    }
    doc = {
        "episodes": len(xs),
        "episodes_in_export": n_all,
        "episodes_replay_success": n_ok,
        "only_successful": bool(args.only_successful),
        "shards": files,
        "measured": m,
        "eval_range": ev,
        "generation_range": _gen_range_cfg(),
        "covers_eval": bool(all(covers.values())),
        "covers_eval_per_axis": covers,
        "margin": {
            "x": float(min(cx - ev["xy_half_range_m"] - m["x"]["min"],
                           m["x"]["max"] - (cx + ev["xy_half_range_m"]))),
            "y": float(min(cy - ev["y_half_range_m"] - m["y"]["min"],
                           m["y"]["max"] - (cy + ev["y_half_range_m"]))),
            "yaw": float(min(-ev["yaw_half_range_rad"] - m["yaw"]["min"],
                             m["yaw"]["max"] - ev["yaw_half_range_rad"])),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    print(json.dumps({k: doc[k] for k in
                      ("episodes", "episodes_in_export", "episodes_replay_success",
                       "measured", "eval_range", "covers_eval", "covers_eval_per_axis", "margin")},
                     indent=2), flush=True)
    print(f"COVERAGE_DONE: {len(xs)} episodes -> {args.output} (covers_eval={doc['covers_eval']})",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
