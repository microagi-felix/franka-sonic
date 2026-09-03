#!/usr/bin/env python3
"""Does gear_sonic's MotionLibRobot load our dual-FR3 motion library? (CPU only)

    PYTHONUSERBASE=~/env/pyuser-sonic /isaac-sim/python.sh \
        harness/lane_b/check_motion_lib_loads.py \
        --motion-dir ~/runs/franka-sonic/lane_b/2026-09-03_motion_lib/out/motions

This is the P2 "does the loader accept the pkls" check. It builds the same
`motion_lib_cfg` that `gear_sonic/envs/manager_env/mdp/commands.py` (lines
~108-240) assembles from the env, but with **identity** IsaacLab<->MuJoCo maps
(no Isaac import, no GPU): 14 dofs, 17 bodies. `smpl_motion_file: dummy` and
`adaptive_sampling.enable: False` keep it off the SMPL/soma paths, and
`multi_thread: False` keeps it in this process so a traceback is readable.

Reports `_num_unique_motions`, the loaded motion count, and the per-frame
`dof_pos` round-trip error against the pkls' own `dof` (the loader resamples
30 -> `--target-fps`, so a clip is compared at its own frame times).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

DEFAULT_XML = os.path.expanduser(
    "~/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/dual_fr3.xml"
)
DEFAULT_WBC = os.path.expanduser("~/GR00T-WholeBodyControl")
N_DOF, N_BODIES = 14, 17


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion-dir", required=True)
    ap.add_argument("--xml", default=DEFAULT_XML)
    ap.add_argument("--wbc", default=DEFAULT_WBC)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--target-fps", type=int, default=50)
    args = ap.parse_args()

    motion_dir = os.path.abspath(os.path.expanduser(args.motion_dir))
    xml = os.path.abspath(os.path.expanduser(args.xml))
    wbc = os.path.abspath(os.path.expanduser(args.wbc))
    n_pkl = len(glob.glob(os.path.join(motion_dir, "*.pkl")))
    print(f"[loader] {n_pkl} pkls in {motion_dir}")

    sys.path.insert(0, wbc)
    os.chdir(wbc)
    import easydict
    import torch

    from gear_sonic.utils.motion_lib import motion_lib_robot

    ident_dof = list(range(N_DOF))
    ident_body = list(range(N_BODIES))
    cfg = easydict.EasyDict({
        "motion_file": motion_dir,
        "smpl_motion_file": "dummy",
        "asset": {
            "assetRoot": os.path.dirname(xml),
            "assetFileName": os.path.basename(xml),
            "urdfFileName": "",
        },
        "extend_config": [],
        "target_fps": args.target_fps,
        "multi_thread": False,
        "adaptive_sampling": {"enable": False},
        # identity maps: this rig's MuJoCo order IS the order we hand the loader
        "mujoco_to_isaaclab_dof": ident_dof,
        "isaaclab_to_mujoco_dof": ident_dof,
        "mujoco_to_isaaclab_body": ident_body,
        "isaaclab_to_mujoco_body": ident_body,
        "body_indexes": torch.tensor(ident_body, dtype=torch.long),
        "body_indexes_data": ident_body,
        "lower_joint_indices_mujoco": list(range(12)),
        "cat_upper_body_poses": False,
        "freeze_frame_aug": False,
        "randomize_heading": False,
        "randomize_wrist_poses": False,
        "filter_motion_keys": None,
    })

    lib = motion_lib_robot.MotionLibRobot(cfg, num_envs=args.max_num_seqs, device="cpu")
    print(f"[loader] _num_unique_motions = {lib._num_unique_motions}")
    lib.load_motions_for_training(max_num_seqs=args.max_num_seqs)
    n = int(lib._motion_lengths.shape[0])
    print(f"[loader] loaded {n} motions; "
          f"lengths {lib._motion_lengths.min():.2f}..{lib._motion_lengths.max():.2f} s; "
          f"fps {sorted(set(lib._motion_fps.tolist()))}")
    print(f"[loader] dof_pos {tuple(lib.dof_pos.shape)}  body_pos_w {tuple(lib.body_pos_w.shape)}")
    assert lib.dof_pos.shape[-1] == N_DOF, lib.dof_pos.shape
    assert lib.body_pos_w.shape[-2] == N_BODIES, lib.body_pos_w.shape

    # sample a few motion times and check the loaded dof against the source pkl
    import joblib

    keys = list(lib.curr_motion_keys)
    worst = 0.0
    for i, key in enumerate(keys[: min(3, len(keys))]):
        src = joblib.load(os.path.join(motion_dir, f"{key}.pkl"))[key]
        ids = torch.tensor([i], dtype=torch.long)
        for frac in (0.0, 0.25, 0.5, 0.75):
            t = torch.tensor([float(lib._motion_lengths[i]) * frac], dtype=torch.float32)
            res = lib.get_motion_state(ids, t)
            got = res["dof_pos"][0].numpy()
            # nearest source frame at the pkl's own 30 fps
            j = int(round(float(t) * src["fps"]))
            j = min(j, src["dof"].shape[0] - 1)
            worst = max(worst, float(np.abs(got - src["dof"][j]).max()))
        print(f"[loader] {key}: dof vs source pkl max |err| so far {worst:.4f} rad")
    print(f"[loader] max |loaded dof - source dof| at sampled times = {worst:.4f} rad "
          f"(resampling {int(src['fps'])} -> {args.target_fps} fps, so a small "
          "interpolation error is expected)")
    print("[loader] OK: MotionLibRobot loaded the dual-FR3 library on CPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
