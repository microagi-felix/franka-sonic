"""P5 diagnostic: flange position error (cm) of a decoder replay by MuJoCo FK.

    PYTHONUSERBASE=~/env/pyuser-sonic <Kit python.sh> harness/lane_b/fk_flange_error.py <decoder_replay run> [...]

Reads out/replay_trajectories.npz (reference / measured joint trajectories, IsaacLab order) and
reports the left/right flange (link7 + 0.107 m along z) position error per clip and at the grasp
frames of handover_s0_d0 (left 100-250, right 600-750). Uses the INSTALLED dual_fr3.xml (the repo
copy cannot be loaded standalone: its meshdir is missing).
"""
import glob
import os
import sys

import mujoco
import numpy as np

XML = os.path.expanduser("~/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/dual_fr3.xml")
ISAAC = [f"{s}_fr3_joint{i}" for i in range(1, 8) for s in ("left", "right")]
MJ = [f"left_fr3_joint{i}" for i in range(1, 8)] + [f"right_fr3_joint{i}" for i in range(1, 8)]
I2M = [ISAAC.index(j) for j in MJ]


def main(runs):
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("left_fr3_link7", "right_fr3_link7")]

    def flanges(q_isaac):
        d.qpos[:14] = q_isaac[I2M]
        mujoco.mj_forward(m, d)
        return np.array([d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.array([0, 0, 0.107]) for b in bodies])

    for run in runs:
        z = np.load(os.path.join(run, "out", "replay_trajectories.npz"))
        r, t = z["reference"], z["measured"]
        e = np.array([np.linalg.norm(flanges(t[i]) - flanges(r[i]), axis=1) for i in range(len(r))]) * 100
        print(f"{os.path.basename(run):28s} flange cm: L mean {e[:, 0].mean():.1f} max {e[:, 0].max():.1f} | "
              f"R mean {e[:, 1].mean():.1f} max {e[:, 1].max():.1f} | L grasp(100-250) {e[100:250, 0].mean():.1f} "
              f"max {e[100:250, 0].max():.1f} | R grasp(600-750) {e[600:750, 1].mean():.1f}")


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(glob.glob(os.path.expanduser("~/runs/franka-sonic/lane_b/*_decoder_replay*")))[-1:])
