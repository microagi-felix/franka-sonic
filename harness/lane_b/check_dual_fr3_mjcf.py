#!/usr/bin/env python3
"""Sanity-check dual_fr3.xml for both of its consumers (CPU only).

    PYTHONUSERBASE=~/env/pyuser-sonic /isaac-sim/python.sh harness/lane_b/check_dual_fr3_mjcf.py \
        --xml ~/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/dual_fr3.xml

1. mujoco.MjModel.from_xml_path loads it; prints body / joint / actuator names in
   MuJoCo order (the order the SONIC pkl `dof` must use).
2. gear_sonic's Humanoid_Batch parses the same file (needs cwd = GR00T-WholeBodyControl
   for its relative assetRoot) and its fk_batch on a random pose matches MuJoCo's
   mj_forward body positions — the contract the demo->pkl converter builds on
   (pose_aa[:, body] = dof_axis * dof for the body's joint, root = identity).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--wbc", default=os.path.expanduser("~/GR00T-WholeBodyControl"))
    args = ap.parse_args()
    xml = os.path.abspath(os.path.expanduser(args.xml))

    import mujoco

    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    joints = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    acts = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    print("mujoco bodies :", bodies)
    print("mujoco joints :", joints)
    print("mujoco actuators:", acts)
    print(f"nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody}")
    assert m.nq == 14 and m.nu == 14, "expected 14 hinge DoF, no freejoint"
    lo, hi = m.jnt_range[:, 0], m.jnt_range[:, 1]
    print("joint ranges lo:", np.round(lo, 3).tolist())
    print("joint ranges hi:", np.round(hi, 3).tolist())

    # home keyframe
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    for side in ("left", "right"):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_fr3_link7")
        s = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_attachment_site")
        print(f"home {side} link7 xpos={np.round(d.xpos[b], 4).tolist()} "
              f"flange site={np.round(d.site_xpos[s], 4).tolist()}")

    # ---------------------------------------------------------------- Humanoid_Batch
    os.chdir(args.wbc)
    sys.path.insert(0, args.wbc)
    import torch
    from easydict import EasyDict

    from gear_sonic.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

    root = os.path.dirname(xml)
    cfg = EasyDict({"asset": {"assetRoot": root, "assetFileName": os.path.basename(xml)},
                    "extend_config": []})
    hb = Humanoid_Batch(cfg)
    print("humanoid_batch body_names:", hb.body_names)
    print("humanoid_batch num_dof:", hb.num_dof, "has_freejoint:", hb.has_freejoint)
    print("humanoid_batch actuated_joints_idx:", hb.actuated_joints_idx.tolist())
    print("humanoid_batch body_to_joint:", list(hb.mjcf_data["body_to_joint"].items()))
    assert hb.body_names == bodies[1:], "Humanoid_Batch body order != MuJoCo body order (minus world)"
    mj_joint_order = [hb.mjcf_data["body_to_joint"][hb.body_names[i]] for i in hb.actuated_joints_idx]
    assert mj_joint_order == joints, (mj_joint_order, joints)

    rng = np.random.default_rng(0)
    T = 5
    dof = rng.uniform(lo, hi, size=(T, 14)).astype(np.float32)
    # pose_aa in MuJoCo body order: root identity, joint bodies = axis * angle
    nb = len(hb.body_names)
    pose_aa = np.zeros((T, nb, 3), dtype=np.float32)
    axes = hb.dof_axis.numpy().astype(np.float32)  # (14, 3) in joint document order
    for k, bi in enumerate(hb.actuated_joints_idx):
        pose_aa[:, bi, :] = axes[k][None] * dof[:, k : k + 1]
    trans = np.zeros((T, 3), dtype=np.float32)
    out = hb.fk_batch(torch.from_numpy(pose_aa)[None], torch.from_numpy(trans)[None],
                      return_full=True)
    fk_pos = out.global_translation[0].numpy()  # (T, nb, 3)
    fk_dof = out.dof_pos[0].numpy()
    worst = 0.0
    for t in range(T):
        d.qpos[:] = dof[t]
        mujoco.mj_forward(m, d)
        err = np.abs(fk_pos[t] - d.xpos[1:]).max()
        worst = max(worst, err)
    print(f"FK max |Humanoid_Batch - mujoco| body position error over {T} random poses: {worst:.2e} m")
    print(f"dof round-trip max error: {np.abs(fk_dof - dof).max():.2e} rad")
    assert worst < 1e-4, "FK mismatch: check body quats / axes in the MJCF"
    assert np.abs(fk_dof - dof).max() < 1e-5
    print("OK: dual_fr3.xml is consistent for MuJoCo and Humanoid_Batch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
