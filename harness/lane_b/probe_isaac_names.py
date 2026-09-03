#!/usr/bin/env python3
"""Spawn the dual-FR3 articulation in Isaac Lab (headless) and print what PhysX sees.

    cd ~/GR00T-WholeBodyControl && PYTHONUSERBASE=~/env/pyuser-sonic /isaac-sim/python.sh \
        ~/code/franka-sonic/harness/lane_b/probe_isaac_names.py --headless

Prints body names, joint names, fixed-base flag, masses, joint limits and the
default joint targets; asserts the orders in robots/dual_fr3.py; then holds the
ready pose for 2 s and reports the joint drift (gravity sag under the PD gains) and
the two flange (link7) world positions.
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--no-gravity", action="store_true", help="hold test without gravity")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from gear_sonic.envs.manager_env.robots import dual_fr3  # noqa: E402


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

    # spawn from the URDF exactly as training does (UrdfFileCfg converts to USD lazily)
    print("URDF:", dual_fr3.DUAL_FR3_URDF, os.path.exists(dual_fr3.DUAL_FR3_URDF))
    cfg = dual_fr3.DUAL_FR3_CFG.replace(prim_path="/World/Robot")
    if args.no_gravity:
        cfg.spawn.rigid_props.disable_gravity = True
        print("GRAVITY DISABLED for this probe")
    robot = Articulation(cfg)
    sim.reset()
    # like the env's reset: put the joints AT the default pose (the USD starts at q=0)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),
                                   robot.data.default_joint_vel.clone())
    robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7].clone())
    robot.reset()
    sim.step()
    robot.update(sim.get_physics_dt())
    names0 = list(robot.body_names)
    for side in ("left", "right"):
        i = names0.index(f"{side}_fr3_link7")
        print(f"RESET {side} link7 world pos (expect MuJoCo home {side}: "
              f"[+-0.4767, -0.308, 0.5167]):", robot.data.body_pos_w[0, i].cpu().numpy().round(4).tolist())
    v = robot.root_physx_view
    for label, fn in (("max_forces", "get_dof_max_forces"), ("friction", "get_dof_friction_coefficients"),
                      ("stiffness", "get_dof_stiffnesses"), ("damping", "get_dof_dampings"),
                      ("armature", "get_dof_armatures"), ("max_vel", "get_dof_max_velocities")):
        if hasattr(v, fn):
            print(f"PHYSX dof {label}:", getattr(v, fn)()[0].cpu().numpy().round(3).tolist())
    print("USD CACHE:", dual_fr3.DUAL_FR3_USD_DIR, os.listdir(dual_fr3.DUAL_FR3_USD_DIR)
          if os.path.isdir(dual_fr3.DUAL_FR3_USD_DIR) else "missing")
    print("ISAAC body_names :", robot.body_names)
    print("ISAAC joint_names:", robot.joint_names)
    print("ISAAC is_fixed_base:", robot.is_fixed_base, "num_bodies:", robot.num_bodies,
          "num_joints:", robot.num_joints)
    masses = robot.root_physx_view.get_masses()[0].tolist()
    print("ISAAC masses:", [round(m, 3) for m in masses])
    lim = robot.data.joint_pos_limits[0].cpu().numpy()
    print("ISAAC joint limits lo:", lim[:, 0].round(3).tolist())
    print("ISAAC joint limits hi:", lim[:, 1].round(3).tolist())
    print("ISAAC default_joint_pos:", robot.data.default_joint_pos[0].cpu().numpy().round(3).tolist())
    print("ISAAC stiffness:", robot.data.joint_stiffness[0].cpu().numpy().round(1).tolist())
    print("ISAAC damping:", robot.data.joint_damping[0].cpu().numpy().round(1).tolist())

    ok_b = list(robot.body_names) == dual_fr3.DUAL_FR3_ISAACLAB_JOINTS
    ok_j = list(robot.joint_names) == dual_fr3.DUAL_FR3_ISAACLAB_JOINT_NAMES
    print("ORDER CHECK bodies:", "OK" if ok_b else "MISMATCH", " joints:", "OK" if ok_j else "MISMATCH")

    target = robot.data.default_joint_pos.clone()
    dt = sim.get_physics_dt()
    for k in range(400):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        if k in (0, 4, 20, 100, 399):
            d = (robot.data.joint_pos - target)[0].cpu().numpy()
            tq = robot.data.applied_torque[0].cpu().numpy()
            print(f"HOLD step {k}: drift j4 L/R = {d[6]:.4f}/{d[7]:.4f}  applied torque j4 L/R = "
                  f"{tq[6]:.1f}/{tq[7]:.1f}  j2 L/R torque = {tq[2]:.1f}/{tq[3]:.1f}")
    drift = (robot.data.joint_pos - target)[0].cpu().numpy()
    print("HOLD 2 s: joint drift from target (rad):", drift.round(4).tolist())
    print("HOLD 2 s: max |drift| =", float(abs(drift).max()))
    names = list(robot.body_names)
    for side in ("left", "right"):
        i = names.index(f"{side}_fr3_link7")
        print(f"HOLD {side} link7 world pos:", robot.data.body_pos_w[0, i].cpu().numpy().round(4).tolist())
    i = names.index("base")
    print("HOLD base world pos:", robot.data.body_pos_w[0, i].cpu().numpy().round(4).tolist(),
          "root_pos_w:", robot.data.root_pos_w[0].cpu().numpy().round(4).tolist())
    print("PROBE DONE")


if __name__ == "__main__":
    main()
    # simulation_app.close() never returns in this headless configuration on the pod
    # (2026-09-03); gear_sonic's own scripts end with os._exit(0) for the same reason.
    import sys as _sys

    _sys.stdout.flush()
    os._exit(0)
