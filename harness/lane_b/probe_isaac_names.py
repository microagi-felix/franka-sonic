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

    # 1. MJCF -> USD (the Kit experience does not enable the importer; do it here)
    from isaacsim.core.utils.extensions import enable_extension

    ok = enable_extension("isaacsim.asset.importer.mjcf")
    print("ENABLE isaacsim.asset.importer.mjcf:", ok)
    from isaaclab.sim.converters import MjcfConverter

    # The Isaac Sim 5.1 importer writes a layered USD (configuration/*.usd sublayers) and
    # resolves those sublayers wrongly when dest_path is relative -> absolute paths here.
    spawn_cfg = dual_fr3.DUAL_FR3_MJCF_SPAWN.replace(
        asset_path=os.path.abspath(dual_fr3.DUAL_FR3_MJCF_SPAWN.asset_path),
        usd_dir=os.path.abspath(dual_fr3.DUAL_FR3_USD_DIR),
    )
    conv = MjcfConverter(spawn_cfg)
    print("CONVERTED USD:", conv.usd_path, os.path.getsize(conv.usd_path), "bytes")
    assert conv.usd_path.endswith("dual_fr3.usd") and os.path.exists(conv.usd_path), conv.usd_path
    assert os.path.abspath(conv.usd_path) == os.path.abspath(dual_fr3.DUAL_FR3_USD), (
        conv.usd_path, dual_fr3.DUAL_FR3_USD)

    # 2. spawn from the USD exactly as training does
    robot = Articulation(dual_fr3.DUAL_FR3_CFG.replace(prim_path="/World/Robot"))
    sim.reset()
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
    for _ in range(400):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
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
    simulation_app.close()
