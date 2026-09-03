#!/usr/bin/env python3
"""Smallest possible Isaac Lab check under the SONIC user-site: ground plane, sim.reset,
20 steps. Separates 'the app/PhysX hangs on this pod' from 'our articulation hangs'.

    cd ~/GR00T-WholeBodyControl && PYTHONUSERBASE=~/env/pyuser-sonic /isaac-sim/python.sh -u \
        ~/code/franka-sonic/harness/lane_b/probe_isaac_empty.py --headless
"""

import argparse
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--with-robot", action="store_true", help="also spawn the dual-FR3 URDF")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
t0 = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"[probe] app up after {time.time() - t0:.1f}s", flush=True)

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
print(f"[probe] sim context {time.time() - t0:.1f}s device={args.device}", flush=True)
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
robot = None
if args.with_robot:
    from isaaclab.assets import Articulation

    from gear_sonic.envs.manager_env.robots import dual_fr3

    robot = Articulation(dual_fr3.DUAL_FR3_CFG.replace(prim_path="/World/Robot"))
    print(f"[probe] articulation cfg spawned {time.time() - t0:.1f}s", flush=True)
print("[probe] calling sim.reset()", flush=True)
sim.reset()
print(f"[probe] sim.reset done {time.time() - t0:.1f}s", flush=True)
if robot is not None:
    print("[probe] bodies:", robot.body_names, flush=True)
    print("[probe] joints:", robot.joint_names, flush=True)
    print("[probe] fixed_base:", robot.is_fixed_base, flush=True)
for i in range(20):
    sim.step()
print(f"[probe] 20 steps done {time.time() - t0:.1f}s", flush=True)
print("[probe] PROBE DONE", flush=True)
# simulation_app.close() never returns headless on this pod; hard-exit like gear_sonic does.
import os  # noqa: E402

sys.stdout.flush()
os._exit(0)
