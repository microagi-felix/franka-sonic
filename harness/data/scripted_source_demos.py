"""Scripted expert (franka-sonic copy): record Mimic SOURCE demos of the handover without teleop.

Copy of ~/code/franka-bimanual-isaac-sim/mimic/scripts/scripted_source_demos.py with ONE change
for the angled rig: the RIGHT arm uses the mirror image of the left arm's grasp rotation
(mirrored across the rig's x = MID_X symmetry plane). On the angled rig (bases rolled -45/+45 deg)
the upstream world-vertical yaw-90 grasp is reachable for the left arm but stalls ~5 cm above the
centre pad for the right arm (differential IK hits a joint limit); the mirrored rotation makes the
right arm's motion the mirror of the left arm's successful one. Finger axis is unchanged (world x).
Run it with the sim user-site from the franka repo root:

    PYTHONUSERBASE=~/env/pyuser-fr3 /isaac-sim/python.sh harness/data/scripted_source_demos.py \
        --headless --num_demos 10 --rate 50 --dataset <run>/out/sources.hdf5


Drives env-frame waypoints through the Mimic env's own target_eef_pose_to_action
(so it exercises the exact frame math datagen will use) and records episodes with
Isaac Lab's ActionStateRecorderManager (initial_state + actions), the format
annotate_demos.py replays. Success-only export.

    pixi run python mimic/scripts/scripted_source_demos.py --headless --num_demos 10 \
        --dataset ./datasets/mimic/sources.hdf5
"""

import argparse
import os

from frankas_assets.specs.task import add_task_argument

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
add_task_argument(parser, default="Isaac-Stack-Cube-DualFranka-IK-Abs-Mimic-v0")
parser.add_argument("--num_demos", type=int, default=10)
parser.add_argument("--dataset", default="./datasets/mimic/sources.hdf5")
parser.add_argument("--rate", type=float, default=30.0)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import traceback  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
import mimic  # noqa: E402,F401  (registers the Mimic env)
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from tasks.stack_fr3.dual_stack_env_cfg import START_POS, SLOT_CENTER_X, SLOT_END_X, SLOT_Y  # noqa: E402
from frankas_assets.rig import RIG, log_rig, stamp_rig  # noqa: E402

# Gripper-down, yaw 90 (fingers close across the block's 4 cm width): Rz(90) @ Rx(180).
# NOTE: these world-vertical grasp/hover/place waypoints are tuned for the FLAT rig; on
# the angled rig they may be IK-unreachable from the rolled bases (the run then reports
# 0 saved and exits non-zero rather than writing a silent empty dataset).
R_GRASP = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
# Right arm: mirror of R_GRASP across the x-normal plane (M R M, M = diag(-1, 1, 1)) = Rz(-90) @ Rx(180).
R_GRASP_RIGHT = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
R_ARM = {"left": R_GRASP, "right": R_GRASP_RIGHT}
Z_HOVER, Z_GRASP, Z_PLACE = 0.25, 0.030, 0.042  # TCP z; block center 0.02, top 0.04
OPEN, CLOSE = 1.0, -1.0


def pose(x, y, z, device, arm="left"):
    m = torch.eye(4, device=device)
    m[:3, :3] = R_ARM[arm].to(device)
    m[0, 3], m[1, 3], m[2, 3] = x, y, z
    return m


def main() -> None:
    log_rig("expert")
    rate = args.rate
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
    env_cfg.decimation = max(1, round(100.0 / rate))
    env_cfg.sim.dt = 1.0 / (rate * env_cfg.decimation)
    # Sources: tight spawn (generation uses the cfg's wide range at its own resets).
    pr = env_cfg.events.init_cube.params["pose_range"]
    cx, cy = START_POS
    pr["x"], pr["y"], pr["yaw"] = (cx - 0.01, cx + 0.01), (cy - 0.01, cy + 0.01), (-0.1, 0.1)
    # Success term: extract + evaluate manually (no auto-termination mid-demo).
    success_term = env_cfg.terminations.success
    env_cfg.terminations = None
    # Recorder: Isaac format (initial_state + actions), we export success-only manually.
    out = os.path.abspath(args.dataset)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = os.path.dirname(out)
    env_cfg.recorders.dataset_filename = os.path.splitext(os.path.basename(out))[0]
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    device = env.device

    def hold_pose(eef):
        return env.get_robot_eef_pose(eef, env_ids=[0])[0]

    def step_action(poses, grips):
        act = env.target_eef_pose_to_action(
            {k: v for k, v in poses.items()}, {k: torch.tensor([g], device=device) for k, g in grips.items()}
        )
        env.step(act.unsqueeze(0))

    # The upstream step budgets (240 / 20 / 15 env steps) were tuned at 30 Hz. At 50 Hz an env
    # step is 2 physics substeps instead of 3, so the same budget is 1.67x less wall-time and the
    # right arm's hover waypoint stalled at dist ~0.035 > tol 0.02 (3/6 sources on the aborted
    # attempt). Scale every budget with the rate so the expert gets the same simulated time.
    scale = rate / 30.0
    GOTO_STEPS, GRIP_STEPS, SETTLE_STEPS = (int(round(240 * scale)), int(round(20 * scale)),
                                            int(round(15 * scale)))

    def goto(eef, target, grip, other, other_grip, tol=0.02, max_steps=None):
        """Drive one eef to target while the other holds; returns success."""
        dist = 1e9
        for _ in range(max_steps or GOTO_STEPS):
            poses = {eef: target, other: hold_pose(other)}
            step_action(poses, {eef: grip, other: other_grip})
            cur = env.get_robot_eef_pose(eef, env_ids=[0])[0, :3, 3]
            dist = float(torch.linalg.norm(cur - target[:3, 3]))
            if dist < tol:
                return True
        t = [round(float(x), 3) for x in target[:3, 3]]
        print(f"[expert]   goto {eef} -> {t} NOT reached (dist {dist:.3f})", flush=True)
        return False

    def set_grip(eef, grip, other, other_grip, steps=None):
        tgt = hold_pose(eef)
        for _ in range(steps or GRIP_STEPS):
            step_action({eef: tgt, other: hold_pose(other)}, {eef: grip, other: other_grip})
        # debug: fingers + tcp + cube after the grip change
        rob = env.scene["robot" if eef == "left" else "robot_2"]
        ids, _ = rob.find_joints(["fr3_finger_joint1", "fr3_finger_joint2"])
        fingers = [round(float(rob.data.joint_pos[0, i]), 4) for i in ids]
        tcp = [round(float(x), 3) for x in env.get_robot_eef_pose(eef, env_ids=[0])[0, :3, 3]]
        cube = [round(float(x), 3) for x in env.get_object_poses()["cube_1"][0][:3, 3]]
        print(f"[expert]   set_grip {eef} {grip}: fingers={fingers} tcp={tcp} cube={cube}", flush=True)

    saved = 0
    attempt = 0
    while saved < args.num_demos and attempt < args.num_demos * 3:
        attempt += 1
        env.reset()
        ok = True
        cube = env.get_object_poses()["cube_1"][0]
        cx, cy = float(cube[0, 3]), float(cube[1, 3])

        # --- left arm: grasp at spawn -> place middle (right holds) ---
        L, R = "left", "right"
        seq_left = [
            (pose(cx, cy, Z_HOVER, device), OPEN, 0.02),
            (pose(cx, cy, Z_GRASP, device), OPEN, 0.006),
            ("grip", CLOSE, None),
            (pose(cx, cy, Z_HOVER, device), CLOSE, 0.02),
            (pose(SLOT_CENTER_X, SLOT_Y, Z_HOVER, device), CLOSE, 0.02),
            (pose(SLOT_CENTER_X, SLOT_Y, Z_PLACE, device), CLOSE, 0.007),
            ("grip", OPEN, None),
            (pose(SLOT_CENTER_X, SLOT_Y, Z_HOVER, device), OPEN, 0.02),
        ]
        for tgt, g, tol in seq_left:
            if tgt == "grip":
                set_grip(L, g, R, OPEN)
            else:
                ok = goto(L, tgt, g, R, OPEN, tol=tol) and ok
        # re-read where the block actually landed
        cube = env.get_object_poses()["cube_1"][0]
        mx, my = float(cube[0, 3]), float(cube[1, 3])
        print(f"[expert]   after left phase, cube at {[round(float(x),3) for x in cube[:3,3]]}", flush=True)

        # --- right arm: grasp from middle -> place right (left holds, open) ---
        seq_right = [
            (pose(mx, my, Z_HOVER, device, R), OPEN, 0.02),
            (pose(mx, my, Z_GRASP, device, R), OPEN, 0.006),
            ("grip", CLOSE, None),
            (pose(mx, my, Z_HOVER, device, R), CLOSE, 0.02),
            (pose(SLOT_END_X, SLOT_Y, Z_HOVER, device, R), CLOSE, 0.02),
            (pose(SLOT_END_X, SLOT_Y, Z_PLACE, device, R), CLOSE, 0.007),
            ("grip", OPEN, None),
            (pose(SLOT_END_X, SLOT_Y, Z_HOVER, device, R), OPEN, 0.02),
        ]
        for tgt, g, tol in seq_right:
            if tgt == "grip":
                set_grip(R, g, L, OPEN)
            else:
                ok = goto(R, tgt, g, L, OPEN, tol=tol) and ok

        # settle + manual success check
        for _ in range(SETTLE_STEPS):
            step_action({L: hold_pose(L), R: hold_pose(R)}, {L: OPEN, R: OPEN})
        cube_end = env.get_object_poses()["cube_1"][0]
        print(f"[expert]   cube end pos: {[round(float(x),3) for x in cube_end[:3,3]]} "
              f"(end zone ({SLOT_END_X},{SLOT_Y}) r=0.10) reached_all={ok}", flush=True)
        sig = env.get_subtask_term_signals()
        print(f"[expert]   signals: { {k: bool(v[0]) for k, v in sig.items()} }", flush=True)
        success = bool(success_term.func(env, **success_term.params)[0]) if ok else False
        env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
        env.recorder_manager.set_success_to_episodes(
            [0], torch.tensor([[success]], dtype=torch.bool, device=device)
        )
        if success:
            env.recorder_manager.export_episodes([0])
            saved += 1
        print(f"[expert] attempt {attempt}: {'SAVED' if success else 'failed'} ({saved}/{args.num_demos})", flush=True)

    if saved > 0:
        with __import__("h5py").File(out, "a") as _f:  # stamp the rig for the MimicGen pipeline
            stamp_rig(_f)
    print(f"EXPERT_DONE: {saved}/{args.num_demos} source demos -> {out}", flush=True)
    env.close()
    if saved == 0:
        raise SystemExit(
            f"[expert] 0/{args.num_demos} demos succeeded on rig {RIG!r} — waypoints are flat-tuned; "
            f"retune R_GRASP/Z_* for this rig or run with FR3_RIG=flat. No dataset written."
        )


try:
    main()
except Exception:
    traceback.print_exc()
finally:
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
