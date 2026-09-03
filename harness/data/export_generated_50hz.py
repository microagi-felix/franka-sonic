"""Replay MimicGen-generated episodes into the training HDF5 schema — franka-sonic variant.

Based on `mimic/scripts/export_generated.py` (franka-bimanual-isaac-sim @14f0d8a) with the
additions lane A / lane B need (docs/DATASET_SCHEMA.md "MimicGen export fields" + these):

- `--rate 50` (rev 3c) and `sim.render_interval = decimation`, so the cameras render every
  control step (upstream leaves the base cfg's render interval, which at decimation 2 would
  refresh the images only every 2.5 steps).
- observations are taken BEFORE the action that follows them (obs_t, action_t), not after
  (upstream pairs the post-step observation with the action that produced it).
- `obs/joint_target_left|right (T,7)`: the differential-IK controller's joint position targets
  (`Articulation.data.joint_pos_target` after `env.step`, the last physics substep's command) —
  lane A's ABSOLUTE joint labels, the thing the JointPos env consumes.
- `obs/top|wrist_left|wrist_right`: video-backed (h264 mp4, relative path string) at
  `--scale` of the native resolution (0.5 -> 640x360 / 424x240, cv2 INTER_AREA). Raw 720p uint8
  arrays for 80 episodes would be ~400 GB (rule j caps a write at 20 GB).
- `initial_cube_pose (7,)` per demo (env-frame pos + quat wxyz, from the generated episode's
  recorded initial state) so the A-oracle can respawn the block where the demo had it, and the
  attr `replay_success` (mimic.mdp.handover_success after a short settle) so a replay that no
  longer succeeds at this rate is visible and can be excluded downstream.
- `--shard i --num_shards n`: this process exports episodes i, i+n, i+2n, ... so several
  renderers share one GPU.

    PYTHONUSERBASE=~/env/pyuser-fr3 /isaac-sim/python.sh harness/data/export_generated_50hz.py \
        --input <run>/out/generated.hdf5 --output <run>/out/export/demos_shard0.hdf5 \
        --rate 50 --shard 0 --num_shards 4 --headless
"""

import argparse

from frankas_assets.specs.task import add_task_argument

from isaaclab.app import AppLauncher
from frankas_assets.camera_app import configure_camera_app_args

parser = argparse.ArgumentParser()
add_task_argument(parser, default="Isaac-Stack-Cube-DualFranka-IK-Abs-v0")
parser.add_argument("--input", required=True, help="Mimic-generated HDF5 (Isaac format)")
parser.add_argument("--output", required=True, help="training-schema HDF5 to write (video-backed)")
parser.add_argument("--max_episodes", type=int, default=0, help="0 = all (of this shard)")
parser.add_argument("--rate", type=float, default=50.0)
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--num_shards", type=int, default=1)
parser.add_argument("--scale", type=float, default=0.5, help="image downscale factor (INTER_AREA)")
parser.add_argument("--settle_steps", type=int, default=25, help="hold steps before the success check")
parser.add_argument("--no_video", action="store_true", help="store raw uint8 arrays (large!)")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
args.enable_cameras = True
configure_camera_app_args(args)
app = AppLauncher(args).app

import os  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
import tasks  # noqa: E402,F401
from frankas_assets.end_effectors.franka_hand import FINGER_OPEN_M  # noqa: E402
from frankas_assets.cameras import sync_wrist_cam_fabric  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.datasets import HDF5DatasetFileHandler  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from mimic.mdp import handover_success  # noqa: E402
from tasks.stack_fr3.dual_stack_env_cfg import SLOT_END_X, SLOT_Y  # noqa: E402
from teleop.recorder import EpisodeBuffer, next_demo_index, write_episode  # noqa: E402

ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]


def _resize(img: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def main() -> None:
    rate = args.rate
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
    env_cfg.decimation = max(1, round(100.0 / rate))
    env_cfg.sim.dt = 1.0 / (rate * env_cfg.decimation)
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.terminations = None  # replay must not be interrupted
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    print(f"[export] control {rate:g} Hz, physics dt {env_cfg.sim.dt:.4f}, decimation {env_cfg.decimation}, "
          f"render_interval {env_cfg.sim.render_interval}, image scale {args.scale}", flush=True)

    robots = {"left": env.scene["robot"], "right": env.scene["robot_2"]}
    arm_ids = {}
    finger_ids = {}
    for side, rob in robots.items():
        names = list(rob.joint_names)
        arm_ids[side] = [names.index(j) for j in ARM_JOINTS]
        finger_ids[side] = names.index("fr3_finger_joint1")
    print(f"[export] arm joint ids {arm_ids}, finger ids {finger_ids}", flush=True)
    cube_cfg = SceneEntityCfg("cube_1")

    def u8(t):
        return _resize(t[0].detach().cpu().numpy()[..., :3].astype(np.uint8), args.scale)

    def f32(t):
        return t[0].detach().cpu().numpy().astype(np.float32)

    def joints(side):
        jp = robots[side].data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
        grip = np.float32(np.clip(1.0 - jp[finger_ids[side]] / FINGER_OPEN_M, 0.0, 1.0))
        return jp[arm_ids[side]], np.array([grip], dtype=np.float32)

    def targets(side):
        return robots[side].data.joint_pos_target[0].detach().cpu().numpy().astype(np.float32)[arm_ids[side]]

    def obs_frame(obs):
        """Training-schema frame (mirrors sim_server.obs_frame) + obs/object, taken PRE-step."""
        p, c = obs["policy"], obs["rgb_camera"]
        jpL, gL = joints("left")
        jpR, gR = joints("right")
        cube = env.scene["cube_1"]
        obj = np.concatenate(
            [
                (cube.data.root_pos_w[0] - env.scene.env_origins[0]).detach().cpu().numpy(),
                cube.data.root_quat_w[0].detach().cpu().numpy(),
            ]
        ).astype(np.float32)
        return {
            "top": u8(c["top"]),
            "wrist_left": u8(c["wrist_left"]),
            "wrist_right": u8(c["wrist_right"]),
            "joint_pos_left": jpL,
            "joint_pos_right": jpR,
            "gripper_left": gL,
            "gripper_right": gR,
            "eef_left": np.concatenate([f32(p["eef_pos"]), f32(p["eef_quat"])]),
            "eef_right": np.concatenate([f32(p["eef_2_pos"]), f32(p["eef_2_quat"])]),
            "object": obj,
        }

    handler = HDF5DatasetFileHandler()
    handler.open(os.path.abspath(args.input))
    names = list(handler.get_episode_names())
    names = [n for i, n in enumerate(names) if i % args.num_shards == args.shard]
    if args.max_episodes:
        names = names[: args.max_episodes]
    print(f"[export] shard {args.shard}/{args.num_shards}: {len(names)} generated episodes -> {args.output}",
          flush=True)

    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    meta = {
        "fps": float(rate), "task": args.task, "source": "mimicgen", "rate_hz": float(rate),
        "image_scale": float(args.scale), "generated_file": os.path.abspath(args.input),
        "joint_target_source": "differential_ik Articulation.data.joint_pos_target after env.step",
        "obs_timing": "pre-step (obs_t pairs with action_t)",
    }
    exported, successes = 0, 0
    t_start = time.monotonic()
    for name in names:
        t0 = time.monotonic()
        ep = handler.load_episode(name, env.device)
        env.reset()  # clears managers; then set the exact recorded initial state
        init = ep.get_initial_state()
        obs, _ = env.reset_to(init, torch.tensor([0], device=env.device), is_relative=True)
        sync_wrist_cam_fabric(env)
        buf = EpisodeBuffer()
        tgt_l, tgt_r = [], []
        last_action = None
        while True:
            action = ep.get_next_action()
            if action is None:
                break
            frame = obs_frame(obs)  # observation BEFORE this action
            obs, *_ = env.step(action.unsqueeze(0))
            tgt_l.append(targets("left"))  # the joint command this step ran under
            tgt_r.append(targets("right"))
            buf.add(frame, action.detach().cpu().numpy().astype(np.float32))
            last_action = action
        # attach the joint targets as observation columns (same T as the buffer)
        for fr, tl, tr in zip(buf._obs, tgt_l, tgt_r):
            fr["joint_target_left"] = tl
            fr["joint_target_right"] = tr
        # settle, then the task's own success term (the block must be at rest in the end zone)
        if last_action is not None:
            for _ in range(args.settle_steps):
                env.step(last_action.unsqueeze(0))
        success = bool(handover_success(env, cube_cfg, (SLOT_END_X, SLOT_Y))[0])
        successes += int(success)
        idx = next_demo_index(out)
        n = write_episode(out, idx, buf, meta, video=not args.no_video)
        with h5py.File(out, "a") as f:
            g = f["data"][f"demo_{idx}"]
            g.attrs["replay_success"] = success
            g.attrs["source_episode"] = name
            cube_pose = init["rigid_object"]["cube_1"]["root_pose"]
            g.create_dataset("initial_cube_pose", data=cube_pose.reshape(-1).detach().cpu().numpy().astype(np.float32))
        exported += 1
        print(f"[export] {name} -> demo_{idx} ({n} frames, replay_success={success}, "
              f"{time.monotonic() - t0:.0f}s)", flush=True)
    print(f"EXPORT_DONE: {exported} episodes ({successes} replay successes) -> {out} "
          f"in {(time.monotonic() - t_start)/60:.1f} min", flush=True)
    env.close()


try:
    main()
except Exception:
    traceback.print_exc()
finally:
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
