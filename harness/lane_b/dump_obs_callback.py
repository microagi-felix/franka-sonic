"""Observation-dump callback for gear_sonic's eval_agent_trl.py (P3 WP 3.1 validation).

Same plumbing as harness/lane_b/replay_callback.py (hydra ``+eval_callbacks=[dump]`` with
``+callbacks.dump._target_=dump_obs_callback.DumpObsCallback``), but instead of scoring the
replay it records, for every env step of ONE demo clip played from its first frame, exactly
what the trained policy saw and did:

    tokenizer_obs   (T, 1401)  the env's flat tokenizer group (encoder input, env layout)
    actor_obs       (T, 480)   the decoder's proprio vector (10-step histories)
    cmd_mf          (T, 280)   command.command_multi_future  (the g1 encoder input, flat)
    anchor_mf       (T, 60)    command.root_rot_dif_l_multi_future
    time_steps      (T,)       command.time_steps after the step (obs row k has time step k+1)
    future_steps    (T, 10)    command.future_time_steps (absolute motion-lib frame indices)
    ref_joint_pos   (T, 14)    command.joint_pos (reference, isaac order)
    joint_pos/vel   (T, 14)    robot state after the step (isaac order, SONIC angles)
    raw_action      (T, 14)    action applied AT this step (policy mean), isaac order
    processed_action(T, 14)    default + scale * raw_action
    default_joint_pos (14,), joint_names, body_names

Row k is written after env.step k: obs arrays are obs_{k+1}, raw_action is a_k. The offline
labeller (harness/lane_b/label_tokens.py check) compares its own encoder inputs against
cmd_mf/anchor_mf at the same time steps, runs the ONNX encoder+decoder on the recorded obs
and compares against raw_action, and validates the decoder runtime's proprio bookkeeping
against actor_obs. Writes ``out_npz`` when the clip times out (or ``max_steps``).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch


def _np(x) -> np.ndarray:
    return x.detach().cpu().numpy().copy() if torch.is_tensor(x) else np.asarray(x)


class DumpObsCallback:
    def __init__(self, out_npz: str, clip: str = "", max_steps: int = 5000):
        self.out_npz = out_npz
        self.clip = clip
        self.max_steps = int(max_steps)
        self.model = None
        self.rows: dict[str, list] = {}
        self.static: dict = {}
        self.t0 = time.time()
        self.finished = False

    def on_step_end(self, *args, **kwargs):  # TrainerCallback signature; nothing to do
        return None

    @staticmethod
    def _base(env):
        return env.env if hasattr(env, "env") else env

    def _add(self, key: str, value) -> None:
        self.rows.setdefault(key, []).append(_np(value).reshape(-1) if _np(value).ndim <= 2 else _np(value)[0])

    def eval_step(self, env, results) -> bool:
        if self.finished:
            return True
        base = self._base(env)
        obs, _rew, dones, _infos = results[0], results[1], results[2], results[3]
        cmd = base.command_manager.get_term("motion")
        robot = base.scene["robot"]
        act = base.action_manager.get_term("joint_pos")

        if isinstance(obs, dict):
            for k, v in obs.items():
                if torch.is_tensor(v) and v.dim() == 2:
                    self._add(f"obs__{k}", v[0])
        self._add("cmd_mf", cmd.command_multi_future[0])
        self._add("anchor_mf", cmd.root_rot_dif_l_multi_future[0])
        self._add("time_steps", cmd.time_steps[0:1])
        self._add("start_steps", cmd.motion_start_time_steps[0:1])
        self._add("future_steps", cmd.future_time_steps.view(cmd.num_envs, -1)[0])
        self._add("ref_joint_pos", cmd.joint_pos[0])
        self._add("joint_pos", robot.data.joint_pos[0])
        self._add("joint_vel", robot.data.joint_vel[0])
        self._add("raw_action", act.raw_actions[0])
        self._add("processed_action", act.processed_actions[0])
        self._add("robot_anchor_quat_w", cmd.robot_anchor_quat_w[0])
        if not self.static:
            self.static = {
                "default_joint_pos": _np(robot.data.default_joint_pos[0]),
                "joint_names": list(robot.joint_names),
                "body_names": list(robot.body_names),
                "action_scale": _np(act._scale[0]) if torch.is_tensor(getattr(act, "_scale", None)) else np.asarray(getattr(act, "_scale", np.nan)),
                "motion_num_steps": int(cmd.motion_num_steps[0].item()),
                "num_future_frames": int(cmd.num_future_frames),
                "frame_skips": int(cmd.frame_skips),
                "motion_key": str(cmd.motion_lib.curr_motion_keys[int(cmd.motion_ids[0].item())]),
            }

        d = dones.reshape(-1)[0]
        done = bool(d.item()) if torch.is_tensor(d) else bool(d)
        if done or len(self.rows["cmd_mf"]) >= self.max_steps:
            timed_out = None
            tm = getattr(base, "termination_manager", None)
            if tm is not None and hasattr(tm, "time_outs"):
                timed_out = bool(tm.time_outs.reshape(-1)[0].item())
            self._write(timed_out)
            self.finished = True
            return True
        return False

    def _write(self, timed_out) -> None:
        os.makedirs(os.path.dirname(self.out_npz), exist_ok=True)
        arrays = {k: np.stack(v) for k, v in self.rows.items()}
        meta = {
            "clip": self.clip,
            "n_rows": int(arrays["cmd_mf"].shape[0]),
            "ended_by_time_out": timed_out,
            "row_semantics": "row k: obs arrays = obs_{k+1} (after env.step k); raw_action = a_k",
            "wall_clock_s": round(time.time() - self.t0, 1),
            **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.static.items()},
        }
        np.savez_compressed(self.out_npz, **arrays, meta=json.dumps(meta))
        shapes = {k: list(v.shape) for k, v in arrays.items()}
        print(f"[dump] wrote {self.out_npz}: {json.dumps(shapes)} meta={json.dumps(meta)}", flush=True)
