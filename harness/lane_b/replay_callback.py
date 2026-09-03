"""Decoder-replay callback for gear_sonic's eval_agent_trl.py (P2 WP 2.7b).

Plugged into the stock evaluation loop through hydra overrides (no upstream edit):

    PYTHONPATH=~/code/franka-sonic/harness/lane_b \
    /isaac-sim/python.sh gear_sonic/eval_agent_trl.py checkpoint=<…/last.pt> headless=True \
        num_envs=1 use_encoder=g1 '+eval_callbacks=[replay]' \
        +callbacks.replay._target_=replay_callback.ReplayCallback \
        +callbacks.replay.out_json=<run>/out/replay.json +callbacks.replay.clip=<name> \
        ++manager_env.commands.motion.motion_lib_cfg.motion_file=<dir with ONE demo pkl> \
        ++manager_env.commands.motion.start_from_first_frame=True

The eval loop calls ``eval_step(env, results)`` after every ``env.step``; the policy runs
closed-loop (mean action) on the single demo clip from its first frame. Per step we record
the reference joint positions the encoder saw (``command.joint_pos`` — the motion
library's dof for the current frame, IsaacLab joint order), the joint TARGETS the decoder
produced (``processed_actions`` = default + scale * action) and the joints the robot
actually reached. When the clip ends (time_out) or the tracker terminates early, the
errors are written to ``out_json`` and the loop is told to exit.

mean_joint_error_rad = mean |decoder joint target - reference joint| over frames x 14 joints
(the gate's number, PLAN.md P2); mean_measured_joint_error_rad is the closed-loop tracking
error of the simulated arms, and error_body_pos_m the env's own body-position metric.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch


class ReplayCallback:
    def __init__(self, out_json: str, clip: str = "", max_steps: int = 5000):
        self.out_json = out_json
        self.clip = clip
        self.max_steps = int(max_steps)
        self.model = None  # eval_agent_trl sets this when the attribute exists
        self.targets, self.measured, self.refs = [], [], []
        self.body_err, self.joint_err_env = [], []
        self.t0 = time.time()
        self.finished = False

    # eval_agent_trl calls this once before the loop (TrainerCallback signature)
    def on_step_end(self, *args, **kwargs):  # noqa: D401
        return None

    @staticmethod
    def _base(env):
        return env.env if hasattr(env, "env") else env

    def eval_step(self, env, results) -> bool:
        if self.finished:
            return True
        base = self._base(env)
        obs, rew, dones, infos = results[0], results[1], results[2], results[3]
        cmd = base.command_manager.get_term("motion")
        robot = base.scene["robot"]
        act = base.action_manager.get_term("joint_pos")
        ref = cmd.joint_pos[0].detach().cpu().numpy().copy()
        tgt = act.processed_actions[0].detach().cpu().numpy().copy()
        q = robot.data.joint_pos[0].detach().cpu().numpy().copy()
        self.refs.append(ref)
        self.targets.append(tgt)
        self.measured.append(q)
        m = getattr(cmd, "metrics", {})
        if "error_body_pos" in m:
            self.body_err.append(float(m["error_body_pos"][0]))
        if "error_joint_pos" in m:
            self.joint_err_env.append(float(m["error_joint_pos"][0]))

        d = dones.reshape(-1)[0]
        done = bool(d.item()) if torch.is_tensor(d) else bool(d)
        if done or len(self.refs) >= self.max_steps:
            timed_out = None
            tm = getattr(base, "termination_manager", None)
            if tm is not None and hasattr(tm, "time_outs"):
                timed_out = bool(tm.time_outs.reshape(-1)[0].item())
            self._write(timed_out)
            self.finished = True
            return True
        return False

    def _write(self, timed_out):
        refs = np.stack(self.refs)
        tgts = np.stack(self.targets)
        meas = np.stack(self.measured)
        # the target applied at step t aims at the reference of the frame reached at t;
        # both arrays are recorded post-step, so index-aligned comparison is the right one
        e_tgt = np.abs(tgts - refs)
        e_meas = np.abs(meas - refs)
        joint_names = None
        out = {
            "mean_joint_error_rad": float(e_tgt.mean()),
            "mean_measured_joint_error_rad": float(e_meas.mean()),
            "max_joint_error_rad": float(e_tgt.max()),
            "max_measured_joint_error_rad": float(e_meas.max()),
            "per_joint_target_error_rad": e_tgt.mean(0).round(4).tolist(),
            "per_joint_measured_error_rad": e_meas.mean(0).round(4).tolist(),
            "env_error_joint_pos_rad_mean": float(np.mean(self.joint_err_env)) if self.joint_err_env else None,
            "env_error_body_pos_m_mean": float(np.mean(self.body_err)) if self.body_err else None,
            "clip": self.clip,
            "n_frames": int(len(self.refs)),
            "control_hz": 50,
            "ended_by_time_out": timed_out,
            "terminated_early": (not timed_out) if timed_out is not None else None,
            "joint_order": "isaaclab (left_j1, right_j1, left_j2, ...)",
            "joint6_convention": "sonic q6 = fr3 q6 - 2.5307",
            "wall_clock_s": round(time.time() - self.t0, 1),
        }
        os.makedirs(os.path.dirname(self.out_json), exist_ok=True)
        with open(self.out_json, "w") as fh:
            json.dump(out, fh, indent=2)
        npz = os.path.splitext(self.out_json)[0] + "_trajectories.npz"
        np.savez_compressed(npz, reference=refs, targets=tgts, measured=meas)
        print(f"[replay] wrote {self.out_json}: {json.dumps(out)}", flush=True)
