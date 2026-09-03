"""Extra reward terms for the dual-FR3 SONIC embodiment (franka-sonic lane B).

Referenced from the experiment yaml as ``func: sonic_rewards:<name>`` (harness/lane_b must
be on PYTHONPATH — bakeoff's sonic_env() does that). The terms are mounted on slots that
gear_sonic's RewardsCfg already has, because custom_instantiate builds RewardsCfg(**dict)
and an unknown attribute would raise:

  * P2 (jp2):  tracking_joint_pos_error  -> slot tracking_relative_body_ori_weighted
  * P5 (jp3):  tracking_joint_space      -> slot feet_acc            (nulled on the FR3)
               tracking_joint_vel        -> slot undesired_contacts  (nulled on the FR3)
               (tracking_relative_body_ori_weighted goes back to its upstream function)

Why: SONIC's rewards are body-position/orientation based. On the FR3 the tracked link
origins (link2, link4, link7) sit on or near the joint-1 / joint-7 axes, so the joint
angles are only weakly pinned by positions (run lane_b/2026-09-03_sonic_rl-3: body error
6 cm but joint error 0.75 rad, body-rotation error 0.82 rad). The decoder is judged on
joint-target fidelity (gates p2/p5), hence direct joint-space terms.

P5 lesson from jp2: exp(-mean_j dq_j^2 / 1.0^2) is too blunt — one wrapped joint
(dq ~ 3 rad, L_j7 in the replay) saturates the mean and the wide kernel has almost no
gradient near zero. jp3 therefore takes the per-joint MEAN of per-joint kernels at two
scales, so a single bad joint costs at most 1/14 of the term and the tight kernel keeps
pulling once a joint is within ~0.3 rad.
"""

from __future__ import annotations

import torch


def _ref_and_robot(env, command_name: str, what: str):
    """(reference, robot) joint tensors in IsaacLab joint order, trimmed to the motion DoF."""
    command = env.command_manager.get_term(command_name)
    robot = env.scene["robot"].data
    if what == "pos":
        ref, q = command.joint_pos, robot.joint_pos
    else:
        ref, q = command.joint_vel, robot.joint_vel
    if q.shape[-1] != ref.shape[-1]:
        q = q[:, : ref.shape[-1]]
    return ref, q


def tracking_joint_pos_error(env, command_name: str, std: float) -> torch.Tensor:
    """jp2: exp(-mean_j (q_j - q_ref_j)^2 / std^2): reference joints from the motion command."""
    ref, q = _ref_and_robot(env, command_name, "pos")
    err = torch.square(q - ref).mean(dim=-1)
    return torch.exp(-err / (std**2))


def tracking_joint_space(
    env,
    command_name: str,
    std_wide: float = 0.5,
    std_tight: float = 0.15,
    w_wide: float = 0.5,
    w_tight: float = 0.5,
    std_far: float = 1.5,
    w_far: float = 0.0,
) -> torch.Tensor:
    """jp3: w_wide * mean_j exp(-dq_j^2/std_wide^2) + w_tight * mean_j exp(-dq_j^2/std_tight^2).

    Per-joint kernel MEAN (not the exp of the mean square): one wrapped joint must not zero
    the whole term. dq = q - q_ref in IsaacLab joint order, exactly as tracking_joint_pos_error.
    jp6 adds an optional third, far scale (std_far, w_far; w_far=0 keeps jp3's behaviour) that
    is still alive at 1-2 rad, where the 0.5/0.15 kernels are flat.
    """
    ref, q = _ref_and_robot(env, command_name, "pos")
    sq = torch.square(q - ref)
    wide = torch.exp(-sq / (std_wide**2)).mean(dim=-1)
    tight = torch.exp(-sq / (std_tight**2)).mean(dim=-1)
    out = w_wide * wide + w_tight * tight
    if w_far:
        out = out + w_far * torch.exp(-sq / (std_far**2)).mean(dim=-1)
    return out


def tracking_joint_vel(env, command_name: str, std: float = 2.0) -> torch.Tensor:
    """jp3: exp(-mean_j (qd_j - qd_ref_j)^2 / std^2); reference velocities from the motion
    command (the motion library carries finite-difference joint velocities)."""
    ref, qd = _ref_and_robot(env, command_name, "vel")
    err = torch.square(qd - ref).mean(dim=-1)
    return torch.exp(-err / (std**2))


# --------------------------------------------------------------------------- P5 jp4 (hedge)
# ckpt-1000 replay of jp3 (P5, 2026-09-03 15:15 UTC): the right arm tracks to <= 0.1 rad but the
# left arm never follows its big excursions (wrist roll 0.73 -> -3.02 rad ignored, shoulder swing
# to -1.75 rad ignored) and parks in another IK branch with the hand within 5 cm. In that state
# every Gaussian kernel is flat (|dq| > 1 rad -> exp(-4) and below; body-ori error 0.85 rad at
# std 0.4 -> 0.01), so PPO has no gradient out of it. jp4 adds the two DeepMimic-style levers:
# a linear (never flat) joint-error penalty and an early termination on joint error, so the
# wrong branch becomes a terminal state instead of a local optimum.


def joint_pos_error_l1(env, command_name: str) -> torch.Tensor:
    """jp4: mean_j |q_j - q_ref_j| (rad); mount with a NEGATIVE weight. Linear, so the gradient
    is the same at 3 rad as at 0.1 rad — the Gaussian kernels are flat beyond ~1 rad."""
    ref, q = _ref_and_robot(env, command_name, "pos")
    return torch.abs(q - ref).mean(dim=-1)


def bad_joint_pos(
    env,
    command_name: str,
    max_joint_threshold: float = 1.5,
    mean_threshold: float = 0.5,
) -> torch.Tensor:
    """jp4 termination: any joint further than `max_joint_threshold` rad from its reference, or
    the 14-joint mean |dq| above `mean_threshold` rad. Same dq as the tracking terms. Returns a
    bool tensor (num_envs,); time_out=False in the TerminationTermCfg (no bootstrap)."""
    ref, q = _ref_and_robot(env, command_name, "pos")
    err = torch.abs(q - ref)
    return (err.max(dim=-1).values > max_joint_threshold) | (err.mean(dim=-1) > mean_threshold)
