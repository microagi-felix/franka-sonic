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
) -> torch.Tensor:
    """jp3: w_wide * mean_j exp(-dq_j^2/std_wide^2) + w_tight * mean_j exp(-dq_j^2/std_tight^2).

    Per-joint kernel MEAN (not the exp of the mean square): one wrapped joint must not zero
    the whole term. dq = q - q_ref in IsaacLab joint order, exactly as tracking_joint_pos_error.
    """
    ref, q = _ref_and_robot(env, command_name, "pos")
    sq = torch.square(q - ref)
    wide = torch.exp(-sq / (std_wide**2)).mean(dim=-1)
    tight = torch.exp(-sq / (std_tight**2)).mean(dim=-1)
    return w_wide * wide + w_tight * tight


def tracking_joint_vel(env, command_name: str, std: float = 2.0) -> torch.Tensor:
    """jp3: exp(-mean_j (qd_j - qd_ref_j)^2 / std^2); reference velocities from the motion
    command (the motion library carries finite-difference joint velocities)."""
    ref, qd = _ref_and_robot(env, command_name, "vel")
    err = torch.square(qd - ref).mean(dim=-1)
    return torch.exp(-err / (std**2))
