"""Extra reward terms for the dual-FR3 SONIC embodiment (franka-sonic lane B).

Referenced from the experiment yaml as ``func: sonic_rewards:tracking_joint_pos_error``
(harness/lane_b must be on PYTHONPATH — bakeoff's sonic_env() does that). Plugged into the
spare ``tracking_relative_body_ori_weighted`` slot of gear_sonic's RewardsCfg, because
custom_instantiate builds RewardsCfg(**dict) and an unknown attribute would raise.

Why: SONIC's rewards are body-position/orientation based. On the FR3 the tracked link
origins (link2, link4, link7) sit on or near the joint-1 / joint-7 axes, so the joint
angles are only weakly pinned by positions (run lane_b/2026-09-03_sonic_rl-3: body error
6 cm but joint error 0.75 rad, body-rotation error 0.82 rad). The decoder is judged on
joint-target fidelity (gate p2), hence a direct joint-space term.
"""

from __future__ import annotations

import torch


def tracking_joint_pos_error(env, command_name: str, std: float) -> torch.Tensor:
    """exp(-mean_j (q_j - q_ref_j)^2 / std^2): reference joints from the motion command."""
    command = env.command_manager.get_term(command_name)
    ref = command.joint_pos  # (N, num_dof), IsaacLab joint order (motion lib reordered)
    q = env.scene["robot"].data.joint_pos
    if q.shape[-1] != ref.shape[-1]:
        q = q[:, : ref.shape[-1]]
    err = torch.square(q - ref).mean(dim=-1)
    return torch.exp(-err / (std**2))
