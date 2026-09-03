"""Lane A's absolute joint label, derived one way for the dataset AND the A-oracle.

The export (harness/data/export_generated_50hz.py) stores, per step t, the measured joints
`joint_pos_*[t]` (pre-step) and the differential-IK controller's joint command
`joint_target_*[t]` that step t ran under. Which of these is "the recorded joint action" is a
decision (P1, 2026-09-03), measured on one generated episode replayed through the JointPos env:

  ik_target          raw IK command. Replays to success (oracle 6/6 milestones) but jumps by up
                     to ~10 rad — beyond the joint limits — at every segment start, because the
                     damped-least-squares IK saturates when the Cartesian waypoint is far. ~7 %
                     of frames move > 0.5 rad. Physics clamps it; a learner would not.
  next_state         q_{t+1}. Smooth, but the replay lags the demo (PD error is one step of
                     motion instead of a saturated command) and reaches only 4/6 milestones.
  ik_target_clamped  ik_target clipped to the FR3 joint limits. Same saturated command where it
                     matters (the drive is torque-limited long before a 3 rad error), sane range.
  ik_target_delta    ik_target_clamped with the per-step move q_t -> label bounded to
                     +-max_delta_rad; keeps the direction of a saturated command, bounds its size.

`arm_label()` is pure numpy so the GR00T venv (converter) and the Isaac python (oracle) share it.
"""

from __future__ import annotations

import numpy as np

# Franka FR3 joint position limits (rad), fr3_joint1..7 — the values Isaac's articulation
# enforces (frankas_assets pins the same numbers in its FR3 spec).
FR3_JOINT_LIMITS = np.array(
    [
        [-2.7437, 2.7437],
        [-1.7837, 1.7837],
        [-2.9007, 2.9007],
        [-3.0421, -0.1518],
        [-2.8065, 2.8065],
        [0.5445, 4.5169],
        [-3.0159, 3.0159],
    ],
    dtype=np.float32,
)

JOINT_LABELS = ("ik_target_clamped", "ik_target_delta", "ik_target", "next_state")
DEFAULT_JOINT_LABEL = "ik_target_clamped"
DEFAULT_MAX_DELTA_RAD = 0.3


def arm_label(
    joint_pos: np.ndarray,
    joint_target: np.ndarray,
    mode: str = DEFAULT_JOINT_LABEL,
    max_delta_rad: float = DEFAULT_MAX_DELTA_RAD,
) -> np.ndarray:
    """(T, 7) absolute joint label for one arm from the exported (T, 7) pos/target columns."""
    q = np.asarray(joint_pos, dtype=np.float32)
    tgt = np.asarray(joint_target, dtype=np.float32)
    if q.shape != tgt.shape or q.ndim != 2 or q.shape[1] != 7:
        raise ValueError(f"expected (T, 7) pos/target, got {q.shape} / {tgt.shape}")
    if mode not in JOINT_LABELS:
        raise ValueError(f"joint label {mode!r} not in {JOINT_LABELS}")
    if mode == "next_state":
        return np.concatenate([q[1:], q[-1:]], axis=0)
    if mode == "ik_target":
        return tgt.copy()
    lo, hi = FR3_JOINT_LIMITS[:, 0], FR3_JOINT_LIMITS[:, 1]
    out = np.clip(tgt, lo, hi)
    if mode == "ik_target_delta":
        out = q + np.clip(out - q, -max_delta_rad, max_delta_rad)
        out = np.clip(out, lo, hi)
    return out.astype(np.float32)


def describe(mode: str, max_delta_rad: float = DEFAULT_MAX_DELTA_RAD) -> str:
    if mode == "ik_target_delta":
        return f"{mode} (IK joint command clipped to FR3 limits, per-step move <= {max_delta_rad} rad)"
    if mode == "ik_target_clamped":
        return f"{mode} (IK joint command clipped to FR3 joint limits)"
    if mode == "ik_target":
        return f"{mode} (raw differential-IK joint command)"
    return f"{mode} (next measured joint position)"
