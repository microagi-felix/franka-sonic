"""Isaac-free constants for the dual-FR3 SONIC boundary (P3, lane B-2).

Everything lane B's labeller, decoder runtime, policy server and B-oracle need
to move between the three joint conventions, with no isaaclab / gear_sonic
import:

    wire     [Lq7 | Lg | Rq7 | Rg]          the sim's ZmqAct contract, FR3 angles
    mujoco   [left_j1..7 | right_j1..7]      dual_fr3.xml document order, SONIC angles
    isaac    [left_j1, right_j1, left_j2, …] PhysX breadth-first order, SONIC angles
                                             (the decoder's proprio and action order)

SONIC angles differ from FR3 angles in joint 6 only: dual_fr3.xml pre-rotates
link6 by Rz(J6_OFFSET), so ``q6_sonic = q6_fr3 - J6_OFFSET`` (P2 decision,
harness/lane_b/make_dual_fr3_xml.py). The name lists mirror
harness/lane_b/robots_dual_fr3.py (installed upstream, verified by
probe_isaac_names.py on 2026-09-03) — keep them identical.

ONNX contracts of lane_b/2026-09-03_export_onnx-3 (P2 handoff, verified by a
perturbation probe on 2026-09-03: with encoder_index 0 only the two "g1" slices
move the token):

    encoder  input  (1, 1391) = [encoder_index 1 | encoder_index one-hot 3 (ignored)
                                 | command_multi_future 280 | motion_anchor_ori_b 6
                                 | motion_anchor_ori_b_mf 60 | teleop/smpl features 1041]
             output (1, 64)    FSQ token, values on a 1/16 grid inside [-1, 1)
    decoder  input  (1, 544)  = [token 64 | proprio 480]
             output (1, 14)    raw action a, isaac order; joint target = default + 0.25 a

    proprio 480 = concat over terms [gravity_dir 3, base_ang_vel 3, joint_pos_rel 14,
                  joint_vel 14, last_action 14], each a 10-step history flattened
                  (history, dim) with the OLDEST entry first (IsaacLab CircularBuffer).
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------- joint orders
MUJOCO_JOINTS = [f"left_fr3_joint{i}" for i in range(1, 8)] + [
    f"right_fr3_joint{i}" for i in range(1, 8)
]
ISAAC_JOINTS = [f"{side}_fr3_joint{i}" for i in range(1, 8) for side in ("left", "right")]
MUJOCO_BODIES = ["base"] + [f"left_fr3_link{i}" for i in range(8)] + [
    f"right_fr3_link{i}" for i in range(8)
]
ISAAC_BODIES = ["base"] + [f"{side}_fr3_link{i}" for i in range(8) for side in ("left", "right")]

# output[i] = input[mapping[i]]  (gear_sonic IsaacLabMuJoCoConverter semantics)
ISAAC_TO_MUJOCO_DOF = [ISAAC_JOINTS.index(j) for j in MUJOCO_JOINTS]  # mujoco = isaac[this]
MUJOCO_TO_ISAAC_DOF = [MUJOCO_JOINTS.index(j) for j in ISAAC_JOINTS]  # isaac = mujoco[this]
ISAAC_TO_MUJOCO_BODY = [ISAAC_BODIES.index(b) for b in MUJOCO_BODIES]
MUJOCO_TO_ISAAC_BODY = [MUJOCO_BODIES.index(b) for b in ISAAC_BODIES]

ISAAC_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": ISAAC_BODIES,
    "isaaclab_to_mujoco_dof": ISAAC_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": MUJOCO_TO_ISAAC_DOF,
    "isaaclab_to_mujoco_body": ISAAC_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": MUJOCO_TO_ISAAC_BODY,
}

# the 7 bodies the SONIC command tracks (harness/lane_b/sonic_dual_fr3.yaml), anchor first
TRACKED_BODIES = [
    "base",
    "left_fr3_link2",
    "left_fr3_link4",
    "left_fr3_link7",
    "right_fr3_link2",
    "right_fr3_link4",
    "right_fr3_link7",
]
ANCHOR_BODY = "base"

N_DOF = 14
ARM_DOF = 7

# ----------------------------------------------------------------- conventions
J6_OFFSET = 2.5307
J6_MUJOCO_INDICES = (5, 12)  # left_fr3_joint6, right_fr3_joint6 in mujoco order

# FR3 "ready" pose (frankas_assets/fr3.py FR3_CFG init_state = dual_fr3.xml `home`), FR3 angles
FR3_READY = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741], dtype=np.float32)
# FR3 joint limits (menagerie franka_fr3, FR3 angles)
FR3_LO = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159], dtype=np.float32)
FR3_HI = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159], dtype=np.float32)


def fr3_to_sonic(q_mujoco: np.ndarray) -> np.ndarray:
    """FR3 angles -> SONIC angles, mujoco order (..., 14): joint 6 shifted by -J6_OFFSET."""
    q = np.array(q_mujoco, dtype=np.float32, copy=True)
    q[..., list(J6_MUJOCO_INDICES)] -= J6_OFFSET
    return q


def sonic_to_fr3(q_mujoco: np.ndarray) -> np.ndarray:
    q = np.array(q_mujoco, dtype=np.float32, copy=True)
    q[..., list(J6_MUJOCO_INDICES)] += J6_OFFSET
    return q


def mujoco_to_isaac(v: np.ndarray) -> np.ndarray:
    """(..., 14) mujoco order -> isaac order."""
    return np.asarray(v)[..., MUJOCO_TO_ISAAC_DOF]


def isaac_to_mujoco(v: np.ndarray) -> np.ndarray:
    return np.asarray(v)[..., ISAAC_TO_MUJOCO_DOF]


def wire_to_mujoco(state16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[Lq7, Lg, Rq7, Rg] -> (q_mujoco14 FR3 angles, grips[2])."""
    s = np.asarray(state16, dtype=np.float32).reshape(-1)
    if s.shape != (16,):
        raise ValueError(f"state16 must have shape (16,), got {s.shape}")
    q = np.concatenate([s[0:7], s[8:15]])
    g = np.array([s[7], s[15]], dtype=np.float32)
    return q, g


def mujoco_to_wire(q_mujoco14: np.ndarray, grips: np.ndarray) -> np.ndarray:
    q = np.asarray(q_mujoco14, dtype=np.float32).reshape(14)
    g = np.asarray(grips, dtype=np.float32).reshape(2)
    return np.concatenate([q[0:7], g[0:1], q[7:14], g[1:2]]).astype(np.float32)


# ----------------------------------------------------------------- SONIC defaults
SONIC_DEFAULT_MUJOCO = fr3_to_sonic(np.concatenate([FR3_READY, FR3_READY]))  # (14,)
SONIC_DEFAULT_ISAAC = mujoco_to_isaac(SONIC_DEFAULT_MUJOCO)  # decoder's default_joint_pos
SONIC_LO_MUJOCO = fr3_to_sonic(np.concatenate([FR3_LO, FR3_LO]))
SONIC_HI_MUJOCO = fr3_to_sonic(np.concatenate([FR3_HI, FR3_HI]))
ACTION_SCALE = 0.25  # joint target = default + ACTION_SCALE * raw action (DUAL_FR3_ACTION_SCALE)

# ----------------------------------------------------------------- ONNX layouts
TOKEN_DIM = 64
TOKEN_BOUND = 1.25  # NVIDIA's run_vla_inference.py bound; the FSQ grid itself is inside [-1, 1)
ENCODER_INPUT_DIM = 1391
ENC_IDX = 0  # scalar encoder index; 0 = g1 (robot-motion encoder)
ENC_CMD_MF = slice(4, 284)  # command_multi_future_nonflat: [joint_pos 10x14 | joint_vel 10x14]
ENC_ANCHOR_MF = slice(290, 350)  # motion_anchor_ori_b_mf_nonflat: 10 x 6D rotation
NUM_FUTURE_FRAMES = 10

DECODER_INPUT_DIM = 544
PROPRIO_DIM = 480
HISTORY = 10
# Term order = gear_sonic's PolicyCfg CLASS attribute order (not the yaml's defaults list):
# base_ang_vel, joint_pos, joint_vel, actions come first, gravity_dir is declared later in the
# class. Measured on the env's own actor_obs (label_tokens.py check, 2026-09-03): joint_pos_rel
# history at [30:170], joint_vel [170:310], last_action [310:450] (oldest first, a_k at 436),
# gravity (0,0,-1) tiled at [450:480], base_ang_vel (noise around 0) at [0:30].
PROPRIO_TERMS = (("base_ang_vel", 3), ("joint_pos_rel", 14), ("joint_vel", 14),
                 ("last_action", 14), ("gravity_dir", 3))
GRAVITY_DIR = np.array([0.0, 0.0, -1.0], dtype=np.float32)  # fixed upright base
CONTROL_HZ = 50


def build_proprio(hist_q_rel: np.ndarray, hist_v: np.ndarray, hist_a: np.ndarray) -> np.ndarray:
    """(H,14) x3 histories, oldest first -> the decoder's 480-D proprio vector."""
    blocks = {
        "base_ang_vel": np.zeros(3 * HISTORY, np.float32),
        "joint_pos_rel": np.asarray(hist_q_rel, np.float32).reshape(-1),
        "joint_vel": np.asarray(hist_v, np.float32).reshape(-1),
        "last_action": np.asarray(hist_a, np.float32).reshape(-1),
        "gravity_dir": np.tile(GRAVITY_DIR, HISTORY),
    }
    p = np.concatenate([blocks[name] for name, _ in PROPRIO_TERMS]).astype(np.float32)
    assert p.shape == (PROPRIO_DIM,), p.shape
    return p


def proprio_slices() -> dict[str, slice]:
    out, off = {}, 0
    for name, dim in PROPRIO_TERMS:
        out[name] = slice(off, off + dim * HISTORY)
        off += dim * HISTORY
    return out


def _check() -> None:
    assert sorted(ISAAC_JOINTS) == sorted(MUJOCO_JOINTS)
    assert sorted(ISAAC_BODIES) == sorted(MUJOCO_BODIES)
    x = np.arange(14, dtype=np.float32)
    assert np.array_equal(isaac_to_mujoco(mujoco_to_isaac(x)), x)
    assert np.array_equal(mujoco_to_isaac(x)[0], x[0]) and np.array_equal(mujoco_to_isaac(x)[1], x[7])
    total = sum(d for _, d in PROPRIO_TERMS) * HISTORY
    assert total == PROPRIO_DIM and TOKEN_DIM + PROPRIO_DIM == DECODER_INPUT_DIM
    assert ENC_CMD_MF.stop - ENC_CMD_MF.start == 2 * NUM_FUTURE_FRAMES * N_DOF
    assert ENC_ANCHOR_MF.stop - ENC_ANCHOR_MF.start == 6 * NUM_FUTURE_FRAMES


_check()
