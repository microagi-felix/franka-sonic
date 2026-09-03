"""SONIC decoder runtime for lane B (P3): token + streamed joint state -> FR3 joint targets.

Wraps P2's ``model_decoder.onnx`` (lane_b/2026-09-03_export_onnx-*) with the proprio
bookkeeping the trained policy had in the SONIC env, so it can live inside a policy server
(harness/lane_b/serve_gr00t_sonic_joint.py) or an in-process oracle client
(harness/lane_b/eval_oracle_b.py) that only sees the sim's 16-D wire state at 50 Hz.

Decoder input (1, 544) = [token 64 | proprio 480]; proprio = the IsaacLab ``actor_obs`` of
gear_sonic's ``local_dir_hist`` policy group, term by term, each term a 10-step history
flattened oldest-first:

    gravity_dir   (3)  down vector in the anchor (base) frame -> constant (0, 0, -1)
    base_ang_vel  (3)  fixed base -> 0
    joint_pos_rel (14) q - default_joint_pos, isaac order, SONIC angles
    joint_vel     (14) finite difference of the streamed positions at 50 Hz (the env used
                       PhysX's velocity; the difference is the one modelling gap here)
    last_action   (14) the decoder's own previous raw output (zeros after a reset)

History semantics copy IsaacLab's CircularBuffer: after a reset the first observation fills
all 10 slots (the action slots with zeros), then each step appends the newest entry at the end.

Decoder output (1, 14) = raw action a (isaac order); joint target = default + 0.25 a
(JointPositionActionCfg use_default_offset + DUAL_FR3_ACTION_SCALE), converted back to FR3
angles (joint 6 + J6_OFFSET), mujoco order, then optionally clipped to the FR3 joint limits.

Hold token: ``hold_token_from_encoder`` runs the encoder ONNX on a static reference at the
FR3 ready pose (what the labeller's hold_token.json holds) — a zero token is NOT a hold
(the shipped-token trap). The runtime feeds it whenever it is asked to act without a token.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import dual_fr3_orders as O  # noqa: E402


def _session(onnx_path: str, expect_dim: int):
    import onnxruntime as ort

    sess = ort.InferenceSession(os.path.expanduser(onnx_path), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    if list(inp.shape[1:]) != [expect_dim]:
        raise ValueError(f"{onnx_path}: input {inp.shape}, expected (1, {expect_dim})")
    return sess, inp.name


def hold_token_from_encoder(encoder_onnx: str, anchor_row: np.ndarray | None = None) -> np.ndarray:
    """Encode a static reference at the SONIC default (FR3 ready) pose -> (64,) token."""
    sess, name = _session(encoder_onnx, O.ENCODER_INPUT_DIM)
    if anchor_row is None:
        # identity rotation, first two columns of the 3x3 matrix row-major: [m00 m01 m10 m11 m20 m21]
        anchor_row = np.tile(np.array([1, 0, 0, 1, 0, 0], np.float32), O.NUM_FUTURE_FRAMES)
    enc = np.zeros((1, O.ENCODER_INPUT_DIM), np.float32)
    enc[0, O.ENC_IDX] = 0.0
    pos = np.tile(O.SONIC_DEFAULT_ISAAC.astype(np.float32), O.NUM_FUTURE_FRAMES)
    enc[0, O.ENC_CMD_MF] = np.concatenate([pos, np.zeros(O.NUM_FUTURE_FRAMES * O.N_DOF, np.float32)])
    enc[0, O.ENC_ANCHOR_MF] = np.asarray(anchor_row, np.float32).reshape(-1)
    return sess.run(None, {name: enc})[0][0].astype(np.float32)


def load_hold_token(path_or_none: str | None, encoder_onnx: str | None) -> np.ndarray:
    """hold_token.json (labeller) wins; else self-encode from the encoder ONNX."""
    if path_or_none and os.path.exists(os.path.expanduser(path_or_none)):
        with open(os.path.expanduser(path_or_none)) as fh:
            d = json.load(fh)
        return np.asarray(d["token"], np.float32).reshape(O.TOKEN_DIM)
    if encoder_onnx:
        return hold_token_from_encoder(encoder_onnx)
    raise ValueError("need --hold-token-json or --encoder-onnx to build the hold token")


class SonicDecoderRuntime:
    """One decoder + its proprio history. Not thread-safe; one instance per episode stream."""

    def __init__(self, decoder_onnx: str, hold_token: np.ndarray | None = None,
                 clip_targets: bool = True, token_bound: float = O.TOKEN_BOUND):
        self.sess, self.name = _session(decoder_onnx, O.DECODER_INPUT_DIM)
        self.hold_token = None if hold_token is None else np.asarray(hold_token, np.float32).reshape(O.TOKEN_DIM)
        self.clip_targets = clip_targets
        self.token_bound = float(token_bound)
        self.default = O.SONIC_DEFAULT_ISAAC.astype(np.float32)
        self.reset()

    # -------------------------------------------------------------- state
    def reset(self) -> None:
        self.hist_q = None  # (H, 14) joint_pos_rel, oldest first
        self.hist_v = None
        self.hist_a = None
        self.prev_q = None  # isaac order, SONIC angles
        self.last_action = np.zeros(O.N_DOF, np.float32)
        self.steps = 0
        self.max_token_clipped = 0.0

    # -------------------------------------------------------------- helpers
    @staticmethod
    def wire_to_isaac(state16: np.ndarray) -> np.ndarray:
        q_fr3, _ = O.wire_to_mujoco(state16)
        return O.mujoco_to_isaac(O.fr3_to_sonic(q_fr3)).astype(np.float32)

    def _push(self, q: np.ndarray, v: np.ndarray, a_prev: np.ndarray) -> None:
        if self.hist_q is None:
            # IsaacLab CircularBuffer: the first push after a reset fills every slot
            self.hist_q = np.tile(q - self.default, (O.HISTORY, 1)).astype(np.float32)
            self.hist_v = np.tile(v, (O.HISTORY, 1)).astype(np.float32)
            self.hist_a = np.tile(a_prev, (O.HISTORY, 1)).astype(np.float32)
        else:
            self.hist_q = np.roll(self.hist_q, -1, axis=0)
            self.hist_q[-1] = q - self.default
            self.hist_v = np.roll(self.hist_v, -1, axis=0)
            self.hist_v[-1] = v
            self.hist_a = np.roll(self.hist_a, -1, axis=0)
            self.hist_a[-1] = a_prev

    def proprio(self) -> np.ndarray:
        return O.build_proprio(self.hist_q, self.hist_v, self.hist_a)

    # -------------------------------------------------------------- stepping
    def step(self, token: np.ndarray | None, state16: np.ndarray, return_proprio: bool = False):
        """One 50 Hz decoder step.

        token:   (64,) SONIC token for this frame, or None -> hold token.
        state16: the sim's wire state [Lq7, Lg, Rq7, Rg] (FR3 angles).
        Returns the raw action (14, isaac order) [and the proprio vector].
        Use ``targets_wire`` to turn the raw action into joint targets.
        """
        if token is None:
            if self.hold_token is None:
                raise ValueError("no token and no hold token")
            token = self.hold_token
        token = np.asarray(token, np.float32).reshape(O.TOKEN_DIM)
        if not np.isfinite(token).all():
            raise ValueError("non-finite token")
        amax = float(np.abs(token).max())
        if amax > self.token_bound:
            self.max_token_clipped = max(self.max_token_clipped, amax)
            token = np.clip(token, -self.token_bound, self.token_bound)
        q = self.wire_to_isaac(state16)
        if self.prev_q is None:
            v = np.zeros(O.N_DOF, np.float32)  # the env reports zero velocity at reset
        else:
            v = ((q - self.prev_q) * O.CONTROL_HZ).astype(np.float32)
        self.prev_q = q
        self._push(q, v, self.last_action)
        pro = self.proprio()
        x = np.concatenate([token, pro])[None, :].astype(np.float32)
        a = self.sess.run(None, {self.name: x})[0][0].astype(np.float32)
        if not np.isfinite(a).all():
            raise ValueError("decoder produced non-finite actions")
        self.last_action = a
        self.steps += 1
        return (a, pro) if return_proprio else a

    def targets_isaac(self, raw_action: np.ndarray) -> np.ndarray:
        return (self.default + O.ACTION_SCALE * np.asarray(raw_action, np.float32)).astype(np.float32)

    def targets_wire(self, raw_action: np.ndarray, grips: np.ndarray) -> np.ndarray:
        """raw action -> [Lq7, Lg, Rq7, Rg] joint targets in FR3 angles (grips passed through)."""
        q_mujoco = O.sonic_to_fr3(O.isaac_to_mujoco(self.targets_isaac(raw_action)))
        if self.clip_targets:
            lo = np.concatenate([O.FR3_LO, O.FR3_LO])
            hi = np.concatenate([O.FR3_HI, O.FR3_HI])
            q_mujoco = np.clip(q_mujoco, lo, hi)
        return O.mujoco_to_wire(q_mujoco, grips)

    def act(self, token: np.ndarray | None, state16: np.ndarray, grips: np.ndarray) -> np.ndarray:
        """Convenience: step + targets_wire."""
        return self.targets_wire(self.step(token, state16), grips)


def self_test() -> None:
    """CPU smoke without ONNX: the history bookkeeping and the order round trips."""
    rt = SonicDecoderRuntime.__new__(SonicDecoderRuntime)
    rt.default = O.SONIC_DEFAULT_ISAAC.astype(np.float32)
    rt.clip_targets = True
    rt.reset()
    q0 = np.concatenate([O.FR3_READY, [0.0], O.FR3_READY, [0.0]]).astype(np.float32)
    q = rt.wire_to_isaac(q0)
    assert np.allclose(q, rt.default), (q, rt.default)
    rt._push(q, np.zeros(14, np.float32), np.zeros(14, np.float32))
    p = rt.proprio()
    S = O.proprio_slices()
    assert p.shape == (480,) and np.allclose(p[S["gravity_dir"]], np.tile([0, 0, -1], 10))
    assert np.allclose(p[:450], 0)
    a = np.ones(14, np.float32)
    rt._push(q + 0.1, np.full(14, 5.0, np.float32), a)
    p = rt.proprio()
    hq = p[S["joint_pos_rel"]].reshape(10, 14)
    assert np.allclose(hq[:-1], 0) and np.allclose(hq[-1], 0.1)
    ha = p[S["last_action"]].reshape(10, 14)
    assert np.allclose(ha[:-1], 0) and np.allclose(ha[-1], 1)
    hv = p[S["joint_vel"]].reshape(10, 14)
    assert np.allclose(hv[:-1], 0) and np.allclose(hv[-1], 5.0)
    w = rt.targets_wire(np.zeros(14, np.float32), np.array([0.0, 1.0]))
    assert np.allclose(w[0:7], O.FR3_READY, atol=1e-5) and np.allclose(w[8:15], O.FR3_READY, atol=1e-5)
    assert w[7] == 0.0 and w[15] == 1.0
    print("[sonic_decoder] self-test OK")


if __name__ == "__main__":
    self_test()
