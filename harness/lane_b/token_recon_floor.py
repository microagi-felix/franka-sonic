#!/usr/bin/env python3
"""P5 diagnostic: how precisely does the 64-D FSQ token still carry the reference joints?

    PYTHONUSERBASE=~/env/pyuser-sonic /isaac-sim/python.sh harness/lane_b/token_recon_floor.py \
        --ckpt <sonic_rl run>/out/TRL_DualFR3_Track/<exp>/last.pt \
        --tokens <label_tokens run>/out/tokens --obs <label_tokens run>/out/encoder_obs \
        --episodes 0 5 [--log /tmp/franka-sonic/lane_b/p5_recon_floor/recon_floor.log]

The SONIC actor has a *reconstruction* head ``decoders.g1_kin`` (token -> the encoder's own
input) trained with the ``g1_recon`` aux loss.  Running it on the STORED offline tokens gives the
information floor of the token itself: what the decoder could track if its dynamics were perfect.
Compare with the measured decoder replay (~0.055 rad / ~1 cm flange).

Not in the ONNX export: ``model_g1_pair.onnx`` is ``encoder g1 -> decoder g1_dyn`` (in 820 =
tokenizer 340 + proprio 480, out 14 actions), *not* the recon head.  eval_agent_trl.py only
exports g1_dyn pairs, the encoders and the decoder; ``g1_kin`` is never exported.  So the weights
are read straight out of ``last.pt`` (``policy_state_dict``) and the MLP is re-run in numpy.

Layouts (verified here, not assumed):

* ``g1`` encoder input  = cat([command_multi_future_nonflat (10,28),
  motion_anchor_ori_b_mf_nonflat (10,6)], dim=-1).view(340) -> INTERLEAVED per future frame
  ``[28 cmd | 6 anchor] x 10``, *not* ``[cmd 280 | anchor 60]`` (that is the flat ONNX layout).
  MLP 340->2048->1024->512->512->64, SiLU; view (2,32); FSQ levels=[32]*32 -> token, flatten 64.
* ``g1_kin`` recon head = MLP 64->2048->1024->512->512->340, SiLU, view (10,34), split
  ``[:28] = command_multi_future_nonflat``, ``[28:] = motion_anchor_ori_b_mf_nonflat``.
* the 280-D ``command_multi_future`` is ``[dof_pos frames 0..9 (10x14) | dof_vel frames 0..9]``
  (frame-major, IsaacLab order), so its (10,28) "nonflat" view puts *two* future frames of
  positions in each of rows 0..4 and the velocities in rows 5..9 — gear_sonic's own
  ``token_losses.decoder_output_to_egocentric_transforms`` calls this out as a known bug.
  First future frame joint pos = flat[0:14]; its joint vel = flat[140:154].
* ``future_steps[t, 0] == t``: the first future frame IS the current frame, so the reference for
  the first frame equals ``ref_joint_pos_isaac`` exactly (asserted below).
* recon target is RAW (``G1ReconLoss`` compares ``tokenizer_obs`` to the decoder output; the actor
  has ``running_mean_std: false``) — no mean/std to undo.

FK: MuJoCo on the INSTALLED dual_fr3.xml, flange = link7 + 0.107 m along its z (as
harness/lane_b/fk_flange_error.py).

Reports, per episode: per-joint and mean |q_recon - q_ref| over the clip and the grasp windows,
the implied flange error in cm, the joint-velocity recon error, a head-INDEPENDENT floor (how far
apart two reference frames that map to the SAME token can be — no decoder can separate those),
and the same head re-run on the unrounded FSQ code to price the 1/16 grid on its own.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import dual_fr3_orders as O  # noqa: E402

XML = os.path.expanduser(
    "~/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/dual_fr3.xml"
)
I2M = [O.ISAAC_JOINTS.index(j) for j in O.MUJOCO_JOINTS]  # mujoco = isaac[I2M]
JOINT_LABELS = [f"{s[0].upper()}{i}" for i in range(1, 8) for s in ("left", "right")]  # L1 R1 L2 ...
N_FUT = O.NUM_FUTURE_FRAMES  # 10
CMD_PER_FRAME = 2 * O.N_DOF  # 28
ANCHOR_PER_FRAME = 6
FSQ_LEVELS = 32

_LOG = []


def log(msg: str) -> None:
    print(msg, flush=True)
    _LOG.append(msg)


# ------------------------------------------------------------------ numpy MLP replica
def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def mlp_weights(state_dict, prefix: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """gear_sonic BaseModule._build_mlp_layer -> nn.Sequential(Linear, SiLU, Linear, SiLU, ...)."""
    layers, i = [], 0
    while f"{prefix}.{i}.weight" in state_dict:
        w = state_dict[f"{prefix}.{i}.weight"].float().cpu().numpy()
        b = state_dict[f"{prefix}.{i}.bias"].float().cpu().numpy()
        layers.append((w, b))
        i += 2  # activations carry no parameters
    if not layers:
        raise SystemExit(f"no MLP weights under {prefix}")
    return layers


def mlp_forward(x: np.ndarray, layers) -> np.ndarray:
    for k, (w, b) in enumerate(layers):
        x = x @ w.T + b
        if k < len(layers) - 1:
            x = silu(x)
    return x


def fsq_quantize(z: np.ndarray, levels: int = FSQ_LEVELS, round_: bool = True) -> np.ndarray:
    """vector_quantize_pytorch.FSQ with an even, uniform level list and no projections.

    ``round_=False`` returns the *unrounded* code in the same units as the token, i.e. the
    quantiser's input distribution without its rounding step — the only fair way to isolate the
    FSQ grid (the raw pre-bound latent is far outside anything ``g1_kin`` was ever trained on).
    """
    eps = 1e-3
    half_l = (levels - 1) * (1 + eps) / 2.0
    offset = 0.5  # levels even
    shift = np.arctanh(offset / half_l)
    bounded = np.tanh(z + shift) * half_l - offset
    return (np.round(bounded) if round_ else bounded) / (levels // 2)


# ------------------------------------------------------------------------------ FK
class FK:
    def __init__(self):
        import mujoco

        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(XML)
        self.d = mujoco.MjData(self.m)
        self.bodies = [
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in ("left_fr3_link7", "right_fr3_link7")
        ]

    def flanges(self, q_isaac: np.ndarray) -> np.ndarray:
        """(14,) isaac-order SONIC angles -> (2,3) left/right flange positions in the base frame."""
        self.d.qpos[:14] = np.asarray(q_isaac, np.float64)[I2M]
        self.mj.mj_forward(self.m, self.d)
        off = np.array([0.0, 0.0, 0.107])
        return np.array(
            [self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ off for b in self.bodies]
        )

    def batch(self, q: np.ndarray) -> np.ndarray:
        return np.stack([self.flanges(row) for row in q])


# ---------------------------------------------------------------------------- report
def stats(err_rad: np.ndarray, sl: slice) -> tuple[np.ndarray, float]:
    per_joint = np.abs(err_rad[sl]).mean(axis=0)
    return per_joint, float(per_joint.mean())


def window_line(tag: str, q_err: np.ndarray, fl_err_cm: np.ndarray, sl: slice) -> None:
    pj, mean = stats(q_err, sl)
    log(
        f"  {tag:26s} joints mean {mean:.4f} rad  max-joint {pj.max():.4f} ({JOINT_LABELS[int(pj.argmax())]})"
        f" | flange cm L mean {fl_err_cm[sl, 0].mean():.2f} max {fl_err_cm[sl, 0].max():.2f}"
        f" | R mean {fl_err_cm[sl, 1].mean():.2f} max {fl_err_cm[sl, 1].max():.2f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokens", required=True, help="<label_tokens run>/out/tokens")
    ap.add_argument("--obs", required=True, help="<label_tokens run>/out/encoder_obs")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 5])
    ap.add_argument("--log", default="/tmp/franka-sonic/lane_b/p5_recon_floor/recon_floor.log")
    args = ap.parse_args()

    import torch

    ckpt = os.path.expanduser(args.ckpt)
    tokens_dir = os.path.expanduser(args.tokens)
    obs_dir = os.path.expanduser(args.obs)
    log(f"[recon] ckpt {ckpt}")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["policy_state_dict"]
    kin = mlp_weights(sd, "actor_module.decoders.g1_kin.module")
    enc = mlp_weights(sd, "actor_module.encoders.g1.module")
    log(f"[recon] g1_kin MLP {[w.shape for w, _ in kin]}")
    log(f"[recon] g1     MLP {[w.shape for w, _ in enc]}")
    assert kin[0][0].shape[1] == O.TOKEN_DIM, kin[0][0].shape
    assert kin[-1][0].shape[0] == N_FUT * (CMD_PER_FRAME + ANCHOR_PER_FRAME), kin[-1][0].shape

    with open(os.path.join(tokens_dir, "index.json")) as fh:
        index = {e["episode_index"]: e for e in json.load(fh)}
    fk = FK()

    for ep in args.episodes:
        meta = index[ep]
        clip = meta["clip"]
        obs = np.load(os.path.join(obs_dir, f"{clip}.npz"))
        tok_npz = np.load(os.path.join(tokens_dir, meta["file"]))
        token, grip = tok_npz["token"], tok_npz["grip"]
        T = int(obs["num_steps"])
        # dataset frame t <-> motion-lib frame min(t, T-1); use the T frames that map 1:1
        token = token[:T].astype(np.float64)
        cmd_ref = obs["cmd_mf"][:T].astype(np.float64)
        anchor_ref = obs["anchor_mf"][:T].astype(np.float64)
        assert np.array_equal(obs["future_steps"][:, 0], np.arange(T)), "first future frame != t"
        assert np.abs(cmd_ref[:, : O.N_DOF] - obs["ref_joint_pos_isaac"][:T]).max() == 0.0

        log(f"\n[ep {ep}] clip {clip}  T_lib={T} T_dataset={meta['n_frames_dataset']}")

        # --- verification: re-encode the stored encoder input and reproduce the stored token
        enc_in = np.concatenate(
            [cmd_ref.reshape(T, N_FUT, CMD_PER_FRAME), anchor_ref.reshape(T, N_FUT, ANCHOR_PER_FRAME)],
            axis=-1,
        ).reshape(T, N_FUT * (CMD_PER_FRAME + ANCHOR_PER_FRAME))
        latent = mlp_forward(enc_in, enc)
        tok_rebuilt = fsq_quantize(latent)
        d_tok = float(np.abs(tok_rebuilt - token).max())
        log(f"  token replica vs stored: max abs diff {d_tok:.3e} "
            f"({'MATCH' if d_tok < 1e-6 else 'MISMATCH'})")

        # --- reconstruction from the STORED token
        out = mlp_forward(token, kin).reshape(T, N_FUT, CMD_PER_FRAME + ANCHOR_PER_FRAME)
        cmd_rec = out[:, :, :CMD_PER_FRAME].reshape(T, N_FUT * CMD_PER_FRAME)
        anchor_rec = out[:, :, CMD_PER_FRAME:].reshape(T, N_FUT * ANCHOR_PER_FRAME)

        q_rec, q_ref = cmd_rec[:, : O.N_DOF], cmd_ref[:, : O.N_DOF]
        v_rec = cmd_rec[:, N_FUT * O.N_DOF : N_FUT * O.N_DOF + O.N_DOF]
        v_ref = cmd_ref[:, N_FUT * O.N_DOF : N_FUT * O.N_DOF + O.N_DOF]
        q_err, v_err = q_rec - q_ref, v_rec - v_ref
        a_err = np.abs(anchor_rec - anchor_ref)

        fl_err_cm = np.linalg.norm(fk.batch(q_rec) - fk.batch(q_ref), axis=2) * 100.0

        pj, mean = stats(q_err, slice(None))
        log("  per-joint mean |q_recon - q_ref| (rad), first future frame = current frame:")
        log("    " + "  ".join(f"{n}={v:.4f}" for n, v in zip(JOINT_LABELS, pj)))
        log(f"    mean over joints {mean:.4f} rad   max frame-joint {np.abs(q_err).max():.4f} rad")
        log(f"  joint velocity recon: mean |dq| err {np.abs(v_err).mean():.4f} rad/s "
            f"(ref |dq| mean {np.abs(v_ref).mean():.4f})")
        log(f"  anchor 6D recon: mean abs {a_err.mean():.5f} max {a_err.max():.5f}")

        windows = [("clip (all frames)", slice(0, T))]
        close = {}
        for c, side in ((0, "left"), (1, "right")):
            tr = np.flatnonzero(np.diff((grip[:T, c] > 0.5).astype(int)) == 1)
            close[side] = int(tr[0]) if len(tr) else -1
        log(f"  grip close frames from labels: left {close['left']} right {close['right']}")
        windows.append(("left grasp 100-250", slice(100, 250)))
        windows.append(("right grasp 600-750", slice(600, 750)))
        if close["left"] > 0:
            windows.append((f"left approach {max(0, close['left'] - 100)}-{close['left']}",
                            slice(max(0, close["left"] - 100), close["left"])))
        if close["right"] > 0:
            windows.append((f"right approach {max(0, close['right'] - 100)}-{close['right']}",
                            slice(max(0, close["right"] - 100), close["right"])))
        for tag, sl in windows:
            window_line(tag, q_err, fl_err_cm, sl)

        # --- head-INDEPENDENT floor: frames that share one token cannot be told apart by ANY
        #     decoder, however good.  Spread of the reference inside each collision group.
        _, inv, cnt = np.unique(token, axis=0, return_inverse=True, return_counts=True)
        fl_ref = fk.batch(q_ref)
        coll = np.flatnonzero(cnt > 1)
        n_frames_coll = int(cnt[coll].sum())
        spreads_q, spreads_cm = [], []
        for g in coll:
            sel = inv == g
            spreads_q.append(np.ptp(q_ref[sel], axis=0).max())
            spreads_cm.append(np.ptp(fl_ref[sel], axis=0).max(axis=0).max() * 100.0)
        log(f"  token collisions: {len(_)} unique tokens / {T} frames; "
            f"{n_frames_coll} frames ({100.0 * n_frames_coll / T:.1f}%) share a token with another frame")
        if spreads_cm:
            log(f"    within-group reference spread: joints mean {np.mean(spreads_q):.4f} "
                f"max {np.max(spreads_q):.4f} rad | flange mean {np.mean(spreads_cm):.2f} "
                f"max {np.max(spreads_cm):.2f} cm")

        # --- how much of the recon error is the FSQ ROUNDING? same head, unrounded code
        code = fsq_quantize(latent, round_=False)
        log(f"  FSQ rounding size: mean |code - token| {np.abs(code - token).mean():.4f} "
            f"(one grid step = {2.0 / FSQ_LEVELS:.4f})")
        q_lat = mlp_forward(code, kin).reshape(T, N_FUT, CMD_PER_FRAME + ANCHOR_PER_FRAME)[
            :, 0, : O.N_DOF
        ]
        ql_err = q_lat - q_ref
        fl_lat_cm = np.linalg.norm(fk.batch(q_lat) - fk.batch(q_ref), axis=2) * 100.0
        log("  --- same head on the UNROUNDED code (isolates the FSQ grid) ---")
        for tag, sl in windows[:3]:
            window_line("unrounded " + tag, ql_err, fl_lat_cm, sl)
        log(f"  FSQ rounding costs: joints {np.abs(q_err).mean() - np.abs(ql_err).mean():+.4f} rad, "
            f"flange {fl_err_cm.mean() - fl_lat_cm.mean():+.2f} cm "
            f"(rounded {np.abs(q_err).mean():.4f} rad / {fl_err_cm.mean():.2f} cm vs "
            f"unrounded {np.abs(ql_err).mean():.4f} rad / {fl_lat_cm.mean():.2f} cm)")

    out_log = os.path.expanduser(args.log)
    os.makedirs(os.path.dirname(out_log), exist_ok=True)
    with open(out_log, "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    print(f"[recon] wrote {out_log}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
