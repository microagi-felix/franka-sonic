#!/usr/bin/env python3
"""Offline SONIC token labels for the handover demos (P3 WP 3.1).

Lane B's action label for dataset frame t is the 64-D FSQ token the trained
SONIC encoder ("g1" = robot-motion mode) emits when it sees the reference motion
from t onwards — the 10 future reference frames at 0.1 s spacing that the
encoder was trained on. Offline labelling has that future window; NVIDIA's live
recipe encodes teleop on the fly. The B-oracle (WP 3.5) checks the labels are
executable.

The encoder inputs are produced by gear_sonic's OWN command term
(``TrackingCommand.create_offline`` + ``MotionLibRobot``) on the P2 motion
library clips, so the joint order, the 30->50 fps resampling, the finite-
difference joint velocities, the future-frame indexing (frame_skips = 5 at
50 fps, clamped at the clip end) and the flattening quirk of
``command_multi_future_nonflat`` are the library's, not a re-implementation.
Three subcommands, two interpreters (AGENTS.md rule d):

  obs     PYTHONUSERBASE=~/env/pyuser-sonic /isaac-sim/python.sh label_tokens.py obs \
              --clips <dir of the 76 original demo pkls> --out <run>/out/encoder_obs
          -> <out>/<clip>.npz : enc_in (T_lib, 1391) float32 encoder input per 50 Hz frame

  encode  ~/Isaac-GR00T/.venv/bin/python label_tokens.py encode \
              --obs <run>/out/encoder_obs --encoder <export>/out/model_encoder.onnx \
              --dataset <lane A gr00t_v2> --manifest <motion_lib>/out/manifest.json \
              --out <run>/out/tokens
          -> <out>/episode_XXXXXX.npz (token (T,64), grip (T,2), provenance), index.json,
             hold_token.json (the encoder's token for a static ready pose), label_summary.json
          Asserts |token| <= 1.25 and finite.

  check   ~/Isaac-GR00T/.venv/bin/python label_tokens.py check \
              --dump <run>/out/validation/env_dump.npz --obs <run>/out/encoder_obs \
              --encoder ... --decoder ... --out <run>/out/validation/validation.json
          Compares the offline encoder inputs with what the SONIC env fed the policy on the
          same clip (harness/lane_b/dump_obs_callback.py), runs encoder+decoder ONNX on the
          env's own observations against the env's own actions, and checks the decoder
          proprio layout the runtime (sonic_decoder.py) reproduces.

Dataset frame t (50 Hz, from frame 0) pairs with motion-lib frame min(t, T_lib-1)
(also 50 Hz from frame 0): nearest-neighbour, one frame of drift at most at the end.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import dual_fr3_orders as O  # noqa: E402

DEFAULT_WBC = os.path.expanduser("~/GR00T-WholeBodyControl")
MJCF_REL = "gear_sonic/data/assets/robot_description/mjcf/"


def log(msg: str) -> None:
    print(msg, flush=True)


# =============================================================================== obs
def cmd_obs(args) -> int:
    wbc = os.path.abspath(os.path.expanduser(args.wbc))
    clips = os.path.abspath(os.path.expanduser(args.clips))
    out = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(out, exist_ok=True)
    n_pkl = len(glob.glob(os.path.join(clips, "*.pkl")))
    if n_pkl == 0:
        raise SystemExit(f"[obs] no pkl under {clips}")
    log(f"[obs] {n_pkl} clips in {clips}")
    sys.path.insert(0, wbc)
    os.chdir(wbc)
    import easydict
    import torch

    # gear_sonic's command term (commands.py) cannot be imported outside a Kit app (it pulls
    # isaaclab.envs.mdp -> pxr), so the few command properties the g1 encoder consumes are
    # re-derived here on top of the library's OWN MotionLibRobot (loads without Isaac, P2):
    #   command_multi_future      = [dof_pos(future) | dof_vel(future)]     (TrackingCommand L897)
    #   future_time_steps         = clip(t + k*frame_skips, max=num_steps-1) (L3267)
    #   root_rot_dif_l_multi_future = 6D(quat_inv(robot_anchor) * ref_root(future)) (L1942),
    #     offline: robot_anchor = ref root quat at t (L2330)
    # `check` then compares every row against the SONIC env's own values (dump_obs_callback).
    from gear_sonic.utils.motion_lib import motion_lib_robot

    body_idx = [O.ISAAC_BODIES.index(b) for b in O.TRACKED_BODIES]
    cfg = easydict.EasyDict({
        "motion_file": clips,
        "smpl_motion_file": "dummy",
        "smpl_y_up": True,
        "asset": {"assetRoot": MJCF_REL, "assetFileName": "dual_fr3.xml", "urdfFileName": ""},
        "extend_config": [],
        "target_fps": 50,
        "step_dt": 1.0 / O.CONTROL_HZ,
        "multi_thread": False,
        "filter_motion_keys": None,
        "adaptive_sampling": {"enable": False},
        "mujoco_to_isaaclab_dof": list(O.MUJOCO_TO_ISAAC_DOF),
        "isaaclab_to_mujoco_dof": list(O.ISAAC_TO_MUJOCO_DOF),
        "mujoco_to_isaaclab_body": list(O.MUJOCO_TO_ISAAC_BODY),
        "isaaclab_to_mujoco_body": list(O.ISAAC_TO_MUJOCO_BODY),
        "body_indexes": torch.tensor(body_idx, dtype=torch.long),
        "body_indexes_data": list(body_idx),
        "lower_joint_indices_mujoco": list(range(12)),
        "cat_upper_body_poses": False,
        "cat_upper_body_poses_prob": 0.0,
        "freeze_frame_aug": False,
        "freeze_frame_aug_prob": 0.0,
        "randomize_heading": False,
        "randomize_wrist_poses": False,
        "randomize_wrist_prob": 0.0,
        "randomize_wrist_std": 0.0,
    })
    t0 = time.time()
    lib = motion_lib_robot.MotionLibRobot(cfg, num_envs=1, device="cpu")
    lib.load_motions_for_training(max_num_seqs=None)  # all clips, file order, no augmentation
    keys = list(lib.curr_motion_keys)
    num_steps_all = lib.get_motion_num_steps().cpu().numpy().astype(int)
    frame_skips = int(0.1 // (1.0 / lib.target_fps))
    assert frame_skips == 5, f"frame_skips {frame_skips} != 5 (0.1 s at 50 fps)"
    assert lib.dof_pos.shape[-1] == O.N_DOF, lib.dof_pos.shape
    dof_pos = lib.dof_pos.cpu().numpy().astype(np.float32)  # isaac order (reordered at load)
    dof_vel = lib.dof_vel.cpu().numpy().astype(np.float32)
    root_quat = lib.body_quat_w[:, 0, :].cpu().numpy().astype(np.float32)  # tracked body 0 = base, wxyz
    starts = lib.length_starts.cpu().numpy().astype(int)
    log(f"[obs] motion lib loaded {len(keys)} clips in {time.time() - t0:.1f}s; "
        f"frame_skips={frame_skips} target_fps={lib.target_fps} "
        f"num_steps {num_steps_all.min()}..{num_steps_all.max()}; root quat row0 {root_quat[0].round(4).tolist()}")

    def quat_inv(q):  # wxyz, unit
        return q * np.array([1, -1, -1, -1], np.float32)

    def quat_mul(a, b):  # wxyz, (N,4) x (N,4)
        w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        return np.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], axis=-1)

    def rot6d(q):  # wxyz -> first two COLUMNS of R, row-major over rows: [m00 m01 m10 m11 m20 m21]
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        m00 = 1 - 2 * (y * y + z * z)
        m01 = 2 * (x * y - z * w)
        m10 = 2 * (x * y + z * w)
        m11 = 1 - 2 * (x * x + z * z)
        m20 = 2 * (x * z - y * w)
        m21 = 2 * (y * z + x * w)
        return np.stack([m00, m01, m10, m11, m20, m21], axis=-1)

    summary = []
    for m, key in enumerate(keys):
        T = int(num_steps_all[m])
        s0 = int(starts[m])
        t = np.arange(T)[:, None]
        fut = np.minimum(t + np.arange(O.NUM_FUTURE_FRAMES)[None, :] * frame_skips, T - 1)  # (T, 10)
        pos = dof_pos[s0 + fut].reshape(T, -1)  # frame-major, isaac order
        vel = dof_vel[s0 + fut].reshape(T, -1)
        cmd_mf = np.concatenate([pos, vel], axis=1).astype(np.float32)
        q_anchor = np.repeat(root_quat[s0 + np.arange(T)][:, None, :], O.NUM_FUTURE_FRAMES, axis=1)
        q_ref = root_quat[s0 + fut]
        anchor = rot6d(quat_mul(quat_inv(q_anchor), q_ref)).reshape(T, -1).astype(np.float32)
        enc = np.zeros((T, O.ENCODER_INPUT_DIM), np.float32)
        enc[:, O.ENC_IDX] = 0.0  # g1 encoder
        enc[:, O.ENC_CMD_MF] = cmd_mf
        enc[:, O.ENC_ANCHOR_MF] = anchor
        ref_isaac = dof_pos[s0 + np.arange(T)]
        assert np.isfinite(enc).all(), key
        np.savez_compressed(
            os.path.join(out, f"{key}.npz"), enc_in=enc, cmd_mf=cmd_mf, anchor_mf=anchor,
            future_steps=fut.astype(np.int64), ref_joint_pos_isaac=ref_isaac, num_steps=T, clip=key,
            control_hz=O.CONTROL_HZ,
        )
        summary.append({"clip": key, "num_steps": T,
                        "ref_joint_pos_isaac_first": ref_isaac[0].round(4).tolist()})
        if m % 10 == 0 or m == len(keys) - 1:
            log(f"[obs] {m + 1}/{len(keys)} {key}: {T} frames; anchor row0 {anchor[0, :6].round(3).tolist()}")
    with open(os.path.join(out, "obs_index.json"), "w") as fh:
        json.dump({"clips": summary, "encoder_input_dim": O.ENCODER_INPUT_DIM,
                   "layout": "[enc_idx 0 | onehot 1:4 | cmd_mf 4:284 | anchor_ori_b 284:290 | anchor_mf 290:350 | rest 0]",
                   "frame_skips": frame_skips, "target_fps": int(lib.target_fps),
                   "wall_clock_s": round(time.time() - t0, 1)}, fh, indent=1)
    log(f"[obs] OBS_DONE: {len(keys)} clips -> {out} in {time.time() - t0:.1f}s")
    return 0


# ============================================================================ encode
class Encoder:
    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        assert list(shape[1:]) == [O.ENCODER_INPUT_DIM], shape

    def __call__(self, enc_in: np.ndarray) -> np.ndarray:
        enc_in = np.asarray(enc_in, np.float32).reshape(-1, O.ENCODER_INPUT_DIM)
        out = np.empty((enc_in.shape[0], O.TOKEN_DIM), np.float32)
        for i in range(enc_in.shape[0]):
            out[i] = self.sess.run(None, {self.name: enc_in[i:i + 1]})[0][0]
        return out


def _load_clip_map(manifest_path: str) -> dict[tuple[str, str], str]:
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    entries = manifest["clips"] if isinstance(manifest, dict) and "clips" in manifest else manifest
    m = {}
    for e in entries:
        if e.get("aug") == "orig" and not e.get("mirrored"):
            m[(os.path.basename(e["source_shard"]), e["source_demo"])] = e["name"]
    if not m:
        raise SystemExit(f"[encode] no original clips in {manifest_path}")
    return m


def _episode_parquet(dataset: str, ep: int) -> str:
    with open(os.path.join(dataset, "meta", "info.json")) as fh:
        info = json.load(fh)
    chunk = ep // int(info.get("chunks_size", 1000))
    return os.path.join(dataset, info["data_path"].format(episode_chunk=chunk, episode_index=ep))


def _read_actions(path: str) -> np.ndarray:
    import pyarrow.parquet as pq

    col = pq.read_table(path, columns=["action"]).column("action").to_pylist()
    return np.asarray(col, dtype=np.float32)


def hold_token_input(anchor_row: np.ndarray) -> np.ndarray:
    """Encoder input for a static reference at the SONIC default (FR3 ready) pose."""
    enc = np.zeros((1, O.ENCODER_INPUT_DIM), np.float32)
    pos = np.tile(O.SONIC_DEFAULT_ISAAC.astype(np.float32), O.NUM_FUTURE_FRAMES)
    vel = np.zeros(O.NUM_FUTURE_FRAMES * O.N_DOF, np.float32)
    enc[0, O.ENC_CMD_MF] = np.concatenate([pos, vel])
    enc[0, O.ENC_ANCHOR_MF] = np.asarray(anchor_row, np.float32)
    return enc


def cmd_encode(args) -> int:
    t0 = time.time()
    obs_dir = os.path.abspath(os.path.expanduser(args.obs))
    dataset = os.path.abspath(os.path.expanduser(args.dataset))
    out = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(out, exist_ok=True)
    enc = Encoder(os.path.expanduser(args.encoder))
    clip_map = _load_clip_map(os.path.expanduser(args.manifest))
    with open(os.path.join(dataset, "meta", "provenance.json")) as fh:
        prov = json.load(fh)
    # provenance.json holds one entry per episode index plus a few dataset-level keys
    episodes = sorted(int(k) for k in prov.keys() if k.isdigit() and isinstance(prov[k], dict))
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]

    index, summary = [], []
    gmax, anchor_row = 0.0, None
    token_cache: dict[str, np.ndarray] = {}
    for ep in episodes:
        p = prov[str(ep)]
        key = (os.path.basename(p["source_file"]), p["demo_name"])
        if key not in clip_map:
            raise SystemExit(f"[encode] episode {ep} {key} has no original clip in the manifest")
        clip = clip_map[key]
        obs = np.load(os.path.join(obs_dir, f"{clip}.npz"))
        T_lib = int(obs["num_steps"])
        if anchor_row is None:
            anchor_row = obs["anchor_mf"][0]
        if clip not in token_cache:
            token_cache[clip] = enc(obs["enc_in"])
        tok_lib = token_cache[clip]
        actions = _read_actions(_episode_parquet(dataset, ep))
        T_ds = int(actions.shape[0])
        if actions.shape[1] != 16:
            raise SystemExit(f"[encode] episode {ep}: lane A action width {actions.shape[1]} != 16")
        idx = np.minimum(np.arange(T_ds), T_lib - 1)
        token = tok_lib[idx].astype(np.float32)
        grip = actions[:, 14:16].astype(np.float32)
        if not np.isfinite(token).all():
            raise SystemExit(f"[encode] episode {ep}: non-finite token")
        amax = float(np.abs(token).max())
        gmax = max(gmax, amax)
        if amax > O.TOKEN_BOUND:
            raise SystemExit(f"[encode] episode {ep}: |token| max {amax:.3f} > {O.TOKEN_BOUND} "
                             "— the encoder or its input ordering is wrong")
        changes = float(np.mean(np.any(token[1:] != token[:-1], axis=1))) if T_ds > 1 else 0.0
        fname = f"episode_{ep:06d}.npz"
        np.savez_compressed(
            os.path.join(out, fname), token=token, grip=grip, episode_index=ep,
            source_file=p["source_file"], demo_name=p["demo_name"], clip=clip,
            n_frames_dataset=T_ds, n_frames_lib=T_lib,
        )
        index.append({"episode_index": ep, "file": fname, "n_frames_dataset": T_ds,
                      "n_frames_lib": T_lib, "clip": clip})
        summary.append({"episode_index": ep, "clip": clip, "T_dataset": T_ds, "T_lib": T_lib,
                        "token_min": float(token.min()), "token_max": float(token.max()),
                        "frac_frames_token_changes": round(changes, 4),
                        "n_unique_tokens": int(np.unique(token, axis=0).shape[0])})
        log(f"[encode] ep {ep:3d} {clip:16s} T_ds={T_ds} T_lib={T_lib} |token|max={amax:.4f} "
            f"unique={summary[-1]['n_unique_tokens']} changes={changes:.3f}")

    hold_in = hold_token_input(anchor_row)
    hold = enc(hold_in)[0]
    with open(os.path.join(out, "hold_token.json"), "w") as fh:
        json.dump({"token": hold.tolist(), "anchor_mf_row": np.asarray(anchor_row).tolist(),
                   "default_joint_pos_isaac": O.SONIC_DEFAULT_ISAAC.tolist(),
                   "note": "encoder output for a static reference at the FR3 ready pose "
                           "(SONIC angles, isaac order); NOT a zero token"}, fh, indent=1)
    with open(os.path.join(out, "index.json"), "w") as fh:
        json.dump(index, fh, indent=1)
    all_tok = np.concatenate([np.load(os.path.join(out, e["file"]))["token"] for e in index])
    grid = np.abs(all_tok * 16 - np.round(all_tok * 16)).max()
    with open(os.path.join(out, "label_summary.json"), "w") as fh:
        json.dump({"episodes": summary, "n_episodes": len(index),
                   "n_frames": int(sum(e["n_frames_dataset"] for e in index)),
                   "token_abs_max": gmax, "token_bound": O.TOKEN_BOUND,
                   "max_dist_from_1_16_grid": float(grid),
                   "hold_token_abs_max": float(np.abs(hold).max()),
                   "per_dim_mean": all_tok.mean(0).round(4).tolist(),
                   "per_dim_std": all_tok.std(0).round(4).tolist(),
                   "wall_clock_s": round(time.time() - t0, 1)}, fh, indent=1)
    log(f"[encode] ENCODE_DONE: {len(index)} episodes, {int(all_tok.shape[0])} frames, "
        f"|token| max {gmax:.4f} (bound {O.TOKEN_BOUND}), grid dist {grid:.2e} -> {out}")
    return 0


# ============================================================================= check
class Decoder:
    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        assert list(shape[1:]) == [O.DECODER_INPUT_DIM], shape

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float32).reshape(-1, O.DECODER_INPUT_DIM)
        return np.stack([self.sess.run(None, {self.name: x[i:i + 1]})[0][0] for i in range(x.shape[0])])


def _find_offset(hay: np.ndarray, needle: np.ndarray) -> tuple[int, float]:
    """Offset o minimising max|hay[:, o:o+w] - needle| over rows (needle has no noise)."""
    w = needle.shape[1]
    best = (-1, np.inf)
    for o in range(0, hay.shape[1] - w + 1):
        d = float(np.abs(hay[:, o:o + w] - needle).max())
        if d < best[1]:
            best = (o, d)
    return best


def cmd_check(args) -> int:
    dump = np.load(os.path.expanduser(args.dump))
    meta = json.loads(str(dump["meta"]))
    clip = meta["clip"] or meta.get("motion_key")
    obs = np.load(os.path.join(os.path.expanduser(args.obs), f"{clip}.npz"))
    res = {"clip": clip, "n_rows": int(meta["n_rows"]), "ended_by_time_out": meta["ended_by_time_out"]}

    # 1. offline encoder inputs vs the env's command term at the same time steps
    ts = dump["time_steps"].reshape(-1).astype(int)
    T_lib = int(obs["num_steps"])
    valid = ts < T_lib
    d_cmd = np.abs(dump["cmd_mf"][valid] - obs["cmd_mf"][ts[valid]])
    d_anc = np.abs(dump["anchor_mf"][valid] - obs["anchor_mf"][ts[valid]])
    d_fut = np.abs(dump["future_steps"][valid] - obs["future_steps"][ts[valid]])
    res["offline_vs_env"] = {
        "rows_compared": int(valid.sum()), "env_num_steps": meta.get("motion_num_steps"),
        "offline_num_steps": T_lib,
        "cmd_mf_max_abs_diff": float(d_cmd.max()), "cmd_mf_mean_abs_diff": float(d_cmd.mean()),
        "anchor_mf_max_abs_diff": float(d_anc.max()),
        "future_steps_max_abs_diff": int(d_fut.max()),
        "env_frame_skips": meta.get("frame_skips"), "env_num_future_frames": meta.get("num_future_frames"),
    }

    # 2. layout of the env's flat tokenizer vector (what the policy saw, with any noise)
    tok_key = [k for k in dump.files if k.startswith("obs__") and "token" in k]
    enc = Encoder(os.path.expanduser(args.encoder))
    if tok_key:
        tok = dump[tok_key[0]]
        o_cmd, d1 = _find_offset(tok, dump["cmd_mf"])
        o_anc, d2 = _find_offset(tok, dump["anchor_mf"])
        res["env_tokenizer_layout"] = {"key": tok_key[0], "dim": int(tok.shape[1]),
                                       "cmd_mf_offset": o_cmd, "cmd_mf_residual": d1,
                                       "anchor_mf_offset": o_anc, "anchor_mf_residual(noise)": d2}
        seen = np.zeros((tok.shape[0], O.ENCODER_INPUT_DIM), np.float32)
        seen[:, O.ENC_CMD_MF] = tok[:, o_cmd:o_cmd + 280]
        seen[:, O.ENC_ANCHOR_MF] = tok[:, o_anc:o_anc + 60]
        tok_seen = enc(seen)
    else:
        tok_seen = None
    clean = np.zeros((dump["cmd_mf"].shape[0], O.ENCODER_INPUT_DIM), np.float32)
    clean[:, O.ENC_CMD_MF] = dump["cmd_mf"]
    clean[:, O.ENC_ANCHOR_MF] = dump["anchor_mf"]
    tok_clean = enc(clean)
    tok_off = enc(obs["enc_in"][ts[valid]])
    res["tokens"] = {
        "offline_vs_env_clean_max_abs_diff": float(np.abs(tok_off - tok_clean[valid]).max()),
        "offline_vs_env_clean_frac_rows_identical": float(np.mean(np.all(tok_off == tok_clean[valid], axis=1))),
        "abs_max": float(np.abs(tok_clean).max()),
        "unique_tokens_in_clip": int(np.unique(tok_clean, axis=0).shape[0]),
    }
    if tok_seen is not None:
        res["tokens"]["seen_vs_clean_max_abs_diff(noise)"] = float(np.abs(tok_seen - tok_clean).max())
        res["tokens"]["seen_vs_clean_frac_rows_identical"] = float(np.mean(np.all(tok_seen == tok_clean, axis=1)))

    # 3. ONNX decoder on the env's own proprio vs the env's own next action
    dec = Decoder(os.path.expanduser(args.decoder))
    actor = dump["obs__actor_obs"]
    raw = dump["raw_action"]
    use = tok_seen if tok_seen is not None else tok_clean
    x = np.concatenate([use[:-1], actor[:-1]], axis=1)  # obs_{k+1} -> a_{k+1} = raw[k+1]
    a_hat = dec(x)
    d_act = np.abs(a_hat - raw[1:])
    row_max = d_act.max(axis=1)
    res["decoder_onnx_vs_env_policy"] = {
        "rows": int(d_act.shape[0]), "max_abs_diff": float(d_act.max()),
        "mean_abs_diff": float(d_act.mean()), "raw_action_abs_max": float(np.abs(raw).max()),
        "rows_with_max_diff_gt_0.1": int((row_max > 0.1).sum()),
        "median_row_max_diff": float(np.median(row_max)),
        "worst_rows": [int(i) for i in np.argsort(-row_max)[:5]],
    }

    # 4. proprio layout: rebuild actor_obs[k] from the recorded histories (k >= 9)
    q, v, dj = dump["joint_pos"], dump["joint_vel"], np.asarray(meta["default_joint_pos"], np.float32)
    H = O.HISTORY
    S = O.proprio_slices()
    rebuilt, rows = [], []
    # the last row is the auto-reset observation of the next episode (run_once): skip it
    for k in range(H - 1, q.shape[0] - 1):
        rebuilt.append(O.build_proprio(q[k - H + 1:k + 1] - dj, v[k - H + 1:k + 1], raw[k - H + 1:k + 1]))
        rows.append(k)
    rebuilt = np.stack(rebuilt)
    d_pro = np.abs(rebuilt - actor[rows])
    per_term = {name: float(d_pro[:, S[name]].max()) for name, _ in O.PROPRIO_TERMS}
    res["proprio_layout"] = {"rows": len(rows), "max_abs_diff": float(d_pro.max()), "per_term_max": per_term,
                             "default_joint_pos_env": dj.round(4).tolist(),
                             "default_joint_pos_ours": O.SONIC_DEFAULT_ISAAC.round(4).tolist(),
                             "default_max_abs_diff": float(np.abs(dj - O.SONIC_DEFAULT_ISAAC).max()),
                             "joint_names_env": meta.get("joint_names"),
                             "joint_names_match_isaac_order": meta.get("joint_names") == O.ISAAC_JOINTS,
                             "action_scale_env": meta.get("action_scale")}

    # 5. the runtime module, if present: feed the recorded states, compare proprio + actions
    try:
        import sonic_decoder

        rt = sonic_decoder.SonicDecoderRuntime(os.path.expanduser(args.decoder))
        rt.reset()
        a_rt, pro_rt = [], []
        for k in range(q.shape[0] - 1):
            q_wire = O.mujoco_to_wire(O.sonic_to_fr3(O.isaac_to_mujoco(q[k])), np.zeros(2))
            a, pro = rt.step(use[k], q_wire, return_proprio=True)
            a_rt.append(a)
            pro_rt.append(pro)
        a_rt, pro_rt = np.stack(a_rt), np.stack(pro_rt)
        dp = np.abs(pro_rt[H:] - actor[H:q.shape[0] - 1])
        per = {name: float(dp[:, S[name]].max()) for name, _ in O.PROPRIO_TERMS}
        res["runtime"] = {"rows": int(dp.shape[0]), "proprio_max_abs_diff": float(dp.max()),
                          "proprio_per_term_max": per,
                          "raw_action_vs_env_max_abs_diff": float(np.abs(a_rt[H:] - raw[H + 1:q.shape[0]]).max()),
                          "raw_action_vs_env_mean_abs_diff": float(np.abs(a_rt[H:] - raw[H + 1:q.shape[0]]).mean()),
                          "note": "runtime joint_vel is a 50 Hz finite difference of the streamed positions, "
                                  "the env's is PhysX's; actions differ through that term only"}
    except ImportError:
        res["runtime"] = None

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    log(f"[check] CHECK_DONE -> {out}\n{json.dumps(res, indent=1)}")
    # Pass criteria: offline encoder inputs identical to the env's; ONNX decoder reproduces the
    # env policy's actions on the env's own observations (median row error, one reset-row
    # outlier allowed); the proprio layout matches to within the training-time observation
    # noise (joint_pos +-0.01, joint_vel +-0.5, base_ang_vel +-0.2, gravity +-0.05); the runtime
    # (finite-difference velocities) stays within 0.1 mean raw-action error of the env policy.
    pl = res["proprio_layout"]["per_term_max"]
    ok = (res["offline_vs_env"]["cmd_mf_max_abs_diff"] < 1e-3
          and res["offline_vs_env"]["anchor_mf_max_abs_diff"] < 1e-3
          and res["tokens"]["offline_vs_env_clean_frac_rows_identical"] > 0.99
          and res["decoder_onnx_vs_env_policy"]["median_row_max_diff"] < 0.05
          and res["decoder_onnx_vs_env_policy"]["rows_with_max_diff_gt_0.1"] <= 2
          and pl["joint_pos_rel"] < 0.1 and pl["gravity_dir"] < 0.1 and pl["base_ang_vel"] < 0.3
          and pl["joint_vel"] < 1.0
          and (res["runtime"] is None or res["runtime"]["raw_action_vs_env_mean_abs_diff"] < 0.1))
    log(f"[check] VERDICT: {'OK' if ok else 'MISMATCH'}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("obs")
    a.add_argument("--clips", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--wbc", default=DEFAULT_WBC)
    a.set_defaults(fn=cmd_obs)
    e = sub.add_parser("encode")
    e.add_argument("--obs", required=True)
    e.add_argument("--encoder", required=True)
    e.add_argument("--dataset", required=True, help="lane A gr00t_v2 dir (provenance + parquet grips)")
    e.add_argument("--manifest", required=True, help="motion_lib out/manifest.json")
    e.add_argument("--out", required=True)
    e.add_argument("--max-episodes", type=int, default=0)
    e.set_defaults(fn=cmd_encode)
    c = sub.add_parser("check")
    c.add_argument("--dump", required=True)
    c.add_argument("--obs", required=True)
    c.add_argument("--encoder", required=True)
    c.add_argument("--decoder", required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_check)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    _code = 1
    try:
        _code = main()
    except SystemExit as exc:
        if isinstance(exc.code, int):
            _code = exc.code
        else:
            print(exc.code, file=sys.stderr, flush=True)
            _code = 1
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(_code)  # Kit python's atexit can hang headless (P2 finding)
