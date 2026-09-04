#!/usr/bin/env python3
"""Build the lane-B SONIC motion library for the dual-FR3 embodiment (CPU only).

Reads the P0/P1 handover demos (HDF5, 50 Hz, `obs/joint_pos_{left,right}` —
the *measured* FR3 joint positions, never `joint_target_*`, which carry the IK
spikes), resamples them to 30 fps, mirrors them about the rig's x=0 plane and
applies a deterministic set of augmentations. Each resulting clip is written as
one `joblib` pkl in the motion_lib format `gear_sonic`'s
`MotionLibBase.load_motion_with_skeleton` reads:

    {clip_name: {
        "root_trans_offset": (T,3) f32   zeros — `base` is at the origin, fixed base
        "pose_aa":           (T,17,3) f32 MuJoCo body order, [0]=root=identity,
                                          jointed bodies = axis * dof, link0s zero
        "dof":               (T,14) f32  MuJoCo joint order, left 1..7 then right 1..7
        "root_rot":          (T,4) f32   identity, **wxyz** (plan/PLAN.md contract;
                                          NOTE the upstream CSV converter
                                          `convert_soma_csv_to_motion_lib.py` writes
                                          scipy **xyzw** there — ours is wxyz)
        "smpl_joints":       (T,24,3) f32 zeros
        "fps":               30 (int)
    }}

Joint-6 convention (`J6_OFFSET`, see `harness/lane_b/make_dual_fr3_xml.py` and the
"JOINT 6 CONVENTION" line in `dual_fr3.xml`'s header): the MJCF pre-rotates
link6's frame by Rz(2.5307), so `q6_here = q6_fr3 - 2.5307` and joint 6's range
is [-1.9862, 1.9862] instead of the FR3's [0.5445, 4.5169]. Kinematics are
unchanged; the point is that the demos push left joint 6 past pi (up to 4.517),
and `gear_sonic`'s motion library re-derives joint angles through an
axis-angle -> quaternion -> axis-angle round trip that wraps anything beyond pi
by 2*pi. The offset is subtracted from the measured FR3 angles before any
mirroring / augmentation / clipping, and the script asserts every written dof
value satisfies |q| <= pi - 0.01. The mirror is unaffected: joint 6 keeps its
sign under the mirror, so the constant offset commutes with it.

Mirror rule (verified by `--self-test`, not assumed): swap the arms and negate
joints 1, 3, 5, 7 of each arm.  With the arm bases at (±0.1, 0.005, 0.14) the
mirrored configuration reproduces every body position of the original reflected
in x to ~1e-16 m, and the link7 frames differ from the reflected originals by a
*constant* relabelling K = diag(-1,-1,1), which is what pins the sign of joint 7
(joint 7 does not move link7's origin, so a position-only test cannot see it).

Usage
-----
    PYTHONUSERBASE=~/env/pyuser-sonic /isaac-sim/python.sh \
        harness/lane_b/demos_to_sonic_pkl.py --self-test

    PYTHONUSERBASE=~/env/pyuser-sonic /isaac-sim/python.sh \
        harness/lane_b/demos_to_sonic_pkl.py \
        --run-dir ~/runs/franka-sonic/lane_b/2026-09-03_motion_lib --verify 3

Everything it writes goes under `--run-dir` (AGENTS.md rule e). It never
deletes anything: re-running over an existing out/motions overwrites clips with
the same name and leaves anything else in place (AGENTS.md rule o).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import itertools
import json
import os
import re
import socket
import subprocess
import sys
import time
import zlib

import numpy as np

# ---------------------------------------------------------------- constants

DEFAULT_SHARDS = os.path.expanduser(
    "~/runs/franka-sonic/shared/2026-09-03_demos-2/out/export/demos_shard*.hdf5"
)
DEFAULT_XML = os.path.expanduser(
    "~/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/dual_fr3.xml"
)
DEFAULT_WBC = os.path.expanduser("~/GR00T-WholeBodyControl")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# swap the arms and negate joints 1,3,5,7 — see --self-test
MIRROR_SIGNS = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

# SONIC joint 6 = FR3 joint 6 - J6_OFFSET. dual_fr3.xml pre-rotates link6's frame
# by Rz(J6_OFFSET) (see harness/lane_b/make_dual_fr3_xml.py, and the "JOINT 6
# CONVENTION" line in the MJCF header), which shifts joint 6's range from the
# FR3's [0.5445, 4.5169] to [-1.9862, 1.9862] without changing the kinematics.
# Needed because the demos drive joint 6 past pi and the motion library's
# axis-angle -> quaternion -> axis-angle round trip wraps anything beyond pi.
J6_OFFSET = 2.5307
J6_DOF_INDICES = (5, 12)  # left_fr3_joint6, right_fr3_joint6 in MuJoCo dof order
MAX_ABS_DOF = np.pi - 0.01

# skips crop2/noise2 from the P2 menu so the library lands near ~1k clips
DEFAULT_AUGS = "tw080,tw125,off1,off2,off3,noise1,crop1"
ALL_AUGS = ("tw080", "tw125", "off1", "off2", "off3", "noise1", "noise2", "crop1", "crop2")

TARGET_FPS = 30
SRC_FPS_EXPECTED = 50.0
N_BODIES = 17
N_DOF = 14
MIN_FRAMES = 60
CROP_MIN_FRAMES = 90


# ------------------------------------------------------------------- model


class Rig:
    """Everything the converter needs from dual_fr3.xml, read straight from MuJoCo."""

    def __init__(self, xml: str):
        import mujoco

        self.xml = os.path.abspath(os.path.expanduser(xml))
        self.model = mujoco.MjModel.from_xml_path(self.xml)
        m = self.model
        assert m.nq == N_DOF and m.njnt == N_DOF, f"expected {N_DOF} hinge DoF, got nq={m.nq}"
        self.body_names_full = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)
        ]
        assert self.body_names_full[0] == "world"
        # pose_aa body order == Humanoid_Batch body order == MuJoCo bodies minus `world`
        self.body_names = self.body_names_full[1:]
        assert len(self.body_names) == N_BODIES, self.body_names
        self.joint_names = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)
        ]
        expected = [f"{s}_fr3_joint{i}" for s in ("left", "right") for i in range(1, 8)]
        assert self.joint_names == expected, self.joint_names
        self.lo = m.jnt_range[:, 0].astype(np.float64).copy()
        self.hi = m.jnt_range[:, 1].astype(np.float64).copy()
        self.axis = m.jnt_axis.astype(np.float64).copy()  # (14,3)
        # pose_aa row for each joint = its body, with `world` dropped
        self.joint_body_row = np.array([int(b) - 1 for b in m.jnt_bodyid], dtype=np.int64)
        assert (self.joint_body_row >= 0).all()
        # The MJCF must be the J6_OFFSET ("JOINT 6 CONVENTION") revision, otherwise a
        # shifted converter would silently write clips against an unshifted rig.
        for k in J6_DOF_INDICES:
            assert abs(self.lo[k] + 1.9862) < 1e-3 and abs(self.hi[k] - 1.9862) < 1e-3, (
                f"{self.joint_names[k]} range {self.lo[k]}..{self.hi[k]} is not the "
                f"J6_OFFSET={J6_OFFSET} convention — regenerate dual_fr3.xml with "
                "harness/lane_b/make_dual_fr3_xml.py"
            )
        assert (np.abs(self.lo) <= MAX_ABS_DOF).all() and (np.abs(self.hi) <= MAX_ABS_DOF).all(), (
            "some jnt_range endpoint is outside (-pi, pi); the motion library's "
            "axis-angle round trip would wrap it"
        )

    def pose_aa(self, dof: np.ndarray) -> np.ndarray:
        """(T,14) dof -> (T,17,3) axis-angle in MuJoCo body order, root identity."""
        T = dof.shape[0]
        out = np.zeros((T, N_BODIES, 3), dtype=np.float32)
        for k in range(N_DOF):
            out[:, self.joint_body_row[k], :] = (
                self.axis[k][None, :] * dof[:, k : k + 1]
            ).astype(np.float32)
        return out

    def clip(self, dof: np.ndarray, margin: float = 0.0) -> np.ndarray:
        return np.clip(dof, self.lo + margin, self.hi - margin)

    def in_limits(self, dof: np.ndarray, tol: float = 1e-6) -> bool:
        return bool((dof >= self.lo - tol).all() and (dof <= self.hi + tol).all())


def mirror_dof(dof: np.ndarray) -> np.ndarray:
    """Mirror about the rig's x=0 plane: swap arms, negate joints 1,3,5,7."""
    left, right = dof[:, :7], dof[:, 7:]
    return np.concatenate([MIRROR_SIGNS * right, MIRROR_SIGNS * left], axis=1)


# --------------------------------------------------------------- self-test


def self_test(rig: Rig, n_poses: int = 8, seed: int = 0, search: bool = True) -> dict:
    """Verify MIRROR_SIGNS numerically with mj_forward; search 2^7 if it fails."""
    import mujoco

    m, d = rig.model, __import__("mujoco").MjData(rig.model)
    bn = rig.body_names_full
    L = [bn.index(f"left_fr3_link{i}") for i in range(8)]
    R = [bn.index(f"right_fr3_link{i}") for i in range(8)]
    Mx = np.diag([-1.0, 1.0, 1.0])
    rng = np.random.default_rng(seed)
    qs = rng.uniform(rig.lo, rig.hi, size=(n_poses, N_DOF))

    def fk(q):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        return d.xpos.copy(), d.xmat.copy().reshape(-1, 3, 3)

    def score(signs):
        """(max body-position error, spread of the link7 frame relabelling K)."""
        pos_err = 0.0
        Ks = []
        for q in qs:
            qm = np.concatenate([signs * q[7:], signs * q[:7]])
            if not rig.in_limits(qm[None, :]):
                return np.inf, np.inf
            p0, r0 = fk(q)
            p1, r1 = fk(qm)
            pos_err = max(
                pos_err,
                float(np.abs(p1[R] - p0[L] @ Mx).max()),
                float(np.abs(p1[L] - p0[R] @ Mx).max()),
            )
            Ks.append((Mx @ r0[L[7]] @ Mx).T @ r1[R[7]])
        Ks = np.array(Ks)
        return pos_err, float(np.abs(Ks - Ks.mean(0)).max())

    pos_err, k_spread = score(MIRROR_SIGNS)
    ok = pos_err < 1e-6 and k_spread < 1e-6
    res = {
        "pattern": [int(s) for s in MIRROR_SIGNS],
        "max_body_pos_err_m": pos_err,
        "link7_frame_relabel_spread": k_spread,
        "n_poses": n_poses,
        "passed": bool(ok),
    }
    print(
        f"[self-test] mirror pattern {res['pattern']}: "
        f"max body-position error {pos_err:.3e} m, "
        f"link7 frame-relabel spread {k_spread:.3e} -> {'PASS' if ok else 'FAIL'}"
    )
    if not ok and search:
        print("[self-test] declared pattern failed; searching all 2^7 sign patterns...")
        cands = []
        for pat in itertools.product([1, -1], repeat=7):
            p, k = score(np.array(pat, dtype=float))
            if np.isfinite(p):
                cands.append((max(p, k), p, k, list(pat)))
        cands.sort()
        for tot, p, k, pat in cands[:5]:
            print(f"[self-test]   {pat}: pos {p:.3e} m  K-spread {k:.3e}")
        res["search_best"] = [
            {"pattern": pat, "pos_err": p, "k_spread": k} for _, p, k, pat in cands[:5]
        ]
    return res


# ------------------------------------------------------------ resample/aug


def resample(q: np.ndarray, fps_in: float, fps_out: int, speed: float = 1.0) -> np.ndarray:
    """Linear interpolation onto t_k = k/fps_out, optionally played at `speed`x."""
    T = q.shape[0]
    src_t = np.arange(T, dtype=np.float64) / fps_in
    duration = src_t[-1] / speed
    K = int(np.floor(duration * fps_out)) + 1
    dst_t = np.arange(K, dtype=np.float64) / fps_out * speed
    out = np.empty((K, q.shape[1]), dtype=np.float64)
    for j in range(q.shape[1]):
        out[:, j] = np.interp(dst_t, src_t, q[:, j])
    return out


def _rng_for(seed: int, name: str, aug: str) -> np.random.Generator:
    crc = zlib.crc32(f"{name}|{aug}".encode()) & 0xFFFFFFFF
    return np.random.default_rng(np.random.SeedSequence([seed, crc]))


def aug_offset(q: np.ndarray, fps: int, rng: np.random.Generator, max_amp: float = 0.05):
    """Smooth per-joint offset: sum of two low-frequency sinusoids, |off| <= max_amp."""
    T, J = q.shape
    t = np.arange(T, dtype=np.float64) / fps
    amp = rng.uniform(0.3, 1.0, size=J) * max_amp
    w = rng.uniform(0.3, 0.7, size=J)
    f1 = rng.uniform(0.05, 0.25, size=J)
    f2 = rng.uniform(0.15, 0.50, size=J)
    p1 = rng.uniform(0.0, 2 * np.pi, size=J)
    p2 = rng.uniform(0.0, 2 * np.pi, size=J)
    off = (amp * w) * np.sin(2 * np.pi * f1 * t[:, None] + p1) + (amp * (1 - w)) * np.sin(
        2 * np.pi * f2 * t[:, None] + p2
    )
    return q + off


def aug_noise(q: np.ndarray, fps: int, rng: np.random.Generator, sigma: float = 0.01):
    """Gaussian noise low-pass filtered at 3 Hz, rescaled back to sigma rad."""
    from scipy import signal

    T, J = q.shape
    white = rng.normal(0.0, 1.0, size=(T, J))
    if T > 24:
        b, a = signal.butter(2, 3.0 / (0.5 * fps), btype="low")
        smooth = signal.filtfilt(b, a, white, axis=0)
    else:  # too short for filtfilt's edge padding — fall back to a 5-frame mean
        k = np.ones(5) / 5.0
        smooth = np.stack([np.convolve(white[:, j], k, mode="same") for j in range(J)], axis=1)
    std = smooth.std(axis=0, keepdims=True)
    std[std < 1e-9] = 1e-9
    noise = np.clip(smooth / std * sigma, -4 * sigma, 4 * sigma)
    return q + noise


def aug_crop(q: np.ndarray, rng: np.random.Generator, frac_lo=0.6, frac_hi=0.8):
    T = q.shape[0]
    length = int(round(rng.uniform(frac_lo, frac_hi) * T))
    length = max(min(length, T), min(CROP_MIN_FRAMES, T))
    start = int(rng.integers(0, T - length + 1))
    return q[start : start + length]


def apply_aug(aug: str, q30: np.ndarray, q_src: np.ndarray, fps_in: float,
              rng: np.random.Generator) -> np.ndarray:
    """`q30` is the 30 fps base clip; time warps re-resample from `q_src` (50 Hz)."""
    if aug == "":
        return q30
    if aug == "tw080":
        return resample(q_src, fps_in, TARGET_FPS, speed=0.8)
    if aug == "tw125":
        return resample(q_src, fps_in, TARGET_FPS, speed=1.25)
    if aug.startswith("off"):
        return aug_offset(q30, TARGET_FPS, rng)
    if aug.startswith("noise"):
        return aug_noise(q30, TARGET_FPS, rng)
    if aug.startswith("crop"):
        return aug_crop(q30, rng)
    raise ValueError(f"unknown augmentation {aug!r}")


# ------------------------------------------------------------------ output


def make_entry(rig: Rig, dof: np.ndarray, fps: int = TARGET_FPS) -> dict:
    T = dof.shape[0]
    dof32 = dof.astype(np.float32)
    root_rot = np.zeros((T, 4), dtype=np.float32)
    root_rot[:, 0] = 1.0  # identity in WXYZ — see module docstring
    assert np.allclose(root_rot[0], np.array([1.0, 0.0, 0.0, 0.0])), (
        "root_rot must be the wxyz identity [1,0,0,0]; the upstream CSV converter "
        "writes scipy xyzw here, our contract (plan/PLAN.md) is wxyz"
    )
    entry = {
        "root_trans_offset": np.zeros((T, 3), dtype=np.float32),
        "pose_aa": rig.pose_aa(dof),
        "dof": dof32,
        "root_rot": root_rot,
        "smpl_joints": np.zeros((T, 24, 3), dtype=np.float32),
        "fps": int(fps),
    }
    assert entry["pose_aa"].shape == (T, N_BODIES, 3)
    assert not np.isnan(entry["pose_aa"]).any() and not np.isnan(dof32).any()
    return entry


def write_clip(out_dir: str, name: str, entry: dict) -> int:
    import joblib

    path = os.path.join(out_dir, f"{name}.pkl")
    joblib.dump({name: entry}, path, compress=True)
    return os.path.getsize(path)


# -------------------------------------------------------------------- load


def load_demos(patterns: list[str], limit: int | None = None) -> list[dict]:
    import h5py

    files: list[str] = []
    for pat in patterns:
        files.extend(sorted(glob.glob(os.path.expanduser(pat))))
    assert files, f"no HDF5 shards matched {patterns}"
    demos = []
    n_screened_out = 0
    for path in files:
        base = os.path.basename(path)
        # `demos_shard0.hdf5` -> "0". Take the digit run right after "shard": a
        # plain isdigit() filter would swallow the "5" of ".hdf5" and give "05".
        mo = re.match(r"\d+", base.split("shard")[-1])
        shard = mo.group(0) if mo else "x"
        with h5py.File(path, "r") as f:
            fps = float(f.attrs.get("fps", SRC_FPS_EXPECTED))
            assert abs(fps - SRC_FPS_EXPECTED) < 1e-6, f"{path}: fps={fps}, expected 50"
            for key in sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1])):
                g = f["data"][key]
                rs = g.attrs.get("replay_success")
                if isinstance(rs, bytes):
                    rs = rs.decode()
                if str(rs) != "True":
                    continue
                # `jointpos_replay_success` (harness/data/jointpos_screen.py, P7) marks the
                # episodes whose recorded ABSOLUTE joint targets reproduce the handover on the
                # evaluation's own JointPos controller. It is absent on round-1 shards, so a
                # missing attr means True and every round-1 command reproduces byte for byte.
                # P8: the reference library must be the screened set, so lane B's episode count
                # equals lane A's dataset (P7 STATUS 2026-09-04 15:55).
                jp = g.attrs.get("jointpos_replay_success")
                if jp is not None:
                    if isinstance(jp, bytes):
                        jp = jp.decode()
                    if str(jp) != "True":
                        n_screened_out += 1
                        continue
                ql = np.asarray(g["obs/joint_pos_left"], dtype=np.float64)
                qr = np.asarray(g["obs/joint_pos_right"], dtype=np.float64)
                n = int(g.attrs["num_samples"])
                assert ql.shape == (n, 7) and qr.shape == (n, 7), (path, key, ql.shape)
                q = np.concatenate([ql, qr], axis=1)
                # FR3 -> SONIC joint-6 convention, before any mirror/augment/clip.
                for k in J6_DOF_INDICES:
                    q[:, k] -= J6_OFFSET
                demos.append({
                    "name": f"handover_s{shard}_d{key.split('_')[-1]}",
                    "shard": path,
                    "demo": key,
                    "fps": fps,
                    "q": q,
                })
                if limit is not None and len(demos) >= limit:
                    print(f"[load] {len(demos)} demos (limit), "
                          f"{n_screened_out} skipped by jointpos_replay_success", flush=True)
                    return demos
    print(f"[load] {n_screened_out} demos skipped by jointpos_replay_success=False", flush=True)
    return demos


# ------------------------------------------------------------------ verify


def verify_fk(run_dir: str, xml: str, wbc: str, n: int, seed: int) -> dict:
    """Humanoid_Batch fk_batch vs mujoco mj_forward on a few written pkls."""
    import joblib
    import mujoco

    out_dir = os.path.join(run_dir, "out", "motions")
    paths = sorted(glob.glob(os.path.join(out_dir, "*.pkl")))
    assert paths, f"no pkls in {out_dir}"
    rng = np.random.default_rng(seed)
    picks = [paths[i] for i in rng.choice(len(paths), size=min(n, len(paths)), replace=False)]

    m = mujoco.MjModel.from_xml_path(os.path.abspath(os.path.expanduser(xml)))
    d = mujoco.MjData(m)
    bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
    l7, r7 = bn.index("left_fr3_link7"), bn.index("right_fr3_link7")

    wbc = os.path.abspath(os.path.expanduser(wbc))
    sys.path.insert(0, wbc)
    cwd0 = os.getcwd()
    os.chdir(wbc)
    try:
        import torch
        from easydict import EasyDict

        from gear_sonic.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

        cfg = EasyDict({
            "asset": {
                "assetRoot": os.path.dirname(os.path.abspath(os.path.expanduser(xml))),
                "assetFileName": os.path.basename(xml),
            },
            "extend_config": [],
        })
        hb = Humanoid_Batch(cfg)
        results = []
        for p in picks:
            blob = joblib.load(p)
            name = os.path.splitext(os.path.basename(p))[0]
            assert list(blob.keys()) == [name], (p, list(blob.keys()))
            e = blob[name]
            T = e["dof"].shape[0]
            assert e["root_trans_offset"].shape == (T, 3)
            assert e["pose_aa"].shape == (T, N_BODIES, 3)
            assert e["dof"].shape == (T, N_DOF)
            assert e["root_rot"].shape == (T, 4)
            assert e["smpl_joints"].shape == (T, 24, 3)
            for k in ("root_trans_offset", "pose_aa", "dof", "root_rot", "smpl_joints"):
                assert e[k].dtype == np.float32, (p, k, e[k].dtype)
            assert e["fps"] == TARGET_FPS and isinstance(e["fps"], int)
            assert np.allclose(e["root_rot"], np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
            assert np.abs(e["root_trans_offset"]).max() == 0.0
            assert np.abs(e["dof"]).max() <= MAX_ABS_DOF, (p, float(np.abs(e["dof"]).max()))
            out = hb.fk_batch(
                torch.from_numpy(e["pose_aa"])[None],
                torch.from_numpy(e["root_trans_offset"])[None],
                return_full=True,
            )
            fk_pos = out.global_translation[0].numpy()
            fk_dof = out.dof_pos[0].numpy()
            worst = 0.0
            for t in range(T):
                d.qpos[:] = e["dof"][t]
                mujoco.mj_forward(m, d)
                worst = max(
                    worst,
                    float(np.abs(fk_pos[t, l7 - 1] - d.xpos[l7]).max()),
                    float(np.abs(fk_pos[t, r7 - 1] - d.xpos[r7]).max()),
                )
            dof_err = float(np.abs(fk_dof - e["dof"]).max())
            print(
                f"[verify] {name}: T={T} link7 FK error {worst:.3e} m, "
                f"dof round-trip {dof_err:.3e} rad"
            )
            assert worst < 1e-4, f"{name}: FK link7 error {worst}"
            assert dof_err < 1e-5, f"{name}: return_full dof_pos != dof ({dof_err})"
            results.append({"clip": name, "frames": T, "link7_fk_err_m": worst,
                            "dof_roundtrip_err_rad": dof_err})
    finally:
        os.chdir(cwd0)
    return {"checked": results,
            "max_link7_fk_err_m": max(r["link7_fk_err_m"] for r in results),
            "max_dof_roundtrip_err_rad": max(r["dof_roundtrip_err_rad"] for r in results)}


# ---------------------------------------------------------------------- cli


def repo_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", REPO, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"unknown ({exc})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=None,
                    help="run folder (AGENTS.md rule e); out/motions is written under it")
    ap.add_argument("--shards", nargs="+", default=[DEFAULT_SHARDS],
                    help="HDF5 demo shard paths or globs")
    ap.add_argument("--xml", default=DEFAULT_XML, help="dual_fr3.xml (MuJoCo body/joint order)")
    ap.add_argument("--wbc", default=DEFAULT_WBC, help="GR00T-WholeBodyControl checkout")
    ap.add_argument("--augs", default=DEFAULT_AUGS,
                    help=f"comma list, subset of {','.join(ALL_AUGS)} (empty = originals only)")
    ap.add_argument("--no-mirror", action="store_true", help="skip the _M mirrors")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-demos", type=int, default=None, help="smoke: only the first N demos")
    ap.add_argument("--self-test", action="store_true",
                    help="verify the mirror sign pattern with mujoco and exit")
    ap.add_argument("--verify", type=int, default=0,
                    help="after the build, FK-check N random pkls against mujoco")
    args = ap.parse_args()

    rig = Rig(args.xml)
    print(f"[rig] {rig.xml}")
    print(f"[rig] bodies ({len(rig.body_names)}): {rig.body_names}")
    print(f"[rig] joints ({len(rig.joint_names)}): {rig.joint_names}")
    print(f"[rig] lo: {np.round(rig.lo, 4).tolist()}")
    print(f"[rig] hi: {np.round(rig.hi, 4).tolist()}")
    print(f"[rig] J6_OFFSET = {J6_OFFSET} rad subtracted from FR3 joint 6 "
          f"(dof indices {J6_DOF_INDICES}); |dof| bound {MAX_ABS_DOF:.4f} rad")

    st = self_test(rig, seed=args.seed)
    if args.self_test:
        return 0 if st["passed"] else 1
    if not st["passed"]:
        print("[error] mirror self-test failed — refusing to build the library", file=sys.stderr)
        return 1

    assert args.run_dir, "--run-dir is required unless --self-test"
    run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
    out_dir = os.path.join(run_dir, "out", "motions")
    os.makedirs(out_dir, exist_ok=True)

    augs = [a for a in args.augs.split(",") if a]
    for a in augs:
        assert a in ALL_AUGS, f"unknown augmentation {a!r}; known: {ALL_AUGS}"

    demos = load_demos(args.shards, args.limit_demos)
    print(f"[load] {len(demos)} replay_success demos, "
          f"frames {min(d['q'].shape[0] for d in demos)}..{max(d['q'].shape[0] for d in demos)} "
          f"@ 50 Hz")

    config = {
        "args": vars(args),
        "hostname": socket.gethostname(),
        "date": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "repo_sha": repo_sha(),
        "repo": REPO,
        "xml": rig.xml,
        "source_shards": sorted({d["shard"] for d in demos}),
        "n_source_demos": len(demos),
        "augs": augs,
        "mirror": not args.no_mirror,
        "target_fps": TARGET_FPS,
        "j6_offset_rad": J6_OFFSET,
        "mirror_signs": [int(s) for s in MIRROR_SIGNS],
        "mirror_self_test": st,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")

    manifest = []
    total_bytes = 0
    lib_lo = np.full(N_DOF, np.inf)
    lib_hi = np.full(N_DOF, -np.inf)
    max_orig_clip = 0.0
    max_abs_dof = 0.0
    skipped = []
    t0 = time.time()

    for di, dm in enumerate(demos):
        base30 = resample(dm["q"], dm["fps"], TARGET_FPS)
        # the recorded joint_pos can sit a hair outside jnt_range; pull it in so
        # every clip in the library satisfies the limit assert below.
        max_orig_clip = max(max_orig_clip, float(np.abs(rig.clip(base30) - base30).max()))
        base30 = rig.clip(base30)

        for aug in [""] + augs:
            rng = _rng_for(args.seed, dm["name"], aug or "orig")
            q = apply_aug(aug, base30, dm["q"], dm["fps"], rng)
            margin = 0.02 if (aug.startswith("off") or aug.startswith("noise")) else 0.0
            q = rig.clip(q, margin)
            name = dm["name"] + (f"_{aug}" if aug else "")
            variants = [(name, q)]
            if not args.no_mirror:
                variants.append((name + "_M", mirror_dof(q)))
            for cname, cq in variants:
                if cq.shape[0] < MIN_FRAMES:
                    skipped.append({"clip": cname, "frames": int(cq.shape[0]),
                                    "why": f"< {MIN_FRAMES} frames"})
                    continue
                assert not np.isnan(cq).any(), cname
                # J6_OFFSET convention: nothing may reach pi, or the motion
                # library's axis-angle round trip wraps it by 2*pi.
                assert np.abs(cq).max() <= MAX_ABS_DOF, (
                    cname, "|dof| > pi - 0.01", float(np.abs(cq).max()),
                    int(np.abs(cq).max(0).argmax()),
                )
                assert rig.in_limits(cq), (
                    cname,
                    float((rig.lo - cq.min(0)).max()),
                    float((cq.max(0) - rig.hi).max()),
                )
                entry = make_entry(rig, cq)
                nbytes = write_clip(out_dir, cname, entry)
                total_bytes += nbytes
                lib_lo = np.minimum(lib_lo, cq.min(0))
                lib_hi = np.maximum(lib_hi, cq.max(0))
                max_abs_dof = max(max_abs_dof, float(np.abs(cq).max()))
                manifest.append({
                    "name": cname,
                    "source_shard": dm["shard"],
                    "source_demo": dm["demo"],
                    "aug": aug or "orig",
                    "mirrored": cname.endswith("_M"),
                    "frames": int(cq.shape[0]),
                    "fps": TARGET_FPS,
                    "duration_s": round(cq.shape[0] / TARGET_FPS, 4),
                    "bytes": nbytes,
                })
        if (di + 1) % 10 == 0 or di + 1 == len(demos):
            print(f"[build] demo {di + 1}/{len(demos)}  clips so far {len(manifest)}  "
                  f"{total_bytes / 1e6:.1f} MB  {time.time() - t0:.0f}s", flush=True)

    with open(os.path.join(run_dir, "out", "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write("\n")

    n_orig = sum(1 for c in manifest if c["aug"] == "orig" and not c["mirrored"])
    n_mirror = sum(1 for c in manifest if c["mirrored"])
    n_augmented = sum(1 for c in manifest if c["aug"] != "orig")
    summary = {
        "counts": {
            "source_demos": len(demos),
            "originals": n_orig,
            "mirrors": n_mirror,
            "augmented": n_augmented,
            "augmented_unmirrored": sum(
                1 for c in manifest if c["aug"] != "orig" and not c["mirrored"]
            ),
            "total": len(manifest),
        },
        "augs": augs,
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / 1e6, 2),
        "frames": {
            "min": min(c["frames"] for c in manifest),
            "max": max(c["frames"] for c in manifest),
            "total": sum(c["frames"] for c in manifest),
        },
        "fps": TARGET_FPS,
        "joint_min": np.round(lib_lo, 6).tolist(),
        "joint_max": np.round(lib_hi, 6).tolist(),
        "jnt_range_lo": np.round(rig.lo, 6).tolist(),
        "jnt_range_hi": np.round(rig.hi, 6).tolist(),
        "max_original_limit_clip_rad": max_orig_clip,
        "j6_offset_rad": J6_OFFSET,
        "j6_convention": (
            "dof[:,5] and dof[:,12] are FR3 joint 6 minus J6_OFFSET=2.5307 rad; "
            "dual_fr3.xml pre-rotates link6 by Rz(J6_OFFSET) so the kinematics are "
            "unchanged and every joint stays inside (-pi, pi)"
        ),
        "max_abs_dof_rad": max_abs_dof,
        "max_abs_dof_bound": float(MAX_ABS_DOF),
        "mirror_self_test": st,
        "skipped": skipped,
        "build_seconds": round(time.time() - t0, 1),
        "out_dir": out_dir,
    }
    print(f"[build] {summary['counts']}")
    print(f"[build] {summary['total_mb']} MB in {summary['build_seconds']}s; "
          f"frames {summary['frames']['min']}..{summary['frames']['max']}")
    print(f"[build] max limit clip applied to an original: {max_orig_clip:.4f} rad")

    if args.verify:
        summary["fk_verification"] = verify_fk(run_dir, args.xml, args.wbc, args.verify,
                                               args.seed)

    with open(os.path.join(run_dir, "out", "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    print(f"[done] {len(manifest)} clips -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
