#!/usr/bin/env python3
"""BONES-SEED SMPL pkl -> GMR SMPL-X npz.

Mirrors GMR's own ``scripts/smpl_to_smplx.py`` (which only handles npz inputs)
and writes exactly the keys ``general_motion_retargeting/utils/smpl.py::
load_smplx_file`` reads: ``gender``, ``betas`` (16,), ``root_orient`` (T,3),
``pose_body`` (T,63), ``trans`` (T,3), ``mocap_frame_rate``.

Verified on ``/data/lustre/shared/datasets/bones-seed/human-smpl`` (2026-09-03):
the clips are **joblib** pickles, not plain ones (``pickle.load`` raises
``UnpicklingError: invalid load key, 'x'``), holding
``{pose_aa (T,72) f32, transl (T,3) f32, smpl_joints (T,24,3) f32, fps 50.0,
original_pose_aa (T0,72) f32, original_fps 30.0}`` — **no betas**, so
``betas = zeros(16)`` and GMR derives a 1.66 m human for every actor
(``human_height = 1.66 + 0.1*betas[0]``).  ``pose_aa`` columns 66:72 are SMPL's
two hand joints and are dropped, as upstream does.

    python3 smpl_pkl_to_smplx_npz.py --src <pkl|dir> --dst <dir> [--limit N]
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np


def load_pkl(path: pathlib.Path) -> dict:
    try:
        import joblib

        return joblib.load(path)
    except Exception:
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)


def convert(src: pathlib.Path, dst: pathlib.Path, gender: str = "neutral") -> pathlib.Path:
    d = load_pkl(src)
    pose = np.asarray(d["pose_aa"], dtype=np.float32)
    if pose.ndim != 2 or pose.shape[1] < 66:
        raise ValueError(f"{src}: pose_aa has shape {pose.shape}, expected (T,>=66)")
    betas = np.asarray(d.get("betas", np.zeros(16)), dtype=np.float32).reshape(-1)
    if betas.size < 16:
        betas = np.concatenate([betas, np.zeros(16 - betas.size, dtype=np.float32)])
    out = {
        "gender": np.array(gender),
        "betas": betas[:16],
        "root_orient": pose[:, :3],
        "pose_body": pose[:, 3:66],
        "trans": np.asarray(d["transl"], dtype=np.float32),
        "mocap_frame_rate": np.array(float(d.get("fps", 50.0))),
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **out)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=pathlib.Path, required=True, help="pkl file or directory")
    ap.add_argument("--dst", type=pathlib.Path, required=True, help="output directory")
    ap.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
    ap.add_argument("--limit", type=int, default=None, help="convert at most N clips")
    args = ap.parse_args()

    srcs = sorted(args.src.rglob("*.pkl")) if args.src.is_dir() else [args.src]
    if args.limit is not None:
        srcs = srcs[: args.limit]
    ok = bad = 0
    for s in srcs:
        try:
            print(convert(s, args.dst / (s.stem + ".npz"), args.gender))
            ok += 1
        except Exception as e:  # one bad clip must not kill a 1k-clip sweep
            print(f"SKIP {s}: {type(e).__name__}: {e}")
            bad += 1
    print(f"converted {ok}, skipped {bad}, out {args.dst}")


if __name__ == "__main__":
    main()
