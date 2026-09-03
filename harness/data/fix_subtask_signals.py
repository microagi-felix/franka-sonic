"""Make the auto-annotated `left_placed` signal MimicGen-parsable on the 0.20 m angled rig.

`mimic/mdp.py:object_placed_in_zone` fires when the block rests within 0.10 m of the centre pad
with the releasing gripper open. On the angled rig the start slot is exactly 0.10 m from the
centre slot (ARM_SPACING 0.20), so the signal is already TRUE at frame 0 for most spawns. Isaac
Lab Mimic derives a subtask's end as the first non-zero diff of its termination signal
(`datagen_info_pool.py`), which for a signal starting at 1 is the FALLING edge when the gripper
closes — the same frame `grasp_a` rises — and generation dies with
"subtask termination signal is not increasing: 255 should be greater than 255" (2026-09-03).

Fix, applied to a COPY of the annotated file (nothing is edited in place, nothing deleted):
`left_placed` is zeroed before the first rising edge of `grasp_a` in every episode — a place
cannot precede the grasp — so its first transition is the real placement. Prints the subtask
boundaries MimicGen will compute and refuses to write a file that would still fail.

    ~/Isaac-GR00T/.venv/bin/python harness/data/fix_subtask_signals.py \
        --input <run>/out/sources_annotated.hdf5 --output <run>/out/sources_annotated_fixed.hdf5
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

import h5py
import numpy as np

# eef -> ordered termination signals of its non-final subtasks (mimic/env_cfg.py)
SUBTASKS = {"left": ["grasp_a", "left_placed"], "right": ["left_placed", "grasp_c"]}
GATE = "grasp_a"  # left_placed may not be true before this rises
TARGET = "left_placed"


def first_rising(sig: np.ndarray) -> int | None:
    d = np.diff(sig.astype(np.int64))
    hits = np.flatnonzero(d > 0)
    return int(hits[0]) + 1 if len(hits) else None


def mimic_boundaries(sig: dict[str, np.ndarray], n_actions: int) -> dict[str, list[tuple[int, int]]]:
    """Replicates datagen_info_pool._add_episode's parsing (first NON-ZERO diff, either sign)."""
    out = {}
    for eef, names in SUBTASKS.items():
        prev, bounds = 0, []
        for name in names:
            d = np.diff(sig[name].astype(np.int64))
            nz = np.flatnonzero(d)
            if not len(nz):
                raise ValueError(f"{eef}/{name}: signal never changes")
            end = int(nz[0]) + 2
            if end <= prev:
                raise ValueError(f"{eef}/{name}: end {end} <= start {prev} (not increasing)")
            bounds.append((prev, end))
            prev = end
        bounds.append((prev, n_actions))
        out[eef] = bounds
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if os.path.exists(args.output):
        sys.exit(f"[fix-signals] refusing to overwrite {args.output}")
    shutil.copy2(args.input, args.output)
    fixed = 0
    with h5py.File(args.output, "a") as f:
        for name in sorted(f["data"].keys(), key=lambda n: int(n.split("_")[1])):
            g = f["data"][name]
            sig_grp = g["obs"]["datagen_info"]["subtask_term_signals"]
            sig = {k: np.asarray(sig_grp[k]).reshape(-1) for k in sig_grp}
            gate = first_rising(sig[GATE])
            if gate is None:
                sys.exit(f"[fix-signals] {name}: {GATE} never rises — the source is not a valid demo")
            tgt = sig[TARGET].copy()
            before = int(tgt[:gate].sum())
            if before:
                tgt[:gate] = 0
                ds = sig_grp[TARGET]
                ds[...] = tgt.reshape(ds.shape).astype(ds.dtype)
                sig[TARGET] = tgt
                fixed += 1
            bounds = mimic_boundaries(sig, g["actions"].shape[0])
            print(f"[fix-signals] {name}: T={g['actions'].shape[0]} {GATE} rises at {gate}, "
                  f"{TARGET} zeroed on {before} early frames; boundaries {bounds}", flush=True)
    print(f"FIX_SIGNALS_DONE: {fixed} episode(s) patched -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
