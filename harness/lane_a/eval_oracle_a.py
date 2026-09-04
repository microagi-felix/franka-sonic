"""A-oracle replay (P1 WP 1.7, and the WP 1.0 one-episode replay check): the recorded joint
targets, on the recorded cube spawns, through `evaluation.eval` — same env family, same rubric,
same `eval_results.csv`.

    PYTHONUSERBASE=~/env/pyuser-fr3 /isaac-sim/python.sh harness/lane_a/eval_oracle_a.py \
        --demos <dir-or-glob of export HDF5> --run-folder <run>/out/eval \
        --rate 50 --rollouts 20 --max-steps 1500 --no-splat --headless [any evaluation.eval flag]

What it adds on top of the stock evaluator, all from THIS repo (the franka repo is upstream and
is never edited):

- a `TaskContract` + gym id `Isaac-Stack-Cube-DualFranka-JointPos-OracleA-v0` whose only
  difference to the JointPos env is a table-driven cube spawn (`oracle_a_env.py`);
- an in-process client `OracleA` that streams episode k's recorded actions
  `[Lq7, Lg±1, Rq7, Rg±1]` (differential-IK joint targets + the recorded binary gripper) one row
  per control step and holds the last row afterwards;
- `--task` and `--client` are forced; everything else is passed to `evaluation.eval` verbatim,
  so resume, videos, seeds and the run record behave exactly as for the policy run.

Expected ≈ 100 % by construction; a low number means the dataset or the replay path is wrong
and lane A's policy number would be meaningless (PLAN.md, comparability protocol).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import oracle_a_table  # noqa: E402  (Isaac-free)

ORACLE_TASK_ID = "Isaac-Stack-Cube-DualFranka-JointPos-OracleA-v0"
ORACLE_CLIENT = "OracleA"


sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data"))
import joint_labels  # noqa: E402  (harness/data/joint_labels.py — shared with the converter)

JOINT_LABELS = joint_labels.JOINT_LABELS


def arm_labels(g, side: str, joint_label: str, max_delta_rad: float) -> np.ndarray:
    """Lane A's absolute joint label for one arm, (T, 7) — the same function the converter uses."""
    return joint_labels.arm_label(
        np.asarray(g["obs"][f"joint_pos_{side}"], dtype=np.float32),
        np.asarray(g["obs"][f"joint_target_{side}"], dtype=np.float32),
        joint_label, max_delta_rad,
    )


def load_episodes(spec: str, only_successful: bool = True,
                  joint_label: str = joint_labels.DEFAULT_JOINT_LABEL,
                  max_delta_rad: float = joint_labels.DEFAULT_MAX_DELTA_RAD,
                  left_j1_offset_rad: float = 0.0) -> list[dict]:
    """left_j1_offset_rad (P5 tolerance test, default 0 = the P1 protocol): a constant added to
    the LEFT arm's joint-1 target on every row, i.e. a lateral shift of the left hand."""
    import h5py

    if joint_label not in JOINT_LABELS:
        raise SystemExit(f"[oracle-a] --joint-label must be one of {JOINT_LABELS}")
    if os.path.isdir(spec):
        files = sorted(glob.glob(os.path.join(spec, "*.hdf5")))
    else:
        files = sorted(glob.glob(spec))
    if not files:
        raise SystemExit(f"[oracle-a] no HDF5 under {spec}")
    episodes = []
    skipped = 0
    for path in files:
        with h5py.File(path, "r") as f:
            demos = sorted(f["data"].keys(), key=lambda n: int(n.split("_")[1]))
            for name in demos:
                g = f["data"][name]
                # `jointpos_replay_success` (harness/data/jointpos_screen.py, P7) is absent on
                # every export written before the screen existed, and absent means True — so a
                # round-1 export loads exactly as it always did.
                if only_successful and not (bool(g.attrs.get("replay_success", True))
                                            and bool(g.attrs.get("jointpos_replay_success", True))):
                    skipped += 1
                    continue
                if "initial_cube_pose" not in g or "joint_target_left" not in g["obs"]:
                    raise SystemExit(f"[oracle-a] {path}:{name} lacks initial_cube_pose/joint_target_* "
                                     "(export with harness/data/export_generated_50hz.py)")
                acts = np.asarray(g["actions"], dtype=np.float32)  # IK-Abs rows; grips at 7 and 15
                tl = arm_labels(g, "left", joint_label, max_delta_rad)
                tr = arm_labels(g, "right", joint_label, max_delta_rad)
                action16 = np.concatenate(
                    [tl, np.sign(acts[:, 7:8]), tr, np.sign(acts[:, 15:16])], axis=1
                ).astype(np.float32)
                action16[:, 7] = np.where(action16[:, 7] == 0, 1.0, action16[:, 7])
                action16[:, 15] = np.where(action16[:, 15] == 0, 1.0, action16[:, 15])
                if left_j1_offset_rad:
                    action16[:, 0] += np.float32(left_j1_offset_rad)
                episodes.append({
                    "name": f"{os.path.basename(path)}:{name}",
                    "action16": action16,
                    "cube_pose": np.asarray(g["initial_cube_pose"], dtype=np.float32).reshape(7),
                })
    print(f"[oracle-a] {len(episodes)} replayable episodes from {len(files)} file(s) "
          f"({skipped} skipped as replay_success=False)", flush=True)
    return episodes


def completed_episodes(run_folder: str) -> int:
    """Same rule as evaluation.eval.reconcile_episode_ledger: leading run of episode_N_result.json."""
    n = 0
    while os.path.exists(os.path.join(run_folder, f"episode_{n}_result.json")):
        n += 1
    return n


def register_oracle_task() -> None:
    """Declare the contract (Isaac-free) and the gym id; the env cfg module loads at gym.make."""
    import gymnasium as gym
    from frankas_assets.specs.task import HANDOVER_JOINT_POS, TASKS, TaskContract

    base = TASKS[HANDOVER_JOINT_POS]
    TASKS[ORACLE_TASK_ID] = TaskContract(
        gym_id=ORACLE_TASK_ID,
        env_cfg_entry_point="oracle_a_env:OracleAJointPosEnvCfg",
        embodiment=base.embodiment,
        placements=base.placements,
        raytraced_prims=base.raytraced_prims,
        required_scene_frames=base.required_scene_frames,
        required_rig_layout=base.required_rig_layout,
        rubric=base.rubric,
        scorer_contract_version=base.scorer_contract_version,
        env_entry_point=base.env_entry_point,
    )
    gym.register(
        id=ORACLE_TASK_ID,
        entry_point=base.env_entry_point,
        kwargs={"env_cfg_entry_point": "oracle_a_env:OracleAJointPosEnvCfg"},
        disable_env_checker=True,
    )


def register_oracle_client() -> None:
    from evaluation.client import InferenceClient, register

    @register(ORACLE_CLIENT)
    class OracleAClient(InferenceClient):
        """Streams the recorded JointPos-env actions of the episode being replayed."""

        def __init__(self, args, *, fail_closed: bool = False):
            super().__init__(args, fail_closed=fail_closed)
            self._resets = 0
            self._ep = None
            self._t = 0

        def reset(self) -> None:
            idx = min(oracle_a_table.START_EPISODE + self._resets, len(oracle_a_table.EPISODES) - 1)
            self._resets += 1
            self._ep = oracle_a_table.EPISODES[idx]
            self._t = 0
            print(f"[oracle-a] client reset #{self._resets - 1}: replaying table episode {idx} "
                  f"({self._ep['name']}, {len(self._ep['action16'])} rows)", flush=True)

        def infer(self, obs: dict, instruction: str = "") -> np.ndarray:
            rows = self._ep["action16"]
            row = rows[min(self._t, len(rows) - 1)]
            self._t += 1
            return row.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--demos", required=True, help="dir or glob of export HDF5 (video-backed schema)")
    ap.add_argument("--run-folder", required=True)
    ap.add_argument("--include-failed-replays", action="store_true")
    ap.add_argument("--joint-label", choices=JOINT_LABELS, default=joint_labels.DEFAULT_JOINT_LABEL,
                    help="arm label to replay (harness/data/joint_labels.py); must match the dataset")
    ap.add_argument("--max-delta-rad", type=float, default=joint_labels.DEFAULT_MAX_DELTA_RAD)
    ap.add_argument("--start-episode", type=int, default=None,
                    help="table offset (default: the run folder's completed episode count)")
    ap.add_argument("--episode-indices", default=None,
                    help="P7 WP 7.4: comma-separated indices into the loaded episode table; the "
                         "replay table is restricted to exactly these, in this order (default: all)")
    ap.add_argument("--left-j1-offset-rad", type=float, default=0.0,
                    help="P5 tolerance test: constant offset added to the left arm's joint-1 target "
                         "(default 0 = the P1 protocol, unchanged)")
    known, rest = ap.parse_known_args()

    oracle_a_table.EPISODES = load_episodes(known.demos, only_successful=not known.include_failed_replays,
                                            joint_label=known.joint_label, max_delta_rad=known.max_delta_rad,
                                            left_j1_offset_rad=known.left_j1_offset_rad)
    if known.episode_indices:
        picked = [int(v) for v in known.episode_indices.split(",") if v.strip() != ""]
        n_loaded = len(oracle_a_table.EPISODES)
        bad = [i for i in picked if not 0 <= i < n_loaded]
        if bad:
            sys.exit(f"[oracle-a] --episode-indices {bad} out of range (0..{n_loaded - 1})")
        oracle_a_table.EPISODES = [oracle_a_table.EPISODES[i] for i in picked]
        print(f"[oracle-a] episode subset {picked} of {n_loaded}: "
              f"{[e['name'] for e in oracle_a_table.EPISODES]}", flush=True)
    if known.left_j1_offset_rad:
        print(f"[oracle-a] TOLERANCE TEST: left joint-1 targets offset by {known.left_j1_offset_rad:+.3f} rad",
              flush=True)
    print(f"[oracle-a] arm label: {joint_labels.describe(known.joint_label, known.max_delta_rad)}", flush=True)
    oracle_a_table.START_EPISODE = (
        known.start_episode if known.start_episode is not None else completed_episodes(known.run_folder)
    )
    oracle_a_table.ENV_RESETS = 0
    os.makedirs(known.run_folder, exist_ok=True)
    with open(os.path.join(known.run_folder, "oracle_a.json"), "w") as fh:
        json.dump({
            "demos": known.demos,
            "episodes": [e["name"] for e in oracle_a_table.EPISODES],
            "start_episode": oracle_a_table.START_EPISODE,
            "task": ORACLE_TASK_ID,
            "client": ORACLE_CLIENT,
            "joint_label": known.joint_label,
            "left_j1_offset_rad": known.left_j1_offset_rad,
            "action_layout": f"[L fr3_joint1..7 ({known.joint_label}), L grip ±1, R fr3_joint1..7, R grip ±1]",
        }, fh, indent=2)

    register_oracle_task()
    register_oracle_client()

    sys.argv = [sys.argv[0], "--run-folder", known.run_folder, "--client", ORACLE_CLIENT,
                "--task", ORACLE_TASK_ID] + rest
    from evaluation.config import RunRecordMismatch, parse_eval_args
    from evaluation.eval import main as eval_main

    eval_args, policy_args = parse_eval_args("A-oracle replay through evaluation.eval")
    if eval_args.rollouts > len(oracle_a_table.EPISODES):
        print(f"[oracle-a] WARNING: --rollouts {eval_args.rollouts} > {len(oracle_a_table.EPISODES)} "
              "replayable episodes; the last episode would repeat", flush=True)
    try:
        return eval_main(eval_args, policy_args)
    except RunRecordMismatch as mismatch:
        print(f"[eval] REFUSED: {mismatch}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    _code = 1
    try:
        _code = main()
    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(_code)  # dodge Isaac's headless close() hang, like evaluation.eval does
