"""B-oracle replay (P3 WP 3.5): the encoder-labelled token stream of each recorded episode
through the SONIC decoder — no VLA in the loop — on the episode's recorded cube spawn, through
`evaluation.eval`, same env family, same rubric, same seeds, same `eval_results.csv`.

    PYTHONUSERBASE=~/env/pyuser-fr3 /isaac-sim/python.sh harness/lane_b/eval_oracle_b.py \
        --demos <export dir> --tokens <label_run>/out/tokens \
        --decoder-onnx <export>/out/model_decoder.onnx --encoder-onnx <export>/out/model_encoder.onnx \
        --run-folder <run>/out/eval --rate 50 --rollouts 20 --max-steps 1500 --no-splat --headless

Reuses lane A's oracle plumbing from harness/lane_a (never edited): the JointPos env variant
`Isaac-Stack-Cube-DualFranka-JointPos-OracleA-v0` whose reset places the block at the replayed
episode's recorded pose (`oracle_a_env.py`, keyed through `oracle_a_table`), so episode k of
this oracle, of the A-oracle and of the dataset are the same recording. The in-process client
`OracleB` runs harness/lane_b/sonic_decoder.py (the same runtime the lane-B policy server
uses) on episode k's token labels at 50 Hz and returns `[Lq7, Lg±1, Rq7, Rg±1]`; the grippers
are the dataset's own grip labels (identical to lane A's). After the last labelled frame the
last token is held.

This is lane B's ceiling: a low number means the controller (token space + decoder) lost, not
the VLA (plan/PLAN.md, comparability protocol).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LANE_A = os.path.join(os.path.dirname(HERE), "lane_a")
for p in (HERE, LANE_A):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_oracle_a  # noqa: E402  (registers nothing at import; provides the task registration)
import oracle_a_table  # noqa: E402
import sonic_decoder  # noqa: E402

ORACLE_CLIENT = "OracleB"
ORACLE_TASK_ID = eval_oracle_a.ORACLE_TASK_ID


def load_episodes(tokens_dir: str, demos_dir: str) -> list[dict]:
    import h5py

    files = sorted(glob.glob(os.path.join(tokens_dir, "episode_*.npz")))
    if not files:
        raise SystemExit(f"[oracle-b] no episode_*.npz under {tokens_dir}")
    episodes = []
    handles: dict[str, h5py.File] = {}
    try:
        for path in files:
            z = np.load(path)
            src = str(z["source_file"])
            demo = str(z["demo_name"])
            # the labels carry the absolute source path; fall back to the same file name under --demos
            cand = src if os.path.exists(src) else os.path.join(demos_dir, os.path.basename(src))
            if cand not in handles:
                handles[cand] = h5py.File(cand, "r")
            g = handles[cand]["data"][demo]
            if "initial_cube_pose" not in g:
                raise SystemExit(f"[oracle-b] {cand}:{demo} lacks initial_cube_pose")
            episodes.append({
                "name": f"{os.path.basename(cand)}:{demo}",
                "episode_index": int(z["episode_index"]),
                "clip": str(z["clip"]),
                "token": np.asarray(z["token"], dtype=np.float32),
                "grip": np.asarray(z["grip"], dtype=np.float32),
                "cube_pose": np.asarray(g["initial_cube_pose"], dtype=np.float32).reshape(7),
            })
    finally:
        for h in handles.values():
            h.close()
    episodes.sort(key=lambda e: e["episode_index"])
    print(f"[oracle-b] {len(episodes)} labelled episodes from {tokens_dir}", flush=True)
    return episodes


def register_oracle_client(decoder_onnx: str, hold_token: np.ndarray, clip_targets: bool) -> None:
    from evaluation.client import InferenceClient, register

    @register(ORACLE_CLIENT)
    class OracleBClient(InferenceClient):
        """Streams episode k's token labels through the SONIC decoder at the control rate."""

        def __init__(self, args, *, fail_closed: bool = False):
            super().__init__(args, fail_closed=fail_closed)
            self._resets = 0
            self._ep = None
            self._t = 0
            self._decoder = sonic_decoder.SonicDecoderRuntime(
                decoder_onnx, hold_token=hold_token, clip_targets=clip_targets)

        def _dump_trace(self) -> None:
            # P5 diagnostic: per-step closed-loop trace (state seen, wire targets sent) per episode,
            # written when ORACLE_B_TRACE_DIR is set; off by default (the P3 protocol is unchanged).
            tdir = os.environ.get("ORACLE_B_TRACE_DIR")
            if tdir and self._ep is not None and getattr(self, "_trace", None):
                os.makedirs(tdir, exist_ok=True)
                np.savez(os.path.join(tdir, f"trace_reset{self._resets - 1}.npz"),
                         state16=np.asarray([t[0] for t in self._trace], np.float32),
                         wire=np.asarray([t[1] for t in self._trace], np.float32),
                         token_index=np.asarray([t[2] for t in self._trace], np.int32),
                         name=self._ep["name"], clip=self._ep["clip"])
            self._trace = []

        def reset(self) -> None:
            self._dump_trace()
            idx = min(oracle_a_table.START_EPISODE + self._resets, len(oracle_a_table.EPISODES) - 1)
            self._resets += 1
            self._ep = oracle_a_table.EPISODES[idx]
            self._t = 0
            self._decoder.reset()
            print(f"[oracle-b] client reset #{self._resets - 1}: replaying table episode {idx} "
                  f"({self._ep['name']}, clip {self._ep['clip']}, {len(self._ep['token'])} tokens)", flush=True)

        def infer(self, obs: dict, instruction: str = "") -> np.ndarray:
            tokens, grips = self._ep["token"], self._ep["grip"]
            i = min(self._t, len(tokens) - 1)
            self._t += 1
            state16 = np.asarray(obs["state16"], dtype=np.float32)
            raw = self._decoder.step(tokens[i], state16)
            wire = self._decoder.targets_wire(raw, grips[i])
            if os.environ.get("ORACLE_B_TRACE_DIR"):
                self._trace.append((state16.copy(), wire.copy(), i))
                if len(self._trace) % 1500 == 0:
                    self._dump_trace()  # the last episode gets no reset; flush at the horizon
            g_l = -1.0 if grips[i, 0] > 0.5 else 1.0  # JointPos env: +1 open, -1 closed
            g_r = -1.0 if grips[i, 1] > 0.5 else 1.0
            return np.concatenate([wire[0:7], [g_l], wire[8:15], [g_r]]).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--demos", required=True, help="export dir with the demo HDF5 shards (cube spawns)")
    ap.add_argument("--tokens", required=True, help="label_tokens run's out/tokens dir")
    ap.add_argument("--decoder-onnx", required=True)
    ap.add_argument("--encoder-onnx", default=None)
    ap.add_argument("--hold-token-json", default=None, help="default: <tokens>/hold_token.json")
    ap.add_argument("--no-clip-targets", action="store_true")
    ap.add_argument("--run-folder", required=True)
    ap.add_argument("--start-episode", type=int, default=None,
                    help="table offset (default: the run folder's completed episode count)")
    known, rest = ap.parse_known_args()

    hold_json = known.hold_token_json or os.path.join(known.tokens, "hold_token.json")
    hold = sonic_decoder.load_hold_token(hold_json if os.path.exists(hold_json) else None, known.encoder_onnx)
    oracle_a_table.EPISODES = load_episodes(known.tokens, known.demos)
    oracle_a_table.START_EPISODE = (
        known.start_episode if known.start_episode is not None
        else eval_oracle_a.completed_episodes(known.run_folder)
    )
    oracle_a_table.ENV_RESETS = 0
    os.makedirs(known.run_folder, exist_ok=True)
    with open(os.path.join(known.run_folder, "oracle_b.json"), "w") as fh:
        json.dump({
            "tokens": known.tokens, "demos": known.demos, "decoder_onnx": known.decoder_onnx,
            "encoder_onnx": known.encoder_onnx, "hold_token_json": hold_json,
            "hold_token_abs_max": float(np.abs(hold).max()),
            "episodes": [e["name"] for e in oracle_a_table.EPISODES],
            "clips": [e["clip"] for e in oracle_a_table.EPISODES],
            "start_episode": oracle_a_table.START_EPISODE,
            "task": ORACLE_TASK_ID, "client": ORACLE_CLIENT,
            "clip_targets_to_fr3_limits": not known.no_clip_targets,
            "action_layout": "[L fr3_joint1..7 (decoder targets), L grip ±1, R fr3_joint1..7, R grip ±1]",
        }, fh, indent=2)

    eval_oracle_a.register_oracle_task()
    register_oracle_client(known.decoder_onnx, hold, clip_targets=not known.no_clip_targets)

    sys.argv = [sys.argv[0], "--run-folder", known.run_folder, "--client", ORACLE_CLIENT,
                "--task", ORACLE_TASK_ID] + rest
    from evaluation.config import RunRecordMismatch, parse_eval_args
    from evaluation.eval import main as eval_main

    eval_args, policy_args = parse_eval_args("B-oracle replay through evaluation.eval")
    if eval_args.rollouts > len(oracle_a_table.EPISODES):
        print(f"[oracle-b] WARNING: --rollouts {eval_args.rollouts} > {len(oracle_a_table.EPISODES)} "
              "labelled episodes; the last episode would repeat", flush=True)
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
