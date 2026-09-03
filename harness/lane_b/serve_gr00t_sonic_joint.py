#!/usr/bin/env python
"""Lane B policy server: GR00T N1.7 -> 40-step SONIC token chunk -> decoder ONNX (inside this
server, 50 Hz) -> dual-FR3 joint targets, over the stock ZmqAct wire.

    ~/Isaac-GR00T/.venv/bin/python harness/lane_b/serve_gr00t_sonic_joint.py \\
        --model-path <run>/out/checkpoints/checkpoint-2000 --port 8000 \\
        --decoder-onnx <export>/out/model_decoder.onnx --encoder-onnx <export>/out/model_encoder.onnx \\
        [--hold-token-json <label_run>/out/tokens/hold_token.json]

Identical wire contract to harness/lane_a/serve_gr00t_joint.py (the sim side is the same
client, binding, rubric and rate for both lanes — plan/PLAN.md comparability protocol):

    {"type": "reset"}                                        -> {"ok": True, "episode": n}
    {"type": "act", "state": f32[16] [Lq7, Lg, Rq7, Rg], "top"/"wrist_left"/"wrist_right": uint8}
                                                             -> {"action": f32[16] [Lq7, Lg01, Rq7, Rg01]}

What differs is inside:

  * GR00T's action keys are motion_token (64) | left_grip (1) | right_grip (1)
    (harness/lane_b/modality_config_dual_fr3_sonic.py), predicted as a 40-step chunk
    (0.8 s at 50 Hz) and replanned every --replan-every rows (20 -> 2.5 Hz).
  * Every 50 Hz request runs ONE decoder step (harness/lane_b/sonic_decoder.py): the row's
    token (clipped to |t| <= 1.25) + the decoder's own proprio history built from the streamed
    state -> raw action -> joint targets = default + 0.25 a -> FR3 angles (joint 6 + 2.5307),
    clipped to the FR3 joint limits.
  * Reset clears the chunk AND the decoder history. A "hold" token (the encoder's output for
    a static reference at the ready pose — never a zero token) is fed whenever no chunk row is
    available (--warmup-steps decoder steps after a reset, non-finite GR00T output).
  * Grippers pass straight through from GR00T's grip columns ([0,1], 1 = closed).

Images: same 640x360 INTER_AREA resize as lane A (--image-size).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lane_a"))
import serve_gr00t_joint as base  # noqa: E402  (lane A's server: wire, images, REP loop)
import dual_fr3_orders as O  # noqa: E402
import sonic_decoder  # noqa: E402

DEFAULT_MODALITY_CONFIG = HERE / "modality_config_dual_fr3_sonic.py"
ACTION_KEYS = ("motion_token", "left_grip", "right_grip")
CHUNK_DIM = O.TOKEN_DIM + 2  # 66


def chunk_from_action(action: dict) -> np.ndarray:
    """(1,40,64)+(1,40,1)+(1,40,1) -> (40,66) = [token | left_grip | right_grip]."""
    missing = [k for k in ACTION_KEYS if k not in action]
    if missing:
        raise KeyError(f"policy returned no {missing}; got keys {sorted(action)}")
    parts = [np.asarray(action[k], dtype=np.float32)[0] for k in ACTION_KEYS]
    chunk = np.concatenate(parts, axis=-1).astype(np.float32)
    if chunk.shape != (base.ACTION_HORIZON, CHUNK_DIM):
        raise ValueError(f"action chunk has shape {chunk.shape}, expected (40, {CHUNK_DIM})")
    if not np.isfinite(chunk).all():
        raise ValueError("action chunk contains non-finite values")
    return chunk


class SonicPolicyServer(base.JointPolicyServer):
    """REP socket + 40-step token chunk + the SONIC decoder in the loop."""

    def __init__(self, args: argparse.Namespace):
        self.decoder = None
        self.hold_token = None
        if args.decoder_onnx:
            self.hold_token = sonic_decoder.load_hold_token(args.hold_token_json, args.encoder_onnx)
            self.decoder = sonic_decoder.SonicDecoderRuntime(
                args.decoder_onnx, hold_token=self.hold_token, clip_targets=not args.no_clip_targets)
            print(f"[serve] decoder {args.decoder_onnx} loaded; hold token |max| "
                  f"{float(np.abs(self.hold_token).max()):.4f} first8 {np.round(self.hold_token[:8], 4).tolist()}",
                  flush=True)
        elif not args.dry_run:
            raise ValueError("--decoder-onnx is required unless --dry-run is set")
        self.warmup_steps = int(args.warmup_steps)
        self.token_stats = {"rows": 0, "abs_max": 0.0, "clipped_rows": 0}
        super().__init__(args)

    # ------------------------------------------------------------- inference
    def _predict(self, observation: dict, state: np.ndarray) -> np.ndarray:
        if self.policy is None:
            # --dry-run: hold token for the whole horizon, grippers open
            tok = self.hold_token if self.hold_token is not None else np.zeros(O.TOKEN_DIM, np.float32)
            row = np.concatenate([tok, np.zeros(2, np.float32)])
            return np.repeat(row[None, :], base.ACTION_HORIZON, axis=0).astype(np.float32)
        action, _info = self.policy.get_action(observation)
        return chunk_from_action(action)

    def _next_row(self, request: dict) -> np.ndarray:
        observation, state = base.build_observation(request, self.instruction, self.image_scale, self.image_size)
        state16 = np.asarray(request["state"], dtype=np.float32).reshape(16)
        if self.chunk is None or self.step_in_chunk >= self.replan_every:
            started = time.perf_counter()
            self.chunk = self._predict(observation, state)
            latency_ms = (time.perf_counter() - started) * 1e3
            self.step_in_chunk = 0
            self.replans += 1
            tok_abs = float(np.abs(self.chunk[:, :O.TOKEN_DIM]).max())
            print(f"[serve] REPLAN #{self.replans} at request {self.requests} latency_ms={latency_ms:.1f} "
                  f"|token|max={tok_abs:.3f} grips={np.round(self.chunk[0, O.TOKEN_DIM:], 3).tolist()}",
                  flush=True)
        row = self.chunk[self.step_in_chunk]
        self.step_in_chunk += 1
        token, grips = row[:O.TOKEN_DIM], row[O.TOKEN_DIM:CHUNK_DIM]
        amax = float(np.abs(token).max())
        self.token_stats["rows"] += 1
        self.token_stats["abs_max"] = max(self.token_stats["abs_max"], amax)
        if amax > O.TOKEN_BOUND:
            self.token_stats["clipped_rows"] += 1
        grips = np.clip(grips, 0.0, 1.0)
        if self.decoder is None:
            # dry run without a decoder: hold the current state (lane A's wire test)
            return state16.copy()
        if self.decoder.steps == 0 and self.warmup_steps > 0:
            for _ in range(self.warmup_steps):
                self.decoder.step(None, state16)
        raw = self.decoder.step(token, state16)
        return self.decoder.targets_wire(raw, grips)

    # ---------------------------------------------------------------- handlers
    def handle(self, request: dict) -> dict:
        if isinstance(request, dict) and request.get("type") == "reset":
            if self.decoder is not None:
                self.decoder.reset()
            print(f"[serve] token stats so far: {json.dumps(self.token_stats)}", flush=True)
        return super().handle(request)


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = __doc__
    parser.set_defaults(modality_config_path=str(DEFAULT_MODALITY_CONFIG))
    parser.add_argument("--decoder-onnx", default=None, help="P2 model_decoder.onnx (required unless --dry-run)")
    parser.add_argument("--encoder-onnx", default=None, help="P2 model_encoder.onnx (self-encodes the hold token)")
    parser.add_argument("--hold-token-json", default=None,
                        help="labeller's hold_token.json; wins over --encoder-onnx")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="decoder steps with the hold token after a reset before the first chunk row "
                             "(0 = the SONIC env's own start: zero action history)")
    parser.add_argument("--no-clip-targets", action="store_true",
                        help="do not clip the decoder's joint targets to the FR3 joint limits")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dry_run and not args.model_path:
        print("ERROR: --model-path is required unless --dry-run is set", flush=True)
        return 2
    base._install_signal_handlers()
    try:
        return SonicPolicyServer(args).serve()
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"SERVER_FAILED: {type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
