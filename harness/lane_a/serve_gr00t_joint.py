#!/usr/bin/env python
"""Lane A policy server: GR00T N1.7 -> dual-FR3 joint targets, over the stock
ZmqAct wire.

Run it with the GR00T interpreter:

    ~/Isaac-GR00T/.venv/bin/python harness/lane_a/serve_gr00t_joint.py \\
        --model-path <run>/out/checkpoints/checkpoint-2000 --port 5602

WIRE CONTRACT (evaluation/client.py::ZmqActClient of
franka-bimanual-isaac-sim -- upstream, never patched by us)
-----------------------------------------------------------------------------
zmq REQ/REP, pickled objects (send_pyobj / recv_pyobj), one round trip per
control step at 50 Hz. The client never chunks; replanning lives here.

    {"type": "reset"}                                   -> any dict (we send
                                                           {"ok": True, "episode": n})
    {"type": "act",
     "state": float32[16],                              [Lq7, Lg, Rq7, Rg]
     "top":         uint8 (720, 1280, 3),               native sim resolution
     "wrist_left":  uint8 (480, 848, 3),
     "wrist_right": uint8 (480, 848, 3)}
                                                        -> {"action": float32[16]}
                                                           [Lq7, Lg01, Rq7, Rg01]

Grippers on the wire are in [0,1] with 1 = closed; the client thresholds them
into its own +-1 convention. A reply that contains "error" makes the client
raise, so every exception is caught, reported and the server keeps serving.

ORDER CONVERSION (the one thing that is easy to get wrong)
-----------------------------------------------------------------------------
    wire     [Lq7 | Lg | Rq7 | Rg]      indices 0:7, 7, 8:15, 15
    dataset  [Lq7 | Rq7 | Lg | Rg]      indices 0:7, 7:14, 14, 15

The dataset order is what harness/data/convert_hdf5_to_gr00t_v2.py wrote and
what harness/lane_a/modality_config_dual_fr3.py slices into joint_pos_l /
joint_pos_r / grip. This server converts in both directions.

TIMING (plan/PLAN.md rev 3c): GR00T predicts a 40-step chunk (0.8 s at 50 Hz);
we play it out one row per request and replan every --replan-every rows
(20 -> 2.5 Hz).

IMAGES: the training frames are the sim's native frames scaled by 0.5 with
cv2.INTER_AREA (640x360 top, 424x240 wrists). --image-scale applies the same
transform to every live frame; use 1.0 only if the dataset was not downscaled.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np


DEFAULT_MODALITY_CONFIG = (
    Path(__file__).resolve().parent / "modality_config_dual_fr3.py"
)
DEFAULT_INSTRUCTION = "hand the block from the left arm to the right"
CAMERAS = ("top", "wrist_left", "wrist_right")
ACTION_HORIZON = 40
VECTOR_DIM = 16
ARM_DOF = 7

# module-level stop flag, flipped by SIGTERM/SIGINT so the recv loop can exit
_STOP = False


def _install_signal_handlers() -> None:
    def handler(signum, _frame):
        global _STOP
        _STOP = True
        print(f"[serve] signal {signum} received, shutting down", flush=True)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


# ----------------------------------------------------------------- conversions


def wire_to_dataset(state: np.ndarray) -> np.ndarray:
    """[Lq7, Lg, Rq7, Rg] -> [Lq7, Rq7, Lg, Rg]."""
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape != (VECTOR_DIM,):
        raise ValueError(f"state must have shape (16,), got {state.shape}")
    return np.concatenate(
        [state[0:ARM_DOF], state[ARM_DOF + 1 : 2 * ARM_DOF + 1],
         state[ARM_DOF : ARM_DOF + 1], state[2 * ARM_DOF + 1 : VECTOR_DIM]]
    ).astype(np.float32)


def dataset_to_wire(row: np.ndarray) -> np.ndarray:
    """[Lq7, Rq7, Lg, Rg] -> [Lq7, Lg, Rq7, Rg]."""
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    if row.shape != (VECTOR_DIM,):
        raise ValueError(f"action row must have shape (16,), got {row.shape}")
    return np.concatenate([row[0:7], row[14:15], row[7:14], row[15:16]]).astype(np.float32)


def scale_image(image: np.ndarray, scale: float) -> np.ndarray:
    """Downscale a live frame the same way the dataset frames were downscaled."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected an (H,W,3) frame, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if scale == 1.0:
        return np.ascontiguousarray(image)
    import cv2

    height, width = image.shape[:2]
    target = (int(round(width * scale)), int(round(height * scale)))
    return np.ascontiguousarray(
        cv2.resize(image, target, interpolation=cv2.INTER_AREA)
    )


def build_observation(
    request: dict, instruction: str, image_scale: float
) -> tuple[dict, np.ndarray]:
    """Turn one 'act' request into a GR00T observation dict.

    Returns (observation, dataset_order_state) -- the state is handed back so
    --dry-run can echo it without rebuilding.
    """
    state = wire_to_dataset(request["state"])
    video = {}
    for camera in CAMERAS:
        if camera not in request:
            raise KeyError(f"act request is missing camera '{camera}'")
        frame = scale_image(request[camera], image_scale)
        video[camera] = frame[None, None, ...]  # (B=1, T=1, H, W, 3) uint8
    observation = {
        "video": video,
        "state": {
            "joint_pos_l": state[None, None, 0:7].astype(np.float32),
            "joint_pos_r": state[None, None, 7:14].astype(np.float32),
            "grip": state[None, None, 14:16].astype(np.float32),
        },
        "language": {"annotation.human.task_description": [[instruction]]},
    }
    return observation, state


def chunk_from_action(action: dict) -> np.ndarray:
    """(1,40,7)+(1,40,7)+(1,40,2) -> (40,16) in dataset order."""
    missing = [k for k in ("joint_pos_l", "joint_pos_r", "grip") if k not in action]
    if missing:
        raise KeyError(f"policy returned no {missing}; got keys {sorted(action)}")
    chunk = np.concatenate(
        [
            np.asarray(action["joint_pos_l"])[0],
            np.asarray(action["joint_pos_r"])[0],
            np.asarray(action["grip"])[0],
        ],
        axis=-1,
    ).astype(np.float32)
    if chunk.shape != (ACTION_HORIZON, VECTOR_DIM):
        raise ValueError(f"action chunk has shape {chunk.shape}, expected (40, 16)")
    if not np.isfinite(chunk).all():
        raise ValueError("action chunk contains non-finite values")
    return chunk


# --------------------------------------------------------------------- server


class JointPolicyServer:
    """REP socket + a 40-step chunk buffer around a Gr00tPolicy."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.instruction = args.instruction
        self.replan_every = int(args.replan_every)
        if self.replan_every < 1:
            raise ValueError("--replan-every must be >= 1")
        self.image_scale = float(args.image_scale)
        self.policy = None if args.dry_run else self._load_policy()
        self.chunk: np.ndarray | None = None
        self.step_in_chunk = 0
        self.episode = 0
        self.requests = 0
        self.replans = 0

    def _load_policy(self):
        config_path = Path(self.args.modality_config_path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"--modality-config-path not found: {config_path}")
        # Same as gr00t/experiment/launch_finetune.py::load_modality_config --
        # the checkpoint carries its own modality config, but the registry must
        # still know the tag before the policy is built.
        sys.path.append(str(config_path.parent))
        importlib.import_module(config_path.stem)
        print(f"[serve] loaded modality config {config_path}", flush=True)

        from gr00t.policy.gr00t_policy import Gr00tPolicy

        started = time.perf_counter()
        policy = Gr00tPolicy(
            embodiment_tag=self.args.embodiment_tag,
            model_path=str(Path(self.args.model_path).expanduser()),
            device=self.args.device,
        )
        print(
            f"[serve] policy ready in {time.perf_counter() - started:.1f}s "
            f"({self.args.model_path} on {self.args.device})",
            flush=True,
        )
        return policy

    # ------------------------------------------------------------- inference

    def _predict(self, observation: dict, state: np.ndarray) -> np.ndarray:
        if self.policy is None:
            # --dry-run: hold the current state for the whole horizon. Played
            # back through dataset_to_wire this returns the request's own state,
            # which is what the wire test asserts.
            return np.repeat(state[None, :], ACTION_HORIZON, axis=0).astype(np.float32)
        action, _info = self.policy.get_action(observation)
        return chunk_from_action(action)

    def _next_row(self, request: dict) -> np.ndarray:
        observation, state = build_observation(request, self.instruction, self.image_scale)
        if self.chunk is None or self.step_in_chunk >= self.replan_every:
            started = time.perf_counter()
            self.chunk = self._predict(observation, state)
            latency_ms = (time.perf_counter() - started) * 1e3
            self.step_in_chunk = 0
            self.replans += 1
            print(
                f"[serve] REPLAN #{self.replans} at request {self.requests} "
                f"latency_ms={latency_ms:.1f}",
                flush=True,
            )
        row = self.chunk[self.step_in_chunk]
        self.step_in_chunk += 1
        return dataset_to_wire(row)

    # ---------------------------------------------------------------- handlers

    def handle(self, request: dict) -> dict:
        if not isinstance(request, dict):
            raise TypeError(f"expected a dict request, got {type(request).__name__}")
        kind = request.get("type")
        if kind == "reset":
            self.chunk = None
            self.step_in_chunk = 0
            self.episode += 1
            print(f"[serve] reset -> episode {self.episode}", flush=True)
            return {"ok": True, "episode": self.episode}
        if kind == "act":
            self.requests += 1
            action = self._next_row(request)
            if self.args.log_every and self.requests % self.args.log_every == 0:
                print(
                    f"[serve] heartbeat requests={self.requests} replans={self.replans} "
                    f"episode={self.episode} step_in_chunk={self.step_in_chunk}",
                    flush=True,
                )
            return {"action": action}
        raise ValueError(f"unknown request type {kind!r}")

    def serve(self) -> int:
        import zmq

        context = zmq.Context.instance()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.RCVTIMEO, 500)  # so SIGTERM is noticed promptly
        socket.setsockopt(zmq.LINGER, 0)
        endpoint = f"tcp://{self.args.host}:{self.args.port}"
        socket.bind(endpoint)
        print(
            f"[serve] endpoint={endpoint} model={self.args.model_path or '<dry-run>'} "
            f"replan_every={self.replan_every} image_scale={self.image_scale} "
            f"instruction={self.instruction!r}",
            flush=True,
        )
        print(f"SERVER_READY port={self.args.port}", flush=True)
        try:
            while not _STOP:
                try:
                    request = socket.recv_pyobj()
                except zmq.Again:
                    continue
                except zmq.ContextTerminated:
                    break
                try:
                    reply = self.handle(request)
                except Exception as exc:  # noqa: BLE001 - never take the server down
                    traceback.print_exc()
                    reply = {"error": f"{type(exc).__name__}: {exc}"}
                socket.send_pyobj(reply)
        finally:
            socket.close(0)
            print(
                f"[serve] stopped after {self.requests} act requests, "
                f"{self.replans} replans, {self.episode} episodes",
                flush=True,
            )
        return 0


# ----------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-path", default=None, help="checkpoint directory")
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument("--modality-config-path", default=str(DEFAULT_MODALITY_CONFIG))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5602)
    parser.add_argument(
        "--replan-every", type=int, default=20,
        help="rows of the 40-step chunk played before re-querying the policy "
             "(20 at 50 Hz = 2.5 Hz replanning)",
    )
    parser.add_argument(
        "--image-scale", type=float, default=0.5,
        help="cv2.INTER_AREA downscale applied to every live frame; must match "
             "the dataset (0.5 -> 640x360 top, 424x240 wrists). 1.0 = no resize",
    )
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="no model: reply with the request's own state held for 40 steps "
             "(wire tests on CPU)",
    )
    parser.add_argument("--log-every", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dry_run and not args.model_path:
        print("ERROR: --model-path is required unless --dry-run is set", flush=True)
        return 2
    _install_signal_handlers()
    try:
        return JointPolicyServer(args).serve()
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"SERVER_FAILED: {type(exc).__name__}: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
