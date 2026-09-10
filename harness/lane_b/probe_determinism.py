#!/usr/bin/env python
"""P12 WP 12.3 — is lane B's policy function bit-reproducible?

P11 closed with a per-process bistable regime in lane B's late checkpoints: one 200-rollout run
of `checkpoint-17500` scored 3/81, then 100/106, then 1/12 with no relaunch, and the seed, the
checkpoint files, the device, the process environment, the ONNX provider, the ports, code drift,
server latency, rendering, image lag, camera alignment, decoder state and the client loop are all
excluded with data. This probe partitions what is left: it replays REAL requests (captured by
`serve_gr00t_joint.py --dump-requests`) through the policy function alone — GR00T forward ->
(40, 66) token chunk -> optionally one decoder step — with the simulator removed, and asks
whether the same request gives the same chunk

    * K times inside one process (with the RNG re-seeded identically before each forward, exactly
      as `--server-seed 20260905` seeds it for episode 0),
    * across fresh processes,
    * across processes while a 20-rollout screen shares the device.

    forward  one process pass over a dump directory -> <out>.json + <out>.npz (the chunks)
    compare  several passes -> one JSON + one markdown carrying the VERDICT: line

Usage (the driver in the probe run folder does this three times over):

    probe_determinism.py forward --requests <dump dir> --model-path <ckpt> \\
        --decoder-onnx <...>/model_decoder.onnx --encoder-onnx <...>/model_encoder.onnx \\
        --out <run>/out/pass_quiet_1 [--repeats 5] [--limit 400] [--tag quiet_1]
    probe_determinism.py compare --quiet <pass>.json ... --loaded <pass>.json ... \\
        --out-json <run>/out/probe.json --out-md <run>/out/probe.md

Nothing here writes outside the paths it is given, and it never removes anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lane_a"))

TOKEN_DIM = 64
GRID = 16.0          # the token quantisation grid the SONIC labels live on: 1/16
GRID_TOL = 1e-6
SUBGRID = 1.0 / 32   # half a grid step: below this a token difference cannot cross a grid cell


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()[:16]


def grid_fraction(chunk: np.ndarray) -> float:
    """Fraction of the token values of a chunk that sit on the 1/16 grid."""
    tok = np.asarray(chunk, dtype=np.float64)[:, :TOKEN_DIM]
    if tok.size == 0:
        return float("nan")
    scaled = tok * GRID
    return float(np.mean(np.abs(scaled - np.round(scaled)) <= GRID_TOL * GRID))


def load_requests(d: Path, limit: int) -> list[Path]:
    files = sorted(d.glob("request_*.npz"))
    return files[:limit] if limit > 0 else files


# ------------------------------------------------------------------ one process pass


def build_server(args):
    """The policy exactly as the server builds it: same modality config, same checkpoint, same
    image resize, same decoder ONNX — by instantiating the server class itself, not a copy of it.
    `serve()` is never called, so no socket is bound."""
    import serve_gr00t_sonic_joint as sonic
    import serve_gr00t_joint as base

    argv = ["--model-path", args.model_path, "--embodiment-tag", args.embodiment_tag,
            "--host", "127.0.0.1", "--port", "0", "--replan-every", "20",
            "--image-size", args.image_size, "--device", args.device]
    if args.modality_config_path:
        argv += ["--modality-config-path", args.modality_config_path]
    if args.decoder_onnx:
        argv += ["--decoder-onnx", args.decoder_onnx]
    if args.encoder_onnx:
        argv += ["--encoder-onnx", args.encoder_onnx]
    if args.seed is not None:
        argv += ["--seed", str(args.seed)]
    srv_args = sonic.build_parser().parse_args(argv)
    srv = sonic.SonicPolicyServer(srv_args)
    return srv, base


def forward(args) -> int:
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    reqs = load_requests(Path(args.requests).expanduser(), args.limit)
    if not reqs:
        print(f"[probe] no request_*.npz under {args.requests}", flush=True)
        return 2
    t0 = time.time()
    srv, base = build_server(args)
    print(f"[probe] policy ready in {time.time() - t0:.1f}s; {len(reqs)} requests, "
          f"{args.repeats} repeats each", flush=True)

    per_request, chunks = [], {}
    for i, f in enumerate(reqs):
        z = np.load(f)
        request = {"type": "act", "state": z["state"]}
        for cam in base.CAMERAS:
            if cam in z.files:
                request[cam] = z[cam]
        observation, state = base.build_observation(
            request, srv.instruction, srv.image_scale, srv.image_size)

        reseeded, free = [], []
        for k in range(args.repeats):          # primary: identical RNG state before every forward
            srv.episode = 0
            srv._seed_rngs("probe") if srv.seed is not None else None
            reseeded.append(np.asarray(srv._predict(observation, state), dtype=np.float32))
        for k in range(args.repeats):          # secondary: the sampler left free-running
            free.append(np.asarray(srv._predict(observation, state), dtype=np.float32))

        ref = reseeded[0]
        dmax = max(float(np.abs(c - ref).max()) for c in reseeded[1:]) if len(reseeded) > 1 else 0.0
        dmax_tok = (max(float(np.abs(c[:, :TOKEN_DIM] - ref[:, :TOKEN_DIM]).max())
                        for c in reseeded[1:]) if len(reseeded) > 1 else 0.0)
        fref = free[0]
        fmax = max(float(np.abs(c - fref).max()) for c in free[1:]) if len(free) > 1 else 0.0
        rec = z["chunk"] if "chunk" in z.files else None
        rec_d = (float(np.abs(np.asarray(rec, np.float32) - ref).max())
                 if rec is not None and np.asarray(rec).shape == ref.shape else None)
        per_request.append({
            "file": f.name,
            "episode": int(z["episode"]) if "episode" in z.files else None,
            "request": int(z["request"]) if "request" in z.files else None,
            "replanned_here": bool(z["replanned_here"]) if "replanned_here" in z.files else None,
            "sha_reseeded": [sha(c) for c in reseeded],
            "within_process_bitwise_equal": len(set(sha(c) for c in reseeded)) == 1,
            "within_process_max_abs_delta": dmax,
            "within_process_max_abs_delta_token": dmax_tok,
            "free_running_max_abs_delta": fmax,
            "grid_fraction_reseeded_0": grid_fraction(ref),
            "grid_fraction_recorded": grid_fraction(rec) if rec is not None else None,
            "delta_vs_recorded_chunk": rec_d,
        })
        chunks[f.stem] = ref
        if (i + 1) % 25 == 0:
            print(f"[probe] {i + 1}/{len(reqs)} requests", flush=True)

    summary = {
        "tag": args.tag,
        "pid": os.getpid(),
        "host": platform.node(),
        "device_env": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "model_path": args.model_path,
        "requests_dir": str(Path(args.requests).expanduser()),
        "n_requests": len(reqs),
        "repeats": args.repeats,
        "seed": args.seed,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "all_within_process_bitwise_equal": all(r["within_process_bitwise_equal"] for r in per_request),
        "max_within_process_delta": max(r["within_process_max_abs_delta"] for r in per_request),
        "max_free_running_delta": max(r["free_running_max_abs_delta"] for r in per_request),
        "mean_grid_fraction": float(np.mean([r["grid_fraction_reseeded_0"] for r in per_request])),
        "mean_grid_fraction_recorded": float(np.mean(
            [r["grid_fraction_recorded"] for r in per_request
             if r["grid_fraction_recorded"] is not None] or [float("nan")])),
        "requests": per_request,
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1) + "\n")
    np.savez(out.with_suffix(".npz"), **chunks)
    print(f"[probe] wrote {out.with_suffix('.json')} and {out.with_suffix('.npz')} "
          f"in {time.time() - t0:.0f}s", flush=True)
    return 0


# ------------------------------------------------------------------ cross-process comparison


def cross(passes: list[tuple[str, dict, dict]]):
    """passes = [(tag, summary json, {stem: chunk}), ...] -> per-request cross-process stats."""
    if len(passes) < 2:
        return {"n_compared": 0, "bitwise_equal": None, "max_abs_delta": None,
                "max_abs_delta_token": None, "differing": []}
    stems = sorted(set.intersection(*[set(ch) for _, _, ch in passes]))
    dmax, dmax_tok, differing = 0.0, 0.0, []
    for s in stems:
        arrs = [ch[s] for _, _, ch in passes]
        ref = arrs[0]
        d = max(float(np.abs(a - ref).max()) for a in arrs[1:])
        dt = max(float(np.abs(a[:, :TOKEN_DIM] - ref[:, :TOKEN_DIM]).max()) for a in arrs[1:])
        if d > 0:
            differing.append({"file": s, "max_abs_delta": d, "max_abs_delta_token": dt})
        dmax, dmax_tok = max(dmax, d), max(dmax_tok, dt)
    return {"n_compared": len(stems), "tags": [t for t, _, _ in passes],
            "bitwise_equal": dmax == 0.0, "max_abs_delta": dmax,
            "max_abs_delta_token": dmax_tok, "n_differing": len(differing),
            "differing": differing[:20]}


def load_pass(p: str):
    j = json.loads(Path(p).read_text())
    npz = Path(p).with_suffix(".npz")
    ch = dict(np.load(npz)) if npz.exists() else {}
    return (j.get("tag") or Path(p).stem, j, ch)


def verdict_line(quiet, loaded, within_bad, within_delta) -> str:
    if within_bad:
        return (f"VERDICT: differs within a process (max |dchunk| = {within_delta:.3e}) — the policy "
                f"function is not bit-reproducible even with an identical RNG state")
    if quiet["n_compared"] and not quiet["bitwise_equal"]:
        return (f"VERDICT: differs across processes even when quiet (max |dchunk| = "
                f"{quiet['max_abs_delta']:.3e}, max |dtoken| = {quiet['max_abs_delta_token']:.3e})")
    if loaded["n_compared"] and not loaded["bitwise_equal"]:
        return (f"VERDICT: differs across processes only under load (max |dtoken| = "
                f"{loaded['max_abs_delta_token']:.3e})")
    return "VERDICT: bitwise-reproducible within and across processes (quiet and under load)"


def compare(args) -> int:
    qp = [load_pass(p) for p in args.quiet]
    lp = [load_pass(p) for p in args.loaded]
    allp = qp + lp
    within_bad = [t for t, j, _ in allp if not j["all_within_process_bitwise_equal"]]
    within_delta = max([j["max_within_process_delta"] for _, j, _ in allp] or [0.0])
    free_delta = max([j["max_free_running_delta"] for _, j, _ in allp] or [0.0])
    q = cross(qp)
    l = cross(qp[:1] + lp) if lp else {"n_compared": 0, "bitwise_equal": None,
                                       "max_abs_delta": None, "max_abs_delta_token": None,
                                       "differing": []}
    v = verdict_line(q, l, bool(within_bad), within_delta)
    grid = float(np.mean([j["mean_grid_fraction"] for _, j, _ in allp]))
    grid_rec = float(np.nanmean([j.get("mean_grid_fraction_recorded", float("nan"))
                                 for _, j, _ in allp]))
    out = {
        "verdict": v,
        "passes": [{"tag": t, "pid": j["pid"], "device_env": j["device_env"],
                    "n_requests": j["n_requests"], "repeats": j["repeats"],
                    "started": j["started"], "finished": j["finished"],
                    "within_process_bitwise_equal": j["all_within_process_bitwise_equal"],
                    "max_within_process_delta": j["max_within_process_delta"],
                    "max_free_running_delta": j["max_free_running_delta"],
                    "mean_grid_fraction": j["mean_grid_fraction"]} for t, j, _ in allp],
        "within_process": {"all_bitwise_equal": not within_bad,
                           "max_abs_delta": within_delta,
                           "max_free_running_delta": free_delta,
                           "passes_that_differ": within_bad},
        "across_processes_quiet": q,
        "across_processes_under_load": l,
        "grid": {"mean_fraction_on_1_16_grid": grid,
                 "mean_fraction_on_1_16_grid_recorded_chunks": grid_rec,
                 "subgrid_threshold": SUBGRID},
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=1) + "\n")

    def fmt(x):
        return "n/a" if x is None else (f"{x:.6g}")

    md = ["# P12 WP 12.3 — determinism probe of the lane-B policy function", "",
          v, "",
          "One captured `act` request replayed through GR00T + the SONIC head with the simulator",
          "removed. *Reseeded* repeats put the RNG in exactly the state `--server-seed 20260905`",
          "gives at episode 0 before every forward; *free-running* repeats leave the sampler alone",
          "(that column is the sampler's own spread, not a determinism failure).", "",
          "## Passes", "",
          "| tag | pid | CUDA_VISIBLE_DEVICES | requests | repeats | within-process bitwise-equal | max abs delta (reseeded) | max abs delta (free-running) | fraction of token values on the 1/16 grid |",
          "|---|---|---|---|---|---|---|---|---|"]
    for t, j, _ in allp:
        md.append(f"| `{t}` | {j['pid']} | {j['device_env'] or '(unset)'} | {j['n_requests']} | "
                  f"{j['repeats']} | {'yes' if j['all_within_process_bitwise_equal'] else 'NO'} | "
                  f"{fmt(j['max_within_process_delta'])} | {fmt(j['max_free_running_delta'])} | "
                  f"{j['mean_grid_fraction']:.3f} |")
    md += ["", "## Across processes", "",
           f"* **Quiet** ({q['n_compared']} requests compared over {len(qp)} processes): "
           f"{'bitwise-equal' if q['bitwise_equal'] else 'NOT bitwise-equal'}"
           + (f", max |dchunk| = {fmt(q['max_abs_delta'])}, max |dtoken| = "
              f"{fmt(q['max_abs_delta_token'])}, {q.get('n_differing', 0)} of {q['n_compared']} "
              f"requests differ" if q["n_compared"] and not q["bitwise_equal"] else ""),
           f"* **Under load** (a 20-rollout screen on the same device; {l['n_compared']} requests): "
           f"{'bitwise-equal' if l['bitwise_equal'] else ('NOT bitwise-equal' if l['n_compared'] else 'not run')}"
           + (f", max |dchunk| = {fmt(l['max_abs_delta'])}, max |dtoken| = "
              f"{fmt(l['max_abs_delta_token'])}, {l.get('n_differing', 0)} of {l['n_compared']} "
              f"requests differ" if l["n_compared"] and not l["bitwise_equal"] else ""),
           "",
           "## The 1/16 grid", "",
           f"* Mean fraction of the 64 token values per chunk that land on the 1/16 grid: "
           f"**{grid:.3f}** (replayed chunks), {grid_rec:.3f} (the chunks the server actually "
           f"emitted during the capture screen).",
           f"* A token difference below 1/32 = {SUBGRID:.5f} cannot move a value across a grid "
           f"cell; `harness/lane_b/sonic_decoder.py` clips at |t| <= 1.25 and decodes continuous "
           f"values, so such a difference is a sub-grid perturbation. Reported, not interpreted.",
           ""]
    if q["n_compared"] and not q["bitwise_equal"]:
        md += ["### Requests that differ across quiet processes (first 20)", "",
               "| request | max abs delta | max abs delta (token) |", "|---|---|---|"]
        md += [f"| `{d['file']}` | {fmt(d['max_abs_delta'])} | {fmt(d['max_abs_delta_token'])} |"
               for d in q["differing"]]
        md.append("")
    if l["n_compared"] and not l["bitwise_equal"]:
        md += ["### Requests that differ between a quiet process and a loaded one (first 20)", "",
               "| request | max abs delta | max abs delta (token) |", "|---|---|---|"]
        md += [f"| `{d['file']}` | {fmt(d['max_abs_delta'])} | {fmt(d['max_abs_delta_token'])} |"
               for d in l["differing"]]
        md.append("")
    md += ["## What this does and does not settle", "",
           "The probe covers the policy function only: GR00T's forward and the token chunk it",
           "returns. Bit-reproducibility here puts the P11 bistable regime *outside* the policy",
           "function — what stays open is PhysX state carried across resets and the observation",
           "pipeline's timing (both untouched by this probe). A difference here would instead be",
           "quantified against the 1/16 grid above.", ""]
    Path(args.out_md).write_text("\n".join(md))
    print(v, flush=True)
    print(f"[probe] wrote {args.out_json} and {args.out_md}", flush=True)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("forward", help="one process pass over a dump directory")
    f.add_argument("--requests", required=True)
    f.add_argument("--model-path", required=True)
    f.add_argument("--decoder-onnx", default=None)
    f.add_argument("--encoder-onnx", default=None)
    f.add_argument("--modality-config-path", default=None)
    f.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    f.add_argument("--image-size", default="640x360")
    f.add_argument("--device", default="cuda")
    f.add_argument("--seed", type=int, default=20260905)
    f.add_argument("--repeats", type=int, default=5)
    f.add_argument("--limit", type=int, default=0, help="0 = every request in the directory")
    f.add_argument("--tag", default="pass")
    f.add_argument("--out", required=True, help="path without suffix; .json and .npz are written")
    f.set_defaults(func=forward)

    c = sub.add_parser("compare", help="several passes -> JSON + markdown with the VERDICT line")
    c.add_argument("--quiet", nargs="*", default=[])
    c.add_argument("--loaded", nargs="*", default=[])
    c.add_argument("--out-json", required=True)
    c.add_argument("--out-md", required=True)
    c.set_defaults(func=compare)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
