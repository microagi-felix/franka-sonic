#!/usr/bin/env python3
"""GPU device allocator for the FR3 handover bake-off pod.

Run it with the GR00T venv interpreter, which is the one with torch:

    ~/Isaac-GR00T/.venv/bin/python harness/gpus.py probe
    eval "$(~/Isaac-GR00T/.venv/bin/python harness/gpus.py acquire --n 2 --job lane-a-finetune)"
    ~/Isaac-GR00T/.venv/bin/python harness/gpus.py release --job lane-a-finetune
    ~/Isaac-GR00T/.venv/bin/python harness/gpus.py list

Why this exists: the pod has no `nvidia-smi`, and it sees all eight devices on
a node it shares with processes outside the pod (two probes on 2026-09-01
showed identical foreign memory on six of them). Occupancy therefore comes
from `torch.cuda.mem_get_info`, and a device is only handed out when it
reports less than IDLE_GIB used. Claims live in a lock-protected JSON file so
two jobs launched a second apart cannot take the same card; a claim whose PID
is gone is dropped on sight.

`torch.cuda.mem_get_info(i)` creates a CUDA context on device i (~0.4-0.5 GiB
while this process lives), which is exactly why an idle device reads ~0.5 GiB
rather than 0 and why the threshold is 1 GiB and not something tighter.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import json
import os
import socket
import sys
from pathlib import Path

POOL = Path(os.path.expanduser("~/runs/franka-sonic/gpus.json"))
IDLE_GIB = 1.0
GIB = 1024 ** 3


# --------------------------------------------------------------------------- probe
def probe() -> list[dict]:
    """[{index, used_gib, total_gib, free_gib}] for every physical device."""
    # Never inherit a restricted view: the whole point is to see all devices.
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    try:
        import torch
    except ImportError:  # pragma: no cover - environment error, not logic
        sys.exit("torch not importable — run this with ~/Isaac-GR00T/.venv/bin/python")
    if not torch.cuda.is_available():
        sys.exit("torch reports no CUDA devices")
    out = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        out.append(
            {
                "index": i,
                "used_gib": round((total - free) / GIB, 2),
                "free_gib": round(free / GIB, 2),
                "total_gib": round(total / GIB, 2),
            }
        )
    return out


# --------------------------------------------------------------------------- pool file
class Pool:
    """The claims file, held under an exclusive fcntl lock for the whole block."""

    def __init__(self, path: Path = POOL):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None
        self.data: dict = {}

    def __enter__(self) -> "Pool":
        self._fh = open(self.path, "a+")
        fcntl.flock(self._fh, fcntl.LOCK_EX)
        self._fh.seek(0)
        raw = self._fh.read().strip()
        try:
            self.data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self.data = {}
        self.data.setdefault("claims", {})
        self._reap()
        return self

    def __exit__(self, *exc):
        self._fh.seek(0)
        self._fh.truncate()
        json.dump(self.data, self._fh, indent=2, sort_keys=True)
        self._fh.write("\n")
        self._fh.flush()
        fcntl.flock(self._fh, fcntl.LOCK_UN)
        self._fh.close()
        return False

    def _reap(self) -> None:
        """Drop claims whose process is gone (crashed job, killed tmux window)."""
        for job, claim in list(self.data["claims"].items()):
            pid = claim.get("pid")
            if pid is None:
                continue
            try:
                os.kill(int(pid), 0)
            except (ProcessLookupError, ValueError):
                del self.data["claims"][job]
            except PermissionError:
                pass  # alive, owned by someone else

    def claimed(self) -> dict[int, str]:
        return {
            int(d): job
            for job, claim in self.data["claims"].items()
            for d in claim.get("devices", [])
        }


# --------------------------------------------------------------------------- commands
def cmd_probe(_args) -> int:
    rows = probe()
    print(f"{'gpu':>3}  {'used GiB':>9}  {'total GiB':>9}  eligible (<%.1f GiB used)" % IDLE_GIB)
    for r in rows:
        flag = "yes" if r["used_gib"] < IDLE_GIB else "no"
        print(f"{r['index']:>3}  {r['used_gib']:>9.2f}  {r['total_gib']:>9.2f}  {flag}")
    return 0


def cmd_acquire(args) -> int:
    rows = probe()
    with Pool() as pool:
        taken = pool.claimed()
        eligible = [
            r for r in rows if r["used_gib"] < IDLE_GIB and r["index"] not in taken
        ]
        eligible.sort(key=lambda r: (r["used_gib"], r["index"]))
        if args.job in pool.data["claims"]:
            print(
                f"job {args.job!r} already holds "
                f"{pool.data['claims'][args.job]['devices']} — release it first",
                file=sys.stderr,
            )
            return 1
        if len(eligible) < args.n:
            busy = ", ".join(
                f"{r['index']}={r['used_gib']:.1f}GiB" for r in rows
            )
            print(
                f"need {args.n} idle device(s), found {len(eligible)}.\n"
                f"  occupancy: {busy}\n"
                f"  claimed here: {taken or '{}'}\n"
                f"  a device counts as idle below {IDLE_GIB} GiB used "
                f"(the node is shared — see AGENTS.md rule a).",
                file=sys.stderr,
            )
            return 1
        devices = sorted(r["index"] for r in eligible[: args.n])
        pool.data["claims"][args.job] = {
            "devices": devices,
            "pid": args.pid,
            "job": args.job,
            "host": socket.gethostname(),
            "time": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "used_gib_at_acquire": {
                str(r["index"]): r["used_gib"] for r in rows if r["index"] in devices
            },
        }
    line = ",".join(str(d) for d in devices)
    print(f"CUDA_VISIBLE_DEVICES={line}")
    print(f"export CUDA_VISIBLE_DEVICES={line}")
    return 0


def cmd_release(args) -> int:
    with Pool() as pool:
        claim = pool.data["claims"].pop(args.job, None)
    if claim is None:
        print(f"no claim for job {args.job!r} (already released or reaped)")
        return 0
    print(f"released {claim['devices']} held by {args.job!r}")
    return 0


def cmd_list(_args) -> int:
    with Pool() as pool:
        claims = pool.data["claims"]
    if not claims:
        print(f"no claims in {POOL}")
        return 0
    print(f"{'job':<28} {'devices':<12} {'pid':>8}  acquired")
    for job, c in sorted(claims.items()):
        devs = ",".join(str(d) for d in c.get("devices", []))
        print(f"{job:<28} {devs:<12} {str(c.get('pid')):>8}  {c.get('time')}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="GPU device allocator (pool file: %s)" % POOL
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("probe", help="per-device used/total memory").set_defaults(fn=cmd_probe)

    a = sub.add_parser("acquire", help="claim N idle devices and print the export line")
    a.add_argument("--n", type=int, default=1)
    a.add_argument("--job", required=True)
    a.add_argument(
        "--pid",
        type=int,
        default=os.getppid(),
        help="process whose death releases the claim (default: the calling shell)",
    )
    a.set_defaults(fn=cmd_acquire)

    r = sub.add_parser("release", help="drop this job's claim")
    r.add_argument("--job", required=True)
    r.set_defaults(fn=cmd_release)

    sub.add_parser("list", help="show live claims").set_defaults(fn=cmd_list)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
