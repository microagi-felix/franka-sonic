#!/usr/bin/env python3
"""Register the dual-FR3 embodiment in a GMR (General Motion Retargeting) clone.

Idempotent installer.  Re-running is a no-op and prints `no change`.
It never deletes anything and only writes inside the GMR working tree
(``--gmr``, default ``~/code/upstream/GMR``).

What it does (all five touch points, verified against GMR @bb1bbe4):

  a) ``general_motion_retargeting/params.py`` — one entry in each of the four
     dicts: ``ROBOT_XML_DICT``, ``IK_CONFIG_DICT["smplx"]``, ``ROBOT_BASE_DICT``
     (= ``"base"``, the fixed root body of ``dual_fr3.xml``) and
     ``VIEWER_CAM_DISTANCE_DICT``.
  b) ``scripts/smplx_to_robot.py`` — ``dual_fr3`` added to the hardcoded
     ``--robot`` ``choices`` list.
  c) ``general_motion_retargeting/ik_configs/smplx_to_dual_fr3.json`` — copy of
     this directory's ``smplx_to_dual_fr3.json`` (the repo holds the canonical
     copy; see ``README.md`` for the rationale behind every number in it).
  d) ``scripts/smplx_to_robot_dataset.py`` — the two silent clip denylists
     (``assets/hard_motions/{0,1}.txt`` and the filename-substring list
     ``["BMLrub", "EKUT", "crawl", "_lie", "upstairs", "downstairs"]``) become
     switchable with a new ``--no-denylist`` flag.  Default behaviour is
     unchanged; with the flag the batch retarget is 1:1 with its input.
  e) ``general_motion_retargeting/motion_retarget.py`` — seed the mink
     configuration from the model's first keyframe instead of ``qpos0``.
     Without this the dual-FR3 dies on the first frame::

         mink.exceptions.NotWithinConfigurationLimits: Joint 3
         (left_fr3_joint4) violates configuration limits -3.0421 <= 0.0 <= -0.1518

     ``mink.Configuration(model)`` starts at ``qpos0`` = all zeros, and the FR3's
     joint 4 never straightens (range [-3.0421, -0.1518]), so zero is outside its
     limits and ``solve_ik``'s ``check_limits`` raises before any IK happens.
     Every humanoid GMR ships has zero inside every joint range, so upstream
     never hit it.  The patch is generic: if the MJCF has a keyframe, start
     there (ours is ``<key name="home">``), otherwise behave exactly as before.

  f) ``assets/dual_fr3/`` — two symlinks so MuJoCo resolves the model *and* its
     meshes: ``dual_fr3.xml`` -> the installed MJCF under
     ``~/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf``
     and ``dual_fr3_assets`` -> the menagerie ``franka_fr3/assets`` dir that the
     MJCF's ``meshdir`` names.  Symlinks (not copies) so the model tracks
     whatever ``harness/lane_b/install_gear_sonic.sh`` last installed.

Usage:
    python3 harness/lane_b/gmr/register_dual_fr3.py [--gmr DIR] [--check]
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys

ROBOT = "dual_fr3"
HERE = pathlib.Path(__file__).resolve().parent

DEFAULT_GMR = pathlib.Path.home() / "code" / "upstream" / "GMR"
DEFAULT_XML = (
    pathlib.Path.home()
    / "GR00T-WholeBodyControl"
    / "gear_sonic"
    / "data"
    / "assets"
    / "robot_description"
    / "mjcf"
    / "dual_fr3.xml"
)
DEFAULT_MESHDIR = (
    pathlib.Path.home() / "code" / "upstream" / "mujoco_menagerie" / "franka_fr3" / "assets"
)

# the fixed root body of dual_fr3.xml (no freejoint; `base` sits at the origin)
ROBOT_BASE_BODY = "base"
VIEWER_CAM_DISTANCE = 2.0

DENY_MARKER = "no_denylist"
KEYFRAME_MARKER = "franka-sonic: seed the IK from the model's first keyframe"

KEYFRAME_OLD = "        self.configuration = mink.Configuration(self.model)\n"
KEYFRAME_NEW = (
    f"        # {KEYFRAME_MARKER} (harness/lane_b/gmr/register_dual_fr3.py).\n"
    "        # mink starts at qpos0 = zeros; the FR3's joint 4 range is\n"
    "        # [-3.0421, -0.1518], so zero is out of bounds and solve_ik's\n"
    "        # check_limits raises NotWithinConfigurationLimits on frame 1.\n"
    "        q_init = self.model.key_qpos[0].copy() if self.model.nkey > 0 else None\n"
    "        self.configuration = mink.Configuration(self.model, q=q_init)\n"
)


class Changes:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def add(self, msg: str) -> None:
        self.lines.append(msg)

    def __bool__(self) -> bool:
        return bool(self.lines)


def _read(p: pathlib.Path) -> str:
    return p.read_text()


def _write(p: pathlib.Path, text: str, dry: bool) -> None:
    if dry:
        return
    p.write_text(text)


def _insert_before_dict_close(text: str, dict_name: str, entry: str, close: str = "}") -> str:
    """Insert `entry` (a full line, already indented) before the line that closes
    the top-level dict `dict_name = {`."""
    key = f"{dict_name} = {{"
    i = text.index(key)
    j = text.index(f"\n{close}", i)
    return text[: j + 1] + entry + text[j + 1 :]


def patch_params(gmr: pathlib.Path, ch: Changes, dry: bool) -> None:
    p = gmr / "general_motion_retargeting" / "params.py"
    text = _read(p)
    orig = text

    if f'"{ROBOT}": ASSET_ROOT' not in text:
        text = _insert_before_dict_close(
            text,
            "ROBOT_XML_DICT",
            f'    "{ROBOT}": ASSET_ROOT / "{ROBOT}" / "{ROBOT}.xml",\n',
        )
        ch.add(f"params.py: ROBOT_XML_DICT[{ROBOT!r}] = assets/{ROBOT}/{ROBOT}.xml")

    if f'"{ROBOT}": IK_CONFIG_ROOT' not in text:
        # the "smplx" sub-dict of IK_CONFIG_DICT closes with a 4-space-indented `},`
        i = text.index('"smplx":{')
        j = text.index("\n    },", i)
        entry = f'        "{ROBOT}": IK_CONFIG_ROOT / "smplx_to_{ROBOT}.json",\n'
        text = text[: j + 1] + entry + text[j + 1 :]
        ch.add(f'params.py: IK_CONFIG_DICT["smplx"][{ROBOT!r}] = smplx_to_{ROBOT}.json')

    if f'"{ROBOT}": "{ROBOT_BASE_BODY}"' not in text:
        text = _insert_before_dict_close(
            text, "ROBOT_BASE_DICT", f'    "{ROBOT}": "{ROBOT_BASE_BODY}",\n'
        )
        ch.add(f"params.py: ROBOT_BASE_DICT[{ROBOT!r}] = {ROBOT_BASE_BODY!r}")

    if f'"{ROBOT}": {VIEWER_CAM_DISTANCE}' not in text:
        text = _insert_before_dict_close(
            text, "VIEWER_CAM_DISTANCE_DICT", f'    "{ROBOT}": {VIEWER_CAM_DISTANCE},\n'
        )
        ch.add(f"params.py: VIEWER_CAM_DISTANCE_DICT[{ROBOT!r}] = {VIEWER_CAM_DISTANCE}")

    if text != orig:
        _write(p, text, dry)


def patch_choices(gmr: pathlib.Path, ch: Changes, dry: bool) -> None:
    p = gmr / "scripts" / "smplx_to_robot.py"
    text = _read(p)
    if f'"{ROBOT}"' in text:
        return
    needle = '"fourier_gr3"]'
    if needle not in text:
        raise SystemExit(
            f"{p}: cannot find the end of the --robot choices list ({needle!r}); "
            "upstream changed, patch by hand"
        )
    text = text.replace(needle, f'"fourier_gr3", "{ROBOT}"]', 1)
    _write(p, text, dry)
    ch.add(f"scripts/smplx_to_robot.py: --robot choices += {ROBOT!r}")


def patch_denylist(gmr: pathlib.Path, ch: Changes, dry: bool) -> None:
    p = gmr / "scripts" / "smplx_to_robot_dataset.py"
    text = _read(p)
    if DENY_MARKER in text:
        return
    orig = text

    arg_anchor = '    parser.add_argument("--num_cpus", default=4, type=int)\n'
    if arg_anchor not in text:
        raise SystemExit(f"{p}: --num_cpus argparse line not found; upstream changed")
    text = text.replace(
        arg_anchor,
        arg_anchor
        + '    parser.add_argument("--no-denylist", dest="no_denylist", default=False,\n'
        '                        action="store_true",\n'
        '                        help="keep every clip: ignore assets/hard_motions/*.txt and "\n'
        '                             "the filename-substring exclusion list")\n',
        1,
    )

    hard_anchor = "        if motion_name in hard_motions:\n"
    if hard_anchor not in text:
        raise SystemExit(f"{p}: hard_motions filter not found; upstream changed")
    text = text.replace(
        hard_anchor, "        if not args.no_denylist and motion_name in hard_motions:\n", 1
    )

    sub_anchor = "        if any(content in motion_name for content in exclude_file_content):\n"
    if sub_anchor not in text:
        raise SystemExit(f"{p}: substring filter not found; upstream changed")
    text = text.replace(
        sub_anchor,
        "        if not args.no_denylist and any(content in motion_name "
        "for content in exclude_file_content):\n",
        1,
    )

    if text == orig:
        return
    _write(p, text, dry)
    ch.add(
        "scripts/smplx_to_robot_dataset.py: added --no-denylist; both clip denylists "
        "(hard_motions + filename substrings) are now gated by it"
    )


def install_ik_config(gmr: pathlib.Path, ch: Changes, dry: bool) -> None:
    src = HERE / f"smplx_to_{ROBOT}.json"
    dst = gmr / "general_motion_retargeting" / "ik_configs" / f"smplx_to_{ROBOT}.json"
    if not src.is_file():
        raise SystemExit(f"missing canonical IK config {src}")
    if dst.is_file() and dst.read_bytes() == src.read_bytes():
        return
    if not dry:
        shutil.copyfile(src, dst)
    ch.add(f"ik_configs/smplx_to_{ROBOT}.json <- {src}")


def patch_keyframe_seed(gmr: pathlib.Path, ch: Changes, dry: bool) -> None:
    p = gmr / "general_motion_retargeting" / "motion_retarget.py"
    text = _read(p)
    if KEYFRAME_MARKER in text:
        return
    if KEYFRAME_OLD not in text:
        raise SystemExit(
            f"{p}: `self.configuration = mink.Configuration(self.model)` not found; "
            "upstream changed, patch by hand"
        )
    _write(p, text.replace(KEYFRAME_OLD, KEYFRAME_NEW, 1), dry)
    ch.add(
        "general_motion_retargeting/motion_retarget.py: mink.Configuration seeded from "
        "the MJCF's first keyframe when there is one (FR3 joint 4 excludes 0.0, so "
        "qpos0 violates the configuration limits)"
    )


def install_assets(gmr: pathlib.Path, xml: pathlib.Path, meshdir: pathlib.Path,
                   ch: Changes, dry: bool) -> None:
    adir = gmr / "assets" / ROBOT
    if not adir.is_dir():
        if not dry:
            adir.mkdir(parents=True, exist_ok=True)
        ch.add(f"mkdir {adir}")
    for link, target in ((adir / f"{ROBOT}.xml", xml), (adir / "dual_fr3_assets", meshdir)):
        if not target.exists():
            raise SystemExit(f"symlink target does not exist: {target}")
        if link.is_symlink():
            if os.readlink(link) == str(target):
                continue
            raise SystemExit(
                f"{link} is a symlink to {os.readlink(link)!r}, expected {str(target)!r}; "
                "not touching it (nothing is ever deleted on this pod) — fix by hand"
            )
        if link.exists():
            raise SystemExit(f"{link} exists and is not a symlink; not touching it")
        if not dry:
            link.symlink_to(target)
        ch.add(f"symlink {link} -> {target}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gmr", type=pathlib.Path, default=DEFAULT_GMR,
                    help="GMR clone to patch (default: %(default)s)")
    ap.add_argument("--xml", type=pathlib.Path, default=DEFAULT_XML,
                    help="installed dual_fr3.xml to link (default: %(default)s)")
    ap.add_argument("--meshdir", type=pathlib.Path, default=DEFAULT_MESHDIR,
                    help="menagerie franka_fr3 assets dir (default: %(default)s)")
    ap.add_argument("--check", action="store_true",
                    help="report what would change, write nothing (exit 1 if not installed)")
    args = ap.parse_args()

    gmr = args.gmr.expanduser().resolve()
    if not (gmr / "general_motion_retargeting" / "params.py").is_file():
        raise SystemExit(f"{gmr} does not look like a GMR clone")

    ch = Changes()
    patch_params(gmr, ch, args.check)
    patch_choices(gmr, ch, args.check)
    install_ik_config(gmr, ch, args.check)
    patch_denylist(gmr, ch, args.check)
    patch_keyframe_seed(gmr, ch, args.check)
    install_assets(gmr, args.xml.expanduser(), args.meshdir.expanduser(), ch, args.check)

    verb = "would change" if args.check else "changed"
    if ch:
        print(f"register_dual_fr3: {verb} {len(ch.lines)} thing(s) in {gmr}")
        for line in ch.lines:
            print(f"  - {line}")
    else:
        print(f"register_dual_fr3: no change — {ROBOT} already registered in {gmr}")
    if args.check and ch:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
