#!/usr/bin/env python3
"""Build harness/lane_b/dual_fr3.xml: two menagerie `franka_fr3` arms on the angled rig.

    python3 harness/lane_b/make_dual_fr3_xml.py \
        --menagerie ~/code/upstream/mujoco_menagerie/franka_fr3/fr3.xml \
        --out harness/lane_b/dual_fr3.xml

One articulation, ONE root body `base` (fixed to the world, no freejoint), the two
arms as children with every body / joint / geom / site name prefixed `left_` /
`right_`. Geometry = the franka repo's angled rig (frankas_assets/rig.py,
FR3_RIG=angled): bases ARM_SPACING=0.20 m apart along setup-local x, ARM_BACK=0.005 m
back (+y), BASE_Z=0.14 m up, both arms yawed -90 deg (facing setup-local -y) and
rolled -45 deg (left, +x side) / +45 deg (right, -x side) about their own x —
`base_R(roll) = Rz(YAW_DEG) @ Rx(roll)`, quaternion `base_quat_wxyz`. The rig's
world SETUP_POS/SETUP_YAW are NOT applied: the SONIC embodiment lives in the
setup-local frame (the demo joint trajectories are base-frame invariant).

Written for the two consumers that parse this file:
  * gear_sonic's Humanoid_Batch.from_mjcf — reads body `pos`/`quat` attributes
    literally (so every quaternion is written explicitly and normalised), takes
    the DoF order from document order of `<joint>` (left_fr3_joint1..7 then
    right_fr3_joint1..7), needs an integer `axis` on each joint and one named
    `<actuator>` per DoF.
  * Isaac Lab's MjcfFileCfg importer — needs an inertial on every link, so `base`
    and the two `fr3_link0` get explicit ones (menagerie leaves them massless).
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import xml.etree.ElementTree as ET

# --- angled rig, verbatim from frankas_assets/rig.py (setup-local frame) ---------------
ARM_SPACING = 0.20
BASE_Z = 0.14
ARM_BACK = 0.005
YAW_DEG = -90.0
ROLL_LEFT_DEG = -45.0
ROLL_RIGHT_DEG = 45.0
ARM_LOCAL = {
    "left": (ARM_SPACING / 2.0, ARM_BACK, BASE_Z),
    "right": (-ARM_SPACING / 2.0, ARM_BACK, BASE_Z),
}
ARM_ROLL = {"left": ROLL_LEFT_DEG, "right": ROLL_RIGHT_DEG}

# FR3 "ready" pose used by the franka repo's FR3_CFG init_state (frankas_assets/fr3.py)
READY_POSE_FR3 = (0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741)

# --- joint-6 zero shift ---------------------------------------------------------------
# FR3 joint 6 ranges over [0.5445, 4.5169] rad and the handover demos reach 4.517. gear_sonic's
# motion library re-derives joint angles from quaternions (fk_batch: axis-angle -> quat ->
# axis-angle), which wraps any |angle| > pi by 2*pi. So this model rotates link6's frame by
# Rz(J6_OFFSET) and shifts the joint-6 range accordingly: SONIC-side q6 = FR3 q6 - J6_OFFSET,
# range [-1.9862, 1.9862]. Kinematics are identical; the offset is a convention that must be
# applied wherever FR3 joint values cross into or out of the SONIC embodiment (the demo->pkl
# converter, the P3 token labeller / decoder server). Every other joint is inside (-pi, pi].
J6_OFFSET = 2.5307  # (0.5445 + 4.5169) / 2
READY_POSE = tuple(v - J6_OFFSET if i == 5 else v for i, v in enumerate(READY_POSE_FR3))


def quat_mul(a, b):
    """Hamilton product of two wxyz quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def shift_joint6(link6: ET.Element, side: str) -> None:
    """Rotate link6's body frame by Rz(J6_OFFSET) and shift its joint range by -J6_OFFSET."""
    q = tuple(float(x) for x in link6.attrib["quat"].split())
    rz = (math.cos(J6_OFFSET / 2), 0.0, 0.0, math.sin(J6_OFFSET / 2))
    link6.attrib["quat"] = fmt(quat_mul(q, rz))
    joint = link6.find("joint")
    assert joint is not None and joint.attrib["name"] == f"{side}_fr3_joint6", joint.attrib
    lo, hi = (float(x) for x in joint.attrib["range"].split())
    joint.attrib["range"] = fmt((lo - J6_OFFSET, hi - J6_OFFSET), 4)
    joint.attrib["ref"] = "0"  # explicit: the offset lives in the body frame, not in `ref`


def base_quat_wxyz(roll_deg: float, yaw_deg: float = YAW_DEG) -> tuple[float, float, float, float]:
    """q_yaw (x) q_roll, (w, x, y, z) — identical to frankas_assets.rig.base_quat_wxyz."""
    hy, hr = math.radians(yaw_deg) / 2.0, math.radians(roll_deg) / 2.0
    cy, sy, cr, sr = math.cos(hy), math.sin(hy), math.cos(hr), math.sin(hr)
    return (cy * cr, cy * sr, sy * sr, sy * cr)


def fmt(vals, nd=8) -> str:
    out = []
    for v in vals:
        s = f"{float(v):.{nd}f}".rstrip("0").rstrip(".")
        out.append("0" if s in ("-0", "") else s)
    return " ".join(out)


def normalised_quat_attr(q_str: str) -> str:
    q = [float(x) for x in q_str.split()]
    n = math.sqrt(sum(x * x for x in q))
    return fmt([x / n for x in q])


def prefix_subtree(node: ET.Element, prefix: str) -> None:
    """Prefix every name-bearing element under `node` (in place) and normalise body quats."""
    for el in node.iter():
        if "name" in el.attrib:
            el.attrib["name"] = prefix + el.attrib["name"]
        if el.tag == "body" and "quat" in el.attrib:
            el.attrib["quat"] = normalised_quat_attr(el.attrib["quat"])
        if el.tag == "body" and "pos" not in el.attrib:
            el.attrib["pos"] = "0 0 0"
        if el.tag == "body" and "quat" not in el.attrib:
            el.attrib["quat"] = "1 0 0 0"


def build(menagerie_xml: str, meshdir: str) -> ET.ElementTree:
    src = ET.parse(menagerie_xml).getroot()
    src_base = src.find("worldbody").find("body")  # menagerie's `base` (childclass=fr3)
    src_link0 = src_base.find("body")  # fr3_link0 subtree
    assert src_link0.attrib["name"] == "fr3_link0", src_link0.attrib

    root = ET.Element("mujoco", model="dual_fr3")
    ET.SubElement(root, "compiler", angle="radian", meshdir=meshdir)
    ET.SubElement(root, "option", integrator="implicitfast")
    root.append(copy.deepcopy(src.find("default")))
    asset = copy.deepcopy(src.find("asset"))
    # gear_sonic's Humanoid_Batch.load_mesh indexes every <mesh> by its `name` attribute;
    # menagerie leaves it implicit (MuJoCo uses the file stem). Make it explicit.
    for mesh in asset.findall("mesh"):
        if "name" not in mesh.attrib:
            mesh.attrib["name"] = os.path.splitext(os.path.basename(mesh.attrib["file"]))[0]
    root.append(asset)

    worldbody = ET.SubElement(root, "worldbody")
    base = ET.SubElement(
        worldbody, "body", name="base", pos="0 0 0", quat="1 0 0 0", childclass="fr3"
    )
    # A dummy mount mass so the Isaac importer gets a proper root link (menagerie's base
    # is massless; MuJoCo does not care for a static body).
    ET.SubElement(base, "inertial", pos="0 0 0.07", mass="10", diaginertia="0.2 0.2 0.2")

    contact = ET.Element("contact")
    actuator = ET.Element("actuator")
    src_act = {a.attrib["joint"]: a for a in src.find("actuator")}
    qpos_home, ctrl_home = [], []
    for side in ("left", "right"):
        pre = f"{side}_"
        link0 = copy.deepcopy(src_link0)
        prefix_subtree(link0, pre)
        link0.attrib["pos"] = fmt(ARM_LOCAL[side])
        link0.attrib["quat"] = fmt(base_quat_wxyz(ARM_ROLL[side]))
        link6 = [b for b in link0.iter("body") if b.attrib["name"] == f"{pre}fr3_link6"]
        assert len(link6) == 1
        shift_joint6(link6[0], side)
        # explicit inertial for link0 (real FR3 link0 is ~ 2.9 kg; static in menagerie)
        ET.SubElement(
            link0, "inertial", pos="-0.041 0 0.05", mass="2.9", diaginertia="0.02 0.02 0.01"
        )
        # inertial must precede child bodies for readability (MuJoCo does not care)
        link0.remove(link0[-1])
        link0.insert(0, ET.Element(
            "inertial", pos="-0.041 0 0.05", mass="2.9", diaginertia="0.02 0.02 0.01"
        ))
        base.append(link0)
        ET.SubElement(contact, "exclude", body1=f"{pre}fr3_link0", body2=f"{pre}fr3_link1")
        for i in range(1, 8):
            jn = f"fr3_joint{i}"
            a = copy.deepcopy(src_act[jn])
            a.attrib["name"] = pre + jn
            a.attrib["joint"] = pre + jn
            actuator.append(a)
        qpos_home += list(READY_POSE)
        ctrl_home += list(READY_POSE)

    root.append(actuator)
    root.append(contact)
    kf = ET.SubElement(root, "keyframe")
    ET.SubElement(kf, "key", name="home", qpos=fmt(qpos_home, 4), ctrl=fmt(ctrl_home, 4))
    return ET.ElementTree(root)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--menagerie", required=True, help="path to mujoco_menagerie/franka_fr3/fr3.xml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--meshdir", default="dual_fr3_assets",
                    help="mesh directory relative to the xml (a symlink to menagerie's assets/)")
    args = ap.parse_args()
    tree = build(os.path.expanduser(args.menagerie), args.meshdir)
    ET.indent(tree, space="  ")
    header = (
        "<!-- GENERATED by harness/lane_b/make_dual_fr3_xml.py from mujoco_menagerie franka_fr3\n"
        "     (angled rig: 0.20 m apart, +0.14 m, roll -45/+45 deg, yaw -90 deg). Do not hand-edit.\n"
        f"     JOINT 6 CONVENTION: q6_here = q6_fr3 - {J6_OFFSET} (link6 frame pre-rotated by Rz), so\n"
        "     every joint stays inside (-pi, pi) for the motion library's axis-angle round trip. -->\n"
    )
    body = ET.tostring(tree.getroot(), encoding="unicode")
    with open(os.path.expanduser(args.out), "w") as fh:
        fh.write(header + body + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
