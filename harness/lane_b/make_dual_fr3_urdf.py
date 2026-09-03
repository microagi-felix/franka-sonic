#!/usr/bin/env python3
"""Derive dual_fr3.urdf from dual_fr3.xml (the MJCF is the single source of truth).

    python3 harness/lane_b/make_dual_fr3_urdf.py --xml harness/lane_b/dual_fr3.xml \
        --out harness/lane_b/dual_fr3.urdf

Why a URDF: Isaac Lab spawns G1/H2 through the URDF importer; the Isaac Sim 5.1 MJCF
importer wrote empty USD layers and hung on this pod (2026-09-03). The URDF mirrors the
MJCF body tree one-to-one (same link and joint names, same body poses — including the
joint-6 frame shift — same inertials, same joint ranges), so the IsaacLab <-> MuJoCo
mappings in robots_dual_fr3.py hold. Mesh references are the menagerie collision STLs
(`dual_fr3_assets/linkN.stl`, one per link, used for visual and collision alike; the
visuals are irrelevant headless), resolved relative to the URDF's directory — install a
`dual_fr3_assets` symlink next to it exactly like next to the MJCF.

Stdlib only (runs under plain python3).
"""

from __future__ import annotations

import argparse
import math
import os
import xml.etree.ElementTree as ET

# FR3 joint velocity limits (rad/s), Franka datasheet
FR3_VEL = [2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26]


def quat_to_rpy(q):
    """wxyz quaternion -> URDF rpy (extrinsic x-y-z = intrinsic z-y'-x'')."""
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    s = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(s)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return (roll, pitch, yaw)


def f3(v, nd=8):
    out = []
    for x in v:
        s = f"{float(x):.{nd}f}".rstrip("0").rstrip(".")
        out.append("0" if s in ("-0", "") else s)
    return " ".join(out)


def floats(s, default):
    return [float(t) for t in s.split()] if s else list(default)


def build(xml_path: str, meshdir: str) -> ET.Element:
    src = ET.parse(xml_path).getroot()
    robot = ET.Element("robot", name="dual_fr3")
    default_joint = src.find("default").find("default").find("joint")
    default_armature = float(default_joint.attrib.get("armature", "0")) if default_joint is not None else 0.0

    def add_link(body: ET.Element):
        name = body.attrib["name"]
        link = ET.SubElement(robot, "link", name=name)
        inert = body.find("inertial")
        if inert is not None:
            ie = ET.SubElement(link, "inertial")
            ET.SubElement(ie, "origin",
                          xyz=f3(floats(inert.attrib.get("pos"), (0, 0, 0))),
                          rpy=f3(quat_to_rpy(floats(inert.attrib.get("quat"), (1, 0, 0, 0)))))
            ET.SubElement(ie, "mass", value=inert.attrib["mass"])
            d = floats(inert.attrib["diaginertia"], (0, 0, 0))
            ET.SubElement(ie, "inertia", ixx=f3([d[0]]), iyy=f3([d[1]]), izz=f3([d[2]]),
                          ixy="0", ixz="0", iyz="0")
        # one STL per link: the menagerie collision mesh (linkN.stl)
        stem = name.split("_", 1)[1] if "_" in name else name  # left_fr3_link3 -> fr3_link3
        mesh = None
        if stem.startswith("fr3_link"):
            mesh = f"{meshdir}/link{stem[len('fr3_link'):]}.stl"
        if mesh is not None:
            for tag in ("visual", "collision"):
                el = ET.SubElement(link, tag)
                ET.SubElement(el, "origin", xyz="0 0 0", rpy="0 0 0")
                geom = ET.SubElement(el, "geometry")
                ET.SubElement(geom, "mesh", filename=mesh)
        return link

    def walk(body: ET.Element, parent: str | None):
        name = body.attrib["name"]
        add_link(body)
        if parent is not None:
            joint = body.find("joint")
            origin_xyz = f3(floats(body.attrib.get("pos"), (0, 0, 0)))
            origin_rpy = f3(quat_to_rpy(floats(body.attrib.get("quat"), (1, 0, 0, 0))))
            if joint is None:
                j = ET.SubElement(robot, "joint", name=f"{parent}_to_{name}", type="fixed")
                ET.SubElement(j, "origin", xyz=origin_xyz, rpy=origin_rpy)
                ET.SubElement(j, "parent", link=parent)
                ET.SubElement(j, "child", link=name)
            else:
                jname = joint.attrib["name"]
                axis = floats(joint.attrib.get("axis"), (0, 0, 1))
                lo, hi = floats(joint.attrib["range"], (-math.pi, math.pi))
                frc = floats(joint.attrib.get("actuatorfrcrange"), (-87, 87))
                idx = int(jname[-1]) - 1
                j = ET.SubElement(robot, "joint", name=jname, type="revolute")
                ET.SubElement(j, "origin", xyz=origin_xyz, rpy=origin_rpy)
                ET.SubElement(j, "parent", link=parent)
                ET.SubElement(j, "child", link=name)
                ET.SubElement(j, "axis", xyz=f3(axis))
                ET.SubElement(j, "limit", lower=f3([lo]), upper=f3([hi]),
                              effort=f3([abs(frc[1])]), velocity=f3([FR3_VEL[idx]]))
                # MuJoCo's damping/frictionloss are N*m*s/rad and N*m; the Isaac URDF importer
                # turns URDF `friction` into a PhysX *coefficient* (friction torque = coeff x
                # joint constraint force), so 1.137 would glue the joints (probe 2026-09-03).
                # Damping comes from the implicit PD actuator instead. Both zero here.
                ET.SubElement(j, "dynamics", damping="0", friction="0")
        for child in body.findall("body"):
            walk(child, name)

    root_body = src.find("worldbody").find("body")
    walk(root_body, None)
    return robot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meshdir", default="dual_fr3_assets")
    args = ap.parse_args()
    robot = build(os.path.expanduser(args.xml), args.meshdir)
    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ")
    header = (
        "<!-- GENERATED by harness/lane_b/make_dual_fr3_urdf.py from dual_fr3.xml. Do not hand-edit.\n"
        "     Same links/joints/poses as the MJCF, incl. the joint-6 zero shift (q6 = q6_fr3 - 2.5307). -->\n"
    )
    with open(os.path.expanduser(args.out), "w") as fh:
        fh.write(header + ET.tostring(robot, encoding="unicode") + "\n")
    n_links = len(robot.findall("link"))
    n_rev = len([j for j in robot.findall("joint") if j.attrib["type"] == "revolute"])
    print(f"wrote {args.out}: {n_links} links, {n_rev} revolute joints")


if __name__ == "__main__":
    main()
