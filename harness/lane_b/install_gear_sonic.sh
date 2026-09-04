#!/usr/bin/env bash
# Install the dual-FR3 SONIC embodiment from this repo into ~/GR00T-WholeBodyControl
# (an editable install: PYTHONUSERBASE=~/env/pyuser-sonic imports that tree directly).
# Idempotent. Nothing is committed upstream; every mutation is logged in the WORKLOG.
#
#   bash harness/lane_b/install_gear_sonic.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WBC="${WBC:-$HOME/GR00T-WholeBodyControl}"
MENAGERIE="${MENAGERIE:-$HOME/code/upstream/mujoco_menagerie/franka_fr3/assets}"
WORKLOG="$HOME/agents/2026-09-01_franka-sonic/WORKLOG.md"
GS="$WBC/gear_sonic"
stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
changed=()

install_file() {  # src dst
  if ! cmp -s "$1" "$2"; then
    cp "$1" "$2"
    changed+=("$2")
    echo "installed $2"
  fi
}

# 1. MJCF + mesh symlink (meshdir="dual_fr3_assets" is relative to the xml)
MJ="$GS/data/assets/robot_description/mjcf"
install_file "$HERE/dual_fr3.xml" "$MJ/dual_fr3.xml"
if [ ! -e "$MJ/dual_fr3_assets" ]; then
  ln -s "$MENAGERIE" "$MJ/dual_fr3_assets"
  changed+=("$MJ/dual_fr3_assets -> $MENAGERIE")
  echo "symlinked $MJ/dual_fr3_assets"
fi

# 1b. URDF (Isaac side) + the same mesh symlink next to it
UR="$GS/data/assets/robot_description/urdf/dual_fr3"
mkdir -p "$UR"
install_file "$HERE/dual_fr3.urdf" "$UR/dual_fr3.urdf"
if [ ! -e "$UR/dual_fr3_assets" ]; then
  ln -s "$MENAGERIE" "$UR/dual_fr3_assets"
  changed+=("$UR/dual_fr3_assets -> $MENAGERIE")
  echo "symlinked $UR/dual_fr3_assets"
fi

# 2. robot config
install_file "$HERE/robots_dual_fr3.py" "$GS/envs/manager_env/robots/dual_fr3.py"
install_file "$HERE/robots_dual_fr3_stiff.py" "$GS/envs/manager_env/robots/dual_fr3_stiff.py"  # P5 jp19: deployment plant
install_file "$HERE/robots_dual_fr3_lanea.py" "$GS/envs/manager_env/robots/dual_fr3_lanea.py"  # P5 jp20: lane A plant, gravity off

# 3. experiment config (+exp=sonic_dual_fr3)
install_file "$HERE/sonic_dual_fr3.yaml" "$GS/config/exp/sonic_dual_fr3.yaml"
install_file "$HERE/sonic_dual_fr3_jp.yaml" "$GS/config/exp/sonic_dual_fr3_jp.yaml"
install_file "$HERE/sonic_dual_fr3_jp2.yaml" "$GS/config/exp/sonic_dual_fr3_jp2.yaml"
install_file "$HERE/sonic_dual_fr3_jp3.yaml" "$GS/config/exp/sonic_dual_fr3_jp3.yaml"
install_file "$HERE/sonic_dual_fr3_jp4.yaml" "$GS/config/exp/sonic_dual_fr3_jp4.yaml"
install_file "$HERE/sonic_dual_fr3_jp5.yaml" "$GS/config/exp/sonic_dual_fr3_jp5.yaml"
install_file "$HERE/sonic_dual_fr3_jp6.yaml" "$GS/config/exp/sonic_dual_fr3_jp6.yaml"
install_file "$HERE/sonic_dual_fr3_jp7.yaml" "$GS/config/exp/sonic_dual_fr3_jp7.yaml"
install_file "$HERE/sonic_dual_fr3_jp8.yaml" "$GS/config/exp/sonic_dual_fr3_jp8.yaml"
install_file "$HERE/sonic_dual_fr3_jp9.yaml" "$GS/config/exp/sonic_dual_fr3_jp9.yaml"
install_file "$HERE/sonic_dual_fr3_jp10.yaml" "$GS/config/exp/sonic_dual_fr3_jp10.yaml"
install_file "$HERE/sonic_dual_fr3_jp11.yaml" "$GS/config/exp/sonic_dual_fr3_jp11.yaml"
install_file "$HERE/sonic_dual_fr3_jp12.yaml" "$GS/config/exp/sonic_dual_fr3_jp12.yaml"
install_file "$HERE/sonic_dual_fr3_jp13.yaml" "$GS/config/exp/sonic_dual_fr3_jp13.yaml"
install_file "$HERE/sonic_dual_fr3_jp14.yaml" "$GS/config/exp/sonic_dual_fr3_jp14.yaml"
install_file "$HERE/sonic_dual_fr3_jp15.yaml" "$GS/config/exp/sonic_dual_fr3_jp15.yaml"
install_file "$HERE/sonic_dual_fr3_jp18.yaml" "$GS/config/exp/sonic_dual_fr3_jp18.yaml"
install_file "$HERE/sonic_dual_fr3_jp19.yaml" "$GS/config/exp/sonic_dual_fr3_jp19.yaml"
install_file "$HERE/sonic_dual_fr3_jp20.yaml" "$GS/config/exp/sonic_dual_fr3_jp20.yaml"
install_file "$HERE/sonic_dual_fr3_jp21.yaml" "$GS/config/exp/sonic_dual_fr3_jp21.yaml"
install_file "$HERE/sonic_dual_fr3_jp22.yaml" "$GS/config/exp/sonic_dual_fr3_jp22.yaml"
install_file "$HERE/sonic_dual_fr3_jp23.yaml" "$GS/config/exp/sonic_dual_fr3_jp23.yaml"
install_file "$HERE/sonic_dual_fr3_jp24.yaml" "$GS/config/exp/sonic_dual_fr3_jp24.yaml"  # P8 round-2 library
install_file "$HERE/sonic_dual_fr3_jp26.yaml" "$GS/config/exp/sonic_dual_fr3_jp26.yaml"  # P8 round-2 library
install_file "$HERE/sonic_dual_fr3_jp27.yaml" "$GS/config/exp/sonic_dual_fr3_jp27.yaml"  # P8 round-2 library

# 4. robot_mapping entry + order converter (minimal in-place patches, idempotent)
python3 - "$GS" <<'PY'
import sys, pathlib
gs = pathlib.Path(sys.argv[1])
changed = []

p = gs / "envs/manager_env/modular_tracking_env_cfg.py"
s = p.read_text()
if "dual_fr3" not in s:
    s = s.replace(
        "from gear_sonic.envs.manager_env.robots import g1, h2",
        "from gear_sonic.envs.manager_env.robots import dual_fr3, g1, h2", 1)
    entry = (
        '            "dual_fr3": {\n'
        '                "robot_cfg": dual_fr3.DUAL_FR3_CFG,\n'
        '                "action_scale": dual_fr3.DUAL_FR3_ACTION_SCALE,\n'
        '                "isaaclab_to_mujoco_mapping": dual_fr3.DUAL_FR3_ISAACLAB_TO_MUJOCO_MAPPING,\n'
        '            },\n'
    )
    anchor = '            "h2": {\n'
    assert anchor in s, "robot_mapping anchor not found"
    s = s.replace(anchor, entry + anchor, 1)
    p.write_text(s)
    changed.append(str(p))

p = gs / "trl/utils/order_converter.py"
s = p.read_text()
if "Dual_FR3Converter" not in s:
    cls = '''

class Dual_FR3Converter(IsaacLabMuJoCoConverter):
    """Dual-FR3 (franka-sonic lane B) joint/body order converter, IsaacLab <-> MuJoCo."""

    def __init__(self):
        from gear_sonic.envs.manager_env.robots.dual_fr3 import (
            DUAL_FR3_ISAACLAB_JOINTS,
            DUAL_FR3_ISAACLAB_TO_MUJOCO_BODY,
            DUAL_FR3_ISAACLAB_TO_MUJOCO_DOF,
            DUAL_FR3_MUJOCO_TO_ISAACLAB_BODY,
            DUAL_FR3_MUJOCO_TO_ISAACLAB_DOF,
        )

        self.JOINT_NAMES = DUAL_FR3_ISAACLAB_JOINTS
        self.DOF_MAPPINGS = {
            ("isaaclab", "mujoco"): DUAL_FR3_ISAACLAB_TO_MUJOCO_DOF,
            ("mujoco", "isaaclab"): DUAL_FR3_MUJOCO_TO_ISAACLAB_DOF,
        }
        self.BODY_MAPPINGS = {
            ("isaaclab", "mujoco"): DUAL_FR3_ISAACLAB_TO_MUJOCO_BODY,
            ("mujoco", "isaaclab"): DUAL_FR3_MUJOCO_TO_ISAACLAB_BODY,
        }

    VR_3POINTS_BODY_NAMES = ["base", "left_fr3_link7", "right_fr3_link7"]
    FOOT_BODY_NAMES = []
'''
    anchor = "\n\ndef load_qpos_from_csv"
    assert anchor in s, "order_converter anchor not found"
    s = s.replace(anchor, cls + anchor, 1)
    p.write_text(s)
    changed.append(str(p))
for c in changed:
    print("patched", c)
open("/dev/stdout", "w").flush()
PY

# 4b. P5 jp19: robot_mapping entry for the deployment-plant variant (idempotent)
python3 - "$GS" <<'PY2'
import sys, pathlib
gs = pathlib.Path(sys.argv[1])
p = gs / "envs/manager_env/modular_tracking_env_cfg.py"
s = p.read_text()
if "dual_fr3_stiff" not in s:
    s = s.replace(
        "from gear_sonic.envs.manager_env.robots import dual_fr3, g1, h2",
        "from gear_sonic.envs.manager_env.robots import dual_fr3, dual_fr3_stiff, g1, h2", 1)
    entry = (
        '            "dual_fr3_stiff": {\n'
        '                "robot_cfg": dual_fr3_stiff.DUAL_FR3_STIFF_CFG,\n'
        '                "action_scale": dual_fr3_stiff.DUAL_FR3_STIFF_ACTION_SCALE,\n'
        '                "isaaclab_to_mujoco_mapping": dual_fr3_stiff.DUAL_FR3_STIFF_ISAACLAB_TO_MUJOCO_MAPPING,\n'
        '            },\n'
    )
    anchor = '            "h2": {\n'
    assert anchor in s, "robot_mapping anchor not found"
    s = s.replace(anchor, entry + anchor, 1)
    p.write_text(s)
    print("patched", p, "(dual_fr3_stiff)")
s = p.read_text()
if "dual_fr3_lanea" not in s:
    s = s.replace(
        "from gear_sonic.envs.manager_env.robots import dual_fr3, dual_fr3_stiff, g1, h2",
        "from gear_sonic.envs.manager_env.robots import dual_fr3, dual_fr3_lanea, dual_fr3_stiff, g1, h2", 1)
    entry = (
        '            "dual_fr3_lanea": {\n'
        '                "robot_cfg": dual_fr3_lanea.DUAL_FR3_LANEA_CFG,\n'
        '                "action_scale": dual_fr3_lanea.DUAL_FR3_LANEA_ACTION_SCALE,\n'
        '                "isaaclab_to_mujoco_mapping": dual_fr3_lanea.DUAL_FR3_LANEA_ISAACLAB_TO_MUJOCO_MAPPING,\n'
        '            },\n'
    )
    anchor = '            "h2": {\n'
    assert anchor in s, "robot_mapping anchor not found"
    s = s.replace(anchor, entry + anchor, 1)
    p.write_text(s)
    print("patched", p, "(dual_fr3_lanea)")
PY2

# 5. import check
( cd "$WBC" && PYTHONUSERBASE="$HOME/env/pyuser-sonic" /isaac-sim/python.sh -c \
  "from gear_sonic.trl.utils.order_converter import Dual_FR3Converter; import mujoco; \
   m = mujoco.MjModel.from_xml_path('$MJ/dual_fr3.xml'); print('install check OK: nq', m.nq)" \
  2>&1 | grep -v '^\[' | tail -2 )

if [ "${#changed[@]}" -gt 0 ]; then
  printf -- '- %s  franka-sonic P2: installed dual-FR3 SONIC embodiment into %s (harness/lane_b/install_gear_sonic.sh): %s\n' \
    "$stamp" "$WBC" "$(IFS=';'; echo "${changed[*]}")" >> "$WORKLOG"
  echo "WORKLOG updated"
fi
echo "install done"
