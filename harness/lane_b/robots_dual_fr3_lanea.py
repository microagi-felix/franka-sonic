"""Lane A's handover plant for the dual-FR3 SONIC embodiment (P5 "jp20", 2026-09-03).

Canonical copy: franka-sonic/harness/lane_b/robots_dual_fr3_lanea.py
Installed as:   ~/GR00T-WholeBodyControl/gear_sonic/envs/manager_env/robots/dual_fr3_lanea.py
Selected by:    manager_env.config.robot.type: dual_fr3_lanea

= robots_dual_fr3_stiff.py (lane A's 400/80 gains on all joints, armature 0, USD velocity
limits, soft limit 1.0, solver 16/1) PLUS the one thing that matters: lane A spawns its FR3
arms with **gravity disabled** (franka-bimanual-isaac-sim frankas_assets/fr3.py
`FR3_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True`, frankas_assets/specs/fr3_dual.py
`disable_gravity=True` on every chain), i.e. a gravity-compensated arm. The SONIC plant had
gravity ON, so every P5 decoder learned to command targets ABOVE the reference to cancel the
sag (SONIC env: measured - target = +0.05..0.15 rad on the shoulder joints); in the
gravity-free handover env the arm lands ON those targets and the hand ends 5-8 cm too high
at the grasp (B-oracle traces vs demo joints, 21:00 UTC). The stiffness/rate change alone
(dual_fr3_stiff, decoder_replay-35) left the jp18 replay at 0.060 rad; gravity is the gap.
"""
from __future__ import annotations

from gear_sonic.envs.manager_env.robots import dual_fr3_stiff as _stiff
from gear_sonic.envs.manager_env.robots.dual_fr3_stiff import *  # noqa: F401,F403

_RIGID_PROPS_NOGRAV = _stiff._base._RIGID_PROPS.replace(disable_gravity=True)

DUAL_FR3_LANEA_CFG = _stiff.DUAL_FR3_STIFF_CFG.replace(
    spawn=_stiff.DUAL_FR3_STIFF_CFG.spawn.replace(rigid_props=_RIGID_PROPS_NOGRAV),
)
DUAL_FR3_LANEA_ACTION_SCALE = _stiff.DUAL_FR3_STIFF_ACTION_SCALE
DUAL_FR3_LANEA_ISAACLAB_TO_MUJOCO_MAPPING = _stiff.DUAL_FR3_STIFF_ISAACLAB_TO_MUJOCO_MAPPING
