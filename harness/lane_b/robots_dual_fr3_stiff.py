"""Deployment-plant variant of the dual-FR3 SONIC embodiment (P5 "jp19", 2026-09-03).

Canonical copy: franka-sonic/harness/lane_b/robots_dual_fr3_stiff.py
Installed as:   ~/GR00T-WholeBodyControl/gear_sonic/envs/manager_env/robots/dual_fr3_stiff.py
Selected by:    manager_env.config.robot.type: dual_fr3_stiff  (robot_mapping entry added by
                harness/lane_b/install_gear_sonic.sh)

Why. The P5 decoders replay the demos to ~1 cm in the SONIC env (this module's parent,
`dual_fr3.py`) but put the hand 5-8 cm too HIGH at the grasp in lane A's handover env
(B-oracle traces, 2026-09-03 21:00 UTC): the policy learned the SONIC plant's response —
its targets sit 0.05-0.15 rad from where that plant settles (soft 200/20 wrists, armature,
200 Hz physics) — and lane A's stiffer plant lands ON the targets instead. Same asset, same
masses (URDF = MJCF), different actuators. This variant copies lane A's handover JointPos
env plant (franka-bimanual-isaac-sim: frankas_assets/specs/fr3_dual.py FR3_HIGH_PD_GROUPS,
frankas_assets/fr3.py FR3_CFG, tests/goldens/env_characterisation/*.json for the USD-authored
values) so the decoder trains on the plant it is deployed on:

    stiffness / damping    400 / 80 on ALL seven joints   (was 400/80 on 1-4, 200/20 on 5-7)
    armature               0.0                            (was 0.195 / 0.074, menagerie)
    velocity limits        2.62 (1-4), 5.26 / 4.18 / 5.26 (5-7)  (USD-authored; was 2.6 / 5.0)
    effort limits          87 / 12 Nm                     (unchanged)
    soft_joint_pos_limit   1.0                            (was 0.95)
    solver iterations      16 position / 1 velocity       (was 8 / 4)
    physics                100 Hz, decimation 2 -> 50 Hz control  (yaml: sim_dt 0.01, decimation 2;
                           was 200 Hz, decimation 4)

Everything else (asset, joint/body orders, action scale 0.25 rad per unit, ready pose,
joint-6 convention) is the parent's, re-exported below.
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg

from gear_sonic.envs.manager_env.robots import dual_fr3 as _base
from gear_sonic.envs.manager_env.robots.dual_fr3 import *  # noqa: F401,F403  (orders, scale, mapping)

FR3_STIFF_STIFFNESS = 400.0
FR3_STIFF_DAMPING = 80.0
FR3_STIFF_ARMATURE = 0.0
FR3_STIFF_VEL_SHOULDER = 2.62
FR3_STIFF_VEL_FOREARM = {".*_fr3_joint5": 5.26, ".*_fr3_joint6": 4.18, ".*_fr3_joint7": 5.26}

_ARTICULATION_PROPS_STIFF = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False,  # as the parent (decision 2026-09-03)
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
)

DUAL_FR3_STIFF_CFG = _base.DUAL_FR3_CFG.replace(
    spawn=_base.DUAL_FR3_CFG.spawn.replace(articulation_props=_ARTICULATION_PROPS_STIFF),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "fr3_shoulder": ImplicitActuatorCfg(
            joint_names_expr=[".*_fr3_joint[1-4]"],
            effort_limit_sim=_base.FR3_EFFORT_SHOULDER,
            velocity_limit_sim=FR3_STIFF_VEL_SHOULDER,
            stiffness=FR3_STIFF_STIFFNESS,
            damping=FR3_STIFF_DAMPING,
            armature=FR3_STIFF_ARMATURE,
        ),
        "fr3_forearm": ImplicitActuatorCfg(
            joint_names_expr=[".*_fr3_joint[5-7]"],
            effort_limit_sim=_base.FR3_EFFORT_FOREARM,
            velocity_limit_sim=FR3_STIFF_VEL_FOREARM,
            stiffness=FR3_STIFF_STIFFNESS,
            damping=FR3_STIFF_DAMPING,
            armature=FR3_STIFF_ARMATURE,
        ),
    },
)

# Re-exported for the robot_mapping entry (identical to the parent's).
DUAL_FR3_STIFF_ACTION_SCALE = _base.DUAL_FR3_ACTION_SCALE
DUAL_FR3_STIFF_ISAACLAB_TO_MUJOCO_MAPPING = _base.DUAL_FR3_ISAACLAB_TO_MUJOCO_MAPPING
