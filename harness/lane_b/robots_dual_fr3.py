# Dual-FR3 robot configuration for the SONIC (gear_sonic) tracking environment.
#
# Canonical copy: franka-sonic/harness/lane_b/robots_dual_fr3.py
# Installed as:   ~/GR00T-WholeBodyControl/gear_sonic/envs/manager_env/robots/dual_fr3.py
#                 (harness/lane_b/install_gear_sonic.sh; never committed upstream)
#
# Two menagerie franka_fr3 arms on the angled rig (harness/lane_b/dual_fr3.xml), one
# fixed-base articulation, 14 DoF, no grippers. Follows robots/h2.py: name lists,
# IsaacLab <-> MuJoCo index maps, ArticulationCfg, action scale. The index maps are
# derived from the two name lists instead of being typed by hand; the Isaac order was
# read off the imported articulation by harness/lane_b/probe_isaac_names.py and is
# re-asserted there.

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

ASSET_DIR = "gear_sonic/data/assets"

# ------------------------------------------------------------------ MuJoCo order (dual_fr3.xml)
# Body order = depth-first document order of <body> (Humanoid_Batch.from_mjcf), root first.
DUAL_FR3_MUJOCO_BODIES = (
    ["base"]
    + [f"left_fr3_link{i}" for i in range(8)]
    + [f"right_fr3_link{i}" for i in range(8)]
)
# DoF order = document order of <joint> — this is the order of `dof` in the motion pkls.
DUAL_FR3_MUJOCO_JOINTS = [f"left_fr3_joint{i}" for i in range(1, 8)] + [
    f"right_fr3_joint{i}" for i in range(1, 8)
]

# ------------------------------------------------------------------ IsaacLab order (probed)
# PhysX/IsaacLab enumerate the imported articulation breadth-first: the two arms
# interleave level by level. Verified by probe_isaac_names.py on 2026-09-03.
DUAL_FR3_ISAACLAB_JOINTS = ["base"] + [
    f"{side}_fr3_link{i}" for i in range(8) for side in ("left", "right")
]
DUAL_FR3_ISAACLAB_JOINT_NAMES = [
    f"{side}_fr3_joint{i}" for i in range(1, 8) for side in ("left", "right")
]

# output[i] = input[mapping[i]]  (IsaacLabMuJoCoConverter.convert semantics)
DUAL_FR3_ISAACLAB_TO_MUJOCO_DOF = [
    DUAL_FR3_ISAACLAB_JOINT_NAMES.index(j) for j in DUAL_FR3_MUJOCO_JOINTS
]
DUAL_FR3_MUJOCO_TO_ISAACLAB_DOF = [
    DUAL_FR3_MUJOCO_JOINTS.index(j) for j in DUAL_FR3_ISAACLAB_JOINT_NAMES
]
DUAL_FR3_ISAACLAB_TO_MUJOCO_BODY = [
    DUAL_FR3_ISAACLAB_JOINTS.index(b) for b in DUAL_FR3_MUJOCO_BODIES
]
DUAL_FR3_MUJOCO_TO_ISAACLAB_BODY = [
    DUAL_FR3_MUJOCO_BODIES.index(b) for b in DUAL_FR3_ISAACLAB_JOINTS
]

DUAL_FR3_ISAACLAB_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": DUAL_FR3_ISAACLAB_JOINTS,
    "isaaclab_to_mujoco_dof": DUAL_FR3_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": DUAL_FR3_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": DUAL_FR3_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": DUAL_FR3_MUJOCO_TO_ISAACLAB_BODY,
}

# ------------------------------------------------------------------ actuators
# Effort limits from the franka repo's frankas_assets/fr3.py (87 Nm joints 1-4, 12 Nm
# joints 5-7); PD gains from its FR3_HIGH_PD_CFG tier (stiff position tracking, so the
# policy's joint targets stay close to the joint positions it actually reaches).
FR3_EFFORT_SHOULDER = 87.0
FR3_EFFORT_FOREARM = 12.0
FR3_STIFFNESS_SHOULDER = 400.0
FR3_DAMPING_SHOULDER = 80.0
FR3_STIFFNESS_FOREARM = 200.0
FR3_DAMPING_FOREARM = 20.0
# FR3 joint velocity limits (rad/s) from the Franka datasheet, rounded down.
FR3_VEL_SHOULDER = 2.6
FR3_VEL_FOREARM = 5.0

# JOINT 6 CONVENTION (see make_dual_fr3_xml.py): the SONIC embodiment's joint 6 is the FR3
# joint 6 minus J6_OFFSET, so the demos' 4.5 rad excursions stay inside (-pi, pi) for the
# motion library's axis-angle round trip. Apply it at every FR3 <-> SONIC boundary.
J6_OFFSET = 2.5307

# FR3 "ready" pose (frankas_assets/fr3.py FR3_CFG init_state) — also dual_fr3.xml's `home`.
FR3_READY_POSE = {
    ".*_fr3_joint1": 0.0,
    ".*_fr3_joint2": -0.569,
    ".*_fr3_joint3": 0.0,
    ".*_fr3_joint4": -2.810,
    ".*_fr3_joint5": 0.0,
    ".*_fr3_joint6": 3.037 - J6_OFFSET,
    ".*_fr3_joint7": 0.741,
}

# The MJCF is converted to USD ONCE by harness/lane_b/probe_isaac_names.py (Isaac Lab's
# MjcfConverter needs the `isaacsim.asset.importer.mjcf` extension, which the training
# app's Kit experience does not enable; the probe enables it explicitly). Training spawns
# the converted USD, so the importer is never touched inside train_agent_trl.py.
DUAL_FR3_USD_DIR = f"{ASSET_DIR}/robot_description/usd/dual_fr3"
DUAL_FR3_USD = f"{DUAL_FR3_USD_DIR}/dual_fr3.usd"

_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    retain_accelerations=False,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=1000.0,
    max_depenetration_velocity=1.0,
)
_ARTICULATION_PROPS = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=True,
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=4,
)

# Conversion spawner (probe only): MJCF -> instanceable USD under DUAL_FR3_USD_DIR.
DUAL_FR3_MJCF_SPAWN = sim_utils.MjcfFileCfg(
    asset_path=f"{ASSET_DIR}/robot_description/mjcf/dual_fr3.xml",
    usd_dir=DUAL_FR3_USD_DIR,
    usd_file_name="dual_fr3.usd",
    force_usd_conversion=True,
    fix_base=True,
    import_inertia_tensor=True,
    import_sites=False,
    self_collision=True,
    activate_contact_sensors=True,
    rigid_props=_RIGID_PROPS,
    articulation_props=_ARTICULATION_PROPS,
)

DUAL_FR3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=DUAL_FR3_USD,
        activate_contact_sensors=True,
        rigid_props=_RIGID_PROPS,
        articulation_props=_ARTICULATION_PROPS,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),  # the rig's setup-local origin sits on the env origin
        joint_pos=dict(FR3_READY_POSE),
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "fr3_shoulder": ImplicitActuatorCfg(
            joint_names_expr=[".*_fr3_joint[1-4]"],
            effort_limit_sim=FR3_EFFORT_SHOULDER,
            velocity_limit_sim=FR3_VEL_SHOULDER,
            stiffness=FR3_STIFFNESS_SHOULDER,
            damping=FR3_DAMPING_SHOULDER,
            armature=0.195,  # menagerie fr3 joint armature
        ),
        "fr3_forearm": ImplicitActuatorCfg(
            joint_names_expr=[".*_fr3_joint[5-7]"],
            effort_limit_sim=FR3_EFFORT_FOREARM,
            velocity_limit_sim=FR3_VEL_FOREARM,
            stiffness=FR3_STIFFNESS_FOREARM,
            damping=FR3_DAMPING_FOREARM,
            armature=0.074,
        ),
    },
)

# ------------------------------------------------------------------ action scale
# SONIC's `0.25 * effort / stiffness` formula gives 0.054 / 0.015 rad per unit action here
# (stiff PD, small effort limits), which with action_clip_value=20 could not even reach
# the demos' joint-4 excursions from the ready pose. A flat 0.25 rad per unit action
# (G1's shoulder joints sit at 0.35-0.44) covers +-5 rad at the clip.
DUAL_FR3_ACTION_SCALE = {name: 0.25 for name in DUAL_FR3_MUJOCO_JOINTS}
