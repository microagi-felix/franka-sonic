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

# Isaac spawns the URDF derived from the MJCF (harness/lane_b/make_dual_fr3_urdf.py), the
# way G1/H2 are spawned. The Isaac Sim 5.1 MJCF importer is not usable on this pod (it is
# not enabled by the headless Kit experience and, enabled by hand, wrote empty USD layers
# and hung, 2026-09-03). Paths are made absolute at import time because the converters
# resolve sublayers relative to the process cwd otherwise; cwd must be the
# GR00T-WholeBodyControl root anyway (ASSET_DIR is relative).
import os as _os

DUAL_FR3_URDF = _os.path.abspath(f"{ASSET_DIR}/robot_description/urdf/dual_fr3/dual_fr3.urdf")
DUAL_FR3_USD_DIR = _os.path.abspath(f"{ASSET_DIR}/robot_description/usd/dual_fr3_urdf")

_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    retain_accelerations=False,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=1000.0,
    max_depenetration_velocity=1.0,
)
# No self-collision (decision, 2026-09-03): with convex-hull colliders of the FR3 meshes the
# folded ready pose (joint 4 = -2.81) interpenetrates forearm/upper-arm hulls and the contact
# pushes joint 4 open by 0.5-0.76 rad while "holding" the pose (probe_isaac_names.py). The
# demos are collision-free by construction; inter-arm contact is simply not modelled here.
_ARTICULATION_PROPS = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=False,
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=4,
)

DUAL_FR3_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=DUAL_FR3_URDF,
        usd_dir=DUAL_FR3_USD_DIR,
        usd_file_name="dual_fr3.usd",
        fix_base=True,
        merge_fixed_joints=False,  # keep base / link0 as bodies: 17 bodies like the MJCF
        replace_cylinders_with_capsules=False,
        collider_type="convex_hull",
        self_collision=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
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
