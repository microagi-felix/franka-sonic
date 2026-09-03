# Lane A modality config — dual FR3 handover, GR00T direct (joint targets).
#
# Registers a GR00T N1.7 modality config under EmbodimentTag.NEW_EMBODIMENT as
# an *import side effect*, exactly like examples/SO100/so100_config.py. Both
# gr00t/experiment/launch_finetune.py (load_modality_config) and
# gr00t/data/stats.py append this file's parent directory to sys.path and
# import it by module stem, so it must stay importable as a plain module with
# no package-relative imports.
#
#   ~/Isaac-GR00T/.venv/bin/python gr00t/experiment/launch_finetune.py \
#       --embodiment-tag NEW_EMBODIMENT \
#       --modality-config-path <repo>/harness/lane_a/modality_config_dual_fr3.py ...
#
# Rev 3c contract (plan/PLAN.md, "fine-tune alignment protocol"):
#
#   * data rate            50 Hz (recorder --rate 50)
#   * action horizon       40 steps = 0.8 s at 50 Hz (delta_indices range(40))
#   * replan               every 20 steps = 2.5 Hz (evaluation.eval --replan-every 20)
#   * action encoding      absolute joint targets for both arms + absolute
#                          gripper commands; NON_EEF (joint space, not a pose)
#   * language key         annotation.human.task_description
#
# Layout of the 16-D state/action vectors (dataset order, see
# harness/data/convert_hdf5_to_gr00t_v2.py and meta/modality.json):
#
#   joint_pos_l  0:7    left  FR3 joints 1..7      (rad)
#   joint_pos_r  7:14   right FR3 joints 1..7      (rad)
#   grip        14:16   [left, right] gripper      (0 = open, 1 = closed)
#
# NOTE the dataset order is NOT the sim's wire order ([Lq7, Lg, Rq7, Rg]);
# harness/lane_a/serve_gr00t_joint.py owns that conversion.

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


# One ActionConfig per action modality key, in the same order as modality_keys.
# ABSOLUTE for all three: the arms are absolute joint *targets* (the recorded
# differential-IK targets), and the grippers are absolute open/close commands.
# NON_EEF because none of these are end-effector poses.
_ABSOLUTE_JOINT = ActionConfig(
    rep=ActionRepresentation.ABSOLUTE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)

dual_fr3_config = {
    # Video: current frame only. Keys must match "video" in meta/modality.json.
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["top", "wrist_left", "wrist_right"],
    ),
    # State: current proprioception. Keys must match "state" in meta/modality.json.
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["joint_pos_l", "joint_pos_r", "grip"],
    ),
    # Action: 40-step chunk (0.8 s at 50 Hz), replanned every 20 steps (2.5 Hz).
    "action": ModalityConfig(
        delta_indices=list(range(40)),
        modality_keys=["joint_pos_l", "joint_pos_r", "grip"],
        action_configs=[_ABSOLUTE_JOINT, _ABSOLUTE_JOINT, _ABSOLUTE_JOINT],
    ),
    # Language: the handover instruction, read from task_index via meta/tasks.jsonl.
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(dual_fr3_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
