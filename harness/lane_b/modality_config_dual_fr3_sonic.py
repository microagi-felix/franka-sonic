# Lane B-2 modality config — dual FR3 handover, GR00T over SONIC (motion tokens).
#
# Registers a GR00T N1.7 modality config under EmbodimentTag.NEW_EMBODIMENT as
# an *import side effect*, exactly like examples/SO100/so100_config.py and its
# lane A twin harness/lane_a/modality_config_dual_fr3.py. Both
# gr00t/experiment/launch_finetune.py (load_modality_config) and
# gr00t/data/stats.py append this file's parent directory to sys.path and
# import it by module stem, so it must stay importable as a plain module with
# no package-relative imports.
#
#   ~/Isaac-GR00T/.venv/bin/python gr00t/experiment/launch_finetune.py \
#       --embodiment-tag NEW_EMBODIMENT \
#       --modality-config-path <repo>/harness/lane_b/modality_config_dual_fr3_sonic.py ...
#
# WARNING: this file and harness/lane_a/modality_config_dual_fr3.py both claim
# EmbodimentTag.NEW_EMBODIMENT, and register_modality_config() *asserts* the tag
# is still free ("Embodiment tag ... already registered"). Exactly one of the
# two may be imported per process — never point a job at both.
#
# Shape of this config: it mirrors NVIDIA's own "unitree_g1_sonic" entry in
# gr00t/configs/data/embodiment_configs.py field for field (40-step horizon, a
# leading `motion_token` key, ABSOLUTE / NON_EEF / DEFAULT for every action
# group), with the G1's `ego_view` replaced by our three cameras and its
# `left_hand_joints` / `right_hand_joints` by the two Franka Hand commands.
#
# Rev 3c contract (plan/PLAN.md, "fine-tune alignment protocol") — identical to
# lane A except for what the action columns *mean*:
#
#   * data rate            50 Hz (recorder --rate 50)
#   * action horizon       40 steps = 0.8 s at 50 Hz (delta_indices range(40))
#   * replan               every 20 steps = 2.5 Hz (evaluation.eval --replan-every 20)
#   * action encoding      absolute SONIC motion token + absolute gripper
#                          commands; NON_EEF (neither is an end-effector pose)
#   * language key         annotation.human.task_description
#
# Layout of the 66-D ACTION vector (dataset order, see
# harness/lane_b/make_sonic_dataset.py and meta/modality.json):
#
#   motion_token   0:64   64-D SONIC FSQ motion token. The FSQ codebook is a
#                         1/16 grid inside [-1, 1]: every component is one of
#                         -1, -15/16, ..., 15/16 (dequantised token, not an
#                         integer code index). The SONIC decoder turns one
#                         token into the low-level joint targets, so GR00T
#                         never regresses joint angles in this lane.
#   left_grip     64:65   left  Franka Hand command (0 = open, 1 = closed)
#   right_grip    65:66   right Franka Hand command (0 = open, 1 = closed)
#
# The grip columns carry exactly the same values (and the same 0/1 convention)
# as lane A's `grip` 14:16 columns — they are copied straight across by the
# dataset builder, so a lane A and a lane B checkpoint can be compared on the
# gripper channel without any rescaling.
#
# Layout of the 16-D STATE vector — unchanged from lane A (the observation side
# of the dataset is copied byte-for-byte):
#
#   joint_pos_l  0:7    left  FR3 joints 1..7      (rad)
#   joint_pos_r  7:14   right FR3 joints 1..7      (rad)
#   grip        14:16   [left, right] gripper      (0 = open, 1 = closed)
#
# NOTE the dataset order is NOT the sim's wire order ([Lq7, Lg, Rq7, Rg]);
# lane B's policy server owns that conversion, after the SONIC decoder.

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
# ABSOLUTE for all three, exactly as in unitree_g1_sonic: a motion token is an
# absolute point in the FSQ codebook (relative chunking of a quantised latent
# is meaningless), and the grippers are absolute open/close commands. NON_EEF
# because none of these are end-effector poses — that also keeps
# gr00t/data/stats.py's relative-stats pass a no-op (it only walks RELATIVE
# keys), so meta/relative_stats.json stays empty like lane A's.
_ABSOLUTE_TOKEN = ActionConfig(
    rep=ActionRepresentation.ABSOLUTE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)

dual_fr3_sonic_config = {
    # Video: current frame only. Keys must match "video" in meta/modality.json.
    # Same three cameras as lane A (unitree_g1_sonic has the single "ego_view").
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["top", "wrist_left", "wrist_right"],
    ),
    # State: current proprioception, unchanged from lane A. Keys must match
    # "state" in meta/modality.json.
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["joint_pos_l", "joint_pos_r", "grip"],
    ),
    # Action: 40-step token chunk (0.8 s at 50 Hz), replanned every 20 steps
    # (2.5 Hz) — the same horizon unitree_g1_sonic uses.
    "action": ModalityConfig(
        delta_indices=list(range(40)),
        modality_keys=["motion_token", "left_grip", "right_grip"],
        action_configs=[_ABSOLUTE_TOKEN, _ABSOLUTE_TOKEN, _ABSOLUTE_TOKEN],
    ),
    # Language: the handover instruction, read from task_index via meta/tasks.jsonl.
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(dual_fr3_sonic_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
