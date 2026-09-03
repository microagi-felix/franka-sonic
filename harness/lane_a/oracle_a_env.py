"""JointPos handover env for the A-oracle: the block spawns where the replayed demo had it.

`Isaac-Stack-Cube-DualFranka-JointPos-OracleA-v0` = `DualFrankaJointPosEnvCfg` with ONE change:
the reset event `init_cube` writes the recorded initial cube pose of the episode being replayed
(from `oracle_a_table.EPISODES`) instead of sampling a random spawn. Everything else — the
absolute joint-position action space, the rig, the cameras, the rubric — is the stock JointPos
env `evaluation.eval` drives, so the oracle's `eval_results.csv` is directly comparable with the
policy's. Imported by name (`oracle_a_env:OracleAJointPosEnvCfg`) through the gym registration
`eval_oracle_a.py` makes; must be importable as the top-level module `oracle_a_env`.
"""

import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import oracle_a_table
from frankas_assets.specs.task import verify_task_contract
from tasks.stack_fr3.dual_stack_env_cfg import DualFrankaJointPosEnvCfg

ORACLE_TASK_ID = "Isaac-Stack-Cube-DualFranka-JointPos-OracleA-v0"


def set_cube_pose_from_table(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("cube_1")):
    """Reset event: place the block at the replayed demo's recorded initial pose (env frame)."""
    if env_ids is None:
        return
    k = oracle_a_table.ENV_RESETS
    oracle_a_table.ENV_RESETS += 1
    episodes = oracle_a_table.EPISODES
    idx = min(max(0, oracle_a_table.START_EPISODE + k - 1), len(episodes) - 1)
    pose = torch.as_tensor(episodes[idx]["cube_pose"], dtype=torch.float32, device=env.device).reshape(1, 7)
    asset = env.scene[asset_cfg.name]
    for cur_env in env_ids.tolist():
        p = pose.clone()
        p[:, :3] += env.scene.env_origins[cur_env, :3]
        ids = torch.tensor([cur_env], device=env.device)
        asset.write_root_pose_to_sim(p, env_ids=ids)
        asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=ids)
    print(f"[oracle-a] env reset #{k}: cube at recorded pose of table episode {idx} "
          f"({episodes[idx]['name']}): {[round(float(x), 3) for x in pose[0, :3]]}", flush=True)


@configclass
class OracleAJointPosEnvCfg(DualFrankaJointPosEnvCfg):
    task_id: str = ORACLE_TASK_ID

    def __post_init__(self):
        super().__post_init__()
        self.events.init_cube = EventTerm(
            func=set_cube_pose_from_table,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg("cube_1")},
        )
        verify_task_contract(self, OracleAJointPosEnvCfg)
