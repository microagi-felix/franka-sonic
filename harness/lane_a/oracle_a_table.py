"""Isaac-free shared state for the A-oracle replay: the recorded episodes and a reset counter.

Filled by `eval_oracle_a.py` BEFORE the simulator boots; read by `oracle_a_env.py` (the cube
spawn event, which runs inside every `env.reset`) and by the in-process `OracleA` client
(which streams the recorded joint actions). Both key the same table the same way:

    reset #0  = evaluation.eval's startup `env.reset(seed)`      -> table[start_episode] (unused)
    reset #k  = episode start_episode + k - 1                   -> table[that episode]

`evaluation.eval` calls `client.reset()` once per episode (after the env reset), so the client
counts from start_episode directly.
"""

EPISODES: list = []  # [{name, action16 (T,16) float32 JointPos-env layout, cube_pose (7,)}]
START_EPISODE: int = 0
ENV_RESETS: int = 0
