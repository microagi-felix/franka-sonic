# Patches to the upstream evaluation tree

`~/code/franka-bimanual-isaac-sim` is an upstream working tree; AGENTS.md forbids committing in
it, so a change that has to live there is kept here as a patch file and applied to the working
tree by hand. Each patch records the upstream commit it was cut against.

| patch | upstream base | what |
| --- | --- | --- |
| `2026-09-05_fresh_first_obs.patch` | `14f0d8a` | P11: `--fresh-first-obs`, an opt-in render + camera-sensor refresh after `env.reset()`, so an episode's first image is its own rather than the previous episode's last frame. Isaac Lab's `ManagerBasedEnvCfg.num_rerenders_on_reset` defaults to 0 and its own docstring says sensor data after a reset "will be stale". Off by default, so every run recorded in rounds 1-3 stays reproducible. |

Apply with `git -C ~/code/franka-bimanual-isaac-sim apply <patch>`; check with
`git -C ~/code/franka-bimanual-isaac-sim diff --stat`.
