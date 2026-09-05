# Patches to the upstream evaluation tree

`~/code/franka-bimanual-isaac-sim` is an upstream working tree; AGENTS.md forbids committing in
it, so a change that has to live there is kept here as a patch file and applied to the working
tree by hand. Each patch records the upstream commit it was cut against.

| patch | upstream base | what |
| --- | --- | --- |
| `2026-09-05_fresh_first_obs.patch` | `14f0d8a` | P11: `--fresh-first-obs`, an opt-in render + camera-sensor refresh after `env.reset()`, so an episode's first image is its own rather than the previous episode's last frame. Isaac Lab's `ManagerBasedEnvCfg.num_rerenders_on_reset` defaults to 0 and its own docstring says sensor data after a reset "will be stale". Off by default, so every run recorded in rounds 1-3 stays reproducible. |

Apply with `git -C ~/code/franka-bimanual-isaac-sim apply <patch>`; check with
`git -C ~/code/franka-bimanual-isaac-sim diff --stat`.

## `2026-09-05_fresh_first_obs.patch` was TESTED AND REJECTED — do not enable it

Validated under load on 2026-09-05 19:48 as a paired test: three 20-rollout runs of lane B
`checkpoint-10000` **with** the flag against three **without**, launched together on six
devices, same checkpoint, same `--seed 12345`.

| arm | runs | successes | dead episodes |
| --- | --- | --- | --- |
| `--fresh-first-obs` | `lane_b/2026-09-05_eval-17/-18/-19` | **4/20, 5/20, 5/20** | 0, 0, 0 |
| without it | `lane_b/2026-09-05_eval-20/-21/-22` | **19/20, 19/20, 9/20** | 0, 0, 5 |

The three flagged runs produced an almost identical episode-by-episode pattern
(`00010000100001010000`, `00011000100001010000`, `00011000100001010000`), i.e. a
deterministic and much worse policy — not a noisier one. The mechanism the patch targets is
real, but the implementation changes more than the first frame: two extra `sim.render()`
calls plus `sensor.update(0.0, force_recompute=True)` most plausibly offset the annotator
buffers for the rest of the episode, so every later capture is one frame further behind.

**Kept for the record only. No P11 row carries it.** A later attempt should try a single
`sim.render()` without the forced sensor update, or a warm-up capture discarded before the
first `client.infer`, and re-run this same paired test under load.
