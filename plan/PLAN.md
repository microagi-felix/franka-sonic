# PLAN — FR3 handover bake-off (condensed from the brain wiki, rev 3b)

Source of record: `microagi-felix-brain/wiki/fr3-handover-bakeoff-plan.md` +
`wiki/bakeoff-instance-harness-proposal.md`. This file is the executable
checklist; when the two disagree, the wiki wins and this file gets fixed.

## Thesis

One task (dual-FR3 block **handover**), one demo set, one eval, two control
stacks:

- **Lane A — GR00T direct.** GR00T N1.7 fine-tuned as `NEW_EMBODIMENT`,
  emitting joint targets for both arms.
- **Lane B — GR00T over SONIC.** A SONIC encoder/decoder pair is trained for a
  "dual-FR3 embodiment"; GR00T is fine-tuned to emit the **64-D SONIC token +
  2 grippers**, and the decoder turns each token into joint targets at 50 Hz
  (NVIDIA's own `UNITREE_G1_SONIC` pattern).

Prototype goal: **every stage runs end to end once with tiny numbers and both
lanes log a success rate.** A pipeline proof, not a result.

## Decisions already taken (do not relitigate)

- Rig = `angled` (repo default: 0.20 m apart, +0.14 m, ±45° roll) everywhere.
- No human teleop available -> demos are **scripted MimicGen sources**.
- The demo episodes **are** included in lane B's motion library.
- Pod `franka-sonic` is ours with all 8 GPUs; **lanes run in parallel** via
  `harness/gpus.py` (A fine-tune 2 ‖ SONIC RL 4 ‖ MimicGen/eval 1–2).
- Allocator hands out only devices with **< 1 GiB used** until infra confirms
  exclusivity (six of eight showed foreign memory on 2026-09-01).

## Comparability protocol (the reason this is a bake-off and not two demos)

1. **Collect episodes ONCE** on the plain robot: joint trajectories +
   Cartesian commands + 3 cameras.
2. **Derive lane B's labels offline** — run the trained SONIC encoder
   (robot-motion mode, needs the future window, which offline has) over each
   episode's joint trajectory -> one 64-D token per frame. Same episodes, same
   observations, same VLA base, same fine-tune budget/horizon, same eval; only
   the action space and controller differ, which *is* the experiment.
3. **Oracle replays calibrate both lanes** and are cheap — run them:
   - A-oracle: replay recorded joint actions on the recorded seed -> ≈100 % by
     construction.
   - B-oracle: encoder-labelled token stream through the decoder, no VLA ->
     lane B's ceiling. A low B-oracle means the *controller* lost, not the VLA.
   Report both next to the two policy numbers.
4. **Decoder lives in the policy server.** Both lanes present the identical
   contract to the sim: `{state16, 3 cams} -> [Lq7, Lg, Rq7, Rg]` joint
   targets, one registered joint client, same binding, same rubric, same
   `--rate`. Run both at `--rate 50`. The Cartesian/openpi route for lane A is
   an optional third run, not the baseline.

## Task and scoring

Block starts on the left pad -> left arm places it on the centre pad -> right
arm carries it to the end pad. Success = block at rest in the END zone, both
grippers open. Spawn randomisation ±6 cm / ±0.5 rad. Prompt = the
`franka_dual` binding's "hand the block from the left arm to the right".
Rubric = `handover_rubric()`, 6 ordered milestones (left reaches -> lifts ->
placed at centre -> right reaches -> lifts -> placed at end).

Per lane, over 20 seeded rollouts: per-milestone reach rate, full-success
rate, steps-to-success, action jerk, GPU-hours, plus the two oracle replays.
"Better" = full-success rate first, milestones second.

## Phases and gates

### P0 — environments + demo set (shared) — GATE `harness/gates/p0.sh`

- [ ] `env/bootstrap.sh` completes cleanly and is idempotent
- [ ] user in group `isaac-sim`; `/isaac-sim/python.sh` executable
- [ ] sim stack imports: `import isaacsim, isaaclab; import evaluation, tasks`
      under `PYTHONUSERBASE=~/env/pyuser-fr3 /isaac-sim/python.sh`
- [ ] `evaluation.eval --help` runs
- [ ] `harness/bakeoff.py run shared p0.smoke` produces
      `out/eval/eval_results.csv` (stub policy server + 1 rollout, 100 steps)
- [ ] `import gr00t` in `~/Isaac-GR00T/.venv`; `nvidia/GR00T-N1.7-3B` in the HF cache
- [ ] `import gear_sonic` under `PYTHONUSERBASE=~/env/pyuser-sonic` (WARN only)
- [ ] tmux present; `harness/gpus.py probe` runs

Then the demo set (P0 continued, gated by the replay check):

- [ ] `mimic/scripts/scripted_source_demos.py --headless --num_demos 10`
- [ ] `annotate_sources.py --auto` -> `generate_parallel.py --total 80 --procs 4`;
      record the keep-rate (target ≥ 50 %)
- [ ] `export_generated.py` -> `convert_sim_hdf5.py` (LeRobot v3) ->
      `convert_v3_to_v2.py` (GR00T v2) + `meta/modality.json`
      (state `joint_pos_l 0:7`, `joint_pos_r 7:14`, `grip 14:16`; action same
      split; video `top`, `wrist_left`, `wrist_right`)
- [ ] replay gate: one episode stepped through `…-JointPos-v0` reaches
      `handover_success = True`

### P1 — lane A: GR00T direct — GATE `harness/gates/p1.sh`

- [ ] `modality_config_dual_fr3.py`: video `[0]`, state current, action
      `delta_indices=range(16)`, arms `RELATIVE / NON_EEF`, grippers
      `ABSOLUTE`, horizon 16; registered under `EmbodimentTag.NEW_EMBODIMENT`
- [ ] `gr00t_finetune … --embodiment-tag NEW_EMBODIMENT --modality-config-path …
      --max-steps 2000 --save-steps 500 --global-batch-size 32` (2 GPUs)
- [ ] `open_loop_eval.py` on ckpt 500/1000/1500/2000 -> MSE falls
- [ ] joint policy server (GR00T -> joint chunk) + registered joint client
      closes the loop at `--rate 50`
- [ ] 20 rollouts evaluated, success rate logged

### P2 — lane B-1: dual-FR3 SONIC embodiment — GATE `harness/gates/p2.sh`

- [ ] `dual_fr3.xml` (2× menagerie `franka_fr3` at rig poses, fixed bases, no
      freejoint) + URDF/USD + `gear_sonic/…/robots/dual_fr3.py` (14 DoF, 2×7
      actuator groups, Isaac<->MuJoCo maps) + order converter
- [ ] `sonic_dual_fr3.yaml`: `anchor_body` = a fixed base body,
      `vr_3point_body` = [left hand, right hand, base], `reward_point_body` =
      the two hands; **disable** `foot_pos_xyz`, `feet_acc`,
      `undesired_contacts`; keep `ee_body_pos_adaptive`, tracking terms,
      `joint_limit`, `action_rate_l2`; robot-motion encoder only,
      `smpl_motion_file: dummy`, soma off
- [ ] body-name overrides pass a `num_envs=1` smoke
- [ ] motion library = GMR arms-only retarget of a ~1k-clip upper-body subset
      **plus the P0 demos as clips** + `_M` mirrors
- [ ] RL on 4 GPUs (≈2048–4096 envs), 1–2 h
- [ ] `eval_agent_trl.py +export_onnx_only=True` -> encoder + decoder ONNX
- [ ] decoder replays a demo clip with mean joint error < ~0.1 rad

### P3 — lane B-2: GR00T over SONIC — GATE `harness/gates/p3.sh`

- [ ] `label_tokens.py`: encoder ONNX over each demo's joint trajectory ->
      `token (T,64)`, subsampled to the policy rate (50 Hz encoder, nearest)
- [ ] dataset variant with `action = [token 64 | grip 2]`, modality config
      `ABSOLUTE / NON_EEF`; same fine-tune budget as P1
- [ ] policy server = GR00T -> token chunk -> decoder ONNX at 50 Hz with its
      own proprio history; clip `|t| <= 1.25` before the decoder; self-encode a
      "hold" token from the ready pose for resets
- [ ] 20 rollouts evaluated, success rate logged

### P4 — compare + decide — GATE `harness/gates/p4.sh`

- [ ] one table: success / milestones / steps-to-success / action jerk /
      wall-clock / GPU-h per lane, plus A-oracle and B-oracle
- [ ] decision memo: which lane scales, what the real run needs (demo count,
      full retarget, 4096 envs × 8 GPUs, longer fine-tune)

## Work-package parallelism (8 GPUs)

| lane | job | GPUs |
|---|---|---|
| shared | MimicGen generation, evals | 1–2 |
| A | GR00T fine-tune | 2 |
| B | SONIC RL | 4 |

Never a training rank and an Isaac eval on the same device.

## Contracts that bite

- Quaternions: the franka repo is **wxyz** internally, the policy **wire** is
  **xyzw**; `evaluation/codec.py` owns that conversion. The dataset `ee_pose`
  is **rot6d**, not a quaternion. The SONIC pkl boundary (`root_rot` wxyz) is
  ours to get right. Assert with a known pose at each boundary.
- Rates: recorder 30 Hz, SONIC decoder 50 Hz, GR00T chunk = horizon 16 at the
  training fps. Datasets at 30 Hz; hold the token between VLA frames to feed
  the decoder at 50.
- Token bounds `|t| <= 1.25` (FSQ grid) — clip the VLA output.
- Safety caps from `policy_convention.yaml`: 0.40 m / 0.60 rad per step,
  `control_hz: 15` global default.
- `evaluation.eval` refuses to resume a run folder with a different task/rate.
  One episode = one sample; always `--rollouts N`.

## Risks (ranked)

1. SONIC assumes a floating humanoid (anchor/feet/contact terms, root
   re-anchoring, `base_ang_vel` in the decoder obs) — P2 is the phase that can
   eat the schedule. `num_envs=1` smoke first; stub, do not rewrite.
2. Token space learned from locomotion-heavy motion may not cover tabletop
   reaching — the demos are in the library to prevent this. If the decoder
   still cannot replay demos, lane B is answered early (that is a finding).
3. FR3 J4 never straightens (−3.04 … −0.15 rad) -> retargeted swings saturate;
   keep the subset upper-body/reaching.
4. Two Isaac stacks + GR00T's venv on one box — separate user-sites, serialise
   GPU use per device.
5. 60–80 MimicGen episodes give noisy success rates for both lanes; the
   prototype compares pipelines, not policies.
6. LeRobot v3 -> v2 fidelity (video codec, `episodes.jsonl`, index columns) —
   validate with GR00T's loader before any training.
7. wxyz/xyzw/rot6d mix-ups at three boundaries.
8. Foreign memory on six of eight GPUs, no `nvidia-smi` — allocator rule (a).
