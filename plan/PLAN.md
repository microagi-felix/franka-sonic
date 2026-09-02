# PLAN — FR3 handover bake-off (condensed from the brain wiki, rev 3c)

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
  exclusivity (six of eight showed foreign memory on 2026-09-01; on 2026-09-03
  only devices 5 and 7 were allocatable — plan for **2**, take more if the
  allocator offers them).

## Fine-tune alignment protocol (rev 3c — both lanes identical but the action keys)

Verified against NVIDIA's own SONIC post-train (`GR00T-WholeBodyControl`
`docs/source/tutorials/vla_workflow.md`, `vla_inference.md`,
`gear_sonic/data/features_sonic_vla.py`) and Isaac-GR00T @ab88b50. NVIDIA's
SONIC post-train **is the ordinary GR00T fine-tune** — same script, same
defaults; only the embodiment tag and the modality config differ.

| knob | lane A | lane B |
|---|---|---|
| tag / projector | `NEW_EMBODIMENT` (id 10, fresh action head) | `NEW_EMBODIMENT` (id 10, fresh action head) |
| script | `gr00t/experiment/launch_finetune.py` | same |
| hyperparameters | `--num-gpus 2 --max-steps 2000 --save-steps 500 --global-batch-size 32 --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 --dataloader-num-workers 4`; default LR/optimizer, no LoRA | same |
| data rate | **50 Hz** (recorder `--rate 50`) | same frames, second action table |
| action horizon | **40** (`delta_indices = range(40)`) | 40 |
| action encoding | `left_arm 7 \| left_grip 1 \| right_arm 7 \| right_grip 1`, **ABSOLUTE**, `NON_EEF` | `motion_token 64 \| left_grip 1 \| right_grip 1`, **ABSOLUTE**, `NON_EEF` |
| state keys | joints 14 + grippers 2 (+ EEF poses if present) | same |
| video | `top` + `wrist_left` + `wrist_right`, one resolution | same |
| language | `annotation.human.task_description` = "hand the block from the left arm to the right" | same |
| inference | **2.5 Hz replan**, 40-step chunk played at 50 Hz (`--rate 50 --replan-every 20`) | same |
| labels | recorded joint targets | encoder over the recorded joint trajectory (offline) |

Consequences that are easy to get wrong:

- The recorder runs at **50 Hz** (`scripted_source_demos.py --rate 50`), not the
  script's 30 Hz default. Both datasets are the **same frames** with two action
  tables.
- ABSOLUTE in **both** lanes: lane A's arms are absolute joint targets, so the
  only difference to lane B is the action space itself.
- 2000 steps for both lanes in the prototype (10 % of NVIDIA's 20 000).
  **Never compare lanes trained for different step counts.**
- Rev 2's "horizon 16, RELATIVE arms, 30 Hz" is superseded — if you find those
  numbers anywhere, they are stale.

## Comparability protocol (the reason this is a bake-off and not two demos)

1. **Collect episodes ONCE** on the plain robot at 50 Hz: joint trajectories +
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
   targets, one joint client, same binding, same rubric, same `--rate 50`. The
   Cartesian/openpi route for lane A is an optional third run, not the baseline.
   Prefer making the **server** speak the wire the stock `ZmqAct` client
   already speaks over patching the franka repo's client registry (that repo is
   upstream: never commit in it).

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

## Artifact conventions (gates read these paths — keep to them)

Run folders come from `harness/bakeoff.py` (AGENTS.md rule e), under
`~/runs/franka-sonic/<lane>/<YYYY-MM-DD>_<stage>/`. The gates search by name
substring, so the *stage* names matter:

| phase | run folder (stage substring) | artifact the gate reads |
|---|---|---|
| P0 | `shared/…_p0.smoke` | `out/eval/eval_results.csv` |
| P0 | `shared/…_demos` | `out/*.hdf5` (sources + generated) |
| P1 | `shared/…_dataset` or `lane_a/…_dataset` | `out/gr00t_v2/meta/modality.json` |
| P1 | `lane_a/…_finetune` | `out/checkpoints/checkpoint-2000/` |
| P1 | `lane_a/…_open_loop` | `logs/run.log` (+ `out/open_loop_eval.json`) |
| P1 | `lane_a/…_eval` | `out/eval/eval_results.csv` ≥ 20 episodes |
| P1 | `lane_a/…_oracle_a` | `out/eval/eval_results.csv` |
| P2 | repo `harness/lane_b/` | `dual_fr3.xml`, `robots_dual_fr3.py`, `sonic_dual_fr3.yaml` |
| P2 | `lane_b/…_motion_lib` | `out/motions/*.pkl` (≥ 100) |
| P2 | `lane_b/…_export_onnx` | `out/model_encoder.onnx`, `out/model_decoder.onnx` |
| P2 | `lane_b/…_decoder_replay` | `out/replay.json` with `mean_joint_error_rad` |
| P3 | `lane_b/…_label_tokens` | `out/gr00t_v2_sonic/meta/modality.json` |
| P3 | `lane_b/…_finetune` | `out/checkpoints/checkpoint-2000/` |
| P3 | `lane_b/…_eval` | `out/eval/eval_results.csv` ≥ 20 episodes |
| P3 | `lane_b/…_oracle_b` | `out/eval/eval_results.csv` |
| P4 | repo | `plan/REPORT.md` with a table naming both lanes |

Code lives in the repo (`harness/lane_a/`, `harness/lane_b/`, `harness/data/`,
`harness/report/`), never in a run folder; data and logs live in run folders,
never in the repo.

## Phases and gates

### P0 — environments + demo set (shared) — GATE `harness/gates/p0.sh`

- [ ] `env/bootstrap.sh` completes cleanly and is idempotent
- [ ] user in group `isaac-sim`; `/isaac-sim/python.sh` executable; Kit's
      portable-mode dirs writable (`/isaac-sim/kit/{cache,logs,data}`,
      `/isaac-sim/extscache` — see AGENTS.md rule j exception)
- [ ] sim stack imports: `import isaacsim, isaaclab; import evaluation, tasks`
      under `PYTHONUSERBASE=~/env/pyuser-fr3 /isaac-sim/python.sh`
- [ ] `evaluation.eval --help` runs
- [ ] `harness/bakeoff.py run shared p0.smoke` produces
      `out/eval/eval_results.csv` (stub policy server + 1 rollout, 100 steps)
- [ ] `import gr00t` in `~/Isaac-GR00T/.venv`; `nvidia/GR00T-N1.7-3B` in the HF cache
- [ ] `import gear_sonic` under `PYTHONUSERBASE=~/env/pyuser-sonic` (WARN only)
- [ ] tmux present; `harness/gpus.py probe` runs

Then the demo set (P0 continued, gated by the replay check):

- [ ] `mimic/scripts/scripted_source_demos.py --headless --num_demos 10
      --rate 50` (**50 Hz**, rev 3c — the script defaults to 30)
- [ ] `annotate_sources.py --auto` -> `generate_parallel.py --total 80 --procs 4`;
      record the keep-rate (target ≥ 50 %)
- [ ] `export_generated.py` -> HDF5 in the `…_demos` run folder
- [ ] replay gate: one episode stepped through `…-JointPos-v0` reaches
      `handover_success = True`
- [ ] the LeRobot v3 -> GR00T v2 conversion may be done here or at the start of
      P1; **the p1 gate owns it** (`out/gr00t_v2/meta/modality.json`)

### P1 — lane A: GR00T direct — GATE `harness/gates/p1.sh`

- [ ] dataset: HDF5 -> LeRobot v3 -> GR00T v2 (`convert_v3_to_v2.py`) +
      `meta/modality.json` at **50 Hz** (state `joint_pos_l 0:7`,
      `joint_pos_r 7:14`, `grip 14:16`; action the same split; video `top`,
      `wrist_left`, `wrist_right`; `annotation.human.task_description`)
- [ ] `harness/lane_a/modality_config_dual_fr3.py`: video `[0]`, state current,
      action `delta_indices = range(40)`, arms **ABSOLUTE / NON_EEF**, grippers
      ABSOLUTE; registered under `EmbodimentTag.NEW_EMBODIMENT`
- [ ] `launch_finetune.py --base-model-path nvidia/GR00T-N1.7-3B
      --embodiment-tag NEW_EMBODIMENT --modality-config-path …
      --num-gpus 2 --max-steps 2000 --save-steps 500 --global-batch-size 32`
      with the SONIC-tutorial colour-jitter params
- [ ] `open_loop_eval.py` on ckpt 500/1000/1500/2000 -> MSE trend recorded
- [ ] policy server (GR00T -> 40-step joint chunk) + a joint client that speaks
      `{state16, 3 cams} -> [Lq7, Lg, Rq7, Rg]`
- [ ] `evaluation.eval --embodiment franka_dual --rate 50 --replan-every 20
      --rollouts 20` -> eval run folder
- [ ] A-oracle replay -> its own run folder with an `eval_results.csv`

### P2 — lane B-1: dual-FR3 SONIC embodiment — GATE `harness/gates/p2.sh`

- [ ] `harness/lane_b/dual_fr3.xml` (2× menagerie `franka_fr3` at rig poses,
      fixed bases, no freejoint) + `harness/lane_b/robots_dual_fr3.py`
      (14 DoF, 2×7 actuator groups, Isaac<->MuJoCo maps, order converter),
      installed into `gear_sonic/envs/manager_env/robots/dual_fr3.py`
- [ ] `harness/lane_b/sonic_dual_fr3.yaml`: `anchor_body` = a fixed base body,
      `vr_3point_body` = [left hand, right hand, base], `reward_point_body` =
      the two hands; **disable** `foot_pos_xyz`, `feet_acc`,
      `undesired_contacts`; keep `ee_body_pos_adaptive`, tracking terms,
      `joint_limit`, `action_rate_l2`; robot-motion encoder only,
      `smpl_motion_file: dummy`, soma off
- [ ] body-name overrides pass a `num_envs=1` smoke
- [ ] motion library = GMR arms-only retarget of a ~1k-clip BONES-SEED
      upper-body subset (GMR cloned to `~/code/upstream/GMR` @bb1bbe4 and
      registered per `wiki/gmr-new-robot-recipe.md`) **plus the P0 demos as
      clips** + `_M` mirrors
- [ ] RL (`train_agent_trl.py`): `num_envs=1` smoke, then a 1–2 h run on the
      devices the allocator gives (4 if available, 2 today)
- [ ] `eval_agent_trl.py +export_onnx_only=True` -> encoder + decoder ONNX
- [ ] decoder replays a demo clip with mean joint error < ~0.1 rad
      (`out/replay.json`)

### P3 — lane B-2: GR00T over SONIC — GATE `harness/gates/p3.sh`

- [ ] `harness/lane_b/label_tokens.py`: encoder ONNX over each demo's joint
      trajectory at **50 Hz** -> `token (T,64)` per frame
- [ ] dataset variant with `action = [token 64 | grip 2]`, modality config
      **ABSOLUTE / NON_EEF, horizon 40**; the P1 fine-tune recipe unchanged
- [ ] policy server = GR00T -> token chunk -> **decoder inside the server** at
      50 Hz with its own proprio history; clip `|t| <= 1.25` before the decoder;
      self-encode a "hold" token from the ready pose for resets
- [ ] `evaluation.eval … --rate 50 --replan-every 20 --rollouts 20`
- [ ] B-oracle replay (encoder tokens -> decoder, no VLA) -> its own run folder

### P4 — compare + decide — GATE `harness/gates/p4.sh`

- [ ] `harness/report/aggregate.py` reads both eval run folders + both oracles
- [ ] `plan/REPORT.md`: one table — per-milestone rate, full-success rate,
      steps-to-milestone, action jerk, wall-clock, GPU-hours per lane, plus
      A-oracle and B-oracle
- [ ] scale-up deltas list: what the real run needs (demo count, full retarget,
      4096 envs × 8 GPUs, 20 000 fine-tune steps)

## Autonomous driver

`harness/driver.sh` runs the phases unattended in tmux window `bakeoff:driver`:
for each phase it checks `plan/STATUS.md` for `GATE PN: PASS`, and otherwise
runs `claude -p --dangerously-skip-permissions --effort xhigh` on
`plan/prompts/PN.md` (fresh context per attempt, up to 3 attempts, `timeout 6h`
each). It stops at the first `BLOCKED:`. See README.md for start/stop and the
`DRIVER: resume` escape hatch.

## Work-package parallelism (8 GPUs)

| lane | job | GPUs |
|---|---|---|
| shared | MimicGen generation, evals | 1–2 |
| A | GR00T fine-tune | 2 |
| B | SONIC RL | 4 |

Never a training rank and an Isaac eval on the same device. Only devices with
< 1 GiB used are allocatable — with 2 today, run the lanes sequentially and say
so in STATUS.md.

## Contracts that bite

- Quaternions: the franka repo is **wxyz** internally, the policy **wire** is
  **xyzw**; `evaluation/codec.py` owns that conversion. The dataset `ee_pose` is
  **rot6d**, not a quaternion. The SONIC pkl boundary (`root_rot` wxyz) is ours
  to get right. Assert with a known pose at each boundary.
- Rates (rev 3c): recorder and datasets **50 Hz**; GR00T chunk = **horizon 40**
  (0.8 s); VLA replan **2.5 Hz** = `--replan-every 20` at `--rate 50`; SONIC
  decoder native 50 Hz.
- Token bounds `|t| <= 1.25` (FSQ grid) — clip the VLA output before the decoder.
- Safety caps from `policy_convention.yaml`: 0.40 m / 0.60 rad per step;
  `control_hz: 15` is the global *default*, overridden per run by `--rate`.
- `evaluation.eval` refuses to resume a run folder with a different task/rate.
  One episode = one sample; always `--rollouts N`.
- Motion pkls for SONIC: `root_trans_offset (T,3)`, `pose_aa (T,nb,3)`,
  `dof (T,ndof)` in MuJoCo order, `root_rot (T,4)` **wxyz**, `smpl_joints
  (T,24,3)` zeros allowed, fps 30, `_M` mirrors.

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
9. GMR needs SMPL-X body models (`SMPLX_*.pkl`, registration-walled) — if they
   are not on the pod, the human-motion half of the library is not obtainable
   autonomously; decide, record the decision, and continue with the demo clips
   plus augmentations (P2 prompt says how).
