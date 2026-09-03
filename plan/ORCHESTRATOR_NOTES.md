# Orchestrator notes for the running phase agent (P5)

Echoed by every `harness/bakeoff.py` call. Newest first. Act on them; log what you did in STATUS.md.

## 22:05 UTC — jp20 series is still rising: extend it past the 22:33 cap

Ckpt 500 1/20 → ckpt 1000 7/20 → ckpt 1500 4/11 so far (progress 0.80). The curve has not flattened, so:
1. When jp20 hits its 22:33 wall-clock cap, relaunch it immediately as a warm start from its last checkpoint (same jp20 recipe, same plant, `--hours 2.0`, one device) and keep the replay + ceiling-test loop every 500 iterations. Attempt 3 has budget until ~02:39 UTC; use it on this one variant only. No std-clamp or reward changes (jp21 showed the warm start breaks).
2. Ceiling-test every 500 as long as the count rises; stop testing when two consecutive checkpoints do not improve, and take the best one as the P5 final.
3. If a checkpoint reaches ≥ 15/20, write the gate marker, copy encoder+decoder ONNX + tokens dataset to `~/runs/franka-sonic/lane_b/final/p5/` and let the driver proceed to P6 (lane B redo: relabel with that encoder, refinetune GR00T, eval). P6 is only justified on a passing gate.
4. The 20:05 wrap-up rule stands: STATUS line with the series table (ckpt → replay rad → grasp-frame L/R cm → B-oracle N/20 → mean progress) before the attempt ends.

## 21:40 UTC — ceiling #7 (jp20 ckpt 500) partial: the plant fix works, the residual is the RIGHT re-grasp

Read at 21:38 from oracle_b-7's run log: 16 episodes in, 1 success (episode 12, done in 728 steps), 9 at progress 0.667, 5 at 0.167, 1 at 0. The 0.667 group = left grasp, lift and place succeed, the right arm's blind re-grasp misses, exactly the failure mode the tolerance test showed at +0.05 rad (right grasp tolerates 1–3 cm). So:
1. The decisive metric for the 1000/1500/2000 checkpoints is not the flange-vs-demo error but the RIGHT hand's position relative to the ACTUAL cube at the right grip-close frame (the left arm places the cube where the decoder put it, not where the demo did). Add that number to the FK report (cube pose is in the trace) and pick the ceiling-test checkpoint by it.
2. If the 1000-iteration checkpoint is not clearly better on that metric, spend the remaining budget on one precision variant on the lane-A plant (jp20 recipe, tighter kernels 0.15 → 0.10 rad on joints 1–2 and 7, lower exploration std clamp) rather than on more of the same, and ceiling-test the better of the two at ~22:33.
3. Whatever the count at wrap-up, the honest result is "plant mismatch found and fixed; ceiling opened from 0/20 to N/20 with the right re-grasp as the remaining gap"; the gate's 15/20 stays as is. Do not run P6 unless the gate passes.
4. The 0.167 group (5 of 16) is the left grasp missing: check whether those are the episodes with the largest left-hand error at grip-close or a cube-spawn cluster; one line in STATUS is enough.

## 2026-09-03 20:30 UTC — measure the token's precision floor (5 minutes, CPU) before betting on longer RL

Longer RL only pays if the token still carries the target more precisely than the decoder tracks it.
The model has the `g1_kin` reconstruction head (token → `command_multi_future_nonflat`, which
includes the reference joint positions). Run it on the jp15 label tokens of episode 0 and 5 (the
exported pair or the checkpoint in-process) and report: mean |reconstructed q − reference q| per
joint (rad) and, by FK, the implied hand error (cm) at the grasp frames. Reading:
- recon error ≪ decoder error (e.g. 0.3 cm vs 1 cm at the hand) ⇒ the FSQ token is not the floor,
  precision training (fingertip points, tighter kernels, low noise, more hours) can still gain;
- recon error ≈ decoder error ⇒ the 64-D × 32-level FSQ quantisation IS the floor and no amount of
  RL will grasp a ~1 cm-margin cube blind — then the demos' grasp margin (regenerate with a fully
  open, centred, slow pre-grasp) is the only real fix, to be decided by Felix tomorrow.
Put the numbers in STATUS next to the tolerance-test result.

## 2026-09-03 20:15 UTC — add one cheap diagnostic: slow-motion B-oracle

After the A-oracle tolerance test, run a **time-stretched B-oracle** on the best decoder (jp15 or
jp14 last): resample the 76 demo clips 1.5× slower (motion pkl at the same fps with 1.5× the
frames; joint velocities scale down accordingly), re-label tokens with the same encoder, and
replay through `oracle_b` with the grip labels stretched the same way (horizon 1500 steps holds
1.5 × 812). Same episodes, same spawns, protocol otherwise unchanged; tag the run folder `_slow`.
Reading: if successes appear, the residual is dynamic lag (PD + policy), and precision training /
a slower approach in the demos will fix it; if it stays at zero with the hand still 1 cm off, the
error is a static bias and the demos' grasp margin is the constraint. Record both hand-error
(FK, grasp frames) and the oracle result. ~30 min on one device; do it before the 21:11 caps.

## 2026-09-03 20:05 UTC — attempt 3 budget: adopt jp16/jp17/jp18, tolerance test first, then wrap up

- The three continuation trainers launched 19:40 (jp16 = jp14 cont., jp17 = jp15 cont., jp18 =
  jp15 + fingertip reward points; caps ~21:11) ARE the "one more variant" budget from P5.md §6 —
  adopt them, do not add more. Item 1 of the 19:45 entry (kill the six hung trainer pids) still
  stands; do it first, it takes a minute.
- Run the A-oracle tolerance test (19:45 item 2b; j1 offsets 0.02 / 0.05 / 0.10 rad, 20 rollouts
  each, ~6 min each on one device) while the trainers run. Also record, from the demo hdf5, the
  gripper opening at the pre-grasp frame vs the cube width.
- At the 21:11 caps: replay + FK grasp-frame hand error for all three; ceiling test on the best one
  (and on jp18 if its fingertip error is the lowest, since it targets the failure directly).
- Then write the P5 result in STATUS as it stands (decoder table, tolerance numbers, best B-oracle),
  copy finals for the tested decoders, run the gate, and end the turn. If the gate is < 15/20 the
  driver will mark P5 BLOCKED — that is the honest outcome; P6 must not run on a closed ceiling.
- Harness debts to leave in STATUS for tomorrow (do not fix tonight while runs are live): bakeoff's
  finalizer must wait on the python pid and SIGKILL after a grace period; the allocator's job name
  must be unique across run roots.

## 2026-09-03 19:45 UTC — six hung trainers to kill; measure the task's tolerance before more precision runs

1. **Six trainer python processes survived their caps** (Kit's `close()` hang, a known rake): pids
   341689 (jp6), 387915 (jp11), 425660 (jp12), 462198 (jp13), 519858 (jp14), 522702 (jp15) — all
   reparented to init, ~115 % CPU each, ~12 GiB GPU each, no checkpoint written since their caps.
   They are yours (recorded launcher pgids 341684/387910/425655/462193/519853/522697): `kill -9` the
   six python pids now, confirm with `gpus.py list` + per-device memory, log it. Harness debt for
   later: bakeoff's finalizer must wait on the python pid and SIGKILL after a grace period.
2. **Ceiling #4/#5 (jp14/jp15) are 0/17 with ~65 % reaching milestone 1** — precision went from 1.5
   to 1.0 cm at the grasp and it still closes on air. Before any further precision variant, measure
   what the task actually tolerates (CPU + one short GPU job, ~15 min total):
   a. From the demo hdf5: gripper opening at the pre-grasp frame vs the cube width, per episode — the
      lateral margin per finger. If it is ≤ 1 cm, no 50 Hz PD tracker will reach it with our noise.
   b. **A-oracle tolerance test**: replay the A-oracle (recorded joint targets) with a constant
      offset added to the left arm's j1 targets of +0.02 / +0.04 rad (≈ 1 / 2 cm lateral at the
      hand), 20 rollouts each (two `oracle_a` runs with a target-offset option in
      `eval_oracle_a.py`, default 0, protocol otherwise unchanged). The offset at which the
      A-oracle drops from 20/20 to ~0 IS the ceiling requirement. Put both numbers in STATUS.
   c. Only if the requirement is within ~2× of jp14/jp15's grasp-frame error: one last variant,
      warm from jp15 last, `--hours 1.0`: jp14's kernels + `std_clamp_max` 0.5 → 0.1 (less action
      noise = tighter tracking), and confirm the ONNX export uses the MEAN action. Ceiling test at
      its cap. Otherwise stop training.
3. **If the ceiling is not ≥ 15/20 by 20:40 UTC** (attempt 2 ends 21:02): write the P5 result as it
   stands — decoder solved (0.055 rad / 1 cm), ceiling closed by grasp precision vs the demos'
   margin (numbers from 2a/2b), best B-oracle progress — copy finals for the best two decoders to
   `~/runs/franka-sonic/lane_b/final/p5/`, and leave the marker to the gate. Attempt 3 (auto,
   6 h) then reads P5.md §6 and continues ONLY with what 2b justifies; it must not re-run the
   variant ladder.

## 2026-09-03 18:20 UTC — 0/20 at 0.089 rad ⇒ the plant differs; train on the deployment plant

Two decoders under 0.1 rad and still 0/20 means the SONIC training env and lane A's JointPos env
are different plants and the policy has learned its own plant's response (soft 200/20 wrists ⇒ it
over-commands targets; in a 400/80 plant the same targets overshoot). Fix on the training side,
one variant, warm-started, `--hours 1.0`, then ceiling test:

- **jp14 = jp13 + the deployment plant + reset noise**:
  1. Copy lane A's JointPos env actuator config into `harness/lane_b/robots_dual_fr3.py` EXACTLY:
     stiffness/damping per joint (all seven), effort and velocity limits, armature/joint friction,
     `soft_joint_pos_limit_factor`, and the gripper mass/inertia on link7 (the SONIC MJCF has no
     grippers; add an equivalent point mass at the flange, or the URDF's hand link, so link7's
     inertia matches). Also match sim dt × decimation = 20 ms and the physics solver settings that
     matter (substeps, solver iterations) if they differ.
  2. `manager_env.commands.motion.joint_position_range: [-0.6, 0.6]` (reset noise) so the policy
     learns to converge onto the reference from off-reference starts such as FR3_READY.
  3. Rewards/terminations as jp13. Re-install, Hydra-compose check, launch from the best
     checkpoint (jp12/jp13). Its in-env error will jump at first — fine.
- While it trains: finish the per-step trace on 2 episodes and confirm the diagnosis by plotting
  decoder target vs measured q for L_j5–j7 in the first 2 s (overshoot/oscillation = gains; a
  slow monotone approach = start pose). Put the number in STATUS either way.
- Oracle-side, allowed and cheap: hold the frame-0 token for the first 0.5–1 s (same as the
  server's ready hold). Not allowed: changing the env's reset, gains or rate.
- Ceiling test on jp14's first checkpoint whose replay is < 0.15 rad; if it scores ≥ 10/20 keep
  training and re-test at the cap; if still ~0, stop and write the trace findings to STATUS as the
  P5 result — do not spend attempt 3 on more reward variants.

## 2026-09-03 17:58 UTC — runtime is faithful (good); now train the start-pose gap away, keep the oracle env as is

- The teacher-forced 0.227 rad settles it: the FR3-side path is right. Remaining gap = the closed loop
  in lane A's env: (a) wrist-roll error at grasp time, (b) the FR3_READY start pose 0.3–0.7 rad off
  the demos' frame 0, which the SONIC env's on-reference reset never trained.
- **Do NOT change the oracle env's reset or controller** — lane A's eval and the P6 lane-B eval must
  stay identical to P1/P3 (comparability protocol). Fix it on the decoder side:
  - **jp14** (warm from the best jp12/jp13 checkpoint, `--hours 0.75`): jp13's rewards +
    `manager_env.commands.motion.joint_position_range: [-0.6, 0.6]` (reset noise on every joint, so
    the policy learns to converge onto the reference from off-reference starts like FR3_READY) and,
    if the config has it, a wider reset-velocity range. Expect the in-env error to jump at first and
    settle within a few hundred iterations.
  - Runtime option that is legitimate for BOTH oracle and server: hold the frame-0 token for the
    first ~0.5–1 s of an episode (the server does this already for its ready-pose hold token —
    make the oracle do the same thing, nothing else).
- Ceiling test #2 (jp11 ckpt 1000) is the read-out of (a)+(b) together; run #3 on jp12 ckpt 1000
  (0.063 rad) as soon as a device is free, in parallel with #2's tail. If jp12 scores ≥ 10/20, the
  wrist roll was the main term and jp14 is the polish; if it stays ~0, the start pose is the main
  term and jp14 is the fix.
- Harness debt (fix only when NO launcher is live, e.g. after the 18:32 cap): the allocator's job
  name must include the run root or a unique id (17:48 incident). Until then check `gpus.py list`
  by hand before every launch, as you already do.

## 2026-09-03 17:35 UTC — the 0/20 on jp7 is almost certainly the oracle path; find it before more variants

A decoder at 0.226 rad / 1.6 cm in its own env scoring 0/20 with progress 0.008 (below P3's 0.033) means
the FR3-side runtime, not the policy. Tests that discriminate, cheapest first (CPU, minutes each):

1. **Teacher-forced runtime replay** (what P3 did at 08:46): feed the demo's RECORDED FR3 joints +
   the jp7 offline tokens through `harness/lane_b/sonic_decoder.py` exactly as `eval_oracle_b.py`
   calls it; compare decoder targets to the reference per joint. Expected ≈ 0.2 rad if the FR3 →
   SONIC → FR3 mapping is right. If it is ≫ 0.5 rad, the bug is in that mapping: joint ORDER
   (IsaacLab breadth-first left_j1, right_j1, left_j2 … vs the FR3 env's [Lq1..7 | Rq1..7]),
   the joint-6 offset (SONIC q6 = FR3 q6 − 2.5307, applied on the way IN for proprio and on the
   way OUT for targets, never twice, never zero times), joint velocities (rad/s, same order),
   `joint_pos_rel` = q_sonic − SONIC default pose (the P2 default, not the FR3 ready pose),
   `last_action` = the raw policy action (not the target), history filled with the FIRST real obs
   or zeros exactly as the SONIC env does at reset.
2. **Round-trip identity**: run the A-oracle's recorded FR3 joint targets through the runtime's
   FR3→SONIC and SONIC→FR3 conversions; must be identical to 1e-6.
3. **Initial state**: the SONIC env always starts AT the reference pose (RSI). Confirm the oracle
   episode's first token is the demo's frame-0 token and the FR3 env's reset pose equals the demo's
   frame-0 joints; if the env resets to a different ready pose, hold frame-0 tokens for ~1 s first.
4. **Plant**: compare the JointPos env's actuator gains / effort limits with SONIC's 400/80 + 200/20;
   plot decoder target vs measured q for one oracle episode (the 4 PNG frames + per-step logging).
   A stiffer/softer plant changes the closed-loop feel but does not zero the task — a convention
   bug does.
5. Fix in `sonic_decoder.py` (the policy server shares it — P6 depends on this), re-run
   `oracle_b` on the best checkpoint (jp11/jp12 by replay), then the gate. The trainers keep
   running to their caps; do not launch further variants until the oracle path is proven with a
   ≥ 10/20 result.
6. Record the per-joint teacher-forced numbers in STATUS whichever way it goes.

## 2026-09-03 16:30 UTC — jp6 is close; test the ceiling NOW, and attack the left wrist roll in parallel

jp6 ckpt-1000 replays at 0.309 rad with the 13-joint mean at 0.105 rad and body error 1.8 cm — the
only miss is the left wrist roll (0.73 → −3.02 rad). **Task success is the gate, not joint error**,
and a parallel gripper grasps a cube just as well rolled by 180°, so the ceiling test may already
pass. Six devices are free (0–4, 6). Do all of this concurrently, one device each:

1. **Ceiling test now** on jp6 ckpt-1000: `label_tokens` (ALL steps, new encoder ⇒ new tokens + new
   `gr00t_v2_sonic`) → `oracle_b --rollouts 20`. Repeat on every later checkpoint whose replay
   error improves. If ≥ 15/20: gate, finals, PASS — do not wait for the training cap.
2. Keep jp6 training and its replay loop.
3. Launch, warm-started from jp6's newest `model_step`, `--num-envs 4096 --hours 2.0`:
   - **jp7**: the linear joint penalty per joint instead of mean-diluted (−0.3 × Σ_j |dq_j|, i.e.
     −0.3 rad⁻¹ per joint, ~4× the current per-joint gradient) + wrist joints (j5–j7, both arms)
     weighted ×3 inside `tracking_joint_space` (per-joint weight vector, default ones).
   - **jp8**: jp7 + ASYMMETRIC reward-point offsets — the current ±0.05 m x pair is symmetric under
     a 180° roll, so the position kernel cannot see the roll at all; use one point at +0.05 m x and
     one at +0.05 m y per hand (the encoder input width must stay 1391; else revert this item).
   - **jp9**: jp7 + `std_clamp_max` 1.5 + `max_grad_norm` 1.0.
   Replay each at 500/1000; ceiling-test any that beats jp6.
4. Mind the CPU (384 cores, load ~30 from other tenants): if jp6's iteration time doubles, stop
   the newest variant by its recorded pgid.

## 2026-09-03 16:10 UTC — all 8 GPUs are allocatable now; use them

- `harness/gpus.py` threshold raised 1 → 40 GiB (pulled onto the pod). Devices 0–4 and 6 hold
  5–19 GiB from processes outside this pod that are idle (measured ~380 TFLOPS bf16 on each,
  identical memory numbers since 2026-09-01). Felix: "don't we have 8 on 1 instance" — yes, use them.
- **Do now, in this order:**
  1. Run the export → replay loop (WP 5.2) on a free device immediately for the newest jp3 and jp5
     checkpoints instead of waiting for the caps; keep doing it every ~30 min.
  2. Launch up to **4 more warm-started variants in parallel** (`--checkpoint` the best jp3/jp5
     `model_step` so far, `--num-envs 4096 --hours 2.5`), one per device, each its own yaml:
     - jp6: jp5 + `algo.config.actor.backbone.aux_loss_coef.g1_recon` 0.01 → 1.0 (the token must
       carry the reference joints; the recon target includes joint pos/vel)
     - jp7: jp5 + `std_clamp_max` 0.5 → 1.5 and `max_grad_norm` 0.1 → 1.0 (exploration for 3 rad
       excursions is otherwise ~0.1 rad)
     - jp8: jp6 + jp7 together
     - jp9: jp8 on a motion library WITHOUT the mirrored clips (keep the augmentations) — see the
       mirror check in P5.md §6 item 3; skip jp9 if the mirror check passes
     Keep at least ONE device free for export/replay/label/oracle at all times. Mind the CPU: if
     iteration time on the existing runs doubles after the launches, stop the newest variant by
     its recorded pgid.
  3. At each cap: export + replay every variant, run `label_tokens` + `oracle_b` on the best one
     (lowest replay error), then the next best if it scores ≥ 10/20 but < 15/20.
- STATUS.md stays append-only and yours; the orchestrator writes only this file and P5.md.
