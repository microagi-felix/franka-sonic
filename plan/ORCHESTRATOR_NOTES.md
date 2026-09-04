# Orchestrator notes for the running phase agent (P5)

Echoed by every `harness/bakeoff.py` call. Newest first. Act on them; log what you did in STATUS.md.

## 17:55 UTC (2026-09-04) — the dense pass is DROPPED. Your reasoning is accepted; do not re-open P7

You were right to refuse it at P8 start: the gate had passed, and a new lane-A
dataset mid-phase would break the very episode-count equality that keeps the two
lanes comparable. That invariant matters more than the density argument, and the
awkwardness was mine — I asked for the split after the gate instead of before
generation.

**Decision, so nobody re-opens this:** round 2 runs on the 891 screened episodes.
No dense pass, no union dataset, no second lane-A dataset. Reasons, for the
record: re-opening costs ~5 h on the critical path (generate, export, screen,
rebuild, then relabel ~1900 episodes with P8's winning encoder just to restore
the equality invariant), against an uncertain gain — the training support is
continuous and already contains the evaluation box with 7.4 cm of margin, so a
policy trained on 891 episodes spread over it has roughly 25 episodes inside the
box and interpolates the rest. The honest statement belongs in P10's limits
section, not in another day of data work:

> Training spawns cover ±9 cm / ±0.75 rad while the evaluation only visits
> ±1.5 cm / ±0.4 rad, so roughly 25 of the 891 training episodes start inside
> the evaluated region; both lanes share this exactly.

**Where the spare capacity goes instead** (this is what "we can do more" buys):
the six decoder variants you are already running, then P9 at 4 GPUs per lane and
30 000 steps with all 12 checkpoints per lane kept and screened afterwards, then
P10 at 200 rollouts per row. That sequence is the plan; nothing else is queued.

Two small things: `lane_b/label_tokens` failed rc=1 at 17:46 — if that was a
ceiling-test labelling run, say in STATUS what the error was and whether the
retry succeeded, because a silent labelling failure would invalidate a ceiling
number. And keep one device unclaimed for the ceiling tests so a variant never
waits on the allocator.

## 15:45 UTC (2026-09-04, P7) — start the dense generation NOW, in parallel with the CPU rebuild

Timing check against your attempt cap (11:25 → 19:25): dataset rebuild ends
~16:00, then dense generation ~90 min, export ~50 min, screen ~35 min, union
rebuild ~30 min = 19:25 exactly. Too tight, and it wastes the pod: the rebuild
you are running is **CPU-only** while all 8 GPUs sit idle.

Do this:

1. **Launch the dense generation immediately**, alongside the running rebuild —
   `--spawn-xy 0.03 --spawn-yaw 0.5 --arm-noise-std 0.05 --n-generated 1000
   --n-procs 32`, into a new demos run folder (or a distinct worker prefix in
   the existing one, your choice; whichever keeps the merge simple).
2. **Let the current rebuild finish** — it is nearly done and it makes the P7
   gate satisfiable on the wide half alone, which is your safety net if the
   dense half runs long.
3. **Build the union dataset once** at the end, from all screened episodes of
   both halves. That is the dataset P8 and P9 use; say so explicitly in STATUS
   so P8 does not pick up the wide-only one by "newest" resolution.
4. Screen the dense half exactly as you screened the wide half, and add its
   numbers (kept, screened, in-box) to `out/coverage.json`.
5. Drop the optional mid pass at ±6 cm unless everything above is finished by
   ~18:15 UTC.

If the dense half cannot finish inside this attempt, that is fine: write what
exists into STATUS, let the gate pass on what you have, and note that the union
rebuild is the first task of the next attempt. Do not burn the cap on a partial
export.

## 16:00 UTC (2026-09-04) — MORE COMPUTE AUTHORISED (Felix: "we can do more"). Re-size P7 tail, P8, P9, P10

The 15:50 note's sizing is superseded where it is smaller than this. The pod is
ours overnight; the binding constraint is wall-clock, not GPUs, so spend GPUs to
buy quality and keep every stage parallel across all 8 devices.

**P7 tail — more data, not less.** Dense pass target **1000 kept** (not 600) at
`--spawn-xy 0.03 --spawn-yaw 0.5 --arm-noise-std 0.05`, 32 workers as before,
screened the same way. If it runs faster than expected, add a **mid pass** of
~500 at `--spawn-xy 0.06`. One dataset from the union of all screened halves;
report per-half counts and `n_in_eval_box`.

**P8 — six variants, not one.** Six parallel single-GPU runs from the start, all
warm from `final/p5/jp20_last_it2848`, all on the lane-A plant: two at
`--num-envs 4096`, two at `8192`, two with one reward knob each (your choice,
one knob per variant, recorded). 6 h cap each, ceiling-tested every 500
iterations on the same 20 episodes. Extend the leader warm while its ceiling
series still rises. Target ≥ 18/20; floor 15/20.

**P9 — full-length training, screening afterwards.** With compute to spare the
screening no longer has to interleave:

- Both lanes **4 GPUs each**, `--train-steps 30000 --save-steps 2500
  --save-total-limit 16` (12 checkpoints per lane, ~340 GB per lane — check
  `df` first, fall back to `/tmp/franka-sonic` for **both** lanes together).
- When both fine-tunes finish, screen **all** checkpoints of both lanes in
  parallel across the 8 GPUs, 20 rollouts each (~45 min for all 24).
- The stopping rule becomes a **selection** rule on the finished series: best =
  highest success, ties by mean progress, then the earlier checkpoint. Still
  write `P9 BEST lane_a=…` / `P9 BEST lane_b=…`. Report both learning curves.
- Kill a lane early only on a hard failure (NaN loss, trainer death).

**P10 — 200 rollouts per row, not 100.** Four rows at 200 seeded rollouts
(episodes 0–199), two GPUs per row, all four in parallel: ~3.5 h and the 95 %
interval tightens from roughly ±10 points to ±7. Keep everything else identical
between lanes. Gate `P10_MIN_ROLLOUTS=200 P10_MIN_ORACLE=200 bash
harness/gates/p10.sh` — the default is 100, so pass the overrides or the gate
reads the smaller floor.

Unchanged and non-negotiable: identical data, command, budget and evaluation for
both lanes; nothing deleted; storage floor 600 GB; only recorded pids killed.

## 15:50 UTC (2026-09-04, P8 preview) — do not let the 4 h cap decide the decoder

P8's prompt says one variant, 4 h, one GPU. That was sized on round 1, where the
winner reached B-oracle 19/20 in 1.5 h (848 iterations warm from jp18 ckpt 2000,
2848 total). The round-2 reference library is ~12× larger, so the same recipe
has a harder tracking problem and its plateau may sit further out. 4 h at round
1's rate is only ~2300 iterations. Three changes:

1. **Use the idle GPUs.** Run **three variants in parallel from the start**, one
   GPU each: (a) jp24 = jp20 recipe, warm from `final/p5/jp20_last_it2848`;
   (b) the same with `--num-envs 8192` (more samples per iteration, the library
   is bigger); (c) the same as (a) but warm-started and with the reward's far
   kernel slightly relaxed if your first replay curve stalls — your call, one
   knob only. Ceiling-test each on the same 20 episodes; the ceiling test is the
   only valid selector (round-1 finding).
2. **Extend rather than stop.** If the best variant's ceiling series is still
   rising at its cap, continue it warm for another 2 h and keep testing. Stop
   when two consecutive ceiling tests fail to beat the best, or at ≥ 18/20.
3. **Budget guard.** P8 owns the pod until P9 needs 6 GPUs. If the decoder is
   still below 15/20 after ~7 h total, take the best you have, write the number
   plainly, and go on to P9 anyway — lane B's row is then read against its own
   B-oracle, exactly as round 1 taught. Do not let P8 eat P9's day.

Same hard rules: kill only recorded pids, mark NEEDS-CLEANUP, nothing deleted.

## 15:20 UTC (2026-09-04, P7) — screening accepted; now do the dense pass, then one dataset

Your covariate analysis is the right call and it refutes both of my hypotheses
cleanly. Screening every episode on the JointPos env and marking (never
deleting) is the correct fix. Two follow-ups:

1. **The dense pass is still owed** (my 12:00 / 12:20 / 13:00 notes). After the
   screen and the rebuild, run one generation at `--spawn-xy 0.03 --spawn-yaw
   0.5 --arm-noise-std 0.05`, target ~600 kept, screen it the same way, then
   rebuild the dataset **once** from the union of both screened halves. Reason
   unchanged and now sharper: the evaluation only visits ±1.5 cm, so the wide
   set contributes ~23 in-box episodes out of ~830, while ~600 dense episodes
   contribute ~150. Time: generation ~40–60 min (the keep rate should be better
   than 40.6 % at short reaches), export ~30–50 min, screen ~20 min. You are at
   15:20 in an attempt that runs to 19:25, so it fits — and if it does not, cut
   the dense pass with `--gen-deadline-min` rather than skipping it.

2. **Record one caveat for P10's report**: the screen keeps episodes whose
   recorded *absolute joint targets* replay open-loop, which is lane A's action
   space. Both lanes get the identical episode set, so the comparison stays
   like-for-like, but the selection criterion is defined in lane A's terms and
   may drop episodes lane B's decoder could execute. State it in the report's
   limits section; do not change the criterion — a clean shared set matters
   more, and it keeps the A-oracle row honest.

## 14:40 UTC (2026-09-04, P7) — the 3/5 replay check is the phase's real finding: diagnose it before the dataset is used

`bakeoff shared/demos` returned rc=1 because the 5-episode replay check scored
3/5 (episode 2 stalled at progress 0.333). Round 1 replayed 76 of 80. Do not
treat this as a flaky check and do not build P8/P9 on this dataset until it is
understood — a dataset whose episodes do not reproduce in the evaluation env
teaches both lanes targets that fail there, and it would silently cap round 2
the way the gravity mismatch capped round 1.

**Diagnose first (parallel, ~15–30 min):**

1. Replay a **stratified sample of 60** episodes through `eval_oracle_a.py`,
   sharded across the 8 GPUs the way generation was: 20 with cube spawn radius
   ≤ 3 cm from the evaluation centre (0.4375, −0.78), 20 in 3–6 cm, 20 > 6 cm.
   Report success per stratum.
2. Separate the two candidate causes:
   - *Wide spawn*: success falls with spawn radius → the wide half is
     intrinsically less valid (near joint limits, longer reaches).
   - *Arm-start mismatch*: the generation ran with `--arm-noise-std 0.05` while
     the replay resets the arms differently. Check what `eval_oracle_a.py`
     actually sets at reset (round 1's trace said it resets on the demo's frame
     0) and whether the recorded start pose is reproduced. If it is not, this is
     a harness bug, not a data problem, and the fix is free.
   Record which one the numbers support, with the numbers.

**Then act on what you find:**

- *Harness bug*: fix it, re-run the check, keep all 1024 episodes.
- *Wide spawn is the cause*: replay-validate **all** 1024 episodes (sharded, it
  is the same cost per episode as generation and buys certainty), keep only the
  valid ones, and build the dataset from those. Then run the dense pass
  (`--spawn-xy 0.03 --spawn-yaw 0.5`) from my 12:00/13:00 notes — dense
  episodes should validate at a much higher rate, which is the second reason to
  have them.
- Either way `out/coverage.json` gains `replay_valid` counts per stratum, and
  the dataset is built **only** from replay-valid episodes.

The gate's floor is 600 episodes; 1024 minus the invalid ones plus a dense pass
should clear it comfortably. If it does not, say so in STATUS with the numbers
rather than lowering the standard — a smaller clean set beats a large dirty one.

Budget: you are 3 h into an 8 h attempt with the driver's 3-attempt retry behind
you. Spending an hour here is correct.

## 13:00 UTC (2026-09-04, P7) — the dense pass must reach the SAME export and dataset

Measured from outside at 12:53: 250 MiB across the 32 worker files, and round
1's merged set was 28.1 MB for 80 episodes (0.35 MB/episode), so the wide pass
is at roughly 750 episodes after 50 min and will reach 1000 around 13:10 —
well inside its deadline. Your `--steps generate,export,coverage,replay` then
rolls straight into export, which is why this note matters.

**Whatever state you are in when you read this, the target is one dataset built
from both spawn widths:**

- *Generate still running*: let it finish the 1000 wide episodes, then run the
  dense pass (`--spawn-xy 0.03 --spawn-yaw 0.5 --arm-noise-std 0.05`,
  `--n-generated 1000`, its own worker prefix) **before** export, merge both
  into the file export reads, and export once.
- *Export already started or finished on the wide half*: keep it. Run the dense
  pass, export it into the **same export directory** with a distinct shard
  prefix, and build the single `gr00t_v2` from the union. Do not re-export the
  wide half.
- Either way `out/coverage.json` reports both halves separately and the merged
  set, with `n_in_eval_box`.

Budget check before you commit: at ~0.35 MB/episode and this rate the dense
pass should cost ~40–60 min, and export of ~2000 episodes at 16 shards ~2–3 h.
That still lands P7 inside its 8 h attempt. If your own measurements say
otherwise, cut the dense pass short with `--gen-deadline-min` rather than
dropping it — even 400 dense episodes are worth ~100 in-box, four times what
the wide half alone gives.

## 12:20 UTC (2026-09-04, P7) — the 60/40 split still stands; the running generation is the WIDE half

You launched at 12:03 with `--spawn-xy 0.09` and a 170 min deadline, i.e. the
pure wide setting my 12:00 note supersedes. Do not throw that work away — treat
it as the **wide half** and add the dense half after it:

1. **Stop the running generation at ~13:15 UTC** (≈70 min of generation) by the
   pids you recorded, exactly as `--deadline_min` would: your merge keeps every
   episode the workers have flushed. Expect roughly 40 % of the 1000 target.
2. **Then run the dense half** into the same run folder with
   `--spawn-xy 0.03 --spawn-yaw 0.5 --arm-noise-std 0.05` and
   `--gen-deadline-min 70`, worker files under a distinct prefix.
3. **Merge both passes** into the single `generated*.hdf5` the export step
   reads (your worker merge already handles N files; both passes share the task,
   the sources and the episode schema). One export, one `gr00t_v2`.
4. `out/coverage.json` reports the merged set: per-half counts and spawn stats,
   the merged distribution, `n_in_eval_box` (|x−0.4375| ≤ 0.015, |y+0.78| ≤
   0.015, |yaw| ≤ 0.4) and `covers_eval`.

Why, once more, because it is the whole point of the phase: the evaluation only
ever visits ±1.5 cm / ±0.4 rad. At ±9 cm about 1 episode in 36 lands there; at
±3 cm about 1 in 4. The split buys ~130 in-box episodes instead of ~28 while
keeping the evaluation region interior to the training support. If the dense
half's keep rate is much better than the wide half's (likely — shorter reaches,
fewer IK failures), let it overshoot its share rather than stopping it early.

Good catch on the `nan` keep-rate bug: round 1's "24 % keep rate" in the prompt
is not from this pipeline's logs, so treat any keep-rate planning number as
unknown until this run measures it.

**Amendment (12:25), because notes only reach you on a bakeoff call and you are
mid-poll:** if you read this *after* the wide generation has already finished,
do not undo anything. Keep every wide episode, then run one dense pass of about
60 % of the wide count (`--spawn-xy 0.03 --spawn-yaw 0.5 --arm-noise-std 0.05`,
`--gen-deadline-min 70`), merge both passes into one export and one `gr00t_v2`,
and report both halves in `out/coverage.json`. A larger merged set than 1000 is
welcome as long as export and the P9 fine-tunes still fit their budgets — say
so in STATUS if you shorten the dense pass to protect the schedule.


## 12:00 UTC (2026-09-04, P7) — SPLIT the generation spawn width; my ±9 cm target is superseded

Your WP 7.1 finding is right and it changes the recipe. The evaluation box is
**±1.5 cm / ±0.4 rad**, not the ±6 cm / ±0.5 rad the prompt claimed (that was
the generation range). So "cover the evaluation range with margin" was already
true in round 1, and pushing generation to ±9 cm spends the data budget on
configurations the evaluation never visits: ~1/36 of episodes land in the
evaluation box, ~28 of 1000.

**Do this instead — one dataset, two spawn widths:**

- **60 % dense**: `--spawn-xy 0.03 --spawn-yaw 0.5` (4× the evaluation area in
  x/y, 1.25× in yaw) → ~150 of 600 episodes inside the evaluation box.
- **40 % wide**: `--spawn-xy 0.09 --spawn-yaw 0.75` as you configured → the
  robustness half, and it keeps the evaluation region interior to the training
  support rather than at its edge.
- Keep `--arm-noise-std 0.05` for both halves.
- Merge both generations into **one** export and **one** `gr00t_v2`. Both lanes
  train on the identical mixture, so comparability is untouched; P8 relabels
  that same set.
- Use `--gen-deadline-min` to split the 3 h budget between the two runs (e.g.
  100 min dense, 80 min wide) so the phase still fits its budget. Record the
  achieved counts and the in-box fraction of the merged set in
  `out/coverage.json` (add a `n_in_eval_box` field) and in STATUS.

Rationale: with a fixed episode budget, local sample density near the tested
distribution is what buys success rate; breadth buys robustness that this
evaluation cannot see. The split gets ~5× more in-box episodes than the pure
wide run and still trains on a support that contains the evaluation region.

If your probe shows the keep rate is much worse at ±9 cm than at ±3 cm, shift
the split further toward dense (70/30) and say so — do not spend the budget
chasing 1000 episodes at the wide setting.

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
