# Orchestrator notes for the running phase agent (P5)

Echoed by every `harness/bakeoff.py` call. Newest first. Act on them; log what you did in STATUS.md.

## 09:45 UTC (2026-09-05, P10 attempt 1) — AUTHORISED: start lane B's P11 fine-tune now on the idle devices; lane A's waits for GATE P10

Lane B's headline row is final (eval-7: 92/200 = 46.0 %, held-out 20–199
80/180 = 44.4 %), B@20000 is minutes from 200, and every lane-B P11 input is
verified: `shared/2026-09-05_dataset` (1855), `lane_b/2026-09-05_motion_lib`
(1855 orig), `lane_b/2026-09-05_label_tokens` (1855 tokens, VERDICT OK,
`gr00t_v2_sonic` 1855). Devices 7 and 0 would otherwise idle ~2 h until the
lane-A rows end (~11:50). Same precedent as the early P10 rows: this is
load-smoothing recorded as such, not a phase change — P10 still owns its
report and gate, and P11 starts when the waiter's driver picks it up.

Do this, once device 0 is released by eval-8:

```
BAKEOFF_RUN_ROOT=/tmp/franka-sonic python3 harness/bakeoff.py run lane_b finetune --gpus 2 \
    --dataset ~/runs/franka-sonic/lane_b/2026-09-05_label_tokens \
    --init-from /tmp/franka-sonic/lane_b/2026-09-05_finetune/out/checkpoints/checkpoint-20000 \
    --train-steps 20000 --save-steps 2500 --save-total-limit 12
```

Then, within ten minutes, the warm-start verification from `plan/prompts/P11.md`
WP 11.4, recorded in STATUS as the literal line `P11 INIT lane_b=<that
checkpoint path>` followed by an indented line with: the loading summary from
`logs/run.log` (no missing / unexpected / mismatched keys beyond
`mask_token`), the mean loss over steps 10–100 (expect near round 2's END,
~0.032, not its start, 0.13), and whether the new run's processor
`statistics.json` is byte-identical to the checkpoint's or re-derived. Write
`out/finetune.pid` / `.pgid`. Append a **RESUME POINTER** naming the run
folder, the pid and the device set so the P11 agent adopts the run instead of
launching a second one (its prompt says a live fine-tune is never restarted;
`ps -p` first). If the loss over steps 10–100 is near 0.13, the warm start did
not take: stop that run (recorded pgid, then the python pid), write what you
saw, and leave lane B for P11 proper — do not iterate on it inside P10.

Lane A's P11 fine-tune stays where the prompt puts it: after `GATE P10: PASS`,
by the P11 agent. Nothing else changes; the mirror keeps running.

## 09:00 UTC (2026-09-05, P10 attempt 1) — home is draining again (~140 GB/h); an insurance mirror is running, do not redo it

`df` went 818 → 738 GB free between 08:22 and 08:56 — from outside, as
overnight. The floor (600) is ~1 h away at that rate and zero ~5 h. What that
means and what is already done:

- Below the floor `bakeoff.py` routes **new** run folders to `/tmp` by itself;
  P11's fine-tunes are on `/tmp` from step 0 anyway (`BAKEOFF_RUN_ROOT`).
- The running rows write to home. An orchestrator mirror
  (`/tmp/franka-sonic/p11/mirror_p10.sh`, pid in `mirror_p10.pid`, log
  `mirror_p10.log`) copies every P10 row folder and every P11 prep artefact
  (`demos-union`, `2026-09-05_dataset*`, `motion_lib`, `label_tokens*`) to
  `/tmp/franka-sonic/p10_mirror/<lane>/<run>` every 10 min with `cp -au`
  (newer files only, nothing ever removed). **Do not start a second one.**
- If a row dies on ENOSPC, its per-episode `episode_*_result.json` files are
  the record: rebuild `eval_results.csv` from them (same columns: episode,
  episode_length, success, progress) — do not rerun the row. Say so in STATUS.
- If home is below ~50 GB when the P11 fine-tunes launch, point `--dataset`
  at the `/tmp/franka-sonic/p10_mirror/...` copies (verify `meta/info.json`
  and one parquet + mp4 open first) and record it as `(instance-local, not
  persistent)`.
- Your WP 11.2 library is right (1855 orig clips, indices 0–23); the prompt's
  "2048" line was mine and is fixed in this push.

## 08:40 UTC (2026-09-05, P10 attempt 1) — your WP 11.0 numbers are right; the P11 gate and prompt now match them

Your union folder is correct and better than the prompt: the wide export has
**16** shards (`demos_shard0..15`), so eval-matched as `demos_shard16..23` is
the right renumbering, and the converter's `replay_success` keep filter makes
the union **1855** episodes (891 + 964), not 1915. `harness/gates/p11.sh` now
defaults `P11_MIN_EPISODES` to **1800** and `plan/prompts/P11.md` carries the
same numbers — `git pull --rebase` before your next push picks both up. Nothing
else changed. Carry on: WP 11.1 is running, WP 11.2/11.3 next, fine-tunes only
after `GATE P10: PASS`.

## 08:15 UTC (2026-09-05, P9 -> P10 -> P11) — lane A's last checkpoint screens 16/20; P11 is decided and its harness is in the repo

**Correction first (also in watcher.log at 08:12).** The watcher's series line
`lane_a step 20000 1/7 … eval-9` is a mis-attribution: `newest_eval_run()` took
the newest lane-A eval folder by mtime, which was the 200-rollout row of
ckpt-7500 (`eval-9`, 7 episodes in). The real screen is
`lane_a/2026-09-05_eval-7`: **16/20, progress 0.933, four failures all at
0.667**. Lane A's best-by-screen is therefore `checkpoint-20000`; append a
corrected series line (the stopping rule keeps the last line per step), stop
the watcher, write the two `P9 BEST` lines, run the gate, launch the A-20000
row on device 5 and the A-oracle on 2/3 when the export releases them, and copy
`checkpoint-20000` (not 17500) to `final/p9`. Both lanes now show the same
shape — flat for three quarters, everything in the last quarter, still climbing
at 20 000 — which is what P11 is for.

**P11 = round 3, decided by Felix (08:05).** Warm-restart each lane from its own
round-2 `checkpoint-20000` for 20 000 more steps (fresh optimizer + cosine) on
the **union** of the 891 wide and the 1024 eval-matched demos. Recipe, layout
and checks are in `plan/prompts/P11.md`; the gate is `harness/gates/p11.sh`.
Harness changes pushed with this note (pull before you use them):

- `bakeoff.py run <lane> finetune --init-from <checkpoint dir>` → GR00T's
  `--base-model-path` (weights + processor from the dir, fresh optimizer).
- `BAKEOFF_RUN_ROOT=/tmp/franka-sonic` forces the run root — both P11
  fine-tunes go to `/tmp` from step 0; home drained ~1.6 TB from outside
  overnight and is at ~850 GB.
- `stage_dataset` feeds the converter `out/export/*.hdf5` **and**
  `out/export/*/*.hdf5`, so a union demo folder can keep each source set in
  its own subdirectory with distinct shard basenames (the label pipeline keys
  clips by shard basename + demo name and names clips by the shard index).
- `harness/driver.sh` knows P11; a fresh driver skips P0–P10 (PASS) and runs
  it. A waiter in tmux window `driver7` starts that driver once `driver6` has
  exited with P10 PASS — you do not launch it.

**During P10** (Addendum A5 in `plan/prompts/P10.md`): WP 11.0–11.3 (union
folder, lane-A union dataset, union motion library, union tokens) are CPU work
and may run now, so P11's fine-tunes start the minute GATE P10 passes. Every
P10 rule from the 03:40/05:40/07:12 notes stands: 200 rollouts, top three per
lane by (successes, m6, m5, step), held-out 20–199, milestone vector first,
two numbers per lane, explicit paths on every command.

## 05:40 UTC (2026-09-05, P9 -> attempt 2) — lane B is done and 13/20 at 17500. Start P10's lane-B rows early on the idle devices.

**Read this first if you are attempt 2.** Nothing is broken. Three jobs are
alive and **must not be restarted**: lane A fine-tune **pid 2311371**
(`/tmp/franka-sonic/lane_a/2026-09-05_finetune`, at ~15 650/20 000, finishing
~07:40), the screening watcher **pid 2404198**, and the round-3 generation's
last worker. The RESUME POINTER v2 in STATUS is authoritative; the older one at
`STATUS.md:800` names dead pids by design (a mid-run storage switch), not by
failure. **The early stop stays OFF.** Both lanes run to `checkpoint-20000`;
WP 9.3's rule only *selects*.

**Where P9 stands.** Lane B has finished all 20 000 steps and its series is the
result of the whole bake-off so far:

| step | 2500 | 5000 | 7500 | 10000 | 12500 | 15000 | 17500 |
|---|---|---|---|---|---|---|---|
| lane B | 0/20 | 0/20 | 0/20 | 0/20 | 3/20 | 5/20 | **13/20 (progress 0.775, milestones 100/80/80/75/65/65)** |
| lane A | 0/20 | 0/20 | 1/20 | 0/20 | 0/20 | screening | — |

Lane B was still rising at 17 500, so screen `checkpoint-20000` as your first
priority — it may well be the best checkpoint of either lane.

### DO THIS with the idle devices (authorised, and it is not "starting the next phase")

Lane B's trainer has exited and generation is done, so devices are free while
lane A trains alone on 1,5 and screening uses one more. **Once B@20000 is
screened** — and only then, so the pick stays pre-registered — launch lane B's
**three top checkpoints at 200 rollouts plus the B-oracle row**, one device
each, on the free devices:

- These are P10 rows measured early, not a phase change: record them in STATUS
  as such, with `--rollouts 200`, the same eval binding as every screen, and the
  B-oracle through a **verified** export (`[check] VERDICT: OK`).
- **Lane A's training and the screening watcher keep absolute priority** for
  devices. Never take 1 or 5, and leave one device free for the watcher.
- Do not start a row you cannot finish; each is ~4.2 h.

The reason is load-smoothing, not wall-clock: with all eight rows starting at
once after lane A finishes, eight concurrent Isaac processes would contend
badly. Spreading four of them across the next two hours costs nothing and
de-risks the rest.

### When lane A reaches 20 000 (~07:40)
Screen its remaining checkpoints, then write the two literal lines from
`stopping_rule.py` and run the gate:

```
P9 BEST lane_a=<abs path>
P9 BEST lane_b=<abs path>
```

Then P10 per my 03:40 note: **top three per lane at 200 rollouts**, headline
only from the 200-rollout row of the pre-registered best-by-screen pick, every
row reported on all 200 *and* on held-out episodes 20–199, and **the
six-milestone vector leading each row rather than mean progress** — lane A's
7500 screen is the standing proof that a falling mean can accompany a policy
getting further.

Round-3 data (`shared/2026-09-04_demos-2`, eval-matched spawns) is generated;
its export → replay → `jointpos_screen` can proceed whenever it does not
compete with the above. It feeds P11, which does not start until P10's gate
passes.

## 03:40 UTC (2026-09-05, P9 -> P10) — the series oscillates; P10 evaluates the top THREE per lane, and the headline never comes from a screen

Both lanes have now produced successes (A 1/20 at 7500, B **3/20** at 12500),
and both series swing hard between adjacent checkpoints:

```
A   0.175 (0/20)  0.467 (0/20)  0.392 (1/20)  0.108 (0/20)
B   0.242 (0/20)  0.108 (0/20)  0.425 (0/20)  0.100 (0/20)  0.275 (3/20)
```

The swing is concentrated in **milestone 1**: "left reaches" goes 100 → 75 → 25
for A and 100 → 25 → 40 for B. A policy regressing on the *easiest* milestone
while deeper milestones improve is not ordinary binomial noise. GR00T N1.7
samples actions stochastically, so there is per-rollout variance on top of the
20-rollout sampling error, and checkpoint-to-checkpoint variance on top of that.

**Consequence, and it is the same lesson as P8's decoder ceiling:** the
best-of-eight checkpoint chosen from 20-rollout screens is a **selected
maximum**, not an estimate. Do not let any headline number in the report come
from a screening row.

### P10 changes (supersedes the top-2 instruction in my 23:40 note)

1. **Evaluate the top THREE checkpoints per lane at 200 rollouts**, not two.
   With eight devices free once training ends, the rows are: A-best, A-2nd,
   A-3rd, B-best, B-2nd, B-3rd, A-oracle, B-oracle = 8 rows, one device each,
   ~4.2 h wall-clock. **Priority order if devices are short:** the four
   canonical rows first (A-best, B-best, A-oracle, B-oracle), then the two
   2nd-place rows, then the two 3rd-place rows. Never start a row you cannot
   finish.
2. **The headline is the 200-rollout number for the pre-registered pick**
   (best-by-screen, chosen before the 200-rollout runs are seen). The 2nd and
   3rd rows are the spread, and that spread is itself a finding: report the
   range across the three, because it measures how much of "which checkpoint
   won" is luck.
3. **Report each row twice**: all 200 episodes, and episodes 20–199 only. The
   screens used seeds 0–19 of the same sequence, so 20–199 is genuinely held
   out from the selection.
4. **Lead every policy row with the milestone vector, not mean progress.** Lane
   A at 7500 shows why: mean progress *fell* 0.467 → 0.392 while the task got
   further than ever (milestone 4: 0 % → 40 %, first completed handover). For a
   staged task a falling mean can mean the policy stopped playing safe. Give
   the six-milestone vector for every row and treat mean progress as secondary.
5. Gate overrides stay `P10_MIN_ROLLOUTS=200 P10_MIN_ORACLE=200`.

### For the record, on the early stop
Your own 03:2x STATUS entry makes the point better than my estimate did: lane A
was one non-improving screen from being stopped at 12500, with its working 7500
checkpoint already in hand. Keep that sentence in the report — it is the
evidence for the 23:40 decision, not a hypothetical.

## 02:10 UTC (2026-09-05, P9) — your RESUME POINTER is stale and it is the one thing that can lose the night's work

Series looks good — A 0.175 → 0.467, B 0.242 → 0.108 → 0.425, both lanes past
the grasp wall and both now stalling at milestone 4. Horizon is ruled out: the
oracles finish all six milestones in a median of 722 (A) / 733 (B) steps against
the 1500-step horizon, so the policy has 2.05x the oracle's budget and the right
arm is underfit, not out of time. Nothing to change there.

**The problem is the handover to attempt 2.** `plan/STATUS.md:800`, the entry
literally headed "RESUME POINTER for attempt 2 (single place to look)", still
names the pre-switch jobs:

| it says | actually |
|---|---|
| lane A pid 2006685, `~/runs/.../lane_a/2026-09-04_finetune-3` | **pid 2311371**, `/tmp/franka-sonic/lane_a/2026-09-05_finetune` |
| lane B pid 2006968, `~/runs/.../lane_b/2026-09-04_finetune-3` | **pid 2350298**, `/tmp/franka-sonic/lane_b/2026-09-05_finetune` |
| watcher pid 2044639 | **pid 2404198** (restarted 01:13) |

Every one of those pids is dead. Attempt 2 starts at 05:39, reads that entry,
finds a dead pid and a home run folder whose newest checkpoint is 5000 (lane A)
or 7500 (lane B), and the obvious wrong conclusion is "training died, restart
it" — which throws away everything since 21:56 and makes the two lanes
non-comparable. STATUS is append-only, so do not edit line 800; **append a new,
explicitly superseding RESUME POINTER now**, before anything else, saying:

- the live pids and their `/tmp` run folders, per the table above, measured with
  `ps` rather than copied from an earlier entry;
- that `plan/STATUS.md:800`'s pointer is **superseded** and its pids are dead by
  design (the storage switch), not by failure;
- that each lane trained across two run folders and the pre-switch folders on
  home are kept, not stale;
- that no trainer, watcher or generation job may be restarted, and the early
  stop stays off;
- the live round-3 generation pid, measured now (the 2211731 in the old entry is
  also worth re-checking);
- that P9 ends when both lanes reach `checkpoint-20000` (A ~07:20, B ~05:28),
  every checkpoint is screened, and the two `P9 BEST` lines are written from
  `stopping_rule.py`.

**Also write the pid files the harness convention expects** — both
`/tmp/franka-sonic/<lane>/2026-09-05_finetune/out/finetune.pid` and
`.pgid` are missing, so there is currently no on-disk record of the live
trainers outside STATUS prose. Write them from `ps` now.

Round-3 generation is on track for the record: worker 0 at 26/75 successful
demos (34.7 % hit rate), 64 needed per worker, so ~05:07 against the 06:57
deadline. Leave it alone.

## 00:45 UTC (2026-09-05, P9) — the switch broke the watcher. Fix it before lane A's 7500 (~01:46).

Lane A's resume is correct and verified (5095/20000, LR 8.941e-05 = cosine at
5000) — good work, and the LR check is exactly the evidence I wanted. But the
switch has a consequence nobody wrote down:

**`/tmp/franka-sonic/p9/watcher.py:33-35` hardcodes the home checkpoint dirs.**
Lane A now writes to `/tmp/franka-sonic/lane_a/2026-09-05_finetune/out/checkpoints`,
so the watcher will never see `checkpoint-7500` and lane A's screening stops
silently after 5000 — no error, just an empty series. Lane B inherits the same
break the moment it switches.

Fix before lane A's 7500 lands (~01:46 at 1.76 s/it from 5253):

1. Make `CKPTS[lane]` a **list** of directories and scan all of them,
   deduplicating by step (the pre-switch checkpoints 2500/5000 stay on home, the
   post-switch ones are on /tmp — the series must span both).
2. **Editing the file is not enough** — `python3 watcher.py` (pid 2044639)
   loaded it at 22:04 and will not pick up the change. Restart it: `touch
   /tmp/franka-sonic/p9/watcher.stop`, wait for it to exit (it exits only when
   nothing is running, which is the behaviour you want), edit, relaunch. Its
   `watcher_state.json` makes the restart safe — already-screened steps are not
   re-screened.
3. After the restart, confirm from the log that it enumerates both roots per
   lane, and that `series.txt` still holds the three existing rows.
4. Add lane B's post-switch dir at the same time you switch lane B, so you
   restart the watcher once rather than twice.

Also: `watcher.py:87` builds the eval run root from
`~/runs/franka-sonic/<lane>` — leave that on home. Eval outputs are ~150 MB and
home has 1613 GB; only the 34 GB checkpoints needed moving.

**Screening is the critical path now.** The whole point of the 20-rollout series
is the trend, and the trend is what decides whether the 20 000-step budget was
ever the right call: A@2500 0/20 progress 0.175, B@2500 0/20 progress 0.242,
B@5000 0/20 progress **0.108** — lane B went *down*, and its "left reaches" rate
fell 95 % → 20 %. If A@5000 and B@7500 confirm a decline rather than noise, say
so plainly in STATUS; that finding outranks finishing the step budget.

## 00:30 UTC (2026-09-05, P9) — the mirror is done (I ran it); now switch lane A. And your poll filter is hiding my notes.

**Your `poll.sh` line 54 is `grep -a "^lane_"`**, so every orchestrator line I
inject into `series.txt` starting with `#` is filtered out — that is why the
00:15 note did not reach you. Either widen that grep or read
`plan/ORCHESTRATOR_NOTES.md` at the top of every poll. I am prefixing this one
`lane_ORCHESTRATOR` so your existing filter catches it (the stopping rule's
regex requires `lane_[ab]\s+step`, so it ignores such lines safely).

**Done for you, already running, do not redo:** `/tmp/franka-sonic/p9/mirror.sh`
(nohup setsid, log `mirror.log`, `df.log`) copies every settled checkpoint to
`/tmp/franka-sonic/p9/ckpt_mirror/<lane>/checkpoint-N` and samples `df` every
2 min. All four existing checkpoints are mirrored and size-verified at 34 GB
each. It copies only, never moves, never deletes.

**Correction to my 00:15 note:** `resume_from_checkpoint` is a **bool**, not a
path (`gr00t/configs/training/training_config.py:76`) — HF Trainer resumes from
the *last checkpoint inside `output_dir`*. So the new `/tmp` output dir must
already contain the checkpoint to resume from. That is what the mirror is for.
`save_only_model` is False (default), so the checkpoints carry optimizer,
scheduler and RNG state — `check_resume_compatibility` exists precisely to
enforce that pairing, and 34 GB per checkpoint (≈6 GB bf16 weights + ≈24 GB
optimizer) confirms the state is there.

### Switch lane A NOW, lane B at its 7500 boundary

Home was 1618 GB at 00:26 and is falling ~900 GB/h; the 600 GB floor arrives
about 01:30. Lane A's `checkpoint-5000` has landed and is mirrored, so
switching costs only the ~300 steps since it. Lane B's 7500 lands ~00:53 —
wait for it (that costs nothing) **unless home drops below 1100 GB first**, in
which case switch lane B from 5000 immediately.

Per lane, at its boundary:

1. `mkdir -p /tmp/franka-sonic/<lane>/2026-09-05_finetune/out/checkpoints` and
   `cp -a` the mirrored `checkpoint-<N>` into it (the mirror is the source; the
   home copy stays untouched).
2. SIGTERM the recorded pgid, SIGKILL the python by pid, verify with `ps`.
3. Relaunch the **same command** from that lane's `cmd.sh` with two changes:
   `--output-dir /tmp/franka-sonic/<lane>/2026-09-05_finetune/out/checkpoints`
   and the resume flag (confirm its exact spelling with
   `launch_finetune --help`; the field is `resume_from_checkpoint`). Everything
   else — base model, dataset path, modality config, `--max-steps 20000`,
   `--save-steps 2500`, `--global-batch-size 32`, colour jitter, 4 workers,
   `--save-total-limit 12`, the device set, the master port — stays identical.
   Machine-diff it against `cmd.sh` as you did at launch and record the diff.
4. **Verify the resume took** before walking away: the progress bar must start
   near N/20000, not 0/20000, and the first logged `learning_rate` must sit on
   the cosine curve at step N (step 2500 logged 9.847e-5 and it decreases from
   there). If it starts at 0, kill it, restore the home-side trainer if it is
   still alive, and write `BLOCKED: resume did not restore global step`.

**Staggering the two lanes is fine** — I withdraw the "both or neither" wording
from 00:15. Storage location does not touch the optimisation; a resume restores
optimizer, scheduler and RNG state, so each lane's trajectory continues as if
uninterrupted. What must stay identical between lanes is the recipe, not the
filesystem.

Record in STATUS, per lane: the switch step, the old and new run folders, the
verified resumed step and LR, and that the pre-switch folder on home is kept
(never deleted). The P10 report must say each lane trained across two run
folders and why.

## 00:15 UTC (2026-09-05, P9) — URGENT: home is draining from outside. Move both fine-tunes to /tmp at their next checkpoint boundary.

**Measured, not projected.** Home free: 2495 GB at 21:40, 2472 at 22:06, 1859 at
00:07, **1848 at 00:09** — 11 GB in 40 s. Our own writes over 22:06→00:07 were
three checkpoints (~102 GB), so roughly **255–900 GB/h is being consumed by
other users** of the shared `/research` Lustre volume (71 TB, 98 % full). We
still need ~440 GB for the remaining 13 checkpoints. At the low end of that
drain we cross the 600 GB floor around 03:20 UTC; at the high end, before 02:00.
Both are before lane B finishes (~05:50) and long before lane A (~07:55).

`/tmp` (instance-local `/dev/md0`) has **10.7 TB free**. That is the answer.

### DO THIS — a clean switch, not a crash

`resume_from_checkpoint` is supported end to end (`gr00t/experiment/experiment.py:344`,
`launch_finetune.py:127`), so HF Trainer restores optimizer, LR-scheduler state
and step count: a stop-and-resume is a **faithful continuation of the same run**,
not a schedule restart, and does not confound the comparison.

1. **Mirror what exists now.** Copy (never move, never delete) every existing
   `checkpoint-*` to `/tmp/franka-sonic/p9/ckpt_mirror/<lane>/`, and each new one
   as it lands. ~102 GB today, minutes on local disk. This alone makes an ENOSPC
   death recoverable instead of fatal — do it first, before anything else.
2. **Switch each lane at its next checkpoint boundary** — lane A at
   `checkpoint-5000` (~00:25), lane B at `checkpoint-7500` (~00:53). At the
   boundary: SIGTERM the recorded pgid, SIGKILL the python by pid, verify with
   `ps`, then relaunch the *same command* with `--output-dir` under
   `/tmp/franka-sonic/<lane>/` and `resume_from_checkpoint` pointing at that
   checkpoint. Waiting for the boundary costs ~0 work; stopping mid-interval
   throws away up to 70 min, so do not stop early.
3. **Verify the resume took**: the trainer must log the resumed global step
   (5000 / 7500), not 0, and the first logged LR must match the cosine schedule
   at that step (~9.85e-5 was step 2500's value; it decreases). If it restarts
   at step 0, kill it, say so in STATUS, and fall back to letting the home-side
   run continue while you mirror aggressively.
4. **Sample `df -BG ~` every 5 min** into `/tmp/franka-sonic/p9/df.log` with a
   timestamp, so the drain rate is a measured series in the report rather than
   my two-point estimate.
5. **Nothing on home is deleted or moved** — the existing run folders stay
   exactly where they are, marked in STATUS as the pre-switch half of each run.
   At the end of the phase, copy the winning checkpoints back to
   `~/runs/franka-sonic/<lane>/final/p9/` if home allows, else `NEEDS-COPY`.
6. Point the round-3 generation's remaining stages at `/tmp/franka-sonic` too.
   It is small (~10 GB) but there is no reason to spend home on it now.

**Both lanes switch, or neither** — never one lane on home and one on /tmp, the
same rule that governed the launch. Record the switch step per lane in STATUS;
the P10 report must state that each lane trained across two run folders and that
the resume restored optimizer and scheduler state.

If the drain stops on its own, still finish the switch once started — a
half-switched pair is worse than either end state.

## 00:05 UTC (2026-09-05, P9 + round-3 prep) — the failure is reach-then-stall, and the spawn box is 1.5 % of the training volume. Start the targeted generation now.

**What both lanes actually do at checkpoint-2500** (I read the videos and the
per-episode jsons): the left arm reaches the block in 85 % (A) / 95 % (B) of
episodes, then **stalls over it and never closes the gripper** — every episode
ends on `termination_reason: horizon` at 1500 steps, none on failure. Milestone
2 ("left lifts") is where the whole thing dies: A 85/10/10/0/0/0, B
95/25/20/5/0/0. Grippers are binary 0/1 in the dataset against the client's 0.5
threshold, so the encoding is not the bug; images are 360x640 on both sides, so
that is not it either. This is a precision failure at the grasp, which is the
expected shape of an underfit policy at 12.5 % of its step budget — it is not
evidence of a broken harness.

**The structural finding, measured** (`shared/2026-09-04_demos/out/coverage.json`
against `dual_stack_env_cfg._block_spawn_event`):

| | training (891 eps) | evaluation |
|---|---|---|
| x half-range | 0.0893 m | 0.015 m |
| y half-range | 0.0900 m | 0.015 m |
| yaw half-range | 0.750 rad | 0.400 rad |

The evaluated box is **1.5 % of the training spawn volume** (0.168 x 0.167 x
0.533), i.e. **~13 of 891 episodes start where the policy is scored**. Round 1
was worse in the same way (~4 of 76), so this is not a round-2 regression — but
it is the clearest lever we have, and P7 dropped exactly this ("dense +/-3 cm
pass") to protect the episode-count equality invariant. Widening the generation
box was my call and it was right for robustness; the cost is that 98.5 % of the
data trains a distribution nobody measures.

### DO NOW — round-3 dataset, broad + targeted (this does not touch P9)

Generate a **second, eval-matched demo set** alongside the running fine-tunes:

- `MIMIC_SPAWN_XY=0.015 MIMIC_SPAWN_YAW=0.4 MIMIC_ARM_NOISE_STD=0.02` — the
  evaluation box exactly, arm noise matched to the eval's 0.02 rad (the wide set
  already carries 0.05 rad diversity, so the union keeps it).
- 1024 episodes through the same P7 pipeline (sources -> annotate -> fixsignals
  -> generate -> export -> replay), then `harness/data/jointpos_screen.py` on
  every episode, exactly as P7 did. Expect ~890 to pass.
- New run folder under `~/runs/franka-sonic/shared/`; **nothing about the P7 set
  changes and nothing is deleted**.
- Round 3 trains on the **union**: 891 wide + ~890 narrow, one dataset, both
  lanes identical. Lane B relabels the union with the P8 decoder through a
  **verified** export (P8 did 891 in 10 min).

**Budget guard — P9 owns the node.** Baseline step times are **1.74 s/it (A)**
and **1.42 s/it (B)**. Cap the generation at **2 GPUs** and pick the worker
count so both stay under **2.00 / 1.65 s/it**; sample both every 10 min and
halve the workers if either is over for two consecutive samples. If it cannot
be kept under those numbers at any worker count, stop the generation and say so
— P9's trainers have priority over round-3 prep, always.

Detach it (`nohup setsid`) and record the pid/pgid and the run folder in
STATUS: your session ends ~05:39 and attempt 2 must pick this up, not restart
it. Screening and the two fine-tunes keep priority for devices at all times.

### Also record, for the P10 report
The reach-then-stall milestone profile and the 1.5 % spawn-overlap number
belong in the report as the round-2 headline finding, whatever the final
success counts are.

## 23:40 UTC (2026-09-04, P9/P10) — early stop OFF, 30k ceiling withdrawn, 22:50 oracle note withdrawn

Read this one even if you read nothing else tonight. Three decisions, all mine,
all final for this round.

**1. WP 9.3's early stop is OFF. Both trainers run to 20 000 regardless of the
screens.** The rule stays as the *selection* rule (best = highest success/20,
ties to progress, then the earlier checkpoint) and the STATUS entry says where
it *would* have fired — but no trainer is stopped before `checkpoint-20000`.
Why: (a) a 20-rollout screen has SE ≈ 0.1, so the running best is a biased
maximum, and two consecutive "no-beats" against it happen by chance in roughly
40–50 % of genuinely rising curves (best 9/20 drawn at p = 0.35; the next two
checkpoints at p = 0.40 and 0.45 both fail to beat 9 with probability
0.755 × 0.591 = 0.45); (b) the schedule is cosine to zero at 20 000 (LR at
2500 = 9.85e-5 matches cosine over 20k with 5 % warmup exactly), and the low-LR
tail is where these fine-tunes usually gain; (c) a false stop is irreversible
and asymmetric between the lanes; (d) GPU-hours are not the constraint.
Earliest point the old rule could have fired: lane B's 10 000 screen at ~02:17
UTC — so this is in force before that.

**2. The 30 000 ceiling in my 22:30 note is withdrawn.** With the LR at zero at
20 000, "continue warm to 30 000" is a schedule restart, i.e. a recipe change
and a confound. 20 000 is the run for both lanes; select the best of the eight.
If a lane's series is still rising at 20 000, say so in WP 9.4 — that is the
round-3 headline, not something to fix tonight.

**3. The 22:50 oracle note is withdrawn.** P10 runs its rows in parallel on
eight free devices and every 200-rollout row costs ~4.2 h on one device, so the
oracles were never on the critical path. Leave devices 0/3/4 idle; screening
keeps its two.

**For P10 (record now, act then):** (i) evaluate the top-**2** checkpoints per
lane at 200 rollouts, not just the winner — the headline stays the
pre-registered pick (best-by-screen), the runner-up is a robustness row, and
they run on separate devices so P10's wall-clock does not change; (ii) report
each row twice: all 200 episodes, and episodes 20–199 (held out from the
20-rollout screens, which used seeds 0–19 of the same sequence); (iii) the
gate overrides stay `P10_MIN_ROLLOUTS=200 P10_MIN_ORACLE=200`.

Acknowledge this note with one STATUS line so the next attempt inherits it;
the session clock (8 h from 21:39) ends before lane A's last checkpoint, and
attempt 2 must not restart training or re-apply the early stop.

## 22:50 UTC (2026-09-04, P9/P10) — run P10's two oracle rows NOW on the idle devices

Four devices (0, 2, 3, 4) are idle until the first checkpoints land at ~23:15,
and they will only be partly busy after that: a 20-rollout screen costs ~22 min
and checkpoints arrive every ~70 min per lane, so screening uses roughly half
the spare capacity.

P10 needs four rows at 200 rollouts, and **two of them do not depend on P9's
outcome at all**: the A-oracle (recorded joint targets replayed) and the
B-oracle (the P8 winner's tokens through its decoder). Run both tonight, on the
idle devices, at the full 200 rollouts over episodes 0–199 of the P7 set:

- `oracle_a` on 2 devices, `oracle_b` on 2 devices, both `--rollouts 200`.
- The B-oracle must use a **verified** export of the P8 winner — re-export and
  confirm `VERDICT: OK` first if the one you have is not recorded as verified.
- Give screening priority: if a checkpoint is waiting and no device is free,
  let the oracle runs finish their current episode budget rather than starting
  another, and resume them in the gaps. Screening is on the critical path;
  oracles are not.

That moves ~1.5 h of P10 off tomorrow's critical path and turns idle capacity
into two of the four final rows. Record both runs in STATUS as P10 rows measured
early, with the export verification path for the B-oracle, so the report can use
them directly instead of re-running.

On the rank count, for the record: 2 GPUs per lane is not a preference, it is the
only configuration that has ever completed a fine-tune in this container (4 ranks
faulted tonight, 3 was never smoke-tested because you went straight to the proven
one). Do not restart the running trainers to try 3 — an hour of progress is worth
more than the throughput.

## 22:30 UTC (2026-09-04, P9) — my call on the budget, and you have four free devices

You stated the trade instead of hiding it, which is what I asked for. Two things:

**1. Screen concurrently — four devices are free.** Your 20:xx reasoning ("with
all 8 devices training there is no spare device to screen on") was written for
the 4+4 plan. You are running 2+2: the allocator shows lane_a on 1,5 and lane_b
on 6,7, with **0, 2, 3, 4 idle**. So WP 9.2 works as designed: screen each
`checkpoint-N` as it lands, 20 rollouts, on the free devices. Checkpoints arrive
about every 70 min and a screen costs ~22 min, so screening never becomes the
critical path — and it restores WP 9.3's early stop, which your note correctly
said was lost. It also means we see the learning curves tonight instead of
tomorrow.

**2. The budget: 20 000 steps stands as the floor, 30 000 as the ceiling,
decided by the curve.** At 32 samples per step, 20 000 steps is 640 k samples
against 722 k frames — 0.89 epochs, slightly *less* per-datum training than
round 1's 1.25 epochs. So do not treat 20 000 as the end:

- Let both lanes run to 20 000 as launched (no restart, you would throw away an
  hour).
- If a lane's screened series is **still improving** at 20 000 (its last two
  checkpoints did not both fail to beat its best), continue that lane warm from
  its last checkpoint to 30 000, same recipe, same save cadence.
- If a lane's series has flattened, stop it there — that is the rule working.
- Both lanes get the same rule and the same 30 000-step ceiling. Realised steps
  may differ between lanes; that is the stopping rule doing its job, not an
  asymmetry, and the report should say which lane stopped where and why.

At 1.42–1.67 s/it a continuation costs ~4–4.7 h, so the ceiling is affordable
even with P10's 200-rollout rows after it. If both lanes flatten early, we
finish sooner and nothing is lost.

Keep the trainers detached as you have them, and keep reporting each screened
checkpoint as a one-line series entry.

## 22:00 UTC (2026-09-04, P9) — 4 ranks is the failure, not your launch. Fall back by measurement, in this order

Both fine-tunes died with `CUDA error: an illegal memory access was encountered`
inside the NCCL watchdog — P1's finding at a larger rank count. My 16:00 note
asked for 4 GPUs per lane; that is what broke, so this supersedes it.

**Decide by smoke test, one at a time, not both lanes at once** (two 4-rank jobs
starting together across all 8 devices is itself a variable):

1. **3 ranks, one lane alone, 200 steps.** If it survives, launch both lanes at
   3 GPUs each and keep 2 devices for screening. 30 000 steps at the ~1.1
   steps/s that implies is ~7.5 h.
2. **If 3 ranks faults: 2 ranks, one lane alone, 200 steps.** 2 ranks is the only
   configuration this container has ever completed a fine-tune on (round 1 and
   P6, 0.749 steps/s). Then both lanes at 2 GPUs each, 4 devices free for
   screening, 30 000 steps ≈ 11 h — long, but it is an overnight window and the
   spare devices let you screen every checkpoint as it lands instead of
   afterwards.
3. **If 2 ranks also faults**, something changed since P6: check that
   `NCCL_P2P_DISABLE=1` and `NCCL_SHM_DISABLE=1` are actually in the launched
   environment (`stage_finetune` sets them with `setdefault`, so an inherited
   value would win), and that no Isaac process from P8 still holds memory on the
   devices you claimed. Then retry 2 ranks.

**Do not reduce the step budget to make a bigger rank count fit.** Both lanes
must get the same number of steps, and 30 000 is the number in the plan. If the
only working configuration makes 30 000 steps too slow for the window, tell me
in STATUS with the measured rate and I will decide the trade — do not silently
choose 20 000.

Whatever you land on: identical rank count, step budget, save cadence and
`save-total-limit` for both lanes, and the three-token `cmd.sh` diff verified
before they run. Record the smoke-test result and the final configuration in
STATUS so P10's report can state what hardware path the numbers came from.

## 21:35 UTC (2026-09-04, P8) — export non-determinism: excellent diagnosis; now contain it everywhere, not just in the ceiling script

Your finding is the most consequential of round 2: `export_onnx` is
non-deterministic — two exports of the same checkpoint differ in exactly the
g1 encoder's ten tensors by ~15 % relative, the two encoders then agree on 0.0 %
of token rows, and the incidence is ~1/3 and concentrated in exports that ran
concurrently with other exports. The `check` clause that kept "failing" is an
export verification and it was right every time (9 of 9 on reproducibility).

Three things follow, and they matter beyond P8:

1. **Every ONNX that anything downstream consumes must carry a recorded
   `VERDICT: OK`** — not only ceiling tests. That includes the export used for
   the full 891-episode relabel, the export lane B's policy server loads in P9's
   screening, and the one it loads in P10. Re-export and re-verify rather than
   reusing an unverified artifact, and put the verifying run's path in STATUS
   next to each. A bad export in the relabel would poison lane B's entire
   training set, which is exactly the failure mode round 1 escaped by luck.
2. **Serialise exports.** Since 11 of 12 bad draws happened while another export
   ran, take a simple lock so only one `export_onnx` runs at a time. An export
   costs ~1 min; the parallelism buys nothing and evidently costs correctness.
   Keep the verification anyway — serialising is a mitigation, not a proof.
3. **Find the mechanism if it is cheap** (30 min, not more): ten tensors of one
   sub-module differing suggests the export captured a partially-loaded or
   concurrently-mutated module — a shared cache or temp path between concurrent
   exports, a `torch.load` racing a writer, or a module registry keyed on
   something not unique per process. If it is not cheap, stop and rely on
   verification plus serialisation; write the hypothesis down for the next round.

For the record: round 1's 17 exports all verified, so P1–P6's numbers stand.
Say so explicitly in the P10 report, along with this bug and how it was caught —
a reviewer will otherwise wonder whether round 1's B-oracle was measured through
a good export.

Selection: proceed on verified-export numbers only, as you are doing. If the
eligible set is thin, re-export and re-test the leaders rather than admitting an
unverified number.

## 20:15 UTC (2026-09-04, P8) — RULE: a checkpoint whose labelling check fails cannot be the winner

Three more `label_tokens` rc=1 at 20:06–20:07 (runs -30, -31, -32), on the
`last.pt` checkpoints — the ones most likely to be selected. Meanwhile -21, -22
and -25 passed cleanly. So the mismatch is checkpoint-dependent, and P8 is about
to choose among candidates of both kinds.

Until the mismatch is explained, apply this rule, which is safe either way and
costs you nothing given six variants and ~7900 iterations each:

**Any checkpoint whose `label_tokens` run reports `[check] VERDICT: MISMATCH`
is ineligible to be P8's winner.** Select only from checkpoints whose full
labelling — validate, obs, encode, check — returned rc=0. If that leaves fewer
than three candidates for the 60-episode round, test more intermediate
checkpoints from the leaders rather than admitting a mismatching one.

Report in STATUS, as one line: how many of the ceiling tests so far came from
mismatching labelling runs, and which reported numbers those were. Those numbers
stay in the record but are marked unreliable and are not used for selection.

This is not a substitute for the diagnosis I asked for at 18:45 — it is what
makes the phase safe to finish while the diagnosis is still open. If you find
the mismatch is benign and prove it on one clip, lift the rule, say so, and the
excluded numbers come back into play.

If the rule leaves you with no eligible checkpoint at all, that is itself the
finding: stop selection, write it plainly, and treat the labelling path as the
phase's blocking bug.

## 18:45 UTC (2026-09-04, P8) — ANSWER THE MISMATCH BEFORE YOU DECLARE A WINNER

`label_tokens` has now failed rc=1 five times (17:46, 17:58, 18:07, 18:31, and
at least one more) with `[check] VERDICT: MISMATCH`, while `oracle_b` runs
straight afterwards and reports ceiling numbers. That means the ceiling series
is being produced from labelling runs whose own self-check says the runtime
observation does not match the env observation. Either the check is wrong or the
tokens are wrong, and P8 cannot declare a winner until you know which.

This is the last thing I will ask of P8 before selection. In STATUS, answer:

1. **Does the ceiling pipeline consume tokens from a run whose check failed?**
   If `encode` completes and `check` is the only failing step, say so plainly.
2. **Is the mismatch benign?** The log attributes it to `joint_vel` being a
   50 Hz finite difference at runtime versus PhysX's in the env. Prove it on one
   clip: plot or tabulate the per-frame difference and show that the large values
   sit only in the frames where that term is expected to differ (right after a
   reset, or at a velocity spike), and that the token sequence produced with and
   without the offending term is effectively identical. If the difference lives
   in `joint_pos_rel` or in the command terms instead, the tokens are wrong and
   every ceiling number so far is void.
3. **Why round 2 and not round 1?** Round 1 passed this check every time. Name
   what changed: the 891-episode reference set, `--max-episodes`, the new
   screened-episode skip, or the clip that the check happens to draw.

If the answer is "benign, and here is the proof", record it and carry on with
selection. If it is "the tokens are wrong", stop the ceiling series, fix it, and
re-run the tests on the leaders only — you have the iterations to spare, since
the trainers reach ~7900 against the 848 round 1 needed.

Do not relax the tolerance, and do not let the phase end with five silent rc=1s
in the log and a winner declared on top of them.

## 18:25 UTC (2026-09-04, P8) — select the winner on 60 episodes across the top THREE, not just confirm one

Your non-monotonicity finding (A: 18 → 14 → 6 over iterations 500/1000/2000; F:
13 → 17; replay error flat at 0.045–0.051 throughout and correlated with
nothing) is the most useful thing this phase has produced, and your own method
caveat is exactly right: the maximum of ~15–20 tests on one 20-episode set is
biased upward.

Strengthen the fix one step: **run the 60-episode test on the top three
checkpoints, not only on the leader, and select on those 60-episode results.**
Confirming a single pre-selected maximum still inherits the selection; comparing
three candidates on a larger, common set does not. Then run the
protocol-identical 20-rollout test on the chosen one last, so the gate reads the
stated bar. Report all three 60-episode numbers, not just the winner's.

If two candidates come out within a few episodes of each other on 60, prefer the
one from the **earlier** iteration and the plainer recipe (fewer knobs changed
from jp20) — it is the less over-fitted choice and it keeps P9's story simple.

Still open from my 18:10 note: `label_tokens` failed rc=1 three times (17:46,
17:58, 18:07) with `[check] VERDICT: MISMATCH`. Ceiling tests have run since, so
either those were checkpoint-specific or the path differs. Say in STATUS which
it was and whether any reported ceiling number came from a run whose check
mismatched — if one did, drop that number from the series.

Also record for the report: round 1's 19/20 was itself the maximum of seventeen
ceiling tests, so it carries the same upward bias. The round-2 winner's
60-episode number is the first unbiased decoder figure this project will have.

## 18:10 UTC (2026-09-04, P8) — `label_tokens` has now failed twice with `[check] VERDICT: MISMATCH`; treat it as a regression, not noise

`label_tokens-12` (17:46) and `label_tokens-14` (17:58) both exit 1 at the
`check` step. From `-14`'s log: `raw_action_vs_env_max_abs_diff` **14.78** with
mean 0.11, per-term proprio max `joint_vel` 0.64, `last_action` 1.42. Round 1
passed this check every time (`VERDICT OK`), so something round 2 changed. Until
it is understood, **no ceiling number from a mismatching labelling run is
trustworthy** — the encoder may be reading different inputs than it trained on,
and that is exactly the class of bug that cost round 1 its first two attempts.

Strongest hypothesis, cheapest to test first: **`--max-episodes` mis-aligns the
clip set against the dataset provenance mapping.** You added that flag today and
it narrows both the `obs` step's clip set and `encode`'s own limit. If the two
narrowings pick different subsets — different order, different filter, or the
screened-episode skip applied on one side only — the check compares runtime obs
from clip *i* against env obs from clip *j*. That produces exactly this
signature: a huge max difference with a small mean, because most frames still
line up. Test: run one `label_tokens` with `--max-episodes 0` (all 891, round-1
behaviour) and see whether the check passes. If it does, the flag is the bug.

If it passes at 0 and fails at 20, fix the alignment, re-run the ceiling tests
whose labelling mismatched, and say in STATUS which ceiling numbers were
affected. If it fails at 0 as well, the cause is in the round-2 reference set or
the runtime path: compare a single clip's runtime `joint_vel` and `last_action`
against the env's, frame by frame, the way P5's trace did — and remember the
note in the log says `joint_vel` is a 50 Hz finite difference at runtime versus
PhysX's in the env, so check whether the 14.78 lives in frames right after a
reset, where that difference is largest.

Do not relax the check's tolerance to make it pass. If the difference turns out
to be benign, prove it on one clip and write the proof in STATUS.

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
