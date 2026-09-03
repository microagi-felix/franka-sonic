# Orchestrator notes for the running phase agent (P5)

Echoed by every `harness/bakeoff.py` call. Newest first. Act on them; log what you did in STATUS.md.

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
