# Lane B — dual-FR3 in GMR (General Motion Retargeting)

Registers the `dual_fr3` embodiment in the GMR clone at `~/code/upstream/GMR`
(@`bb1bbe4`) so BONES-SEED human motion can be retargeted onto the rig and fed
to the SONIC motion library (P2, `plan/PLAN.md`).

Canonical copies of everything live here in the repo; `register_dual_fr3.py`
installs them into the GMR working tree. **Never `git commit` in
`~/code/upstream/GMR`** — it is an upstream clone (AGENTS.md rule n / the P2
prompt); `git diff` there is the record of what we changed.

| file | what |
|---|---|
| `register_dual_fr3.py` | idempotent installer, six touch points, `--check` for a dry run |
| `smplx_to_dual_fr3.json` | the arms-only IK config (canonical copy) |
| `smpl_pkl_to_smplx_npz.py` | BONES-SEED SMPL pkl -> GMR SMPL-X npz adapter |

```bash
python3 harness/lane_b/gmr/register_dual_fr3.py            # install / re-install
python3 harness/lane_b/gmr/register_dual_fr3.py --check    # exit 1 if not installed
```

## Status (2026-09-03)

**Registered and constructible; not runnable end to end — the SMPL-X body models
are missing.** See "Blocker" below. What was verified on the pod without them:

```
$ cd ~/code/upstream/GMR && PYTHONUSERBASE=$HOME/env/pyuser-sonic /isaac-sim/python.sh -c \
   "from general_motion_retargeting import GeneralMotionRetargeting as G; \
    r = G(src_human='smplx', tgt_robot='dual_fr3', actual_human_height=1.66, verbose=False); \
    print(r.xml_file, r.model.nq)"
/home/felixminzenmay/code/upstream/GMR/general_motion_retargeting/../assets/dual_fr3/dual_fr3.xml 14
```

and a synthetic retarget (a hand-built `{joint: (pos, quat_wxyz)}` frame dict of
the shape `utils/smpl.py::get_smplx_data_offline_fast` returns, so no body model
is needed) over five stylised standing poses: link7 position error **≤ 0.5 mm**
once the arms have travelled there, joint 4 lands at −1.9 … −2.6 rad (inside its
[−3.0421, −0.1518] range), and the pelvis pin holds — moving the synthetic human
from world (0, 0, 0.92) to (3, −2, 0.95) leaves the retarget bit-identical.

## Touch points the installer patches

1. `general_motion_retargeting/params.py` — `ROBOT_XML_DICT`,
   `IK_CONFIG_DICT["smplx"]`, `ROBOT_BASE_DICT` (`"base"`),
   `VIEWER_CAM_DISTANCE_DICT` (2.0, viewer only).
2. `scripts/smplx_to_robot.py` — `dual_fr3` into the hardcoded `--robot`
   `choices` list.
3. `general_motion_retargeting/ik_configs/smplx_to_dual_fr3.json` — copy of the
   canonical config here.
4. `scripts/smplx_to_robot_dataset.py` — new `--no-denylist` flag that switches
   off **both** silent clip filters (`assets/hard_motions/{0,1}.txt` and the
   filename-substring list `["BMLrub", "EKUT", "crawl", "_lie", "upstairs",
   "downstairs"]`). Default behaviour unchanged. BONES-SEED clip families hit
   `crawl` / `_lie`, so a 1:1 sweep needs the flag.
5. `general_motion_retargeting/motion_retarget.py` — **the FR3-specific fix.**
   `mink.Configuration(model)` starts at `qpos0` = all zeros; the FR3's joint 4
   range is [−3.0421, −0.1518], so zero is *outside* its limits and the first
   `solve_ik` dies with

   ```
   mink.exceptions.NotWithinConfigurationLimits: Joint 3 (left_fr3_joint4)
   violates configuration limits -3.0421 <= 0.0 <= -0.1518
   ```

   The patch seeds the configuration from the MJCF's first keyframe when the
   model has one (ours is `<key name="home">`) and is otherwise a no-op, so no
   upstream robot changes behaviour. Every humanoid GMR ships has 0 inside every
   joint range, which is why upstream never hit this.
6. `assets/dual_fr3/` — `dual_fr3.xml` -> the *installed* MJCF under
   `~/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf/`,
   and `dual_fr3_assets` -> `~/code/upstream/mujoco_menagerie/franka_fr3/assets`
   (the MJCF's `meshdir`). Symlinks, so the GMR model tracks whatever
   `harness/lane_b/install_gear_sonic.sh` last installed; a symlink to the whole
   `mjcf/` directory would have exposed `g1_29dof_rev_1_0.xml` and `h2.xml` to
   GMR as well, hence the per-file link.

## The IK config, and every guess in it

`smplx_to_dual_fr3.json`. JSON has no comments; this is the comment file. The
schema is the one annotated in GMR's `DOC.md`: robot **body** name ->
`[human_body, pos_weight, rot_weight, pos_offset_xyz, rot_offset_quat_wxyz]`.

```
robot_root_name          "base"     the fixed root body of dual_fr3.xml
human_root_name          "pelvis"
ground_height            0.0
human_height_assumption  1.66
use_ik_match_table1      true       (table2 is an inert spare copy)
human_scale_table        pelvis 0.0, left_wrist 1.2, right_wrist 1.2
ik_match_table1          base           <- pelvis        pos   0  rot 1
                         left_fr3_link7 <- left_wrist    pos 100  rot 0
                         right_fr3_link7 <- right_wrist  pos 100  rot 0
```

**`pelvis: 0.0` is the fixed-base trick, not a typo.** `motion_retarget.py::
scale_human_data` computes `scaled_root_pos = human_scale_table[root] * root_pos`
and then places every other tracked joint at `scaled_root_pos + (pos - root_pos)
* scale`. A scale of 0 pins the human pelvis to the world origin — which is
exactly where the rig's `base` body sits — and turns the wrist targets into
*pelvis-relative* positions. Any other value would let an AMASS/BONES clip's
global translation (metres of walking) drag the targets away from a robot that
cannot move. Verified: same clip frame at two different world pelvis positions
gives an identical retarget.

**Why `base <- pelvis` exists at all even though `base` is welded to the world.**
Not for the IK — `base` has a zero Jacobian, so the task moves nothing. GMR
*requires* it: `scale_human_data` keeps only joints listed in
`human_scale_table` (plus the root), and `offset_human_data` then indexes
`pos_offsets1[body]` for every surviving joint. `pos_offsets1` is only populated
for table1 rows with a **non-zero weight**. So `pelvis` must be in the scale
table (the root always is) *and* must have a non-zero-weight row in table1, or
retargeting dies with `KeyError: 'pelvis'`. Hence pos 0 / rot 1: inert, but
present. Note this also means `human_scale_table` and the non-zero rows of
`ik_match_table1` must stay in exact correspondence.

**Arm scale 1.2** (a decision, not a measurement). The rationale, from a
120k-sample forward-kinematics sweep of the rig (`nq=14`, uniform over joint
ranges):

- rig geometry: arm mounts at (±0.10, 0.005, 0.14); the J2 pivot ("shoulder")
  ends up at (±0.335, 0.005, 0.375), i.e. the rig's shoulders are 0.67 m apart
  and 0.375 m above the base — roughly twice a 1.66 m human's 0.34 m shoulder
  width. Home-pose flange sits at (±0.424, −0.384, 0.464), 0.60 m from its mount.
- with the pelvis pinned at the origin, a set of stylised pelvis-relative wrist
  targets (arms down / reach forward at chest / both hands at the midline /
  reach up / low table / cross-body) is **reachable at every scale from 0.65 to
  1.6** — position alone does not choose the scale.
- what does choose it is elbow posture. Median joint 4 over the sampled
  configurations that land within 4 cm of those targets: −2.43 (scale 0.65),
  −2.41 (0.8), −2.24 (1.0), −2.10 (1.2), −1.87 (1.4), against a range of
  [−3.04, −0.15] whose healthy middle is ≈ −1.6. **Scaling the human *down*
  folds the FR3 up**, because the FR3's 0.855 m reach is much longer than a
  1.66 m human's ≈ 0.5 m shoulder-to-wrist and its shoulders are wider. 1.2 is
  the compromise: elbows off the fold stop, targets 0.32–0.67 m from their
  mount (well inside 0.855 m), and `reach_up` still under the rig's ceiling.
  1.4 gives slightly better elbows but pushes overhead reaches to 0.80 m out.
- the brief asked for "a standing human's shoulders ~0.3 m above the rig base".
  **GMR cannot place the human vertically at all** — the only global position
  knob is the root scale, and the offsets in `ik_match_table*` are applied in
  each human joint's own (per-frame rotating) frame, so there is no additive
  world translation. Pelvis pinned at the origin puts the shoulders at
  ≈ 0.47 · 1.2 = 0.56 m above the base (implied — the shoulders are not
  tracked, only the two wrists are) — higher than the requested 0.3 m and
  above the rig's own 0.375 m shoulder height, and that is the direction that
  keeps the arms out of their fold limit. Recorded as a deliberate deviation.

**`human_height_assumption: 1.66`, not G1's 1.8.** GMR multiplies the whole
scale table by `actual_human_height / human_height_assumption`, and
`load_smplx_file` derives `actual_human_height = 1.66 + 0.1 * betas[0]`. The
BONES-SEED SMPL clips carry **no betas** (verified, below), so every actor is
1.66 m; with the assumption set to 1.66 the ratio is exactly 1.0 and the numbers
in the table mean what they say. The pelvis pin (0.0) survives any ratio.

**Rotation weight 0 on both wrists — deliberate, and the biggest open question.**
Mapping the SMPL-X wrist frame onto the FR3 flange frame needs a `rot_offset`
quaternion per side (this is where every other GMR robot's tuning time goes, cf.
`smplx_to_r1pro.json`'s `[0.707, 0, -0.707, 0]` / `[0, 0.707, 0, 0.707]` pair).
It cannot be validated without body models and a viewer, and a wrong offset with
a non-zero weight actively wrecks the solve. Position-only is also what SONIC
will actually reward for this rig — `harness/lane_b/sonic_dual_fr3.yaml` has
`reward_point_body` = the two hands, tracked by position. With 7 DoF per arm and
a 3-DoF target the IK is redundant; mink's damped least-squares picks the
minimum-velocity solution, which keeps the trajectory continuous frame to frame.
**When body models land, tune the two rot offsets in the viewer and raise the
rotation weight to ~10 before trusting the flange orientation.**

**Never call `retarget(..., offset_to_ground=True)` with this config.**
`offset_human_data_to_ground` looks for a body whose name contains `foot`; our
scale table has none, so `lowest_body_name` is never bound and it raises
`UnboundLocalError`. Both stock scripts leave it `False`.

## Blocker — SMPL-X body models are not on this pod

`load_smplx_file` builds `smplx.create(assets/body_models, "smplx", ...)`, which
needs `~/code/upstream/GMR/assets/body_models/smplx/SMPLX_{NEUTRAL,MALE,FEMALE}.pkl`
(GMR's README also has you flip `ext` npz->pkl in `smplx/body_models.py`). These
are registration-walled at smpl-x.is.tue.mpg.de and **absent from the pod**
(`find ~ /data/lustre/shared -maxdepth 6 -iname 'SMPLX_*'` -> nothing,
2026-09-03). Do not attempt to download or register for them.

Consequence for P2: the human half of the motion library is not obtainable
autonomously. `plan/PLAN.md` risk 9 already calls this: continue with the P0
demo clips plus augmentations (`harness/lane_b/demos_to_sonic_pkl.py`). The
registration here is the part that can be done now, so the human half is one
`.pkl` drop away.

## How to run it, once body models exist

Outputs go to a run folder, never the repo tree (AGENTS.md rule e):
`~/runs/franka-sonic/lane_b/<YYYY-MM-DD>_motion_lib/` with `out/motions/*.pkl`
(the p2 gate reads `out/motions/*.pkl`, >= 100).

```bash
RUN=~/runs/franka-sonic/lane_b/$(date +%F)_motion_lib      # via harness/bakeoff.py
SRC=/data/lustre/shared/datasets/bones-seed/human-smpl     # READ-ONLY

# 0. bones SMPL pkl -> SMPL-X npz (needs the tar extracted somewhere writable first)
python3 harness/lane_b/gmr/smpl_pkl_to_smplx_npz.py \
    --src <dir of bones *.pkl> --dst $RUN/out/smplx_npz --limit 1000

# 1a. single clip, viewer, for tuning the rot offsets  -- NEEDS A DISPLAY
cd ~/code/upstream/GMR && PYTHONUSERBASE=$HOME/env/pyuser-sonic /isaac-sim/python.sh \
    scripts/smplx_to_robot.py --robot dual_fr3 \
    --smplx_file $RUN/out/smplx_npz/<clip>.npz --save_path $RUN/out/motions/<clip>.pkl

# 1b. batch, no viz  -- NEEDS A GPU (see landmines)
cd ~/code/upstream/GMR && PYTHONUSERBASE=$HOME/env/pyuser-sonic /isaac-sim/python.sh \
    scripts/smplx_to_robot_dataset.py --robot dual_fr3 --no-denylist \
    --src_folder $RUN/out/smplx_npz --tgt_folder $RUN/out/motions --num_cpus 16
```

### Landmines in the stock scripts (all verified in source, none patched)

- **Both scripts assume a floating base.** They slice the retargeted `qpos` as
  `root_pos = qpos[:3]`, `root_rot = qpos[3:7]`, `dof_pos = qpos[7:]`. Our
  `nq` is **14** with no freejoint, so that reads joints 1–3 as a position,
  joints 4–7 as a quaternion, and keeps only **7 of the 14** joint angles. For
  `dual_fr3` the whole of `qpos` is the dof vector. A retarget driver of our own
  (or a post-hoc fix) is required before either script's `--save_path` output is
  usable; nothing was patched here because that is a rewrite, not a
  registration.
- `scripts/smplx_to_robot.py` opens `mujoco.viewer.launch_passive` unconditionally
  -> needs a display; it will not run headless on this pod.
- `scripts/smplx_to_robot_dataset.py` hardcodes `device = "cuda:0"` for its
  `KinematicsModel` FK pass -> it is **not** a CPU-only job, and on this pod the
  device must come from `harness/gpus.py acquire` (AGENTS.md rule a), never
  `cuda:0` by hand.
- It also does a `HEIGHT_ADJUST` pass (drops the model so the lowest body sits on
  the ground) and a `ROOT_ORIGIN_OFFSET` pass on `root_pos` — both meaningless
  for a bolted-down rig, both operating on the mis-sliced `root_pos`.
- `--override` is needed to redo clips that already have an output.
- Resampling: `tgt_fps = 30` is hardcoded, but `frame_skip = int(src_fps/tgt_fps)`
  is an **integer**, so a 50 fps source gives `frame_skip = 1` and passes through
  **unchanged at 50 fps** (`aligned_fps = 50`). BONES-SEED is 50 fps, so its clips
  are *not* downsampled to 30 — contrary to the note in
  `wiki/gmr-new-robot-recipe.md`. 120 fps AMASS does land on ~30.

### Still to write: GMR pkl -> SONIC motion_lib pkl

GMR writes, per clip,
`{fps, root_pos (T,3), root_rot (T,4) **xyzw**, dof_pos (T,nq-7), local_body_pos,
link_body_list}`. SONIC's `MotionLibBase.load_motion_with_skeleton` wants
`{root_trans_offset, pose_aa, dof, root_rot **wxyz**, smpl_joints, fps}`. For
this rig the conversion is trivial *because the base is fixed*: GMR's `root_pos`
and `root_rot` carry no information (and, per the landmine above, are actually
mis-sliced joint angles) — only the 14 joint angles matter.

The contract is already implemented in
`harness/lane_b/demos_to_sonic_pkl.py` (sibling deliverable): `make_entry(rig,
dof)` builds `root_trans_offset` zeros `(T,3)`, `pose_aa (T,17,3)` in MuJoCo body
order, `dof (T,14)`, `root_rot` = the wxyz identity `[1,0,0,0]`, `smpl_joints`
zeros `(T,24,3)`, `fps` 30. Reuse it: take the full 14-wide `qpos` per frame from
a fixed-base-aware retarget driver, resample 50 -> 30 fps with its `resample()`,
and call `make_entry`. Mind that module's joint-6 convention (`J6_OFFSET`,
`q6_here = q6_fr3 - 2.5307`) — it is baked into `dual_fr3.xml`, so anything that
comes out of *this* MJCF is already in the shifted convention and must not be
shifted again.

## Facts established on the pod (2026-09-03)

- **BONES-SEED SMPL clips are `joblib` pickles, not plain ones.**
  `pickle.load` on
  `/data/lustre/shared/datasets/bones-seed/human-smpl/bones_seed_smpl.tar.part_aa
  :smpl_filtered/jump_ff_360_R_001__A373_M.pkl` raises
  `UnpicklingError: invalid load key, 'x'`; `joblib.load` works. The adapter
  tries joblib first and falls back to pickle.
- One clip's fields:
  `pose_aa (119,72) f32`, `transl (119,3) f32`, `smpl_joints (119,24,3) f32`,
  `fps 50.0`, `original_pose_aa (72,72) f32`, `original_fps 30.0`.
  **No `betas`** — so `betas = zeros(16)`, `gender = neutral`, and GMR treats
  every actor as 1.66 m tall (per-actor limb proportions are lost). This
  confirms the open question in `wiki/gmr-new-robot-recipe.md`.
- Human joint names on the SMPL-X side come from `smplx.joint_names.JOINT_NAMES`
  (`pelvis`, `left_wrist`, `right_wrist`, ...).
- `~/env/pyuser-sonic` already had `mujoco`, `smplx`, `qpsolvers`, `daqp`,
  `torch`, `scipy`, `rich`, `tqdm`, `psutil`, `imageio`. Only **`mink`** had to
  be added for `import general_motion_retargeting` to work (plus GMR itself,
  editable, `--no-deps`). `natsort` is still missing and is imported by
  `scripts/smplx_to_robot_dataset.py` — install it before the batch run.

### Open, unverified (needs body models)

- **Clip heading is not canonicalised.** With the pelvis pinned, a wrist target's
  direction in the world is whatever way the actor happened to face. The rig is
  not rotationally symmetric — both arms face −y — so clips whose actor faces +y
  will retarget behind the rig and saturate against joint limits. Fix in the
  adapter: rotate `root_orient`/`trans` about z so frame 0 faces −y. Not
  implemented (untestable today), but it is the first thing to add.
- The SMPL-X world axis convention (which way "forward" is for a zero
  `root_orient`) was not checked; the synthetic test above assumed z-up and
  forward = −y.
- The two wrist `rot_offset` quaternions are identity placeholders.
