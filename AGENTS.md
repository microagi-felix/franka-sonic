# AGENTS.md — operating rules on pod `franka-sonic` (10.5.9.216)

These are not suggestions. Every one of them exists because breaking it has
cost someone a day. Read `plan/PLAN.md` for *what* to do; this file is *how*.

## a. GPUs go through the allocator

Every GPU job acquires a device set first and exports it:

```bash
eval "$(~/Isaac-GR00T/.venv/bin/python harness/gpus.py acquire --n 1 --job p0-smoke)"
# -> exports CUDA_VISIBLE_DEVICES=5   (or exits 1 if not enough idle devices)
...run the job...
~/Isaac-GR00T/.venv/bin/python harness/gpus.py release --job p0-smoke
```

Never hand-pick a device. Never run a training rank and an Isaac eval on the
same device (measured −80 % on both). The node is shared with processes
outside this pod: a device counts as usable only when it reports **< 1 GiB
used**. `harness/bakeoff.py` does the acquire/release for you.

## b. Never kill by pattern

No `pkill -f`, no `killall`, no `kill -9 $(pgrep …)`. Other tenants share the
node. Kill only PIDs your own run folder recorded (`<run>/out/*.pid`).

## c. `/data/lustre/shared` is READ-ONLY

Read freely. Never write, move, delete or `mkdir` there. It is multi-user
storage; the guard hook blocks the obvious forms, but the rule is yours.

## d. Install only under `~`

The container overlay (`/opt`, apt, system pip) is lost on every fresh pod;
`~` is Lustre and survives. Two separate user-sites so their pins cannot
fight:

- sim stack (Isaac Lab 2.3.2 + the franka repo): `PYTHONUSERBASE=~/env/pyuser-fr3`
- `gear_sonic` (SONIC training): `PYTHONUSERBASE=~/env/pyuser-sonic`

Both installed with `/isaac-sim/kit/python/bin/python3 -m pip install --user`
and run with `PYTHONUSERBASE=… /isaac-sim/python.sh …`. GR00T keeps its own
`~/Isaac-GR00T/.venv`. Never `sudo pip`, never a system-wide install.

## e. Every run has a folder

`~/runs/<family>/<YYYY-MM-DD>_<slug>/` — family is `franka-sonic/<lane>`
(`shared`, `lane_a`, `lane_b`). Each folder holds:

- `README.md` — what and why, written **at launch**, not afterwards
- `cmd.sh` — the exact commands, re-runnable
- `logs/` — every stdout/stderr, `logs/run.log` is the main one
- `out/` — artifacts (`out/eval/` for `evaluation.eval` run folders)
- `config.json` — stamped by `harness/bakeoff.py`: args, host, date, repo
  SHAs, `CUDA_VISIBLE_DEVICES`

Nothing outside a run folder. No logs in the repo tree, no scratch in `~`.

## f. Phases move only through gates

`bash harness/gates/pN.sh` must exit 0 before phase N+1 starts. Gates read run
folders and exit codes, never prose. A gate that "basically passes" has failed.

## g. Commit after every completed step

Append a dated line to `plan/STATUS.md`, `git commit`, `git pull --rebase`,
`git push`. The orchestrator on the Mac reads STATUS.md and the tmux pane —
that is the only channel. Push, or it did not happen.

STATUS.md is **append-only**: never rewrite, reorder or tidy it. More than one
agent pushes to `main` (the interactive session, the driver's phase sessions),
so always `git pull --rebase` immediately before pushing.

## h. Log pod-state mutations outside git

Anything git cannot see — `apt-get install`, `usermod`, symlinks, pip
installs, HF downloads — gets a timestamped line in
`~/agents/2026-09-01_franka-sonic/WORKLOG.md` (the `~/agents/README.md`
convention: one dated WORKLOG per campaign).

## i. Wait in the foreground

Bounded blocking loops, never background wakeups (they fail silently):

```bash
for i in $(seq 1 120); do kill -0 "$PID" 2>/dev/null || break; sleep 15; done
tail -50 <run>/logs/run.log
```

## j. STOP and write `BLOCKED: <reason>` in STATUS.md (then push) before

- anything that would use **> 20 GB** of disk
- anything installed or written **outside `~`**
- any `kubectl`
- any change to **another user's files**
- using any GPU that is **not idle** per the allocator

**This rule is not "no sudo".** Overlay-only, pod-local `sudo` that the
campaign needs is expected — do it, make it idempotent in `env/bootstrap.sh`,
and log it in the WORKLOG. The standing example, needed on every fresh pod:

```bash
sudo chown -R $USER:research /isaac-sim/kit/cache /isaac-sim/kit/logs \
                             /isaac-sim/kit/data /isaac-sim/extscache
```

Kit writes its caches and config under `/isaac-sim/kit`; without write access
the RTX renderer fails to create a scene renderer and **every Isaac env
creation dies** (it broke the first P0 smoke). Widen to `/isaac-sim/kit` if
more `Permission denied` appears.

## k. Lustre is 99 % full

1.3 TB free on a 70 TB shared filesystem. Keep datasets and checkpoints small,
delete failed-run artifacts, and treat "just cache it" as a decision, not a
reflex.

## l. Sync only via git

This repo and `plan/STATUS.md` are the entire interface to the orchestrator.
No side channels, no assumptions that anyone is watching the terminal.

## m. The driver owns the phase loop

`harness/driver.sh` runs in tmux window `bakeoff:driver` and walks P0 → P4
unattended, one fresh `claude -p --effort xhigh` session per phase from
`plan/prompts/PN.md`, 3 attempts of at most 6 h each, stopping at the first
`BLOCKED:`. Consequences for you:

- **The interactive window `bakeoff:claude` is for P0 and for debugging only.**
  Never run a phase by hand that the driver is running: two sessions in one
  working tree fight over the index and the GPU claims.
- If you are a phase session: do your phase, pass its gate, write
  `GATE PN: PASS`, push, **end the turn**. Never start the next phase — the
  driver does that.
- Stuck? `BLOCKED: <reason>`, push, end the turn. The driver stops the
  campaign; a human clears it by appending `DRIVER: resume` to STATUS.md.
- Start/stop: see README.md. `tmux kill-window -t bakeoff:driver` stops it.

## n. Subagents are allowed — under the same rules

Use the `Agent` tool (`.claude/agents/executor.md`) for parallelisable,
well-scoped work: authoring configs while a conversion runs, an oracle replay
while a fine-tune trains, a retarget batch on CPU. Rules that do not bend:

- **Only the parent acquires GPUs.** The parent runs `harness/gpus.py acquire`
  and hands the exported `CUDA_VISIBLE_DEVICES` set to the subagent's commands.
  Never a second allocation for the same job, and at most **one GPU-heavy job
  per allocated device set**.
- **Foreground bounded waits only** — background wakeups fail silently in
  subagents. Same `for i in $(seq …); do kill -0 …; sleep 15; done` pattern.
- **No `git push` from a subagent.** It reports back; the parent commits,
  appends to STATUS.md and pushes.
- Parallel subagents only for CPU or authoring work; serialise anything that
  touches a GPU or the git index.
