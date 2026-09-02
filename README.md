# franka-sonic — FR3 handover bake-off workspace

Executable workspace for the dual-FR3 block-handover bake-off: **lane A** =
GR00T N1.7 fine-tuned to emit joint targets directly, **lane B** = GR00T
emitting a 64-D SONIC token decoded to joint targets by a SONIC decoder.
Goal of the prototype: every stage runs end to end once and both lanes log a
success rate — a pipeline proof, not a result.

Plan and design live in the brain wiki (`microagi-felix-brain`):
`wiki/fr3-handover-bakeoff-plan.md` and
`wiki/bakeoff-instance-harness-proposal.md`. The condensed, executable copy is
`plan/PLAN.md`; live state is `plan/STATUS.md`.

Start on the pod: `bash env/bootstrap.sh`, then `tmux new -s bakeoff -c ~/code/franka-sonic`
and `claude` inside it. Read `AGENTS.md` before touching anything.

## Running the bake-off unattended

`harness/driver.sh` walks P0 → P4 on its own: per phase it checks
`plan/STATUS.md` for `GATE PN: PASS`, and otherwise runs a **fresh** Claude
Code session on `plan/prompts/PN.md`:

```
claude -p --dangerously-skip-permissions --effort xhigh --output-format text "$(cat plan/prompts/PN.md)"
```

up to 3 attempts per phase, `timeout 6h` each, `git pull --ff-only` and a GPU
claim listing before every attempt, everything logged to
`~/runs/franka-sonic/driver/<YYYY-MM-DD>/PN-attempt-<k>.log` (the effective
command line, effort flag included, is echoed at the top of each log). It stops
at the first `BLOCKED:` and writes `DRIVER: all phases passed` when P4 passes.

```bash
mkdir -p ~/runs/franka-sonic/driver
tmux new-window -t bakeoff -n driver -c ~/code/franka-sonic \
  'bash harness/driver.sh 2>&1 | tee -a ~/runs/franka-sonic/driver/driver.log'

tmux capture-pane -p -t bakeoff:driver | tail -20   # what is it doing
tmux kill-window -t bakeoff:driver                  # stop it
```

- **The interactive window `bakeoff:claude` is for P0 and for debugging only.**
  The driver waits (60 s poll, up to 4 h) for that session to write
  `GATE P0: PASS` or `BLOCKED:` before it touches P0 itself. Never run a phase
  by hand while the driver is running the same phase.
- Re-running the driver is safe: phases already marked `GATE PN: PASS` are
  skipped.
- **Unblocking:** the driver exits 2 as soon as the last stop marker in
  STATUS.md is a `BLOCKED:` line. Fix the cause, append a line ending in
  `DRIVER: resume` to `plan/STATUS.md`, then start the window again.
- Knobs (env vars): `DRIVER_EFFORT` (default `xhigh`), `DRIVER_ATTEMPTS` (3),
  `DRIVER_PHASE_TIMEOUT` (`6h`), `DRIVER_P0_WAIT_POLLS` (240 × 60 s = 4 h),
  `DRIVER_CLAUDE` (path to the CLI).

Phase prompts live in `plan/prompts/`; gates in `harness/gates/p0.sh … p4.sh`.
Both are the contract between phases — edit the prompt, not the driver, when a
phase needs different work.
