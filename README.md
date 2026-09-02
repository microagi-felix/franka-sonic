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
