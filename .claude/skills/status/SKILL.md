---
name: status
description: Show the current bake-off state — the tail of plan/STATUS.md, live GPU claims, and the tmux windows. Use for "/status", "where are we", "what is running", or before starting any new stage.
---

# /status

Three reads, then one compact summary. Do not guess any of them.

```bash
tail -25 plan/STATUS.md
~/Isaac-GR00T/.venv/bin/python harness/gpus.py list
~/Isaac-GR00T/.venv/bin/python harness/gpus.py probe
tmux list-windows -a 2>/dev/null || echo "no tmux server"
git -C . status --short && git -C . log --oneline -3
```

Report as bullets:

- **phase** — the last gate verdict in STATUS.md, and what it unblocks
- **running** — live GPU claims (job, devices, pid) and tmux windows; flag any
  claim whose PID is dead as stale (the allocator reaps those on the next call)
- **free GPUs** — how many devices are eligible (< 1 GiB used) right now
- **uncommitted** — anything dirty or unpushed in this repo, since git is the
  only channel to the orchestrator (AGENTS.md rule l)
- **next** — the next unchecked box in `plan/PLAN.md`
