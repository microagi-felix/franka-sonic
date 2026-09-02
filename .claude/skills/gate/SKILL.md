---
name: gate
description: Run the phase gate for phase N (harness/gates/pN.sh), record the verdict in plan/STATUS.md, commit and push. Use when a phase looks finished, or when asked "/gate 0", "run the P0 gate", "are we through P1".
---

# /gate N

Phases move only through gates (AGENTS.md rule f). A gate that "basically
passes" has failed.

1. Run it and keep the full output:

   ```bash
   bash harness/gates/p$N.sh 2>&1 | tee /tmp/gate-p$N.out; echo "rc=${PIPESTATUS[0]}"
   ```

2. Read every non-PASS line. A `FAIL` is a stop: fix the cause, then re-run the
   whole gate — never a single check in isolation.

3. Append the verdict to `plan/STATUS.md`, one line, dated:

   ```
   - YYYY-MM-DD HH:MM  GATE P<N>: PASS (2 warnings)   [or]   GATE P<N>: FAIL — <the failing checks>
   ```

   Then paste the failing/warning check lines underneath, indented, so the
   orchestrator can read the reason without the terminal.

4. `git add -A && git commit -m "gate p<N>: <PASS|FAIL>" && git push`.

5. Report the verdict in chat with the failing lines verbatim. On PASS, say
   which phase is now unblocked; do **not** start it unless asked.
