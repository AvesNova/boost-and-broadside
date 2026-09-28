# Agent prompt for perception/belief phases

Copy the block below into a fresh session and fill in the phase number. It is
deliberately short. The spec and the status document carry the detail; the
prompt carries the budget and the stopping rule, which is what the previous
prompts lacked.

---

```
You are implementing Phase <N> of the perception/belief plan in the
boost-and-broadside repository.

Read, in this order, and nothing else before starting:
1. docs/internal/perception-belief-overarching-plan.md — the "Decided model
   semantics" section and the Phase <N> section only.
2. docs/internal/perception-belief-status.md — all of it (under 250 lines).
3. STYLE_GUIDE.md.
Then read the code the phase names. Do not read the archived handoffs unless a
specific number in the status document is unclear.

Work on branch feat/perception-belief-phases-4-8. Commit as you go with short conventional messages.

## Scope

Do exactly the Phase <N> scope. The phase's "Not in scope" line is binding. If
you find a bug outside the scope, note it in the status document's caveats and
move on; fix it only if it blocks the gate or is a one-line change.

## Budget and what to run

- Run the focused tests for the files you change, as often as you like. They
  take seconds.
- Run the full default `pytest` suite at most twice in the whole phase: once
  when you believe the code is complete, and once more only if the first run
  failed and you changed code. It takes about seven minutes. Never run it
  after a documentation-only change.
- Run `bnb smoke` once, at the end, only if the phase touches evaluation
  modes, the wrapper, or the match runner.
- Run `ruff check` and `ruff format --check` on the files you changed before
  each commit. Do not fix pre-existing violations in files you did not touch.
- Run an end-to-end benchmark only if the Phase <N> gate names one, exactly
  once, with the command in the status document. Do not run a second arm, a
  variant, or a comparison the gate does not ask for.
- Never start a training run, an SPS experiment, a parity harness, a
  calibration, or any process longer than the benchmark above. If you believe
  one is needed, write why in the status document and stop.
- Never write a script, loop or monitor that waits on your own process
  output. If something needs to run long, it is out of budget.
- Do not run experiments to compare alternatives the spec did not ask you to
  compare. Implement what the spec says; if you disagree, say so in one
  paragraph in the status document and implement the spec anyway.

## Stopping rule

The Phase <N> gate in the plan is the definition of done. When every line of
it is met:
1. Update docs/internal/perception-belief-status.md as the plan's "Status
   update requirement" describes: move the phase to Done in three to ten
   lines, adjust caveats and open questions, add a numbers row only if the
   gate required a benchmark. Keep it under 250 lines. No test counts, no
   test wall times, no commit hashes, no incremental validation tables.
2. Commit and push.
3. Stop. Do not re-run tests after the status edit. Do not add a commit hash
   to the document. Do not tidy anything the gate did not ask for.

If the gate cannot be met, write what is blocking it in the status document
under a "Blocked" heading, commit, push, and stop. Do not keep trying
variations.

## Style of the status update

Plain sentences about what changed and what the next phase must respect.
Write for someone who will continue the work, not for someone auditing it.
If a sentence exists to show you were careful, delete it.
```

---

## Notes for the person running the phases

- The old spec's fourteen-item handoff checklist is what generated the
  700-line handoffs, the repeated test runs, and the commit-hash loop. It is
  gone. If you want a record of what was run, `git log` has it.
- The phases are sized so that a single session finishes one. If a phase
  reports "Blocked", read the block before restarting; usually the fix is to
  narrow the phase, not to give the agent more budget.
- Phase 8 is training runs. Decide the budget yourself and put the run list in
  the prompt; do not let the agent choose which runs to start.
- If an agent starts a training run or a long benchmark anyway, the prompt's
  budget section is the thing to strengthen, not the spec.
