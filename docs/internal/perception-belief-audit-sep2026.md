# Perception/belief work: code audit and process review

September 28, 2026. Covers Phases 0–3 of the perception/belief plan as they
stand at `ff32f9f`, and the way the phases were run.

Two parts. Part 1 flags things in the code that look wrong, questionable, or
worth a cheap agent's attention. Part 2 explains why the phase agents have been
slow and expensive, and what in the instructions caused it. The rewritten spec
(`perception-belief-overarching-plan.md`), the consolidated status document
(`perception-belief-status.md`) and the agent prompt
(`perception-belief-agent-prompt.md`) implement Part 2's recommendations.

---

## Part 1 — Code audit

Coverage: I read the Phase 3 modules in full (`physical_belief.py`,
`belief.py`, the composition path in `observation.py`, the label/loss/join
code in `ppo.py` and `opponents.py`, `NextStateHead`) and the Phase 2 code
that lives in the same files (`write_pending_action_view`, action-belief
storage). Phase 0's latch fix, Phase 1's calibration script and the tests
were checked against the handoffs only, not read. Phases 0 and 1 are small
and well-described, and Phase 3 rewrote most of what Phase 2 touched, so this
is where the risk is, but it is not a full four-phase audit.

Headline: **no correctness bug found in the Phase 3 data plane.** The label
timing (`privileged_means` is read before the step, so `truth[t]` pairs with
`obs[t]`; labels are `believed[t] → truth[t+1]`), the stream-join order
(`advance → reset → observe → compose`), the bivariate NLL (the per-axis split
sums to the exact joint likelihood), and the one-selection composition all check
out against the spec. What follows is graded by how much I think it matters.

### Should be fixed or decided (cheap agent)

1. **Persistence-initialized position forecast: not a defect, by owner
   decision.** The head starts out predicting zero delta (persistence) and the
   Phase 3 probe shows dead reckoning beating it 34x on visible ships. I first
   recommended a dead-reckoning prior. The owner's answer: the next-state head
   exists as a *hard* auxiliary task that helps the policy learn, and giving it
   the physics for free defeats that purpose. So: no prior. The dead-reckoning
   diagnostic stays as the bar the head should eventually clear, and the open
   question is whether it does at convergence.

2. **`shield_delay` has no upper clamp** (`physical_belief.py`,
   `from_ship_config`). The handoff records it. The recharge delay lives on
   `FrontlineConfig`, which the spec does not see. Either pass it in or accept
   the caveat explicitly in the spec. Low risk; a bounded recursion is the
   point of the clamps.

3. **`physical_means_from_state` takes a `ship_config` argument and deletes it**
   (`physical_belief.py`). Every caller threads `ship_config` through for
   nothing. Remove the parameter and the three call sites' argument.

4. **`BELIEF_UNCERTAINTY` for truth-sourced ship slots is read from the belief
   store, not set to the certainty floor at composition**
   (`env/observation.py`, end of `observation_from_state`). It is correct today
   only because `BeliefTracker.observe` stamps `_certain` on every visible ship
   and own ships are always visible. If an own ship is ever truth-sourced by
   ownership while not "visible" (a dead ship in a non-respawn mode, a future
   visibility change), the policy reads a stale forecast spread for a slot it is
   seeing truth on. One `torch.where(from_truth, certain, belief.uncertainty)`
   at the composition site would make the contract local. Same remark for
   `TIME_SINCE_OBSERVATION`.

5. **Hidden-token label semantics are intended.** The label is
   `truth[t+1] − believed[t]`, so on a hidden ship the head learns to correct
   the belief toward truth, with the sigmas saying how much to trust it. That is
   the owner's design: reuse the aux-head machinery to estimate where hidden
   ships might be. The v1 plan did not say this in so many words; the v2 spec
   does.

### Worth knowing, no action required

6. **Dead code is deliberately left for Phase 4** (`features.py` predictor
   machinery, `BELIEF_TARGETS`, `_believed_rotary_tables`, and so on). The
   Phase 3 handoff lists it precisely; the list is reproduced in the status
   document. Phase 4 is a deletion phase and should be cheap.

7. **The full Phase 2 → Phase 3 throughput story is fine.** Phase 2 cost 26%
   SPS, Phase 3 recovered it and then some (+13.5% over the Phase 0 baseline,
   −472 MiB). The only regression is compile startup (three Inductor graphs
   instead of one). Nothing to do unless startup time starts to matter.

8. **The 24-step compile parity at zero tolerance differs at the ULP level.**
   Expected float reassociation. The handoff's own point stands: `error_count`
   is the criterion, not the hash.

9. **A diagnostic block in `ppo.py` can kill a run.** The `Categorical`
   simplex check crash was fixed in `33a4a0e`, but the metric block is not
   otherwise defensive. If another metric raises, the update dies. A
   `try/except` around no-grad diagnostics that logs and continues would be
   cheap insurance; a matter of taste.

10. **Belief tracks own ships as well as enemies.** `observe` assimilates every
    visible ship, including the observer's own. Harmless (own slots are always
    truth-sourced), costs eleven channels of `where` per own ship per step.
    Not worth touching.

11. **`state.num_zones > 0` is the "respawn is on" flag** in
    `BeliefTracker.source`. Implicit coupling between map layout and lifecycle
    semantics. It works; a named property on the config would be clearer.

12. **The Phase 0 handoff's claim that "compiled perception is not paying for
    itself"** was a two-update measurement. Phase 3 changed the graph
    structure anyway, so it is moot.

### Things I checked and found correct

- Bivariate Gaussian NLL, including the `0.25·log(1−ρ²)` per-axis split and
  the `clamp_min(1e-6)` guard.
- Attitude residual wrapped onto the circle before the scalar NLL.
- `advance` before `reset` before `observe` in the primary rollout; the league
  slots keep the same order across three call sites.
- `ns_mask = belief_valid & ~terminated & transition_contiguous` masks
  never-seen tokens and respawn destinations.
- Spawn voids belief validity before assimilation, so a revealed respawn is a
  clean re-acquisition.
- `write_pending_action_view`: authoritative enemy commands never reach the
  destination; spawn-revealed slots read the exact null one-hot.
- Uncertainty is replaced, not accumulated; visible = floor, never-seen =
  ceiling; the encoder divides log sigmas by the clamp bound and leaves the
  correlation latents alone.
- Physical channels promoted to fp32 in the rollout buffer, for the stated
  reason (bf16 resolution versus the 4 px/s velocity scale).

---

## Part 2 — Why the agents are slow, and what to change

### What the handoffs show

Reading the four handoffs and the git log, the pattern is consistent:

- **Every phase ran the full suite (400–1000 s) and `bnb smoke` at least twice,
  plus a "validation during development" table of ten to fifteen partial
  runs**, each with an exact pass count and wall time. Phase 2 lists thirteen
  such rows. That table is not useful to the next agent; it exists because the
  spec's "Required phase handoff" asks for "tests run and their results".
- **Every phase ran an end-to-end CUDA benchmark** (`rl_pipeline_profile.py`,
  roughly ten minutes each), a compile-parity harness, and in Phase 3 a
  555-second baselines probe, because the exit gates say "throughput cost is
  measured" and "benchmarked against the Phase-0 baseline". Phase 1 even
  explains why it did *not* run the benchmark, which shows the agent felt
  obliged to.
- **Commit hashes in the handoff**: the spec requires "branch and relevant
  commit SHAs". A handoff commit cannot contain its own hash, so the agent
  commits, records the hash, which changes the tree, which needs a new commit,
  and it re-runs the tests before each one because the spec also asks for test
  results. Phase 0 wrote a paragraph explaining this loop. Phase 3 has three
  handoff commits (`d74f66a`, `847c2aa`, `705acc4`) before the code was even
  finished.
- **Phase 2 spent an experiment on a linear enemy head**, measured it,
  reverted it, and kept both profiles in history, because the exit gate says
  "throughput cost is measured" and the agent read that as an invitation to
  optimize. The spec never asked for the experiment.
- **A separate "independent evaluation" agent re-ran Phase 0** and wrote 450
  lines. Nothing in it changed a decision. That doubled the cost of the phase.
- **Handoffs are 200–700 lines** and mostly prose about what the agent did
  and why it was careful, written for a reader who is auditing rather than
  continuing. The next agent has to read all of it, plus the prior ones, and
  then reproduces the style.

### The instruction flaws, specifically

1. **The handoff checklist rewards evidence of effort.** Fourteen mandatory
   items, several of which ("tests run and their results", "benchmarks/profiles
   run and their results", "commit SHAs") can only be satisfied by doing more
   runs and writing more. An agent optimizing for a complete checklist will run
   the suite again so it has a fresh number to report. The style guide's own
   rule ("never hardcode counts in prose, they rot") is violated by every
   handoff because the checklist demands it.

2. **Exit gates are phrased as measurements, not as decisions.** "Throughput
   cost is measured" has no stopping condition. "SPS within 10% of baseline,
   else stop and report" does. Without a threshold the agent measures, then
   measures again under another arm to be sure, then writes it up.

3. **"Independently testable" plus "make each phase leave a durable handoff"
   plus "do not silently change semantics"** reads, to a careful agent, as
   "prove you did not break anything", which means the full suite, the smoke
   suite, the parity harness and a benchmark, every time. Nothing in the
   instructions says which checks are sufficient, so the agent runs all of
   them, and runs them again after each documentation edit.

4. **No token, time, or run budget anywhere.** Nothing tells the agent that a
   ten-minute benchmark is expensive or that a training run is off limits.
   Multi-hour training and speculative SPS experiments are the natural result
   of "benchmark performance-sensitive changes rather than assuming" with no
   cap.

5. **The plan's principles are correct but their tone is absolute** ("must",
   "never", "by contract", "authoritative"). A model reads that register and
   mirrors it: it becomes defensive, over-verifies, and writes handoffs that
   sound like legal depositions. The Phase 3 handoff's prose is a direct
   imitation of the plan's.

6. **No prompt-level guidance on what to do when done.** The agent has no
   instruction such as "when the exit gate is met, write the handoff and stop".
   So it keeps finding things to tighten (the Phase 2 linear-head experiment,
   the Phase 3 24-step parity re-run, the extra handoff ordering commits).

7. **Monitors watching their own output** is a symptom of the same thing: the
   agent starts a long process because it was told to benchmark, then has to
   wait for it, and the only tool for waiting is a watcher. Not starting the
   long process is the fix.

### Are the phases too rigid?

Partly. The *semantics* section of the plan is right to be rigid; those are
real design decisions and re-litigating them per phase would be worse. The
*phase gates* are too rigid in the wrong dimension: they are precise about what
to measure and silent about when to stop. Two concrete changes:

- **Merge Phase 4 into Phase 5's preamble** rather than run it as its own
  phase with its own benchmark and handoff. It is a deletion; the test suite is
  the gate.
- **Drop the per-phase end-to-end benchmark requirement.** Replace it with:
  run the benchmark once, at the end of a phase that touches the rollout hot
  path, only if the phase's own micro-timing suggests a change larger than
  10%, and never more than once per phase. Phases 4, 6 (which has its own
  micro-benchmark) and 8 do not need it.

### What the new documents do about it

- The spec keeps the decided semantics verbatim, folds the completed phases
  into a two-page "what exists" section, and rewrites the remaining phases with
  a short scope, a short gate with thresholds, and an explicit "not in scope"
  line each.
- The handoff requirement shrinks to about eight lines with a hard length
  cap, no test counts, no commit hashes, no benchmark unless the phase says so.
- The agent prompt carries an explicit budget (what to run, how often, what
  never to run), a stopping rule, and a ban on watchers, training runs and
  unrequested experiments.
- The four handoffs and the Phase 0 evaluation are archived, not deleted, so
  their numbers remain reachable; the status document carries only what the
  next agent needs.
