# Phase 0 handoff — correctness, lifecycle, and measurement foundation

Plan: [`perception-belief-overarching-plan.md`](perception-belief-overarching-plan.md),
Phase 0. Completed September 27, 2026.

Branch: `fix/seat-symmetry-pending-action`. Starting commit `590a25b`
("docs: add the overarching perception and belief implementation plan").
Ending commit: the last one on this branch, which is the one that added this
file. `git log --oneline 590a25b..` lists the phase exactly.

Phase 0 is the first phase, so there was no preceding handoff. Its inputs were
the overarching plan plus the two September investigations it grew out of:
[`next-state-head-audit-sep2026.md`](next-state-head-audit-sep2026.md) and
[`seat-symmetry-bug-sep2026.md`](seat-symmetry-bug-sep2026.md).

## Commits, in order

| commit | what it established |
|---|---|
| `375cf12` | `TensorEnv.step` clears and re-fills the spawn-reveal latch, so the reveal lasts exactly one decision in the direct physics path as well as the wrapper's. |
| `c41063f` | The null pending action at every spawn and respawn, end to end through `MatchRunner` and at the training path's initial queue. |
| `b933da0` | The belief diagnostic's temporal alignment, measured rather than assumed; and `belief/<cell>/position_beyond_legal_frac`, which separates a leaked teleport from a broken model. |
| `62c31d4` | Seat symmetry as a mirrored-game comparison over every observation channel and the policy's outputs, not just the channels two seats of one game share. |
| _this commit_ | The recorded baseline — throughput, phase wall-time, VRAM, compiled/eager — and this handoff. A commit cannot carry its own hash; `git log 590a25b..` on this branch gives the exact range. |

## What changed, and why

### The spawn-reveal latch in the direct physics path

`state.ship_spawned` is the latch `team_visibility_from_state` ORs into both
teams' sight when `spawn_reveal` is on. `YemongEnvWrapper.step` cleared it at the
start of each decision and re-filled it from each tick's respawns;
`TensorEnv.step` did neither. So in every path that drives `TensorEnv` directly,
the opening reset set the latch and nothing ever cleared it: **every ship stayed
permanently visible to both teams**, and `spawn_reveal` silently meant
omniscience.

Training was never affected — it steps the wrapper. Affected, and therefore
measuring a fog-free world at `30790f6` and before:

* the live Elo evaluator, so run 747's live Elo and the scripted anchor it is
  calibrated against are omniscient-world numbers;
* every `MatchRunner` consumer: `evaluation/tournament.py`, `noise-calibration`,
  `ar-report`, `capture`, `feature-stats`;
* `modes/interactive.py`'s `state_only` render path.

The fix puts the same three lines the wrapper owns into `TensorEnv.step`, which
is the level at which "a decision has passed" is known. The duplication is
deliberate and commented on both sides: the wrapper steps `tick` directly
because rewards and episode statistics have to accumulate per physics tick, so
it never reaches `TensorEnv.step`'s body.

This was already diagnosed in the next-state audit's "Harness bug found while
measuring this"; that section is now marked fixed, and the audit's numbers were
taken with the probe's own workaround, so they stand.

### Null pending action at spawn and respawn

Already correct, now pinned end to end. `PendingActionState.commit` zeroes the
queue wherever `actuator_contiguous` is false, and `actuator_contiguous` is
`alive & ~respawned` accumulated across the decision — so a ship that respawned
during decision `t` enters decision `t+1` on exactly `(0, 0, 0)`. The unit-level
rule was already covered against the decision-runtime oracle in
`tests/runtime_semantics`; what was missing was the statement over real
lifecycle events, which `tests/env/test_spawn_lifecycle.py` now makes.

**Deferred to Phase 2, with the gap measured.** The decided semantics say both
teams know the null pending action during the spawn/reveal decision. Today the
spawned ship's own team reads `(0, 0, 0)` and the other team reads the private
categories `(3, 7, 2)`. Phase 2 owns the spawn override together with the 42-way
joint representation that replaces those categories; implementing it against the
factorized representation now would be work Phase 2 deletes, and it would change
the observation distribution the Phase-0 performance baseline was taken under.

### The belief diagnostic: aligned, but leaking

The plan carried this as a "known timing discrepancy", from the audit's
hypothesis that `_precompute_belief_diagnostics` pairs `rollout_predictions[t]`
with `privileged_targets[t+1]` off by one decision.

**The hypothesis is refuted.** Full evidence in
[`belief-diagnostic-alignment-sep2026.md`](belief-diagnostic-alignment-sep2026.md).
Replacing the forecast with the believed target vector — the identity, which for
an absolute Fourier-moment predictor is *not* a zero prediction vector — and
scoring it against `privileged_targets` at three offsets gives 0.019 px, 8.77 px
and 17.61 px. `buf.obs[t]` and `buf.privileged_targets[t]` describe the same
instant to within the bf16 and Fourier-ladder floor, and production scores the
forecast made at `t` against `t+1`, as intended.

The 22× anomaly is the 0.2% of tokens that `transition_contiguous` removes. They
average 1726 px each — respawn teleports — and that is enough to move the mean of
seventeen thousand samples eighteen-fold. Dropping that one filter reproduces the
historical number from today's code: 0.208 px becomes 3.862 px. The audit's own
`enemy_visible_anyalive` cell, which drops the same filter, recorded 3.55 px and
was dismissed on a different cell's number.

Which commit leaked them is not recoverable from the checkpoint, and run 747 is
being discarded for the seat-symmetry contamination anyway. What is fixable is
the statistic: `belief/<cell>/position_beyond_legal_frac` reports the fraction of
a cell whose one-decision position error exceeds ten times the furthest a ship
can legally travel in one decision. On the visible cell that is unreachable by
physics, so it *is* the leak rate; on the belief cells it reads how far the
recursion has drifted. Measured on run 747's checkpoint: 0.0000 visible, 0.906
hidden.

### Seat symmetry

Per the plan, the live policy stays on Team 0 (`elo_eval` slots 0–3) and nothing
randomizes the live seat. Coverage was strengthened instead.

The existing invariants compare the two seats of one game, which limits them to
the channels both sides can see, on the ships that happen to be visible to
everyone. The two new tests play one recorded command script twice, relabelling
the two sides in the second run, and require seat 1 of the game and seat 0 of its
mirror to be **the same tensors in every channel**, and then to produce the same
policy action, logprob, value, `pred_next` and hidden state. Nothing physical
depends on the labels and the win condition is symmetric, so the two runs stay in
lockstep — which the test checks first, so a failure is asymmetry rather than
divergence.

A recorded script rather than a controller, because a controller would have to be
provably side-blind *and* consume its randomness identically in both runs.
Non-Frontline, because a zone carries the side that owns it, so mirroring the
ships without mirroring the map is not a mirrored game — pinning the mirror for
Frontline's asymmetric map objects is worth doing and is listed below as deferred.

Mutation-checked: labelling opponents from a fixed team instead of the observer's
fails both new tests.

## Surprising repository facts worth knowing

* **`flip_team_obs` on a *raw* observation is not the inverse of building the
  other seat's view.** Hidden ships are masked to `TEAM_ID` 0, and zero is also
  Team 0's real id, so flipping turns a masked zero into a confident 1 where a
  native team-0 view would read 0. The contract only holds on the **composed**
  view, where `BeliefTracker.compose` restores `TEAM_ID` from its cache. This is
  the third instance of the zero-collides-with-a-legal-value pattern the
  seat-symmetry write-up names (`TEAM_ID`, `ZONE_ROLE`, and now this).
* **A ship dies on the tick it takes damage while already at zero health**, not
  the tick its health reaches zero (`_apply_damage_source` tests
  `health_before <= 0.0`). Provoking a respawn inside a single-tick decision
  therefore needs health set to exactly 0, not to 1.
* **Position is an absolute predictor.** `FourierMomentPredictor` extends
  `AbsolutePredictor`, so `apply_prediction` *discards* the current state. A zero
  prediction vector decodes to the origin of moment space — about 28,000 px of
  error — not to "no change". Any probe that wants an identity baseline must pass
  the believed target vector, or `compute_labels(believed, believed)` in scaled
  space.
* **Belief-cell measurements need a long warm-up.** Frontline ships deploy about
  2400 px apart and a decision is 1/60 s, so a few hundred decisions is a couple
  of seconds of game time in which no enemy is ever in sight and every belief
  cell is empty. A first attempt at 100 warm-up decisions measured `n=0` visible
  enemies; 3000 is where contact lives.
* **The belief diagnostic assumes a full buffer.** It reads `buf.num_steps`
  rather than `buf.ptr`, so a partially filled buffer decodes zeroed steps as
  truth and produces nonsense. Production always fills; probes must too.
* `pytest.ini` sets `-m "not slow"`, so the figure-rendering suite does not run
  by default.
* `ruff` is not clean on `main`: `src/.../train/rl/ppo.py` has an unsorted import
  block (`POWER_SLICE` after the `PRIVATE_*` names) and `tests/train/test_ppo.py`,
  `src/.../modes/interactive.py` and two `docs/internal` probe scripts are
  unformatted. All of that predates this phase and was left alone.

## Tests

Continuously during development, then broadly at the end.

| when | command | result |
|---|---|---|
| baseline, before any change | `pytest` (full suite) | 1706 passed in 421 s |
| after the latch fix | `pytest tests/env/ tests/train/test_belief.py tests/runtime_semantics/ tests/evaluation/` | 417 passed |
| latch fix, with `env.py` reverted | `pytest tests/env/test_spawn_lifecycle.py` | 4 of 5 failed, as intended |
| after the diagnostic change | `pytest tests/train/test_ppo.py::TestBeliefDiagnosticAlignment ::TestAuxPredictionMetrics` | 4 passed |
| after the mirror tests | `pytest tests/env/test_seat_symmetry.py` | 7 passed |
| mirror tests, with opponents labelled from a fixed team | same | 4 failed, as intended |
| final | `pytest` (full suite) | **1718 passed** in 430 s — 1706 baseline plus the 12 added here |
| final | `bnb smoke` | **all 16 isolated cases passed**, checkout unchanged |
| final | `benchmarks/belief_compile_parity.py --device cuda --seed 271828 --steps 12` | **`parity_passed: true`**, `error_count: 0`; both teams' belief state identical between the compiled and eager trackers over 12 steps |

New test files and classes:

* `tests/env/test_spawn_lifecycle.py` — 7 tests: the latch across `TensorEnv.step`
  and the wrapper, a respawn revealed for exactly one decision, a respawn on an
  early tick of a held action surviving to the end of it, fog actually resuming
  over 40 decisions of direct stepping, the two entry points agreeing, and the
  null pending action through `MatchRunner` and at the training path's initial
  queue.
* `tests/train/test_ppo.py::TestBeliefDiagnosticAlignment` — 3 tests: index
  agreement between the composed observation and privileged truth, which slices
  of the truth tensor the statistic responds to, and the leak series firing on a
  planted teleport.
* `tests/env/test_seat_symmetry.py` — 2 added: the mirrored-game view comparison
  and the same pair through the policy.

## Baseline performance

Artifact: [`perception-belief-phase0-baseline.json`](perception-belief-phase0-baseline.json).
Measured on the `rl` profile — 5v5 Frontline, 960 environments, 128 steps, 4
rollouts per update, so 491,520 environment steps per update — on an RTX 4070
Laptop (8 GB), torch 2.13.0+cu130, one warm-up and two measured updates each.

```
uv run --no-sync python benchmarks/rl_pipeline_profile.py --profile rl \
    --timing {wall|sync} [--perception eager] --updates 2 --warmup 1 --no-checkpoint
```

| run | perception | SPS | s/update | rollout s/update | update s/epoch | peak alloc | peak reserved |
|---|---|---|---|---|---|---|---|
| `rl-wall` | compiled (shipped) | **2,810** | 174.95 | 103.84 | 21.39 | 3,461 MiB | 4,656 MiB |
| `rl-wall-eager-perc` | eager | **2,864** | 171.60 | 105.50 | 19.85 | 3,461 MiB | 4,648 MiB |
| `rl-sync` | compiled (shipped) | 2,269 | 216.60 | 139.42 | 23.56 | 3,461 MiB | 4,656 MiB |

`--timing wall` syncs only at update boundaries, so its per-update total is the
truth while phase attribution under-reports queued GPU work; `--timing sync`
brackets every region with `cuda.synchronize`, so attribution is correct and the
total inflates by whatever overlap the syncs destroy. **Compare SPS across wall
runs only; read phase splits from the sync run.** The two are listed side by side
because doing otherwise is the standard way to misread this benchmark.

Perception and belief, from the sync run, per update:

| phase | s/update | calls/update |
|---|---|---|
| `01m_rollout/belief_compose` | 14.65 | 2048 |
| `01n_rollout/belief_advance` | 5.99 | 2048 |
| `01o_rollout/privileged_obs` | 3.06 | 516 |
| `01p_rollout/target_vector` | 2.31 | 1548 |
| **total** | **26.00** | |

That is **12% of the sync-attributed update** and is the number Phase 3 has to
beat, or knowingly pay. For context in the same run: `01k_rollout/elo_step` is
81.8 s/update — 59% of the rollout and the single largest line in the profile, far
larger than anything perception does. Worth knowing before optimizing the wrong
thing, and it is not Phase 3's to fix.

**Compiled perception is not currently paying for itself.** Eager perception
measured 2,864 SPS against the shipped compiled path's 2,810, a 1.9% *advantage*
for eager, at identical peak memory. Two measured updates cannot separate 1.9%
from run-to-run noise, so the honest statement is that the compiled perception
path is worth between nothing and nothing much on this hardware at this size — not
that eager is faster. Either way it removes "perception compilation is load-bearing
for throughput" as a constraint on Phase 3's redesign. If Phase 3 wants to settle
it, run both arms with more updates; the flag is `--perception {compiled|eager}`.

Compiled-versus-eager *behaviour* (as opposed to timing) is covered by the
existing `tests/env/test_compiled_tick.py` and `benchmarks/belief_compile_parity.py`,
both of which pass in the final validation below.

**Not measured, deliberately:** the `bc` profile. It shares `rl`'s environment
geometry, model config and perception/belief path and differs in the loss, so
profiling it would re-measure an identical code path for about half an hour. One
command away if a later phase wants it.

## Exit gate

| requirement | status |
|---|---|
| Spawn reveal lasts exactly one decision in every relevant stepping path | **met** — `TensorEnv.step`, `YemongEnvWrapper.step`, `step_interactive`; tested including the two agreeing |
| Respawn pending action is null and visible as specified | **partly** — the queue is null and each ship's own team sees it; the enemy team still reads the private category instead of the null. Deferred to Phase 2, which owns the spawn override |
| Fog resumes normally after reveal | **met** — tested at the latch level and as measured enemy visibility over 40 decisions of direct stepping |
| Seat-symmetry tests are clean | **met** — 7 tests, including a full mirrored-game comparison; mutation-checked |
| Belief diagnostics are temporally aligned | **met** — measured at 0.019 px, and pinned by tests. The discrepancy it was meant to resolve was not alignment |
| Baseline performance numbers are recorded | **met** — SPS, per-phase wall-time, VRAM and a compiled/eager arm, on the `rl` profile, in `perception-belief-phase0-baseline.json`; compiled/eager *behaviour* parity verified separately |

## Intentionally deferred

* The spawn override in the pending-action *observation*, so both teams read the
  null command for a spawning ship. Phase 2.
* A mirrored-game seat-symmetry test for Frontline, where zone roles and the
  front make the map itself carry a side. The mirror has to relabel zone roles
  and the front sign together with the ships; `flip_team` already swaps
  `ZONE_ROLE` in the observation, so the missing piece is a state-level mirror.
* Making seat identity unrepresentable rather than merely absent, by emitting a
  self-relative team feature (ego / ally / enemy) per perspective. This would
  delete `flip_team`'s `TEAM_ID` swap and the whole zero-collides-with-a-legal-value
  class. It touches the observation schema, so it belongs with Phase 3's direct
  view composition, not here.
* The two consumers that still merge by the observation's team id —
  `evaluation/next_state.py` and `modes/ar_report.py` — are correct today only
  because they receive composed observations. Left as the seat-symmetry write-up
  found them.
* Tracing which commit leaked the respawn teleports into run 747's
  `belief/visible/position_px`. Not recoverable from the checkpoint, and the run
  is being discarded.
* The pre-existing `ruff` findings listed above.

## Compatibility

No schema, checkpoint or config change. `state.ship_spawned` already existed and
kept its meaning; the fix is that one more caller now maintains it. The one new
metric key, `belief/<cell>/position_beyond_legal_frac`, is additive — no existing
series changed name or meaning.

Evaluation numbers taken through `TensorEnv` before `375cf12` are not comparable
with numbers taken after it: they were measured without fog. That includes every
live-Elo series and every `bnb` evaluation mode's historical output.

## For the next phase

Phase 1 (one-time physical delta calibration) is the recommended next block, and
it is unblocked. What it needs from here:

* **Use the wrapper, or `TensorEnv.step` at `375cf12` or later.** A calibration
  collected through a pre-fix `MatchRunner` would be measured in a fog-free
  world. Fog does not change truth→truth deltas, but it changes what the
  scripted agents do, and Phase 1's deltas are supposed to be representative of
  the trained environment.
* **Exclude respawns with `transition_contiguous`, or with `ship_respawned`
  after the step, which is the same fact.** Phase 0's measurement of what a leak
  costs is the argument for being strict here: 0.2% of tokens moved a mean
  eighteen-fold. Phase 1 should report the excluded fraction alongside its
  constants.
* Position and velocity each need **one shared scale** across x and y, and zero
  physical delta must map to exactly zero normalized delta. Note that the
  *current* position predictor is absolute, not a delta — Phase 3 is where the
  11 physical mean deltas replace it, so Phase 1 is measuring for a model that
  does not exist yet.
* `docs/internal/next-state-head-audit-sep2026.json` already holds per-dimension
  label statistics for the current encoding on run 747's checkpoint. Useful as a
  cross-check, but it is in scaled label space, not physical units, and the run
  is contaminated.
