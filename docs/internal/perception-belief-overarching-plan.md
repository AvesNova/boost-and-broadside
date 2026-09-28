# Boost & Broadside — Perception and belief plan (v2)

Repository: https://github.com/AvesNova/boost-and-broadside

This is the implementation contract for partial observability, action belief,
physical state belief, the global/game token, and global density supervision.
It replaces the v1 plan (archived at
`archive/perception-belief-overarching-plan-v1.md`). Phases 0–3 are complete;
their result is summarized in `perception-belief-status.md`, which is the
single handoff document for this work from now on.

It is not a line-by-line recipe. The phase owner has the repository, the tests
and the status document in front of it and should make ordinary engineering
decisions locally. It should not change the semantics in "Decided model
semantics" without saying so in the status document and stopping.

## Guiding principles

- Correctness and information-flow safety before model experiments.
- Privileged truth may be an auxiliary-loss target; it never enters policy
  inputs.
- Team 0 and Team 1 are symmetric. Differences are bugs, not noise to average
  over. The live policy stays on Team 0.
- Build legal observations from the correct sources. Never build a privileged
  observation and patch it.
- One representation, one source of truth, no compatibility layers kept past
  the comparison that needed them.
- Maintain throughput, but **measure once and only when the phase says to**.
  Benchmarks are expensive; a phase's gate says whether one is required.
- Each phase is small enough to finish in one agent session. If it is not, the
  phase is wrong, not the agent.

---

# Decided model semantics

These are settled. Do not reopen them inside a phase.

**Spawn and respawn.** Respawn is instantaneous; no persistent dead ships. On
initial spawn and every respawn a ship is fully visible to both teams for that
decision, its pending action is exactly the null action, and both teams see that
null action. Fog and action belief resume afterward. Next-state losses whose
destination is a respawn are masked.

**Pending actions.** The observation is exactly 42 floats, the physical joint
action space. Own/allied pending actions are exact one-hots. Hidden enemy
pending actions are the stored 42-way predicted distribution. True enemy
commands are supervision targets only and never enter policy input outside the
spawn reveal.

**Enemy action prediction.** A dedicated head, separate from but identically
shaped to the actor head, sharing the trunk, trained with cross-entropy against
the committed joint enemy action, produced by the same forward pass. Physical
ship indexing is authoritative; canonicalization relabels perspective and never
reorders slots.

**Belief state is physical.** Eleven means per ship in fixed order — position
x/y, velocity x/y, attitude, angular velocity, shield delay, health, power,
cooldown, natural-log local index — in physical units. Thirteen uncertainty
terms: full 2D covariance (two log sigmas, one unconstrained correlation
latent) for position and for velocity, one log sigma each for the seven scalar
channels. Uncertainty stays in log/unconstrained form everywhere. Visible truth
carries a finite certainty floor; a never-seen slot carries the ceiling.

**Next-state model.** Predicts the eleven normalized mean deltas and the
thirteen uncertainty terms. Gaussian NLL: bivariate for position and velocity,
scalar for the rest. Fixed scales from the Phase 1 calibration (one shared
scale for x/y position, one for x/y velocity; zero maps to exactly zero). The
head sees the current uncertainty and states the next one directly; nothing
accumulates variance. Propagation uses the predicted mean deterministically.
The label is `truth[t+1] − believed[t]`, so on a hidden ship the head learns to
correct the belief toward truth, with its sigmas saying how far to trust that
correction. This is deliberate: the head began as a hard auxiliary task that
helps the policy learn, and it now doubles as the estimator of where hidden
ships are. **Do not give it a dead-reckoning or any other physics prior**;
learning the dynamics is the point. Dead reckoning stays as a baseline
diagnostic only.

**Composition.** Every ship slot takes its physical state from exactly one
source in one selection: truth where the observer owns or currently sees it,
the observer's belief where it cannot, zero where nothing was ever observed.
Then canonicalize, then run the ordinary encoder. Derived multi-ship features
(local presence, relational geometry) are computed from the composed legal view.
Each observer composes its own view from its own belief.

**Global/game token.** A permanent token in every mode, with its own
split-encoder input path, then the same Yemong trunk as ship tokens (spatial
attention, recurrent layers, FFNs, residuals). Recurrent/query set = ships +
global token; fields/zones stay K/V-only. Spatial location: map center. Game
mode is categorical when modes are mixed. TeamPMA unchanged.

**Global density target.** The global token predicts *current* ally and enemy
density on a fixed hex grid over the playable circle (roughly 271–331 cells),
using the same Gaussian-density + `log1p` formulation as `local_presence`, as a
flat vector, with MSE. Targets use privileged truth; inputs stay legal.

**Explicitly deferred.** Per-enemy categorical position grids; cross-entropy
for the density field; SDF targets; mixtures or codebooks; per-harmonic state
prediction; online delta scaling; sampling belief means; manual variance
accumulation; actor/enemy-head sharing; TeamPMA replacement before the
global-token ablation.

---

# What exists (Phases 0–3, complete)

Read `perception-belief-status.md` for the invariants, the file map, the known
caveats and the measured numbers. In one paragraph: the spawn-reveal lifecycle
is correct in every stepping path and pinned by tests; seat symmetry is tested
as a mirrored game; the delta scales are calibrated and committed in
`train/rl/physical_deltas.py`; the 42-way pending-action observation and the
dedicated enemy head are in; the belief plane is physical, composition is a
single selection, the next-state head is 11 + 13 wide with a Gaussian
objective, and the rollout composes after the stream join. Throughput is above
the Phase 0 baseline. The old encoded-belief machinery is inert, untested, and
waiting to be deleted.

---

# Remaining phases

Each phase below has a scope, a gate, and a "not in scope" line. The gate is
the stopping rule: when it is met, write the status update and stop. Nothing
that is not in the gate needs to be measured.

## Phase 4 — Delete the legacy belief machinery

**Scope.** Delete everything on the removal list in the status document
(`ObsKey.BELIEF_TARGETS`/`BELIEF_SUBSTITUTE`, the `Predictor` hierarchy and
`predictor=`/`label_scale=` arguments in `features.py`, the coordinator's
target/prediction/uncertainty dimension helpers, `_believed_rotary_tables`,
`aux_weights`, and any `Fourier.invert`/`UnitCircle` code with no remaining
reader). Update `docs/architecture.md` and `docs/training.md` where they
describe the removed path. Fix audit items 2–4 from
`perception-belief-audit-sep2026.md` while in the same files. Optionally leave
the evaluator's single-observer builder eager if it shortens startup and
nothing else moves.

**Gate.** `ruff check src tests benchmarks` clean; the default `pytest` run
passes; `bnb smoke` passes once. No benchmark: this is a deletion.

**Not in scope.** Any change to head width, belief layout, or composition.
Any new diagnostic. Any training run.

## Phase 5 — Promote the global/game token

**Scope.** Make the global token permanent across modes with its own
split-encoder input; route ships + global token through the full
query/recurrent trunk; keep fields/zones K/V-only; remove any assumption that
recurrent-token count equals ship count; keep ship heads ship-only; keep
TeamPMA unchanged; keep rollout/update recurrent equivalence; represent mode
categorically. **Keep a config switch that builds the model without the global
token.** "Permanent" means on by default in every mode, not unconfigurable:
Phase 8's variant A needs the token off, and adding that back later would be a
rewrite of the kind Phase 3 did to Phase 2.

**Gate.** Tests show: the global token receives recurrent state and trunk
updates; map objects remain K/V-only; ship heads do not consume the global
token; rollout and sequence re-evaluation agree. Run the end-to-end benchmark
**once** (`benchmarks/rl_pipeline_profile.py --profile rl --timing wall
--updates 2 --warmup 1 --no-checkpoint`) and record SPS and peak VRAM against
the Phase 3 row in the status document. If SPS drops more than 10%, report it
and stop rather than optimize.

**Not in scope.** Density targets or head. TeamPMA changes. Large-fleet
scaling studies.

## Phase 6 — Hex density target as a benchmarked primitive

**Scope.** Fixed-count hex grid over the playable circle with stable cell
ordering; two privileged targets (ally and enemy current density) with the
`local_presence` kernel and `log1p`; exact computation, fused/compiled, without
materializing a sample × ship × cell × coordinate intermediate. A dedicated
micro-benchmark for the target alone.

**Gate.** A numerical test against a naive reference on a small grid; the
micro-benchmark shows target generation costs under 5% of a rollout step's
wall time at production width (960 envs, 5v5). No end-to-end benchmark.

**Not in scope.** Attaching a loss. Truncated or approximate kernels (only
after the exact one is measured too slow). A custom CUDA kernel unless compiled
PyTorch misses the 5% bar.

## Phase 7 — Global density auxiliary head

**Scope.** Read the final global-token embedding, predict both density
vectors, MSE on the log-density targets, privileged truth only in the target
path, gradient diagnostics (norm and cosine against actor, critic, next-state
and enemy-action losses), and a coefficient chosen from those diagnostics.

**Gate.** Tests show target/view alignment and no privileged leakage; the
loss and its diagnostics appear in the metric stream through a PPO smoke test;
one end-to-end benchmark run as in Phase 5, with the same 10% rule.

**Not in scope.** Any training run longer than a PPO smoke test. Coefficient
sweeps.

## Phase 8 — Controlled architecture experiments

Only after 5–7 are stable, and only with the user's explicit go-ahead on
budget, because this phase is training runs.

| Variant | Global token | Density supervision | TeamPMA |
|---|---|---|---|
| A | No | No | Yes |
| B | Yes | No | Yes |
| C | Yes | Yes | Yes |

A is the Phase 5 switch off; B is the switch on with the Phase 7 coefficient at
zero; C is both on. No new code should be needed for any arm.

Questions: does the recurrent global token help (A→B)? Does density
supervision add value (B→C)? Screening runs first; multi-seed only where a
screening difference is worth confirming. Also answer the open question from
Phase 3: does the next-state head beat dead reckoning at convergence, and at
what hidden age?

---

# Status update requirement

There are no per-phase handoff files any more. A phase ends by editing
`perception-belief-status.md`:

- move the phase from "Remaining" to "Done" with three to ten lines: what
  changed, in which files, and any decision that a later phase must respect;
- add caveats or open questions to the existing lists, remove the ones the
  phase closed;
- if the gate required a benchmark, add one row to the numbers table;
- update the invariants list only if an invariant changed, and say why.

Hard limits: the status document stays under 250 lines total. No test counts,
no wall times of test runs, no commit hashes, no incremental validation tables,
no narrative of what was tried and reverted. Git history carries that.
