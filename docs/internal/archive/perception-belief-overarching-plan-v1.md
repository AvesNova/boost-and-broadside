# Boost & Broadside — Overarching Implementation Plan

Repository: https://github.com/AvesNova/boost-and-broadside

This document is the high-level implementation contract for the next sequence of work on partial observability, action belief, physical state belief, the global/game token, and global density supervision.

It is intentionally **not** a line-by-line implementation recipe. Each phase should be owned by an agent with the repository, tests, profiling tools, and the previous phase handoff in front of it. The phase owner should make reasonable local engineering decisions, but should not silently change the model semantics established here.

## Guiding principles

- Correctness and information-flow safety come before model experiments.
- The live policy may remain Team 0. Team 0 and Team 1 are intended to be symmetric; differences should be exposed as bugs, not averaged away by randomizing seats.
- Privileged truth may be used for auxiliary-loss targets, but must never leak into policy inputs.
- Truth state and belief state should share the same physical meanings and observation slots wherever possible.
- Build legal observations directly from the correct sources. Do not construct privileged observations and then zero/patch them afterward.
- Prefer one simple representation and one source of truth over parallel special-case pathways.
- Maintain or improve throughput. Benchmark performance-sensitive changes rather than assuming they are cheap.
- Make each phase independently testable and leave a durable handoff for the next phase owner.
- Avoid carrying experimental compatibility layers forever. Keep old paths only long enough to make controlled comparisons, then delete them once the replacement is validated.

---

# Decided model semantics

These are already decided and should not be reopened casually inside implementation phases.

## Spawn and respawn

- Respawn is instantaneous; there are no persistent dead ships.
- On initial spawn and every respawn, a ship is fully visible to both teams for that decision.
- Its pending action is exactly the null action: no thrust, no turn, no shoot.
- Both teams know that null pending action during the spawn/reveal decision.
- Normal fog/action-belief semantics resume afterward.
- All next-state losses whose destination is a respawn are masked.

## Pending actions

- Pending action observation is exactly **42 floats**, matching the physical joint action space.
- No extra “unknown” action category is needed.
- Own/allied pending actions are exact one-hot vectors.
- Hidden enemy pending actions are predicted 42-way probability distributions.
- Enemy true actions may be used as privileged supervision targets.
- Outside explicit spawn/reveal semantics, true enemy pending actions must not leak into policy inputs.

## Enemy action prediction

- Use a **dedicated enemy-action prediction head**, separate from the actor head.
- It may initially share the normal Yemong trunk.
- Train it with cross-entropy against the actual committed joint enemy action.
- No extra policy forward pass should be required.
- Physical ship indexing must remain authoritative; team canonicalization must not create index remapping or cross-seat leakage.
- Measure trunk-gradient interaction with the policy loss rather than preemptively isolating the auxiliary task.

## Next-state belief means

Predict 11 physical mean deltas per ship:

1. position x fundamental phase delta
2. position y fundamental phase delta
3. velocity x delta
4. velocity y delta
5. attitude wrapped angular delta
6. angular-velocity delta
7. shield-delay delta
8. health delta
9. power delta
10. cooldown delta
11. local log-index delta

Input encoding remains the normal policy encoding. Position/attitude may still be Fourier-expanded for the network; velocity and other features may still use their existing input transforms. The belief system itself should store physical state, not Fourier moments.

## Delta scaling

Use a one-time scripted-agent calibration.

- Measure truth→truth one-decision deltas.
- Exclude death→respawn transitions.
- Choose fixed constants for the first implementation.
- **Position uses one shared scale for x and y.**
- **Velocity uses one shared scale for vx and vy.**
- Other scalar features may have their own scales.
- Zero physical delta must map to exactly zero normalized delta.
- Record the measurement and chosen constants so the result is reproducible.

Do not introduce an online/moving delta scaler in the initial implementation.

## Uncertainty

Use full 2D covariance for position and velocity.

Position uncertainty:
- log sigma x
- log sigma y
- unconstrained correlation latent mapped to rho in (-1, 1)

Velocity uncertainty:
- log sigma vx
- log sigma vy
- unconstrained correlation latent mapped to rho in (-1, 1)

Scalar log-sigma terms:
- attitude
- angular velocity
- shield delay
- health
- power
- cooldown
- local log-index

Total: **13 uncertainty outputs**.

Uncertainty is kept in log/unconstrained form in observations/head outputs. Visible truth uses a finite certainty floor rather than mathematical log(0).

Each prediction describes the **next belief uncertainty directly**. Do not manually accumulate old variance. The model sees the current uncertainty as input and predicts the next uncertainty.

Belief propagation uses the predicted mean deterministically; do not sample a state from the Gaussian.

## Global/game token

- A permanent global/game token exists in every game mode.
- It has its own split-encoder input path.
- After encoding, it traverses the same Yemong trunk path as ship tokens, including spatial attention, Griffin/recurrent layers, FFNs, and residual paths.
- Do not create a bespoke second trunk just for the global token.
- The recurrent/query set becomes ships + global token.
- Fields/zones remain K/V-only map memory.
- Use map center as the global token’s spatial location unless implementation evidence reveals a concrete problem.
- Game mode should be represented categorically when multiple modes are trained together.
- TeamPMA remains unchanged until later experiments justify replacing it.

## Global density target

- The global token predicts **current** ally density and **current** enemy density.
- Targets may use privileged truth; policy inputs remain fog-legal.
- Use a fixed-count hex grid centered on the playable circular area, roughly 271–331 cells initially.
- Output is a flat vector; no CNN/grid-specific architecture is required.
- Reuse the same conceptual Gaussian-density + `log1p` formulation as current `local_presence`, initially using the same physical radius.
- This is a continuous, cardinality-carrying density field, so the initial loss is **MSE, not cross-entropy**.
- Density target generation must be engineered as a performance-sensitive operation. Do not accept a naïve implementation that materially reduces end-to-end training throughput.

---

# Phase order

## Phase 0 — Correctness, lifecycle, and measurement foundation

### Purpose

Make the observation/evaluation machinery trustworthy before introducing new model behavior.

### Scope

- Fix direct `TensorEnv.step()` spawn-reveal lifecycle so the reveal lasts exactly one decision, consistent with the intended semantics.
- Enforce/test null pending action at every initial spawn and respawn.
- Keep the live policy on Team 0.
- Strengthen seat-symmetry regression coverage rather than randomizing live seat.
- Resolve the known timing discrepancy in belief diagnostics.
- Establish baseline performance measurements for the current perception/belief path: SPS, relevant wall-time components, VRAM, and compiled/eager behavior.

### Seat-symmetry philosophy

Team identity should be a symmetry. Build explicit paired/mirrored tests that compare canonicalized observations and, where practical, model outputs/logits/value/auxiliary predictions. If Team 0 and Team 1 differ under the intended symmetry, treat that as a defect.

### Exit gate

- Spawn reveal lasts exactly one decision in every relevant stepping path.
- Respawn pending action is null and visible as specified.
- Fog resumes normally after reveal.
- Seat-symmetry tests are clean.
- Belief diagnostics are temporally aligned.
- Baseline performance numbers are recorded.

---

## Phase 1 — One-time physical delta calibration

### Purpose

Measure the fixed normalization constants needed by the new physical-delta next-state model before changing that model.

### Scope

- Run representative scripted-agent trajectories.
- Collect truth→truth one-decision deltas.
- Exclude transitions landing on respawn.
- Measure robust distributions for all predicted physical quantities.
- Choose and document the fixed scaling constants.
- Position x/y share one scale.
- Velocity x/y share one scale.
- Keep zero exactly zero.

### Exit gate

A reproducible calibration artifact exists with:
- collection conditions;
- distributions/statistics;
- selected constants;
- justification for sparse/event-like features;
- confirmation that position and velocity each use one symmetric shared scale.

---

## Phase 2 — Joint pending-action belief and dedicated enemy-action prediction

### Purpose

Put action belief onto its final representation before training the new physical dynamics model.

### Scope

- Replace the current factorized pending-action observation with the 42-way joint representation.
- Allies/own-team ships use exact one-hot pending actions.
- Spawn/respawn uses exact null-action one-hot for everyone.
- Hidden enemies use the stored 42-way predicted distribution.
- Add a dedicated enemy-action prediction head.
- Train with privileged realized committed enemy actions using cross-entropy.
- Preserve physical ship indexing and avoid extra forward passes.
- Define and test exact temporal alignment between prediction at decision `t`, committed action at `t`, and pending-action belief visible in observation `t+1`.

### Diagnostics

At minimum measure:
- enemy-action CE;
- probability assigned to the realized action;
- entropy/calibration-style diagnostics;
- simple baselines such as uniform and action persistence;
- trunk gradient norm/cosine versus the policy loss.

### Exit gate

- No action-information leakage.
- No index mismatch under team canonicalization.
- Spawn override is correct.
- Temporal alignment is explicitly tested.
- The auxiliary task behaves sensibly relative to trivial baselines.
- Throughput cost is measured.

---

## Phase 3 — Physical belief state, direct legal-view composition, and Gaussian next-state model

### Purpose

Replace the current “masked observation + restore/override belief” architecture with a simple physical belief data plane and train the new physical-delta model against it.

This is the central architectural phase.

### Desired data model

Maintain observer-specific physical belief state as needed by the training mode.

For each observer, belief carries the ship state needed to stand in for hidden truth, including:
- physical means;
- 42-way pending-action belief;
- 13 uncertainty terms;
- observation age / visibility-related epistemic state.

### Belief update

Visible enemy:
- assimilate truth into the belief;
- reset uncertainty to the certainty floor;
- reset age;
- use permitted action visibility semantics.

Hidden enemy:
- add predicted mean deltas;
- replace uncertainty with the newly predicted uncertainty;
- carry the stored predicted pending-action distribution;
- advance age.

Respawn:
- assimilate fully visible truth for both teams;
- set null pending action;
- reset uncertainty/age;
- mask the incoming next-state training loss.

### Direct observation construction

Do not:
- build privileged ship observations and then zero hidden channels;
- restore hidden metadata piecemeal;
- copy encoded belief targets into selected columns later;
- maintain a separate Fourier-moment belief substitution path.

Instead:

1. Start from authoritative truth and observer-specific physical belief.
2. Select the legal physical source for each ship.
3. Apply ego/team canonicalization.
4. Run normal feature processing/encoding on that already-legal view.
5. Place the resulting features directly into the observation.

For a Team-0 view, own ships come from truth and opponent ships come from Team-0’s belief. Because visible opponents are assimilated into that belief before composition, the source rule remains simple.

Construct both team views in a vectorized/compiled way where useful. It is acceptable to concatenate the two legal views and run shared Fourier/scaling/derived-feature processing over a `2B` batch.

Derived features that depend on multiple ships, such as local presence or relational geometry, must be computed from the legal observer-specific battlefield view, not from privileged truth and patched afterward.

### Next-state model

- 11 mean deltas.
- 13 uncertainty outputs.
- Full 2D Gaussian NLL for position and velocity.
- Scalar Gaussian NLL for scalar features.
- Fixed scales from Phase 1.
- Current uncertainty is part of the observation.
- No mixtures, codebooks, Huber, or per-harmonic state prediction initially.

### Exit gate

- Information-flow tests demonstrate that hidden truth cannot reach opponent observation slots through the new composition path.
- Truth and belief use the same normal feature encoder.
- Respawn semantics and loss masking are correct.
- New belief diagnostics are coherent.
- Hidden-state performance is compared with persistence/dead-reckoning baselines.
- New path performance is benchmarked against the Phase-0 baseline.
- The old belief system may remain temporarily only if needed for a controlled comparison.

---

## Phase 4 — Remove legacy belief/perception machinery

### Purpose

Once Phase 3 is validated, simplify the codebase so the old architecture cannot silently return or diverge from the new one.

### Scope

Remove or retire machinery that exists only for the old encoded-belief pathway, including the concepts corresponding to:
- zero-hidden-then-repair observation construction;
- encoded `BELIEF_TARGETS` / substitution logic;
- belief-column override machinery;
- Fourier-moment-specific belief substitution;
- special believed rotary overrides;
- stale restoration/caching paths and obsolete comments/docs.

Update architecture/training documentation and checkpoint/schema handling as needed.

### Exit gate

- There is one clear production belief/observation path.
- Tests cover its information-flow contract.
- No dead compatibility path remains without an explicit reason.
- Performance is at least acceptable relative to the recorded baseline.

---

## Phase 5 — Promote the global/game token

### Purpose

Create a persistent global workspace without yet conflating the experiment with density supervision.

### Scope

- Make the global token permanent across modes.
- Keep its dedicated split-encoder input path.
- Route ships + global token through the full query/recurrent Yemong path.
- Keep fields/zones K/V-only.
- Remove assumptions that recurrent-token count always equals ship count.
- Keep ship-specific heads ship-only.
- Keep TeamPMA unchanged.
- Preserve rollout/update-time recurrent equivalence.
- Represent mode/global match metadata cleanly.

### Exit gate

- Global token really receives recurrent state and full trunk updates.
- Map objects remain K/V-only.
- Ship heads do not accidentally consume the global token.
- Rollout and sequence re-evaluation agree.
- 5v5 and large-fleet throughput/VRAM impacts are measured.

---

## Phase 6 — Optimized fixed hex-density target generation

### Purpose

Build and benchmark the density target as an efficient primitive before attaching a new training loss.

### Scope

- Fixed-count hex grid over the playable circle.
- Stable cell ordering.
- Two privileged truth targets: ally and enemy current density.
- Same conceptual Gaussian kernel + `log1p` semantics as `local_presence`.
- Exact computation first.
- Avoid retaining a giant sample × ship × cell × coordinate intermediate.
- Prefer a fused/compiled accumulation.
- If ordinary compiled PyTorch is not fast enough, use an appropriate purpose-built GPU kernel.
- Only consider truncated/local approximations after benchmarking the exact implementation.

### Exit gate

- Numerically validated target.
- Dedicated performance benchmark exists.
- End-to-end cost is quantified.
- Implementation is cheap enough to justify adding to training.

---

## Phase 7 — Global density auxiliary head

### Purpose

Test whether explicit battlefield-aggregation supervision improves the global token.

### Scope

- Read from the final global-token embedding.
- Predict ally and enemy density vectors.
- Use MSE on the continuous log-density targets.
- Use privileged truth only in the target path.
- Add gradient diagnostics for interaction with actor, critic, next-state, and enemy-action losses.
- Choose the auxiliary coefficient from measured behavior rather than arbitrary magnitude matching.

### Exit gate

- Correct target/view alignment.
- No privileged leakage.
- Stable training.
- Density accuracy and gradient interaction are measured.
- End-to-end throughput impact is known.

---

## Phase 8 — Controlled architecture experiments

Run controlled comparisons only after the preceding systems are stable.

Primary comparison:

| Variant | Global token | Density supervision | TeamPMA |
|---|---|---|---|
| A | No | No | Yes |
| B | Yes | No | Yes |
| C | Yes | Yes | Yes |

Questions:
- A → B: does the recurrent global token help by itself?
- B → C: does explicit global-density supervision add value?

Use screening runs before expensive full-budget/multi-seed experiments. Compare sample efficiency as well as endpoint performance.

Only after the global token demonstrates value should a later experiment consider replacing TeamPMA.

---

# Explicitly deferred

Do not add these unless later evidence motivates them:

- per-enemy categorical position grids;
- cross-entropy for the current density-field target;
- SDF-to-nearest-ship as the primary density target;
- trajectory mixtures or codebooks;
- mixture-density next-state heads;
- per-Fourier-harmonic state prediction;
- online/moving next-state delta scaling;
- stochastic sampling of belief means;
- manual variance accumulation across hidden steps;
- actor/enemy-action head sharing;
- TeamPMA replacement before the global-token ablation is complete.

---

# Required phase handoff

Every implementation phase must leave a durable handoff for the next agent. The exact repository filename may follow existing project conventions, but the handoff must contain:

- phase name and intended scope;
- branch and relevant commit SHAs;
- what was actually implemented;
- important design decisions made locally and why;
- files/components materially changed;
- tests run and their results;
- benchmarks/profiles run and their results;
- new metrics or diagnostics introduced;
- known defects, caveats, or unresolved questions;
- anything intentionally deferred;
- compatibility/checkpoint implications;
- exact preconditions for the next phase;
- recommended next phase/block;
- any surprising repository facts the next agent should know.

The next phase owner should be able to begin by reading the overarching spec, the latest handoff, and the relevant code—without rediscovering the previous phase’s history from scratch.
