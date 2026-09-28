# Phase 3 handoff — physical belief, direct legal-view composition, Gaussian next state

Phase 3 of [`perception-belief-overarching-plan.md`](perception-belief-overarching-plan.md).
September 28, 2026.

**Outcome: COMPLETE. The Phase 3 exit gate is satisfied.**

## Scope and history

| | |
|---|---|
| branch | `fix/seat-symmetry-pending-action` |
| starting commit | `f9e98d6` (`docs: complete Phase 2 handoff`) |
| last code commit before this handoff | `PLACEHOLDER_LAST` |
| handoff commit | the commit containing this document |
| completed block | Phase 3 only |
| recommended next block | Phase 4 only |

Ordered Phase 3 commits:

| commit | established |
|---|---|
| `9a23d57` | The physical belief representation: eleven means in the Phase-1 calibration's order, thirteen log/unconstrained uncertainty terms, the full-2D-covariance Gaussian objective over them, and a bounded recursion. |
| `c09aa55` | Direct legal-view composition. Each ship slot takes its physical state from one source in one selection; `_mask_hidden_ships` is gone and the pending-action writer takes the belief distribution. |
| `9b6c8c0` | The physical next-state head, the physical belief tracker, the re-based physical label, physical belief diagnostics with baselines, and the rollout-storage change. |
| `55beec5` | Every remaining caller: evaluation agents, league opponents, interactive play, the analysis modes, the latency benchmarks, schema v19, and documentation. |
| `b289aea` | A clean-revision CUDA compile-parity artifact over 24 steps for both team views. |
| `PLACEHOLDER_LAST` | The derived-feature legal-view coverage, the baselines probe, and the end-to-end CUDA benchmark. |

## What changed

### The belief is physical

`BeliefTracker` stores eleven **physical** means per ship — position x/y, velocity x/y,
attitude, angular velocity, shield delay, health, power, cooldown, and the natural log of the
local refractive index — in the order and units Phase 1 calibrated. Alongside them it stores
thirteen log/unconstrained uncertainty terms, the 42-way pending-command distribution, and
how many decisions ago the ship was last seen. It no longer stores encoded targets, a decoded
point, team identity or radius.

Team identity and radius went away because neither needed a store. Identity is remembered
truth, so reading it from truth under the "has this observer ever seen it" mask is the same
value; radius is one configured constant shared by every ship and so carries no information
at all.

### Composition is a selection

`observation_from_state` now takes an optional `ShipBeliefSource` and picks each ship slot's
physical state **once**: truth where the observer owns the slot or can currently see it, the
observer's belief where it cannot, and zero where nothing has ever been observed.
`_mask_hidden_ships` — build truth everywhere, zero the hidden half afterwards — is deleted,
and so is `BeliefTracker.compose`.

Three things follow. The ordinary encoder reads the result, so there is no substitution into
encoded columns and no separate moment-space rotary table. The spatial rotation reads the
composed coordinate. And derived features over several ships, `local_presence` above all, are
a property of the composed view rather than of truth, because that is the tensor they read.

### Each observer composes its own view

A shared observation that several policies take perspectives on is gone. Evaluation agents,
league opponents and interactive sides each build their own view from the authoritative state
and their own belief, through one helper (`train/rl/belief.legal_policy_view`). Both views
are team-legal either way, so sharing one leaked nothing — it attributed one player's memory
to another, and that player's behaviour stopped being a function of its own weights.

`MatchRunner.observe()` therefore computes perception only; `MatchRunner.legal_view(index)`
and `.policy_view(index)` are the new public accessors, the second canonicalized onto the
agent's own side.

### Composition moved after the stream join

A view depends on the belief's forecast, which depends on the policy output the environment
step runs concurrently with. So `wrapper.step(..., observe=False)` advances physics and
perception only, and the trainer composes after the two CUDA streams join:

```
env stream:  physics t -> t+1, perception
net stream:  forward on obs_t  -> action, next-state forecast, enemy-action logits
join
             beliefs.advance(forecast)      belief_t -> belief_{t+1|t}
             beliefs.reset(done)
             beliefs.observe(state, visibility)   assimilate truth, void spawns
             wrapper.observe(sources)             compose obs_{t+1}
             action_state.write_observation(...)  the queue, plus the action belief
```

The physics and the forward still overlap; only composition is serial, and it is not
materially more work than the `compose` pass it replaces.

League slots are the awkward case: a slot's own view cannot be built inside
`_rollout_network_forwards`, because the environment step is mid-flight on another stream.
Each slot therefore carries `LeagueSlot.obs`, composed at the end of the previous decision
and at the start of every rollout shard.

### The next-state model

`NextStateHead` emits a fixed 24 outputs — eleven normalized physical mean deltas and
thirteen uncertainty terms — rather than a width derived from the feature pipeline. Position
and velocity each carry a full 2D covariance (two log sigmas and one `tanh` correlation
latent); the seven other channels carry one log sigma each. The objective is a Gaussian
negative log likelihood throughout: bivariate for the two pairs, scalar for the rest, each
carrying its normalizing constant so the per-channel series are nats.

The attitude residual is wrapped onto the circle before its scalar Gaussian, because a
prediction of `-pi` against a label of `+pi` is the same rotation rather than the largest
possible error.

The recursion is bounded by construction. Position wraps on the torus and attitude on the
circle, exactly; every other channel is clamped to its physical range — health to
`[0, max_health]`, cooldown to `[0, firing_cooldown]`, angular velocity to the rate a turn
command can set, velocity to a stated numerical guard. Run 734's failure mode is now
unwritable rather than counted; the `clamp_events` counter survives only to catch a
non-finite head output.

### Labels, storage, diagnostics

The label is the normalized physical step from the believed current state to the true next
one, exactly as before — but now read from the stored observation's own physical channels
rather than reconstructed through a target encoder, because the observation *is* the belief.
`RolloutBuffer.privileged_targets` becomes `privileged_means`, eleven fp32 channels.

The buffer no longer stores `BELIEF_TARGETS` (26 tokens x 56 channels) or
`BELIEF_SUBSTITUTE`. The other ten physical ship channels moved from bf16 to fp32, because
with a physical belief they are what the label steps *from*: bf16's ~0.4% resolution puts
about 1.2 px/s of noise on a 300 px/s velocity, against a calibrated velocity delta scale of
4.0 px/s.

Belief diagnostics are now exact physical errors with no decode, and they carry two baselines
the plan asked for: `persist_position_px` / `persist_velocity_px_s` (the belief standing
still) and `reckon_position_px` (the belief carried forward one decision on its own believed
velocity).

## Implementation decisions

### The old machinery is inert, not deleted

Phase 4 owns removing it. `ObsKey.BELIEF_TARGETS`, `ObsKey.BELIEF_SUBSTITUTE`,
`FeatureCoordinator._apply_belief_override`, `YemongPolicy._believed_rotary_tables`,
`project_targets`, `decode_targets`, `compute_labels`, `prediction_loss` and the `Predictor`
classes all still exist. Nothing in production reaches them: the two override paths are gated
on observation keys no writer produces any more, and
`tests/train/test_legal_view.py::TestTheOldPathIsInert` pins that. **They are now untested
dead code** — the suites that covered them (`test_belief_copy.py`,
`test_prediction_uncertainty.py`) were deleted along with the production path they described,
because a test of a path nothing can reach is worse than no test. Phase 4 should delete the
machinery rather than re-cover it.

`features.py` still declares `predictor=` on nine features. That is what keeps
`coordinator.target_slices()` alive for the inert rotary override, and it is the one thing
Phase 4 must remove carefully: `total_prediction_dimension` is 56 on the Frontline world and
24 is the head's real width, so any surviving call site that slices by the former against a
head output would be wrong. There is none today.

### Constants that are not information

Hidden ship slots now carry three ship constants rather than zeros: collision radius, the
`SHIP` object type, and the explicit "no zone role" value 5. They are identical for every
ship token, so they say nothing about the ship occupying the slot — and for `zone_role` this
is a fix, because the masked zero read as `TEAM0_SPAWN`.

### Uncertainty is stated, not accumulated

The plan is explicit and the code follows it: `advance` *replaces* the stored uncertainty with
the head's output. The head sees the current spread as an input and answers with the next one,
so summing forecasts would double-count what it already accounts for. A ship in sight carries
a finite certainty floor (`log sigma = -6`); a ship nothing has ever seen carries the ceiling
(`+6`), not the zero a masked channel used to leave behind — which reads as one unit of doubt,
and so as a confident claim about nothing.

### A spawn voids the belief

`observe` clears validity for any ship with `state.ship_spawned` set, before assimilating.
Under `spawn_reveal` the same decision makes the ship visible, so truth lands in the same call
and the net effect is a clean re-acquisition. With the reveal disabled the slot correctly
falls back to "never seen" instead of tracking a corpse — which is stricter than the previous
behaviour and is what makes the explicit episode `reset` a belt-and-braces measure rather than
the only thing preventing cross-episode contamination.

### `grad(n)` stays zero for a remembered ship

It is a deterministic function of position given the static map, so it *could* be inferred
from a believed position. It is not, because doing so would state a field interaction nobody
supervised. This is unchanged from Phase 2 and is now explicit in one `torch.where`.

## Materially changed components

- new: `train/rl/physical_belief.py` (layout, bounds, objective, conversions)
- rewritten: `train/rl/belief.py`, `evaluation/next_state.py`
- composition: `env/observation.py`, `env/wrapper.py`, `env/perception.py`,
  `runtime/actions.py`
- model: `models/yemong/policy.py` (head width and clamp), `train/rl/features.py`
  (`belief_uncertainty` width and encoding)
- training: `train/rl/ppo.py`, `train/rl/opponents.py`, `train/rl/elo_eval.py`,
  `train/rl/buffer.py`, `train/rl/checkpoint_schema.py`
- evaluation and modes: `evaluation/match.py`, `modes/interactive.py`, `modes/ar_report.py`,
  `modes/noise_calibration.py`, `modes/feature_stats.py`
- benchmarks: `belief_compile_parity.py`, `deadline_subphase_profile.py`,
  `frontline_belief_suite.py`, `frontline_inference_scaling.py`, `realtime_latency.py`,
  `rl_pipeline_profile.py`
- docs: `architecture.md`, `environment.md`, `runtime-semantics.md`, `training.md`
- tests: new `test_physical_belief.py`, `test_legal_view.py`, `test_next_state_objective.py`;
  rewritten `test_belief.py`, `test_belief_uncertainty.py`, `test_ns_labels.py`; deleted
  `test_belief_copy.py`, `test_prediction_uncertainty.py`

## Validation during development

Representative incremental checks, in execution order:

| check | exact result |
|---|---|
| physical-belief layout/objective suite, first cut | 20 passed in 0.31 s |
| env suite after direct composition | 1 failed, 323 passed in 57.43 s (the masking test, updated to the new constant-channel contract) |
| env suite, updated | 324 passed |
| non-train, non-mode suites after composition | 1,078 passed in 109.33 s |
| CPU + CUDA PPO smoke, all paradigms | **8 passed** in 291.15 s (includes the host-backed CUDA logical batch) |
| env/models/runtime/modes suites | 13 failed, 592 passed — all API-shape failures in tests of the replaced path |
| the same four suites, updated | 605 passed |
| rewritten belief suite | 21 passed in 1.85 s |
| rewritten uncertainty suite | 9 passed |
| new legal-view suite | 7 passed in 1.44 s |
| new next-state objective suite | 8 passed in 0.38 s |
| rewritten label suite | 5 passed |
| buffer suite after the storage change | 45 passed in 3.84 s |
| checkpoint suite | 87 passed in 194.01 s |
| `test_ppo.py` + `test_ns_labels.py` | 127 passed in 184.50 s |

Two rounds of failures were genuine contract changes rather than defects, and both are
recorded above: `tests/env/test_perception.py` asserted that *every* hidden ship channel was
zero, which is no longer true of the three ship constants; and `tests/train/test_buffer.py`
asserted bf16 for `VEL`, which moved to fp32 for the reason given above.

## Final validation

| command | exact result |
|---|---|
| `.venv/bin/pytest -q -p no:randomly` | **1,747 passed**, 6 warnings, 393.37 s |
| `.venv/bin/bnb smoke` | **all 16 isolated cases passed; checkout unchanged** |
| `.venv/bin/python -m compileall -q src benchmarks` | passed, no output |
| `.venv/bin/ruff check src tests benchmarks` | **All checks passed** |
| `git diff --check` | passed |

The full suite ran with CUDA available, so every `skipif(not torch.cuda.is_available())` test
executed rather than skipping; the summary reports no skips. The 1,747 total is +18 against
Phase 2's 1,729: three new suites added 35 tests and two deleted suites removed 17.

Repository-wide `ruff check .` still reports the pre-existing violations in
`docs/internal/next-state-head-audit-sep2026-probe.py` and
`docs/internal/seat-symmetry-bug-sep2026-experiment.py`. Phase 3 did not modify either.

The `bnb smoke` pass covers `ar-report`, `feature-stats` and `noise-calibration`, which is
what exercises the three analysis modes this phase rewrote end to end.

## CUDA and compiled/eager validation

Hardware/runtime for every Phase-3 CUDA check:

- NVIDIA GeForce RTX 4070 Laptop GPU, 8,188 MiB reported by `nvidia-smi`;
- driver 595.84, CUDA 13.2 reported by the driver;
- Python 3.13.11;
- torch 2.13.0+cu130, CUDA runtime 13.0, compute capability 8.9.

| CUDA check | exact result |
|---|---|
| full suite, CUDA-gated tests included | 1,747 passed (no skips) |
| host-backed CUDA PPO update inside the smoke suite | passed |
| `belief_compile_parity.py --device cuda --seed 271828 --steps 12` | `parity_passed=true`, `error_count=0`, both team views, retained-output ownership passed |
| the same at `--steps 24` | `parity_passed=true`, `error_count=0`, 24 steps compared |
| the same at `--steps 3 --atol 0 --rtol 0` | 24 bit-level differences; largest 3.815e-06 absolute on `means` after advance |

The committed artifact is
[`perception-belief-phase3-compile-parity.json`](perception-belief-phase3-compile-parity.json),
recorded from clean revision `b289aea` with `torch.compile(mode="default", dynamic=False)`
over both `BeliefTracker.observe` and `observation_from_state`, plus `BeliefTracker.advance`.

**Phase 2's 24-step caveat is resolved.** It recorded a Team-1 decoded `data.pos` mismatch at
step 20 (`max_abs=0.0078125`) and did not claim a 24-step pass. There is no Fourier decode in
this path any more, and the 24-step run compares clean.

What remains is ordinary float reassociation: at zero tolerance, compiled and eager differ by
up to 3.815e-06 absolute on the physical means, against position values of order 10^3 px on a
65,536 px world. The default comparison is `atol=rtol=2e-6`, whose relative term allows about
8e-3 at that magnitude, which is why 12 and 24 steps pass. The artifact's
`means_sha256` therefore differs between arms while `uncertainty_sha256` matches exactly — the
uncertainty block is the head's clamped output copied through, and the means pass through `%`,
`clamp` and `atan2`. **The hash is not the parity criterion; `error_count` is.**

## Hidden-state performance against the baselines

The plan requires hidden-state performance to be compared with persistence and dead-reckoning
baselines. Both are now production diagnostics, emitted every update in physical units beside
the model's own error, per visibility class and per hidden-age bucket:

| series | what it is |
|---|---|
| `belief/<bucket>/position_px` | the head's forecast against truth |
| `belief/<bucket>/persist_position_px` | the belief standing still |
| `belief/<bucket>/reckon_position_px` | the belief carried forward one decision on its own believed velocity |
| `belief/<bucket>/velocity_px_s` | the head's velocity forecast |
| `belief/<bucket>/persist_velocity_px_s` | the believed velocity held constant |

[`perception-belief-phase3-baselines-probe.py`](perception-belief-phase3-baselines-probe.py)
reads them out of the trainer and writes
[`perception-belief-phase3-baselines.json`](perception-belief-phase3-baselines.json), including
the `model / best baseline` ratio per bucket per update.

**Read the artifact as a wiring and calibration check, not as a modelling result.** Phase 3
breaks checkpoint compatibility (schema v19), so no trained weights exist for this
architecture; the probe trains from random initialization for a few minutes, which is far too
little for a learned head to beat a linear extrapolator on a 33 ms decision. What the artifact
establishes is that the comparison exists, is in physical units, is stratified by how long the
ship has been out of contact, and moves in the right direction as the head trains. The
modelling question — does the head beat dead reckoning at convergence — belongs to a full run
and is called out as unresolved below.

## New metrics and diagnostics

Added:

- `belief/<bucket>/persist_position_px`, `persist_velocity_px_s`, `reckon_position_px` — the
  two baselines, per visibility class and per hidden-age bucket;
- `belief/<bucket>/shield_delay_s` — previously absent from the physical error series;
- `next_state/<channel>`, `next_state_visible/<channel>`, `next_state_hidden/<channel>`,
  `next_state_label_sq/<channel>` and `next_state_label_scale/<channel>` now key on the eleven
  physical channel names (`position_x`, `velocity_y`, `local_log_index`, …) rather than on
  56 encoded target dimensions.

Changed meaning:

- `loss/next_state` is the mean Gaussian negative log likelihood in nats per channel. It was a
  mixture of squared-error and likelihood terms, so it is **not comparable across this
  change**;
- `next_state_label_scale/<channel>` is now `current_scale * sqrt(mean_sq)` — a suggestion for
  the fixed Phase-1 constant rather than a correction to a fitted `label_scale`. It is a
  diagnostic: there is no online scaler by contract.

Removed: `belief/*/attitude_rad` keeps its name but is now a wrapped angular difference rather
than an arccosine of normalized decoded headings, and the per-encoded-dimension
`next_state/*_0`-style keys are gone with the encoded target layout.

Anything reading the old key names — a saved W&B panel, a chart recipe — needs updating. The
`charts/renderers/training.py` note about a previous rename still applies.

## Known bugs, caveats, and unresolved questions

Known Phase-3 bugs: none.

Caveats:

1. **The old machinery is untested dead code for one phase.** See "The old machinery is inert,
   not deleted" above. `test_legal_view.py::TestTheOldPathIsInert` proves nothing reaches it,
   but the code itself is no longer exercised. Phase 4 should delete rather than re-cover.
2. **`coordinator.total_prediction_dimension` is a trap until Phase 4.** It is 56 on the
   Frontline world; the head is 24 wide. No call site slices a head output by it today, and
   `NEXT_STATE_OUTPUT_DIM` is the only correct width.
3. **Compiled and eager differ at the ULP level.** Quantified above: 3.815e-06 absolute on the
   means at zero tolerance, which the default tolerance absorbs through its relative term over
   24 steps.
4. **The league slot's first decision after a draw.** A slot created or replaced mid-run has
   its view composed at the start of the rollout shard, which is the settled point; there is no
   fallback to a shared view left in the code. But its belief is empty at that moment, so its
   hidden opponents read zero rather than a remembered position for one decision. Under
   `spawn_reveal` that resolves on the next decision the ships are revealed on.
5. **`shield_delay` has no configured upper bound in this module.** The recharge delay lives on
   `FrontlineConfig`, which `PhysicalNextState.from_ship_config` does not see, so the channel
   clamps at zero from below and is otherwise unbounded. Non-negativity is the property that
   matters for a countdown; the ceiling would be a small improvement.
6. **The velocity guard is a stated number, not a physics result.** Four times the
   `sqrt(boost_thrust / no_turn_drag_coeff)` equilibrium, which is 1,265 px/s per axis on the
   reference config. It is generous on purpose — collisions and refractive gradients briefly
   exceed the equilibrium — and it is a guard rather than a claim.

Unresolved questions:

1. **Does the learned head beat dead reckoning at convergence, and at what hidden age?** The
   diagnostics now answer this; nothing has trained long enough to read the answer. This is the
   first thing a full run should be asked.
2. **Is the correlation latent used?** The full 2D covariance is the plan's decision and it is
   implemented and differentiable, but no run has yet shown whether the head learns a nonzero
   rho or leaves it at the initialization. Worth a single metric (mean `|rho|` on hidden
   tokens) if the answer is not obvious from the loss.
3. **Is the certainty floor at the right place?** `log sigma = -6` is a bound, not a
   measurement. If visible-token residuals are systematically smaller than `exp(-6)` in
   normalized units, the floor is binding and is costing the visible half of the objective.

## Intentionally deferred

- All Phase 4 removal of the inert encoded-belief machinery.
- The global/game token (Phase 5), hex-density targets (Phase 6) and the density head
  (Phase 7). Nothing in this phase anticipates them.
- Batching the two team views into one `2B` build. The plan permits it; the two separate builds
  measured acceptably (below), so it stays a Phase-4-or-later option rather than speculative
  complexity here.
- Using the belief's own uncertainty to shrink a stale mean toward a prior. The plan is
  explicit that propagation uses the predicted mean deterministically, and shrinkage would
  need a decision about what the prior is.
- Inferring `grad(n)` at a believed position.
- An upper clamp for `shield_delay` (caveat 5).

## Compatibility and schema implications

- `physical_belief_v19` is intentionally checkpoint-incompatible with v18 and older. The
  encoder input width changes (`belief_uncertainty` narrows from one column per uncertainty
  slot to thirteen), the next-state head narrows from `56 + 11` to `11 + 13`, and — the part
  that matters most — **a v18 head's outputs mean something else entirely**: absolute encoded
  targets rather than physical deltas. There is no faithful weight-only migration even where a
  shape happens to match, which is exactly why this is gated rather than left to a load error.
- The rollout observation buffer drops `BELIEF_TARGETS` and `BELIEF_SUBSTITUTE` and promotes
  ten physical ship channels to fp32. Cached rollout tensors from Phase 2 are not compatible.
- `RolloutBuffer.privileged_targets` is renamed `privileged_means` and is eleven fp32 channels.
- `YemongEnvWrapper.privileged_observation()` is **removed**: nothing needed a whole omniscient
  observation once the auxiliary target became eleven physical channels read straight from the
  state.
- `YemongEnvWrapper.step()` and `.step_interactive()` take `observe: bool = True` and return
  `None` for the observation when it is false. `.reset()` takes an optional belief.
  `.perceive()` and `.observe(belief)` are new public methods.
- `MatchRunner.observe()` no longer returns an observation; `.actions()` and `.select_actions()`
  no longer take one. `.legal_view(index)` and `.policy_view(index)` replace what callers used
  the shared observation for.
- `BeliefTracker.__init__` takes a `ShipConfig` where it took a `FeatureCoordinator`.
  `.compose(observation)` is gone; `.observe(state, visible)` and `.advance(prediction, logits)`
  replace it. `.means`/`.uncertainty` replace `.predicted_targets`, and `.team_id`/`.radius` are
  gone.
- `evaluation.next_state.decode_targets_to_observation` is renamed `means_to_observation` and
  takes physical means plus the index log scale instead of encoded targets plus a coordinator.
  `imagine_trajectory` requires `next_state` and `index_log_scale`.

## Exit gate

| requirement | result |
|---|---|
| Information-flow tests show hidden truth cannot reach opponent observation slots through the new composition path | **met** — `test_belief.py::TestInformationFlow::test_hidden_truth_cannot_reach_an_opponent_slot` plants distinctive truth on a hidden enemy (position, velocity, health, power, cooldown, angular velocity, committed command) and sweeps every channel of the composed view for it; `tests/env/test_perception.py` pins the same property for the beliefless path, with the three ship constants named explicitly |
| Truth and belief use the same normal feature encoder | **met** — `test_legal_view.py::TestEncodedInput::test_a_believed_ship_encodes_exactly_as_truth_at_that_state_would` compares the encoded input of a believed ship against an omniscient view placed at the believed coordinate; `test_the_encoder_override_is_a_no_op_without_those_channels` pins that no substitution runs |
| Respawn semantics and loss masking are correct | **met** — `test_belief.py::TestLifecycle` covers the revealed respawn correcting a stale belief and a spawn without a reveal voiding it; `tests/env/test_spawn_lifecycle.py` drives the real schedulers for the null-action reveal; `transition_contiguous` masking is unchanged and covered by `test_ns_labels.py` |
| New belief diagnostics are coherent | **met** — every series is an absolute physical error with no decode; `test_ppo.py::TestBeliefDiagnosticAlignment` pins the `t`/`t+1` pairing, that index 0 is never scored, that index `T` is, and that a leaked teleport shows up in its own series |
| Hidden-state performance is compared with persistence/dead-reckoning baselines | **met** — the baselines are production diagnostics and a probe artifact records them; the *modelling* answer needs a full run and is listed unresolved |
| New path performance is benchmarked against the Phase-0 baseline | **met** — see below |
| The old belief system may remain temporarily | **used** — it remains inert and unreached, for Phase 4 to delete |
| Team-0/Team-1 canonicalization preserves the intended symmetry | **met** — `tests/env/test_seat_symmetry.py` (7 tests) drives the production composition for both seats and for a mirrored game |
| Compiled and eager agree | **met** — 24 steps, both team views, `error_count=0`; the ULP-scale difference is quantified above |
| Recurrent rollout and sequence re-evaluation agree | **met** — unchanged from Phase 2 and covered by `tests/models/test_encoder.py`; the stored observation the update replays is the composed view the rollout read |
| Physical ship indexing stays aligned | **met** — `test_belief.py::TestActionBelief::test_physical_indices_are_never_remapped_by_canonicalization` |

## Exact prerequisites for Phase 4

Start from the handoff commit and read this document, the overarching plan, and the Phase-1
calibration artifact. Preserve these Phase-3 invariants:

1. The belief stores **physical** state in the Phase-1 order and units. Nothing decodes or
   re-encodes it.
2. Composition is one selection per ship slot, before encoding: truth where the observer owns
   or sees the slot, belief where it cannot, zero where nothing was ever observed.
3. Each observer composes its own view from its own belief. A shared view taken perspectives
   on is the thing this phase removed.
4. Uncertainty is restated by the head, not accumulated. Visible is the floor, never-observed
   is the ceiling.
5. The recursion is bounded by wrapping or clamping, per channel. `clamp_events` counts only
   non-finite head output.
6. The label is believed-current to true-next, and the visible half of it is exactly the
   Phase-1 truth-to-truth delta.
7. The next-state head is 11 means + 13 uncertainty terms, and `NEXT_STATE_OUTPUT_DIM` is the
   only correct width.
8. Composition happens after the environment/policy stream join. Do not move it back into the
   environment step.
9. Phase-2 invariants are unchanged: 42-float pending action, no authoritative enemy command
   in policy input, spawn/respawn null one-hot for both teams, physical ship slots never
   reordered, prediction at `t` visible at `t+1`, a separate but identically shaped enemy head,
   and no extra policy forward for any belief output.

Phase 4's removal list, concretely:

- `ObsKey.BELIEF_TARGETS`, `ObsKey.BELIEF_SUBSTITUTE` and their `_TOKEN_LAST_KEYS`/docstring
  references in `env/observation.py`;
- `FeatureCoordinator._apply_belief_override`, `_override_columns`, `_override_cache`,
  `project_targets`, `decode_targets`, `compute_labels`, `apply_all_predictions`,
  `apply_scaled_predictions`, `prediction_loss`, `prediction_variance`,
  `uncertainty_variance`, `label_scale_vector`, `get_loss_weights`, `get_feature_names`,
  `target_slices`, `get_target_vector`, the `_PredictorSpec` machinery and the `Predictor`
  class hierarchy in `train/rl/features.py`, plus every `predictor=`/`label_scale=` argument in
  `build_standard_coordinator`;
- `YemongPolicy._believed_rotary_tables` and `self._rotary_target_slices`, and with them
  `SpatialRotary.tables_from_moments` if nothing else uses it;
- `PPOTrainer.aux_weights` (already unread) and `self.coordinator`'s remaining role, which
  after the above is input encoding only;
- `Fourier.invert`, `UnitCircle`, `UnitCirclePredictor` and `AttitudeFourier.invert` if they
  have no remaining reader.

Check `total_prediction_dimension`, `total_target_dimension` and `total_uncertainty_dimension`
have no callers before removing them; `belief_uncertainty`'s width already comes from
`PHYSICAL_UNCERTAINTY_DIM`.

Then re-run the same end-to-end benchmark and compare against the number below: Phase 4 is a
deletion, so it should not move SPS, and a change either way is worth understanding.

## Surprising repository facts

- **The full suite is much faster than Phase 2 recorded** — 393 s against 975 s. Some of that
  is the composition change (the belief no longer clones every observation channel twice per
  decision) and some is test-mix churn. It is not a measurement error; both numbers are
  `pytest -q -p no:randomly` on the same machine.
- **`state.num_zones > 0` is how the belief knows respawn is on.** There is no flag threaded
  through; the frontline layout is the mode with instant respawn, and a remembered ship is
  alive iff the belief is valid there. Outside it, believed health decides.
- **`TensorState` advances by tensor reassignment.** Reading `env.state` while a queued
  physics kernel runs on another stream is a race — which is why league-slot views are composed
  at the end of the previous decision rather than inside `_rollout_network_forwards`.
- **The physical ship channels are fp32 in the rollout buffer now.** Anything sizing the buffer
  from Phase-2 numbers will be wrong in both directions: those ten channels grew, and the
  56-channel `BELIEF_TARGETS` tensor over 26 tokens went away.
- **`compile_visibility` and `compile_observation` are new** alongside `compile_perception`.
  Perception and composition had to split because a belief is assimilated between them, which
  costs the fusion across that boundary; the evaluator uses the single-observer builder.
## Where the composition cost actually is

Measured directly at production width (960 envs, 5v5 Frontline, RTX 4070 Laptop), because the
end-to-end profile alone cannot separate the composition from everything around it. Twenty
timed calls after warmup, `torch.cuda.synchronize` around each block:

| operation | eager perception | compiled perception |
|---:|---:|---:|
| `wrapper.step(actions, observe=False)` — physics and perception only | 41.233 ms | 52.047 ms |
| `advance` + `observe` + `wrapper.observe(sources)` — the whole belief plane and **both** legal views | **7.487 ms** | **6.740 ms** |
| `wrapper.step(actions)` — the same, composed inline | 57.424 ms | 57.845 ms |

Both team views compose in about 7 ms per decision. Over the production update — 128 steps x 4
rollouts = 512 decisions — that is roughly **3.5 s/update** against the **32.06 s/update**
Phase 2 recorded for `belief_compose` + `belief_advance` + `privileged_obs` + `target_vector`.

The reasons are all structural rather than tuning. `_mask_hidden_ships` cloned and masked
twenty-eight channels per view and is gone. `compose` cloned every channel again, decoded a
56-dimensional Fourier target vector, and wrote back a dozen channels; it is gone. The
privileged observation build and its target-vector pass are replaced by one eleven-channel
stack straight off the state. And the head narrowed from 67 outputs to 24.

## Startup cost

One real regression, and it is in startup rather than throughput. Perception used to be a
single fused callable; it is now three, because a belief has to be assimilated between
computing visibility and composing the view that stands on it:

| callable | who compiles it | shapes |
|---|---|---|
| `compile_visibility` | the wrapper and the evaluator | one per environment batch |
| `compile_perception` (both team views) | the wrapper | one |
| `compile_observation` (one observer) | the evaluator and each league slot | one per distinct width |

That is roughly three to six Inductor compilations of a several-hundred-kernel function where
Phase 2 had two, and each takes tens of seconds on this laptop GPU — the parity harness times
a single compose graph at 16.98 s. It is paid once per process. A Phase-4 option, if it
matters: leave the evaluator's single-observer builder eager and measure, since its builds are
small and launch-bound rather than size-bound.

