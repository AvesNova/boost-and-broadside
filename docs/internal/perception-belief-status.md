# Perception and belief — status

The single handoff document for the plan in
`perception-belief-overarching-plan.md`. Updated at the end of every phase.
Last updated September 28, 2026, after Phase 6 (the Phase 5 and 6 GPU benchmarks
are deferred until after Phase 7; see Deferred GPU benchmarks). Phase 3 ended at
`ff32f9f` on `fix/seat-symmetry-pending-action`; Phases 4–8 all go on
`feat/perception-belief-phases-4-8`.

Everything a new phase owner needs is here, the spec, and the code. The
per-phase handoffs that preceded this document are in `archive/` and are
history, not required reading.

## Done

**Phase 0 — lifecycle and measurement.** `TensorEnv.step` now maintains the
spawn-reveal latch the wrapper already maintained, so the reveal lasts one
decision in every stepping path (before this, every non-training path was
fog-free). Null pending action at spawn/respawn is pinned end to end
(`tests/env/test_spawn_lifecycle.py`). Seat symmetry is tested as a mirrored
game over every observation channel and the policy outputs
(`tests/env/test_seat_symmetry.py`). The belief diagnostic's alignment was
verified, not fixed: the "timing discrepancy" was respawn teleports leaking
into a mean. Baseline numbers are in the table below.

**Phase 1 — delta calibration.** `train/rl/physical_deltas.py` defines the
eleven physical deltas and their fixed scales, measured over 5.75M scripted
5v5 Frontline transitions with respawn destinations excluded
(`perception-belief-phase1-calibration.json`,
`benchmarks/physical_delta_calibration.py`). Position 2.5 px, velocity 4.0
px/s, attitude 0.1 rad; sparse channels use their event magnitude (health 10,
cooldown 0.1 s, shield delay 5 s) rather than a zero-dominated RMS.

**Phase 2 — joint pending action and enemy head.** `previous_action` is 42
floats per token. Allies and spawn-revealed ships are exact one-hots; ordinary
enemy slots are zero in the raw view and filled only from the belief's stored
softmax of the enemy head's previous prediction. The enemy head
(`Linear(D,2D) → RMSNorm → GELU → Linear(2D,42)`) is disjoint from but
identical in shape to the actor head and is returned by the same forward.
Trained with cross-entropy masked to alive, committed enemy decisions. Metrics:
`loss/enemy_action`, realized probability, entropy, Brier, top-1, persistence
and uniform baselines, trunk gradient norm and cosine. Schema v18.

**Phase 3 — physical belief plane.** `BeliefTracker` (`train/rl/belief.py`)
stores eleven physical means, thirteen uncertainty terms, the action belief and
an age per ship per observer. `observation_from_state` takes a
`ShipBeliefSource` and selects each slot's physical state once
(`_ShipChannels.select`); `_mask_hidden_ships` and `BeliefTracker.compose` are
gone. `NextStateHead` emits 11 + 13 (`NEXT_STATE_OUTPUT_DIM`);
`PhysicalNextState` (`train/rl/physical_belief.py`) owns the scales, bounds,
label and Gaussian NLL. The rollout composes after the env/net stream join:
`advance → reset → observe → wrapper.observe → write_observation`. Every
evaluation agent, league slot and interactive side composes its own view
through `legal_policy_view`. `RolloutBuffer.privileged_means` is eleven fp32
channels; the ten physical ship channels are fp32 in the buffer. Persistence
and dead-reckoning baselines are production diagnostics
(`belief/<bucket>/persist_*`, `reckon_position_px`). Schema v19.

**Phase 4 — Delete the legacy belief machinery.** Removed the whole predictor/target-space
path from `FeatureCoordinator` (`train/rl/features.py`): the `Predictor` hierarchy,
`target_encoder`/`predictor=`/`label_scale=` on `Feature`, and every method that built,
scaled or decoded a target vector. `Feature` now takes only an accessor and an input
encoder. Removed `ObsKey.BELIEF_TARGETS`/`BELIEF_SUBSTITUTE`, `YemongPolicy`'s believed-rotary
override, `SpatialRotary.tables_from_moments`, and `UnitCircle`/every `Transform.invert` (all
unread once the target path was gone). `physical_means_from_state` dropped its unused
`ship_config` parameter. `BELIEF_UNCERTAINTY`/`TIME_SINCE_OBSERVATION` are now forced to the
certainty floor/zero at the composition site for truth-sourced ship slots
(`env/observation.py`), rather than relying on `BeliefTracker.observe`'s invariant. No head
width, belief layout or composition semantics changed; `docs/architecture.md` needed only a
stale-paragraph trim, since it already described the post-Phase-3 shape.

**Phase 5 — Promote the global/game token.** Every environment, combat included, now
presents one global token directly after the ships (`env/observation.py`, `NUM_GLOBAL_TOKENS`,
`ObjectType.GLOBAL`, `FeatureScope.GLOBAL`). It sits at the map centre and carries a
`GameMode` one-hot, the clock and the front. `ModelConfig.global_token` (on by default, and
set explicitly in `defaults.py`) makes ships plus that token the query/recurrent set in
`YemongPolicy`. Fields and zones stay K/V-only, every head reads `[:N]`, and TeamPMA is
unchanged. With the switch off, the token becomes an ordinary K/V map object, so Phase 8's
variant A drops the promotion but keeps the information. `num_recurrent_tokens` (N+G) is the
only hidden-state stride: league slots size from their own policy, beliefs from the ship
count, and the buffer and microbatch splits read it off the stored tensor. The policy raises
on a mismatched hidden state. Gate tests: `tests/models/test_global_token.py`. Schema v20.

**Phase 6 — Hex density target.** `train/rl/hex_density.py` builds the privileged
ally/enemy density target: `HEX_DENSITY_RINGS` hexagonal rings (331 cells), stored in units
of the playable radius and scaled per env around `map_center`, so larger maps are a zoom of
the same cells. Outer-ring corners sit on the boundary. Cell order (ring by ring from the
centre, each ring starting on +x and walking counter-clockwise) is part of the target's
meaning; do not change it once a head trains on it. The kernel, radius and `log1p` are
`local_presence`'s, over every living ship, visible or not, with minimum-image distance.
`HexDensityTarget(state, observer_team)` returns `(B, 2C)`, ally cells then enemy cells,
optionally compiled. No loss is attached and nothing calls it in training yet. Tests:
`tests/train/test_hex_density.py`; micro-benchmark: `benchmarks/hex_density_target.py`.

## Invariants to preserve

1. `previous_action` is exactly 42 floats; raw enemy slots are zero; only the
   belief writes predicted values; spawn/respawn is the null one-hot for both
   teams; a prediction made at decision `t` is the belief seen at `t+1`.
2. Canonicalization relabels perspective and never reorders physical ship
   slots.
3. The enemy head is separate from, and shaped like, the actor head. No belief
   output costs an extra policy forward.
4. The belief stores physical state in `PHYSICAL_MEAN_NAMES` order and units.
   Nothing decodes or re-encodes it.
5. Composition is one selection per slot before encoding: truth where the
   observer owns or sees it, belief where it cannot, zero where never seen.
6. Each observer composes its own view from its own belief. No shared view.
7. Uncertainty is restated by the head, never accumulated. Visible = floor
   (`log σ = −6`), never-seen = ceiling (`+6`).
8. The recursion is bounded: position and attitude wrap, everything else
   clamps to its physical range. `clamp_events` counts only non-finite output.
9. The label is `believed[t] → truth[t+1]`, normalized by the fixed scales.
   `NEXT_STATE_OUTPUT_DIM` is the only correct head width.
10. Composition happens after the stream join. Do not move it into the
    environment step.
11. The live policy stays on Team 0; seat symmetry is a test, not a
    randomization.
12. The token axis is ships, global token, fields, zones, so the query/recurrent
    set is a prefix of it. Size hidden state from `num_recurrent_tokens`, never
    from the ship count. (New in Phase 5: the global token made the two differ.)

## Numbers

Production `rl` profile (5v5 Frontline, 960 envs × 128 steps × 4 rollouts),
RTX 4070 Laptop 8 GB, torch 2.13.0+cu130. Command:

```
uv run --no-sync python benchmarks/rl_pipeline_profile.py --profile rl \
    --timing wall --updates 2 --warmup 1 --no-checkpoint
```

| after | SPS | s/update | peak alloc MiB | peak reserved MiB | artifact |
|---|---:|---:|---:|---:|---|
| Phase 0 | 2,810 | 175.0 | 3,461 | 4,656 | `perception-belief-phase0-baseline.json` |
| Phase 2 | 2,087 | 235.5 | 3,766 | 5,052 | `perception-belief-phase2-benchmark.json` |
| Phase 3 | 3,190 | 154.1 | 2,989 | 3,678 | `perception-belief-phase3-benchmark.json` |

Two-update measurements have a few percent of noise. Compare SPS across
`--timing wall` runs only. Compile parity artifacts:
`perception-belief-phase{2,3}-compile-parity.json`; `error_count` is the
criterion, not the hash.

Phase 6 density target versus one rollout step (primary + aux + evaluator), 960 envs,
5v5, `uv run --no-sync python benchmarks/hex_density_target.py` (GPU, compiled, by default):

| device | compile | target ms | step ms | fraction | artifact |
|---|---|---:|---:|---:|---|
| CPU (4 cores) | none | 10.3 | 3,670 | 0.28% | `perception-belief-phase6-density-target.json` |

Untrained-head baseline probe (Phase 3, 327k steps,
`perception-belief-phase3-baselines.json`): on visible ships, dead reckoning
0.07 px, persistence 2.5 px, model 2.5 px. The model equals persistence because
its last layer initializes near zero. See open question 1.

## Known caveats

- `shield_delay` has no upper clamp in `PhysicalNextState` (the recharge delay
  lives on `FrontlineConfig`, which the spec does not see). Not fixed in Phase
  4: it needs a new parameter threaded through `from_ship_config` and every
  caller, not a one-line change.
- A metric block in `ppo.py::_compute_minibatch_loss` can raise and kill an
  update (one such crash was fixed in `33a4a0e`). It is not defended.
- Compile startup grew: perception is three Inductor graphs
  (`compile_visibility`, `compile_perception`, `compile_observation`) instead
  of one, tens of seconds each on the laptop GPU, once per process.
- A league slot created mid-run has an empty belief for one decision; hidden
  opponents read zero until the next reveal.
- `state.num_zones > 0` is how `BeliefTracker.source` knows respawn is on.
- Checkpoints older than schema v20 cannot be loaded; there is no weight
  migration. Evaluation numbers taken through `TensorEnv` before Phase 0's
  latch fix were measured without fog.
- Pre-existing `ruff` violations remain in the two `docs/internal/*-probe.py`
  and `*-experiment.py` scripts. Leave them.
- `map_read_mode="full_attention"` still sends fields and zones through the trunk
  as non-recurrent queries. Only `kv_memory` (production) keeps them K/V-only.
- In combat the global token's position and radius are zero; the unbounded arena
  has no map centre. For the same reason every hex density cell collapses onto the
  origin in combat; the target is only meaningful on Frontline.
- `benchmarks/rl_kernel_profile.py` still calls `_prepare_league_slots` with an argument
  and skips the aux scales; it crashes on the current trainer.
- On a CPU-only cloud container, the `capture` smoke case and three default-suite
  tests fail for environmental reasons: no `ffmpeg`, no CUDA driver
  (`test_a_cuda_graph_mode_...`), and a CPU bf16 simplex check
  (`test_bf16_logits_do_not_break_the_marginal_entropies`).

## Open questions

1. Does the learned head beat dead reckoning at convergence, and at what
   hidden age? Only a training run answers it. A physics prior is ruled out by
   design (the head is meant to learn the dynamics); the baseline is the bar,
   not a shortcut.
2. Is the correlation latent used at all? One metric (mean `|ρ|` on hidden
   tokens) would tell.
3. Is the certainty floor at −6 binding on visible tokens?

## Deferred by design

Batching the two team views into one `2B` build; shrinking a stale mean toward
a prior; inferring `grad(n)` at a believed position; a Frontline mirrored-game
seat-symmetry test (the map itself carries a side); a self-relative team
feature (ego/ally/enemy) that would make seat identity unrepresentable.

## Repository facts worth knowing

- `pytest` defaults to `-m "not slow"`. The default suite is roughly 1,750
  tests and takes about seven minutes on the laptop GPU. Do not run it more
  than the phase prompt says.
- `bnb smoke` runs 16 isolated end-to-end cases and is the check for the
  analysis modes (`ar-report`, `feature-stats`, `noise-calibration`).
- `TensorState` advances by tensor reassignment; reading `env.state` while a
  queued kernel runs on another stream is a race. That is why league views are
  composed at the end of the previous decision.
- A ship dies on the tick it takes damage while already at zero health, not the
  tick it reaches zero.
- Belief-cell diagnostics need thousands of decisions of warm-up before any
  enemy is in sight on Frontline.

## Deferred GPU benchmarks

Owner's decision: the GPU benchmarks for Phases 5 and 6 are deferred until Phase 7 is
done, and the owner runs them on the laptop. Phase 7 does not wait for them.

- Phase 5: its gate benchmark has not been run (no GPU in the cloud container;
  `benchmarks/rl_pipeline_profile.py` hard-codes CUDA). Run the command in Numbers once
  and add a row. If SPS is more than 10% below Phase 3's 3,190, look for the cause
  before Phase 8.
- Phase 6: the micro-benchmark was run on CPU only (eager, 0.28% against the 5% bar),
  and the compiled-versus-eager test is CUDA-only and was skipped. Run
  `benchmarks/hex_density_target.py` once with its defaults and add a GPU row, and run
  `tests/train/test_hex_density.py` on the GPU.

## Remaining

Phase 7, then the deferred GPU benchmarks above, then Phase 8 as written in the plan.
