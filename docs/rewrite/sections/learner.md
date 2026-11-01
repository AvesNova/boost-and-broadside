# Section 6: PPO and labels (the learner)

Status: decisions settled, 2026-11-01. Spec: `learner-spec.md`.

## Decided (author, 2026-11-01)

- **Scope:**
  - the PPO learner over Yemong, with v1's math (per-level GAE, the categorical critic,
    the return scale, the auxiliary losses);
  - the labels;
  - streaming the rollout through CPU RAM;
  - several games in one update.

  BC (section 7) and the league (section 8) stay separate.
- **Several games in one batch, in the same tensors (not v1's "scales").**
  - Team deathmatch and Frontline, fogged and clear, the squad and battle maps: all in
    one batch, one shape (same `N`, `M`, `Z`).
  - Modes without zones carry absent zone slots. Absent tokens are neutral, with a
    `present` input of 0 and no attention mask, so the fused attention kernel stays
    usable.
  - Simple first; a measurement can revisit it.
- **No coefficient calibration here.** Every training run starts warm from BC (section
  7). BC must converge the policy, the auxiliary heads and especially the value heads.
  Here, v1's coefficients are the starting values.
- **Precision is decided by measurement, not argument.** This section delivers the
  matrix:
  - configurations: float32, TF32 and bf16, each with the rollout's log-probabilities or
    with old log-probabilities recomputed by `unroll` at the start of the update;
  - measures: rollout and update time, step/unroll ratio drift, clip fraction, KL, and
    short seeded learning comparisons for stability.

  In any configuration, recurrent state, norm statistics and accumulators stay float32:
  low precision drops small additions, a bug v1 had.
- **Shot labels at every age** of each flight, generated on the GPU per rollout shard at
  update time from the stored launch states. The trajectory head trains on all ages at
  once.
- **The rollout streams through pinned CPU RAM** (v1 did this):
  - an asynchronous copy each tick from a GPU staging ring;
  - micro-batches prefetched back during the update;
  - a measured budget of under 5% of throughput;
  - replay compaction as well (the means duplicate the observation for visible ships).
- **The toy learner** is the fixed-size relational MLP (relative heading, distance,
  bearing) at 2v2 team deathmatch, which must tell ally from enemy. It must learn to beat
  the chaser.
- **Shot re-simulation reuses the game's own launch, flight and hit code.** The test
  checks the composition: capture timing, the field layout, which rows a flight tests
  against. A re-simulated shot equals the bullet the game fires from that state.
- **Acceptance (proposed, agreed):**
  - the toy learner beats the chaser;
  - re-simulated shots match the game;
  - each loss term matches an independent computation;
  - micro-batch accumulation equals the unsplit minibatch;
  - the throughput of the 5v5 training cells.

## Later (author's idea, 2026-11-01)

If the game is fast enough relative to the update, roll out to the episode's end, so
value targets need no bootstrapping (Monte Carlo returns) and the critic gets exact
targets.

## Settled (author, 2026-11-01)

- **The battle map keeps 10 fields, zoomed** (option b). The point is long-range
  geometry, and the zero-shot claim covers fields too: train at 10, evaluate at 50v50's
  40.
- **Mixed batches are their own section, 3c** (`mixed-spec.md`):
  - the mode chosen per pool, by static slices of the batch, so no mode pays for
    another's rules;
  - the arena radius per environment;
  - absent zones as a learned vector, with the global token's rotation (no rotation
    would leak the per-team position offset).

  Whatever is fastest wins.
