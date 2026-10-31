# Section 5 (Yemong): notes

Branch `section/yemong`. The spec (`yemong-spec.md`) is implemented in `bnb.models.yemong`,
with a controller in `bnb.controllers.yemong`. `uv run --no-sync pytest` runs the Yemong
equivalence tests (not skipped): 42 of 43 pass. The exception is `test_local_presence`,
which is off by about 1e-7, at float32 level, against the spec's formula in float64.
See request 1. Ruff is clean.

## Design

### Layout

| Module | Job | Lines |
|---|---|---:|
| `model.py` | `YemongConfig`; the `Carry`, `Replay`, `Decision` and `Unrolled` pytrees; `Yemong.step`, `Yemong.unroll`, `shot_trajectory` | 327 |
| `inputs.py` | `compose`: ship, global and field tokens from a view and its belief; `truth` (observed state in belief order) | 117 |
| `belief.py` | `Belief`; `observe` (sight) and `advance` (store the forecast, count non-finite entries) | 69 |
| `state_code.py` | `ShipStateCode`: means and spreads to the 490-wide code and back; the 3.2 constants | 156 |
| `codes.py` | the 3.3 primitives: circular, position, scalar (with moment matching), velocity, grouped softmax, baseline floor | 347 |
| `network.py` | `Encoder` (kinds, shared projection, map memory), `Trunk` (`step`, `sequence`) | 101 |
| `spatial.py`, `temporal.py`, `layers.py` | the sublayers: attention over queries and map memory; causal convolution, RG-LRU and the scan; RMSNorm, rotary, gated MLP | 226 |
| `heads.py` | `MLPHead`, `NextStateHead`, `critic_values`, `ShotTrajectoryHead` | 91 |
| `policy.py` | joint-action Gumbel-max sampling, log-probability and entropy | 24 |
| `critic.py` | value bins, two-hot, expectation, outcome classes and lambda targets, `ReturnScale` | 93 |
| `density.py`, `shots.py`, `features.py` | the hex density target and Poisson terms; shot codes and age features; symlog, respawn features, local presence | 221 |
| `controllers/yemong.py` | `YemongController`: sample, set the previous action, fill `Outputs` | 39 |

The adapter `tests/reference/yemong/yemong_arm.py` has 263 lines. Each method
builds a module from the fixture weights or calls a function. The translations are:
- the RG-LRU's flat weight names to the block's nested `recurrence.*`;
- `global.*` to `global_token.*`, because `global` is a Python keyword and FX codegen
  failed on it during ONNX export;
- `(B, T)` to time-first;
- `(blocks, B)` to batch-first;
- `done` to starts;
- `None` action logits to zeros, whose softmax is uniform;
- age × tick for `time_since_seen`.

### The carry

`Carry` is a pytree with leading `(R,)`:
- `state (R, blocks, Q, D)` and `convolution_buffer (R, blocks, Q, 3, D)`;
- `belief`: `valid`, `age` (ticks), `means (R, N, 12)`, `spreads (R, N, 15)`,
  `action (R, N, 42)`;
- `previous_action (R, S)` int32 joint;
- `nonfinite (R,)` int64, the running count of spec 5.2 step 5.

`Yemong.initial_carry(views, S, N, device)` builds it in the model's dtype. The runtime
stores it, `copy_`s into it and captures it like any carry.

### step

`step(carry, obs, reset)` follows spec 5.2:
1. Clear every reset view (a `where` per leaf). The counter is not cleared.
2. `belief.observe`.
3. Build the `Replay`.
4. `compose`, then the encoder, then `Trunk.step`, then the decision heads (policy, enemy
   action, next state, values, outcome).
5. `decode_logits`, then `belief.advance`.

It returns a `Decision`, whose carry still holds the old previous action. The controller
samples the joint action with Gumbel-max from the default generator, writes the sample into
`previous_action`, and returns `Outputs(logp, entropy, value = values.sum(-1), values,
outcome, replay)`.

### unroll

`unroll(carry0, obs, replay, reset)` works on `(L, R, ...)`:
- it clears `carry0` where `reset[0]` (as `step` would);
- it composes tokens from `obs` and `replay` for all `L·R` views at once;
- the encoder and the spatial sublayers run batched;
- each temporal sublayer runs `TemporalBlock.sequence` over time, with `reset[1:]` as
  episode starts. A start zeroes the decay and cuts the convolution taps. The scan is
  Hillis-Steele: log2 L rounds of elementwise work, with no loop over time;
- it returns every `step` output, plus value logits, density log-rates (from the global
  token), shot-outcome logits and shot latents.

`Yemong.shot_trajectory(latents, ages)` queries the trajectory head at any number of
ages in parallel.

### Compile and capture assumptions

- No `torch.compile`, `torch.cuda` calls or global settings in section code. The runtime
  compiles `step` inside its act stage (`fullgraph=True, dynamic=False`) and captures it.
- No host syncs and no data-dependent shapes in `step` or `unroll`.
- Per-level constants are built on the device: no `torch.tensor(list)` and no list
  indexing. Eager CUDA-graph capture (`compile=False` on a GPU) works too. Two cases
  failed there before this fix: `critic_values` indexing with a list, and the circular
  decode's level tables.
- Shapes are static per run. `S` comes from the carry's `previous_action` and `N` from
  the observation, so one set of weights runs at any size.
- Randomness: the Gumbel noise is `torch.rand_like` on the default generator. A captured
  graph draws fresh noise on every replay (tested).
- ONNX: the single step exports with the dynamo exporter and matches eager in ONNX
  Runtime (`tests/contract/test_yemong_export.py`). Two changes made export work:
  - clamp bounds are floats (the decomposition rejected an int bound);
  - the boolean `where` became `&` (ONNX Runtime's CPU `Where` has no boolean kernel).

## Decisions and deviations

1. **Decoded positions are centred** (`state_code.decode`). The spec decodes the position
   code to `mean mod P`, in `[0, P)`. A hidden ship at x = -100 m would then be believed
   at 26 144 m: wrong for presence and for the rotary phases, and outside the arena. I
   map the mean into `[-P/2, P/2)`. This agrees with the spec modulo P, and the arena
   (at most about 13 km across in any preset) fits inside. Request 2.
2. **The circular code skips the `mod P`** in its coordinate (`codes.circular_encode`).
   `(m mod P)/p_l·c` and `m/p_l·c` differ by a multiple of `c^(l+1)`, and only the
   fraction and the floor mod `c` are used, so the math is the same. In float32,
   `x mod 26 244` rounds to 2 mm. That is 0.15% of a 4 m level's colour, and it made the
   compiled and eager steps disagree by 1e-3 in the code and 2% in the next-state
   logits. Without the mod they agree to 1e-5. The goldens pass at 1e-9 either way.
3. **The replay stores the action belief for enemy rows only** (`Replay.enemy_action`,
   `(R, N - S, 42)`). Own rows' pending action is the previous action, so their action
   belief is never read. This halves the largest part of the replay: at the 40-ship cell,
   the replay went from 2.0 GB to 1.4 GB, and that cell then fit in 8 GB.
4. **The replay includes `valid`.** Spec 5.2 lists means, spreads, action belief and
   previous action, but step 3 also reads the valid flag (local presence counts valid
   ships). Since the replay is defined as what step 3 read, `valid` belongs in it.
5. **`time_since_seen` comes from the observation's column**, not from the belief's age.
   Spec 4 composes the token from the view, and the replay (5.2) does not include the age.
   The two agree except for ships not seen since the carry was created. The belief still
   carries `age` (spec 5.1). No input reads it; it is kept as the spec's carry. Request 3.
6. **Hidden ships' gradient columns are masked by `visible`.** The observation already
   zeroes them. Acceptance item 3 asks that perturbing them changes nothing, so the model
   never reads them.
7. **"With fog off, every ship's spread is zero" is tested for living ships.** Without
   fog the game shows every *living* ship. A dead enemy is hidden even then, and its row
   keeps a forecast with a spread. Request 4.
8. **Initialisation details the spec leaves open:**
   - code columns use `N(0, 1/G)` read as variance `1/G`;
   - the trajectory head's age projection (`W_age`, no bias) is orthogonal with gain √2,
     like the latent it adds to;
   - the convolution is torch's `Conv1d` default for a depthwise kernel of 4.
9. **Reward levels.** The rewards section is not written, so `YemongConfig.levels`
   defaults to `("reward", "outcome")`, with `outcome_level = "outcome"`. Only the count
   and the outcome's position matter to the model. At least one per-ship level is
   required.
10. **`values` are the bin expectations as they are.** The spec says "raw units". Any
    return-scale normalisation is the learner's (section 6). Request 5.
11. **bf16 autocast is an option, off by default** (`YemongConfig.bfloat16_matmuls`). It
    covers the encoder, the trunk and the heads. The residual stream, the RG-LRU state,
    the codes, the belief and the critic expectations stay float32. The measurements
    below show it faster, but PPO's ratio needs `step` and `unroll` to agree, which
    float32 gives (1e-5) and bf16 does not (about 1e-2 in logits). So the default stays
    float32, and the learner section decides.
12. **Rotary frequencies are as written:** `(2π/64 m)·2^k`, `k = 0..7`. The periods are
    64 m down to 0.5 m. Request 6.
13. **The density target and the return scale are learner-side and eager.** `hex_counts`
    builds its cell lookup on the host, and `ReturnScale.update` uses a boolean mask. Both
    are spec 8-9 targets, which the learner section owns. Neither runs in `step`.

## Measurements

### Spec 3.2: velocity reach and nose rate (chaser vs chaser, one 60 s episode)

| preset, fog | ship-steps | share > 319 m/s | share \|nose rate\| > 8 | max speed (m/s) | max \|nose rate\| | gun reach p50 (m) |
|---|---:|---:|---:|---:|---:|---:|
| duel, on | 901k | 0 | 0 | 204 | 4.9 | 197 |
| duel, off | 917k | 0 | 0 | 226 | 5.0 | 197 |
| squad, on | 2.12M | 0 | 0 | 309 | 7.0 | 202 |
| squad, off | 2.14M | 0 | 0 | 245 | 5.5 | 199 |
| battle, on | 2.87M | 5.1e-5 | 0 | 337 | 7.4 | 213 |
| battle, off | 2.79M | 0 | 0 | 242 | 5.7 | 206 |

Both shares are below 1e-3 everywhere. The largest world speed seen was 337 m/s, in a
fogged battle; the reach is 319 m/s. The largest |nose rate| was 7.4 rad/s. The ranges
are kept.

### Spec 4: the presence radius

The gun's reach at fire time: the distance a bullet travels in its 0.4 s life, from its
launch proper speed (ship velocity plus muzzle), `ln(1 + k_b w T)/k_b`. The median is
197-213 m by preset (the table above), so 0.8 of the reach is 158-171 m. From rest the
reach is 186 m, and 0.8 of that is 149 m. `r = 150 m` is about 0.75 of the typical
reach, close to v1's 0.78. I kept it.

### Speed (acceptance item 7)

Bench sizes (seed 20261031), RTX 4070 Laptop, float32 (TF32 off), Yemong as learner and
league slot, chaser and random pools as in `TrainConfig`; min of 3 rollouts after 2 warm-up.

| cell | precision | rollout env-ticks/s | ms per tick | peak VRAM (GB) | MLP skeleton env-ticks/s |
|---|---|---:|---:|---:|---:|
| 114 ships, 22 fields, R 9654 m, 360 envs (540 learner views) | fp32 | 4,028 | 89.4 | 6.63 | 111,834 |
| 40 ships, 8 fields, R 6146 m, 1024 envs (1536 learner views) | fp32 | 11,302 | 90.6 | 6.66 | 438,961 |

Not finished when the run was cut short: the 26-ship cell, the bf16 rollout cells, and
the `unroll` forward and backward cells (L = 128; squad R = 32, 40 ships R = 8). The
script is `/tmp/bnb-scratch-yemong/bench_yemong.py` (CELLS env var lists them).

Step profile (40-ship size, 1792 views, compiled): 88 ms in float32, 52 ms with
`bfloat16_matmuls` (1.7x). In float32, sgemm takes 36 ms at about 54% of peak, the gated
MLP's gelu-mul (memory-bound) takes 11.5 ms, and the codes, norms, rotary and carry copies
take the rest. Fusing gate and up into one matrix product measured no gain (88.4 vs
87.8 ms) and was reverted. The act stage dominates the rollout; the game is about 1-2 ms
per tick.

## Runtime and contract changes

- **`contracts/controller.py`:** `Outputs` gains `values (R, S, K)`, `outcome (R, 4)` and
  `replay` (an opaque pytree), each defaulting to None. Its docstring says `value` is the
  sum of `values`.
- **`contracts/trajectory.py`:** `Trajectory` gains `values`, `outcome`, `replay` and
  `initial_carry`, each defaulting to None.
- **`runtime/rollout.py`** (mechanical):
  - `_learner_buffers` runs one probe decision of the learner's controller on the first
    view at construction. It allocates `(L, ...)` buffers shaped like that decision's
    `values`, `outcome` and `replay`, and a clone of the learner's carry for
    `initial_carry`.
  - The act stage `index_copy_`s the three into the buffers each tick, as it does
    `logp`.
  - `run()` copies the learner's carry into `initial_carry` before the first tick.
  - The trajectory returns all four, or None where the controller gives nothing. The
    MLP skeleton gives nothing, so it is unchanged.
  - The controller carries are created before the buffers, so the probe can use the
    learner's.
- The stateful controller needed no other runtime change. The act stage already
  `_assign`s a returned carry, and `done` is the reset flag.
- `docs/rewrite/contracts.md` (the current-schema section) does not list the new fields.
  I may not edit it. Request 7.

## Requests and open questions

1. **`features_presence` golden.** It disagrees with spec 4's formula, evaluated in
   float64, by up to 9.8e-8 absolute. Every other golden matches to 1e-9. The test
   tolerance is 1e-9, so this one test fails.
   - What I measured: each pair's implied weight deviates by 1e-8 to 1e-6 relative, with
     mixed signs. That is the size of float32 rounding in the exponent. A float32
     constant would give a consistent sign; this is rounding noise.
   - What I tried: float32 positions, differences, squares, sums, exponent and exp in
     every combination; `cdist` in both modes; the matmul form; offsets and torus wraps.
     None reproduces it exactly.
   - Likely cause: v1 computes some part of the presence in float32.
   - Request: regenerate this golden with v1's presence in float64, or say which formula
     v1 used. I implement the spec's formula in the model's dtype and did not imitate an
     unknown float32 path.
2. **Position decoding range** (deviation 1). Please confirm `[-P/2, P/2)` in the spec.
3. **The belief's `age`** (deviation 5) is carried but not read. Either drop it from spec
   5.1, or say that `time_since_seen` should come from it (it would then join the replay).
4. **"With fog off every ship's spread is zero"** (deviation 7). Should a dead enemy's row
   be certain too? It could hold its last state with zero spread while dead, or the test
   could stay "living ships".
5. **Return scale and `values`.** Should `Outputs.values` be the raw bin expectation (as
   now), or that times the learner's return scale? Section 6 decides where the scale
   lives.
6. **Rotary periods.** `(2π/p)·2^k` with `p = 64 m` gives periods from 64 m down to 0.5 m.
   Ships fight at 100-1000 m, so the finest pairs alias heavily. Did the spec mean
   `2^-k` (periods 64 m to 8 km)? The constants are `YemongConfig.rotary_period` and
   `rotary_octaves`; the sign is one line in `Yemong.__init__`.
7. **contracts.md current schema:** add `Outputs.values/outcome/replay` and
   `Trajectory.values/outcome/replay/initial_carry`.
8. **Replay memory.** The replay is about twice the size of the stored observation (70
   floats per ship row, against the observation's 24). If memory binds at large sizes,
   two options:
   - store the replay's means and spreads for hidden rows only (visible rows are truth
     with zero spread). This needs a gather;
   - or store the pre-observation belief and re-run `observe` in `unroll`. That is cheap,
     but it changes the replay's definition.
9. **bf16** (deviation 11). The learner section should decide whether to run rollouts and
   updates under bf16 with the ratio computed against a bf16 `unroll`.

## Fix round (2026-11-01)

This section supersedes the earlier text where they differ, including deviations 1, 2, 5 and
11 and requests 1–7. `uv run --no-sync pytest` passes in full, with all 43 Yemong
equivalence tests running (the presence golden was regenerated on `v2`). Ruff is clean.

### Design change: a per-view position offset
- Each view draws an offset in `[0, P)` per axis, P = 26 244 m:
  - at every reset, and in `initial_carry`;
  - from the default generator, so it is capture-safe like the action noise.
- It lives in `Carry.offset` and `Replay.offset`.
- `truth` adds it to the observed ship positions, and `compose` adds it to the field
  centres and the map centre. The belief, the codes, the forecasts and the rotary
  positions all live in that shifted frame.
- The global token has its position code back (the shifted map centre, sigma 0), and its
  rotary position is that centre.
- Decoded forecast positions come back in `[0, P)`. They are mapped to the copy nearest
  the prior belief (`_nearest`), so the frame stays continuous and differences never
  wrap. *Deviation:* the coordinator's text says the centring "goes away". Without
  `_nearest`, a ship near the period seam would appear 26 km from its neighbours in
  presence and rotary.
- A row with no belief sits at the map centre, not at the frame origin. *Deviation:* at
  the origin it would reveal the offset to rotary attention.
- Tests check that resets draw different offsets and keep them within a match. They also
  check that, with the position-code columns zeroed, a decision does not change when the
  history is moved to a frame shifted by delta (1e-8 in float64).

### Float32 precision in the shifted frame
- The frame reaches about 40 km. Two quantities then lost float32 precision, and fused
  (compiled) and unfused kernels rounded them differently:
  - the finest position-code level (a coordinate near 2e4);
  - the rotary angles (thousands of radians).
- Both now reduce first:
  - the circular code reduces modulo each level's period, which is exact for integer
    periods (spec-equivalent);
  - rotary uses `2π (x mod p)/p`, with the periods held as exact powers of two.
    `rotary_tables` takes periods; the adapter converts frequencies to periods
    (`2π/f`).
- The compiled and eager step tokens now agree to 4e-6 (was 1e-3), and the values after
  one step to 1.4e-6 (was 1.2e-4).

### Answers applied
- **Rotary erratum:** periods run from 8192 m down to 64 m (`rotary_period = 8192`).
- **`time_since_seen`** is `age × tick`, from the belief. `Replay.age` carries it.
- **Fog off:** "zero spread" applies to living ships only (deviation 6 stands).
- **`Outputs.values`** stays the raw expectation; bf16 stays opt-in.

### Review findings
1. **Scan memory.** Measured at (L, B, D) = (128, 1024, 128), compiled, forward and
   backward, under the bench lock:

   | Scan | Time | Peak extra |
   |---|---:|---:|
   | Hillis-Steele (before) | 27.2 ms | 1152 MiB (18× one input) |
   | Plain loop | 416 ms | 322 MiB |
   | Chunked, chunk 8 | 12.9 ms | 448 MiB |
   | Chunked, chunk 16 | 19.1 ms | 472 MiB |
   | Chunked, chunk 32 | 40.8 ms | 420 MiB |

   Adopted: chunked with `SCAN_CHUNK = 8`. L is padded to whole chunks.
2. **Tests from a carry with history.**
   - Split segment: unroll of steps 32–63 from the step loop's carry at t = 32 matches
     to 1e-9; with the trunk carry zeroed it does not (asserted to be more than 1e-3 off).
   - `test_unroll_reproduces_the_stored_rollout` runs the rollout twice, so the stored
     initial carry has history.
   - Found while doing this: the test helper randomised only `last.weight`, which missed
     the grouped heads' `lasts.*`. The next-state head was therefore dead in tests. It
     now randomises them, and asserts they are live.
3. **Step speed.**
   - Fused: gate and up; the two RG-LRU gates; the first linear of the heads that share
     a token (`HeadGroup`: own ships = policy + value; ships = enemy action + next
     state). The adapter concatenates the fixture weights.
   - `step` runs the enemy-action and next-state heads, and the decode, on enemy rows
     only. `Decision.enemy_action_logits` and `next_state_logits` are `(R, N - S, ...)`.
   - Carry copies: under compile, inductor already folds the reset `where`s into their
     consumers. The step's new carry is written once and copied by the runtime. I left
     the conv-buffer shift (a ring buffer would need a per-view tick and a gather).
   - Step profile, 1792 views at the 40-ship size, compiled: 88.4 → 85.6 ms in float32
     and 52.4 → 48.4 ms with bf16 matmuls. The trunk dominates (sgemm about 34 ms, the
     gated MLP's gelu-mul 11.6 ms). The heads were a small share.
   - Drift, compiled step loop against compiled unroll, 128 steps at squad:

     | Precision | Policy logits | Values | PPO \|ratio − 1\| max | mean |
     |---|---:|---:|---:|---:|
     | float32 | 1.8e-6 | 8.3e-7 | 1.4e-6 | 2.4e-7 |
     | bf16 matmuls | 1.9e-2 | 9.3e-3 | 1.6e-2 | 2.7e-3 |
     | TF32 (set at the entry point) | 1.1e-3 | 4.7e-4 | 7.3e-4 | 1.3e-4 |

   - Not resolved: the compiled and captured rollout's stored values drift from unroll by
     up to 1.5e-4 over 64 ticks, growing through the recurrence. A step compiled and
     captured on its own (with in-place carry `copy_`) agrees to 2e-6. So the cause is
     specific to compiling the step inside the whole act stage. I did not isolate it.
     The runtime storage test uses 1e-3.
4. **Replay in bf16.** The enemy-action probabilities and the spreads are stored in bf16,
   and `step` composes from the same rounded values, so step and unroll stay exact.
   - Measured on real forecast spreads: rounding changes the state code by up to 5.4e-3
     absolute, and the next-state baseline `log(code + 1e-3)` by up to 0.44.
   - I judge the spreads **not clearly safe**. Recommendation: store spreads in float32
     (one line, `inputs.STORED_DTYPE` applied only to `enemy_action`). I did not have
     time to make and re-measure that change.
   - The 26-ship cell re-run and the new rollout and unroll cells were still running when
     I handed back. Results will be in `/tmp/bnb-scratch-yemong/bench_run2.log`.
6. Hidden-columns test: now runs after 40 ticks of history, and asserts that hidden ships
   hold beliefs.
7. Cooldown residuals: derived from the card (`ShipStateCode.spread_count`).
8. `density.HexGrid` builds its table once. `ReturnScale.update` uses `nanquantile` with
   precomputed levels and has no host sync.
9. The rollout probes the learner's first row only, then allocates `(L, R, ...)`.

Nits:
- full-word names (`sandwich`, `shots`, `density`, `codes`, `state_code`);
- `TEAMS` imported from `bnb.game.rules`;
- the bf16 docstring corrected;
- `Decision.outcome_probabilities` against `Unrolled.outcome_logits`.

### Fix round, part 2: the act-stage drift and the replay dtypes

**Cause of the act-stage drift.** It was an inductor numerics setting, not an ordering or
aliasing bug, and not reassociation through the recurrence.
- The replay stored the spreads as bf16, and the step composed its tokens from that rounded
  copy.
- Inside the compiled act stage, inductor fused the fp32 → bf16 → fp32 round trip into the
  kernels that consume it, and by default drops the rounding there
  (`torch._inductor.config.emulate_precision_casts = False`).
- So the compiled step read unrounded spreads while the stored replay held rounded ones.
  `unroll` then saw slightly different inputs at every tick, and the recurrence carried the
  difference forward.

**How it was isolated:** the same act stage, values compared with `unroll` over 64 ticks
(`drift_bisection.py`):

| run | max value error |
|---|---:|
| eager, captured | 6e-7 |
| compiled, bf16 spreads | 1.4e-4 |
| compiled, bf16 spreads, `emulate_precision_casts = True` | 2.4e-6 |
| compiled, float32 spreads | 2.3e-6 |
| compiled, not captured | 2.7e-6 |

The checks the coordinator listed came out clean:
- TF32 and matmul precision: off everywhere ("highest").
- The replay's previous action equals the stored action of the step before, cleared at
  resets.
- The offset is constant within an episode.
- Visible rows' stored means equal the shifted truth from the stored obs, to 2e-7.

**Fix:**
- The spreads are stored in float32.
- The enemy-action forecasts are stored as int16 fixed point, `round(p * 32767)`:
  2 bytes, resolution 3e-5. The step composes its tokens from those stored integers. Integer
  rounding is not elided, so the step reads exactly what the replay holds. (bf16 storage
  would have the same elision problem.)
- With this, the compiled, captured rollout agrees with `unroll` to 2e-6 over 64 ticks, flat
  in time. The runtime storage test is back to 1e-5.

**Rule for the learner section:** never store a rounded copy that a compiled stage also
reads unrounded. Round with integer storage, or store at full precision.

**Benchmark (bench_run2).** Re-run after the spreads change. The cells still running at
hand-back write their results to `/tmp/bnb-scratch-yemong/bench_run2.log`; I have not
copied them here.

**Rollout cells from bench_run2** (after the spreads change; Yemong as learner and league;
TF32 off; min of 3 runs after 2 warm-up runs):

| Cell | float32 env-ticks/s | float32 VRAM | bf16 env-ticks/s | bf16 VRAM | Before the fix round (float32 / bf16) |
|---|---:|---:|---:|---:|---|
| 114 ships, 360 envs | 4,152 | 6.14 GB | 7,513 | 5.92 GB | 4,028 / 6,941 |
| 40 ships, 1024 envs | 11,729 | 6.17 GB | 21,347 | 5.95 GB | 11,302 / 19,771 |
| 26 ships, 1576 envs | out of memory in compile and capture | | not run | | out of memory |

The unroll cells were still running at hand-back. Their results are in
`/tmp/bnb-scratch-yemong/bench_run2.log`.
