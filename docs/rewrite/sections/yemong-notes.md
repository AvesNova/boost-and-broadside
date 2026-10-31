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
