# Yemong equivalence fixtures

v2's Yemong keeps v1's ideas and math in new code. These tests pin that down. Each
golden (`golden/<case>.npz`) holds random inputs and random weights (every parameter,
norms included), plus the outputs v1's own modules gave for them. They were computed in
float64 on the CPU at tag `v1-torch-final`. The generator lives outside this tree.

- **Run:** `uv run --no-sync pytest tests/reference/yemong`. The tests skip until
  `yemong_arm.make_arm()` returns an implementation.
- **Fixed:** the tests, the helpers, the conftest and the goldens. The Yemong section
  writes only `yemong_arm.py`, a thin adapter from this interface to `bnb`.
- **Validated:** the tests pass on v1 and fail on each of 37 deliberate mutants of it,
  every one caught by a value comparison. Examples: the reset step off by one, rotary pairs
  split in halves, tanh GELU, an RMSNorm epsilon of 1e-6, the near-miss sides swapped.
- **Tolerances:** float64 against the goldens must agree to 1e-9, relative to each array's
  own scale. The network tests also run in float32 at 2e-4.
- **No wrap:** v1's world was a torus; v2's arena is an open disc. Every case keeps its
  displacements far inside v1's world, so v1's minimum-image step was the identity.
  Implement without wrapping.

## Conventions

These are what the goldens encode. Each was checked by a mutant.

- **Layers:**
  - `Linear` weights are `(out, in)`, as in `torch.nn.Linear`.
  - RMSNorm has a learned scale and uses torch's default epsilon (the dtype's machine
    epsilon).
  - GELU is the exact (erf) form.
  - Attention scale is `1/sqrt(head_width)`.
- **Batch and time:** shapes are `(..., tokens, width)`. Sequences are `(B, T, ...)` for
  the temporal pieces and `(T, B, ...)` for the trunk.
- **`done[t]`:** true when the episode ended on step `t`, so step `t + 1` starts fresh.
  - Recurrence and convolution taps never cross that boundary.
  - The step at which the episode ended still reads its own history.
- **Carries:**
  - `state` is the RG-LRU state, `(blocks, B, queries, width)`.
  - `convolution_buffer` is `(blocks, B, queries, 3, width)`: the last 3 inputs to the
    convolution, oldest first.
  - On the rollout path the caller zeroes both for an environment after its episode ends.
    The update path's `done` flags must then give identical outputs.
- **Queries:** the ships, then the global token. Map tokens (fields) are key/value memory
  only.

## Interface

Every method takes and returns torch tensors, except configuration values, which are
Python numbers or lists. `weights` is a flat dict keyed by the names below.

### Temporal

| method | returns |
|---|---|
| `parallel_scan(decay, contribution, initial)`: `(B, T, D)`, `(B, T, D)`, `(B, D)` | `h` `(B, T, D)` and the final `(B, D)` for `h_t = decay_t h_{t-1} + contribution_t` |
| `rg_lru(weights, inputs, initial, done)`: `(B, T, D)`, `(B, D)`, `(B, T)` | outputs `(B, T, D)` and the final state |
| `temporal_block(weights, inputs, initial, convolution_buffer, done)`: buffer `(B, 3, D)`; `done` may be `None` | outputs, final state, final buffer |

RG-LRU weights:
- `log_decay` `(D)`;
- `recurrence_gate.{weight,bias}`;
- `input_gate.{weight,bias}`.

The decay exponent constant is 8.

Temporal block weights:
- `input_norm.weight`;
- `recurrent_input.weight` (no bias);
- `convolution.weight` `(D, 4)`, depthwise, with column `k` applied to the input
  `3 - k` steps back;
- `convolution.bias`;
- the RG-LRU weights;
- `gate_branch.weight`;
- `output.weight`;
- `feedforward_norm.weight`;
- `feedforward.{gate,up,down}.weight` (gated MLP, 4x width, no biases).

### Spatial

| method | returns |
|---|---|
| `rotary_tables(frequencies_x, frequencies_y, frequencies_heading, position, heading)`: frequencies in radians per unit; `position` `(..., N, 2)`; `heading` `(..., N, 2)` cos/sin, `(0, 0)` for headless tokens | `(cos, sin)`, each `(..., N, 1, R)`, R = total frequencies |
| `apply_rotary(values, cos, sin)`: `(..., N, H, W)` | the first `2R` dims rotated as interleaved (even, odd) pairs, the rest passed through |
| `gated_mlp(weights, inputs)`: `{gate,up,down}.weight` | `down(gelu(gate x) * up x)` |
| `spatial_block(weights, heads, tokens, map_memory, token_tables, map_tables)`: `(B, Q, D)`, `(B, M, Dm)` | `(B, Q, D)` |

Spatial block weights:
- `attention_norm.weight`;
- `query_key_value.weight` `(3D, D)`;
- `output.weight`;
- `map_norm.weight` `(Dm)`;
- `map_key_value.weight` `(2D, Dm)`;
- `feedforward_norm.weight`;
- `feedforward.{gate,up,down}.weight`.

The queries attend over the query keys followed by the map keys in one softmax, and the
map keys are rotated by `map_tables`.

### Model

| method | returns |
|---|---|
| `encoder(weights, ship, global_token, field)`: `(..., N, Fs)`, `(..., 1, Fg)`, `(..., M, Ff)` | queries `(..., N+1, D)` and map memory `(..., M, Dm)` |
| `trunk_sequence(weights, config, tokens, map_memory, token_tables, map_tables, state, convolution_buffer, done)`: `(T, B, Q, D)`, `(T, B, M, Dm)`, tables `(T, B, tokens, 1, R)`, `done` `(T, B)` | `(T, B, Q, D)` |
| `trunk_step(weights, config, tokens, map_memory, token_tables, map_tables, state, convolution_buffer)`: one step, no time axis | `(tokens, state, convolution_buffer)` |

- **Encoder:**
  - Each kind (`ship`, `global`, `field`) has its own first projection:
    `{kind}.first.{weight,bias}` to `2D`, then `{kind}.first_norm.weight`, then GELU.
  - All kinds then share the second: `shared.{weight,bias}`, then `shared_norm.weight`.
  - Map memory is `map_memory.weight` (no bias), then `map_memory_norm.weight`, applied to
    the field tokens.
- **Trunk:**
  - The config holds `width`, `heads`, `map_width`, `blocks` and `spatial_per_block`.
  - Each block is `spatial_per_block` spatial sublayers, then one temporal sublayer.
  - Weights are named `blocks.{i}.spatial.{j}.<spatial name>` and
    `blocks.{i}.temporal.<temporal name>`.

### Heads

| method | returns |
|---|---|
| `mlp_head(weights, inputs)`: `first.{weight,bias}`, `norm.weight`, `last.{weight,bias}` | `last(gelu(norm(first x)))` |
| `next_state_head(weights, inputs, code)` | `log(code + 1e-3) + mlp_head(inputs)` |
| `critic_values(local_logits, outcome_logits, bins, local_levels, outcome_level)`: `(..., N, K_local, bins)`, `(..., 4)` | `(..., N, K)`: per-ship levels as the bin expectation, the outcome level as P(win) - P(loss) broadcast over ships |

### Targets and codes

- **Critic:**
  - `value_bins(count, limit)`;
  - `two_hot(values, bins)`;
  - `expectation(logits, bins)`;
  - `outcome_values(probabilities)`;
  - `outcome_class(result)`: 0 win, 1 tie, 2 loss;
  - `outcome_targets(probabilities, result, terminated, next_probabilities, gamma, lam)`;
  - `return_scales(returns, masks, decay, floor)`: the scale after each batch.
- **Circular:** `circular_encode(period, colours, levels, mean, sigma)` gives
  `(..., levels, colours)`. `circular_decode(period, colours, levels, probabilities)` gives
  `(mean, spread)`.
- **Position:** `position_encode(size, levels, mean, sigma)` gives `(..., levels, 9)` on a
  `size = (x period, y period)` torus code. `position_decode(size, levels, probabilities)`
  is the inverse.
- **Scalar:**
  - `scalar_sharp(low, high, bins, values)`;
  - `scalar_encode(low, high, bins, mean, sigma)`;
  - `scalar_decode(low, high, bins, probabilities)` gives `(mean, spread)`.
- **Velocity:**
  - `compress_velocity(velocity, knee)` and `expand_velocity(compressed, knee)`;
  - `velocity_jacobian(velocity, knee)` gives the packed `(xx, xy, yy)` Jacobian and its
    inverse;
  - `velocity_sharp(bins, spacing, knee, velocity)`;
  - `velocity_encode(bins, spacing, knee, mean, covariance)`, with the covariance packed
    `(xx, xy, yy)`;
  - `velocity_decode(bins, spacing, knee, probabilities)` gives `(mean, covariance)`.
- **Grouped codes:** a code vector is a run of softmax groups; `runs` is
  `[(groups, size), ...]` in vector order.
  - `grouped_log_softmax(logits, runs)`.
  - `grouped_cross_entropy(logits, target, runs)` gives the cross-entropy per group.
  - `remove_baseline_floor(logits, runs, floor)` takes the group softmax, then undoes the
    floor's mixture with uniform, clipping at zero and renormalising.
- **Density:**
  - `hex_cells(rings)` gives the axial coordinates `(C, 2)` in the grid's order and the
    centres `(C, 2)` in units of the arena radius.
  - `hex_counts(x, y, is_ally, alive, arena_radius, rings)` gives `(B, 2C)`: ally counts,
    then enemy counts.
  - `poisson_terms(log_rate, counts)` gives `(per_cell, saturated)`.
- **Shots:**
  - `shot_trajectory_code(parameters, snapshots)`: snapshots are `(..., 5)` (displacement
    x, y; velocity x, y; refractive index);
  - `shot_outcome_code(parameters, hit, distance, side)`: hit 0 none, 1 enemy, 2 ally;
    side 1 enemy, 2 ally;
  - `shot_time_features(ages, max_age, harmonics)`.

  `parameters` holds `displacement_period`, `displacement_levels`, `direction_levels`,
  `top_speed`, `speed_bins`, `index_span`, `index_bins`, `collision_radius`, `near_bins`,
  `near_span_radii` and `near_sigma_bins`.
- **Features:**
  - `symlog(values)`;
  - `respawn_features(seconds, horizon, harmonics)`;
  - `local_presence(position, team, source, radius)`: team 0/1 ships, 2 map tokens;
    `source` is who may contribute.

### Belief

- `belief_observe(belief, truth, visible, spawned, tick_seconds)`:
  - `belief` is a dict with `valid` `(B, N)`, `age` (ticks), `means`, `uncertainty` and
    `action` (probabilities);
  - it returns the same keys plus `time_since_seen` (seconds).
- `belief_advance(belief, prediction, enemy_action_logits or None)`:
  - `prediction` is means, then uncertainties;
  - it returns `means`, `uncertainty`, `action` and `nonfinite` (the count of non-finite
    prediction entries).
