# Yemong spec (section 5)

Status: spec, 2026-11-01. Decisions behind it: `yemong.md`. Equivalence fixtures:
`tests/reference/yemong/` (README there).

Yemong is v1's model with the same ideas and math in new code. Where v1's math is kept,
this text states it exactly, and the fixtures check it. Changes from v1 are marked
**(v2)**; they are decided and not open to reinterpretation. Constants marked
**(measure)** are proposals in the new units: report the measurement the text asks for,
and keep the value unless the measurement shows it is wrong.

Notation:
- `D` is the model width.
- `N` is ships per view, `S` own ships, `M` fields, `Q = N + 1` query tokens (ships, then
  the global token).
- `R` is views, `L` is steps.
- `σ(·)` is the logistic function, and `gelu` is the exact erf form.
- `norm` is RMSNorm with a learned scale and torch's default epsilon (the dtype's machine
  epsilon).
- `eps` and `tiny` are `torch.finfo(dtype).eps` and `.tiny`.
- `Linear` weights are `(out, in)`.

## 1. Scope

- **Model:** the model in `bnb/models/yemong/`, with `step` (one decision) and `unroll` (a
  stored segment). It owns its inputs, its carry and its belief.
- **Controller:** a controller over it in `bnb/controllers/`.
- **Targets and codes** the learner will need: the critic distributions, the ship-state
  code, the hex density target, the shot codes and the return scale. They are built and
  tested here. The losses, labels and training loop are section 6.
- **Contract additions** (section 10) and the mechanical runtime storage they need.

## 2. Configuration

These are the production defaults (v1's known-good run). Nothing in the weights depends on
`N`, `M` or the arena, so one set of weights plays at every size.

| name | value | notes |
|---|---|---|
| width `D` | 128 | |
| blocks | 2 | each: 2 spatial sublayers, then 1 temporal |
| spatial heads | 2 | head width 64 |
| map memory width `D_m` | 64 | fields are key/value memory only |
| head hidden width | `2D` | every MLP head |
| value bins | 51 | over `symexp(linspace(−2, 2, 51))` |
| reward levels `K`, outcome level | from config | rewards section; the outcome is one level, read off the global token |
| hex rings | 10 | 331 cells |
| actions | 42 **(v2)** | joint index `power·14 + stick·2 + shoot` (`bnb.contracts.action`) |

## 3. Ship state and its code

### 3.1 Belief channels (v2)

Means `C = 12`, in this order:
1. `x`, `y` (m);
2. `velocity_x`, `velocity_y` (m/s, world);
3. `heading`, an angle in `[−π, π)` from `atan2(heading_sin, heading_cos)`;
4. `nose_rate` (rad/s);
5. `angle_of_attack` (rad);
6. `shield_recharge_wait` (s);
7. `shield`;
8. `energy` (J);
9. `gun_cooldown` (s);
10. `log_refractive_index`.

Spreads `U = 15`, in this order:
- position sigma;
- velocity covariance `xx, xy, yy`;
- sigmas of heading, nose rate, angle of attack, shield wait, shield, energy and log index;
- 4 cooldown residuals.

Zero spread is certainty. One decoded prediction is `C + U = 27` wide. `proper_speed` is
not a channel: it equals `|v| · n` and is never read.

### 3.2 Code layout (v2)

The code is one vector of softmax groups. These are its runs in order, as
(name, groups × size):

| run | groups × size | code |
|---|---|---|
| position | 9 × 9 | position code, period `P_pos = 26 244 m` per axis (`4 · 3⁸`; finest level 4 m), 9 levels **(v2)** |
| velocity | 3 × 81 | velocity code, knee 50 m/s, 81 bins, spacing 2.5 (reach `\|v\| ≤ 50(e² − 1) ≈ 319 m/s`) **(measure)** |
| heading | 4 × 4 | circular, period 2π, 4 colours |
| nose_rate | 1 × 41 | scalar, ±8 rad/s **(measure)** |
| angle_of_attack | 1 × 21 | scalar, ±2 `α_s` (the lift curve's clip; the largest `α_s` over types) |
| shield_recharge_wait | 1 × 21 | scalar, 0 to the shield delay (6 s) |
| shield | 1 × 21 | scalar over the **fraction** of the ship type's maximum, 0 to 1 **(v2)** |
| energy | 1 × 21 | scalar over the **fraction** of the type's `E_max`, 0 to 1 **(v2)** |
| log_refractive_index | 1 × 21 | scalar, ±ln 2 (field levels span `n` from ½ to 2) |
| gun_cooldown | 1 × 4 | `centres = 0, h, 2h, 3h`: bins = 1 + the largest ticks between shots over types |

Width 490, 23 groups. Shield and energy are coded as fractions. The belief stores physical
values, and decoding multiplies the fraction and its sigma back by the type's maximum,
read from the ship card the model was built with.

**(measure)** On a scripted rollout at each preset, with fog on and off, report:
- the share of ship-steps whose world speed exceeds the velocity reach;
- the share whose `|nose_rate|` exceeds 8.

Both should be below 1e-3. If not, widen the range and say so.

### 3.3 Code primitives (v1, exact)

**Circular code** (period `P`, colours `c ≥ 3`, levels `L`). Level periods are
`p_l = P / c^l`.

- **Encode** `(mean m, sigma s)` to `(…, L, c)`. With `u_l = (m mod P)/p_l · c`,
  `f = u_l − ⌊u_l⌋` and `j = ⌊u_l⌋ mod c`:
  - the sharp two-hot puts `1 − f` on colour `j` and `f` on `(j + 1) mod c`;
  - attenuation `a_l = exp(−½ (2π s / p_l)²)`;
  - `probability = sharp · a_l + (1 − a_l)/c`.
- **Decode** to `(mean, spread)`. Angle step `θ = 2π/c`. Per level:
  - `z = Σ_k p_k e^{i k θ}`;
  - `phase = atan2(Im z, Re z) mod 2π`;
  - `sector = min(⌊phase/θ⌋, c − 1)`;
  - `λ = phase − sector θ`;
  - chord inversion: `f = clip(sin λ / max(sin θ cos λ + (1 − cos θ) sin λ, eps), 0, 1)`;
  - `ρ = sqrt((1 − f + f cos θ)² + (f sin θ)²)`;
  - attenuation `a = min(|z|/ρ, 1)`;
  - `local mean_l = (sector + f) p_l / c`;
  - precision `π_l = a² (P/p_l)²` if `a > 1e-4`, else 0.
- **Mean:**
  - start at local mean₀ with total precision `π₀`;
  - for `l ≥ 1`: `unwrapped = lm_l + round((mean − lm_l)/p_l) p_l`, then
    `total += π_l`, then `mean += π_l / max(total, tiny) · (unwrapped − mean)`;
  - return `mean mod P`.
- **Spread:**
  - set `a = 1` where `a > 1 − 1e-6`;
  - informative levels are those with `a > 1e-4`;
  - per level `κ_l = ½(2π/p_l)²` and weight `w_l = informative · a²`;
  - `var = max(0, −Σ w κ ln max(a, 1e-4) / max(Σ w κ², tiny))`;
  - `var = 0` if every level has at most two nonzero probabilities;
  - `spread = sqrt(var)` if any level is informative, else `P`.

**Position code** (`(P_x, P_y)`, `L` levels):
- **Encode:** a 3-colour circular code per axis, both with the same sigma. The joint
  `(…, L, 9)` is the outer product, x-major: entry `3 i + j`.
- **Decode:** decode the two marginals. The mean is per axis;
  `spread = sqrt((s_x² + s_y²)/2)`.

**Scalar code** (`low`, `high`, `n ≥ 3` bins). Spacing `Δ = (high − low)/(n − 1)`.
Coordinate `k = clip((m − low)/Δ, 0, n − 1)`, lower bin `b = min(⌊k⌋, n − 2)`,
`t = k − b`.

- **Sharp:** `1 − t` at bin `b` and `t` at `b + 1`.
- **Encode** `(m, s)`:
  - `last = n − 1`;
  - `var = min((s/Δ)², max(k(last − k) − t(1 − t), 0))`;
  - offsets `o_i = i − b`; `width = max(var, 1)`; `weight = min(var, 1)`;
  - `G(d) = exp(−½ d²/width) / sqrt(2π width)`;
  - `smooth_i = (1 − t) G(o_i) + t G(o_i − 1)`;
  - fold the tails with `r = sqrt(2 width)`:
    - `smooth_0 += ½[(1 − t) erfc((b + ½)/r) + t erfc((b + 1.5)/r)]`;
    - `smooth_last += ½[(1 − t) erfc((last − b + ½)/r) + t erfc((last − b − ½)/r)]`;
  - `p = (1 − weight) · sharp + weight · smooth`, normalised to sum 1;
  - then **match moments** to mean `t` and variance `var + t(1 − t)`, both about bin `b`.
- **Match moments** `(p, offsets o, target mean μ*, target variance v*)`. Tolerance
  `τ = 8 eps`; interior bins are `0 < i < n − 1`.
  - `μ = Σ p o`; `s₂ = Σ p o²`; `src = p · interior`; `mass = max(Σ src, tiny)`.
  - `δ = (μ* − μ)/mass`, set to 0 if `|δ| ≤ τ`.
  - `δ₂ = |δ| Σ src (2 sign(δ) o + 1)`.
  - `ν = max((v* + μ*² − s₂ − δ₂)/(2 mass), 0)`, set to 0 if `≤ τ`.
  - `e = max(2ν + |δ|, 1)`; then `ν /= e` and `δ /= e`.
  - `up = max(δ, 0) src`, `down = max(−δ, 0) src`, `spread = ν src`.
  - Result: `p − (up + down + 2 spread) + shift_up(up + spread) + shift_down(down + spread)`.
    Here `shift_up` moves mass to bin `i + 1` and `shift_down` to bin `i − 1`.
- **Decode:**
  - `k̄ = clip(Σ p_i i, 0, n − 1)`; `var = Σ p_i (i − k̄)²`; `t = k̄ − ⌊k̄⌋`;
  - `physical = var − t(1 − t)`, set to 0 if `≤ τ`;
  - `mean = low + k̄ Δ`; `spread = sqrt(physical) Δ`.

**Velocity code** (knee `κ`, `n` bins, spacing `δ`):
- **Compression:**
  - `compress(v) = v κ ln(1 + |v|/κ)/|v|`, and 0 at 0;
  - `expand(c) = c κ (e^{|c|/κ} − 1)/|c|`, and 0 at 0.
- **Packed Jacobian** `(xx, xy, yy)` of `T I + (R − T) v̂v̂ᵀ` with `v̂ = v / max(|v|, eps)`:
  - at `v`: `R = 1/(1 + |v|/κ)` and `T = κ ln(1 + |v|/κ)/max(|v|, eps)` (`T = 1` at 0);
  - inverse: `R = 1 + |v|/κ` and `T = |v| / max(κ ln(1 + |v|/κ), tiny)` (`T = 1` at 0).
- **`sandwich(A, S) = A S A`** for packed symmetric 2×2 matrices.
- **Projections** onto three axes at 0°, 120° and 240°:
  `(x, −x/2 + (√3/2) y, −x/2 − (√3/2) y)`.
- **Axis code:** a scalar code over `±½(n − 1)δ`.
- **Sharp:** the axis sharp of the projections of `compress(v)`.
- **Encode** `(mean, covariance)`:
  - `(xx, xy, yy) = sandwich(J(mean), covariance)`;
  - axis variances `[xx, ¼xx − (√3/2)xy + ¾yy, ¼xx + (√3/2)xy + ¾yy]`;
  - axis-encode the projections of `compress(mean)` with sigma `sqrt(max(var, 0))`.
- **Decode:**
  - axis-decode to `(p_i, s_i)`;
  - `compressed = (2/3)(p₀ − ½(p₁ + p₂), (√3/2)(p₁ − p₂))`;
  - with `v_i = s_i²`: `xx = v₀`, `xy = (v₂ − v₁)/√3`, `yy = (2(v₁ + v₂) − xx)/3`;
  - clip `xx, yy ≥ 0` and `|xy| ≤ sqrt(xx yy)`;
  - `mean = expand(compressed)`, `covariance = sandwich(J⁻¹(mean), (xx, xy, yy))`.

**Grouped codes:**
- `log_softmax` within each group.
- Cross-entropy per group: `−Σ target · log p` over the group.
- **Baseline floor removal:** take the group softmax `p`, then
  `p' = max((1 + size · ε) p − ε, 0)`, then `p' / max(Σ p', tiny)`, with `ε = 1e-3`.
- **Baseline:** `log(code + 1e-3)`.

### 3.4 The ship-state codec

- **`encode(means, spreads)`** concatenates the runs of 3.2 in order:
  - position from `(x, y)` with the position sigma;
  - velocity from its mean and covariance;
  - heading from its angle and sigma;
  - the five scalars, each with its sigma (shield and energy divided by the type's
    maximum);
  - `gun_cooldown` as `clamp_min(sharp(cooldown) + residual, 0)`.
- **`sharp(means)`** is `encode` with every spread zero.
- **`decode(probabilities)`** inverts each run. For the cooldown:
  `mean = Σ p_i centre_i` and `residual = p − sharp(mean)`. Heading is wrapped to
  `[−π, π)`.
- **`decode_logits(logits)`** is floor removal per group, then `decode`.

## 4. Inputs: what the model composes from a view

Per view, the tokens are ships `0 … N−1` (rows as observed: own `S` first, then the
enemies), then the global token, then fields `0 … M−1`. Each ship row's physical state is
the belief's (section 5), so it is truth when visible.

**Ship token** (width 490 + 63 = 553):
- the position code (81) and state code (409) of the belief means and spreads;
- `team_relative` one-hot (2);
- `ship_type` one-hot (types);
- `alive`, `visible`, `fired` (3);
- `symlog(time_since_seen)` (1), with `symlog(x) = sign(x) ln(1 + |x|)`;
- respawn features of `respawn_wait` (9):
  `[s/H, sin(2πk s/H), cos(2πk s/H)]` for `k = 1…4`, with `H = 16 s`;
- pending action (42), from the carry (5.3);
- `log_index_gradient / g*` (2), with `g* = 1.875 · 2 ln 2 / 40 m`. It reads zero for a
  hidden ship, as the observation gives it;
- local presence (2), below.

**Field token:** position code of `(x, y)` with zero sigma (81), `radius / 1000 m`,
`band_width / 40 m`, `level / 2`.

**Global token (v2):** `episode_progress`, `arena_radius / 2600 m`, `fogged`. Mode features
join with the modes section. v1's global position code is dropped: the centre is always
the origin.

**Local presence** (v1 with v2 range). For ship token `i`:
- `ally_i = Σ_{j ≠ i, same team, j valid} exp(−|p_i − p_j|²/2r²)`;
- `enemy_i = Σ_{other team, j valid} exp(−|p_i − p_j|²/2r²)`;
- both through `ln(1 + ·)`;
- `r = 150 m` **(measure**: about 0.8 of the gun's reach, as v1's 330 px was of its
  421 px).

Positions are the belief means; "valid" is the belief's valid flag; there is no wrap.

**Rotary geometry:**
- tokens: ships at their belief position and heading; the global token at the origin with
  no heading; fields at their centres with no heading;
- frequencies `(2π/8192 m) 2^k`, `k = 0…7`, on each of x and y (periods from 8192 m
  down to 64 m), and harmonics `1, 2, 4, 8` on the heading **(v2**: metre periods; there is
  no torus, so periodicity no longer fixes the coarsest. Erratum 2026-11-01: the first
  text inverted this, giving periods from 64 m down to 0.5 m);
- 20 pairs, 40 of the 64 head dims.

**Initialising the encoder's code columns** (v1): each first projection's columns for
code inputs are drawn from `N(0, 1/G)`, where `G` is the number of softmax groups in that
kind's input (ships 23, fields 9).

## 5. Carry and belief

### 5.1 The carry

Per view:
- the trunk's `state (blocks, Q, D)` and `convolution_buffer (blocks, Q, 3, D)`;
- the belief: `valid (N)`, `age (N)` ticks, `means (N, 12)`, `spreads (N, 15)`,
  `action (N, 42)` probabilities;
- the previous own action `(S)` as a joint index.

It is a pytree of tensors, so the runtime can store it, `copy_` it and capture it.

### 5.2 One decision: `step(carry, obs, reset)`

`reset (R,)` means the environment reset before this observation.

1. **Reset:** clear the trunk carry, the belief (valid false, everything zero, action
   uniform) and the previous action (joint 0, the null command).
2. **Observe** (v1):
   - `valid ← (valid ∧ ¬spawned) ∨ visible`;
   - means are truth where visible, the belief where valid, else 0;
   - spreads are 0 where visible or not valid;
   - age is 0 where visible, `age + 1` where valid, else 0;
   - action is uniform where not valid.

   Truth is the row's columns in 3.1's order, with `heading = atan2(sin, cos)`.
3. **Compose the inputs** (section 4). Pending action:
   - own ship: one-hot of the previous action;
   - enemy: the one-hot of the null command if it spawned this tick, else the belief's
     action.
4. **Encode, then run the trunk one step** (6.4). Then the heads:
   - action logits for own ships;
   - the enemy-action logits;
   - the next-state logits for every ship;
   - the values and the outcome.
5. **Advance** (v1):
   - `prediction = decode_logits(next-state logits)`;
   - count non-finite entries into a running counter in the carry, then replace them
     with 0;
   - where valid, means and spreads become the prediction's;
   - `action = softmax(enemy-action logits)` for every row. Only enemy rows are read
     later. Before any forecast exists, the action is uniform.
6. **Act:** the controller samples the joint action (Gumbel-max), and the previous action
   becomes the sample.

`step` returns the **replay record** (section 10) besides the outputs: what step 3 read
from the belief and the carry. That is the means, spreads, action belief and previous
action after step 2. With it, `unroll` recomputes the segment in parallel without
replaying the belief.

### 5.3 `unroll(carry₀, obs, replay, reset)`

Over `(L, R, …)`, `unroll` gives every output `step` gave, from the segment's starting
trunk carry and the stored replay:
- it composes the inputs from `obs` and `replay`;
- the encoder and spatial sublayers run over `L·R` views at once;
- the temporal sublayers scan over `L` with episode starts from `reset`;
- the heads run on all of it.

It also returns:
- the density log-rates (from the global token);
- the shot-head latents (for queries at bullet ages).

**Acceptance:** `unroll` must equal a loop of `step` over the same segment (section 11).

## 6. Network (v1, exact)

### 6.1 Encoder

- **Per kind:** `h = gelu(norm(W x + b))`, width `2D`.
- **Shared:** `t = norm(W h + b)`, width `D`.
- **Map memory:** `norm(W_m t_field)`, no bias, width `D_m`.

The query tokens are the ships and the global token. Fields enter only as map memory.

### 6.2 Rotary

`angles = [x f_x, y f_y, ψ f_h]`, where `ψ = atan2(heading_sin, heading_cos)` (0 for
`(0, 0)`). `cos` and `sin` are shaped `(…, tokens, 1, 20)`.

Rotation acts on the first `2R` dims of each head, as interleaved pairs `(2i, 2i + 1)`:
`(e cos − o sin, e sin + o cos)`. The rest pass through.

### 6.3 Spatial sublayer

- **Block:** `x ← x + W_o attention(norm₁ x)`, then `x ← x + mlp(norm₂ x)`.
- **Gated MLP:** `mlp(x) = W_down(gelu(W_gate x) ⊙ W_up x)`, hidden `4D`, no biases.
- **Attention:**
  - `q, k, v` are the thirds of `W_qkv x`, split into heads (head `h` is dims
    `[h·64, (h + 1)·64)`). Rotate `q` and `k` with the token tables.
  - The map keys and values are the halves of `W_map norm_m(memory)`, keys rotated with
    the map tables.
  - One softmax over `[query keys; map keys]`, scale `1/sqrt(64)`, no mask.

### 6.4 Temporal sublayer (per query token, over time)

- `n = norm₁ x` and `u = W₁ n`. Write `reset[t]` for "an episode starts at `t`".
- **Convolution:**
  - over `[buffer (3, oldest first); u]`:
    `c_t = Σ_{k=0}^{3} w_k ⊙ in[t + k] · valid(t, 3 − k) + bias`;
  - a tap at lag `j` is valid when no episode starts at `t − j + 1 … t`;
  - the stored buffer continues the segment, so step 0 does not count as a start.
- **New buffer:** the last 3 entries of `[buffer; u]`, unmasked. On the rollout path the
  runtime clears the carry after a reset.
- **RG-LRU** on `c`:
  - `r = σ(W_a c + b_a)` and `i = σ(W_x c + b_x)`;
  - `a = σ(Λ)^{8r}`;
  - contribution `sqrt(max((1 − a)(1 + a), 1e-6)) ⊙ i ⊙ c`;
  - `a_t = 0` where an episode starts at `t` (`t > 0`);
  - `h_t = a_t h_{t−1} + contribution_t`. Use any scan whose float64 result matches.
- **Output:** `x ← x + W_out(h ⊙ gelu(W₂ n))`, then `x ← x + mlp(norm₂ x)`.
- **Initialisation:** `Λ = linspace(0, 4, D)`; torch defaults otherwise.

### 6.5 Trunk

For each block: two spatial sublayers over the `Q` queries (reading map memory), then one
temporal sublayer.

## 7. Heads and initialisation

Every MLP head is `Linear(D, 2D) → norm → gelu → Linear(2D, out)`.

| head | reads | out |
|---|---|---|
| action | own ships | 42 |
| enemy action | every ship (only enemy rows are used) | 42 |
| next state | every ship | `log(code + 1e-3) + head(x)`, 490 |
| value (per-ship levels) | own ships | `(K − 1) × 51` bin logits |
| outcome | global token | 4: win, tie, loss, unresolved |
| density | global token | `2 × 331` log-rates: ally cells, then enemy cells |
| shot outcome | every ship | 19 |
| shot trajectory **(v2)** | every ship, queried at ages | see below |

- **Values:**
  - a per-ship level is the expectation over the bins;
  - the outcome level is `P(win) − P(loss)`, the same for every own ship;
  - `values (…, S, K)` keeps the configured level order.
- **Shot trajectory (v2: a second hidden layer):**
  - `h = W_latent x + b + W_age φ(age)`, with `φ(a) = [a/A, sin(2πk a/A), cos(2πk a/A)]`,
    `k = 1…4`, `A` = the bullet lifetime in moves (12);
  - then `norm → gelu → Linear(D, D) → norm → gelu → Linear(D, 154)`.

  The latent part is computed once per ship-step and broadcast over the queried ages, so
  any number of ages can be queried in parallel. Section 6 trains it on every point of
  each bullet's flight.
- **Initialisation** (v1). First Linear orthogonal with gain √2, last orthogonal with gain
  0.01, biases zero. Then:
  - the next-state head's last weight is zero, so it starts as "nothing changes";
  - the density head's last bias is `ln(max(N, 2)/2 / 331)`, `N` at construction;
  - the shot latent is orthogonal √2; the trajectory's new middle Linear is orthogonal √2
    with zero bias; its last is orthogonal 0.01.

## 8. Critic distributions and the return scale (v1, exact)

- **Bins:** `symexp(linspace(−L, L, n))`, `symexp(x) = sign(x)(e^{|x|} − 1)`.
- **Two-hot:**
  - clamp `x` to the outer bins;
  - `upper = clip(searchsorted(bins, x, right), 1, n − 1)`, `lower = upper − 1`;
  - `w = clip((x − b_lower)/(b_upper − b_lower), 0, 1)`;
  - `1 − w` goes to `lower` and `w` to `upper`.
- **Expectation:** `Σ softmax · bins`.
- **Outcome:** the classes are win 0, tie 1, loss 2, unresolved 3. The value is
  `p₀ − p₂`, and the class of a result is win if > 0, loss if < 0, else tie.
- **Outcome targets:** backwards over `t = L−1 … 0`, with `p_L = Q_L` the bootstrap
  probabilities `p(s_L)`:
  - `Q_t = γ((1 − λ) p_{t+1} + λ Q_{t+1}) + (1 − γ) e_unresolved`;
  - `Q_t = onehot(class(result_t))` instead where the episode ended on transition `t`.
- **Return scale:** EMAs (decay `d`) of the 5th and 95th percentiles of the masked
  returns.
  - Use `torch.quantile` with linear interpolation; both start at 0; an empty batch skips.
  - The scale is `max(high − low, floor)`.

## 9. Targets (v1, exact; v2 ranges)

**Hex density:**
- **Cells:** `1 + 3R(R + 1)`.
  - **Order:** the centre, then ring by ring. Each ring starts at axial `(r, 0)` and walks
    the steps `(−1, 1), (−1, 0), (0, −1), (1, −1), (1, 0), (0, 1)`, `r` times each,
    ending one step before `(r, 0)`.
  - **Centres:** `(q + r/2, r√3/2) / R`, in units of the arena radius.
- **Each living ship splits one unit across its triangle,** by barycentric weight:
  - with `x, y` divided by `max(R_a, 1e-6)`: `f_r = y R / (√3/2)` and
    `f_q = x R − f_r/2`;
  - `u` and `v` are their fractional parts;
  - if `u + v ≤ 1`, the vertices are `(0, 0), (1, 0), (0, 1)` with weights
    `1 − u − v, u, v`;
  - otherwise they are `(1, 0), (0, 1), (1, 1)` with weights `1 − v, 1 − u, u + v − 1`.

  Vertices outside the grid drop their weight. Ally means `team_relative = 0`. The target
  reads `Privileged` truth (every living ship).
- **Poisson terms:** `l = clip(logit, −20, 2)`; per cell `e^l − y l`; saturated
  `y − y ln max(y, 1e-12)` where `y > 0`, else 0.

**Shot codes:**
- **Trajectory code** of a snapshot `(dx, dy, vx, vy, n)`:
  - the position code of the displacement (period `P_d` on both axes, 8 levels), sigma 0;
  - the circular code of `atan2(vy, vx)` (4 colours, 5 levels), sigma 0;
  - the scalar sharp of `hypot(vx, vy)` over `[0, top]` (41 bins);
  - the scalar sharp of `ln max(n, 1e-6)` over `±ln 2` (21 bins).

  Width 72 + 20 + 41 + 21 = 154; the trajectory head outputs the same width.
- **v2 constants:**
  - `top` = (the largest muzzle speed + the velocity reach of 3.2) × 2 (the lowest
    index, ½);
  - `P_d = 2.5 · top ·` the bullet life (12 moves);
  - `r_c` = the smallest hit radius over types.
- **Outcome code** (19):
  - the near code is a scalar code from `ln r_c` in steps of `ln 6 / 7`, with 9 bins;
    `clear = e^{high}`;
  - spread = `encode(ln max(d, r_c), 0.75 Δ)`;
  - at `d ≥ clear` it is one-hot on bin 8 instead;
  - classes `[enemy hit, ally hit, enemy near 1–8, ally near 1–8, clear]`: the near bins
    go to the side of the closest pass (enemy unless that side is ally), and clear takes
    bin 8;
  - a hit is a one-hot on its class.

## 10. Contract additions (v2)

These are exact, and they are the only contract changes in this section.

- **`Outputs`** gains:
  - `values (R, S, K)`: per level, raw units, the outcome column as above;
  - `outcome (R, 4)`: the outcome probabilities;
  - `replay`: an opaque pytree with leading `(R,)`.

  `value` is the sum of `values` over levels.
- **`Trajectory`** gains `values`, `outcome`, `replay` (each `(L, R, …)`) and
  `initial_carry` (the learner's carry at the segment's start).
- **The runtime** stores these for the learner's views when the controller provides them.
  These are mechanical changes; list each one.

## 11. Acceptance

1. **Equivalence:** `tests/reference/yemong` passes through `yemong_arm.py`, in float64
   and float32. The goldens and tests are fixed.
2. **`unroll == loop(step)`.** On a real rollout at `squad`, half the environments fogged,
   long enough for several resets: logits, values, outcome, next-state and enemy-action
   logits agree.
   - Tolerances: 1e-9 in float64 on the CPU; 1e-4 relative to scale in float32, eager and
     compiled.
3. **Belief:**
   - a visible ship's code is the sharp code of its observed state;
   - spawning voids a belief;
   - with fog off, every ship's spread is zero;
   - the model reads no hidden-zero column of a hidden ship (perturbing them changes
     nothing).
4. **Symmetry and size:**
   - permuting ships within a team permutes every per-ship output;
   - one set of weights runs at `duel`, `squad` and `battle`.
5. **Codes:**
   - the round trips at the v2 ranges;
   - the measurements of 3.2 and the presence radius (4), reported.
6. **Client export:** the single-step forward (one view, the carry in and out, logits and
   values out) exports to ONNX and matches eager (scaffold rule 9).
7. **Speed:**
   - `step` is compiled and capturable: no host syncs, static shapes, fresh randomness on
     every graph replay;
   - report rollout env-ticks/s with Yemong as the learner at the bench sizes, and
     `unroll` forward and backward time and peak memory for `L = 128`;
   - bf16 autocast for matmuls is allowed where measured faster, with codes, belief and
     critic expectations in float32 and the equivalence tests run in float32.

## 12. Deliberate changes from v1 (all decided)

- **Actions:** 42 joint actions (3 power × 7 stick × 2 shoot).
- **No bullets:** no bullet inputs and no bullet encoder.
- **Units and wrap:** metres and seconds, and no world wrap.
- **New state:** the ship-state code covers the new state (angle of attack in; health and
  power become shield and energy fractions).
- **Shot trajectory head:** a second hidden layer, queried at any number of ages.
- **Global token:** drops v1's position code.
- **Relational bias:** dropped (off in v1's run).
- **Ownership:** the model owns the belief and the pending-action input. There is no
  environment wrapper.

Not now: separate ally and enemy encoders, the Yemong-review design, rotary positions for
hidden ships from the forecast. These come later, as head-to-head options.
