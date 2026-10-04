# Semantic rewards, categorical state and map-scale training — plan

Status: agreed design, not yet implemented. Base: `dev` at `4eba8a1`.

Four independent changes, built and measured one at a time (see
[Build order](#6-build-order)):

1. a five-level semantic reward hierarchy that replaces the current 16 value streams;
2. a categorical next-state head and categorical state inputs;
3. a categorical critic;
4. per-episode map-scale randomisation.

Physics numbers used for sizing are collected in
[Appendix A](#appendix-a-physics-calibration).

---

## 1. Rewards

### 1.1 Principles

- **Rewards own spatial credit.** Every reward is written onto the ship that earned it.
  PPO solves temporal credit only, through per-component GAE. The only quantity broadcast
  across a team is the global outcome.
- **Delete the lambda matrix** (`_lambda_matrix`, `_precompute_lambda_aggregates`,
  `enemy_neg_k`, `ally_zero_k`, `_LOCAL_COMPONENTS`). Today every per-ship component is
  already diagonal and the team components are bit-identical across teammates, so this is
  a simplification with no intended behaviour change. The reward-decomposed gradient
  diagnostic must be rewired to read per-component advantages directly.
- **Opposing outcomes stay in separate components** wherever they can cancel within a
  discount horizon: kill enemy vs kill ally, zone captured vs zone reset, damage dealt vs
  taken, healing vs charge-back. Same-sign parts of one event are merged (final shot +
  assist).
- **Keep** scalar rewards, scalar per-component GAE, the per-component `ReturnScaler`, the
  tier scales and the existing per-tier gammas and lambdas.

### 1.2 Balance rule: offensive premium annealed to zero-sum

The current scheme is kept: an event charges the side it happens to `w` and pays the side
that caused it `ratio × w`. `kill_payout_ratio`, `damage_payout_ratio` and
`capture_payout_ratio` start at 2.0 and `offensive_bias` anneals them to 1.0, at which
point every event is exactly zero-sum.

Every charge therefore needs a payee. The rule:

- if the cause is a ship on the opposing side, pay that ship (the attribution in each
  component below);
- if nobody on the opposing side caused it (boundary damage or death, an enemy's friendly
  fire, healing of boundary damage), split the counterpart evenly over the opposing team
  (`enemy_team_share`, kept for exactly these cases).

Each component's fallback is stated below. A test must assert that, at all ratios = 1,
every component's per-env sum over both teams is zero on every tick.

### 1.3 Components

Fifteen components, down from sixteen.

| Level | Component | Sign | Recipient | Replaces |
|---|---|---|---|---|
| 1 | `outcome` | ± | global token, broadcast to allies | `ally_win`, `enemy_win`, `outcome` |
| 2 | `zone_gain` | + | attacking ships (ledger) | `front_advance` |
| 2 | `zone_loss` | − | defending ships (ledger) | `front_advance` |
| 3 | `ally_zone_captured` | − | absent defenders | `capture_progress` |
| 3 | `ally_zone_reset` | + | present defenders | `capture_progress` |
| 3 | `enemy_zone_captured` | + | present attackers | `capture_progress` |
| 3 | `enemy_zone_reset` | − | absent attackers | `capture_progress` |
| 4 | `death` | − | dying ship, any cause | `combat_death`, `boundary` |
| 4 | `kill_enemy` | + | attackers (50/50) | `kill_shot`, `kill_assist` |
| 4 | `kill_ally` | − | friendly-fire attackers (50/50) | `kill_ally_shot`, `kill_ally_assist` |
| 5 | `damage_taken` | − | damaged ship, all sources | `combat_damage_taken`, `boundary_damage` |
| 5 | `damage_dealt_enemy` | + | attacker | same |
| 5 | `damage_dealt_ally` | − | friendly-fire attacker | same |
| 5 | `shield_recharge` | + | healed ship | same (charge half moved out) |
| 5 | `healed_damage` | − | attackers whose damage was healed | new |

Removed outright: `ally_win`, `enemy_win`, the old `outcome` probe, `boundary`,
`boundary_damage`, and the unused shaping and `ally_/enemy_combat_*` components
(currently weight 0).

### 1.4 Level 1 — outcome (pure 3-way categorical)

- Classes **W / T / L**. Utility: win = 1, **tie = ½**, loss = 0 — equivalently +1 / 0 / −1.
  The scalar value is `V = P(W) − P(L)` on the ±1 scale, scaled by `win_weight`.
- Read from the global token by `GlobalValueHead`; advantage broadcast to every allied ship.
- **Discounting.** Run 737 showed γ = 1 bootstrapping collapsing (win explained variance
  0.994 → 0.42), so γ stays 0.9997. The discount is expressed as probability mass leaking
  into the class worth 0 on the ±1 scale, which is T. The categorical λ-target is built
  backwards like GAE:

  ```
  Q_t = γ · [ (1−λ) · p(s_{t+1}) + λ · Q_{t+1} ] + (1−γ) · e_T
  Q_t = onehot(result)                     on the terminal transition
  ```

  Its expectation equals the scalar λ-return, so the scalar GAE advantage and the
  categorical critic target agree exactly. Loss: cross-entropy against `Q_t`.

### 1.5 Level 2 — zone capture events

Per active zone, keep two ledgers over the current capture attempt, with progress `P`
running from 0 to 1:

- **attacker presence** `C_i`: on each tick the meter moves by `Δ > 0` toward the
  attacker, `C_i += Δ / n_present` for each present attacking ship;
- **defender absence** `A_j`: on the same tick, `A_j += Δ / n_absent` for each absent
  defending ship. Dead ships count as absent.

So `Σ C_i = Σ A_j = P`. On a reversal by `r`, both ledgers scale pro rata:
`C_i, A_j ← · (1 − r/P)`. When the meter completes:

```
zone_gain_i = ratio · w2 · [ ½ · present_i / n_present  +  ½ · C_i / P ]
zone_loss_j =       − w2 · [ ½ · absent_j  / n_absent   +  ½ · A_j / P ]
```

Both ledgers clear on completion and when roles rotate. The final-tick half reuses the
current non-empty fallback: if the charged set is empty, charge the whole opposing side.

### 1.6 Level 3 — live zone progress

The same per-tick split as today's `_ZoneCreditReward` (present side paid `ratio ×` and
shared among the present ships; absent side charged and shared among the absent ships,
dead included), but written into four components keyed on zone ownership and meter
direction (§1.3). A zone is captured, reset or idle on any tick, never two at once.

### 1.7 Level 4 — kills and deaths

- **`death`:** −`U` to the dying ship, any cause.
- **Damage ledger:** the existing attacker→target matrix `O[a,t]` holds outstanding
  applied damage, cleared at respawn as today. When target `t` recharges by `h`, with
  `S_t = Σ_a O[a,t]`:

  ```
  O[a,t] ← O[a,t] · (1 − h / S_t)          (pro rata; no FIFO queue)
  ```

- **`kill_enemy` / `kill_ally`** for a death of `t`:

  ```
  share_a = ½ · raw_final[a,t] / Σ_a raw_final[·,t]  +  ½ · O[a,t] / S_t
  ```

  `raw_final` is the raw impact on the death tick. The killing hit lands on zero shield,
  so its applied damage is 0. Paid `±kill_payout_ratio · U · share_a`. A death with no
  enemy cause (boundary) pays `kill_enemy` to the opposing team by `enemy_team_share`. A
  friendly-fire kill pays its counterpart to the opposing team, as today. A ship is never
  blamed for its own death.
- GAE λ for the merged component: start at 0.95, between the old 0.87 (shot) and 0.97
  (assist).

### 1.8 Level 5 — damage

- **`damage_taken`:** −`V · applied` from every source, boundary included. This is what
  blocks recharge farming at the boundary.
- **`damage_dealt_enemy`:** `+damage_payout_ratio · V` per unit, shared among simultaneous
  hits by applied damage as today. Boundary damage pays this to the opposing team by
  `enemy_team_share`.
- **`damage_dealt_ally`:** `−damage_payout_ratio · V` per unit to the friendly-fire shooter.
  Its counterpart is paid to the opposing team.
- **`shield_recharge`:** `+V · h` to the healed ship.
- **`healed_damage`** (charge-back): `−damage_payout_ratio · V · h · O[a,t] / S_t` to each
  attacker, taken before the ledger update above. It exactly reverses that attacker's
  payment for the damage that no longer exists. Recharge of unattributed damage charges
  the opposing team by `enemy_team_share`.

### 1.9 Weights and schedules

`component_weights` is rewritten around the same event weights (`win_weight`,
`death_weight`, `damage_weight`, `capture_progress_weight`, `front_advance_weight` →
`zone_event_weight`) and the three payout ratios. `kill_shot_fraction` is fixed at 0.5 by
the attribution rule and removed. Tier map: outcome and levels 2–3 → `outcome_scale`,
level 4 → `kill_death_scale`, level 5 → `damage_scale`.

---

## 2. Next-state head and state representation

### 2.1 General

- The network reads **probabilities**, never logits.
- The head outputs **residual logits** on a baseline built from the current
  (believed or true) state, and is trained with cross-entropy on the absolute
  distribution:

  ```
  z_{t+1} = b_t + f(h_t),    b_t = log p̃_t,    loss = −Σ q_{t+1} log softmax(z_{t+1})
  ```

  `f`'s last layer is zero-initialised, so the head starts as "nothing changes".
  `p̃` is floored by ε (0.01–0.05 for small categoricals).
- Targets are built on the fly from truth and never stored.
- RoPE stays on physical coordinates: truth for visible ships, decoded `μ̂` for hidden
  ones.
- The 13 Gaussian uncertainty terms are removed. `time_since_observation` stays.

### 2.2 Position — 12-level nested 5-colour code

- Base `2+i` in the Gaussian integers. The level-0 lattice is
  `L₀ = (65 536 / (2+i)) · ℤ[i]` (spacing 29 309 px, rotated −26.57°), so level 0's 5
  colours tile the 65 536 px torus exactly once. **Every point on the map has a unique
  code**, and every level wraps cleanly on the torus.
- `L_ℓ = L₀ / (2+i)^ℓ`, spacing `d_ℓ = 29 309 · 5^(−ℓ/2)`. Colour = class modulo
  `L_{ℓ−1}`, so a level's colours repeat every `d_{ℓ−1} = √5·d_ℓ`.
- Each level is a bilinear ≤4-hot over the 4 corners of its cell; those corners always
  have distinct colours. 12 × 5 = **60 logits**.

| ℓ | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d_ℓ (px) | 29 309 | 13 107 | 5 862 | 2 621 | 1 172 | 524 | 234 | 105 | 47 | 21 | 9.4 | 4.2 |

The finest cell, 4.2 px, matches cruise motion per decision (3.3 px at 100 px/s, §A.4), so
small moves still shift bilinear weights enough to train on. With 9.4 px a 1 px move is
about 0.02 nats; with 4.2 px it is about 4× stronger.

**Targets.**
- Ship **visible at t** (input is truth): fully sharp encoding of the true next position.
- Ship **hidden at t**: encoding of the true next position blurred by the realised error
  `e = |truth_{t+1} − μ̂_t|`. Per level: no blur while `e ≤ d_ℓ`; ramp to uniform by
  `e ≈ 1.5·d_ℓ`, past the ambiguity radius `≈ 1.12·d_ℓ`. Implement it as a Gaussian blur
  of width `e`, by Monte Carlo (about 8 samples) or by the λ curve below, with the
  deadzone applied per level.

**Belief, decode and re-encode.**
- The belief stores `μ̂` (2 numbers) and `σ̂` (1 number) per ship. On observation, reset
  `μ̂ = truth` and `σ̂ = floor`.
- **λ curve.** The code is self-similar and the blur isotropic, so blurring level ℓ by σ
  is approximately `uniform + λ(σ/d_ℓ)·(sharp − uniform)`. Tabulate the single curve λ
  once.
- **Decode.**
  1. `μ̂` from the cross-level projection `argmin_x D(p, F(x))`, solved coarse to fine with
     closed-form `u, v` inside each cell. Use a squared-error `D`: cross-entropy snaps
     estimates to cell corners.
  2. Per level, `λ̂_ℓ` = projection of `p_ℓ − uniform` onto `F_ℓ(μ̂) − uniform`.
  3. `σ̂` from weighted least squares of `log λ̂_ℓ` against the λ curve. Closed form.
- **Next input:** the blurred encoding `B_σ̂(μ̂)`, which is also the residual baseline. An
  uncertain fine level already has a near-uniform baseline, so the head never has to
  cancel a sharp pattern, and every level keeps its residual connection.

### 2.3 Velocity — symlog with a knee, three triangle axes

- `u = v̂ · v₀ · log(1 + |v| / v₀)`, with **v₀ ≈ 100 px/s**. The encoding is roughly linear
  across the playing band and logarithmic only in the unlimited-boost tail. Cells are
  nearly isotropic (about 1.4× elongation, against 4.6× with today's knee at 1 px/s).
- Project `u` onto unit vectors at **0°, 120° and 240°**. Each projection is a 1D two-hot
  over **81 bins** with spacing 5 u-units, covering `|u| ≤ 199` (631 px/s): **243
  logits**.
- Physical spacing (px/s, along / across the velocity): 6/6 at 20 px/s, **10/7 at
  100 px/s**, 16/9 at 215 px/s, 37/16 at 631 px/s. Today's knee at 1 px/s with spacing 1.5
  gives 152/33 at 100 px/s.
- Sharp targets; HL-Gauss smoothing (σ about ½ bin) optional, to restore a sense of
  distance between bins. The belief stores the three marginals. Decode by least squares;
  the leftover from the three-way fit is a consistency check.
- Rejected for V1: one flat hex grid (about 4 900 categories at this resolution), nested
  7-colour hex (wraps at fine levels, needs gating), and polar (measured worse earlier).

### 2.4 Other channels — store the distribution

These are ordinary 1D histograms with no aliasing. Use sharp targets, store the predicted
distribution in the belief, and feed it back. The baseline is `log p` of what was stored.

| Channel | Bins | Notes |
|---|---|---|
| attitude | 72, circular, 5° | attitude = velocity heading + turn offset |
| angular velocity | 5 | exactly `turn_offset/dt` ∈ {0, ±5°, ±15°}/dt (`physics.py:160`); consider tying it to the enemy-action head's turn marginal |
| health (shield) | 11, 0–100 | hits are 1–10 hp; recharge 15 hp/s after a 5 s delay |
| power | 21, step 5 | |
| shield delay | ~21 over 0–5 s | deterministic countdown, reset on damage |
| cooldown | **4** | 0.1 s = 3 ticks at Frontline's 30 Hz (`frontline.py:136`) |
| local log n | 5 | the five bulk index levels; two-hot inside transitions |

Belief memory is about 380 floats per ship.

---

## 3. Critic (stage 2)

- Keep scalar BF16 returns in the rollout and keep the `ReturnScaler`. Convert the
  normalised scalar return to a categorical target only when computing the critic loss.
- **HL-Gauss** target over **21 bins**, symlog-spaced and dense near zero, with tails to
  about ±10 normalised units so sparse components are not clipped.
- Value = expectation over bins. Persistent memory about 0.
- Level 1 is already categorical (§1.4).

---

## 4. Map-scale randomisation

- **Only the map size changes.** Ship count, token count, episode length (9 000 steps),
  physics, vision and rewards are all fixed.
- Per episode, sample a scale `s` and apply it through `scaled_frontline_geometry`
  (playable radius, zone radius, zone ring, field sizes and widths). Those quantities
  already live per environment in `TensorState`, so `s` is sampled at reset per env.
- **Range:** `s ∈ [1, s_max]`, with `s_max = 12` (playable radius 31 200 px, under the
  32 768 px half-period the geometry check enforces).
- **Distribution, heavily favouring the standard map:** with probability 0.5, `s = 1`;
  otherwise `s = exp(x)` with `x` drawn from an exponential of rate 1.5, truncated to
  `[0, ln s_max]`.

| | P |
|---|---|
| s = 1 exactly | 0.50 |
| s ≤ 1.25 | 0.65 |
| s ≥ 2 | 0.17 |
| s ≥ 3.16 (50v50 eval scale) | 0.08 |
| s ≥ 6 | 0.023 |
| s ≥ 10 | 0.004 |

Rate 2.0 instead gives 0.047 at the eval scale and 0.010 at s ≥ 6.

- Expect timeouts to rise with `s`: at s = 12 the zone ring radius is 14 400 px, about
  106 s at energy-limited cruise against a 300 s episode. The rare large maps mainly train
  long occlusions, coarse-level belief corrections and the low-frequency RoPE pairs.
- Log every per-update metric bucketed by `s` (below).

---

## 5. Diagnostics to add

- Position cross-entropy per level, split visible vs hidden. Bucket hidden by realised
  error. Watch level 11 for fast visible ships: if it stays at the baseline loss, gate it
  by speed.
- `σ̂` calibration against realised position error, by hidden age.
- Velocity error per axis and decoded, against the current Gaussian head and dead
  reckoning.
- Zone occupancy, front advances and timeout rate (the run 735 collapse signature),
  bucketed by map scale.
- Per-component gradient share after the reward refactor. Zero-sum residual per component
  at ratio 1.
- Outcome head calibration: predicted P(W/T/L) against realised outcomes.

---

## 6. Build order

One variable at a time, each A/B'd against the step before it:

1. **Rewards** (§1), including lambda-matrix removal. The critic stays scalar except the
   categorical outcome.
2. **Next-state head and state inputs** (§2). Critic still scalar.
3. **Categorical critic** (§3).
4. **Map-scale randomisation** (§4).

Each step changes the checkpoint schema; no weight migration.

---

## Appendix A. Physics calibration

Measured at 60 Hz (dt = 1/60 s), in px/frame unless noted. Frontline runs at
**dt = 1/30** (`frontline_ship_config`), so px/s values carry over but per-tick values
double per decision. They should be re-checked at 30 Hz once.

### A.1 Steady-state speed (px/frame; ×60 for px/s)

| n | Coast straight | normal | sharp | Boost (energy-limited) straight | normal | sharp | Boost (unlimited) straight | normal | sharp |
|---|---|---|---|---|---|---|---|---|---|
| 0.500 | 3.331 | 2.715 | 1.310 | 3.586 | 3.131 | 1.940 | 10.519 | 8.576 | 4.138 |
| 0.707 | 2.356 | 1.920 | 0.926 | 2.847 | 2.486 | 1.541 | 7.442 | 6.068 | 2.927 |
| 1.000 | 1.666 | 1.358 | 0.655 | 2.260 | 1.974 | 1.224 | 5.265 | 4.292 | 2.071 |
| 1.414 | 1.178 | 0.960 | 0.463 | 1.794 | 1.567 | 0.972 | 3.724 | 3.036 | 1.465 |
| 2.000 | 0.833 | 0.679 | 0.328 | 1.424 | 1.244 | 0.772 | 2.634 | 2.147 | 1.036 |

Anchors: n=1 coast cruise 100 px/s; n=1 energy-limited boost 136 px/s; highest
energy-limited 215 px/s (n=0.5); highest unlimited 631 px/s (n=0.5); lowest sustained
sharp-turn coast 19.7 px/s (n=2). **Normal play spans about 20–215 px/s.** `max_speed=180`
in `ShipConfig` is not a cap.

### A.2 Turning

- Sustained turn radius, independent of n and power: **normal 66.2 px, sharp 35.3 px**.
- Largest sustained angular rate: 7.40°/frame = 444°/s (unlimited boost, n=0.5, normal
  turn).
- Commanded turn offsets: 0, ±5°, ±15°. Attitude = velocity heading + offset. Angular
  velocity = offset/dt, giving 5 values.
- One-tick lateral acceleration model (median error 1.3%, worst 8.6%):
  `a_lat ≈ lift·v² + thrust·sin(offset)/n²`, with lift 0.015 (normal) / 0.027 (sharp),
  coast thrust 8 px/s², boost thrust 80 px/s².

### A.3 Acceleration and response

- Largest sustained lateral acceleration: energy-limited 0.148 px/frame² (533 px/s²,
  n=0.5, normal); unlimited 1.109 px/frame² (3 992 px/s²).
- Initial unlimited boost acceleration `a₀ = 1/(45 n²)` px/frame²: at most 0.0889
  (320 px/s², n=0.5).
- Unlimited boost response `v(t) = (5.270/n)·tanh(t / (237 n))` px/frame; time constant
  `τ = 3.95 n` s (n=1: 50/90/99% at 130/349/627 frames).
- No-energy / passive regen response (n=1): terminal 2.26 px/frame; 50/90/99% at
  74/360/782 frames. Full thrust ends at 25 px/s; afterwards
  `a ≈ 0.00926/(n² v) − drag`.

### A.4 Per-decision quantities at 30 Hz (sizing inputs)

| Quantity | Value |
|---|---|
| Displacement per decision | 0.7 px (20 px/s), **3.3 px (100 px/s)**, 7.2 px (215 px/s), 21 px (631 px/s) |
| Max forward Δv per decision | ~10.7 px/s (320 px/s²) |
| Max lateral Δv per decision | ~17.8 px/s energy-limited; ~133 px/s unlimited (extreme) |
| Measured Δv per decision (ally, Sept 30 pilot) | RMS ~3.9 px/s, p99.9 ~18.6 px/s |
| Cooldown | 3 ticks (0.1 s) |
| Shield recharge | 5 s delay (150 ticks), then 0.5 hp per tick |

### A.5 Geometry

- World torus 65 536 × 65 536 px; map centre uniformly random per episode.
- 5v5 standard: playable radius 2 600 px, zone ring 1 200 px, zone radius 330 px.
  Lengths scale with `s = √(ships/10)` for fixed density (50v50: s = 3.16, playable
  radius 8 222 px).
- Vision range 1 024 px.
