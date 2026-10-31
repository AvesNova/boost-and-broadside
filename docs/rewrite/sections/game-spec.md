# Game specification (v2, section 1)

Derived from the physics race's SPEC v3 (`physics-race`, branch `v3-general`). The v2
changes are marked **(v2)** where they occur:
- friendly fire, never the shooter (7.1);
- bullet drag on proper speed (6.2);
- a shuffled and rotated spawn formation (8);
- point-symmetric maps (9);
- per-tick spawn randomness (2);
- no bullet perception (10);
- v1 combat defaults (1).

The decisions behind them are in `game.md`.

This document is the only description of the game you are implementing. It is
mathematical on purpose: implement it in whatever form serves the framework best, as long
as the results match. Where a formula is marked **normative**, reproduce that exact
arithmetic (same approximations, same order of operations up to floating-point
reassociation). The acceptance tests compare your results with a float64 reference
implementation of this text.

Conventions: SI units (m, s, kg, J, W, rad). The plane is the standard x/y plane with y
up; angles are counter-clockwise; "left" means counter-clockwise. A vector written
`(a, b)` is `a x̂ + b ŷ`. `×` between 2D vectors is the scalar `a × b = a_x b_y − a_y b_x`.
`clip(v, lo, hi)` clamps. Every quantity is float32 in the implementation unless stated
otherwise; integer counters are int32.

## 1. Sizes and presets

A batch holds `B` independent environments. Each has `N` ships (even; ships `0 … N/2−1`
are team 0, the rest team 1), `M` circular fields, and `K` bullet slots per ship.

| preset | N (`num_ships`) | M (`num_fields`) | arena radius `R_a` (m, `arena_radius`) |
|---|---:|---:|---:|
| `duel` | 2 | 4 | 1200 |
| `squad` | 10 | 10 | 2600 |
| `battle` | 100 | 40 | 8200 |

The map scale is `s_a = R_a / 2600`. All other constants are the same for every preset.
Constants marked † differ by ship type (section 1.1); the value shown is the fighter's.

| constant | symbol | value | key in `tests/reference/golden/presets.json` |
|---|---|---|---|
| tick | `h` | 1/30 s | `h` |
| episode length | `T_ep` | 1800 ticks | `episode_ticks` |
| bullet slots per ship | `K` | 6 | `slots` |
| ticks between shots † | — | 2 | `fire_interval` |
| bullet lifetime | — | 12 moves | `bullet_moves` |
| muzzle proper speed † | `w_m` | 500 m/s **(v2)** | `muzzle_speed` |
| bullet drag **(v2)** | `k_b` | 8e-4 1/m | `bullet_drag` |
| spread (per axis, proper) | `σ_b` | 12 m/s | `spread` |
| energy per bullet † | `c_b` | 300 J **(v2)** | `bullet_cost` |
| hit radius † | `r_hit` | 8 m | `hit_radius` |
| damage per hit † | `D` | 12 | `damage` |
| head-on damage fraction | `f_0` | 0.3 **(v2)** | `head_on_fraction` |
| shield maximum † | `S_max` | 100 | `shield_max` |
| shield recharge delay | — | 6 s | `shield_delay` |
| shield recharge rate | — | 12 per s | `shield_regen` |
| boundary damage | — | 5 + 0.05 · depth (m) per s | `boundary_base`, `boundary_per_m` |
| respawn wait | — | 240 ticks | `respawn_ticks` |
| spawn proper speed | — | 30 m/s | `spawn_speed` |
| spawn energy † | — | 4000 J | `spawn_energy` |
| spawn offset | — | 0.8 `R_a` | `spawn_offset` |
| spawn disc radius | `ρ_s` | 120 `s_a` m | `spawn_radius` (before scaling) |
| field band width | `b` | 40 m | `field_width` |
| vision range | `V` | 1024 m | `vision` |

The flight constants are in section 3.2. The same table is in machine-readable form in
`tests/reference/golden/presets.json`.

### 1.1 Ship types

Every ship has a type, `ship_type`: 0 **fighter** or 1 **heavy**. Within each team, slots
alternate starting with a fighter: ship `i` is a heavy when `i − (first slot of its team)`
is odd (the first slot is 0 for team 0 and `N/2` for team 1). The type is fixed for the
whole game, through respawns and resets.

Wherever this text uses a constant marked †, it means the value of the relevant ship's
type: the shooter's for everything about firing and the bullet's damage, the target's for
its hit radius, and the ship's own for its flight, shield and spawn energy. The heavy's
values (all others are shared; `tests/reference/golden/presets.json` → `types` → `heavy`):

| constant | fighter | heavy |
|---|---:|---:|
| `C_max` (`cl_max`) | 0.027 1/m | 0.022 1/m |
| `a_max` | 216 m/s² | 160 m/s² |
| `C_D0` (`cd0`) | 6.9612e-4 1/m | 8.5e-4 1/m |
| `P` (`power`) | 1200 W | 1500 W |
| `P_min` (`p_min`) | 863.2 W | 1100 W |
| `F_s` (`f_static`) | 20 N | 24 N |
| `E_max` (`e_max`) | 10 000 J | 14 000 J |
| `ω_c` (`omega_c`) | 22.0 rad/s | 16.0 rad/s |
| shield maximum (`shield_max`) | 100 | 160 |
| hit radius (`hit_radius`) | 8 m | 11 m |
| damage per hit (`damage`) | 12 | 16 |
| muzzle proper speed (`muzzle_speed`) | 500 m/s | 450 m/s |
| energy per bullet (`bullet_cost`) | 300 J | 360 J |
| ticks between shots (`fire_interval`) | 2 | 3 |
| spawn energy (`spawn_energy`) | 4000 J | 5600 J |

Derived flight quantities (the corner speed `u_c`) follow the ship's own constants.

### 1.2 Custom sizes

Besides the three presets, a game can have any size: `N` even and at least 2, `M` from 1
to 64, and any `R_a > 0`, given as `{num_ships, num_fields, arena_radius}`. Every other
constant is the shared value of the table above (and section 1.1), `s_a = R_a / 2600`
scales the spawn disc and the field radii as for the presets, and `K` stays 6. Nothing in
sections 2-10 depends on which preset a game came from.

## 2. The tick

`step(state, actions) → (state', events, observation)` advances every environment by one
tick. `actions` is `(B, N, 3)` integers: power ∈ {0 coast, 1 boost, 2 reverse},
stick ∈ {0 … 6}, shoot ∈ {0, 1} (42 joint actions). The tick runs these phases in order;
"at tick start" means the value in `state`.

1. **Fire** (section 6.1), from the tick-start state.
2. **Ships fly** (section 3). Only ships alive at tick start move; dead ships keep every
   flight variable unchanged.
3. **Bullets fly** (section 6.2). Every bullet with `ttl > 0` moves, including any fired
   in phase 1.
4. **Hits** (section 7.1), between the bullet and ship motions of this tick.
5. **Damage, death and shields** (section 7.2).
6. **Clocks:** each bullet that was active in phase 3 has `ttl ← ttl − 1`; a bullet that
   hit gets `ttl ← 0`. Every ship has `gun_wait ← max(gun_wait − 1, 0)`.
7. **Death and respawn** (section 8).
8. **Episode clock:** `t ← t + 1`. An environment with `t ≥ T_ep` is reset (section 9)
   and reports `done`.
9. **Observation** (section 10), from the final state (after any reset).

Randomness enters in exactly two places **(v2)**:
- **The bullet spread:** phase 1, two standard normals per ship per tick, drawn whether
  or not the ship fires.
- **The tick uniforms:** `U = 4M + 1 + 2 + N` uniforms in [0, 1) per environment per
  tick, laid out as
  - `u[0 … 4M]`: the map, used only by environments that reset (section 9);
  - `u[4M+1], u[4M+2]`: the spawn formation rotation of team 0 and team 1;
  - `u[4M+3 … 4M+2+N]`: one formation key per ship.

  The rotations and keys are used by any ship spawning this tick, by respawn or reset
  (section 8).

The test interface passes these in explicitly; production code draws them itself.

## 3. Ship flight

### 3.1 State

Per ship:

| name | meaning |
|---|---|
| `x, y` | position (m), map frame, no wrap |
| `c, s` | nose heading, unit vector (cos, sin) |
| `w` | proper speed (m/s), `w = n · |v|` |
| `alpha` | angle of attack (rad): the flight path is the nose rotated by `−alpha` |
| `ca, sa` | cos and sin of `alpha`, carried alongside it |
| `r` | nose rotation rate (rad/s) |
| `E` | energy store (J), in [0, `E_max`] |
| `logn, gx, gy` | `ln n` and `∇ ln n` at the ship, cached from the last field evaluation |

The world (coordinate) velocity is `v = (w/n) · p̂` with the path direction
`p̂ = (c·ca + s·sa, s·ca − c·sa)`.

### 3.2 Constants

| symbol | value | meaning | key in `presets.json` → `flight` |
|---|---|---|---|
| `m` | 1 kg | mass | `mass` |
| `g` | 27 m/s² | lift needed to hold level flight | `g` |
| `C_max` | 0.027 1/m | maximum lift coefficient | `cl_max` |
| `α_s` | 15° | stall angle of attack | `alpha_s` (rad) |
| `a_max` | 216 m/s² | g limit | `a_max` |
| `C_D0` | 6.9612e-4 1/m | zero-lift drag | `cd0` |
| `k` | 11.4 | induced-drag factor | `k_ind` |
| `P` | 1200 W | engine power below the knee | `power` |
| `P_min` | 863.2 W | engine power at a full store | `p_min` |
| `e_k` | 0.5 | store fraction where engine power starts to fall | `e_knee` |
| `F_s` | 20 N | static-thrust cap | `f_static` |
| `E_max` | 10 000 J | energy store | `e_max` |
| `β` | 0.02 | store-end fade band (fraction of `E_max`) | `band` |
| `a_0` | 3 m/s² | coast exchange | `a0` |
| `a_c` | 27 m/s² | reverse (climb) exchange | `a_climb` |
| `a_d` | 27 m/s² | boost (dive) exchange | `a_dive` |
| `p` | 16 | soft-min exponent of the lift budget | `soft_p` |
| `ω_c` | 22.0 rad/s | nose frequency at corner speed | `omega_c` |
| `ζ` | 0.7 | nose damping ratio | `zeta` |
| `ω_min` | 1.0 rad/s | nose frequency floor | `omega_floor` |
| `ε` | 1e-6 m/s | speed floor | `eps` |

Derived: corner speed `u_c = sqrt(a_max / C_max)`.

`C_max`, `a_max`, `C_D0`, `P`, `P_min`, `F_s`, `E_max` and `ω_c` differ by ship type
(section 1.1); every formula in sections 3.3-3.6 uses the flying ship's own values.

Stick levels by index: `σ = [0, −0.4, +0.4, −0.75, +0.75, −1, +1]` (positive turns left).
Power mode by index: `π = [0, −1, +1]` (coast, boost, reverse).

### 3.3 Functions

- Soft minimum: `softmin_q(a, b) = a · (1 + (a/b)^q)^(−1/q)` for `a ≥ 0`. Implementations
  must stay finite for proper speeds up to 1000 m/s.
- Engine fall: `φ(E) = t²(3 − 2t)`, `t = clip((E/E_max − e_k)/(1 − e_k), 0, 1)`.
- Thrust: `T(E, u) = softmin_4(P(E)/u, F_s) / m`, `P(E) = P − (P − P_min) φ(E)`.
- Lift coefficient: `C_L(α) = C_max (2x − x|x|)`, `x = clip(α/α_s, −2, 2)`.
- Lift slope: `C_L'(α) = (C_max/α_s)(2 − 2|α/α_s|)` if `|α/α_s| < 2`, else 0.
- **Normative small-angle forms** (used wherever this text writes `cs(θ)`):
  `cos θ ≈ 1 − θ²(1/2 − θ²(1/24 − θ²/720))`,
  `sin θ ≈ θ(1 − θ²(1/6 − θ²(1/120 − θ²/5040)))`.
- **Normative 2×2 exponential** `expm(A, h)`: with `B = A h/2`, form
  `H = I + B(I + (B/2)(I + (B/3)(I + B/4)))` (Horner, innermost first) and return `H·H`.

### 3.4 The flight setting

`setting(u, E, σ, π)` gives the lateral lift `ℓ`, the exchange acceleration `a`, the
vertical lift `z` and the thrust `T`. All four are held fixed over each half kick.

```
a_cmd   = a_0 (1 − φ(E))   if π = 0
        = +a_c             if π > 0
        = −a_d             if π < 0
L       = softmin_p(C_max u², a_max)                         lift budget
ℓ_c     = sign(σ) · softmin_p(|σ| C_max u², a_max)            commanded lateral lift
N       = sqrt(ℓ_c² + g²)
q       = softmin_p(N, max(L, 1e-12)) / max(N, 1e-12)        share of the command flown
ℓ       = ℓ_c q,   z = g q
χ       = clip(|ℓ| / max(L, 1e-12), 0, 1)
a       = a_cmd · sqrt(1 − χ²) · q − (1 − q) · a_d
a       = a · clip((E_max − E)/(β E_max), 0, 1)    if a > 0
        = a · clip(E/(β E_max), 0, 1)              otherwise
T       = T(E, u)
```

Rates, for a speed `ũ`, lift coefficient `C`, `cos α` value `ĉ`, and `1/u²` value `ι`
(the arguments are chosen in 3.5):

```
drag(ũ, C, ι) = (C_D0 + k (C² + (z ι)²)) ũ²
du/dt = (T ĉ − drag − a) / n
dE/dt = m ũ a / n
```

### 3.5 The kick (normative)

`kick(state, n, ∇ln n, σ, π, Δ)` advances `w, α, ca, sa, r, E` by `Δ` with the pose
held. Write `u₀ = w`, `α₀ = alpha`, `(ĉ₀, ŝ₀) = (ca, sa)`, and let `(γ_x, γ_y)` be `∇ln n`
in the nose frame: `γ_x = c g_x + s g_y`, `γ_y = −s g_x + c g_y`.

1. `ū₀ = max(u₀, ε)`; first setting `(·, a₁, z₁, T₁) = setting(ū₀, E, σ, π)`.
2. With `C₀ = C_L(α₀)`: `(u̇₀, Ė₀) = rates(ũ = u₀, C₀, ĉ₀, ι = 1/ū₀², setting 1)`.
3. Mid speed `u = max(u₀ + ½Δ u̇₀, ε)`; mid store `E_m = E + ½Δ Ė₀`; second setting
   `(ℓ, a, z, T) = setting(u, E_m, σ, π)`. Steps 4-9 use the second setting.
4. Path turn rate `θ̇₀ = (C₀ u² + T ŝ₀)/(u n) + (u/n)(ĉ₀ γ_y + ŝ₀ γ_x)`.
5. Path-rate slope `κ = (C_L'(α₀) u² + T ĉ₀)/(u n)`.
6. Nose frequency `Ω = max(ω_c u/u_c, ω_min) / n`.
7. Commanded angle of attack (inverts the lift curve): `c = clip(|ℓ|/(C_max u²), 0, 1)`,
   `Δα = sign(ℓ) α_s (1 − sqrt(1 − c)) − α₀`.
8. Nose oscillator, solved exactly for the linearized system:
   `A = [[−κ, 1], [−Ω² + 2ζΩκ, −2ζΩ]]`, `[[m₁₁, m₁₂], [m₂₁, m₂₂]] = expm(A, Δ)`,
   `e_s = Δα`, `z_s = κ Δα`, `y₁ = −e_s`, `y₂ = (r − θ̇₀) − z_s`,
   `e = e_s + m₁₁ y₁ + m₁₂ y₂`, `r' = θ̇₀ + z_s + m₂₁ y₁ + m₂₂ y₂`.
9. Mid attitude: `(ch, sh) = cs(e/2)`, `ĉ_m = ĉ₀ ch − ŝ₀ sh`, `C_m = C_L(α₀ + e/2)`.
   Two-stage speed and store, with `ι = 1/u²` from step 3:
   `(u̇₁, ·) = rates(u₀, C_m, ĉ_m, ι)`;
   `(u̇₂, Ė₂) = rates(max(u₀ + ½Δ u̇₁, ε), C_m, ĉ_m, ι)`.
10. Outputs: `w' = max(u₀ + Δ u̇₂, ε)`, `alpha' = α₀ + e`, `(c₁, s₁) = cs(e)`,
    `ca' = ĉ₀ c₁ − ŝ₀ s₁`, `sa' = ŝ₀ c₁ + ĉ₀ s₁`, `r'`, `E' = clip(E + Δ Ė₂, 0, E_max)`.

### 3.6 One flight tick (normative)

With the cached `n = exp(logn)` and `∇ln n`:

1. `kick(Δ = h/2)`.
2. Drift. Body-frame velocity `(v_x, v_y) = (w ca / n, −w sa / n)`. Correct it to the
   index at the step midpoint: `ξ = (h/2)(γ_x v_x + γ_y v_y)`,
   `(v_x, v_y) ← (v_x, v_y) / (1 + ξ(1 + ξ(1/2 + ξ/6)))`. Exact arc of the nose
   rotation: `φ = r h`, `A₁ = (1 − φ²(1/6 − φ²/120)) h`, `A₂ = φ(1/2 − φ²(1/24 − φ²/720)) h`,
   `(b_x, b_y) = (v_x A₁ − v_y A₂, v_x A₂ + v_y A₁)`, `x ← x + c b_x − s b_y`,
   `y ← y + s b_x + c b_y`, `(c_φ, s_φ) = cs(φ)`, `(c, s) ← (c c_φ − s s_φ, s c_φ + c s_φ)`.
   Renormalizing `(c, s)` and `(ca, sa)` afterwards is allowed.
3. Evaluate the medium at the new position (section 4) and update `logn, gx, gy`.
4. `kick(Δ = h/2)` with the new `n` and `∇ln n` (rotated into the new nose frame).

## 4. Fields

Field `i` has a center `(f_x, f_y)`, a radius `R_i`, a band width `b_i` (40 m) and a level
`ℓ_i ∈ {−2, −1, +1, +2}`; its target log index is `L_i = ℓ_i ln √2` (so `n` spans ½ to 2).

### 4.1 Profile

For a point at distance `d` from the center: `ζ = ½ − (d − R_i)/b_i`,
`z = clip(ζ, 0, 1)`, `α_i = 6z⁵ − 15z⁴ + 10z³`. Its gradient is
`∇α_i = −30 z²(z − 1)² / b_i · (p − f)/d` when `0 < ζ < 1` and `d > 1e-6`, else 0.

### 4.2 Union composition

Fields may overlap, nest or coincide. With `W = Σα_i`, coverage `A = 1 − Π(1 − α_i)` and
weighted mean `μ = Σ α_i L_i / W` (0 where `W ≤ 1e-12`):

```
ln n = A μ
∇A   = Σ_i ∇α_i Π_{j≠i}(1 − α_j)          (leave-one-out products; never divide by 1 − α_i)
∇μ   = (Σ ∇α_i L_i − μ Σ ∇α_i) / W         (0 where W ≤ 1e-12)
∇ln n = μ ∇A + A ∇μ
```

With no fields, `ln n = 0` and `∇ln n = 0`.

### 4.3 What the medium does

Inside a field ships and bullets keep proper speed while their world speed is `w/n`, and
their direction bends toward higher `n` at rate `(w/n)(v̂ × ∇ln n)`. Sections 3.5-3.6 and
6.2 already contain this; nothing else is needed.

## 5. Arena

The arena is the disc of radius `R_a` around the origin. There is no wrap and no wall.
A living ship whose post-move distance from the origin exceeds `R_a` by `δ > 0` takes
`(5 + 0.05 δ) h` boundary damage this tick (section 7.2).

## 6. Bullets

### 6.1 Firing (phase 1)

A ship fires when it is alive, `shoot = 1`, `gun_wait = 0` and `E ≥ c_b`. Then
`E ← E − c_b`, `gun_wait ←` its ticks between shots, and its slot `cursor` receives a
bullet (`c_b`, the ticks between shots and `w_m` are the shooter's, section 1.1):

- position = the ship's tick-start position;
- proper velocity `W = w p̂ + w_m (c, s) + σ_b (ν_x, ν_y)` with the ship's tick-start
  state and the spread normals `ν`; the bullet stores its proper speed `w_b = |W|` and
  direction `d̂ = W/|W|`;
- `logn, gx, gy` copied from the ship; `ttl = 12`;

and `cursor ← (cursor + 1) mod K`. `fired` is set for this tick (false otherwise).

### 6.2 Bullet motion (normative)

Per active bullet, with `v = w_b / n`:

1. Half bend: rotate `d̂` by `cs((h/2) v (d̂ × ∇ln n))`.
2. Drift: `ξ = (h/2) v (d̂ · ∇ln n)`, position `+= d̂ v h / (1 + ξ(1 + ξ(1/2 + ξ/6)))`.
3. Evaluate the medium at the new position; update `logn, gx, gy`.
4. Half bend with the new values (`v = w_b / n` with the new `n`).

**(v2) Drag.** Before step 1, apply half a tick of drag with the start index:
`w_b ← w_b / (1 + k_b w_b (h/2) e^{−ln n})`. After step 4, apply half a tick with the new
index. This is the exact solution of `dw/dt = −k_b w² / n` (quadratic drag on proper
speed) for each half with `n` held. In steps 1, 2 and 4, `v = w_b / n` uses the current
`w_b`. A bullet from rest travels `ln(1 + k_b w_m T)/k_b ≈ 186 m` in its 0.4 s life.
Inactive bullets (`ttl = 0`) are frozen.

## 7. Combat

### 7.1 Hits (relative swept test)

For bullet `j` (owner `o`) active in phase 3 and ship `k ≠ o` alive at tick start, of
either team **(v2: friendly fire; a bullet never hits its own shooter)**: with `D₀ = p_j − q_k` at tick start and `D₁` the same after the
moves of this tick, `Δ = D₁ − D₀`,

```
τ = clip(−(D₀ · Δ) / max(|Δ|², 1e-12), 0, 1),   miss distance² = |D₀ + τΔ|²
```

The pair is a contact when the miss distance² is below `r_hit²`, with the target's hit
radius. A bullet hits only its
**first** contact: the smallest `τ`, ties to the lowest ship index. A bullet can hit an
ally of its shooter, never the shooter itself. A ship can be hit by several bullets in
one tick.

Damage of a hit: with `θ` the angle between `−d̂` (the bullet's post-move direction) and
the target's post-move nose `(c, s)`,
`D (1 − (1 − f_0) exp(−4θ²/π))` with the shooter's `D`: a tenth from dead ahead, nearly full from the side or
rear.

### 7.2 Damage, death and shields (phase 5)

For each ship alive at tick start, with `S⁰` its tick-start shield:

- `total` = sum of its hit damage + its boundary damage.
- It **dies** if `S⁰ ≤ 0` and it took a hit or boundary damage this tick (the exposed
  core: a ship that starts the tick with no shield dies to anything). In words:
  - damage beyond the current shield breaks it, and the overkill is ignored;
  - damage equal to the shield also breaks it;
  - any damage to an already broken shield kills.
- `S ← max(S⁰ − total, 0)`.
- Shield recharge: let `W⁰` be the tick-start `shield_wait`. If `total > 0`,
  `shield_wait ← 6`; else `shield_wait ← max(W⁰ − h, 0)`. If the ship took no damage and
  did not die, `S ← S + min(S_max − S, 12 · clip(h − W⁰, 0, h))` with the ship's own
`S_max`.
- `killer` = owner of the hit with the smallest `τ` among the bullets that hit it (ties
  to the lowest bullet index `o·K + slot`), when it died and was hit; else −1.

Ships dead at tick start keep `shield` and `shield_wait` unchanged.

## 8. Death and respawn (phase 7)

A ship that died this tick: `alive ← false`, `respawn ← 240`, `w ← ε`, `r ← 0`; it stays
where it is. A ship that was already dead at tick start: `respawn ← respawn − 1`; if it
reaches 0 it **spawns** now. `spawned` is true exactly for ships that spawned in this
tick (also every ship of a reset environment).

Spawning ship `i` **(v2: shuffled, rotated formation)**:
- **Spawn centre:** team 0 spawns around `−0.8 R_a â` and team 1 around `+0.8 R_a â`,
  where `â` is the environment's spawn axis.
- **Formation rank:** with `S = N/2` and the tick uniforms of section 2, ship `i`'s rank
  `k_i ∈ {0 … S−1}` is the rank of its key among its team's keys, ties to the lower slot.
- **Position:** with `ψ_T` its team's rotation, centre `+ ρ_s (cos 2πφ, sin 2πφ)` with
  phase `φ = k_i / S + ψ_T`.
- **Spawn state:**
  - nose pointing at the other team's spawn centre (`+â` for team 0, `−â` for team 1);
  - `alpha = 0`, `ca = 1`, `sa = 0`, `r = 0`;
  - the medium evaluated at the spawn point;
  - `w = 30 · n`;
  - `E` = its spawn energy, shield = its `S_max` (section 1.1);
  - `shield_wait = 0`, `gun_wait = 0`, `respawn = 0`, `alive = true`.

Ships spawning in the same tick take distinct, evenly spread places, and a ship
respawning alone lands at a uniformly random angle. Nothing about the formation depends
on the team's index.

## 9. Reset

A reset environment draws its map from the map part of its tick uniforms `u` **(v2:
point-symmetric)**. With `P = ⌊M/2⌋`:
- **Field `i < P`** (uniforms `u[4i … 4i+3]`): centre at distance `0.85 R_a sqrt(u₀)`
  and angle `2π u₁`; radius `s_a (30 + 460 u₂)`; band 40 m; level
  `[−2, −1, +1, +2][min(⌊4 u₃⌋, 3)]`.
- **Field `P + i`:** field `i` rotated by 180° about the centre (centre negated, same
  radius, band and level).
- **When `M` is odd:** field `M − 1` sits at the centre, with radius and level from
  `u[4(M−1)+2]` and `u[4(M−1)+3]`.

The other map uniforms are unused. The map is identical seen from either team (the spawn
centres are mirror images too). The spawn
axis is `â = (cos 2π u[4M], sin 2π u[4M])`. Then `t ← 0`, every bullet `ttl ← 0` (other
bullet values become 0), every `cursor ← 0`, `fired ← false`, and every ship spawns
(section 8).

## 10. Perception and observation

### 10.1 Visibility (after the tick)

Each field has an opaque core of radius `ρ_i = max(R_i − b_i/2, 0)` (a field too small to
have a flat interior has no core and blocks nothing). A sight line from point `a`
to point `p` is **blocked** by core `i` when the segment comes within `ρ_i` of its center
(closest point on the segment, strictly less than `ρ_i`), unless both `a` and `p` are
strictly inside that core. It is **clear** when no core blocks it.

- Ship `k` is visible to team `T` when `k` is on team `T`, or `k` is alive and
  (it fired this tick, or it spawned this tick, or some living ship of team `T` is within
  `V` of it with a clear sight line).
- **(v2)** Bullets are not perceived. They are not in the observation: a firing ship is
  visible anyway, and shot labels are rebuilt from stored truth.

Range tests use `distance² ≤ V²`.

### 10.2 Observation

**(v2)** The observation is the contract in `docs/rewrite/contracts.md` and
`src/bnb/contracts/obs.py`:
- per viewing team;
- rows ordered by `team_rel`;
- times in seconds;
- `since_seen` and `spawned` columns;
- no bullets.

Hidden ships show only `alive, visible, respawn, type, team_rel, since_seen, spawned`. The
reference tests compare it, through an adapter, with the race's layout (ships in sim
order with an `ally` column, fields, globals `t / T_ep, R_a`).

### 10.3 Events

Per tick: `damage (B, N, N)` hit damage from shooter to target; `boundary_damage (B, N)`;
`died (B, N)`; `killer (B, N)`; `fired (B, N)`; `done (B,)`.

## 11. Out of scope

Rewards, zones and objectives, ship-to-ship collisions, more than two teams, ship types
beyond the two of section 1.1, rendering and input.
