# Frontline redesign plan: flight model, rewards, critic and belief

Status: **agreed design, not implemented.** Base: `dev` at `4eba8a1`.

The plan has two parts:

* **Part I, physics.** The ship's nose moves continuously instead of jumping, a
  lateral-acceleration limit (a g-limit) gives the flight envelope a corner speed, and
  the flight constants are solved from handling targets rather than set by hand.
* **Part II, learning.** Five per-level reward heads with derived zero-sum weights; a
  categorical critic per head with DreamerV3's return normalisation and retuned PPO
  settings; a categorical next-state head and a belief stored as moments, with new
  position, velocity and attitude codes; per-episode map-scale randomisation.

Part I lands first and is validated with scripted play only. Part II is then built
against the new physics and trained from one cold start (§12).

The guiding rule for Part II is to follow published systems rather than invent:
DreamerV3 and Dreamer 4 for the critic and advantage plumbing, Honor of Kings for the
reward structure, OpenAI Five for the zero-sum construction and team mixing (§6.5).

---

## 1. Conventions

* **Tick.** Frontline runs at dt = 1/30 s with `action_repeat` 1, so a tick is a
  decision. Per-tick discounts follow `COMPONENT_GAMMAS_PER_TICK` in
  `config/defaults.py`; a horizon is 1 / (1 − gamma).
* **Proper and world speed.** The local field index n scales motion: a ship's proper
  speed is `u = n|v|`. Part I states speeds as proper speeds in px/s at n = 1, because
  handling is the same in every field in those units. Part II's codes encode world
  velocity and world turn rate, so their ranges are sized for the smallest index,
  n = 0.5, where world values are twice the proper ones.
* **Angles** are in degrees and rates in °/s unless stated. Values called starting
  points are meant to be tuned.
* Measured numbers for today's physics are in Appendix A; game constants used for
  sizing are in Appendix B.

---

# Part I. Physics

## 2. Goals

### 2.1 The target feel

The target is a responsive Second World War fighter:

* **Low and medium speed:** the nose can be thrown hard for a moment, a shot taken, and
  the nose snaps back to the flight path.
* **High speed:** the nose moves only a few degrees, and any real direction change is a
  committed, wide turn.
* **Very low speed:** little turning authority.
* **Turning fights trade energy.** Speed is not free turning power. Above corner speed
  a ship turns slower and wider, so it must spend speed (sharp-pull drag, reverse
  thrust) to turn its best.

### 2.2 Why the current model misses it

* **Turn radius is fixed:** 35 px sharp and 66 px normal at every speed, so turn rate
  grows with speed without limit. Faster is strictly better in every turning fight.
* **The nose jumps.** The gun moves 15° in one tick (450°/s), and 21–28° on a
  left-to-right reversal.
* **There is no stall in practice.** `min_speed` is 1 px/s.

### 2.3 Handling, not coefficients

Designers should choose quantities they can reason about: turn radius, peak and
sustained turn rate, corner speed, top speed, nose onset time. The physics constants
(thrusts, drag and lift coefficients, energy rates) are solved from those. The
constants are tightly coupled: raising top speed through regeneration also raises the
speed and rate of the sustained turn. A solver handles that coupling explicitly (§5).

### 2.4 How the game compares with real aircraft

Absolute numbers do not carry over. The game turns much faster and in much less room
than an aircraft, as an arcade game should. Ratios do carry over once two scales are
fixed:

* **Length:** one ship length L = 20 px (render size 10 px centre to tip; collision
  radius 10 px).
* **Time:** the best sustained 360° takes 3.0 s in the game and 17.5–27.5 s for WWII
  piston fighters. Rotation therefore runs **k ≈ 6–7 times faster** than real (k ≈ 4
  for coasting turns).

| quantity | today | planned (§4) | WWII piston fighters |
|---|---:|---:|---:|
| best instantaneous / best sustained turn rate | 1.75 at 136 px/s, unbounded above | 1.30 | 1.5–1.9 |
| speed of best sustained turn / top speed | 0.54 | 0.54 | 0.49–0.59 |
| corner speed / top speed | no corner | 0.74 | 0.69–0.86 |
| turn rate at top speed / at corner | rises with speed | 0.74 | 0.69–0.86 |
| top speed / lowest flying speed | 136 | ≈ 3.9 | 2.8–3.5 |
| turn onset, straight to full pull | 1 tick | 0.16 s | 0.6–1.5 s; ÷ k ≈ 0.1–0.2 s |
| speed lost per second in a max-rate turn at corner | 42% at 100 px/s | 42% | ≈ 9%; × k ≈ 55–60% (rough) |
| sustained radius / L | 1.8 sharp, 3.3 normal | same below corner | 21–36 |
| sustained radius / gun reach | 0.07 | 0.07 below corner, 0.35 at 215 px/s | ≈ 0.5–1 |
| top speed, L/s | 6.8 (10.5 on a dash) | 6.8 | 14–20 |
| muzzle speed / top speed | 3.7 | 3.7 | 4.5–6 |

"Top speed" here is the energy-limited boost terminal, 136 px/s at n = 1. For
aircraft, corner speed is taken at a 6 g pilot limit, and onset is roll-to-bank plus
the pull. The bleed estimate assumes a lift-to-drag ratio of about 5 at maximum lift
and a thrust-to-weight ratio of 0.3. Aircraft data is in Appendix C.

What this means for the plan:

* **Already in range:** the speed of the best sustained turn, and the bleed in a hard
  turn once time is scaled. Keep them.
* **Brought into range:** a corner speed, turn rate falling above it, a low-speed
  floor, and nose onset time. The planned onset of 0.16 s falls inside the
  time-scaled real range.
* **Short of range:** instantaneous over sustained turn rate is 1.30, against 1.5–1.9.
  The game's top speed sits close to the speed of its best sustained turn, so corner
  placement trades this ratio against corner/top. A corner at 120 gives 1.56 and 0.88;
  a corner at 100 gives 1.30 and 0.74. Raising top speed to about 150–160 px/s with
  the corner at 120 would put both in range. With the handling solver that is a change
  of targets, not of mechanics (§13).
* **Deliberately out of range: compactness.** Relative to ship length and gun reach,
  turning circles are about 12 times tighter than an aircraft's. A sharp 180° fits in
  about a sixth of gun reach. Matching aircraft would need sustained radii of
  250–500 px, which is a different game. Change compactness only as an explicit
  decision (§13).

## 3. Flight model

### 3.1 State

Two new per-ship fields:

* **slip** α, the nose angle relative to the velocity direction;
* **slip rate** α̇.

The attitude is `velocity_direction · e^{iα}`, as today with α in place of the
per-action offset. When α settles at a commanded value, the per-tick force equations
are today's. `ship_ang_vel` becomes the true heading rate: the velocity turn rate plus
α̇. Slip is observable as attitude minus velocity heading, and α̇ as angular velocity
minus the velocity turn rate, so the state stays Markov.

### 3.2 Lift and drag as functions of slip

With `x = |α| / stall_angle`:

```
C_L(α) = sign(α) · C_L_max · (1 − (1 − x)²)      x ≤ 1
C_D(α) = C_D0 + b·x² + c·x⁴
```

* **Lift.** The parabola inverts in closed form:
  `α = stall_angle · (1 − sqrt(1 − C_L/C_L_max))`.
* **Drag.** b and c follow from the drag at the normal and the stall slip.
* **Past stall,** lift falls on the mirrored parabola, clamped at zero. Commands never
  exceed the stall angle, and the nose response does not overshoot.
* **Application.** Drag is applied as the exact quadratic-drag scale and lift as an
  exact rotation of the velocity, as the field path does today.

Today's points are reproduced exactly by `stall_angle` 15°, `C_L_max` 0.027 (0.015 at
5°) and `C_D0` 8e-4, b 3.525e-3, c 6.75e-4 (0.0012 at 5°, 0.0050 at 15°).

### 3.3 Pull levels and the g-limit

The turn actions are straight, plus left and right at two pull levels:

* **sharp:** maximum lift, `C_L_max`;
* **normal:** a fixed lower lift level, capped at half the available lateral
  acceleration.

The command is a lift coefficient, capped so that lateral acceleration `C_L · u²`
stays within `max_lateral_accel`:

```
sharp_cl  = min(C_L_max,         max_lateral_accel / u²)
normal_cl = min(normal_lift,     max_lateral_accel / (2u²))
alpha_cmd = sign · lift_inverse(cl)
```

* **Corner speed** is `sqrt(max_lateral_accel / C_L_max)`. Below it the commands are
  today's; above it the nose angle shrinks with 1/u² and the radius grows with u².
* **Proper speed.** The cap uses proper speed, so the corner speed is the same in every
  field. World turn rate stays reciprocal in n at a fixed proper speed.
* **Lift only.** The cap applies to lift. Slipped boost thrust adds up to about
  80 · sin 15° ≈ 21 px/s² of lateral force on top, so measured lateral acceleration
  can sit slightly above `max_lateral_accel`. Validation allows for this (§10.1).
* **Starting point:** corner speed **100 px/s**, so `max_lateral_accel` = 270 px/s².
  The normal level's own corner is then about 95 px/s.

Lift-only values at n = 1 for this starting point:

| u | normal α | sharp α | normal R | sharp R | sharp rate today | sharp rate capped |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 5.0° | 15.0° | 67 | 37 | 62 | 62 |
| 100 | 4.4° | 15.0° | 74 | 37 | 155 | 155 |
| 120 | 2.9° | 6.7° | 107 | 53 | 186 | 129 |
| 136 | 2.2° | 4.8° | 137 | 69 | 210 | 114 |
| 160 | 1.5° | 3.3° | 190 | 95 | 248 | 97 |
| 215 | 0.8° | 1.7° | 342 | 171 | 333 | 72 |
| 300 | 0.4° | 0.9° | 667 | 333 | 464 | 52 |

The lift curve is flat near its peak, so the sharp nose angle falls quickly just above
corner speed. This is what makes high-speed flight feel committed.

Turns that settle below both corners are unchanged: every coasting turn, and the
boosted sharp turn at 73 px/s. The boosted normal turn settles at 118 px/s today, so it
will widen and slow.

### 3.4 Nose response

α tracks the command as a **critically damped** second-order system with natural
frequency ω, under a slip-rate limit `max_slip_rate`.

* **Exact step.** The step is computed exactly over the tick with the command held
  constant, so it is stable at any ω·dt and has no overshoot or ringing.
* **Onset.** The time to reach 90% of the command is about 3.9/ω. The starting point
  ω = 25 rad/s gives 0.16 s, about 5 decisions.
* **Rate limit.** Onset from straight to sharp peaks at about 138°/s. A limit near
  150°/s leaves onset alone and stretches a full left-to-right reversal to about
  0.25 s.
* **Release.** The nose returns to the flight path over the same ~0.16 s.
* **ω does not scale with speed.** Speed dependence comes from the g-limit and the
  low-speed fade. Scaling ω with speed would make low-speed nose throws sluggish.
* **Field index: open.** Path turns and every other time in the game scale with n (the
  field acts as time dilation). Whether the nose response should also run in proper
  time, ω and `max_slip_rate` divided by n in world time, is undecided (§13). The
  angular-velocity code is sized to cover both choices (§8.6).

### 3.5 Low speed

* **Fade.** Below `slip_fade_speed`, the command amplitude is scaled by
  `u / slip_fade_speed`. Starting point: 35 px/s, below the slowest sustained turn (the
  coasting sharp turn, 39 px/s).
* **Stall.** Below `min_speed`, lift is zero and the absolute attitude is held, as
  today. α is re-derived from the held attitude with α̇ = 0, so leaving the stall does
  not jump.

### 3.6 Air brakes removed

`AIR_BRAKE` and `SHARP_AIR_BRAKE` are removed from `TurnActions`. The scripted
controller never selects them (0% of decisions, Appendix A.8). Slowing remains
available through reverse thrust and sharp-pull drag (Appendix A.5). Before removing
them, measure a trained policy's air-brake usage (§12, step 1).

### 3.7 One flight path

The no-field path (`num_fields == 0`, `_update_kinematics`) applies lift as an
explicit velocity kick. That multiplies speed by `sqrt(1 + (C_L·u·dt)²)` every tick, so
lift adds energy:

* At 60 Hz its energy-limited boost terminal is 224 px/s, against 135 on the field
  path.
* A boosted normal turn there outruns straight flight.
* With power held full, a normal turn runs away to NaN after about 29 s.

No current profile takes this path. Replace it with the field path's flight steps,
using a plain drift in place of field transport. At n = 1 that is the field path
exactly, and the slip update then exists in one place.

### 3.8 Compilation and GPU constraints

The tick must stay compatible with `torch.compile(dynamic=False)` and CUDA-graph
capture.

* **No data-dependent control flow.** Caps, fade, stall gate and rate limit are
  `minimum`, `clamp` and `where`. There is no `.item()`, boolean indexing or
  `nonzero`.
* **No per-tick tensor construction.** All constants are plain floats on the frozen
  `ShipConfig`.
* **No iterative solves.** The lift inverse is a `sqrt`, and the damped step is
  multiply-adds with precomputed coefficients.
* **Static shapes.** The two new fields are `(B, N)` float32, zeroed on reset and
  respawn, and advanced by reassignment.
* **Cost** is a few dozen elementwise operations per ship. Confirm it with the
  throughput benchmark.

## 4. Knock-on changes outside the learner

* **Observation.** The `ang_vel` normalisation must be re-derived for the continuous
  range (§8.6).
* **Action space.** The turn head and the joint action id shrink. Checkpoints, BC data
  and replays recorded with air-brake actions become invalid.
* **Scripted controllers.** `scripted_utils.turn_toward` assumes an instant nose and
  will chatter with a 0.16 s lag. It needs hysteresis or a lead term, and each scripted
  opponent needs re-tuning against the new turn-rate curve. The scripted agents are the
  behaviour-cloning teacher, so this must be done before BC data for Part II exists.

## 5. Handling parametrization

### 5.1 The two sets of variables

**Physics constants θ** of the new model (16):

| group | constants |
|---|---|
| thrust | `base_thrust`, `boost_thrust`, `reverse_thrust` |
| energy | `power_speed_constant` K, `passive_power_gain` P |
| drag | `C_D0`, drag at normal slip, drag at stall slip |
| lift | `C_L_max`, normal lift level, `max_lateral_accel` |
| nose | `stall_angle`, ω, `max_slip_rate`, `slip_fade_speed`, `min_speed` |

`max_power` is fixed at 100. Power units are a free gauge: K appears only as
K·max_power, K·P and K·bullet cost. Bullet cost is in power units, so changing K
changes how much speed a shot costs. Keep that in view, or state bullet cost in energy
units.

**Handling quantities h.** Most are power laws in θ:

| quantity | closed form |
|---|---|
| cruise speed | (T_c / C_D0)^½ |
| top speed (energy-limited boost) | (K·P / C_D0)^⅓ |
| unlimited top speed | (T_b / C_D0)^½ |
| boost acceleration, reverse deceleration | T_b, \|T_r\| |
| recharge time | max_power / P |
| tank, as a speed from rest | (2·K·max_power)^½ |
| sharp radius, normal radius | 1 / C_L_max, 1 / normal lift |
| best sustained sharp speed | (K·P / C_D,stall)^⅓ |
| best sustained sharp rate | that speed · C_L_max |
| normal sustained speed and rate | the same with normal drag and lift |
| coasting turn speeds | (T_c / C_D)^½ at each level |
| corner speed | (max_lateral_accel / C_L_max)^½ |
| max turn rate | (max_lateral_accel · C_L_max)^½ |
| turn rate at top speed | max_lateral_accel / top speed |
| onset time | ≈ 3.9 / ω |
| reversal time | ≈ 2·stall_angle / max_slip_rate |
| sharp gun offset, fade speed, stall speed | stall_angle, slip_fade_speed, min_speed |

Some quantities are not power laws and have no closed form; the correction stage
(§5.2) handles them:

* the normal nose angle (through the lift inverse);
* dash peak and duration;
* speed bleed at corner;
* slipped thrust's share of turning, which makes the sharp turn about 5–7% tighter
  than the closed form.

Many quantities depend on each other; for example, corner speed = max turn rate ×
sharp radius. They are not independent targets.

### 5.2 Solving

* **Linear core.** In logs, the power laws are linear: `log h = A · log θ + const`.
  Pinning a set of targets is a linear system:
  * **square and full rank:** unique and exact;
  * **underdetermined:** the smallest change from a base configuration, with the
    unconstrained directions reported;
  * **redundant or inconsistent:** the dependency among targets is reported, with
    residuals.
* **Correction.** Starting from the linear solution, adjust using the flight-envelope
  harness as the forward model until every pinned target is met in simulation.
* **Feasibility.** The solver must reject specs that break any of these:
  * sustained-turn speed < corner < top speed;
  * normal radius > sharp radius;
  * fade speed < coasting sharp-turn speed;
  * drag rising with slip.
* **Integration.** A profile states a handling spec. The flight constants in
  `ShipConfig` are derived from it once, at configuration time, and recorded with the
  run's configuration. Physics reads plain floats, as now.

A prototype of the linear core for the current model is `benchmarks/flight_spec.py`
(13 constants, 18 quantities). It recovers today's constants exactly from today's
handling and reports dependencies and free directions. Simulation agrees with its
closed forms within 2% on every speed; the sharp turn is the 5–7% exception above.

### 5.3 Starting spec

The new model's first spec keeps today's handling, plus the planned changes:

| target | value |
|---|---:|
| cruise speed | 100 |
| top speed | 136 |
| unlimited top speed | 316 |
| boost acceleration / reverse deceleration | 80 / 80 px/s² |
| recharge time | 10 s |
| tank, as a speed | 200 |
| sharp radius / normal radius | 37 / 67 px |
| best sustained sharp rate | 114 (closed form; 119 measured) |
| corner speed | 100 (so max turn rate 155°/s) |
| stall angle | 15° |
| onset time | 0.16 s |
| reversal time | ≈ 0.25 s |
| fade speed / stall speed | 35 / 1 |

---

# Part II. Learning

## 6. Rewards

### 6.1 Principles

* **Rewards own spatial credit.** Every reward is written onto the ship that earned
  it. PPO solves temporal credit only, through per-head GAE. The only quantity
  broadcast across a team is the match outcome.
* **Each level is zero-sum.** An event charges the side it happens to and pays the
  side that caused it (§6.3).
* **Rewards, returns and advantages are scalars.** Only the critic's output is a
  distribution (§7).

### 6.2 Five heads

Events are grouped into five levels. Each level is **one critic head** whose reward is
the sum of that level's events. Events are still computed and logged one by one; they
do not get their own value heads.

| head | events summed into it | gamma | GAE lambda | horizon | read from |
|---|---|---:|---:|---|---|
| outcome | match result | 0.9997 | 0.97 | ~3 300 decisions, the match | global token, categorical (§7.2) |
| zone capture | completion ledger | 0.999 | 0.97 | ~1 000 decisions | ship token |
| zone progress | live meter movement | 0.995 | 0.95 | ~200 decisions | ship token |
| kill and death | death, kill of an enemy, kill of an ally | 0.995 | 0.95 | ~200 decisions, an engagement | ship token |
| damage | damage taken and dealt, friendly fire, recharge, charge-back | 0.991 | 0.90 | ~110 decisions, an exchange | ship token |

**Why five, not one per event.** GAE is linear in rewards and values, so at a shared
gamma and lambda the advantage of a sum equals the sum of the advantages. Splitting
events inside a level therefore does not change the policy gradient; it only changes
how the critic is fitted. The reason split components existed, letting a scalar critic
tell "an uncertain ±1 is coming" from "certainly nothing", is covered by the
categorical critic, which represents those as different distributions with the same
mean. One thing a merged kill-and-death head cannot tell apart is a mutual trade (kill
and die, net about zero) from a quiet spell, because the realised return is the same.
That does not affect the policy gradient.

**Why not one head.** The levels need different horizons, the outcome is read from the
global token while the rest are per ship, and a sparse ±1 outcome summed with dense
damage would dominate the critic's error.

**Why zone capture and zone progress are separate.** They have different time scales,
the same relationship as death to damage. Progress is paid on the tick a ship's
presence moves the meter (one tick when uncontested). A completion is paid only when
the meter fills, at least `capture_seconds` (10 s, 300 decisions) later at a one-ship
lead. Gamma sets how far back credit reaches, and the decision that earns progress is
choosing a zone and flying to it: adjacent zones on the 1 200 px ring are about
1 400 px apart, roughly 14 s at cruise. That is an engagement-length horizon, so
progress takes the kill-and-death values; damage's 0.991 would be too short.

This is the Honor of Kings layout: zero-sum reward groups, per-agent rewards, one value
head per group, advantages combined by weight.

### 6.3 Balance

* **Rule.** An event charges the side it happens to `w` and pays the side that caused
  it `ratio × w`. `kill_payout_ratio`, `damage_payout_ratio` and `capture_payout_ratio`
  start at 2.0; the schedule's `offensive_bias` holds them to 50M steps, anneals them
  linearly to 1.0 by 300M, then holds 1:1. At 1:1 every event is exactly zero-sum.
  The premium exists because priced evenly, an even trade is worth nothing and a policy
  that cannot reliably win it declines it.
* **Payee.** If a ship on the opposing side caused the event, that ship is paid. If
  nobody on the opposing side did, the counterpart is split evenly over the opposing
  team with `enemy_team_share`, dividing by the full team slot count including dead
  ships. The only non-bullet damage source is the play-area boundary, so this covers
  boundary damage and death, an enemy's friendly fire, and recharge of boundary damage.
* **Invariant, tested.** At all ratios 1, each head's reward summed over both teams is
  zero on every tick.
* **Free parameters.** One weight per level, plus the offensive bias. Everything else
  is derived. Starting values carry the current balance: win 1.0, zone capture 2.0,
  zone progress 2.0, death 0.283 per death, damage 0.274. The schedule's tier scales
  multiply them as today (outcome scale for the outcome and both zone heads,
  kill/death scale, damage scale).
* **The unit.** A win is 1. This is the unit DreamerV3's advantage floor is measured in
  (§7.3).

### 6.4 Events

**Outcome.** Win, tie or loss at the terminal transition; a 9 000-step timeout is a
real tie. Read and valued per environment from the global token and broadcast to every
allied ship.

**Zone capture.** Per active zone, two ledgers over the current capture attempt, with
progress P running from 0 to 1:

* attacker presence C_i: each tick the meter moves by Δ > 0 toward the attacker,
  `C_i += Δ / n_present` for each present attacking ship;
* defender absence A_j: on the same tick, `A_j += Δ / n_absent` for each absent
  defending ship, dead ships counting as absent.

So Σ C_i = Σ A_j = P. On a reversal by r both ledgers scale pro rata,
`C, A ← C, A · (1 − r/P)`. On completion:

```
zone_gain_i = ratio · w · [ ½ · present_i / n_present + ½ · C_i / P ]
zone_loss_j =       − w · [ ½ · absent_j  / n_absent  + ½ · A_j / P ]
```

Ledgers clear on completion, whenever roles rotate (which also zeroes every other
zone's meter), and when the meter returns to zero (guard P = 0). If the charged set is
empty the whole opposing side is charged; this is never reached in practice, because
`_advance_capture_state` moves a meter only on a strict alive majority inside the zone,
so the other side always has a ship dead or elsewhere.

**Zone progress.** Today's `_ZoneCreditReward` split, summed into one head: on every
tick a meter moves, the favoured side's ships inside the zone share `ratio × Δ`, and
the other side's ships outside it, dead included, share the charge `Δ`. Ships of the
losing side who are inside are contesting and are not charged. The completion tick is
excluded (the zone capture head pays it).

**Kill and death.**

* Death: `−U` to the dying ship, any cause.
* Outstanding-damage ledger O[a, t]: applied damage from attacker a to target t, the
  existing `cumulative_damage_matrix`, cleared at respawn. When t recharges by h, with
  S_t = Σ_a O[a, t]: `O[a, t] ← O[a, t] · (1 − h / S_t)` (pro rata, no queue).
* Share of a death of t:
  `share_a = ½ · raw_final[a, t] / Σ raw_final[·, t] + ½ · O[a, t] / S_t`.
  `raw_final` is the raw impact on the death tick: the killing hit lands on zero
  shield, so its *applied* damage is 0. If S_t = 0 (the shield was emptied by the
  boundary) the whole share goes to the raw final hit.
* Kill of an enemy: `+ratio · U · share_a`. Kill of an ally: `−ratio · U · share_a`,
  with the counterpart paid to the opposing team. A boundary death pays the opposing
  team by `enemy_team_share`. A ship is never blamed for its own death.

**Damage.**

* Damage taken: `−V · applied`, every source including the boundary. This is what
  stops boundary recharge farming.
* Damage dealt to an enemy: `+ratio · V` per unit, shared among simultaneous hits by
  applied damage. Boundary damage pays this to the opposing team.
* Damage dealt to an ally: `−ratio · V` per unit to the shooter; the counterpart is
  paid to the opposing team.
* Shield recharge: `+V · h` to the recharging ship.
* Charge-back: `−ratio · V · h · O[a, t] / S_t` to each attacker, taken before the
  ledger update, exactly reversing the payment for damage that no longer exists.
  Recharge of unattributed damage charges the opposing team. Recharge starts 150
  decisions after the last hit (5 s delay), so at gamma 0.991 the charge-back is worth
  0.991^150 ≈ 0.26 of the payment it reverses: it is correct bookkeeping, not the
  anti-farming mechanism.

**Removed:** `ally_win`, `enemy_win`, the token-weight `outcome` probe, `boundary`,
`boundary_damage`, the `ally_/enemy_combat_*` pair, the shaping components (facing,
closing speed, shoot quality, shooting penalty, speed), and `kill_shot_fraction`
(fixed at ½ by the share rule).

### 6.5 Team mixing

The lambda matrix (`_lambda_matrix`, `_precompute_lambda_aggregates`, `enemy_neg_k`,
`ally_zero_k`, `_LOCAL_COMPONENTS`) is deleted. Removing it does not change behaviour:
every component with nonzero weight is already diagonal, or bit-identical across a
side (`tests/train/test_global_value.py` asserts the latter). In its place each head
keeps one scalar, OpenAI Five's "team spirit":
`adv_i + s · mean_{j ≠ i, same side} adv_j`, default **0** (pure per-ship credit), so
a selfish equilibrium has a cheap remedy. The reward-decomposed gradient diagnostic
reads per-head advantages directly.

## 7. Critic, advantages and PPO

### 7.1 Raw rewards and a categorical critic

* **Rewards are stored raw.** The `symlog` applied at storage in `RolloutBuffer` is
  removed; GAE runs on raw rewards. Today's storage symlog is not what DreamerV3 does:
  a discounted sum of symlogged rewards is not the symlog of the return, and it
  compresses dense and sparse components differently.
* **The critic's output is a distribution; everything else stays scalar.** Each head
  emits logits over fixed bins whose locations are symlog-spaced. Its value V is the
  expectation, Σ probability × bin value, in raw units; that V is what GAE uses.
* **Target.** The realised scalar lambda-return, two-hot over the bins (DreamerV3),
  trained by cross-entropy. The target's mean equals the return, so what the value
  means is unchanged; only how the critic is fitted changes.
* **51 bins per head.** DreamerV3 uses 255 over `symexp(linspace(-20, 20))` because it
  spans many domains. At 51 bins that span would give a spacing of 0.8 symlog units,
  with the first bins at ±1.2 and ±4.0 raw: far too coarse for these returns. The
  range is narrowed to each head's measured largest |return| with margin, for example
  ±5 symlog units (±147 raw), spacing 0.2.
* **Removed:** the per-component `ReturnScaler` for the critic (the bin spacing is the
  single place values are compressed), the Huber value loss and any value clipping.
  Explained variance is computed on the expectation.

### 7.2 Outcome head: four classes

Classes **win, tie, loss, unresolved**. Value on the ±1 scale is V = P(win) − P(loss).

Gamma stays 0.9997: at gamma 1, with ~8 600-step episodes and 128-step rollouts, the
value bootstraps about 67 times before any terminal grounds it with no contraction to
damp error, and run 737 measured win explained variance falling from 0.994 to 0.42.
The discount is modelled as probability mass leaking into **unresolved**, worth 0:

```
Q_t = gamma · [(1 − lambda) · p(s_{t+1}) + lambda · Q_{t+1}] + (1 − gamma) · e_unresolved
Q_t = onehot(result)        on the terminal transition
```

Its expectation equals the scalar lambda-return, so the categorical target and the
scalar GAE advantage agree exactly. Loss: cross-entropy against Q_t.

The fourth class keeps **tie** honest. Without it the discount mass would have to go
into tie, and at gamma 0.9997 most of the target would be discount rather than
outcome:

| decisions remaining | mass on win + loss | mass from discount alone |
|---:|---:|---:|
| 9 000 | 0.07 | 0.93 |
| 4 500 | 0.26 | 0.74 |
| 1 000 | 0.74 | 0.26 |
| 100 | 0.97 | 0.03 |

Calibration is then checkable as P(win | resolved) against realised wins.

### 7.3 One advantage, normalised once

Each head's scalar advantage is weighted by its level weight, summed, and the sum is
normalised once, exactly as DreamerV3 does. Settings verified against the official
source (`github.com/danijar/dreamerv3`, main branch, fetched October 5, 2026:
`dreamerv3/configs.yaml`, `imag_loss` in `dreamerv3/agent.py`, `Normalize` in
`embodied/jax/utils.py`):

| setting | DreamerV3 |
|---|---|
| return normaliser | `impl: perc, rate: 0.01, limit: 1.0, perclo: 5.0, perchi: 95.0, debias: False` |
| scale | `max(limit, hi − lo)` of EMA 5th and 95th percentiles of the lambda-returns (EMA decay 0.99) |
| advantage | `(return − value) / scale`; the mean is **not** subtracted |
| value and advantage normalisers | none |
| critic and reward heads | symexp two-hot, 255 bins, `symexp(linspace(-20, 20))` |
| discount, lambda | horizon 333 (gamma ≈ 0.997), 0.95 |
| actor entropy | 3e-4 |
| critic regulariser | toward an EMA copy of the critic, rate 0.02 (not adopted: PPO's fixed rollout targets do that job) |

So the scale is the summed return's 5th–95th percentile spread, EMA decay 0.99,
floored at 1; the summed advantage is divided by it; PPO's per-minibatch advantage
standardisation is dropped.

This replaces the per-component `AdvantageScaler`, which normalised each component to
unit RMS *before* weighting and so cancelled the derived weights. After the change the
level weights reach the policy exactly as derived, and splitting or merging events
inside a level cannot change its gradient share.

**Why a win is the unit.** The floor means advantages are never amplified while the
return spread is under one unit. With a win at 1 the floor binds only when outcomes
barely vary across the batch, as in run 735's stalemate where 90% of games timed out;
in that regime small kill and damage differences stay at their natural size rather
than being blown up into the whole update.

### 7.4 PPO retune

Today's PPO settings were tuned for advantages standardised to unit spread per
minibatch and for a Huber value loss. Both go, so every setting that is measured
against the advantage or value-loss scale is retuned:

| setting | today | why it moves |
|---|---|---|
| `entropy_coef` | 0.005 | Advantages are now in win units with a floor of 1; dense-head advantages can be around 0.01, so the entropy bonus would dominate at today's value. DreamerV3 uses 3e-4 with this normaliser. |
| `clip_coef` | profile value | The clip bounds the probability ratio; how often it binds depends on advantage size. |
| `target_kl`, `high_winrate_target_kl` | 0.1, 0.02 | The KL bound currently ends most updates early (`train/epochs_completed`); with smaller advantages it may stop binding. |
| `learning_rate` | peak 3e-4 | Set together with `target_kl`, as the existing schedule comment explains. |
| `value_function_coef` | 1.0 | Cross-entropy over bins has a different scale and gradient from Huber. |
| `behavior_cloning_coef` | 2.0 | Set from measured gradient norms against the policy-gradient term, whose size changes. |
| `num_epochs` | 4 | Read with `target_kl`. |

Method: short runs from the cold-start initialisation, reading per-term trunk gradient
shares with `benchmarks/gradient_decomposition.py`, policy entropy, clip fraction,
approximate KL and epochs completed. Starting points: `entropy_coef` 3e-4, the rest at
today's values. The PPO objective itself is unchanged.

### 7.5 Precedent

| system | reward and credit | critic | advantage scaling | policy loss |
|---|---|---|---|---|
| DreamerV3 | one raw scalar reward | symexp two-hot, 255 bins | percentile spread, max(1, S) | actor-critic |
| Dreamer 4 | same | symexp two-hot reward and value | none | PMPO, sign of the advantage, KL to the BC prior |
| Honor of Kings | five zero-sum groups (farming, KDA, damage, pushing, win/lose), per hero | one value head per group | per-group weights | dual-clip PPO |
| OpenAI Five | zero-sum by subtracting the enemy mean; "team spirit" mixes own and team reward | single | — | PPO |
| MAPPO | shared team reward | centralised | running mean and std of value targets | PPO |

Learned credit-assignment methods (COMA, QMIX, partial reward decoupling, learned
internal rewards in Capture the Flag) target cooperative games with one shared reward.
This environment attributes every event to a ship already; the strongest results in
team-versus-team games use hand-built zero-sum per-agent rewards.

## 8. Next-state head and belief

### 8.1 What the current head does

Measured on run 747's final checkpoint (`next-state-head-audit-sep2026.md`, 2.68M
scored transitions), mean absolute error:

| | ally visible | enemy visible | enemy hidden |
|---|---|---|---|
| position | 0.24 px, 4× worse than `pos + vel·dt` | 0.20 px, 3.9× worse | 527 px, equal to "nothing changed" |
| velocity | 4.78 px/s, 1.5× worse than "nothing changed" | 4.69 px/s, 1.7× worse | 83.5 px/s, equal |
| health | 1.26 hp, 4.5× worse | 1.27 hp, 3.5× worse | 3.3 hp, equal |
| cooldown | 1.15× worse | 1.19× worse | equal |

For hidden enemies the belief was 93× worse than dead reckoning at 0.1–0.5 s of
occlusion and only overtook it after about 10 s; 70% of hidden tokens are under 5 s
old. Nine of ten position Fourier harmonics on a visible ship were 10× to 500 000×
worse than predicting no change; only the one harmonic carrying a sigma trained, and
the visible-ship sigmas were 3–6× too wide, which throttled the Gaussian likelihood's
gradient where the dynamics are learnable.

The design below addresses the gradient starvation (per-level cross-entropy with no
learned sigma), the cost of "no change" (a residual head) and the sigma calibration.
It deliberately gives the head **no physics prior**: the head exists as a hard
auxiliary task, and learning the dynamics is its job. Dead reckoning is the bar it is
measured against, not a shortcut.

### 8.2 The head

* Residual logits on a baseline built from the current input:
  `z_{t+1} = log p̃_t + f(h_t)`, with `f`'s last layer zero-initialised so the head
  starts as "nothing changes", and `p̃` floored by a small epsilon so the log is finite.
* **Targets are sharp for every channel and every token, hidden ones included.**
  Cross-entropy against the sharp truth is a proper scoring rule: at a level the head
  cannot resolve, its optimum is the conditional distribution, which is near uniform.
  Targets are built on the fly from truth and never stored.
* **Hidden-ship inputs and baselines are smoothed** by the belief's sigma (§8.3). With a
  calibrated sigma the smoothed baseline already equals the best prediction at an
  unresolvable level, so the head outputs zeros there instead of memorising and
  cancelling a stale sharp pattern. Smoothing the input, not the target, keeps the loss
  proper and lets the head beat its prior wherever it has information.
* The 13 Gaussian uncertainty terms are removed. `time_since_observation` stays.
* The ship-token input becomes mostly sparse multi-hot codes; initialise the encoder's
  first layer per input unit, like an embedding table.
* `next_state_coef` is set from a measured trunk-gradient share (the Gaussian head held
  76% of the trunk in run 748).

### 8.3 Belief

**Storage: moments, not distributions.** Storing the full categorical distribution of
every channel would add about 1.1 GB of GPU buffer per resident rollout shard and
3.2 GB of host RAM for the stored shards (one shard is 128 steps × 960 envs × 10 ships
= 1.23M ship tokens, about 440 extra bf16 channels each). The belief and the rollout
buffer store moments instead:

| channel | stored | floats |
|---|---|---:|
| position | mean (fp32) and one sigma | 3 |
| velocity | 2D mean and 2×2 covariance, in raw world velocity | 5 |
| attitude | mean angle and spread | 2 |
| angular velocity | mean and sigma | 2 |
| health, power, shield delay, local log index | mean and sigma each | 8 |
| cooldown | the distribution (4 bins) | 4 |

24 floats per ship, the same as today.

**Rebuilding the input.** Inside the forward pass the categorical input, which is also
the head's baseline, is rebuilt from the moments: position and attitude by the
per-level blur curve, each velocity axis as a 1D Gaussian (§8.5), scalars as a
discretised Gaussian over their bins. These activations are transient and
micro-batched; the buffer stays at today's size.

**Visible ships: mean = truth, sigma = 0.** The input of a visible ship has no
uncertainty, so it is the exact code of the true value: the bilinear multi-hot for
position, the two-hot for every binned channel. Allies are always visible. The blur
curve is defined so that sigma 0 gives exactly that code, and the decoder clamps its
sigma fit at 0. On a fast ship the sharp baseline puts only epsilon on the cell the
ship moves into (at 215 px/s a ship crosses about two finest position cells per
decision); learning that motion is the head's job (§8.1), and epsilon sets how much
loss the zero-initialised head starts with there.

**Hidden ships use only model outputs, carried forward from the last sighting.**
Nothing is computed from physics or the map: no attitude from velocity heading plus
slip, no log index from the field map, no angular velocity from the enemy-action head.

**Decoding is closed-form least squares only: no sampling, no branching search over
candidates.**

* Coded channels (position, attitude): per level, the probability-weighted circular
  mean of the colour positions as a complex phase; levels combine coarse to fine by
  unwrapping each finer phase against the coarser estimate. Sigma is a weighted
  least-squares fit of log per-level sharpness against the blur curve, clamped at 0.
* Velocity: §8.5.
* Scalars: mean and variance of the histogram.

**Round trip must be exact.** With a zero residual, decoding the rebuilt input must
return the same moments, or the belief drifts by itself every step. Each code has a
round-trip unit test, including sigma 0, and the scalars near their range edges where
clipping distorts the Gaussian.

**Accepted limitations.**

* The belief is unimodal: it cannot say "left or right of the zone".
* Hidden health and shield delay are really bimodal (the ship took a hit or did not;
  the delay is counting down or reset to 5 s). A mean and sigma blurs that into one
  bump. Log hidden health and shield-delay error by hidden age.
* When a ship disappears its belief is the last visible state at sigma 0, a sharp
  baseline. Widening and moving it is a learned skill with no prior; this is where the
  design is most likely to fall short. Watch hidden-age position error against dead
  reckoning from the first updates.

### 8.4 Position: 9-colour nested code

**Construction.** Base 3 on the square lattice. Level ℓ has cell spacing
`d_ℓ = 65 536 / 3^(ℓ+1)` px. A point's colour at level ℓ is (x mod 3, y mod 3) of its
cell corner in units of d_ℓ, nine colours. Each level is a bilinear multi-hot over the
cell's four corners, which always have distinct colours. Every level wraps the
65 536 px torus exactly. Nine levels, 81 logits:

| level | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spacing (px) | 21 845 | 7 282 | 2 427 | 809 | 270 | 90 | 30 | 10.0 | 3.3 |

The finest cell, 3.3 px, matches cruise motion per decision (3.3 px at 100 px/s), so
small moves shift the bilinear weights enough to train on. The code splits by axis:
each level is a pair of independent base-3 codes, so decoding runs per axis.

**Why 9 colours.** A code that wraps the square torus and is self-similar from level
to level must step by multiplication by a Gaussian integer a + bi. The colour count is
then a² + b² and each level rotates by atan(b/a), so the possible counts are 4, 5, 8,
9, 10, 13, 16, 17, 18, 20, 25, … (3, 6, 7 and 12 belong to hexagonal lattices, which
cannot tile a square torus; 2 fails because two cell corners share a colour).
Information is ln N minus the entropy of the bilinear target, averaged over positions
in a cell, in nats:

| colours | base | rotation | per axis | levels to ~4 px | finest | logits | step | info mean | info min | per logit |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2 | 0° | yes | 14 | 4.0 px | 56 | 2.00× | 0.39 | 0.00 | 0.097 |
| 5 | 2+i | 26.6° | no | 12 | 4.2 px | 60 | 2.24× | 0.61 | 0.22 | 0.122 |
| 8 | 2+2i | 45° | no | 10 | 2.0 px | 80 | 2.83× | 1.08 | 0.69 | 0.135 |
| **9** | **3** | **0°** | **yes** | **9** | **3.3 px** | **81** | **3.00×** | **1.20** | **0.81** | **0.133** |
| 10 | 3+i | 18.4° | no | 9 | 2.1 px | 90 | 3.16× | 1.30 | 0.92 | 0.130 |
| 13 | 3+2i | 33.7° | no | 8 | 2.3 px | 104 | 3.61× | 1.57 | 1.18 | 0.120 |
| 16 | 4 | 0° | yes | 7 | 4.0 px | 112 | 4.00× | 1.77 | 1.39 | 0.111 |
| 17 | 4+i | 14.0° | no | 7 | 3.2 px | 119 | 4.12× | 1.83 | 1.45 | 0.108 |
| 25 | 5 | 0° | yes | 6 | 4.2 px | 150 | 5.00× | 2.22 | 1.83 | 0.089 |

With few colours the bilinear target lights nearly all of them, so a level says little
(the 4-colour code, the analogue of today's dyadic Fourier ladder, carries nothing at a
cell centre). With many, logits grow faster than the information they carry, the error
from losing one level grows with the step, and the belief blur moves in coarser steps.
Information per logit peaks at 8–10 colours; 9 sits at the peak and decodes per axis.
10 (3+i) is the fallback if square-lattice axis bias shows up, since its 18.4°
rotation per level averages the bias out.

**Known properties.** Losing a middle level costs about one coarser cell of error (the
residue-code failure mode; the old Fourier ladder had it at 512–4 096 px). Levels 0–1
barely change within the 2 600 px playable radius at map scale 1 and only encode the
random map centre; they matter on large maps (§9).

**RoPE** stays on physical coordinates: truth for visible ships, the decoded mean for
hidden ones.

### 8.5 Velocity: three axes

**Code.** `u = v̂ · v₀ · log(1 + |v| / v₀)` with v₀ ≈ 100 px/s, on world velocity:
roughly linear across the playing band (about 20–215 px/s) and logarithmic only in the
unlimited-boost tail. Cells are nearly isotropic (about 1.4× elongation). u is
projected onto unit vectors at 0°, 120° and 240°; each projection is a 1D two-hot over
81 bins of 5 u-units, covering |u| ≤ 199 (631 px/s, the highest world speed, unlimited
boost at n = 0.5). 243 logits.

| speed (px/s) | 20 | 100 | 215 | 631 |
|---|---:|---:|---:|---:|
| spacing along / across velocity (px/s) | 6 / 6 | 10 / 7 | 16 / 9 | 37 / 16 |

Today's knee at 1 px/s gives 152 / 33 px/s at 100 px/s. The measured per-decision Δv
is about 3.9 px/s RMS, about 0.4 bin.

**Moments in raw velocity.** The belief stores the mean and covariance Σ_v of raw world
velocity: that is the physical quantity, the one dead reckoning and the diagnostics use,
and it does not depend on the code's compression. The code lives in compressed space,
so the two are linked by the Jacobian of the compression at the mean,

```
J = f′(r) · v̂v̂ᵀ + (f(r) / r) · (I − v̂v̂ᵀ),   f(r) = v₀ log(1 + r/v₀),   r = |v|
```

which is the identity at r = 0.

* **Encode:** u-mean = compression of the v-mean; Σ_u = J Σ_v Jᵀ; axis k is the 1D
  Gaussian with mean `n_k · u` and variance `n_kᵀ Σ_u n_k`.
* **Decode:** the u-mean is the least-squares solution from the three axis means, and
  the v-mean its inverse compression; the three axis variances determine Σ_u exactly;
  Σ_v = J⁻¹ Σ_u J⁻ᵀ.

The linearisation is approximate for a wide hidden-ship belief, but the round trip is
exact, because both directions use the same Jacobian at the same mean.

Three marginals cannot represent a bimodal joint (turned left or right). The belief is
unimodal anyway, and the three axis variances recover the covariance exactly, so the
round trip loses nothing.

### 8.6 Attitude and angular velocity

**Attitude: 4-colour circular code.** The 1D analogue of the position code. Level ℓ has
4 colours spaced `360° / 4^(ℓ+1)` apart and repeats every `360° / 4^ℓ`; each level is a
two-hot between the two neighbouring colours, and the code wraps the circle exactly.
Four levels with spacings 90°, 22.5°, 5.6° and 1.4°: **16 logits, 1.4° finest.**

Aiming needs about 1.5°: a 10 px hull at 200–400 px subtends 1.4°–2.9°. A flat 72-bin
code (5°) carries the angle only as interpolation weights, and today's input is four
Fourier harmonics whose finest period is 45°.

| colours per level | levels | logits | finest | info mean | info at midpoint | per logit |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 8 | 16 | 1.4° | 0.19 | 0 | 0.10 |
| 3 | 5 | 15 | 1.5° | 0.60 | 0.41 | 0.20 |
| **4** | **4** | **16** | **1.4°** | **0.89** | **0.69** | **0.22** |
| 6 | 3 | 18 | 1.7° | 1.29 | 1.10 | 0.22 |

With the slip model (§3.4) attitude moves continuously: per decision at most the path
turn plus the slip-rate limit, (155/n + 150)/30 ≈ 10° at n = 1 and 15° at n = 0.5. An
enemy's next attitude, whose command is unknown, is a smooth spread of that width
rather than today's five spikes; the finest level (1.4°) sits near uniform for enemies
during manoeuvres. Training is unaffected (per-level cross-entropy is proper). For
allies the command is known, and slip and slip rate are observable (§3.1), so every
level is predictable.

**Angular velocity: continuous.** Today it is exactly `turn_offset / dt`, five values.
With the slip model it is the path turn rate plus α̇, continuous. It is a two-hot over
**41 bins spanning ±610°/s**, 30.5°/s apart (1° of attitude per decision). The range is
(155 + 150)/n at n = 0.5: the path turn rate is reciprocal in n, and the range also
covers a nose response run in proper time (§3.4). It is rebuilt from mean and sigma
like the other scalars. Under the critically damped response the rate can change by
more than 100°/s within one decision during onset, so the per-decision change is large
against the bin; hidden-ship angular velocity is expected to sit near its prior.

The enemy-action head's turn prediction is a diagnostic to compare against, not a
source for this channel.

### 8.7 Other channels

Bins are sized to resolve the events that matter with natural values on bin centres;
slow deterministic drift is cheap because the residual baseline makes "no change" free.

| channel | range | bins | spacing | event resolved | drift per decision |
|---|---|---:|---|---|---|
| health (shield) | 0–100 | 21 | 5 hp | a 10 hp hit is 2 bins | recharge 0.5 hp, 0.1 bin |
| power | 0–100 | 21 | 5 | boost drain | ~0.15 bin |
| shield delay | 0–5 s | 21 | 0.25 s | the reset to 5 s is the last bin | countdown, 0.13 bin |
| local log index | −ln 2 to ln 2 | 21 | 0.069 | the five plateaus (n = ½ … 2 in steps of √2) on bins 0, 5, 10, 15, 20 | ~0.05, 0.7 bin |
| cooldown | 0–0.1 s | 4 | 1 tick | exactly 3 ticks | — |

* **Local log index** transitions between plateaus matter; at 5 bins the per-decision
  change moved a two-hot by only 0.14 bin.
* Finer bins (51) for health and shield delay are worth trying only if those channels
  sit at the "no change" baseline after training.

## 9. Map-scale randomisation

* **No training at other fleet sizes.** Zero-shot fleet-size transfer is part of the
  project's point: it worked for dogfighting and is the goal for Frontline. The
  crossover evaluation is the transfer evidence.
* **Purpose: representation coverage.** Positions at 50v50 span about 16 000 px and use
  coarse position levels and low-frequency RoPE pairs that the 5v5 map never exercises.
  Only the map size changes: ship count, episode length (9 000 steps), physics, vision
  and rewards are fixed. At ten ships, scale s divides areal density by s², so
  per-scale metrics are sparse-map metrics, not transfer evidence. Log ships-in-sight
  alongside s.
* **Mechanism.** s is sampled per environment at reset and applied through
  `scaled_frontline_geometry` (playable radius, zone radius, zone ring, field sizes and
  widths), whose quantities already live per environment in `TensorState`. s ≤ 12
  keeps the playable radius (31 200 px) under the 32 768 px half-period.
* **Required before it is turned on.** Field generation (`field_generation.py`) and the
  scripted strategy (`frontline_strategy.py`, scaled once in `evaluation/agents.py`)
  read scalar radii; on a rescaled environment the behaviour-cloning teacher and the
  scripted opponent would be wrong. Both must read per-environment geometry.
* **Range and distribution: open (§13).** One candidate: probability 0.5 at s = 1,
  otherwise s = exp(x) with x exponential at rate 1.5, truncated to [0, ln 12]:

  | | P |
  |---|---:|
  | s = 1 | 0.50 |
  | s ≤ 1.25 | 0.65 |
  | s ≥ 2 | 0.17 |
  | s ≥ 3.16 (50v50 scale) | 0.08 |
  | s ≥ 6 | 0.023 |

  At s = 12 the zone ring radius is 14 400 px, about 106 s at cruise against a 300 s
  episode, so large maps are nearly all timeouts at full cost. A cap near s = 4 covers
  the 50v50 scale with margin.

---

# Part III. Validation and delivery

## 10. Validation and diagnostics

### 10.1 Physics

Use `benchmarks/flight_envelope.py`, run before and after. Its output for today's
physics is `docs/internal/flight-envelope-baseline-oct2026.json` (Appendix A).

* **Settled envelope:**
  * below corner, within 2% of Appendix A.2 at every index;
  * above corner, matching §3.3, with lateral acceleration allowed up to the slipped
    thrust's share above `max_lateral_accel`;
  * excluded: the boosted normal turn and the power-held-full rows, which settle above
    corner.
* **Transients:** onset to 90% in 0.13–0.20 s; snap-back to within 1° of the path in
  0.13–0.20 s; reversal time; nose throw (0.5 s sharp from straight at 60, 100 and
  160 px/s: peak gun angle and path heading change).
* **No jitter:** α̇ never changes sign during a held turn, and α never passes the stall
  angle.
* **One flight path:** with no fields, lift alone keeps speed constant; energy never
  rises without thrust or regeneration; a held-power turn stays finite; the no-field
  step matches the field path at n = 1.
* **Compilation:** no graph breaks or recompiles across a rollout; captured-tick replay
  matches eager; throughput within noise of today.
* **Play:** scripted duels report hit rate, time-to-kill, mean speed and time above
  corner, against today's physics.

### 10.2 Learning: gates before the run

* The per-head zero-sum test at ratio 1.
* The encode/decode round-trip tests for every code (§8.3), including sigma 0 and the
  velocity Jacobian.
* The outcome head's categorical target has the scalar lambda-return as its mean.

### 10.3 Learning: diagnostics during the run

* Position cross-entropy per level, visible against hidden; hidden bucketed by
  realised error. Watch the finest level for fast visible ships.
* Sigma calibration against realised error, by hidden age, for every channel.
* Hidden-age position and velocity error against dead reckoning and persistence
  (already logged as production diagnostics).
* Hidden health and shield-delay error by hidden age.
* Zone occupancy, front advances and timeout rate (run 735's collapse signature: zone
  occupancy fell from 0.075 to 0.0006 of live ship-steps and 90% of games timed out),
  bucketed by map scale and ships-in-sight.
* Per-head gradient share and per-head zero-sum residual at ratio 1.
* The charge-back's realised share of the damage head.
* Outcome calibration: P(win | resolved) against realised wins.
* The PPO series of §7.4: entropy, clip fraction, approximate KL, epochs completed.

Read the run against the dead-reckoning and persistence baselines, not against the
Gaussian head. Check cooldown first: making "no change" free was the audit's diagnosis
for it.

## 11. Code touched

* **Physics:** `env/physics.py` (slip state, lift and drag curves, g-limit, one flight
  path), `ShipConfig` and the handling solver (from `benchmarks/flight_spec.py`),
  `TurnActions` and the joint action id, the `ang_vel` observation normalisation,
  `scripted_utils.turn_toward` and the scripted opponents.
* **Rewards:** `env/rewards.py` and `env/frontline.py` (events and ledgers).
* **Critic and PPO:** `train/rl/ppo.py` (lambda matrix, advantage combination, value
  loss), `train/rl/buffer.py` (raw rewards, normaliser), `config/defaults.py` (PPO
  schedule).
* **Belief:** `models/yemong/policy.py` (heads), `train/rl/physical_belief.py`,
  `train/rl/belief.py`, `train/rl/features.py`, `env/observation.py` (codes and belief).
* **Map scale:** `env/field_generation.py` and `agents/frontline_strategy.py`
  (per-environment geometry).
* The checkpoint schema and the tests.

## 12. Order of work

1. **Measure air-brake usage** of a trained policy (§3.6).
2. **Unify the flight path** (§3.7) as its own change, so later measurements compare
   one change at a time.
3. **Extend the solver** to the new model's constants and quantities, add the
   correction stage, and derive `ShipConfig` flight constants from a handling spec.
   Today's constants must round-trip.
4. **Implement the slip state,** lift and drag curves, g-limit, fade and air-brake
   removal.
5. **Measure** with the harness against §10.1.
6. **Re-tune the scripted controllers** and run the duels. Physics is now fixed; no RL
   run is needed for this part.
7. **Build Part II** against the new physics: rewards, critic, next-state head and
   belief, per-environment geometry for map scale. One checkpoint-schema bump.
8. **Pass the gates** of §10.2.
9. **Retune PPO** and the gradient-balanced coefficients (`next_state_coef`, the BC
   loss balance) with short runs from the cold start (§7.4).
10. **Train once from the cold start** and read it against §10.3. No weight migration
    and no intermediate comparisons of Part II pieces.

## 13. Open questions

1. **Corner speed versus top speed.** A corner at 100 matches real corner/top ratios
   but leaves instantaneous/sustained at 1.30. Raising top speed with the corner at
   about 120 would bring both into range. Decide from the duels.
2. **Compactness.** Turning circles are about a twelfth of an aircraft's relative to
   gun reach. Halving lift doubles every radius but changes the sustained envelope.
   Leave compactness unchanged unless it is chosen deliberately.
3. **Nose response in proper time.** Whether ω and `max_slip_rate` scale with the field
   index like every other time in the game (§3.4).
4. **Deep nose throws.** A sharp ceiling past stall (25–30°) at low speed would trade
   lift for drag steeply, which is closer to real aircraft. Try it as a variant after
   the 15° version works.
5. **Thrust vectoring** is out of scope. If it returns, cap its rotation rate and scale
   it with applied thrust. A near-stationary turret is the abuse case to test.
6. **Map-scale range and distribution** (§9).

---

## Appendix A. Today's flight envelope (measured)

Measured with `benchmarks/flight_envelope.py` on the live configuration:
`frontline_ship_config(SHIP_CONFIG)`, dt = 1/30, field path. Each probe ship sits in a
uniform field of index n. Speeds are proper speeds; rates are world °/s.

### A.1 Straight-line speeds

| n | coast | boost, energy-limited | dash peak from cruise (time to peak) | boost, power held full |
|---:|---:|---:|---:|---:|
| 0.5 | 100 | 107 | 203 (0.9 s) | 315 |
| 0.707 | 100 | 121 | 206 (1.3 s) | 315 |
| 1 | 100 | 135 | 209 (1.9 s) | 316 |
| 1.414 | 100 | 152 | 214 (2.8 s) | 316 |
| 2 | 100 | 171 | 221 (4.3 s) | 316 |

The energy-limited terminal follows (K·P / C_D0)^⅓ · n^⅓.

### A.2 Sustained turns (n = 1)

| power | turn | speed | rate | radius | 360° |
|---|---|---:|---:|---:|---:|
| coast | normal | 81.5 | 70.4 | 66.3 px | 5.1 s |
| coast | sharp | 39.3 | 63.7 | 35.3 px | 5.7 s |
| boost | normal | 118.3 | 102.5 | 66.3 px | 3.5 s |
| boost | sharp | 73.2 | 119.2 | 35.4 px | 3.0 s |
| power held full | normal | 257.3 | 222.1 | 66.4 px | 1.6 s |
| power held full | sharp | 124.1 | 200.4 | 35.5 px | 1.8 s |

Radius does not depend on index or power. World turn rate at the same proper speed is
reciprocal in n.

### A.3 Instantaneous turning (one tick, coast, n = 1)

| u | 40 | 100 | 120 | 136 | 160 | 300 | 600 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sharp rate | 65 | 154 | 184 | 208 | 244 | 448 | 864 |
| sharp lateral accel (px/s²) | 45 | 267 | 382 | 489 | 672 | 2 291 | 8 641 |
| sharp du/dt (px/s²) | −0.3 | −42 | −63 | −83 | −117 | −421 | −1 630 |
| normal rate | 35 | 86 | 103 | 117 | 137 | 256 | 507 |
| normal du/dt (px/s²) | +6.0 | −4.0 | −9.3 | −14 | −23 | −99 | −414 |

### A.4 Time to turn 180° from straight flight (n = 1, full power)

| entry u | 40 | 70 | 100 | 136 | 180 | 215 | 300 |
|---|---:|---:|---:|---:|---:|---:|---:|
| coast, sharp | 2.80 | 1.97 | 1.47 | 1.13 | 0.87 | 0.73 | 0.53 |
| boost, sharp | 1.37 | 1.17 | 1.03 | 0.87 | 0.73 | 0.63 | 0.50 |
| coast, normal | 4.07 | 2.87 | 2.17 | 1.67 | 1.30 | 1.10 | 0.80 |
| boost, normal | 1.90 | 1.67 | 1.47 | 1.27 | 1.07 | 0.93 | 0.73 |

The lateral offset at 180° is 66–75 px for sharp turns and 129–134 px for normal
turns, whatever the entry speed. Times scale with n.

### A.5 Acceleration and braking (n = 1)

* **Boost from rest:** 100 px/s in 1.3 s, 136 in 1.8 s; peak 199 at 2.9 s.
* **Boost from cruise at full power:** 200 in 1.7 s; peak 209 at 1.9 s, when power
  runs out.
* **From 215 to 100 px/s:**

  | method | time |
  |---|---:|
  | coast | 43.6 s |
  | air brake | 7.6 s |
  | sharp air brake | 1.2 s |
  | reverse | 1.2 s |
  | reverse with sharp air brake | 0.6 s |

* **Reverse from cruise to standstill:** 1.2 s, returning 25 power from kinetic energy.
* **Index:** times scale with n.

### A.6 Gun response (coast, n = 1)

Hold sharp for 0.5 s from straight flight:

| u | 40 | 60 | 100 | 160 | 215 | 300 |
|---|---:|---:|---:|---:|---:|---:|
| gun at 0.1 s | 19° | 21° | 25° | 31° | 36° | 44° |
| gun at 0.5 s | 45° | 58° | 81° | 114° | 141° | 179° |
| path at 0.5 s | 32° | 46° | 71° | 105° | 134° | 173° |

The gun steps 15° on the first tick, and the slip clears in one tick on release.
Today's "nose throw" is mostly a real path turn.

### A.7 Bullets (n = 1)

* **Reach in 1 s:** 421 px from rest, 490 at 100 px/s, 565 at 215 px/s.
* **Time to 200 px** at cruise: 0.37 s.
* **Muzzle speed:** 500 px/s relative to the ship.

### A.8 Scripted play

`StochasticScriptedAgent` against itself on the `rl` profile, 1.92 M ship-decisions.

* **Proper speed:**

  | p1 | p5 | p25 | p50 | p75 | p95 | p99 |
  |---:|---:|---:|---:|---:|---:|---:|
  | 28 | 41 | 80 | 113 | 133 | 152 | 166 |

  Above 100: 61%. Above 120: 42%. Above 136: 19%.
* **Heading rate:** zero on 59% of decisions; p95 154°/s; p99 217°/s.
* **Lateral acceleration:** p95 292 px/s²; p99 434 px/s².
* **Actions:** boost straight 46%, normal turns 21%, sharp turns 20%, air brakes 0%.
* **Local index:** n = 1 50%, n = 2 18%, n = 1.414 10%, n = 0.707 7%, n = 0.5 7%,
  transitions 7%.

## Appendix B. Game constants used for sizing

World speeds (px/s) unless stated; these set the code ranges in Part II.

* **Speed band:** normal play spans about 20–215 px/s world (the energy-limited
  terminal of 107 proper at n = 0.5 is 214 world). The highest world speed is 631 px/s
  (unlimited boost at n = 0.5). `max_speed` 180 in `ShipConfig` is not a cap.

| quantity | value |
|---|---|
| displacement per decision | 0.7 px (20 px/s), 3.3 px (100), 7.2 px (215), 21 px (631) |
| largest forward Δv per decision | ~10.7 px/s |
| largest lateral Δv per decision (today) | ~17.8 px/s energy-limited; ~133 px/s unlimited |
| measured Δv per decision (allies) | RMS ~3.9 px/s, p99.9 ~18.6 px/s |
| cooldown | 3 ticks (0.1 s) |
| shield recharge | 5 s delay (150 ticks), then 0.5 hp per tick |

The lateral Δv figures fall above corner speed under the g-limit and are re-measured
after Part I.

* **Geometry.** World torus 65 536 × 65 536 px; map centre uniformly random per
  episode. 5v5: playable radius 2 600 px, zone ring 1 200 px, zone radius 330 px.
  Lengths scale with `s = sqrt(ships / 10)` for fixed density (50v50: s = 3.16,
  playable radius 8 222 px). Vision range 1 024 px.

## Appendix C. Reference aircraft

### C.1 WWII fighters

Sea-level figures from the IL-2 Great Battles aircraft database (v5.204), which is
simulator data compiled from historical tests. Corner speed is the stall speed × √6 (a
6 g pilot limit), and the rate at corner is g·√35 / V_c. The derived Spitfire radius
(211 m) matches the RAE estimate of 700 ft (212 m).

| aircraft | length | top speed | stall | sustained rate | sustained radius | corner | rate at corner | inst/sust | turn/top | corner/top | top/stall | radius / L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Spitfire Mk IXc | 9.47 m | 542 km/h | 153 km/h | 20.3°/s | 211 m | 375 km/h | 31.9°/s | 1.57 | 0.50 | 0.69 | 3.5 | 22 |
| Yak-9 | 8.55 m | 537 km/h | 156 km/h | 20.6°/s | 209 m | 382 km/h | 31.3°/s | 1.52 | 0.50 | 0.71 | 3.4 | 24 |
| Bf 109 G-6 | 8.95 m | 529 km/h | 168 km/h | 16.7°/s | 257 m | 412 km/h | 29.1°/s | 1.74 | 0.51 | 0.78 | 3.2 | 29 |
| P-51D-15 | 9.83 m | 592 km/h | 177 km/h | 18.0°/s | 256 m | 434 km/h | 27.6°/s | 1.53 | 0.49 | 0.73 | 3.3 | 26 |
| Fw 190 A-8 | 9.00 m | 558 km/h | 192 km/h | 14.9°/s | 300 m | 470 km/h | 25.5°/s | 1.71 | 0.50 | 0.84 | 2.9 | 33 |
| P-47D-28 | 11.0 m | 557 km/h | 196 km/h | 13.1°/s | 391 m | 480 km/h | 24.9°/s | 1.90 | 0.58 | 0.86 | 2.8 | 36 |
| Me 262 A | 10.6 m | 759 km/h | 182 km/h | 10.7°/s | 666 m | 446 km/h | 26.9°/s | 2.50 | 0.59 | 0.59 | 4.2 | 63 |

Stall is the midpoint of the database's range. Sustained rate comes from the
database's best 360° time at sea level.

### C.2 Other figures

* **F-16 Block 50, clean:** sustained 21.5°/s at 9 g and Mach 0.7; instantaneous about
  25°/s; corner 350–450 KCAS. Its instantaneous/sustained ratio is about 1.2.
  Fly-by-wire and high thrust flatten the gap that defines the WWII feel.
* **Roll rates** at combat speed are roughly 60–160°/s (Fw 190 near 160, P-47D about
  85 at 250 mph; Spitfire figures are disputed). Banking 60–90° takes about 0.4–1.5 s
  before the pull builds.
* **Muzzle speeds:** AN/M2 .50 calibre 866 m/s; Hispano Mk II 840–880 m/s; MG 151/20
  700–785 m/s.
* **Effective gun ranges:** 250–400 m, about 25–40 ship lengths.

## Appendix D. Sources

* Hafner et al., *Mastering Diverse Domains through World Models* (DreamerV3);
  settings in §7.3 verified against `github.com/danijar/dreamerv3`.
* Hafner, Yan et al., *Training Agents Inside of Scalable World Models* (Dreamer 4),
  arXiv 2509.24527.
* Ye et al., *Towards Playing Full MOBA Games with Deep RL* (Honor of Kings), NeurIPS
  2020.
* Berner et al., *Dota 2 with Large Scale Deep RL* (OpenAI Five), 2019.
* Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*.
* Farebrother et al., *Stop Regressing: Training Value Functions via Classification*,
  2024.
* Sreenivasan and Fiete, *Grid cells generate an analog error-correcting code*, 2011
  (residue codes and their failure mode).
* IL-2 Great Battles aircraft database, v5.204, `aergistal.github.io/il2/`: Spitfire
  Mk.IXc, Yak-9 ser.1, Bf 109 G-6, P-51D-15, Fw 190 A-8, P-47D-28, Me 262 A.
* Wikipedia, "Aircraft of the Battle of Britain" (RAE turning-circle estimates);
  "Messerschmitt Bf 109" (G-6 dimensions); "AN/M2", "Hispano-Suiza HS.404" and
  "MG 151 cannon" (muzzle velocities).
* F-16 turn figures: f-16.net, "F-16 Sustained Turn Performance" and "F-16
  maneuverability data"; boltflight.com, "F-16 Turn Rate".
* Roll rates: NACA Report 868 (1947), as discussed on ww2aircraft.net.
* In this repository: `next-state-head-audit-sep2026.md`,
  `perception-belief-audit-sep2026.md`,
  `docs/engineering/frontline-density-and-front-dynamics.md`,
  `flight-envelope-baseline-oct2026.json`, `benchmarks/flight_envelope.py`,
  `benchmarks/flight_spec.py`.
