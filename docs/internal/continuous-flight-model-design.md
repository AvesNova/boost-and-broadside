# Continuous flight model design

Status: **proposed design, not implemented.** Base: `docs/reward-belief-redesign` at
`54659e8`.

Today the turn action sets the nose angle instantly. This note replaces that with a nose
angle that moves continuously, plus a lateral-acceleration cap (a g-limit). The target
feel is a responsive Second World War fighter:

* **Low and medium speed.** You can throw the nose round hard for half a second, get a
  shot off, and it snaps back to the flight path.
* **High speed.** The nose moves only a few degrees, and any real direction change is a
  committed, wide turn.
* **Very low speed.** You cannot turn much at all.

The settled turns below corner speed stay exactly what they are today. The changes are
the transients, and the turn rate above corner speed.

Out of scope for now: thrust vectoring, and a roll stage before the pull.

---

## 1. The current model

Read from `env/physics.py` (`_update_kinematics`, `_update_kinematics_in_fields`).
Measured numbers for the live configuration are in Appendix A, real aircraft in
Appendix B, and the targets they suggest in §8.

* **Nose angle is a lookup.** Attitude is `velocity_direction · e^{i·offset}`, with
  offset 0, ±5° or ±15° from `turn_offset_table`. It jumps the full amount in one tick.
  A left-to-right sharp reversal moves the gun 30° in one tick.
* **Slip does not drive the aerodynamics.** Lift and drag coefficients come from
  per-action tables (`drag_coeff_table`, `lift_coeff_table`). The offset only sets the
  thrust direction, the gun direction, the hit-angle damage scaling and rendering.
* **Turn radius does not depend on speed.** Lift rotates the velocity by
  `lift_coeff · |v| · dt` per tick, so curvature is `lift_coeff` and turn rate scales
  with speed. Appendix A of `frontline-reward-critic-belief-design.md` measured
  66.2 px (normal) and 35.3 px (sharp), whatever the index and power. Faster is
  therefore strictly better in every turning fight: same radius, higher rate.
* **`ship_ang_vel` echoes the action.** It is `turn_offset / dt`, not the heading
  rate.
* **Air brakes.** `TurnActions.AIR_BRAKE` and `SHARP_AIR_BRAKE` add drag with no offset
  or lift. Nothing outside `constants.py` refers to them by name.
* **Pseudo-altitude already exists.** `E = ½n²|v|² + power_speed_constant · power` is
  conserved across thrust and reverse. Boost acts as a dive and reverse as a zoom climb,
  so a g-limit gives a complete energy-manoeuvrability picture with no new mechanics.

## 2. Design

### 2.1 State

Two per-ship scalars carry the dynamics:

* **slip** α, the nose angle relative to the velocity direction (new field);
* **slip rate** α̇ (new field).

Attitude is rebuilt at the start of each tick as `velocity_direction · e^{iα}`, exactly
as today with α in place of the lookup offset. Once α settles at today's offset, the
per-tick force equations are identical to today's. That is what keeps the envelope
unchanged below corner speed, by construction rather than by tuning.

`ship_ang_vel` becomes the real heading rate: velocity turn rate plus α̇.

### 2.2 Commands: two pull levels and a g-cap

Air brakes are removed. The turn actions are straight, then left and right at two pull
levels:

* **sharp** = maximum pull;
* **normal** = half the available pull.

The command is a lift coefficient. It is capped so that lateral acceleration
`C_L · u²` does not exceed `max_lateral_accel`:

```
sharp_cl  = min(C_L_max,           max_lateral_accel / u²)
normal_cl = min(normal_turn_lift,  max_lateral_accel / (2 u²))
alpha_cmd = sign · lift_inverse(cl)          # closed form, §2.3
```

Corner speed is `sqrt(max_lateral_accel / C_L_max)`:

* **Below it**, both levels command today's 5° and 15° (normal only below its own
  corner, 95 px/s at the chosen numbers).
* **Above it**, the nose angle shrinks with 1/u² and the radius grows with u².

The cap uses proper speed `u = n|v|`. This keeps the corner proper speed independent of
the index, and keeps world turn rate reciprocal in n at a fixed proper speed, matching
today's field behaviour.

**Corner speed is 100 px/s** (`max_lateral_accel` = 0.027 · 100² = 270 px/s²), chosen
in §8.2. Worked numbers at n = 1 follow. The radii here are lift-only; the measured radii
include about 5% from thrust.

| u (px/s) | normal α | sharp α | normal R | sharp R | sharp rate, today | sharp rate, capped |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 5.0° | 15.0° | 67 | 37 | 62°/s | 62°/s |
| 100 | 4.4° | 15.0° | 74 | 37 | 155°/s | 155°/s |
| 120 | 2.9° | 6.7° | 107 | 53 | 186°/s | 129°/s |
| 136 | 2.2° | 4.8° | 137 | 69 | 210°/s | 114°/s |
| 160 | 1.5° | 3.3° | 190 | 95 | 248°/s | 97°/s |
| 215 | 0.8° | 1.7° | 342 | 171 | 333°/s | 72°/s |
| 300 | 0.4° | 0.9° | 667 | 333 | 464°/s | 52°/s |

The normal level's own corner is sqrt(270 / (2 · 0.015)) ≈ 95 px/s, so normal turns
begin to soften just below 100 px/s. One settled turn therefore changes: the boosted
normal turn, which settles at 118 px/s today (Appendix A.2), will widen and slow. Every
coasting turn and the boosted sharp turn (73 px/s) settle below both corners and stay
as they are.

The parabolic lift curve is flat near its peak, so the sharp nose angle falls quickly
just above corner speed: 15° at 100 px/s, 6.7° at 120. That gives high-speed flight its
"committed" feel.

### 2.3 Lift and drag from the actual slip

With `x = |α| / stall_angle` and `stall_angle` = 15°:

```
C_L(α) = sign(α) · C_L_max · (1 − (1 − x)²)      for x ≤ 1
C_D(α) = C_D0 + b·x² + c·x⁴
```

* **Lift.** The fit is exact at both of today's points: `C_L_max` = 0.027 gives 0.0150
  at 5° and 0.0270 at 15°. It inverts in closed form:
  `α = stall_angle · (1 − sqrt(1 − C_L/C_L_max))`.
* **Drag.** `C_D0` = 8e-4, b = 3.525e-3 and c = 6.75e-4 give 0.0012 at 5° and 0.0050 at
  15°, matching `normal_turn_drag_coeff` and `sharp_turn_drag_coeff`. The curve is
  monotonic.
* **Past stall (x > 1).** Lift falls on the mirrored parabola, clamped at zero.
  Commands never exceed 15° (§7, question 1), so only an overshoot could reach this
  branch, and the response in §2.4 has none.

Lift and drag are applied exactly as now: drag as the exact quadratic-drag scale, lift as
an exact rotation of the velocity. Only the coefficients change, from per-action lookups
to functions of α.

### 2.4 Nose response: critically damped, rate-limited

α tracks `alpha_cmd` as a critically damped second-order system with a constant natural
frequency ω. It is stepped exactly, holding the command constant over the tick.

With `decay = exp(−ω·dt)` and `err = α − alpha_cmd`:

```
err'   = decay · ((1 + ω·dt) · err + dt · α̇)
α̇'     = decay · (−ω²·dt · err + (1 − ω·dt) · α̇)
α'     = α + clamp(alpha_cmd + err' − α, ±max_slip_rate · dt)
α̇'     = clamp(α̇', ±max_slip_rate)
```

* **No overshoot or ringing**, because the system is critically damped. The exact step
  is stable for any ω·dt, so nothing needs substeps.
* **ω, dt and `decay` are config constants**, so the step matrix is four Python floats.
* **Onset.** The time to reach 90% is about 3.9/ω, whatever the size of the step:

  | ω (rad/s) | 90% onset | decisions at 30 Hz | ω·dt |
  |---:|---:|---:|---:|
  | 20 | 194 ms | 5.8 | 0.67 |
  | 25 | 156 ms | 4.7 | 0.83 |
  | 30 | 130 ms | 3.9 | 1.00 |

  Start at 25 rad/s.
* **Rate limit.** `max_slip_rate` makes reversals slower than onset without adding
  state.
  * From straight to a 15° sharp turn, the peak rate is ω · step / e (Euler's number), about
    138°/s at ω = 25.
  * A 30° reversal would peak at about 276°/s.
  * A limit near 150°/s leaves onset alone and stretches a full reversal to about
    0.25 s.
* **Release.** The command goes to zero and the nose returns to the flight path over the
  same ~0.16 s. This is the snap-back after a nose throw.

The frequency is deliberately **not** scaled with speed. Scaling it with u would make
low-speed nose throws sluggish, which is the opposite of the target feel. Speed
dependence comes from the g-cap (§2.2) and from the low-speed fade (§2.5).

### 2.5 Low speed

* **Fade.** Below `slip_fade_speed`, the command amplitude is scaled by
  `clamp(u / slip_fade_speed, 0, 1)`, so the nose stays on the flight path.
  * The lowest sustained sharp-turn coast speed is 19.7 px/s world at n = 2, which is
    39 px/s proper.
  * Keep `slip_fade_speed` below about 35 so no sustained turn changes.
* **Stall.** Below `min_speed`, keep today's rule: lift is zero, and the absolute
  attitude is held. α is re-derived as `angle(attitude · conj(velocity_direction))`
  with α̇ = 0, so leaving the stall does not jump.

### 2.6 Removing the air brakes

* `TurnActions` loses both air-brake entries. The joint action id in `runtime/actions.py`
  changes with `NUM_TURN_ACTIONS`.
* Slowing down still comes from reverse thrust, and from the drag of sharp pulls.

### 2.7 Fixing the ambient path

Appendix A.9 records a defect in the no-field path (`num_fields == 0`):
`_update_kinematics` applies lift as an explicit velocity kick, which multiplies speed by
sqrt(1 + (c_L·u·dt)²) every tick. It also drains power linearly in speed, not by exact
work. Lift therefore adds energy, a boosted normal turn outruns straight flight, and
with power held full the turn runs away to NaN.

* **Fix by deletion.** Remove `_update_kinematics`. With no fields, run the field path's
  flight half-steps (`_apply_field_flight_half_step`) around a plain drift
  `pos += v·dt` in place of `_transport_through_fields`. With n = 1 and zero gradient
  that is the field path exactly: exact drag scale, exact lift rotation, exact-work
  thrust, and the same stall rule for reverse. One flight path remains, which is what
  §4 relies on.
* **Order.** Land it as its own commit before the slip change. §6 then compares one
  change at a time, and the field-path baseline in Appendix A does not move.
* **Tests.**
  * Lift alone (zero thrust and drag) keeps speed constant over 10 000 ticks at
    470 px/s and 60 Hz.
  * Generalized energy never rises without thrust or regeneration.
  * A held-power normal turn stays finite.
  * The no-field step matches a probe at n = 1 to float tolerance at the same dt.
* **Fallout.**
  * Any test that pins ambient-path numbers moves to the corrected values.
  * No profile runs this path, so throughput and trained checkpoints are unaffected.

## 3. What changes in play

* **Turning fights get an energy trade-off.**
  * In a two-circle (rate) fight, the best speed is corner speed rather than "as fast as
    possible".
  * In a one-circle (radius) fight, the slower ship wins once the faster one is above
    corner speed.
  * Boom-and-zoom becomes literal: dive (boost) to corner speed, turn, zoom (reverse)
    back up. This is the largest balance change in the note.
* **Snap shots favour the ship at or below corner speed.** Today any ship at any speed
  moves its gun 15° in one tick. With this change the gun takes ~0.16 s to get there,
  and above corner speed it reaches only a few degrees. Expect lower hit rates and
  longer time-to-kill in fast passes.
* **Settled turns below corner speed are unchanged** (§2.1). Onset alone changes the
  first ~0.16 s of each turn, which barely moves sustained-turn outcomes.

## 4. Compilation and GPU constraints

The env tick must stay compatible with `torch.compile(dynamic=False)` and with CUDA-graph
capture (`env/cuda_graph.py`). Graph breaks, recompiles and host syncs on the physics
hot path are regressions.

* **No data-dependent control flow.** The g-cap, fade, stall gate, rate limit and
  post-stall branch are all `torch.minimum`, `clamp` and `torch.where`. There is no
  `.item()`, no boolean indexing and no `nonzero`.
* **Constants come from `ShipConfig`.** It is frozen and hashable. All new parameters
  are plain Python floats there, so they bake into the graph and do not cause
  recompiles. Per-action command levels extend the existing `_get_lookup_tables` cache;
  no tensors are built per step.
* **No iterative solves.** The lift inverse is a `sqrt`, and the damped step is
  multiply-adds with precomputed coefficients. The only transcendental per ship per tick
  is the `polar` that builds the attitude, as today; `angle` runs only in the stall
  re-derivation, inside a `torch.where`.
* **Static shapes.** The two new fields are `(B, N)` float32, allocated at reset and
  zeroed on respawn next to `ship_ang_vel` (`env.py`, `frontline.py`).
* **Reassignment, not in-place mutation.** The fields advance by reassignment, per the
  `TensorState` carve-out in the style guide. `CapturedTick` copies every `TensorState`
  field back after capture, so the new fields need no special handling there.
* **One flight path.** After §2.7 the ambient case (`num_fields == 0`) runs the field
  path's flight steps with no transport, so the slip update exists once and the two
  cannot diverge.
* **Cost** is a few dozen elementwise operations on `(B, N)`, which is negligible next
  to collisions and observation. Confirm it with the `rl-throughput` benchmark, not by
  assumption.

## 5. Effects elsewhere

* **`frontline-reward-critic-belief-design.md`.**
  * §3.6: an enemy's next attitude is no longer a mixture of five spikes. It moves
    smoothly, which makes the finer attitude levels more predictable.
  * §3.7: angular velocity is no longer five exact values and needs a continuous
    binning. Its range is about ±(`max_lateral_accel` / corner speed + `max_slip_rate`),
    roughly ±305°/s at the numbers above (155°/s + 150°/s).
  * Appendix A must be re-measured above corner speed.
* **Observation.** The `ang_vel` normalisation was chosen for the action echo and must be
  re-derived. Slip is visible as attitude minus velocity heading, so the state remains
  Markov with α̇ carried by `ang_vel`.
* **Action space.** The turn head and the joint action id shrink. Checkpoints, BC data
  and replays recorded with air-brake actions become invalid.
* **Scripted agents.** `scripted_utils.turn_toward` switches between turn levels at 5° and 15°
  of heading error, which assumes the nose responds instantly. With a 0.16 s lag it will
  chatter. Add hysteresis or a lead term, and re-tune each scripted opponent against the
  new turn-rate curve.

## 6. Validation

Use a single-ship harness that calls `update_ships` directly, with power held at
maximum. Compare before and after by git ref; per §6.8 of the style guide there is no
runtime switch to the old model.

1. **Settled envelope.** Turn rate, radius and speed against proper speed for both pull
   levels, at n ∈ {0.5, 1, 2}. They should match today to within tolerance below corner
   speed, and match §2.2 above it.
2. **Transients.**
   * Onset: 90% time from straight to each level.
   * Reversal time.
   * Snap-back time after release.
   * Nose throw: hold sharp for 0.5 s from straight at 60, 100 and 160 px/s; record the
     peak nose angle off the original velocity and the path's heading change.
3. **No jitter.** Zero sign changes of α̇ during a held turn at any speed, and α never
   past `stall_angle`.
4. **Compile.** Zero graph breaks and no recompiles across a rollout. Captured-tick
   replay matches eager within float tolerance. Throughput is within noise of today.
5. **Play.** Scripted duels: hit rate, time-to-kill, mean speed and time spent above
   corner speed.
6. **Ambient path.** The §2.7 tests pass. The `flight_envelope.py` reference block
   (`terminal_reference_60hz_ambient`) reproduces the field path at n = 1: energy-limited
   boost near 135 px/s rather than 224, and no missing rows.

## 7. Open questions

1. **Deep nose throws.** Should sharp go past stall at low speed (a 25–30° ceiling,
   where lift drops and drag rises steeply)? That is closer to the real trade-off, and it
   limits itself, but it changes the sustained low-speed sharp turn into a decelerating
   slide. Start at 15° and try a deeper ceiling as a variant.
2. **Corner speed.** Settled at 100 px/s (§8.2): coast cruise, and 0.74 of the
   energy-limited boost terminal at n = 1. The scripted controller is above it on 61% of
   decisions (Appendix A.8), so the cap shapes most fast flight. Revisit only if the
   §6.5 duels show fast passes becoming unusable.
3. **Landing.** The reward, critic and belief redesign already forces a cold start and a
   checkpoint-schema bump. Folding this change into the same cold start avoids a second
   one. Validate the physics with scripted duels first (§6), since those need no
   learning.
4. **Thrust vectoring.** Deferred. If it returns, cap its rotation rate low and scale its
   torque with applied thrust. Boost from near zero speed is almost free under the energy
   law, so a near-stationary turret is the abuse case to test.

## 8. Targets

### 8.1 Comparing a game with aircraft

Absolute numbers do not carry over: the game turns far faster and in far less room than
an aircraft. Ratios do carry over, once two scales are fixed.

* **Length.** One ship length L = 20 px (renderer `ship_size` 10 px centre to tip;
  `collision_radius` 10 px).
* **Time.** The best sustained 360° takes 3.0 s in the game (boost, sharp, n = 1) and
  17.5–27.5 s for the WWII piston fighters in Appendix B. Rotation therefore runs about
  **k ≈ 6–7 times faster** than real. Coasting turns (5.1–5.7 s per 360°) give k ≈ 4.

| quantity | today | §2 (corner 100) | WWII piston fighters |
|---|---:|---:|---:|
| best instantaneous / best sustained turn rate | 1.75 at 136 px/s, unbounded above | 1.30 | 1.5–1.9 |
| speed of best sustained turn / top speed | 0.54 | 0.54 | 0.49–0.59 |
| corner speed / top speed | none | 0.74 | 0.69–0.86 |
| turn rate at top speed / at corner | > 1 at every speed | 0.74 | 0.69–0.86 |
| top speed / lowest flying speed | 136 (`min_speed` 1) | ≈ 3.9 (fade at 35) | 2.8–3.5 |
| turn onset, straight to full pull | 1 tick (33 ms) | 0.16 s | ≈ 0.6–1.5 s; ÷ k ≈ 0.1–0.2 s |
| speed lost per second in a max-rate turn at corner | 42% at 100 px/s | 42% | ≈ 9%; × k ≈ 55–60% |
| sustained radius / L | 1.8 (sharp), 3.3 (normal) | same below corner | 21–36 |
| gun reach / L | 24.5 | 24.5 | ≈ 25–40 |
| sustained radius / gun reach | 0.07 | 0.07 below corner, 0.35 at 215 px/s | ≈ 0.5–1 |
| top speed, L per second | 6.8 (10.5 on a dash) | 6.8 | 14–20 |
| muzzle speed / top speed | 3.7 | 3.7 | 4.5–6 |

"Top speed" is the energy-limited boost terminal, 136 px/s at n = 1. A full-power dash
from cruise peaks at 209 px/s and lasts about 1.9 s (A.5). For aircraft the corner speed
is taken at a 6 g pilot limit, and turn onset is roll-to-bank plus the pull. The bleed
estimate assumes a lift-to-drag ratio of about 5 at maximum lift and a thrust-to-weight
ratio of 0.3, so treat it as an order of magnitude.

What the comparison says:

* **Already in range:** the speed of the best sustained turn, and, to within its rough
  estimate, the bleed in a hard turn once time is scaled.
* **Brought into range by §2:** a corner speed and its falling turn rate above it, a
  low-speed floor, and onset time once time is scaled. The proposed ω = 25 rad/s lands
  inside the scaled real range without being tuned to it.
* **Moved toward range but short:** instantaneous over sustained turn rate is 1.30
  against 1.5–1.9. That is the price of the 100 px/s corner (§8.2): the gap between
  the best sustained turn (73 px/s) and the corner is small.
* **Not addressed by §2: compactness.** Turning circles are about 12 times tighter
  than an aircraft's relative to ship length and gun reach. A sharp 180° fits in about
  a sixth of the ship's own gun reach. §2 keeps the radii below corner by construction
  and only opens them above it. Closing the gap fully would mean sustained radii of
  250–500 px, half the vision range, which is a different game. Decide it explicitly; do
  not let it drift in through tuning.
* **Speed is low relative to turning.** At 6.8 L/s against 14–20 L/s, passes are slow
  and turning dominates. Boom-and-zoom (§3) depends on the dash; the current energy law
  allows one about 2 s long.

### 8.2 Candidate parameter sets

These are all the §2 model. The values are lift-only and analytic, at n = 1. Sustained
rates come from the measured boost sharp equilibrium (73 px/s), which sits below every
corner speed here.

| | A: corner 120 | **B: corner 100 (chosen)** | C: roomier |
|---|---:|---:|---:|
| `C_L_max` | 0.027 | 0.027 | 0.0135 |
| corner speed (px/s) | 120 | 100 | 120 |
| `max_lateral_accel` (px/s²) | 389 | 270 | 194 |
| best sustained sharp rate | 119°/s | 119°/s | ≈ 60°/s |
| sharp rate at corner | 186°/s | 155°/s | 93°/s |
| instantaneous / sustained | 1.56 | 1.30 | ≈ 1.55 |
| sharp rate at 136 / 209 px/s | 164 / 107°/s | 114 / 74°/s | 82 / 53°/s |
| corner / top speed (136) | 0.88 | 0.74 | 0.88 |
| sharp radius at corner / at 209 px/s | 37 / 112 px | 37 / 162 px | 74 / 225 px |
| scripted decisions above corner | 42% | 61% | 42% |
| scripted p99 lateral acceleration (434) / cap | 1.1 | 1.6 | 2.2 |

* **A** keeps the most of today's game and puts corner/top at the upper edge of the real
  range.
* **B**, the choice, matches the real ratios best on corner/top and rate-at-top. It gives
  up some of the instantaneous-to-sustained gap (1.30), and the cap binds on most fast
  decisions.
* **C** halves lift. Every radius doubles and every turn rate halves, and the sustained
  envelope changes. It is the only set that moves compactness at all, and only by 2×. It
  breaks §2.1's "settled turns unchanged", so it needs its own cold-start justification.

The index matters. The energy-limited boost terminal is 107 px/s (proper) at n = 0.5
and 171 at n = 2. A corner at 100 sits just below top speed at n = 0.5 and well below
it at n = 2, so the cap binds in every medium, most strongly in high-index fields (18%
of scripted time).

Decision (2026-10-05): corner speed 100 px/s, set B. The §6.5 duels report time above
corner and hit rate against today. A is the fallback if fast passes stop producing
kills. C stays in reserve until the compactness question is decided.

### 8.3 Acceptance numbers for §6

* **Settled envelope below corner:** terminal speed, turn rate and radius within 2% of
  A.2 at every index, except the boosted normal turn (§2.2) and the power-held-full
  rows, which settle above corner.
* **Onset:** 90% of the commanded slip in 0.13–0.20 s.
* **Snap-back:** release to within 1° of the flight path in 0.13–0.20 s. Today this is
  one tick (A.6).
* **Nose throw:** 0.5 s sharp from straight at 100 px/s. Compare the path heading change
  (71° today) and the gun's lead over the path (10° today).
* **Re-measure:** run `benchmarks/flight_envelope.py` before and after, and diff it
  against `docs/internal/flight-envelope-baseline-oct2026.json`.

---

## Appendix A. Measured baseline

Measured with `benchmarks/flight_envelope.py` at `f3e0f1f`. The raw output is
`docs/internal/flight-envelope-baseline-oct2026.json`. The configuration is the live one:
`frontline_ship_config(SHIP_CONFIG)`, dt = 1/30, field path. Each probe ship sits at the
centre of one large uniform field of index n and is re-centred every tick. Speeds are
proper (u = n|v|) and rates are world °/s, unless stated otherwise.

### A.1 Straight-line speeds (px/s, proper)

| n | coast | boost, energy-limited | dash peak from cruise (time) | boost, power held full |
|---:|---:|---:|---:|---:|
| 0.5 | 100 | 107 | 203 (0.9 s) | 315 |
| 0.707 | 100 | 121 | 206 (1.3 s) | 315 |
| 1 | 100 | 135 | 209 (1.9 s) | 316 |
| 1.414 | 100 | 152 | 214 (2.8 s) | 316 |
| 2 | 100 | 171 | 221 (4.3 s) | 316 |

The coast terminal is sqrt(`base_thrust`/`no_turn_drag_coeff`) at every index. Holding
power full is not a game state; it bounds what a ship with power left approaches.

### A.2 Sustained turns (n = 1)

| power | turn | speed | rate | radius | 360° |
|---|---|---:|---:|---:|---:|
| coast | normal | 81.5 | 70.4°/s | 66.3 px | 5.1 s |
| coast | sharp | 39.3 | 63.7°/s | 35.3 px | 5.7 s |
| boost | normal | 118.3 | 102.5°/s | 66.3 px | 3.5 s |
| boost | sharp | 73.2 | 119.2°/s | 35.4 px | 3.0 s |
| power held full | normal | 257.3 | 222.1°/s | 66.4 px | 1.6 s |
| power held full | sharp | 124.1 | 200.4°/s | 35.5 px | 1.8 s |

Radius is independent of index and power. World turn rate at the same proper speed is
reciprocal in n: boost sharp turns at 189°/s at n = 0.5 and 75°/s at n = 2.

### A.3 Instantaneous performance (one tick, coast, n = 1)

| u (px/s) | 40 | 100 | 120 | 136 | 160 | 300 | 600 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sharp rate (°/s) | 65 | 154 | 184 | 208 | 244 | 448 | 864 |
| sharp lateral accel (px/s²) | 45 | 267 | 382 | 489 | 672 | 2 291 | 8 641 |
| sharp du/dt (px/s²) | −0.3 | −42 | −63 | −83 | −117 | −421 | −1 630 |
| normal rate (°/s) | 35 | 86 | 103 | 117 | 137 | 256 | 507 |
| normal du/dt (px/s²) | +6.0 | −4.0 | −9.3 | −14 | −23 | −99 | −414 |

There is no corner. Rate rises linearly with speed without limit, and lateral
acceleration with its square. Boost adds about 70 px/s² along track at 100 px/s, so a
boosted sharp turn holds speed up to about 125 px/s. The air brakes have the drag of the
matching turn with no lift. The JSON's `excess_power_accel` includes passive power
regeneration (2 000 px²/s³ of energy), which stops once power is full.

### A.4 Turning from straight flight (n = 1, full power)

Time to turn 180°, in seconds:

| entry u (px/s) | 40 | 70 | 100 | 136 | 180 | 215 | 300 |
|---|---:|---:|---:|---:|---:|---:|---:|
| coast, sharp | 2.80 | 1.97 | 1.47 | 1.13 | 0.87 | 0.73 | 0.53 |
| boost, sharp | 1.37 | 1.17 | 1.03 | 0.87 | 0.73 | 0.63 | 0.50 |
| coast, normal | 4.07 | 2.87 | 2.17 | 1.67 | 1.30 | 1.10 | 0.80 |
| boost, normal | 1.90 | 1.67 | 1.47 | 1.27 | 1.07 | 0.93 | 0.73 |

The lateral offset at 180° is 66–75 px for sharp turns and 129–134 px for normal turns,
whatever the entry speed. A faster entry turns the same 180° sooner in the same space.
Times scale with n.

### A.5 Acceleration and braking (n = 1)

* **Boost from rest:** 100 px/s in 1.3 s, 136 in 1.8 s; peak 199 at 2.9 s, when power runs
  out.
* **Boost from cruise at full power:** 136 in 0.53 s, 200 in 1.67 s; peak 209 at 1.87 s,
  when power runs out.
* **From 215 px/s to 100:**

  | method | time |
  |---|---:|
  | coast | 43.6 s |
  | air brake | 7.6 s |
  | sharp air brake | 1.2 s |
  | reverse | 1.17 s |
  | reverse with sharp air brake | 0.6 s |

* **Reverse from cruise to standstill** takes 1.2 s. It returns 33 power within the
  first second: 25 from kinetic energy, the rest from regeneration.
* **Index.** Times scale with n (mass n²). At n = 2 the boost from cruise peaks at
  4.3 s.

### A.6 Gun response (coast, n = 1)

* **Onset.** The gun steps 15° in one tick (450°/s).
* **Reversal.** Sharp left to sharp right moves it 21–28° in one tick (630–835°/s),
  less at higher speed.
* **Release.** The slip returns to zero in one tick.

Nose throw, 0.5 s of sharp from straight flight:

| u (px/s) | 40 | 60 | 100 | 160 | 215 | 300 |
|---|---:|---:|---:|---:|---:|---:|
| gun at 0.1 s | 19° | 21° | 25° | 31° | 36° | 44° |
| gun at 0.5 s | 45° | 58° | 81° | 114° | 141° | 179° |
| path at 0.5 s | 32° | 46° | 71° | 105° | 134° | 173° |

Today's nose throw is mostly a real path turn: above about 60 px/s the path has turned
further than the 15° slip within 0.15 s.

### A.7 Bullets (n = 1, no spread)

| ship speed (px/s) | 0 | 100 | 215 |
|---|---:|---:|---:|
| reach in 1 s (px) | 421 | 490 | 565 |
| time to 200 px (s) | 0.47 | 0.37 | 0.33 |

The muzzle speed is 500 px/s relative to the ship. Drag leaves the bullet with 64–71% of
its launch speed at expiry.

### A.8 Scripted play

`StochasticScriptedAgent` against itself on the `rl` profile: 64 environments, 2 seeds,
1 500 decisions each, 1.92 M ship-decisions.

* **Proper speed:**

  | p1 | p5 | p25 | p50 | p75 | p95 | p99 |
  |---:|---:|---:|---:|---:|---:|---:|
  | 28 | 41 | 80 | 113 | 133 | 152 | 166 |

  Above 100: 61%; above 120: 42%; above 136: 19%; above 160: 2.4%; above 215: 0.
* **Heading rate:** zero on 59% of decisions; |rate| p75 75°/s, p95 154, p99 217.
  Lateral acceleration p95 292 px/s², p99 434.
* **Actions:** boost straight 46%; normal turns 21%; sharp turns 20%; air brakes 0%.
* **Local index:** n = 1 50%, n = 2 18%, n = 1.414 10%, n = 0.707 7%, n = 0.5 7%, in a
  transition 7%.

Trained-policy usage is not measured here. Air-brake usage in particular should be
checked on a policy before §2.6 removes them.

### A.9 The ambient path

With no fields (`num_fields == 0`), `_update_kinematics` integrates lift as an explicit
velocity kick. Each tick multiplies speed by sqrt(1 + (c_L·u·dt)²), so lift adds energy.

* At 60 Hz, the energy-limited boost terminal there is 224 px/s, against 135 on the
  field path.
* A boosted normal turn settles faster (228 px/s) than straight flight.
* With power held full, a normal turn runs away and reaches NaN after about 29 s.

No current profile takes this path. Every profile has fields, and the field path rotates
the velocity exactly. §2.7 removes it.

## Appendix B. Reference aircraft

### B.1 Data

Sea-level figures for six WWII piston fighters and one early jet, from the IL-2 Great
Battles aircraft database (v5.204). That database is simulator data, compiled from
historical tests. The cross-check holds: the derived Spitfire sustained radius (211 m)
matches the RAE's 700 ft (212 m) estimate. Lengths are from the manufacturers'
specifications.

| aircraft | length | top speed | stall | 360° (speed) | max load |
|---|---:|---:|---:|---:|---:|
| Spitfire Mk IXc | 9.47 m | 542 km/h | 143–163 km/h | 17.7 s (270 km/h) | 12.5 g |
| Yak-9 | 8.55 m | 537 km/h | 152–160 km/h | 17.5 s (270 km/h) | 10.5 g |
| Bf 109 G-6 | 8.95 m | 529 km/h | 160–177 km/h | 21.5 s (270 km/h) | 10.5 g |
| P-51D-15 | 9.83 m | 592 km/h | 159–196 km/h | 20.0 s (290 km/h) | 10 g |
| Fw 190 A-8 | 9.00 m | 558 km/h | 177–208 km/h | 24.2 s (280 km/h) | 11 g |
| P-47D-28 | 11.0 m | 557 km/h | 178–215 km/h | 27.5 s (322 km/h) | 11 g |
| Me 262 A | 10.6 m | 759 km/h | 165–200 km/h | 33.5 s (450 km/h) | 12.5 g |

The maximum load is the database's structural figure. The comparison uses 6 g as the
practical limit for a pilot without a g-suit.

### B.2 Derived values

The corner speed is the stall speed × √6. The instantaneous rate at corner is
g·√35 / V_c. Ratios use the same top speed as above.

| aircraft | sustained rate | sustained radius | load | corner | rate at corner | inst/sust | turn/top | corner/top | top/stall | top (L/s) | radius / L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Spitfire IXc | 20.3°/s | 211 m | 2.9 g | 375 km/h | 31.9°/s | 1.57 | 0.50 | 0.69 | 3.5 | 15.9 | 22 |
| Yak-9 | 20.6°/s | 209 m | 2.9 g | 382 km/h | 31.3°/s | 1.52 | 0.50 | 0.71 | 3.4 | 17.4 | 24 |
| Bf 109 G-6 | 16.7°/s | 257 m | 2.5 g | 412 km/h | 29.1°/s | 1.74 | 0.51 | 0.78 | 3.2 | 16.4 | 29 |
| P-51D-15 | 18.0°/s | 256 m | 2.8 g | 434 km/h | 27.6°/s | 1.53 | 0.49 | 0.73 | 3.3 | 16.7 | 26 |
| Fw 190 A-8 | 14.9°/s | 300 m | 2.3 g | 470 km/h | 25.5°/s | 1.71 | 0.50 | 0.84 | 2.9 | 17.2 | 33 |
| P-47D-28 | 13.1°/s | 391 m | 2.3 g | 480 km/h | 24.9°/s | 1.90 | 0.58 | 0.86 | 2.8 | 14.1 | 36 |
| Me 262 A | 10.7°/s | 666 m | 2.6 g | 446 km/h | 26.9°/s | 2.50 | 0.59 | 0.59 | 4.2 | 19.9 | 63 |

### B.3 Modern jet and other figures

* **F-16 (Block 50, clean),** for contrast:
  * sustained 21.5°/s at 9 g and Mach 0.7;
  * instantaneous about 25°/s;
  * corner 350–450 KCAS;
  * inst/sust about 1.2, sustained radius about 42 L.

  Fly-by-wire and high thrust-to-weight flatten the instantaneous-to-sustained gap
  that defines the WWII feel.
* **Roll rates** at combat speed are roughly 60–160°/s. The Fw 190 is near 160°/s and
  the P-47D about 85°/s at 250 mph. Spitfire figures are disputed between NACA Report
  868 and other tests. Banking 60–90° therefore takes about 0.4–1.5 s before the pull
  builds.
* **Muzzle speeds:** AN/M2 .50 calibre 866 m/s; Hispano Mk II 840–880 m/s; MG 151/20
  700–785 m/s. That is about 4.5–6 times top speed. Effective firing ranges of
  250–400 m are about 25–40 ship lengths.

## Appendix C. Sources

* IL-2 Great Battles aircraft database, v5.204: `aergistal.github.io/il2/`. Pages used:
  Spitfire Mk.IXc, Yak-9 ser.1, Bf 109 G-6, P-51D-15, Fw 190 A-8, P-47D-28 and Me 262 A.
* RAE turning-circle estimate for the Spitfire (700 ft at 12 000 ft) and the Bf 109E:
  Wikipedia, "Aircraft of the Battle of Britain".
* Bf 109 G-6 dimensions: Wikipedia, "Messerschmitt Bf 109", specifications.
* F-16 turn figures: f-16.net forum threads "F-16 Sustained Turn Performance" and "F-16
  maneuverability data", and boltflight.com "F-16 Turn Rate". These are secondary; the
  figures agree to within about 1°/s.
* Roll rates: NACA Report 868 (1947), as discussed on ww2aircraft.net ("Roll rate P47 vs
  FW190", "Relative rolling characteristics of WWII fighters").
* Muzzle velocities: Wikipedia, "AN/M2", "Hispano-Suiza HS.404" and "MG 151 cannon".
