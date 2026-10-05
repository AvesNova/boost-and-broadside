# Flight model and handling plan

Status: **planned, not implemented.**

This plan makes two changes:

* **Physics.** The nose angle moves continuously instead of jumping. A lateral
  acceleration limit (a g-limit) gives the flight envelope a corner speed.
* **Parametrization.** The flight constants are solved from handling targets, not set
  by hand.

All values are proper speeds `u = n|v|` in px/s at n = 1, angles in degrees and rates in
°/s, unless stated otherwise. Values given as starting points are meant to be tuned.

---

## 1. Goals

### 1.1 The target feel

The target is a responsive Second World War fighter:

* **Low and medium speed:** the nose can be thrown hard for a moment, a shot taken, and
  the nose snaps back to the flight path.
* **High speed:** the nose moves only a few degrees, and any real direction change is a
  committed, wide turn.
* **Very low speed:** little turning authority.
* **Turning fights trade energy.** Speed is not free turning power. Above corner speed
  a ship turns slower and wider, so it must spend speed (sharp-pull drag, reverse
  thrust) to turn its best.

### 1.2 Why the current model misses it

* **Turn radius is fixed:** 35 px sharp and 66 px normal at every speed, so turn rate
  grows with speed without limit. Faster is strictly better in every turning fight.
* **The nose jumps.** The gun moves 15° in one tick (450°/s), and 21–28° on a
  left-to-right reversal.
* **There is no stall in practice.** `min_speed` is 1 px/s.

### 1.3 Handling, not coefficients

Designers should choose quantities they can reason about: turn radius, peak and
sustained turn rate, corner speed, top speed, nose onset time. The physics constants
(thrusts, drag and lift coefficients, energy rates) are solved from those. The constants
are tightly coupled: raising top speed through regeneration also raises the speed and
rate of the sustained turn. A solver handles that coupling explicitly (§4).

## 2. How the game compares with real aircraft

Absolute numbers do not carry over. The game turns much faster and in much less room
than an aircraft, as an arcade game should. Ratios do carry over once two scales are
fixed:

* **Length:** one ship length L = 20 px (render size 10 px centre to tip; collision
  radius 10 px).
* **Time:** the best sustained 360° takes 3.0 s in the game and 17.5–27.5 s for WWII
  piston fighters. Rotation therefore runs **k ≈ 6–7 times faster** than real (k ≈ 4
  for coasting turns).

| quantity | today | planned (§3) | WWII piston fighters |
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

"Top speed" here is the energy-limited boost terminal, 136 px/s at n = 1. For aircraft,
corner speed is taken at a 6 g pilot limit, and onset is roll-to-bank plus the pull. The
bleed estimate assumes a lift-to-drag ratio of about 5 at maximum lift and a
thrust-to-weight ratio of 0.3.

What this means for the plan:

* **Already in range:** the speed of the best sustained turn, and the bleed in a hard
  turn once time is scaled. Keep them.
* **Brought into range:** a corner speed, turn rate falling above it, a low-speed
  floor, and nose onset time. The planned onset of 0.16 s falls inside the
  time-scaled real range.
* **Short of range:** instantaneous over sustained turn rate is 1.30, against 1.5–1.9.
  The game's top speed sits close to the speed of its best sustained turn, so corner
  placement trades this ratio against corner/top. A corner at 120 gives 1.56 and 0.88;
  a corner at 100 gives 1.30 and 0.74. Raising top speed to about 150–160 px/s with the
  corner at 120 would put both in range. With the handling solver, that is a change of
  targets, not of mechanics (§8).
* **Deliberately out of range: compactness.** Relative to ship length and gun reach,
  turning circles are about 12 times tighter than an aircraft's. A sharp 180° fits in
  about a sixth of gun reach. Matching aircraft would need sustained radii of
  250–500 px, which is a different game. Change compactness only as an explicit
  decision (§8).

## 3. Physics design

### 3.1 State

Two new per-ship fields:

* **slip** α, the nose angle relative to the velocity direction;
* **slip rate** α̇.

The attitude is `velocity_direction · e^{iα}`, as today with α in place of the per-action
offset. When α settles at a commanded value, the per-tick force equations are today's.
`ship_ang_vel` becomes the true heading rate: the velocity turn rate plus α̇.

### 3.2 Lift and drag as functions of slip

With `x = |α| / stall_angle`:

```
C_L(α) = sign(α) · C_L_max · (1 − (1 − x)²)      x ≤ 1
C_D(α) = C_D0 + b·x² + c·x⁴
```

* **Lift.** The parabola inverts in closed form: `α = stall_angle · (1 − sqrt(1 − C_L/C_L_max))`.
* **Drag.** b and c follow from the drag at the normal and the stall slip.
* **Past stall,** lift falls on the mirrored parabola, clamped at zero. Commands never
  exceed the stall angle, and the nose response does not overshoot.
* **Application.** Drag is applied as the exact quadratic-drag scale and lift as an
  exact rotation of the velocity, as the field path does today.

Today's points are reproduced exactly by `stall_angle` 15°, `C_L_max` 0.027 (0.015 at 5°)
and `C_D0` 8e-4, b 3.525e-3, c 6.75e-4 (0.0012 at 5°, 0.0050 at 15°).

### 3.3 Pull levels and the g-limit

The turn actions are straight, plus left and right at two pull levels:

* **sharp:** maximum lift, `C_L_max`;
* **normal:** a fixed lower lift level, capped at half the available lateral
  acceleration.

The command is a lift coefficient, capped so that lateral acceleration `C_L · u²` stays
within `max_lateral_accel`:

```
sharp_cl  = min(C_L_max,         max_lateral_accel / u²)
normal_cl = min(normal_lift,     max_lateral_accel / (2u²))
alpha_cmd = sign · lift_inverse(cl)
```

* **Corner speed** is `sqrt(max_lateral_accel / C_L_max)`. Below it the commands are
  today's; above it the nose angle shrinks with 1/u² and the radius grows with u².
* **Proper speed.** The cap uses proper speed, so the corner speed is the same in every
  field. World turn rate stays reciprocal in n at a fixed proper speed.
* **Starting point:** corner speed **100 px/s**, so `max_lateral_accel` = 270 px/s². The
  normal level's own corner is then about 95 px/s.

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
  ω = 25 rad/s gives 0.16 s, about 5 decisions at 30 Hz.
* **Rate limit.** Onset from straight to sharp peaks at about 138°/s. A limit near
  150°/s leaves onset alone and stretches a full left-to-right reversal to about 0.25 s.
* **Release.** The nose returns to the flight path over the same ~0.16 s.
* **ω does not scale with speed.** Speed dependence comes from the g-limit and the
  low-speed fade. Scaling ω with speed would make low-speed nose throws sluggish.

### 3.5 Low speed

* **Fade.** Below `slip_fade_speed`, the command amplitude is scaled by `u / slip_fade_speed`.
  Starting point: 35 px/s, below the slowest sustained turn (the coasting sharp turn,
  39 px/s).
* **Stall.** Below `min_speed`, lift is zero and the absolute attitude is held, as
  today. α is re-derived from the held attitude with α̇ = 0, so leaving the stall does
  not jump.

### 3.6 Air brakes removed

`AIR_BRAKE` and `SHARP_AIR_BRAKE` are removed from `TurnActions`. The scripted controller
never selects them. Slowing remains available through reverse thrust and sharp-pull
drag. Check a trained policy's air-brake usage before removing them.

### 3.7 One flight path

The no-field path (`num_fields == 0`, `_update_kinematics`) applies lift as an explicit
velocity kick. That multiplies speed by `sqrt(1 + (C_L·u·dt)²)` every tick, so lift
adds energy:

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

## 4. Handling parametrization

### 4.1 The two sets of variables

**Physics constants θ** of the new model (16):

| group | constants |
|---|---|
| thrust | `base_thrust`, `boost_thrust`, `reverse_thrust` |
| energy | `power_speed_constant` K, `passive_power_gain` P |
| drag | `C_D0`, drag at normal slip, drag at stall slip |
| lift | `C_L_max`, normal lift level, `max_lateral_accel` |
| nose | `stall_angle`, ω, `max_slip_rate`, `slip_fade_speed`, `min_speed` |

`max_power` is fixed at 100. Power units are a free gauge: K appears only as K·max_power,
K·P and K·bullet cost. Bullet cost is in power units, so changing K changes how much
speed a shot costs. Keep that in view, or state bullet cost in energy units.

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

Some quantities are not power laws. These have no closed form and are handled by the
correction stage (§4.2):

* the normal nose angle (through the lift inverse);
* dash peak and duration;
* speed bleed at corner;
* slipped thrust's share of turning, which makes the sharp turn about 5–7% tighter than
  the closed form.

Many quantities depend on each other. For example, corner speed = max turn rate ×
sharp radius. They are not independent targets.

### 4.2 Solving

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
handling, and it reports dependencies and free directions. Simulation agrees with its
closed forms within 2% on every speed; the sharp turn is the 5–7% exception above.

### 4.3 Starting spec

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

## 5. Effects elsewhere

* **Belief and next-state targets** (`frontline-reward-critic-belief-design.md`):
  * Enemy attitude moves smoothly rather than in five spikes.
  * Angular velocity needs continuous binning, over a range of about
    ±(max turn rate + `max_slip_rate`) ≈ ±305°/s.
  * That document's physics calibration must be re-measured.
* **Observation.** The `ang_vel` normalisation must be re-derived. Slip is visible as
  attitude minus velocity heading, so the state stays Markov.
* **Action space.** The turn head and the joint action id shrink. Checkpoints, BC data
  and replays recorded with air-brake actions become invalid.
* **Scripted controllers.** `scripted_utils.turn_toward` assumes an instant nose and
  will chatter with a 0.16 s lag. It needs hysteresis or a lead term, and each scripted
  opponent needs re-tuning against the new turn-rate curve.
* **Landing.** The reward, critic and belief redesign already forces a cold start and a
  checkpoint-schema bump. Land this change in the same cold start.

## 6. Validation

Use `benchmarks/flight_envelope.py`, run before and after. Its output for today's
physics is `docs/internal/flight-envelope-baseline-oct2026.json`.

* **Settled envelope:**
  * below corner, within 2% of Appendix A.2 at every index;
  * above corner, matching §3.3;
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

## 7. Order of work

1. **Unify the flight path** (§3.7) as its own change, so later measurements compare
   one change at a time.
2. **Extend the solver** to the new model's constants and quantities, add the
   correction stage, and derive `ShipConfig` flight constants from a handling spec.
   Today's constants must round-trip.
3. **Implement the slip state,** lift and drag curves, g-limit, fade and air-brake
   removal.
4. **Measure** with the harness against §6.
5. **Re-tune the scripted controllers** and run the duels.
6. **Land** with the reward, critic and belief cold start.

## 8. Open questions

1. **Corner speed versus top speed.** A corner at 100 matches real corner/top ratios
   but leaves instantaneous/sustained at 1.30. Raising top speed with the corner at
   about 120 would bring both into range. Decide from the duels.
2. **Compactness.** Turning circles are about a twelfth of an aircraft's relative to
   gun reach. Halving lift doubles every radius but changes the sustained envelope.
   Leave compactness unchanged unless it is chosen deliberately.
3. **Deep nose throws.** A sharp ceiling past stall (25–30°) at low speed would trade
   lift for drag steeply, which is closer to real aircraft. Try it as a variant after
   the 15° version works.
4. **Thrust vectoring** is out of scope. If it returns, cap its rotation rate and scale
   it with applied thrust. A near-stationary turret is the abuse case to test.

---

## Appendix A. Today's flight envelope (measured)

Measured with `benchmarks/flight_envelope.py` on the live configuration:
`frontline_ship_config(SHIP_CONFIG)`, dt = 1/30, field path. Each probe ship sits in a
uniform field of index n. Rates are world °/s.

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

The lateral offset at 180° is 66–75 px for sharp turns and 129–134 px for normal turns,
whatever the entry speed. Times scale with n.

### A.5 Acceleration and braking (n = 1)

* **Boost from rest:** 100 px/s in 1.3 s, 136 in 1.8 s; peak 199 at 2.9 s.
* **Boost from cruise at full power:** 200 in 1.7 s; peak 209 at 1.9 s, when power runs
  out.
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

The gun steps 15° on the first tick, and the slip clears in one tick on release. Today's
"nose throw" is mostly a real path turn.

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

## Appendix B. Reference aircraft

### B.1 WWII fighters

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

Stall is the midpoint of the database's range. Sustained rate comes from the database's
best 360° time at sea level.

### B.2 Other figures

* **F-16 Block 50, clean:**
  * sustained 21.5°/s at 9 g and Mach 0.7;
  * instantaneous about 25°/s;
  * corner 350–450 KCAS.

  Its instantaneous/sustained ratio is about 1.2. Fly-by-wire and high thrust flatten
  the gap that defines the WWII feel.
* **Roll rates** at combat speed are roughly 60–160°/s (Fw 190 near 160, P-47D about
  85 at 250 mph; Spitfire figures are disputed). Banking 60–90° takes about 0.4–1.5 s
  before the pull builds.
* **Muzzle speeds:** AN/M2 .50 calibre 866 m/s; Hispano Mk II 840–880 m/s; MG 151/20
  700–785 m/s.
* **Effective gun ranges:** 250–400 m, about 25–40 ship lengths.

## Appendix C. Sources

* IL-2 Great Battles aircraft database, v5.204, `aergistal.github.io/il2/`: Spitfire
  Mk.IXc, Yak-9 ser.1, Bf 109 G-6, P-51D-15, Fw 190 A-8, P-47D-28, Me 262 A.
* Wikipedia, "Aircraft of the Battle of Britain": the RAE turning-circle estimates.
* Wikipedia, "Messerschmitt Bf 109": G-6 dimensions.
* F-16 turn figures:
  * f-16.net, "F-16 Sustained Turn Performance" and "F-16 maneuverability data";
  * boltflight.com, "F-16 Turn Rate".
* Roll rates: NACA Report 868 (1947), as discussed on ww2aircraft.net.
* Muzzle velocities: Wikipedia, "AN/M2", "Hispano-Suiza HS.404" and "MG 151 cannon".
