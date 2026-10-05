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

Read from `env/physics.py` (`_update_kinematics`, `_update_kinematics_in_fields`):

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

* **Below it**, both levels command today's 5° and 15°.
* **Above it**, the nose angle shrinks with 1/u² and the radius grows with u².

The cap uses proper speed `u = n|v|`. This keeps the corner proper speed independent of
the index, and keeps world turn rate reciprocal in n at a fixed proper speed, matching
today's field behaviour.

Worked numbers for a corner speed of 120 px/s at n = 1 (`max_lateral_accel` ≈ 389). The
radii here are lift-only; the measured radii include about 5% from thrust.

| u (px/s) | normal α | sharp α | normal R | sharp R | sharp rate, today | sharp rate, capped |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 5.0° | 15.0° | 67 | 37 | 62°/s | 62°/s |
| 100 | 5.0° | 15.0° | 67 | 37 | 155°/s | 155°/s |
| 120 | 4.4° | 15.0° | 74 | 37 | 186°/s | 186°/s |
| 136 | 3.3° | 7.9° | 95 | 48 | 210°/s | 164°/s |
| 160 | 2.3° | 5.1° | 132 | 66 | 248°/s | 139°/s |
| 215 | 1.2° | 2.6° | 238 | 119 | 333°/s | 104°/s |
| 300 | 0.6° | 1.3° | 463 | 231 | 464°/s | 74°/s |

The parabolic lift curve is flat near its peak, so the sharp nose angle falls quickly
just above corner speed: 15° at 120 px/s, 7.9° at 136. That gives high-speed flight its
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
* **Both flight paths change in step.** The ambient path (`num_fields == 0`) and the
  field path share one slip-update helper, so they cannot diverge.
* **Cost** is a few dozen elementwise operations on `(B, N)`, which is negligible next
  to collisions and observation. Confirm it with the `rl-throughput` benchmark, not by
  assumption.

## 5. Effects elsewhere

* **`frontline-reward-critic-belief-design.md`.**
  * §3.6: an enemy's next attitude is no longer a mixture of five spikes. It moves
    smoothly, which makes the finer attitude levels more predictable.
  * §3.7: angular velocity is no longer five exact values and needs a continuous
    binning. Its range is about ±(`max_lateral_accel` / corner speed + `max_slip_rate`),
    roughly ±340°/s at the numbers above.
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

## 7. Open questions

1. **Deep nose throws.** Should sharp go past stall at low speed (a 25–30° ceiling,
   where lift drops and drag rises steeply)? That is closer to the real trade-off, and it
   limits itself, but it changes the sustained low-speed sharp turn into a decelerating
   slide. Start at 15° and try a deeper ceiling as a variant.
2. **Corner speed.** 110–130 px/s puts it between coast cruise (100 px/s) and
   energy-limited boost at n = 1 (136 px/s). Tune it against measured play speeds, which
   span about 20–215 px/s.
3. **Landing.** The reward, critic and belief redesign already forces a cold start and a
   checkpoint-schema bump. Folding this change into the same cold start avoids a second
   one. Validate the physics with scripted duels first (§6), since those need no
   learning.
4. **Thrust vectoring.** Deferred. If it returns, cap its rotation rate low and scale its
   torque with applied thrust. Boost from near zero speed is almost free under the energy
   law, so a near-stationary turret is the abuse case to test.
