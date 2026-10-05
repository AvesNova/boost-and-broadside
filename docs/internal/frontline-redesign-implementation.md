# Frontline redesign: implementation plan and log

Implements `frontline-redesign-plan.md` (the spec). This note is the working plan:
the phases, what each delivers, how it is checked, and a running log of deviations
from the spec with the reason for each. Section references (§) are to the spec.

Definition of done for every phase: `pytest -n auto` green apart from the known
pre-existing failures, `bnb smoke` green, tests added for the new behaviour, docs
that the change makes stale updated, one commit per idea, pushed.

## Part I. Physics

### Phase 1. One flight path (§3.7)

* Delete `_update_kinematics`. The no-field path runs the field path's flight
  half-steps with a plain drift (`pos += vel·dt`, wrapped) in place of optical
  transport, at n = 1.
* Tests: lift alone keeps speed constant; held-power turn stays finite; energy
  never rises without thrust or regeneration; the no-field tick matches the
  field tick at n = 1.
* Gravity, which only the old no-field path computed in its own form, moves to the
  shared half-step (it already exists there).

### Phase 2. Handling solver in the package (§5)

* Move the linear core of `benchmarks/flight_spec.py` into
  `boost_and_broadside/config/handling.py`, extended to the new model's 16
  constants and quantities (§5.1), with the feasibility checks of §5.2.
* `HandlingSpec` (frozen dataclass of targets) → `ShipConfig` flight constants,
  derived once at configuration time and recorded with the run configuration.
* The correction stage (harness as forward model) lives with the harness in
  `benchmarks/`, because it needs the simulator; it reports how far the
  simulated envelope sits from the targets and can return corrected targets.
* Today's constants must round-trip.

### Phase 3. Slip state, lift and drag curves, g-limit (§3.1–3.6)

* `TensorState`: `ship_slip`, `ship_slip_rate` (B, N) float32, zeroed on reset
  and respawn.
* `ShipConfig`: replace the per-action angle, drag and lift constants with
  `stall_angle`, `max_lift_coeff`, `normal_lift_coeff`, `zero_slip_drag_coeff`,
  `normal_slip_drag_coeff`, `stall_drag_coeff`, `max_lateral_accel`,
  `nose_frequency`, `max_slip_rate`, `slip_fade_speed` (`min_speed` stays).
* Tick: command from proper speed (g-limit cap, fade), exact critically damped
  nose step at ω/n with the slip-rate limit at max_slip_rate/n, attitude from
  velocity direction and slip, lift and drag from the slip, the existing
  half-step/transport/half-step, stall gate.
* Remove `AIR_BRAKE`, `SHARP_AIR_BRAKE`: 5 turn actions, 30 joint actions.
  Every consumer of the action layout (scripted marginals, BC targets, replays,
  interactive mode) updated.
* Compile-safe: `where`/`clamp`/`minimum` only, plain floats from config.

### Phase 4. Measure (§10.1)

* Update `benchmarks/flight_envelope.py` for the new model: drop air-brake
  scenarios, add onset, snap-back, reversal and nose-throw transients, jitter
  checks and field-scaling checks.
* Write `docs/internal/flight-envelope-slip-oct2026.json`; record headline
  numbers against §10.1 here. Fix the physics if a check fails materially;
  do not chase every percent.

### Phase 5. Scripted controllers (§4)

* Measure first: `benchmarks/scripted_duels.py` reports scripted play (damage per
  shot, kills per ship-minute, speed, time above corner, turn-command changes and
  side flips) and plays side-balanced matches of a candidate configuration
  against the default controller.
* Change the controllers only where the measurements show a problem.

## Part II. Learning

### Phase 6. Rewards: five heads (§6)

* New `REWARD_HEADS = (outcome, zone_capture, zone_progress, kill_death, damage)`.
  Per-ship (B, N, 5) rewards; events computed and logged one by one.
* Ledgers in `TensorState`: zone attacker/defender ledgers (B, Z, N); the
  outstanding-damage ledger is `cumulative_damage_matrix` with pro-rata recharge
  reduction and clearing at respawn.
* Payee rule, `enemy_team_share`, payout ratios under `offensive_bias`.
* Delete the lambda matrix, `_LOCAL_COMPONENTS`, shaping components, the old
  component registry. Team spirit scalar per head, default 0.
* Test: per-head zero-sum at ratio 1 on random rollouts.

### Phase 7. Critic and advantages (§7)

* Raw rewards in the buffer (no storage symlog).
* Per-head categorical critic: 51 symlog-spaced bins per ship head, two-hot
  cross-entropy on the lambda-return; outcome head reads the global token with
  four classes (win, tie, loss, unresolved) and the discount-leak target.
* One advantage: weighted sum of head advantages, divided by the DreamerV3
  percentile scale (EMA 5th–95th spread of the summed return, decay 0.99,
  floor 1); no mean subtraction; no per-minibatch standardisation.
* Delete `ReturnScaler`, `AdvantageScaler`, Huber value loss, value clipping.
* Starting PPO settings: `entropy_coef` 3e-4, the rest unchanged (§7.4).
* Test: categorical target mean equals the scalar lambda-return.

### Phase 8. Next-state head and belief (§8)

* Codes, each with encode (moments → logits/probabilities) and decode
  (probabilities → moments) and an exact round-trip test: position 9-colour
  nested (81), velocity three-axis (243), attitude 4-colour (16), angular
  velocity 41 bins, health/power/shield delay/local log index 21 bins,
  cooldown 4 bins.
* Belief stores moments (24 floats per ship); visible ships at sigma 0; hidden
  ships carried from model outputs.
* Residual categorical head: logits = log(baseline + eps) + f(h), last layer
  zero-initialised; per-level cross-entropy against sharp truth.
* Encoder reads the codes; the Gaussian uncertainty terms are removed;
  `time_since_observation` stays. RoPE reads truth or the decoded mean.

### Phase 9. Map-scale randomisation (§9)

* Per-environment scale `s` sampled at reset (0.5 at s = 1, else
  exp(Exponential(1.5)) truncated, cap configurable, default 4), applied through
  `scaled_frontline_geometry`.
* Field generation and the scripted Frontline strategy read per-environment
  geometry. Ships-in-sight logged beside s.

### Phase 10. Schema bump, diagnostics, docs

* One checkpoint-schema bump for Part II.
* §10.3 diagnostics that are cheap to add: per-head zero-sum residual,
  charge-back share, outcome calibration, per-level position cross-entropy
  visible against hidden.
* Reader-facing docs that the redesign makes stale.

### Phase 11. PPO retune and the cold start (§7.4, §12 steps 8–9)

* Short runs only if the 8 GB dev GPU allows; otherwise hand over with the
  reading list of §7.4. The cold-start run itself is the user's to launch.

## Deviation log

Entries are added as the work proceeds.

### Part I

1. **Attitude is re-aligned to the end-of-tick flight path.** The spec sets the
   attitude from the velocity direction and the slip; doing that at the start of
   the tick (as the old code did with its offset) leaves the observed
   attitude-minus-heading off the stored slip by the path turn within the tick
   (up to 5° at 155°/s). The tick now flies with the start-of-tick attitude and
   stores `end_direction · e^{iα}`, and a stalled ship's slip is re-derived
   against the end-of-tick path, so slip is exactly attitude minus velocity
   heading (§3.1's Markov claim) and `ship_ang_vel` is the true heading rate.
2. **No-overshoot clamp.** Under the slip-rate limit the state can carry an
   approach speed the exact critically damped step would take past the command.
   Arrival is clamped (error and rate zeroed when the step crosses the command),
   and |α| is bounded by max(stall, |α_prev|).
3. **Drag is held at its 2×-stall value beyond it.** Only a ship leaving a stall
   with a large held slip reaches there; the quartic would otherwise stop it dead
   (C_D ≈ 14.5 at 180°).
4. **`nose_slew_rate` replaces `reversal_time` as the handling quantity** that
   pins `max_slip_rate`. The closed form 2·stall/rate (0.20 s) is not the
   measured reversal (0.27 s at 150°/s: onset and arrival are not rate-limited),
   and pinning reversal at 0.25 s through it would have set the limit to 120°/s,
   which binds on onset. The spec's 150°/s is kept; the measured reversal is
   reported by the harness.
5. **The correction stage runs offline** (`python -m benchmarks.flight_spec
   --correct`), not at configuration time: it needs the simulator. It converges
   in two iterations to within 0.2% on every measured target. The live config
   stays on the closed-form solve because §5.3 states the sharp rate in closed
   form (114; 119.3 measured). Onset is tick-quantised (5 ticks = 0.167 s), so the
   correction leaves it out.
6. **Bistable boosted sharp turn at n ≥ 1.41.** Entered at 100 px/s with full
   power, the boosted sharp turn settles on a second, above-corner equilibrium
   (130 proper at n = 1.41, 160 at n = 2) instead of the stall-slip one (82, 92):
   world-time regeneration supplies enough power to hold a low-slip g-limited
   turn there. This is the physics as specified, not a bug; the validation
   excludes those rows from the below-corner comparison and lists them.
7. **Step-response checks are read between fade and corner speed.** Above corner
   the command grows as speed bleeds, so "90% of the final slip" is not a step
   response (it reads 0.23–0.30 s at 160–300 px/s).
8. **Reverse no longer pushes a stationary ship backwards on the field-free path**
   (Phase 1): the field path's exact energy step stops reverse at the
   minimum-energy point.
9. **Throughput.** Compiled `update_ships` at 960 envs: 0.87 → 0.98 ms per tick
   (+13% on the ship-physics kernel; a small share of a full step).
10. **Drag validation is non-strict** (`C_D0 ≤ C_D,normal ≤ C_D,stall`) so that
    drag-free test configurations remain expressible.

11. **The scripted controllers are unchanged** (§4 predicted chatter and asked for
    hysteresis or a lead term). Measured instead of assumed: under the slip model
    the stochastic controller's turn command flips side on 0.4% of decisions
    (1.0% before) and `turn_toward` settles a 40° turn with 2–4 command changes
    inside its 5° deadband at 60–160 px/s. Two candidate retunes were played
    head-to-head against the unchanged controller, 256 full matches per seed
    (score SE ≈ 0.03): a nose lead of 0.1 s (0.53, 0.52, 0.47), 0.2 s (0.46,
    0.41), a combat governor fading boost above corner when in shooting range
    (0.52, 0.51), and both (0.50, 0.44). None helps, so none was kept.

### Measured (Phase 4)

`docs/internal/flight-envelope-slip-oct2026.json`, all §10.1 checks pass:
below-corner envelope within 0.53% of the October baseline at every index;
onset to 90% 0.167 s; snap-back 0.20 s; reversal 0.27 s (below corner) and
0.20 s (above); peak settled lateral acceleration 283 px/s² against 270 + 21
slipped-thrust allowance; no slip-rate sign changes in held turns; nose times
scale with n within a tick; the field-free terminal matches the field path at
n = 1 to 1e-3. The boosted normal turn widened from 66 to 126 px (59.5°/s at
131 px/s). A 0.5 s sharp throw from 300 px/s now turns the gun 24° (179°
before). Scripted play with the unmodified controller: median proper speed 113
→ 124 px/s, median heading rate 0 → 19°/s (the nose no longer jumps), p95
lateral acceleration 292 → 233 px/s².

### Measured (Phase 5)

Scripted self-play, 2 seeds × 256 envs × 2000 decisions, slip model against the
old physics (same controller): damage per shot 3.13 vs 3.33 (−6%), kills per
ship-minute 1.70 vs 1.88 (−9%), mean proper speed 114 vs 105 px/s, time above
corner 70% vs 60%, turn-command change rate 0.25 vs 0.31, side flips 0.4% vs
1.0%. Fights cost a little more under the g-limit, as intended (§2.1).
