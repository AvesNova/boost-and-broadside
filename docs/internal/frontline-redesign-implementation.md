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
  `stall_angle`, `lift_coeff_max`, `normal_lift_coeff`, `drag_coeff_zero`,
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

* `turn_toward` and the stochastic scripted agent: decide turns against a
  predicted nose (lead term from slip rate) with hysteresis read from the
  previous action, so the controllers do not chatter under the 0.16 s onset.
* Short scripted duels before/after (hit rate, time-to-kill, mean speed, time
  above corner) via the existing scripted suites. Re-tune ramps only as far as
  the duels say is needed.

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
