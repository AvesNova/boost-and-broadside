# lean/game

The game step: elimination on one fixed map with no shrink, any team sizes, batched over
environments. Pure functions of tensors, no Python loops over ships or bullets. Decisions:
`docs/rewrite/core-decisions.md`. History (v2 parity, waste fixes, fog, the flight changes):
`docs/rewrite/core-worklog.md`. Types shared with the other packages: `lean/contract.py`.

## Entry points

`from lean.game import GameConfig, reset, deal, take_deal, step, draw_noise, observe`
(`lean/game/core.py`).

- `GameConfig()` is the core game: arena 4,200 m, boundary ramp 300 m, clock 5,400 ticks (30 Hz),
  "mixed" spawn, lineup `("p51d",)`, ten fields drawn per environment at each deal, ghost bullets,
  fog, a store of two loops at corner speed with 5% edge bands, firing cuts thrust to
  `fire_thrust_fraction` (0). `field_count=0`, `ghost_bullets=False`, `fog=False` are ablations.
- `reset` / `deal` / `take_deal`: fresh states at given team sizes (unused slots `exists=False`);
  the learner draws deals once per segment and takes them as environments finish.
- `step(config, state, action, noise)`: one tick. The tick applies `state.pending` (the previous
  tick's action, neutral after a deal) and stores `action` as the new pending one: the one-tick
  delay, for every agent. Stages: `_fire`, `_fly`, `_move_bullets`, `_hit_test` (swept closest
  approach, first contact), `_damage`, `_boundary_damage`, `_shields`, `_end_check`,
  `_bullet_ends`, then the fog's `perceive`.
- `observe(config, state, team)`: that team's `Obs`, built under fog from `team_view` only.
- `card.py` (the ship card, one row per type; today the P-51D) and `tuning.py` (not imported
  by the step): ship design, option C (`docs/rewrite/p51d-model.md`, "Design inputs"). A
  reference ship from the documents (`reference`: the airframe's point mass fitted to the
  flight test), then the design inputs (`DESIGN_INPUTS`, one table: what each sets, its
  measure, closed form or solved, provenance, slider) turned into a row by `design_row`
  (closed forms, then Newton on the solved targets); `design_of`, `implied`, `warnings`
  (plausibility, never refusals), `performance`, `envelope`, `time_scale`. The card holds no
  stored numbers for a real aircraft: `card.CARD` is derived on first use by
  `tuning.load_card` (`design_row` of each type's reference), cached on disk
  (`~/.cache/bnb-card`, `$BNB_CARD_CACHE`) by a hash of the aircraft and the flight model's
  source, so a physics change or a table change re-solves by itself (about 3-5 s once; a
  cached load about 1 ms). Adding, removing or swapping a design input is an edit of
  `DESIGN_INPUTS` only.
- **Profiles** (`tuning.PROFILES`): named design-input tables. Each card type names its own
  (`RealAircraft.profile`, in its cache key); the default is `"documents"` (`DESIGN_INPUTS`),
  and the P-51D uses it. A profile with a `base` takes its reference values from that
  profile's solved row (`design_of` under its own table; `reference(name, profile)`,
  `reference_in`), so the same ship starts under either table. Warnings name a card field
  (`Check.field`) and point at whichever input sets it in the reference's table. The solve,
  the card and the tool have no code for any one profile.
  - `"performance_points"` (author 2026-11-24): g limit and corner speed (closed: the lift is
    n g / v_corner^2), then top level speed, best climb, best sustained speed and the gap,
    solved for zero-lift drag, power, induced drag and separation drag. The best sustained
    rate, tightest speed, corner rate and best-climb speed are readouts; the thrust softness
    passes through. At the P-51D the condition number is 9. Moving any input by 1% or 3%
    (corner speed -3% included) converges in 3 Newton iterations (1.8 s), or 2-3 fast
    iterations (0.1-0.2 s), with condition numbers 8.0-10.0. A cold solve from the point mass
    also lands back on the documents row.
  - Why not the author's first set (corner speed, corner rate, best sustained speed and rate,
    tightest speed, top speed, g)? A point mass is homogeneous: scale the power and every
    drag together and neither the top speed nor the sustained line moves. Without a climb
    input the power's level is nearly free: condition 335, and 1% of best sustained rate
    doubled the solved power and raised the climb 64%. The corner rate is set by corner speed
    and g, apart from the lift softness: 1% of corner rate needs 22% of softness.

## Rules that matter

- **End:** a wipe, or at the clock more living ships, then more total shield, else a draw.
- **Spawns** (a setting chosen at launch): "mixed", every ship uniform over a disc of 0.85 R,
  heading uniform; "sides", each team on a half disc of a random axis, noses toward the other
  side within 15 degrees. Under both, a ship within 1,000 m of the boundary pointing outward is
  mirrored. `training_spawn` (set by `lean.train` only): per ship speed uniform from the stall
  speed to the top level speed, shield and store uniform 0 to full; an eighth of environments
  (`standard_spawn_share`) take the standard spawn (full shield, half store, 145.3 m/s).
- **Flight:** weight always on the wing, the engine drives speed only, dive and climb trade
  store for speed; fields bend paths (a no-fields path for F = 0).
- **Stick** (author 2026-11-24): a level is a fraction of the most lateral acceleration the
  ship can give at its speed, sqrt(budget^2 - held^2) with the budget the wing's lift against
  the g limit (one soft minimum: the corner). Full stick is all of it (the envelope's top line,
  no tax); the sustained level is derived in `_derived_characteristics` (`stick_table`: the
  stick that holds energy at the best sustained turn, so it passes through that point); the
  fine level is the card's `fine_stick_fraction` (0.5) of it. P-51D: 0.440, 0.881, 1.
- **Automatic flaps** (off for the P-51D, `flap_lift` 0; switchable per type): scheduled by
  stick demand and blown back by the corner speed, they raise the maximum lift when the stick
  asks for more than the clean wing gives (C2 in speed and demand; drag on the extra lift only;
  the turn rate never falls with speed below corner).
- **Separation drag** rises toward the wing's maximum lift and stops there (a transient can
  ask for twice it; P-51D 0.41 at exponent 16, solved by the gap 0.85, docs/rewrite/p51d-model.md
  "Design inputs"). `steady_rates` / `quasi_steady` evaluate the same
  functions with the nose settled; they give the derived constants, `tuning`, and the web tool.
- **Constants:** `ship_card` / `ship_constants` build a run's `contract.Constants` once; a
  value equal for the whole lineup is a Python number, one that differs is a per-slot row. The
  scripted agent reads every number it uses from these by name.
- **Fog:** `perceive` (sight by v2's rule, or a shot or a real bullet's hit that tick),
  `enemy_view`, `team_view`. `Obs.pending` holds own living ships' joint action, `NO_PENDING`
  (42) elsewhere. The ship row has no position scalars; `Obs.pos` carries position for the
  model's position code and rotary.
- **Shields:** recharge after a quiet spell (coast, at most the fine stick, no trigger).
- `StepOut` carries what the learner needs: damage, deaths, done and result, each bullet's
  outcome and firing tick (the hit head's labels), the attacker-by-target damage matrix,
  boundary damage and shield restored (the rewards' ledger).

## Numbers

- Operations per step, uncompiled, CPU, E = 256, 5v5 (`docs/rewrite/sections/scoreboard/game_ops.py`,
  2026-11-24): step 1,625 compute + 210 view (no fields 1,287 + 162); observe 102 + 54;
  `reset_done` 265 + 82; scripted `act` for both teams 439 + 109. Counts are the measure; CPU
  times vary by about 25% between runs. The tripwire is a signal to look for bloat, not a limit.
- Ship feature scales, fixed for every type, from scripted 4v4 play with the training spawn
  (`.../scoreboard/feature_ranges.py`, 256 environments x 3,000 ticks, 2026-11-23). Raw 1st / 99th
  percentiles in brackets.

| Feature | Transform and scale | 1st / 99th |
|---|---|---|
| world velocity x, y; world speed | symlog(x / 200 m/s) | -327 / 327; 77 / 457 m/s |
| nose rate | symlog(x / 1 rad/s) | -1.72 / 1.71 rad/s |
| angle of attack | linear / 0.25 rad | -0.245 / 0.244 rad |
| store, shield fraction | linear | 0.002 / 0.68; 0 / 1 |
| shield wait | linear / 3 s, clipped | 3 / 3 s |
| inside the boundary | linear / 1,000 m, clipped to [-1, 1] | 752 / 4,049 m |
| gun cooldown | whole ticks | 0 / 0 |

## Tests

`lean/game/tests`: flight rules, damage, fields, fog, ghost bullets, invariants (finite, inert
padding, team swap), the clock and wipe, both spawn rules, the training spawn, the one-tick delay,
the stick (full stick the envelope's maximum to 0.1%, every level its fraction of it, the
sustained level through the best sustained point, the fine level its fraction), the flaps on a
flapped test row (deployment by demand, C2 by refinement), the turn rate rising with speed to
the corner (flaps off and on), the tightest sustained turn 10% slower than the best rate, and
tuning (the derived card meets its targets: the flight test and its point mass's best
sustained rate; plausible solved coefficients; the cache's key).

## Open

- Inputs that carry little: the angle of attack's cos and sin repeat the angle, the gun cooldown
  is always 0 at one shot a tick, the shield wait sits at its 3 s maximum. Kept for other types;
  a model step may drop them.
- The clock (5,400 ticks) waits for the scripted wipe-time measurement (backlog, step X).
- Engine power and the ship roster wait for the WWII research (core-decisions.md).
- On the GPU: the dense (E, N K, N) hit test's memory at large team sizes.
