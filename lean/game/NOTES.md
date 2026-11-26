# lean/game

The game step: elimination on one fixed map with no shrink, any team sizes, batched over
environments. Pure functions of tensors, no Python loops over ships or bullets. Decisions:
`docs/rewrite/core-decisions.md`. History (v2 parity, waste fixes, fog, the flight changes):
`docs/rewrite/core-worklog.md`. Types shared with the other packages: `lean/contract.py`.

## Entry points

`from lean.game import GameConfig, reset, deal, take_deal, step, draw_noise, tick, observe`
(`lean/game/core.py`).

- `GameConfig()` is the core game: arena 4,200 m, boundary ramp 300 m, clock 1,800 ticks (60 s at 30 Hz),
  "mixed" spawn, lineup `("p51d",)`, ten fields drawn per environment at each deal, ghost bullets,
  fog, a store of two loops at corner speed with 5% edge bands, firing cuts thrust to
  `fire_thrust_fraction` (0). `field_count=0`, `ghost_bullets=False`, `fog=False` are ablations.
- `reset` / `deal` / `take_deal`: fresh states at given team sizes (unused slots `exists=False`);
  the learner draws deals once per segment and takes them as environments finish.
- `step(config, state, action, noise)`: one tick. The tick applies `state.pending` (the previous
  tick's action, neutral after a deal) and stores `action` as the new pending one: the one-tick
  delay, for every agent. Stages: `_fire`, `_fly`, `_move_bullets`, `_hit_test` (swept closest
  approach, first contact), `_damage`, `_boundary_damage`, `collisions`, `_shields`, `_end_check`,
  `_bullet_ends`, then the fog's `perceive`. `tick(config, state, action, generator)` draws the
  noise and steps: the game half of the rollout's, the evaluation runner's and the sandbox's tick.
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
  (`RealAircraft.profile`, in its cache key). Two today: `"documents"` (`DESIGN_INPUTS`, the
  default table: top speed and climb solved for the flight test) and `"airframe_and_engine"`
  (base the documents: C_D0 and the shaft power closed inputs, nothing solved, top speed and
  climb readouts). A profile with a `base` takes its reference values from that profile's
  solved row (`design_of` under its own table; `reference(name, profile)`, `reference_in`),
  then the aircraft's own values for inputs the base has not (the P-51D's shaft power).
- **The P-51D's card** (author 2026-11-24 night): the airframe with a 2,200 hp engine (late-war
  and post-war engines; the P-51H had about 2,200 hp), under `"airframe_and_engine"`. The
  documents still fit the airframe to the 1,630 hp flight test (375 mph, 3,600 ft/min; C_D0
  0.0160); the card keeps that airframe and its propeller efficiency (0.875) at 2,200 hp:
  416 mph, 4,882 ft/min at 191 mph, best sustained 19.97 deg/s at 188 mph, corner 286 mph
  at 33.6 deg/s. The one warning is the structure (the 8 g limit, 1.22 x its 6.6 g). Warnings name a card field (`Check.field`) and
  point at whichever input sets it in the reference's table. The solve, the card and the tool
  have no code for any one profile.
- **The documents' inputs** (author 2026-11-24): stall speed, g limit, top level speed and
  best climb (the last two solved for the zero-lift drag and the power, nothing else solved);
  span efficiency e (0.85), the separation drag's strength (0.3) and exponent (8) and the
  thrust softness (2.83, what the old solve gave for the 165 mph best-climb speed), closed;
  every other input as before. The best-climb speed, the best sustained turn, the corner and
  the gap are readouts. Tried and dropped the same day: the "performance points" profile (g
  limit, corner speed, top speed, best climb, best sustained speed and gap solved for drag,
  power, induced and separation drag) and solving e, the best-climb speed and the gap.

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
- **Stick levels** (author 2026-11-26, `card.LEVEL_NAMES`; stick indices unchanged, 1/2
  Sustain, 3/4 Max, 5/6 Snap; `stick_table`, `sustain_stick_table`, `deep_stall_table`,
  `stick_level_shape`). The envelope's top line is sqrt(budget^2 - held^2), the budget the
  wing's maximum lift against the g limit (one soft minimum: the corner).
  - **Max**: the top line at every speed.
  - **Snap**: Max's line at and above corner speed. Below it the angle of attack opens past the
    stall to the card's `deep_stall_angle` (45 deg) by a smootherstep over `deep_stall_band`
    (0.30) of corner speed; near stall speed it closes back to the stall angle
    (`_holding_angle`) so Snap never gives up the 1 g.
  - **Sustain**: the fraction of Max's line whose turn holds energy (Ps = 0) coasting at the
    spawn store, from a per-type table over speed (`_sustain_table`: 97 speeds, stall speed to
    cruise speed, read linearly; no solve per tick). 1 (Max) where Max gains energy, 0 at and
    above cruise speed (no sustained turn there: straight). Dive and climb use the same fraction
    of their own top line.
- **Post-stall** (`GameConfig.post_stall`, on; off for comparison): past the stall the lift falls
  monotonically by a smootherstep in the angle to the flat plate's (`card.POST_STALL_LIFT_FRACTION`
  0.75 of the maximum) at 45 deg, then C sin(2a), 0 at 90 deg. The post-stall drag rises smoothly
  from the stall to the card's `post_stall_drag_ratio` (6) x the drag at the stall at 45 deg,
  bounded at twice that. Past the stall the angle is the wing's whole: the held vertical comes
  out of the normal force first (`_path_lift`, a smootherstep over `POST_STALL_SHARE_WIDTH`;
  for Snap once its deep angle opens), and the wing never carries more than its maximum lift
  against the g limit. No cap at the envelope: near stall speed Snap's thrust turns it a little
  more than Max.
- **The 1 g near stall:** what holds the vertical is softmin(C_max v^2 + thrust |sin a|, g limit),
  a the angle flown (the tick: the current angle; `steady_rates` resolves the setting twice).
  The turn's budget is unchanged; the wing's share of the hold is softmin(held, budget), so where
  the thrust carries the rest the turn fades smoothly.
- **Nose** (`_kick`, `card.NOSE_STIFFNESS`): a spring toward the commanded angle with fixed
  friction, total damping coefficient 2 nose_damping omega, nose_damping 1/sqrt(3). Going out
  (the target beyond the angle on its side, or on the other side of centre) the stiffness is
  1/3, 2/3, 1 x omega^2 for Sustain, Max, Snap: damping ratios 1, 1/sqrt(2), 1/sqrt(3). Going
  back toward the flight path (a release, a smaller level on the same side, a reversal until the
  nose crosses centre) it is always the full spring. Chosen per tick, no state.
- **Firing below stall** (`GameConfig.fire_store_band` 0.1, `_fire`, `_fly`): the thrust a shot
  withholds is paid by the speed from (1 + band) x stall speed up, by the store at stall speed
  and below (the withheld thrust's power over the tick), a smootherstep between. With an empty
  store a ship below (1 + band) x stall speed cannot fire; its gun stays ready.
- **Separation drag** rises toward the wing's maximum lift and stops there (a transient can
  ask for twice it; P-51D 0.3 at exponent 8, a design input: about +30% lift-dependent drag at
  the maximum lift, starting near 0.8 of it). `steady_rates` / `quasi_steady` evaluate the same
  functions with the nose settled; they give the derived constants, `tuning`, and the web tool.
- **Constants:** `ship_card` / `ship_constants` build a run's `contract.Constants` once; a
  value equal for the whole lineup is a Python number, one that differs is a per-slot row. The
  scripted agent reads every number it uses from these by name.
- **Fog:** `perceive` (sight by v2's rule, or a shot, a real bullet's hit or a collision that tick),
  `enemy_view`, `team_view`. Sight (`sight_pairs`, per target and observer; `sighted_by_enemy`,
  any observer) is an ellipse with the observer at one focus, stretched along its nose:
  |d| - e (d . nose) <= p, d the offset to the target, p the card's vision range (the side
  reach) and e its `vision_eccentricity` (0 <= e < 1, a card column and design input, default
  0 = the old circle, until the author has tested it; author 2026-11-25). Reach p / (1 - e)
  ahead, p to the side, p / (1 + e) behind: with p = 1,500 m, e 0.2 gives 1,875 / 1,250 m,
  e 0.5 gives 3,000 / 1,000 m, e 0.6 gives 3,750 / 938 m. From a wanted forward reach F:
  e = 1 - p / F (the rear reach is then p F / (2F - p)). Tested without a square root as
  |d|^2 <= (p + e d . nose)^2, which at e = 0 is the old d^2 <= p^2 exactly; one dot product
  more per pair. Fields' cores block lines as before; a team sees what any of its ships sees.
  **Tracking circle** (author 2026-11-26, `GameConfig.fog_tracking`, on): a ship visible to the
  enemy team on the tick before, by any means (sight, its shot, a hit, a collision), stays
  visible while some living enemy observer has it within the circle of that observer's forward
  reach p / (1 - e), with a clear line (the same field-core test); once no observer holds it
  (and nothing reveals it) it is hidden and needs the ellipse or a reveal again. "Visible the
  tick before" is read from the state, no new field: `tracked_by_enemy` = ticks since seen 0 and
  known (the deal starts every row at ticks since seen 0 but unknown, so a never-seen ship is not
  tracked; the deal's own perception sees by the ellipse alone). Per pair the reach is
  `torch.where(tracked, p / (1 - e), p + e d . nose)`, still squared against |d|^2. The circle
  contains the ellipse (|d| <= p + e |d| gives |d| <= p / (1 - e)) and equals it at e = 0, so
  today's default is unchanged (tested). The switch is a GameConfig rule like `fog` and
  `ghost_bullets`, not a card column: it is one on/off rule for every type, and the circle's
  radius comes from the card's p and e, so the card gains no number. `Obs.pending` holds own
  living ships' joint action, `NO_PENDING`
  (42) elsewhere. The ship row has no position scalars; `Obs.pos` carries position for the
  model's position code and rotary.
- **Bullet damage** (`_damage`): the owner's hit damage x kinetic (the explosive share plus
  the rest by the squared ratio of the bullet's speed to its reference speed) x closeness
  (`graze`: full inside the core radius, a smoothstep to 0 at the hit radius, on the impact
  parameter) x the nose cone (`aspect_factor`, author 2026-11-25): no damage when the bullet's
  path is within the card's inner angle (30 deg) of the target's nose, full from the outer
  angle (45 deg) round to dead astern, a smootherstep in the cosine between (t = (cos 30 - c) /
  (cos 30 - cos 45), no acos; the card stores the angles, the constants their cosines).
- **Deflector** (proposal 2026-11-27, branch `bullet-deflector`, unmerged;
  `GameConfig.deflector`, `card.DEFLECTOR_PRESETS`, `core.deflection`, `core.hit_fraction`,
  `core._deflect`). A field around every ship, the same for every type and at any shield (shield
  down included): a bullet whose straight in-tick path passes at b from a ship's centre is
  treated as passing at b + D, D by the bullet's path against the ship's nose (front D inside the
  inner angle, side D beyond the outer, the nose cone's smootherstep in the cosine between).
  Contact: b + D < hit radius (`_hit_test`); damage: `hit_fraction`, graze(b + D; core, hit
  radius) x the nose cone weighted by the hit model's `nose_cone` (1 keeps the rule, 0 drops it).
  A bullet that touches nothing and whose pass's closest approach lies in this tick (0 <= tau < 1,
  so once per pass) inside the field (b < hit radius + D) is pushed at that point by D_eff =
  min(D, hit radius + D - b) to the side it was passing (the smooth sign m / sqrt(b^2 + eps^2),
  eps `card.DEFLECTOR_SIGN_SOFTNESS` 0.1 m) and turned away by atan(D_eff / deflector_length): its
  direction and the rest of its move rotate, its speed is kept (no energy added). Ghosts run the
  same code. Noted: a second ship later in the same tick sees the unbent path; the bent bullet's
  cached medium stays the unbent end's (metres off); a bullet within about eps of dead centre is
  barely pushed or turned (frontal: it passes through with no damage, since contact and damage
  use |b| + D). Presets: "off" (D 0, nose cone on; also skips the code), "symmetric" (D 3 m
  everywhere, nose cone on), "frontal" (D 12 m = the hit radius in the front 30 deg, fading to
  3 m by 45 deg, nose cone off: every frontal bullet glances off). Bend length 48 m in all
  (frontal glance 14 deg, side push 3.6 deg). The scripted agent's fire-worth reads
  `hit_fraction` (one call). Measurements: docs/rewrite/sections/bullet-deflector/progress.md.
- **The hit model at runtime** (proposal 2026-11-27): every hit-model number (`core.HIT_MODEL`:
  hit and core radius, hit damage, bullet spread, deflector front/side/angles/length, the nose
  cone's weight and angles) is a row of the per-slot table (`core.RUNTIME_CONSTANTS`), never a
  Python number baked into the compiled graph: `core.with_hit_model(constants, **inputs)` changes
  it for every ship between calls of the compiled tick with no recompile (tested under
  `torch.compile(fullgraph=True)`). The workbench `python -m lean.tool.hitbench` (effective hit
  radius, apparent size, hit chance against aim error, damage per second, time to kill, from the
  step on synthetic bullets, about 2 s) and the sandbox's "Hit model (live)" sliders use it.
- **Collisions** (`collisions`, author 2026-11-25, first touch 2026-11-26; in the step after
  the move, with the bullet hits and the boundary damage, before `_shields`): every pair of
  ships alive at tick start, any teams, on the two straight moves: p the gap, w the relative
  move, R the two hit radii. A pair is scored once per overlap, on the tick of first touch: the
  gap is at least R at tick start and the swept move p + s w (0 <= s <= 1) comes within R. Ships
  that start a tick inside R are never scored; they must start a tick at or beyond R again
  first. State-free (the start and end positions decide it) and once per overlap by
  construction, also with both ships turning. b = |m|, m = p + tau w, tau = -p.w / |w|^2
  unclamped: the closest distance of the tick's relative line extended past the tick (the
  pass's predicted closest approach). Each ship's blame is 1 - aspect / 2 at the cosine between
  its nose (after the move) and the line of centres at first touch, n = m / R - sqrt(max(0, 1 -
  b^2 / R^2)) w / |w|: 1 within 30 deg, 0.5 beyond 45 deg (the bullets' zone). Damage = max
  shield x (|w| / tick) / (2 x cruise speed) x graze(b; the two core radii, R) x blame: linear
  in the closing speed, a full shield head-on at cruise; full when the cores would meet, a
  partial graze when only the hit radii do. Zero sum: each ship's collision damage is in
  `hit_damage` as dealt by the other (rewards, kill credit and scoring treat it as a hit;
  team-mates included, like friendly fire), in `damage_taken`, and reveals both ships under
  fog. The shield rule is the bullets': the damage empties the shield (clamped at 0), and a ship
  whose shield was already 0 at tick start dies. (Until 2026-11-26 a pass was scored on the tick
  holding its closest approach; with both ships turning that fell in two ticks or none.)
- **Shields:** recharge after a quiet spell (coast, straight, no trigger; author 2026-11-26:
  every turn level restarts the wait, `card.QUIET_STICK_MAX` 0).
- `StepOut` carries what the learner needs: damage, deaths, done and result, each bullet's
  outcome and firing tick (the hit head's labels), the attacker-by-target damage matrix (real
  bullets and collisions), boundary damage and shield restored (the rewards' ledger).

## Numbers

- Operations per step, uncompiled, CPU, E = 256, 5v5 (`docs/rewrite/sections/scoreboard/game_ops.py`,
  2026-11-24): step 1,625 compute + 210 view (no fields 1,287 + 162); observe 102 + 54;
  `reset_done` 265 + 82; scripted `act` for both teams 439 + 109. Counts are the measure; CPU
  times vary by about 25% between runs. The tripwire is a signal to look for bloat, not a limit.
- Nose cone and collisions (2026-11-26, `docs/rewrite/sections/nose-cone/sanity.py`, the yardstick
  on both teams, `GameConfig()`, 256 games each, seed 0; before = core eba3e474). Collisions per
  game are passes inside the two hit radii (before: passes through, no damage); a head-on pass
  is one inside 100 m with the noses at least 150 deg apart. Game step +82 operations (1,243
  no fields, 1,581 with fields).

| | 1v1 | 1v1 fields | 4v4 | 4v4 fields |
|---|---|---|---|---|
| draws, before / after | 93 / 41 | 43 / 13 | 1 / 3 | 2 / 4 |
| mean game length, s | 58.2 / 32.1 | 50.2 / 30.5 | 34.1 / 27.1 | 40.5 / 33.6 |
| deaths in all | 10 / 227 | 85 / 235 | 1,268 / 1,395 | 1,259 / 1,380 |
| contacts per game | 6.74 / 0.97 | 4.32 / 0.81 | 8.49 / 3.07 | 7.32 / 2.88 |
| head-on passes per game | 6.74 / 2.08 | 4.43 / 1.32 | 8.62 / 3.63 | 7.77 / 3.16 |
| contacts per head-on pass | 0.999 / 0.004 | 0.969 / 0.018 | 0.935 / 0.046 | 0.856 / 0.072 |
| deaths on a collision tick (after) | 24 | 24 | 101 | 112 |
| real hits within 30 deg of the nose | 84.6% / 0.7% | 48.4% / 2.6% | 30.8% / 3.5% | 32.8% / 3.3% |
| of those that did damage (after) | 0 | 0 | 8 | 5 |

  The few damaging "inside 30 deg" hits in 4v4 are most likely the bench's target guess (the
  enemy nearest the bullet's last move), not the rule (the step test: 29 deg does 0).
- Collisions at first touch (2026-11-26, the same sanity.py, which now counts the tree's own
  `core.collisions` contacts; before = core 17e6cf3b, the closest-approach rule;
  `docs/rewrite/sections/first-touch/`). An overlap episode is a run of ticks a pair is scored
  or ends inside the two hit radii; "closed" episodes end with both ships alive, "cut" ones in a
  death or the game's end. Compile check: 5,939 operations (core 5,930, +9), cold 222.9 s (222.4).

| | 1v1 | 1v1 fields | 4v4 | 4v4 fields |
|---|---|---|---|---|
| contacts, before / after | 248 / 290 | 207 / 243 | 787 / 883 | 737 / 826 |
| closed episodes scored 0 times | 0 / 0 | 1 / 0 | 3 / 0 | 5 / 0 |
| closed episodes scored twice | 0 / 0 | 1 / 0 | 4 / 0 | 5 / 0 |
| of those, fast (over 100 m/s): 0 / twice | 0, 0 / 0, 0 | 1, 1 / 0, 0 | 1, 1 / 0, 0 | 2, 4 / 0, 0 |
| deaths on a collision tick | 24 / 72 | 24 / 40 | 101 / 115 | 112 / 97 |
| deaths in all | 227 / 237 | 235 / 256 | 1,395 / 1,424 | 1,380 / 1,390 |
| draws | 41 / 49 | 13 / 30 | 3 / 3 | 4 / 3 |
| mean game length, s | 32.1 / 30.7 | 30.5 / 29.1 | 27.1 / 26.7 | 33.6 / 33.0 |

  On the same play (`first-touch/episodes.py`, both rules on the before tree's moves) every
  overlap episode holds exactly one first touch (2 spawned overlapping in 4v4 hold none). The
  extra contacts are cut episodes: a death (mostly a bullet in the same merge) or the game's end
  between first touch and the closest-approach tick, never scored by the old rule (1v1 68 of 316
  episodes, 1v1 fields 45 of 252, 4v4 103 of 886, 4v4 fields 120 of 853). That is also most of
  the rise in deaths on a collision tick: a ship at 0 shield now dies at the touch.

- Post-stall flight (2026-11-26, compile check, laptop 4070, cold): the tick's `advance` graph
  2,856 traced operations (core 2,300, +24%; the new flight terms, not split by cause), total
  6,503 (core 5,947), cold 233.8 s (core 226.9 s). Lines, nose and hold measurements: docs/rewrite/sections/post-stall/progress.md.

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

`lean/game/tests`: flight rules, damage (the nose cone: 29 deg no damage, 46 deg full, both
sides), collisions (the 90-degree crossing at equal speeds 0.5 blame each and closing sqrt 2 v,
the T-bone, head-on at cruise a full shield each, one score per straight pass whatever its
phase, one score for a fast pass with both ships turning that the closest-approach rule scored
twice or missed, none while overlapping and again only after separating, full closeness when
the cores would meet and a graze between, the credit to the other ship including a team-mate
and zero sum, no contact for a wide, later or dead pass),
fields, fog, ghost bullets, the deflector (`test_deflector.py`: the pushed distance in damage
and path, no energy added, one bend per pass, ghosts bend like real bullets, no frontal damage under
"frontal", the hit model's round trip and no recompile), invariants (finite, inert
padding, team swap), the clock and wipe, both spawn rules, the training spawn, the one-tick delay,
the levels (Max the envelope's top line to 0.1% and above every fraction of it, Snap equal to Max
above corner and at the deep angle below its band, Sustain's energy rate about 0 where a
sustained turn exists, the stick tables by index), the nose (damping ratio per level, the
overshoot order Sustain < Max < Snap, the full-spring return), Snap held 60 s staying above
0.95 x stall speed, firing below stall paid from the store and blocked on an empty store, only
straight keeping the shield quiet, Max's turn rate rising with speed to the corner, the tightest sustained turn 10% slower than the best rate, and
tuning (the derived card meets its targets: its airframe and engine and the closed inputs; the
documents' solve meets the flight test; plausible solved coefficients; the cache's key).

## Open

- Inputs that carry little: the angle of attack's cos and sin repeat the angle, the gun cooldown
  is always 0 at one shot a tick, the shield wait sits at its 3 s maximum. Kept for other types;
  a model step may drop them.
- The clock (1,800 ticks, 60 s) is set from scripted wipe times: docs/rewrite/clock-measurement.md.
- Engine power and the ship roster wait for the WWII research (core-decisions.md).
- On the GPU: the dense (E, N K, N) hit test's memory at large team sizes.
