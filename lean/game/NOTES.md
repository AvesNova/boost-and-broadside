# lean/game

The game step: elimination on one fixed map with no shrink, any team sizes, batched over
environments. Pure functions of tensors, no Python loops over ships or bullets. Decisions:
`docs/rewrite/core-decisions.md`. History: `docs/rewrite/core-worklog.md`. Types shared with the
other packages: `lean/contract.py`.

## Entry points

`from lean.game import GameConfig, reset, deal, take_deal, step, draw_noise, tick, observe`
(`lean/game/core.py`).

- `GameConfig()` is the core game: arena 4,200 m, boundary ramp 300 m, clock 1,800 ticks (60 s
  at 30 Hz), "mixed" spawn, lineup `("p51d",)`, ten fields drawn per environment at each deal,
  ghost-only bullets, fog with the tracking circle, the "egg" deflector, the post-stall regime,
  a store of two loops at corner speed with 5% edge bands, firing cuts thrust to
  `fire_thrust_fraction` (0). Ablations: `field_count=0`, `fog=False`, `fog_tracking=False`,
  `deflector="off"`, `post_stall=False`.
- `reset` / `deal` / `take_deal`: fresh states at given team sizes (unused slots `exists=False`);
  the learner draws deals once per segment and takes them as environments finish.
- `step(config, state, action, noise)`: one tick. The tick applies `state.pending` (the previous
  tick's action, neutral after a deal) and stores `action` as the new pending one: the one-tick
  delay, for every agent. Stages: `_fire`, `_fly`, `_move_bullets`, `_hit_test` (swept closest
  approach, first contact), `_damage`, `_deflect`, `_boundary_damage`, `collisions`, `_shields`,
  `_end_check`, `_bullet_ends`, then the fog's `perceive`. `tick(config, state, action,
  generator)` draws the noise and steps: the game half of the rollout's, the evaluation
  runner's and the sandbox's tick.
- `observe(config, state, team)`: that team's `Obs`, built under fog from `team_view` only.
- Relative geometry (`core.py`, for the learner's geometry heads; one source with the step and
  the observation): `boundary_distance` (beyond the edge, negative inside; boundary damage and
  the observation's boundary scalar read it), `field_edge_distance` (to the middle of a field's
  band, negative inside; `medium` reads it), `world_velocity`, `launch_velocity` (`_fire`'s
  launch before the spread), `bullet_reach` / `bullet_flight_time` (the drag law; `gun_range`
  reads `bullet_reach`), `visible_rows` (Obs.visible), `geometry_record` and `pair_geometry`
  (range, range rate, line-of-sight rate, energy difference, train angle, gun solution error).
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
  (`RealAircraft.profile`, in its cache key). Two: `"documents"` (`DESIGN_INPUTS`, the
  default table: top speed and climb solved for the flight test) and `"airframe_and_engine"`
  (base the documents: C_D0 and the shaft power closed inputs, nothing solved, top speed and
  climb readouts). A profile with a `base` takes its reference values from that profile's
  solved row (`design_of` under its own table; `reference(name, profile)`, `reference_in`),
  then the aircraft's own values for inputs the base has not (the P-51D's shaft power).
- **The P-51D's card**: the airframe with a 2,200 hp engine (late-war and post-war engines; the
  P-51H had about 2,200 hp), under `"airframe_and_engine"`. The documents fit the airframe to
  the 1,630 hp flight test (375 mph, 3,600 ft/min; C_D0 0.0160); the card keeps that airframe
  and its propeller efficiency (0.875) at 2,200 hp: 416 mph, 4,882 ft/min at 191 mph, best
  sustained 19.97 deg/s at 188 mph, corner 286 mph at 33.6 deg/s. The one warning is the
  structure (the 8 g limit, 1.22 x its 6.6 g). Warnings name a card field (`Check.field`) and
  point at whichever input sets it in the reference's table. The solve, the card and the tool
  have no code for any one profile.
- **The documents' inputs**: stall speed, g limit, top level speed and best climb (the last two
  solved for the zero-lift drag and the power, nothing else solved); span efficiency e (0.85),
  the separation drag's strength (0.3) and exponent (8) and the thrust softness (2.83, the
  solve's value for the 165 mph best-climb speed when that speed was an input), closed; every
  other input closed. The best-climb speed, the best sustained turn, the corner and the gap are
  readouts.

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
- **Stick levels** (`card.LEVEL_NAMES`; stick indices 1/2 Sustain, 3/4 Max, 5/6 Snap;
  `stick_table`, `sustain_stick_table`, `deep_stall_table`, `stick_level_shape`). The
  envelope's top line is sqrt(budget^2 - held^2), the budget the wing's maximum lift against
  the g limit (one soft minimum: the corner).
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
  value equal for the whole lineup is a Python number, one that differs is a row of the card's
  (C, T) table read through the slot's type (no per-slot copy). The scripted agent reads every
  number it uses from these by name.
- **Fog:** `perceive` (sight, or a shot, a real bullet's hit or a collision that tick),
  `enemy_view`, `team_view`. Sight (`sight_pairs`, per target and observer; `sighted_by_enemy`,
  any observer) is an ellipse with the observer at one focus, stretched along its nose:
  |d| - e (d . nose) <= p, d the offset to the target, p the card's vision range (the side
  reach, 1,500 m) and e its `vision_eccentricity` (0 <= e < 1, a card column and design input;
  0, the circle, until the author has tested the ellipse). Reach p / (1 - e) ahead, p to the
  side, p / (1 + e) behind: with p = 1,500 m, e 0.2 gives 1,875 / 1,250 m, e 0.5 gives 3,000 /
  1,000 m, e 0.6 gives 3,750 / 938 m. From a wanted forward reach F: e = 1 - p / F (the rear
  reach is then p F / (2F - p)). Tested without a square root as |d|^2 <= (p + e d . nose)^2;
  one dot product more per pair than the circle. Field cores block the line; a team sees what
  any of its ships sees.
  **Tracking circle** (`GameConfig.fog_tracking`, on): a ship visible to the enemy team on the
  tick before, by any means (sight, its shot, a hit, a collision), stays visible while some
  living enemy observer has it within the circle of that observer's forward reach p / (1 - e),
  with a clear line (the same field-core test); once no observer holds it (and nothing reveals
  it) it is hidden and needs the ellipse or a reveal again. "Visible the tick before" is read
  from the state, no new field: `tracked_by_enemy` = ticks since seen 0 and known (the deal
  starts every row at ticks since seen 0 but unknown, so a never-seen ship is not tracked; the
  deal's own perception sees by the ellipse alone). Per pair the reach is
  `torch.where(tracked, p / (1 - e), p + e d . nose)`, still squared against |d|^2. The circle
  contains the ellipse (|d| <= p + e |d| gives |d| <= p / (1 - e)) and equals it at e = 0, so
  at today's e = 0 the switch changes nothing. It is a GameConfig rule like `fog`, not a card
  column: one on/off rule for every type, its radius from the card's p and e.
- **Observation:** `Obs.pending` holds own living ships' joint action, `NO_PENDING` (42)
  elsewhere. The ship row has no position scalars; `Obs.pos` carries position for the model's
  position code and rotary.
- **Bullets: the ring is the clock** (core-decisions "Ghost-only bullets"): every ship, alive,
  dead or not existing, writes one bullet a tick into slot `tick mod BULLET_SLOTS` of its ring
  (`_fire`), overwriting the bullet written 12 ticks before, so every bullet lives exactly
  `contract.BULLET_SLOTS` (12) ticks and moves on each of them (`_move_bullets`: no countdown,
  no frozen bullets). The card's `bullet_life` must round to 12 ticks (checked when the
  constants are built, `_per_type`; `tuning.time_scaled` keeps it); a design moving muzzle
  speed or drag moves its gun range, `tuning.ring_gun_range`, a read-out: no design input asks
  for another life. Hit damage 8 and bullet spread 10.42 m/s per axis (5 mrad sigma per axis
  over the launch speed cruise + muzzle = 2,084 m/s, as `lean.tool.hitbench` reads it; the
  number is the setting, the angle moves with the muzzle or cruise speed): `tuning`
  `P51D_VALUES`. One uint8 state per bullet (`State.bullet_state`, `contract.GHOST / REAL /
  CONTACTED`): a real bullet if the ship fired that tick, else a ghost (a dead or absent ship's
  bullets are ghosts). Only a ghost or a real bullet can touch a ship, only a real one damages; a
  contact makes it CONTACTED and it flies on unchanged (no stop, no further contact, no turn by
  an egg). `_bullet_ends` reports each bullet's outcome once (StepOut.shot_outcome): its first
  contact (1 enemy, 2 team-mate) or a miss (0) on its last tick; a fresh deal's slots start
  CONTACTED, so they never report. The firing tick is the ring's age (`bullet_ages`):
  `shot_tick = tick - age`. The observation's bullet share counts REAL bullets.
- **Bullet damage** (`_damage`, `hit_damage`): the owner's hit damage x kinetic (the explosive
  share plus the rest by the squared ratio of the bullet's speed to its reference speed) x
  closeness (`graze`: full inside the core radius, a smoothstep to 0 at the hit radius, on the
  impact parameter) x the nose cone weighted by the hit model's `nose_cone` (`card.NOSE_CONE`,
  0 by default: off). The nose cone (`aspect_factor`): no damage when the bullet's path is
  within the card's inner angle (30 deg) of the target's nose, full from the outer angle (45
  deg) round to dead astern, a smootherstep in the cosine between (t = (cos 30 - c) / (cos 30 -
  cos 45), no acos; the card stores the angles, the constants their cosines). Collision blame
  uses the same angles whatever the weight.
- **Deflector** (the egg; `GameConfig.deflector`, `card.DEFLECTOR_PRESETS`, `core.egg_entry`,
  `core.egg_turn`, `core.hump`, `core._deflect`). An invisible egg around every ship, in its
  frame (nose +x): the front half an ellipse of semi-axis `egg_front` 53.3 m (40/18 x 24)
  along the nose, the rear half one of `egg_rear` 24 m, both of half-width `egg_half_width` 24
  m; the same for every type and at any shield. Not drawn. In the relative frame (bullet minus
  ship, as the hit test): on the tick a bullet's straight in-tick path first crosses a living
  ship's egg inward (a path starting inside never enters: a bullet fired from inside, or
  already turned, flies straight), its direction is turned away from the ship's centre by
  deflector_turn x hump(b / half-width), b the distance of the straight line from the centre,
  hump(x) = 27/4 x (1 - x)^2 on x clamped to [0, 1] (0 at both ends, peak 1 at a third;
  `deflector_turn` 45 deg); the side is the side of the centre the line passes (b = 0 goes
  straight). The relative speed is kept. The rest of the tick's move uses the turned
  direction, and the swept hit test runs on that remainder (entry point to tick end) against
  the hit circle; the bullet's map velocity after the tick is the ship's average velocity over
  the tick plus the turned relative velocity (its map speed changes by the ship's part; its
  cached medium is the unturned end's). Once per pass by construction (only inward
  crossings). Ghosts run the same code. With one egg per bullet and tick: the first egg by
  tau; the bullet's contacts with other ships in that tick stay on the straight path. Damage:
  `hit_damage` at the hitting pass's impact parameter (the turned remainder's line for a turned
  hit) and its path against the nose (the turned direction). Ship hit radius 12 m, core 10 m
  (a soft 2 m edge). Hit width (the band of aim offsets that hit, a still ship): 24 m with no
  turn; with the egg 4.75 m from the side and astern, 2.05 m head-on (`core.hit_widths`, the
  same numbers as the design sketch deflector_humps_sketch.py, tested). Presets: "egg"
  (default), "off" (no turn; also skips the code). The scripted agent reads `hit_widths` (a
  constants column, `core.hit_width` between its 17 aspects) for its target size and its
  trigger's hit chance (with the spread and its lead error). Measurements:
  docs/rewrite/sections/bullet-deflector/progress.md and egg-experiments.md.
- **The hit model at runtime**: every hit-model number (`core.HIT_MODEL`: hit and core radius,
  hit damage, bullet spread, the egg's front, rear and half-width, the largest turn, the nose
  cone's weight and angles; and the derived hit widths) is a row of the constants table
  (`core.RUNTIME_CONSTANTS`), never a Python number baked into the compiled graph:
  `core.with_hit_model(constants, **inputs)` changes it for every ship between calls of the
  compiled tick with no recompile (tested under `torch.compile(fullgraph=True)`). The
  workbench `python -m lean.tool.hitbench` (effective hit radius, apparent size, hit chance
  against aim error, damage per second, time to kill, from the step on synthetic bullets, about
  2 s), the tool's Hit range tab and the sandbox's "Hit model (live)" sliders use it.
- **Collisions** (`collisions`; in the step after the move, with the bullet hits and the
  boundary damage, before `_shields`): every pair of ships alive at tick start, any teams, on
  the two straight moves: p the gap, w the relative move, R the two hit radii. A pair is scored
  once per overlap, on the tick of first touch: the gap is at least R at tick start and the
  swept move p + s w (0 <= s <= 1) comes within R. Ships that start a tick inside R are never
  scored; they must start a tick at or beyond R again first. State-free (the start and end
  positions decide it) and once per overlap by construction, also with both ships turning. b =
  |m|, m = p + tau w, tau = -p.w / |w|^2 unclamped: the closest distance of the tick's relative
  line extended past the tick (the pass's predicted closest approach). Each ship's blame is 1 -
  aspect / 2 at the cosine between its nose (after the move) and the line of centres at first
  touch, n = m / R - sqrt(max(0, 1 - b^2 / R^2)) w / |w|: 1 within 30 deg, 0.5 beyond 45 deg
  (the nose cone's angles). Damage = max shield x (|w| / tick) / (2 x cruise speed) x graze(b;
  the two core radii, R) x blame: linear in the closing speed, a full shield head-on at
  cruise; full when the cores would meet, a partial graze when only the hit radii do. Zero
  sum: each ship's collision damage is in `hit_damage` as dealt by the other (rewards, kill
  credit and scoring treat it as a hit; team-mates included, like friendly fire), in
  `damage_taken`, and reveals both ships under fog. The shield rule is the bullets': the damage
  empties the shield (clamped at 0), and a ship whose shield was already 0 at tick start dies.
- **Shields:** recharge after a quiet spell (coast, straight, no trigger): every turn level
  restarts the wait (`card.QUIET_STICK_MAX` 0).
- `StepOut` carries what the learner needs: damage, deaths, done and result, each bullet's
  outcome and firing tick (the hit head's labels), the attacker-by-target damage matrix (real
  bullets and collisions), boundary damage and shield restored (the rewards' ledger).

## Numbers

- Compile baseline (2026-11-27 ~22:00, laptop 4070,
  `docs/rewrite/sections/scoreboard/compile_baseline.json`): the tick's `advance` graph 3,194
  traced operations.
- Feature scales: the ship scalars' fixed scales, the same for every type, from scripted 4v4
  play with the training spawn (`.../scoreboard/feature_ranges.py`, 256 environments x 3,000
  ticks, 2026-11-23). Raw 1st / 99th percentiles in brackets.

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

`lean/game/tests`: flight rules, damage (the nose cone with its weight at 1: 29 deg no damage,
46 deg full, both sides), collisions (the 90-degree crossing at equal speeds 0.5 blame each and
closing sqrt 2 v, the T-bone, head-on at cruise a full shield each, one score per straight pass
whatever its phase, one score for a fast pass with both ships turning, none while overlapping
and again only after separating, full closeness when the cores would meet and a graze between,
the credit to the other ship including a team-mate and zero sum, no contact for a wide, later
or dead pass), fields, fog, ghost-only bullets (`test_ghost_bullets.py`: every ship writes a
bullet a tick, a dead ship's are ghosts; each bullet reports once, a miss on its last tick; a
fresh deal's slots never label; a contacted bullet never damages or labels again and flies on
to its 12th tick), the deflector (`test_deflector.py`: the hump, one turn per pass, the relative
speed kept, dead centre straight and full damage, a bullet from inside the egg not turned,
ghosts like real bullets, the turned path and hit-or-miss equal the design sketch's
trace_through, the hit widths equal the sketch's, the hit model's round trip and no recompile),
the steady-flight table (`test_steady_table.py`), invariants (finite, inert padding, team
swap), the clock and wipe, both spawn rules, the training spawn, the one-tick delay, the levels
(Max the envelope's top line to 0.1% and above every fraction of it, Snap equal to Max above
corner and at the deep angle below its band, Sustain's energy rate about 0 where a sustained
turn exists, the stick tables by index), the nose (damping ratio per level, the overshoot order
Sustain < Max < Snap, the full-spring return), Snap held 60 s staying above 0.95 x stall speed,
firing below stall paid from the store and blocked on an empty store, only straight keeping the
shield quiet, Max's turn rate rising with speed to the corner, the tightest sustained turn 10%
slower than the best rate, and tuning (the derived card meets its targets: its airframe and
engine and the closed inputs; the documents' solve meets the flight test; plausible solved
coefficients; the cache's key).

## Open

- Inputs that carry little: the angle of attack's cos and sin repeat the angle, the gun cooldown
  is always 0 at one shot a tick, the shield wait sits at its 3 s maximum. Kept for other types;
  a model step may drop them.
- The feature scales were measured on the 2026-11-23 game; the flight, the bullets and the hit
  model have changed since.
- The clock (1,800 ticks, 60 s) is set from scripted wipe times: docs/rewrite/clock-measurement.md.
- Engine power and the ship roster wait for the WWII research (core-decisions.md).
- On the GPU: the dense (E, N K, N) hit test's memory at large team sizes.
