# Lean core: work log

Newest last. Each entry: time (KST), item from core-backlog.md, what was done, how it was
checked, the commit. Numbers are measured unless marked otherwise.

## 2026-11-23

- 02:37 Overnight session starts (author asleep until 09:00). Plan: section A (flight
  physics behind a switch), then B1, then section C when blocked. Machine: laptop,
  platform profile "performance", GPU idle.
- 03:20 A1-A8, A11: the core flight model behind `GameConfig.flight_model` ("v2" default,
  "core" new) and `GameConfig.core()` (core model, Vizia only).
  - What changed (lean/game/core.py): in `_setting`, the vertical demand is held first
    (`held = softmin(demand, budget)`, no `1 - sink_room` exemption) and the vertical lift is
    that held value (re-deriving it through a smooth minimum lost about 4% between near-equal
    values and showed up as sinking at full stick); no charge term (the engine never feeds
    the store); thrust from the card's full-store power (unchanged, constant) times a firing
    scale; smootherstep edges over `store_edge_band` (5%). In `ship_constants`: store =
    `store_loops` (2) x g x 2 x corner radius (corner radius with 1 g on the wing), spawn at
    half; dive and climb rates both the mean of the card's two (38.6 m/s^2 for Vizia); bullet
    cost 0. In `step`: firing scales thrust by `fire_thrust_fraction` (0.0) on the ticks it
    fires. Vizia's store: 32,169 J/kg (v2: 497,563, about 15x more).
  - Engine power choice: the card's full-store power (Vizia 2.04 kW/kg), the power v2 already
    uses for thrust at every store; the 7-9 kW/kg "surplus" that only charged the store is
    gone. So top speeds are unchanged.
  - Checked: tests/test_core_flight.py, 9 assumption tests (store size and spawn; coasting
    straight holds altitude; full stick above the 1 g speed holds altitude; the engine never
    raises the store; below the 1 g speed the ship sinks; dive and climb equal and opposite;
    no dive below empty or climb above full; the edge taper never reverses or overshoots;
    firing costs speed and no store). `pytest lean`: 57 passed, 1 xfailed (the recorded
    "mixed" spawn), 36 s, CPU one thread. v2's parity tests unchanged (v2 is the default).
  - Finding for tuning (A12): at full climb rate the store moves about 36% of the band per
    second at 300 m/s, so the 5% edge band is crossed in about 4 ticks and the taper acts
    almost like a wall. Options for the author: a lower exchange cap, a wider band, or a
    larger store.
  - Sandbox: `--core` flies GameConfig.core() (Vizia by default); README updated.
- 03:45 A9: measurements in docs/rewrite/core-flight-measurements.md (script
  docs/rewrite/sections/turns/measure_core.py). Key: coasting holds altitude (50.0% after
  120 s; v2 climbs to 95%); full stick settles at 111 m/s, 25 deg/s (v2 139, 38.6); the store
  is crossed in about 1.5 s per half (a tuning question); firing costs 6 to 7 m/s per second.
  A10 (core charts) handed to a background agent (chart scripts take FLIGHT_MODEL).
- 03:55 B1: the observation's speed and velocity are in the world frame (proper speed over
  the refractive index, as v2's observe.py), per the author. Test
  tests/test_observe.py (a ship inside a strong field). `pytest lean`: all pass.
- 04:35 B3 and the game half of C4: fields drawn per environment at every reset, and field
  rows in the observation.
  - `GameConfig.field_count` (0: the fixed `fields` as before; `GameConfig.core()` uses 10)
    and v2's real-scale elimination map rules (src/bnb/game/physics/spawning.py `new_map`,
    values from `api.rules_of(PRESETS["real_scale_elimination"])`): point-symmetric pairs
    plus a centre field for an odd count; centres uniform in 0.85 of the arena; radius
    log-normal, median 400 m, sigma 0.6, clamped to [120, 1200] m, times the map scale
    (arena / 2,600 m: 1.615 at 4,200 m, so median 646 m); band 160 m; level -2, -1, +1, +2
    half-octaves, a quarter each. Stored per environment as (E, 1, F) constants;
    `reset_done` redraws them where a game ended; `medium` broadcasts over environments.
  - Contract: `Obs.field` (E, F, 3: radius / 1000 m, band / 40 m, level / 2, v2's input
    scales), `Obs.field_pos` (E, F, 2), `Obs.field_mask` (E, F). The model does not read
    them yet (C4's model half).
  - Sandbox: draws the first environment's drawn fields; Tab keeps the fields.
  - Checked: tests/test_fields.py (5 tests: v2's bounds and symmetry; per-environment fields
    and medium; redraw only where done; observation rows; a 60-tick core game with fields
    stays finite). `pytest lean`: 63 passed, 1 xfailed.
- 05:05 A10 and a fix found by it.
  - A10: the chart scripts draw either model (`FLIGHT_MODEL=core`, files suffixed `-core`);
    v2 charts re-rendered pixel-identical. Core charts: doghouse per ship and comparison,
    who-wins, acceleration, dive-climb, bleed, circles. Store levels for core: 97.5% (inside
    the top band), 50% (spawn), 2.5% (inside the bottom band).
  - Fix: inside an edge band, dive and climb faded only the trade, not their effect on the
    wing, so diving on an empty store still freed lift for the turn (a free turn boost on
    the deck) against the decision "at the edges dive or climb acts as coast". Now the band
    taper fades dive and climb's vertical demand back toward 1 g as well as the trade, and
    a vertical shortfall (sinking) is the only part scaled by the bottom band. A first
    version faded the trade twice (band crossing 3.5 s instead of 1.6 s); corrected, the
    measurements are unchanged from 03:45. New test: at the edges dive and climb act as
    coast on the wing. `pytest lean/game`: 32 passed, 1 xfailed.
  - Core numbers (charts, coast, spawn store; v2 in brackets): best sustained Vizia 160 m/s,
    32 deg/s [260, 54]; Jujak 160, 39 [248, 64]; Yongma 164, 26 [262, 44]. Corner and top
    speed unchanged. Minimum full-stick radius now away from the stall: Vizia 213 m at
    224 m/s. So the three speeds separate: minimum radius about 220 m/s, best sustained
    about 160 m/s, corner 268 m/s (v2: all three at about the corner).
- 05:40 B5: the cheaper damage form behind `GameConfig.damage_model` ("v2" default, "core"
  in `GameConfig.core()`).
  - Head-on term: v2's curve written in the hit angle's cosine, (1 - c) / (1 - cos theta_0)
    in place of (theta / theta_0)^2, which keeps the half-damage angle (45 deg) and differs
    from v2's curve by at most 0.0235 (largest at 68 deg). No atan2 or cross product; no new
    constant to tune. (Rejected: a power of (1 + c) / 2, 0.088 off; a smoothstep in c,
    0.092 off.)
  - Closeness: kept v2's. The agreed shortcut (the hit test's own miss distance) gave 6.8%
    less damage over 150 ticks x 64 games: it is the closest approach within the tick, not
    the distance to the bullet's line of flight (v2's impact parameter), so it changes the
    game. So the saving is smaller than proposed: a few operations, not about 25.
  - Checked: tests/test_core_damage.py (the curve within 0.024 and the same half-damage
    angle; total damage within 2% of v2's on the same hits, the same hits doing damage).
    `pytest lean/game`: 34 passed, 1 xfailed.
- 06:15 B6 (ghost-bullet hit labels, game side) and agreed fix 12 (stored `fired` flag).
  - `GameConfig.ghost_bullets` (False by default, True in `GameConfig.core()`): every living
    ship writes a bullet in its ring every tick (the ring has 11 slots, one per tick of a
    bullet's life); `State.bullet_real` marks real shots; only real bullets damage or earn
    kill credit; a ghost ends at its first contact like a real bullet. `State.bullet_fired_tick`
    stamps each bullet with its tick.
  - `StepOut.shot_outcome` (E, N, K): for every bullet whose flight ended this tick, real or
    ghost: 0 missed, 1 hit an enemy, 2 hit a team-mate (-1 otherwise); `StepOut.shot_tick`
    the tick it left the gun, which is the tick the hit head labels. The learner does not
    read these yet (the hit head is part of the model and learner work).
  - `State.fired` is stored by the step; the observation's "fired last tick" reads it (it was
    inferred from the ring and missed a shot that hit on its first move), and its "bullets in
    flight" share counts real bullets only. The sandbox draws real bullets only.
  - Checked: tests/test_ghost_bullets.py: with the same actions and noise, the game with
    ghosts equals the game without exactly (ships, shields, deaths, kill credit; 400 ticks,
    32 games, with deaths); every living ship stamps a bullet each tick; every bullet reports
    its outcome once, within its life. `pytest lean/game lean/scripted`: 43 passed, 1 xfailed
    (v2 parity unchanged: ghosts are off by default).
- 06:30 F3 (game side): docs/rewrite/scoreboard.md. Step 1,339 operations (v2 config, no
  fields), 1,675 with 10 fields, 1,692 core config; observe 165 with fields; reset_done 266
  to 401; scripted act 172. Over the 1,300 tripwire before fields; fields add about 340
  (fix 11 halves it). Policy, compile and new-size numbers wait for the model port and the
  GPU.
- 07:05 Agreed fix 11 (part): the medium cached in the state (`State.medium` (E, N, 3),
  `State.bullet_medium` (E, N, K, 3)), as v2 caches it: the ships' end-of-tick values are the
  next tick's start values; a new bullet takes its ship's; the observation reads the cache.
  Evaluations per tick: 4 to 2. Step with 10 fields: 1,675 to 1,510 operations; core config
  1,692 to 1,527 and 25.1 to 13.7 ms (E = 256, CPU). v2 parity unchanged (the comparison
  tests take v2's own cached values). Whoever places a ship by hand must refresh the cache
  (the observation test and the sandbox do; `lean.sandbox.world.refresh_medium`). Also fixed:
  the sandbox dropped the drawn fields at start (it rebuilt the constants).
- 07:20 C1-C3, C5-C9 and the model half of C4 (background agent, reviewed): v2's model maths in
  lean/model with the decided changes. New files temporal.py (v2's Griffin block: input
  projection, 4-tap causal convolution, RG-LRU, GeLU gate branch, output projection, then v2's
  4x MLP; paper initialisation, recurrentgemma code for the rest; doubling scan; the
  convolution buffer in the carry, now (blocks, E, 1+N, 4, width)) and position.py (position
  code "fourier" | "grid_cells" | "v2_codes", chosen at construction). policy.py: v2's encoder
  (per-kind first layer, shared second), fields as key/value memory in the same attention,
  element-wise gate (released-code initialisation, mean 0.500 at start), no null key, rotary
  5 bands 64 m to 8.6 km with heading harmonics 1 and 2, the same in both heads, values
  rotated; 4x MLP; 64 linear HL-Gauss bins. Parameters 1,997,681 (was 616,677; v1 about 2.0M).
  - Operations per `step`: 509 compute (+294 view). The 650 tripwire is v1's 649 compute
    operations per policy call, so the model is under it on the same rule (the agent's
    "803" counted views as well).
  - Checked: `pytest lean`: 78 passed, 1 xfailed (23 model tests: carry and convolution
    buffer under resets and masks, field permutation and F = 0, scan against loop, gate start,
    decay range, each position code, compiled step against eager). Smoke on CPU and GPU
    completes (first value loss 4.158 = ln 64).
  - Finding: at the default size (256 envs x 128 ticks) the update runs out of the 8 GB with
    the default 4 minibatches (the model is 3x larger, 4x MLP, Griffin block, and the doubling
    scan keeps its levels for the backward pass). 8 minibatches: 4.8 GB peak, about 2,800
    steps/s; 16: 2.5 GB, about 3,150 steps/s. v2's recipe used 32 (backlog D2).
  - Choices for the author (lean/model/NOTES.md): the game token uses v2's "centre" rotary
    (identity) rather than lean's "zero"; rotary on v2's two axes per band turned 35 deg per
    band rather than lean's three hexagonal axes; heads as v2 (one hidden layer of 256);
    next-state head's last layer orthogonal 0.01, not v2's zero; depth scale 2/6;
    sqrt(1 - a^2) kept at a reset as v2; grid cells with Blair's exponential gain; fields'
    rotary heading 0 as v2's memory tokens.
- 07:50 E1 (part, "tell, do not infer"): the game's constants carry each type's cruise speed,
  corner speed, bullet launch speed and gun range (`core.derived_characteristics`, solved once
  per flight model with the game's own flight functions, cached; v2's definitions). Values
  match the measured top speeds (Yongma 334.9, Jujak 316.2, Vizia 325.4 m/s) and the scripted
  opponent's former Vizia fallbacks exactly (634.8 m, 2,037.4 m/s, 268.2 m/s); the opponent
  reads them per type (it flew Yongma and Jujak with Vizia's). Test in
  lean/scripted/tests. `pytest lean/game lean/scripted`: 44 passed, 1 xfailed.
