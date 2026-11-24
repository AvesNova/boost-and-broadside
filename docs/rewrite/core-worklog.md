# Lean core: work log

Newest last. Each entry: time (KST), item from core-backlog.md, what was done, how it was
checked, the commit. Numbers are measured unless marked otherwise.

## 2026-11-23

- 02:37 Overnight session starts (author asleep until 09:00). Plan: section A (flight
  physics behind a switch), then B1, then section C when blocked. Machine: laptop,
  platform profile "performance", GPU idle.
- 02:43 A1-A8, A11: the core flight model behind `GameConfig.flight_model` ("v2" default,
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
- 02:45 A9: measurements in docs/rewrite/core-flight-measurements.md (script
  docs/rewrite/sections/turns/measure_core.py). Key: coasting holds altitude (50.0% after
  120 s; v2 climbs to 95%); full stick settles at 111 m/s, 25 deg/s (v2 139, 38.6); the store
  is crossed in about 1.5 s per half (a tuning question); firing costs 6 to 7 m/s per second.
  A10 (core charts) handed to a background agent (chart scripts take FLIGHT_MODEL).
- 02:47 B1: the observation's speed and velocity are in the world frame (proper speed over
  the refractive index, as v2's observe.py), per the author. Test
  tests/test_observe.py (a ship inside a strong field). `pytest lean`: all pass.
- 02:52 B3 and the game half of C4: fields drawn per environment at every reset, and field
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
- 02:57 A10 and a fix found by it.
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
- 03:01 B5: the cheaper damage form behind `GameConfig.damage_model` ("v2" default, "core"
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
- 03:06 B6 (ghost-bullet hit labels, game side) and agreed fix 12 (stored `fired` flag).
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
- 03:07 F3 (game side): docs/rewrite/scoreboard.md. Step 1,339 operations (v2 config, no
  fields), 1,675 with 10 fields, 1,692 core config; observe 165 with fields; reset_done 266
  to 401; scripted act 172. Over the 1,300 tripwire before fields; fields add about 340
  (fix 11 halves it). Policy, compile and new-size numbers wait for the model port and the
  GPU.
- 03:10 Agreed fix 11 (part): the medium cached in the state (`State.medium` (E, N, 3),
  `State.bullet_medium` (E, N, K, 3)), as v2 caches it: the ships' end-of-tick values are the
  next tick's start values; a new bullet takes its ship's; the observation reads the cache.
  Evaluations per tick: 4 to 2. Step with 10 fields: 1,675 to 1,510 operations; core config
  1,692 to 1,527 and 25.1 to 13.7 ms (E = 256, CPU). v2 parity unchanged (the comparison
  tests take v2's own cached values). Whoever places a ship by hand must refresh the cache
  (the observation test and the sandbox do; `lean.sandbox.world.refresh_medium`). Also fixed:
  the sandbox dropped the drawn fields at start (it rebuilt the constants).
- 03:13 C1-C3, C5-C9 and the model half of C4 (background agent, reviewed): v2's model maths in
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
- 03:17 E1 (part, "tell, do not infer"): the game's constants carry each type's cruise speed,
  corner speed, bullet launch speed and gun range (`core.derived_characteristics`, solved once
  per flight model with the game's own flight functions, cached; v2's definitions). Values
  match the measured top speeds (Yongma 334.9, Jujak 316.2, Vizia 325.4 m/s) and the scripted
  opponent's former Vizia fallbacks exactly (634.8 m, 2,037.4 m/s, 268.2 m/s); the opponent
  reads them per type (it flew Yongma and Jujak with Vizia's). Test in
  lean/scripted/tests. `pytest lean/game lean/scripted`: 44 passed, 1 xfailed.
- 03:18 E1 (part): the scripted opponent never shoots through a team-mate: a living team-mate
  ahead along the nose, nearer than the target and within 2 of its hit radii of the line of
  fire zeroes the gun gate (`ScriptedConfig.ally_clearance_in_hit_radii`). Test: blocked at
  200 m on the line, clear at 150 m off it. The check costs 21 operations: the scripted call
  is now 193, over the 170 tripwire (question 11 for the author). `pytest lean/scripted`:
  8 passed.
- 03:19 F2: docs/provenance.md, from the literature drafts plus rows for the decisions taken
  since (gate initialisation, null key dropped, Griffin block ported, rotary layout, rotated
  values, the league floor and tally, the core flight model, firing, the damage term, ghost
  bullets, the medium cache). Kinds as in core-brief.md section 3.
- 03:20 Author's answers: the band edges are a ceiling and a floor with a gradual ramp (as
  built); the required order is minimum sustained radius, best sustained rate, corner speed.
  New test (tests/test_core_flight.py): the core model meets the order for all three types.
  Correction to my 02:57 note: the order I gave (best sustained, then minimum radius, then
  corner) used the instantaneous full-stick radius; on the sustained basis the order is the
  author's.
- 03:29 D1 (background agent, reviewed): the league as the author designed it. Half the
  environments (fixed by index) are self-play with two live seats, both training; half are
  league games with one live seat against a member drawn at each reset. `League` (pool.py):
  member 0 the scripted agent plus up to 8 snapshots; draws 7/8 by (1 - p)^2, 1/8 uniform;
  per-member tallies decayed by 0.99 per game, an unplayed member at 0.5; win rates from the
  training games (no lanes). The live policy runs once per tick over all seats; the scripted
  action is still the BC target for every live ship. Tests: lean/learn/tests/test_league.py
  (8). `pytest lean`: 89 passed, 1 xfailed; smoke on the CPU completes.
  - Choices for the author (lean/learn/NOTES.md): the BC gate reads the tally against the
    scripted member (0 until the first scripted game, not the 0.5 prior); a replaced member's
    running games are re-dealt; the self-play/league split fixed by index; memory now 1.5x
    the rows per environment (the default size needs 12 to 16 minibatches on 8 GB).
  - D2 report: docs/rewrite/recipe-v2-vs-lean.md (v2's release run, commit 4cfd1e86). Main
    differences: v2 cloned in a separate 100M-step BC stage and started RL from the clone's
    EMA with no BC term; v2's updates are 1.05M ticks in 32 minibatches with a KL stop at
    0.02; outcome discount 0.9997 (lean 0.998); advantage scaled by a slow EMA of the
    returns' spread; 81 symlog bins to +-19; zero-sum rewards with a damage ledger and kill
    assists; weight averaging (EMA 0.95 and a uniform average, both league members).
- 03:31 Correction: the times in this log from 02:43 on were first written from my estimate,
  not the clock; they now come from the commits' times (git log). The whole session so far,
  02:37 to 03:29, took 52 minutes.
- 03:36 F3 (GPU part): scoreboard.md "GPU". Game step (core) 9.4 ms uncompiled, 0.95 ms
  compiled at 5v5, E = 2048 (cold compile 18 s); policy step 28.9 / 14.4 ms (compile 12 s);
  the policy dominates the rollout. A new team size recompiles once (about 15 s) then never;
  dynamic=True also recompiles once (to investigate). Uncompiled game at 50v50, E = 2048 runs
  out of memory; compiled fits (35.8 ms). Measured from a clean worktree of 25478e10 while
  two agents edit lean/ (learner targets; fog).
- 03:39 F3: the recompile at a new team size is duck sizing (10 slots = 10 fields at 5v5); with duck sizing off and dynamic=True each graph compiles once (game 30 s, policy 18 s) and 10v10 to 50v50 cost nothing. Proposed for D5.
- 03:43 F4 (gate G2, provisional): play on the CPU. Real game and new model at batch size 1, one
  thread: uncompiled p99 7.8 / 10.8 / 19.3 ms at 5v5 / 20v20 / 50v50; compiled p99 4.0 (5v5),
  6.1 (20v20), 12.6 ms (50v50). Compile 31 to 34 s cold, 12 s warm: start uncompiled, switch
  when a background build is ready. Machine loaded by agents: to re-measure quiet.
- 03:45 B7 (fog, game side; background agent, reviewed): `GameConfig.fog` (False by default;
  True in `GameConfig.core()`). v2's sight test ported exactly (all-round, 1,000 m for every
  type, fields block only through their cores; sight.py), plus the author's reveals (firing,
  a real bullet's hit; not boundary damage; that tick only). Last-seen records per ship
  (`State.seen`, `State.ticks_since_seen`), `Obs.visible` and `Obs.since_seen`
  (symlog seconds, as v2's model reads). The observation under fog is built only from the
  team's view (no hidden information: tested by perturbing a hidden enemy); the scripted
  opponent and chaser read their team's view. Fog off is bit-identical to 25478e10 (200
  ticks, both configs). Tests: lean/game/tests/test_fog.py (8). `pytest lean/game
  lean/scripted`: 54 passed, 1 xfailed. Operations with fog: step 1,593 (core), observe 110,
  scripted 208.
  - Choices for the author: no reveal at a spawn (v2 reveals spawns); deaths known only when
    seen (v2 shows all deaths; a bullet kill is always seen since the hit reveals); wrecks
    can be sighted; only real bullets reveal on a hit. The model does not read `visible`
    and `since_seen` yet, so a stale last-seen row looks current to it: the model must
    read them before any training under fog (backlog C10).
- 03:49 D2 (part) and the hit head (background agent, reviewed): v2's five reward levels (damage,
  kill_death, objective_progress, objective, outcome) with v2's weights, discounts (0.982,
  0.990, 0.995, 0.999, 0.9997) and lambdas (0.81 to 0.97); zero-sum damage and kills with
  v2's damage ledger and kill assists; boundary deaths pay the other team; shaping removed.
  Supports from measured returns (damage +-7, kill_death +-8, outcome +-12, objective +-1
  placeholders) and range monitoring every epoch (clamped share and bins in use per level;
  a WARNING line in train.py). The hit head: 3-way (miss, enemy, team-mate) per ship, labels
  written back to the tick the bullet left the gun, loss on labelled own rows only,
  coefficient 0.05; `train.py` plays ghost bullets by default, `--game core` the core game.
  Parameters 2,048,180. `pytest lean`: 113 passed, 1 xfailed; smoke completes.
  - Gap: the rewards rebuild attacker-by-target damage from per-ship totals (approximate when
    one attacker hits two targets, or two attackers one target, on the same tick) and copy the
    game's boundary formula. Fix: StepOut fields hit_damage (E, N, N), boundary_damage and
    shield_restored (next item).
  - Lint: lean's files are written at 120 columns against pyproject's 100 (141 long lines
    before, 176 now in learn/model/train): a cleanup to decide (reformat or raise the limit).
- 03:51 The rewards' gap closed: StepOut carries `hit_damage` (E, N, N) [attacker, target] from
  real bullets (one scatter_add in the step), `boundary_damage` and `shield_restored`
  (the step's own values). `rewards.tick_events` reads them, so the ledger is exact (no
  reconstruction from totals, no copy of the boundary formula). Test: the matrix's rows and
  columns match the per-ship totals over 200 ticks with hits (test_damage_detail.py).
  `pytest lean`: 114 passed, 1 xfailed.
- 03:55 C10: the model reads fog: `Obs.visible` and `Obs.since_seen` enter the ship encoder
  (two inputs), so a last-seen row no longer looks current. Test: a ship's outputs change with
  its own visible flag and time since seen. `pytest lean`: 115 passed, 1 xfailed. A smoke
  run of the whole core game (`python -m lean.train --smoke --game core`: core flight, drawn
  fields, ghost bullets, fog, the new model, the league, v2's rewards) completes on the CPU.
- 04:20 Charts: the core doghouse is one panel (coasting does not depend on the store; the footnote says how dive and climb fade at the edges); THEME=dark renders dark versions. P-51D model and the port of v2's tuning code handed to a background agent (worktree).
- 04:37 Merged the P-51D model and the port of v2's tuning code (lean/game/tuning.py; card types p51d and vizia_plain; SHIP_TYPES 5, so ship features 27 -> 29). Write-up docs/rewrite/p51d-model.md. Under the core model the P-51D flies 4% slower, climbs 14% less and sustains a 17% weaker turn than the real aircraft (the core model's shared terms: drag rise, thrust and lift soft minimums, separation drag); with them off the card reproduces the October fit exactly. Vizia's climb modifications already have no effect in the core model except the dive/climb rate (0.983 g against 1 g). Doghouse titles on two lines. `pytest lean`: 124 passed, 1 xfailed.
- 04:52 B4 merged (game-side waste fixes 1-5 and 8; agent, reviewed): packed constants (uniform ones compiled in, per-type rows in one (E, C, N) table), packed state, dead ships held by one select, fixed tensors built once, velocity stored, a no-fields path. Each fix same within float32 rounding as the one before. Step 1,348 -> 1,148 operations (v2 config, no fields), core 1,599 -> 1,522; reset_done 281 -> 87 (0.56 ms) and core 487 -> 280. Merge with the P-51D work: added `core.per_type_constants` (any card rows, uncached, for tools and tuning; the run's card is cached and holds only its lineup), sized `_per_type` from the card, moved the chart and measurement scripts and the tuning code to the new interface (flight_tick takes the fields), applied the agent's sandbox patch. Checks: `pytest lean` 124 passed, 1 xfailed; measure_core.py's numbers identical to before; the charts re-render pixel-identical; the sandbox runs (both configs). Not done (need lean/learn): one scripted call for both teams, personas at reset, batched resets.
- 05:58 D5 merged (compiled training loop; agent, rebased on 681eb492, no conflicts): `--compile` compiles the tick halves, the snapshot move, the bootstrap, the target pass and the minibatch loss; no graph breaks or recompiles measured on the GPU. bf16 autocast on by default (`LearnConfig.bf16`), minibatches 8, `targets_every_epoch` setting (default True). League snapshots in one batched call (vmap over stacked weights), no per-tick read-back (collect 20.5k -> 26.0k). Steps/s (agent's numbers, laptop 4070, performance profile, league full): v2 game 5,160 at 256 envs and 5,057 at 1,024; core game 4,640 and 4,589; eager about 2,100; fp32 compiled 2,992. Start-up 199 s cold, 72 s warm. Three lean.game workarounds are patched at runtime from lean/learn/compiled.py (None generators, the ship card built outside the trace, `dataclasses.fields` of the instance): to move into lean/game. Checks after the merge: `pytest lean -m "not slow"` 126 passed, 1 xfailed; smoke runs complete on CPU and GPU. Decisions in docs/questions.md question 12.
- 06:34 B4 learn side merged (agent): the three runtime patches moved into lean/game (`contract.random_keywords` leaves out a None generator; built ship cards kept in `core.ship_card`, looked up when traced; `dataclasses.fields` of the instance), so compiled.py patches nothing. Fix 6: one scripted call for both teams (`core.enemy_view` gives each ship as its enemies last saw it); bit-identical to the two single-team calls. Fixes 7 and 9: E deals (sizes, game deal, personas, learner team, offset) drawn at each segment start; the k-th game to end takes deal k with no host read; league members still drawn when games end. Changes which draws each deal gets, not the distribution; more than E endings in a segment reuse deals (counted in `Rollout.repeated_deals`; not seen, shortest episodes 403 ticks against 128-tick segments). Operations per tick (agent's, eager CPU 5v5): v2 2,618 -> 2,345, core 3,273 -> 2,791; scripted teacher 370 -> 185. Scripted call 188 against the 170 tripwire (question 11 updated). GPU (agent's, one run, compiled 256 envs v2): loop 5,160 -> 5,257. My checks after the merge: `pytest lean -m "not slow"` 129 passed, 1 xfailed; CPU smoke completes; sandbox runs headless in both configs.
- 07:25 D6 merged (agent): `crossover` plays every network size at once, in rounds of batched matches (mixed team sizes padded to the largest; the walk plus 2 neighbours either side, then 4 in the walk's direction). Departure from v1: each size starts at the previous evaluation's ratio (parity at the first), since v1's chained start (from the previous size's crossover) cannot run in parallel. Same answers: the agent compared old and new win rates per match (128 to 384 games, sum of z^2 14.9 over 14 comparisons). Timing (agent's, one run each, untrained policy, eager): from parity the new code is slower (CPU [4,8] 47.6 -> 267.4 s; GPU [4,8] 128 games 67.6 -> 83.0 s), from the previous ratios about even (CPU 47.6 -> 58.1 s; [4] 56.0 -> 40.4 s); with every match needed, one batch beats one at a time (202.8 -> 134.9 s). Not shown: a GPU gain; memory at large sizes (`maximum_games` uncapped). My checks: `pytest lean -m "not slow"` 131 passed, 1 xfailed; CPU smoke completes.
- 11:19 Rollout streaming, micro-batches and compact storage merged (agent, restarted after an accidental stop; built on the stopped agent's work). Parity defaults: 2,048 environments x 4 segments x 128 ticks, 32 minibatches, micro-batches of 64 views, targets once per update. Agent's measurements (laptop 4070, compiled, bf16, performance profile): v2 game 6,124 steps/s (peak 5.18 GiB), core game 5,586 (5.58 allocated, 7.27 reserved), targets every epoch 4,860; streamed and all-on-GPU equal where both fit (1,024 x 2: 4,997 against 4,999); all-on-GPU at parity runs out of memory. Bytes per environment tick 3,249 -> 2,094 (v2), 3,564 -> 2,097 (core); store 2.2 GB pinned. Checks: micro-batched step equals one pass (rtol 1e-4); streamed equals unstreamed exactly; compact against full precision at initial weights, log-probability change at most 1.9e-6. Judgement calls for the author in author-review.md 9.1. My checks after the merge: `pytest lean -m "not slow"` 138 passed, 1 xfailed; CPU smoke completes.
- 14:36 Literature review of model choices (docs/rewrite/literature-model-choices.md). Bug it found, fixed: grid cells with 2 phases per axis are even (f(x) = f(-x) to float rounding), so a position and its point reflection got the same code; default grid_phases 3, and a test that fails at 2 and passes at 3. Grid cells are not the default code (Fourier is), so no run used it. `pytest lean -m "not slow"` 139 passed, 1 xfailed.
- 16:05 Chained returns merged (bug fix): GAE (all five levels, both target modes) and hit labels run across an update's 4 segments per seat, bootstrap only after the last segment. Test: equals one GAE pass over the concatenated ticks (rtol 1e-5), with episode ends inside and at a segment end; fails per segment. GPU (agent's, one row, parity, v2 game): loop 6,124 -> 6,002 (-2%, one row each side, may be variation). My checks after the merge: `pytest lean -m "not slow"` 142 passed, 1 xfailed; CPU smoke completes.
- 17:16 Flight tweaks merged (agent): turn has precedence over the climb (1 g, then turn, then climb gets the rest; coast and dive unchanged bit for bit); seven flight terms per ship type (card.FLIGHT_TERMS; Vizia and v2 types unchanged, P-51D row now its point mass: 375 mph, 3,599 ft/min, best sustained 18.6 deg/s at 169 mph, corner 294 mph at 34.0 deg/s); sandbox --time-scale K (tuning.time_scale, sandbox only); P-51D against Vizia doghouses in game units and at 1x (sections/turns/doghouse-p51d-vizia-core*.png). My checks after the merge: `pytest lean -m "not slow"` 147 passed, 1 xfailed; sandbox headless runs with --ship p51d and --time-scale 1.
- 18:12 G1 merged (agent): the core game is the only game (`GameConfig()`; v2 flight and damage, fixed fields, `core()`, `--game`, `--core` removed); only the P-51D row remains (yongma, jujak, vizia were v2's solved rows, not built by the tuning route; vizia_plain gone); step split into named stages; StepOut damage_dealt, friendly_damage, killer, fired and Obs.field_mask removed; team_view built on enemy_view; ShipCard merged into Constants; arena radius in the game's constants. Bit-identical states, outputs, observations and scripted decisions on a 300-tick capture (scripted differs only by reading the real arena radius). Operations: step 1,630 -> 1,552. Source -571/+498, tests -993/+175 lines. My checks: `pytest lean -m "not slow"` 127 passed; smoke completes; sandbox runs.
- 18:34 G2 merged (agent): clock tie-break (wipe; then living ships, then total shield, else draw); spawn setting, default mixed (0.85 R disc), sides over half discs; edge rule: an outward heading within 1,000 m of the boundary is mirrored in the tangent; training spawn (speed stall to top, shield and store uniform, 1/8 standard per environment) on in train.py, forced off in eval; one-tick action delay for every agent, own living ships' pending action observed (42-way one-hot), enemies none; x/R, y/R dropped (18 ship scalars); scales from measured ranges (table in lean/game/NOTES.md). Step 1,552 -> 1,557 operations, observe 94 -> 102. My checks: `pytest lean -m "not slow"` 142 passed; smoke completes; sandbox runs.
- 19:07 S merged (agent): dive/climb power rule from the ship's own numbers (dive when slow or below corner in a hard turn, climb when well above corner or very close, regain the store when far and low); 0.8 target memory and the head-on side (per ship per match) carried by every caller (collect, eval, sandbox); personas per ship on a bounded box at spread > 0, None at spread 0; action hold, fallback numbers and the renaming shim removed; lead corrected for the medium. Scripted 4v4 (256 envs x 5,400 ticks): diving 86% -> 22% of ticks, coasting 4% -> 73%, kills per environment-minute 4.82 -> 5.44; mean speed 0.70 corner speeds and mean store 0.05 in both. One call 186 -> 208 operations. My checks: `pytest lean -m "not slow"` 158 passed; smoke completes; sandbox runs.
- 19:43 M merged (agent): rotary 10 bands at ratio 2 (64 m to 32,768 m), band k turned k x 55.62 deg, heading harmonics 1-2, 22 of 32 pairs (68.75%); fields' rotary heading to the centre; game token at the centre; recurrentgemma init for spatial sublayers; recurrence input 1 when its state starts from zero; last layers orthogonal 0.01; hit biases from measured shares; next-state head 6 columns x 64 HL-Gauss bins in nats (table in lean/model/NOTES.md); pending action as an embedding. Parameters 2,058,420 -> 2,155,565. GPU (agent's, parity, core game now): 5,260 steps/s, 5.73 GiB, first update 403 s cold. My checks: `pytest lean -m "not slow"` 164 passed; smoke completes.
- 20:49 L1 merged (agent): BC stage on the teacher's full distribution (all heads and critics; 96 updates, about 100M ticks), RL from its EMA with a teacher term 0.5 x 0.95^k switched off at the 0.45 gate; EMA (0.95 per update) and uniform averages as side copies; fused Adam, eps 1e-5, clip 1.0, non-finite steps skipped without a host read; warm-up 2,000 Adam steps then constant; per-epoch KL logged with a warning above 0.02; advantages over their standard deviation, level shares logged; crossover is now `python -m lean.eval`; test stand-ins removed (tests on the real packages); modules collect -> rollout, update -> training. GPU (agent's, parity, core game, 3 + 3 updates): 6,669 steps/s, 5.70 GiB. KL: 0.0017 at BC update 0, then 0.09 and 0.18 as BC moves the policy fast (entropy 3.74 -> 2.62); the measure is sound. My checks: `pytest lean -m "not slow"` passed; smoke completes.
- 22:30 L2 merged (agent): members at fixed indices (scripted, EMA, uniform average, up to 12 snapshots); capacity ceil(1.25 L / members) with PFSP draws restricted to free members; one compiled call per network slot on its own rows (the vmap form recompiled on every league change); retiring slots (4) finish a dropped snapshot's games; v2's snapshot steps and retention (newest 8 plus one per 2^k age, cap 12); checkpoints every 10M steps with everything resume needs, `--resume`. League cost per tick at parity: 81.5 ms (every slot on every row) -> 12.75 ms (8 slots), 15.29 ms (11). Checkpoint 249 MB at 10.49M steps. My checks: `pytest lean -m "not slow"` 197 passed; smoke completes with the league.
- 22:37 P merged (agent): league snapshots written once to DIR/snapshots, checkpoints reference them by id; averages' slots rebuilt on resume. Checkpoint constant at about 5 weight units (smoke: 43.5 MB at any snapshot count, was 78 to 113 MB); at parity by arithmetic about 140 MB plus 28 MB per snapshot, about 7.7 GB per 500M steps (was about 15 GB). My checks: `pytest lean -m "not slow"` 198 passed.
- 00:32 E2 merged (agent): game references (best sustained rate, speed, radius; quarter-turn time; store turn); scripted parameter table (refactor bit-identical); situation, tactic weights, committed sense with Shaw's mid-fight switch and overshoot rule, speed-target power, graded defence, lead turn from geometry. Scripted brawl: mean store 0.05 -> 0.23, kills per env-minute 5.44 -> 6.45. New against old 0.413 +- 0.019 (640 games). One call 208 -> 439 operations. My checks: `pytest lean -m "not slow"` 224 passed; smoke completes.
- 00:35 Paused for the author's shutdown. K (compile): the doubling was not a doubling at a fixed size (199 s was at 256 envs, 403 s at parity, whose first update includes about 170 s of real training); the real growth was L2's recompile of slot_actions (a view's base guard), fixed on branch lean-k-compile (282 -> 244 s cold at 256 envs, quiet machine; 0 recompiles); warm cache 89.5 s; tripwire script compile_check.py committed without a baseline; the per-slot CUDA graph tried and reverted (no gain at parity). Not merged: needs a rebase onto E2 and quiet re-timing. W (web tool): nothing built. Agent notes and safeguards copied from /tmp to data/agent-notes-2026-11-24 (tmpfs is wiped at shutdown).
- 02:40 K resumed alone (quiet laptop) and finished on branch lean-k-compile, rebased on 245e537f: the L2 recompile fixed (scripted rows copied, not a view: cold 258.3 -> 237.2 s at 256 environments, 5 -> 4 graphs); quiet compile table D5 to E2 in scoreboard.md "Compile time" (E2: +332 operations in act, +3.9 s); tripwire compile_check.py with compile_baseline.json (5,721 operations, 221.3 s, budget 260 s); CUDA-graph league call and regional compilation tried, neither kept.
- 02:43 K merged: cold compile, quiet, 256 envs: D5 215 s -> L1 215 s -> L2 as merged 254 s (a recompile, 5 graphs) -> E2 258 s -> fixed 237 s (4 graphs, 0 recompiles); warm cache 93 s. E2 added 332 traced ops to act and about 4 s. Tripwire compile_check.py with baseline 4 graphs, 5,721 ops, 221.3 s, budget 260 s. Regional compilation fails to trace in torch 2.15 with dynamic shapes. My checks: `pytest lean -m "not slow"` 224 passed.
- 02:58 Evaluation throughput (agent, rough): scripted vs scripted saturates near 1M environment ticks/s (about 750 4v4 games/s) from E 16k at 0.6 GB; network vs scripted flat about 120k ticks/s (policy-bound). Persona tensors do not recompile; one yardstick float does. Table in scoreboard.md.
- 03:09 W merged (agent): `python -m lean.tool` (after `uv sync --extra tools`) serves a Ships tab (preset, time scale, seven sliders through the fast handles solve, live doghouse and sustained curves against the P-51D reference, readouts; 0.5-0.8 s per slider change) and a plain Sandbox tab (the real World at 30 Hz over a WebSocket); tuning.envelope is the one chart computation, shared with doghouse.py; the Tk sandbox is deleted. My checks: `pytest lean -m "not slow"` 230 passed.
- 03:51 A1 merged (agent): P-51D through the tuning route with the game terms on (drag rise at Mach 0.77 per NACA ACR 4K02, separation drag 0.1, lift softness 16, thrust softness 8), wing 21.67 m^2, CLmax 1.6, solver keeps 375 mph, 3,600 ft/min and the best sustained rate; automatic flaps (+15% lift below 1.3 stall, clean by 1.6 stall, drag only beyond the clean maximum). Tightest sustained turn 202 m at 136.7 mph against best rate 18.99 deg/s at 165.5 mph (ratio 0.826, test limit 0.90). Corner rate 32.2 deg/s, 8% under the point mass (lift softness). Step 1,557 -> 1,625 ops; compile tripwire within budget (225.1 s against 221.3). My checks: `pytest lean -m "not slow"` 232 passed.
- 05:11 V merged (agent): lean/eval runner (rolling slots with a drain, deals every 32 ticks, scripted rows as per-ship tensors, networks through functional_call), SQLite store (agents, conditions with a game version, games, counts view, Sobol), adaptive allocation, Bradley-Terry Elo, CLI (play, report, elo, crossover); crossover moved onto it. Scripted vs scripted 810 games/s at 4,096 envs; network vs scripted 122k env-ticks/s; 0 recompiles for new scripted vectors, networks, team sizes 1-6, arena radius, spawn, clock (fog off compiles a new graph). Drain unbiased (0.4656 +- 0.0019 full vs 0.4654 +- 0.0034 drained). Tripwire 227.3 s, 4 graphs. My checks: `pytest lean -m "not slow"` 239 passed; smoke completes.
- 05:55 Review round 1 fixes merged (agent): B1 retiring slots keep their games (test fails before), B2 allocation, B3 network identity includes lean/model and eval/runner.py (tuning.py out of the game version), B4 atomic checkpoints, B5 evaluation uses the training precision helper, B6 --start needs --stage rl, B7 one spawn path (core.spawn) for deal and the sandbox; shared agent step in lean/model/agent.py (eval no longer imports lean/learn); circular import gone; dead test-only code removed. Tripwire 222.9 s, 4 graphs, 5,771 ops. Not done: one agent step for all three ticks (training's act keeps the direct policy call), wide signatures, the global ship card (author decision). My checks: `pytest lean -m "not slow"` 248 passed.
- 06:06 D merged (agent): every package has one short current NOTES.md (game 129->87 lines, model 173->98, learn 225->116, scripted 117->82, sandbox and tool README renamed NOTES), lean/README.md 124->51 and CONTRACT.md rewritten to the code, LAPTOP_PROMPT.md deleted, author-review.md items marked decided/done/open; every documented command checked on the CPU. Fixed myself: `lean.eval elo` crashed when the store held only self-play (bradley_terry now ignores self pairings), with a test. `pytest lean -m "not slow"` passed.
- 07:19 Screening merged (agent; about 64 min of GPU): skill = mean score against a panel (yardstick, the design BFM persona, 6 box personas) over 18 conditions (radius 3,000/4,200/6,000 m, 3v3/4v4/5v5, mixed/sides). Yardstick 0.609 +- 0.007, better than 95.8% of 4,000 box configurations; BFM persona 0.332. Nine rows carry the effect (total Sobol: lead_turn_scale 0.58, lead_gain 0.14, yo_yo_time 0.12, fight_speed 0.05, closing_time 0.05, plan_threshold 0.04, in_range 0.03, nose_tail 0.03, aligned 0.01); 22 rows show no effect. The lead turn and yo-yo hurt at every value above 0. Rankings carry across radius, team size and spawn (correlations 0.97-1.0). Surrogate R^2 0.862 (noise ceiling 0.904). Broad plateau; lead_gain and fight_speed rise past the top of their boxes. Store: data/scripted-screening/screen.sqlite (1.56M games). Write-up docs/rewrite/scripted-screening.md. My checks: `pytest lean -m "not slow"` passed.
- 07:59 CMA-ES merged (agent, 33 min GPU): candidate mean 0.690 +- 0.005 against the yardstick 0.608 +- 0.005 (9,216 games each); ahead in all 18 conditions; quarter-box neighbourhood 0.676 (96% above the yardstick); 8 held-out personas 0.735 against 0.643; head to head 0.589. Lead turn and yo-yo stayed at 0; plan_threshold and nose_tail ended at their box floors. Defaults unchanged: docs/rewrite/sections/scripted-screening/cma_candidate.json. Next: stop for the author's yardstick choice.
- 08:21 Parity row (agent): loop 6,559 steps/s (collect 49,796, update 7,554), peak 5.76 / 6.40 GiB, first update 380 s cold; against the 2026-11-23 core-game row: loop +17%, collect 2.5x (cause not isolated). Compile cache now defaults to ~/.cache/bnb-inductor. Tripwire 228.2 s, 4 graphs. Waiting for the author's yardstick choice before the clock measurement and the dry runs.
- 10:01 Style counters merged (agent): per game and seat, the share of turning-fight time, two-circle share of it, store, speed in corner speeds, shots per ship-minute, hits and range at hits, climb and dive shares; stored as columns on games, a styles view, `python -m lean.eval styles`. One clean benchmark pair: 775 against 782-793 games/s. Network agents get new ids (runner.py is in the inference version). My checks: `pytest lean -m "not slow"` passed.
- 11:06 Divergence merged (agent, about 45 min GPU): 98,014 stored ship-states; JS from the yardstick: CMA candidate 0.066 nats, BFM persona 0.129, median of the 4,000 0.115. Behaviour ranking differs from skill (fight_speed first for behaviour, fourth for skill). Of the 22 skill-flat rows, engage_range changes behaviour, five change it a little, 16 are flat in both (freeze candidates). The overshoot rule fires on 5 of 98,014 states (its track-crossing gate). No behavioural families: one continuum (MDS axis 1 power balance, axis 2 aim). Style round robin: yardstick 0.410 against the CMA candidate, 0.737 against BFM. Write-up docs/rewrite/scripted-divergence.md. My checks: quick suite passed.
- 11:21 Lead-turn/yo-yo review merged (agent): both features took the nose off the gun inside gun range (gun weight multiplied by (1 - two-circle plan) and (1 - high yo-yo)); the high yo-yo also fired at the beam (no aspect condition). Fixed: the gun owns the nose in range; high yo-yo lag gated to the rear quarter. After the fix the yo-yo is neutral within noise and the lead turn is near neutral (worse in sides spawns, where its lead pursuit beyond gun range still acts). The yardstick is unchanged; the panel's BFM member and personas fly differently, so the yardstick's panel score moved 0.609 -> 0.551. Open (author): the closure target is miscalibrated (y about -1 almost everywhere), which also drives the yardstick's speed target and stick cap. Compile tripwire re-run on a quiet machine: 224.7 s, 4 graphs, within the baseline (the agent's 430 s was at load 40). Quick suite passed.
- 12:05 CMA-ES round 2 merged (agent, about 34 min GPU, fresh store on the fixed code): BFM-default 0.529, BFM-CMA1 0.623, BFM-CMA2 0.649 (+-0.005, 9,216 games each); held-out 0.529 / 0.618 / 0.662; head to head against BFM-default: CMA1 0.591, CMA2 0.625. CMA2 is a flat plateau (neighbourhood 0.651). Divergence from BFM-default: CMA1 0.066 nats, CMA2 0.131 (mostly power: dives more, climbs less, store 0.24 against 0.30), BFM-design 0.102. yo_yo_time drifted to 0.19 (flat row). Waiting for the author's yardstick choice.
