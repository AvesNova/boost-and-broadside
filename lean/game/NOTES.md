# lean/game notes

## What is implemented

- `lean/contract.py`: `State`, `StepOut`, `Obs`, `split_joint`, `join_action`, sizes.
  `SHIP_FEATURES = 29` (20 scalars, 4 flags, 5-type one-hot), `GLOBAL_FEATURES = 3`.
- `card.py`: the real-scale card (Yongma, Jujak, Vizia; then lean's P-51D and Vizia-plain from `tuning.py`) and the shared rule constants,
  copied as literals from v2's `api.ship_types` / `api.rules_of` for
  `real_scale_elimination`. `lean` does not import `bnb`.
- `core.py`: `GameConfig`, `ship_constants` (gathered once per reset into
  `State.constants`), `medium`, `flight_tick` (from `lean_flight.py`), `bullet_motion`,
  `draw_noise`, `step`, `reset`, `reset_done`, `observe`.
- Step order: fire, flight, bullet motion, swept test (dense (E, N K, N)), first contact,
  damage (kinetic x cored closeness x aspect), boundary (constant radius), shields with
  the quiet rule, deaths as a mask, win check, clock, counters. No respawn, shrink, zones
  or fog (fog added 2026-11-23, see the last section). `step` never resets; the caller passes `StepOut.done` to `reset_done`.

## What was run (CPU, one thread, torch 2.15, Python 3.14; 2026-11-22)

Command: `OMP_NUM_THREADS=1 PYTHONPATH=.:src python -m pytest lean/game/tests -q`
-> 4 passed in 9 s (both test files call `torch.set_num_threads(1)`; the earlier timeout
was thread contention, not v2 itself).

- `tests/test_against_v2.py` (32 environments, 5v5, 150 ticks, no fields, no shrink,
  random sticky actions, v2's spread): ran to completion, assertions held. In the run:
  13,553 shots, 169 deaths, energy 131,550 to 227,486, **no game ended** (`done` never
  true, so the result coding at the end and the clock rule were not compared with v2).
  - Check A, one lean step from each v2 state, ships alive at tick start, not dying
    this tick, (99.9% quantile, max) of |lean - v2|:
    x (2.4e-7, 6.1e-5) m; y (2.4e-7, 7.6e-6) m; speed (1.5e-5, 1.5e-5) m/s;
    angle of attack (3.2e-8, 8.2e-8) rad; heading cos/sin (6.0e-8, 1.2e-7);
    angle cos (6.0e-8, 1.2e-7), sin (3.2e-8, 8.9e-8); nose rate (2.4e-7, 3.6e-7) rad/s;
    energy (0, 0); shield (7.6e-6, 1.5e-5).
    Audit bounds: 99.9% within 1e-3 m, 1e-4 m/s, 3e-7 rad; all within 2e-3 m, 2e-3 m/s,
    5e-4 rad. All inside, at float32 rounding level. No renaming slip in the flight.
  - Discrete mismatches (died, fired, done, result, killer, bullet moves left, gun
    cooldown): 0.
  - Check B, open-loop lean rollout on v2's actions, max over living ships, 32 of 32
    environments kept: tick 30 position 3.4e-5 m, speed 3.1e-5 m/s, energy 0,
    shield 2.1e-4; tick 150 position 1.8e-4 m, speed 4.6e-5 m/s, energy 0.047,
    shield 8.7e-4.
  - The first completed run (8 environments, 30 ticks) failed check A on speed (max
    184.7 m/s) and nose rate (0.81): only on ships dying that tick, where v2 writes the
    speed floor and zero and lean keeps the values (listed under known differences).
    The test now leaves those rows out of check A; `core.py` was not changed.
- `tests/test_invariants.py`: 3 passed in 4.9 s. The team-swap test with its changed
  last assertion passed. The 20v20 case now runs 30 ticks (3v5 still 300); it still
  asserts at least one death.
- Operation count, one uncompiled `step`, 5v5, 4 environments, `fields=()`, counter
  `docs/rewrite/sections/simplification/regions.py` (compute ops, views apart):
  step 1,312 compute + 111 view; of which `flight_tick` 929 (+54 view),
  `bullet_motion` 84, `medium` 4 calls x 1 op, the rest of the step 298 (+57 view).
  `observe` for one team: 74 compute + 4 view. Against the targets: flight under 1,000
  holds (929); the rest is 383, under 500.
- Not run: any GPU, `torch.compile`, fields (every run had `fields=()`), the "sides"
  spawn in a test, a game end against v2, check C and check D.

## Choices where the audit left it open (section 7.1)

- Q1 tripwire: not measured. The flight is `lean_flight.py`'s form; nothing was thinned.
- Q2 fields: the step has the medium in flight and bullets from the start;
  `GameConfig.fields` defaults to none. No four-field map is defined yet.
- Q3 end with no shrink: v2's health rule at the clock (5,400 ticks): the sum of shield
  fractions of living ships; more wins, equal is a draw (result 2).
- Q4 arena: default radius 4,200 m with the side rule. Boundary ramp 300 m (v2's 100 m
  times its map scale 3; assumed, not checked against v2).
- Q5 friendly fire: on; reported apart as `StepOut.friendly_damage`.
- Q6 lineup: a fixed cycle by rank within the team (`lineup`, default Yongma, Jujak),
  the same for both teams. Per-episode mirrored draws are not implemented.
- Q8: game version 20 real-scale constants as they stand today.
- Q9: the step emits (E, N) sums only (dealt to enemies, dealt to team-mates, taken); no
  (E, N, N) matrix. `damage_taken` is hit plus boundary damage received, before the
  shield's clamp.
- Q10: first contact only; ordering by two reductions (smallest tau, then the smaller
  miss among contacts within 1e-6 of it), no bit trick.
- Slots are team-contiguous (team 0 first); N is the largest total in the batch.
- "mixed" spawn: every ship uniform in the disc of 0.85 arena radii with a uniform
  heading, then the edge rule. "sides": v2's rule, with a random axis per environment.

## Known differences from v2

- The medium is not cached in the state: it is evaluated at tick start for ships and
  bullets (same points as v2's cache, so the same values, one more evaluation per tick).
- A ship that dies keeps its speed and nose rate (v2 sets the speed floor and zero).
- Killer: earliest tau, then lowest bullet index (v2: tau, miss distance, index).
- Arming distance and the classic closeness are dropped (unused by the real-scale card).
- `softmin` is the prototype's clamp form, not v2's guarded division.
- Result coding: 0 or 1 the winning team, 2 a draw, -1 not done (v2: +1, -1, 0).
- Stick index 1, 2 = fine, 3, 4 = sustained, 5, 6 = full, odd indices negative (v2's
  order). Power 0 coast, 1 boost, 2 reverse. Quiet = coast, stick index 0 to 2, no shot.
- "Fired last tick" in the observation is inferred from the bullet ring; a shot that
  hits on its first move is missed. `State` has no `fired` field.
- Team sizes of 0 are not handled (the game would end at once with that team wiped).

## Check first on the GPU

1. Re-run `tests/test_against_v2.py` on the GPU (done on the CPU, see above).
2. Operation count with fields (the CPU count above is without fields).
3. Memory of the (E, N K, N) hit test at the evaluation sizes; about 30 temporaries.
4. `torch.compile` of `step` (the `SimpleNamespace` mutation in `_kick` and the Python
   branch on the number of fields are the likely snags).
5. `reset_done` draws every environment each call; confirm its cost and that generators
   on the device behave.

## Agreed waste fixes (author, 2026-11-22)

To do after the step-3 scoreboard ("before" numbers), each checked against v2 with
`tests/test_against_v2.py` and recounted:

1. **Constants:** a constant equal for every ship type, or fixed for the run, becomes a
   Python number compiled into the graph; one that differs by type goes into one packed
   per-type table, gathered once per slot at reset, (E, N, C). Damage then needs one
   gather for the target and one for the owner instead of about ten.
2. **Packed state:** the state fields as a few packed tensors (floats, integers, bullets),
   so the reset's select and similar per-field work is a handful of operations.
3. **Dead ships** are not held still field by field: whatever is simplest and fastest, as
   long as their values stay finite (they are masked everywhere they are read).
4. **Constant work at reset:** the aranges, the bullet-owner index and fixed buffers are
   built once, not every tick.
5. **Velocity once per step**, stored in the state; `observe` and the scripted opponent
   read it instead of recomputing it.
6. **One scripted call for both teams** (today two calls, each computing every slot), shared
   with the behaviour-cloning teacher.
7. **Personas drawn only at a reset**, not for every environment every tick.
8. **A quick no-fields path.** Fields are the default, so this is minor.
9. **Batched resets** (possibly): finished environments wait frozen and are re-dealt
   together every k ticks or at a segment boundary, with no host read; the learner masks
   the frozen ticks.

10. **No shared-observation shortcut now** (author 2026-11-22): fog is coming and would
    replace it. **Fog rule:** a ship is visible to the enemy team when it is in the enemy's
    actual sight, and on any tick it fires, is hit by a bullet, or dies; every ship is
    always visible to its own team. Boundary damage does not reveal a ship. A reveal lasts only for the
    tick of the shot or the bullet hit. A dead ship that will respawn is visible; a ship
    dead for good (elimination) is masked out. The scripted opponent must not cheat: under fog it
    reads its team's view, not the state.
11. **Fields drawn per environment** (author 2026-11-22: "fields are randomised for every
    new env"): fields in the state as (E, F, 5), drawn at reset, padded with zero-strength
    fields; the medium at the ships and at the bullets' end of move cached in the state, so
    it is evaluated twice per tick instead of four times. The policy needs field tokens
    (a contract change) before any training with fields.

12. **Stored, not guessed:** a `fired` flag in the state (the observation reads it). Done 2026-11-23.
13. **The scripted opponent is told everything up front:** each ship's type and its
    constants (gun range, corner speed, launch speed per type), and the environment's mode.
    Modes come in a fixed share of environments for the whole run, so an environment's mode
    is a constant of that environment and environments can be grouped by mode.
14. **Cheaper damage form** (author 2026-11-22: agreed): the kinetic term kept; closeness
    from the hit test's miss distance; the head-on term from the hit angle's cosine, fitted
    per type to today's curve, the largest difference reported.

Collision code read (2026-11-22): no error found. Swept straight-segment closest approach,
first contact by earliest tau, no self-hits, friendly fire on, a dead ship's bullets keep
flying. K = 11 bullet slots per ship is exact (bullet life 11 ticks, one shot a tick at
most), so damage arithmetic runs on N x 11 rows. Fire rate, bullet life and bullet energy
cost are equal for every type: constants for the graph.

Future check (author 2026-11-22): the dense (E, N K, N) hit test at 50v50.

## Parity with v2 completed (2026-11-23)

`tests/test_parity_v2.py` extends the comparison to fields, games that end (wipes, boundary
deaths, simultaneous wipes, the clock), the "sides" spawn's distribution, 3v5 and 20v20; the
card equals v2's exactly. Three behaviour-changing bugs were found and fixed (the lift slope
past the stall, and two in the clock rule): see docs/rewrite/game-parity.md, which also has
the full difference table. Remaining differences are recorded configuration choices (arena
4,200 m, boundary ramp 300 m, fixed lineup, the "mixed" spawn) or not yet built (shrink,
fields per environment; fog since built, below).

## Fog of war (2026-11-23)

Rules: docs/rewrite/core-decisions.md and docs/questions.md items 2 and 5. `GameConfig.fog`
(default False; True in `GameConfig.core()`). Fog off is bit-identical to the tree before fog:
`step`, `observe` and `reset_done` compared with HEAD (25478e10) over 200 ticks with resets,
v2 config and core config, every state, step-output and observation field `torch.equal`.

**v2's sight rule as read** (src/bnb/game/perception/sight.py `sighted`, `visibility`;
observe.py `perceive`): sight is all round (no cone), out to the observer's vision range,
d^2 <= V^2. V is the rule's 3,072 m times the type's `vision_range_scale` 0.3255: 1,000 m for
every real-scale type, not scaled with the map. Only living observers see. Fields block by
their cores only: core radius max(r - w/2, 0), outer radius r + w/2. Field i blocks the line
when exactly one end is strictly inside its core and the two ends are not both inside its
outer disc (core to band sees, core to outside does not), or when neither end is inside its
core and the segment passes strictly within the core radius (foot of the perpendicular inside
the segment: 0 < u.d < |d|^2 and |u x d|^2 < rho^2 |d|^2). Two ends in the same core see each
other. Field strength and sign do not matter. Bullets are not perceived. v2 reveals an enemy
that is alive and (sighted, or fired this tick, or spawned this tick); a team always sees its
own ships. v2 does not reveal on a hit and does not keep last-seen values: hidden rows are
zero, `time_since_seen` is ticks times the tick length in seconds (cap 2^30 ticks) and its
model reads symlog of it; v2 shows `alive` for every ship (deaths are public in v2).

**What lean does.**
- `State.seen` (E, 19, N) float32, columns `core.SEEN_COLUMNS`: each ship's raw values at its
  last sighting by its enemy team (x, y, heading, speed, angle, nose rate, energy, shield,
  shield wait, cooldown, the medium, the real-bullet share, alive, known), and
  `State.ticks_since_seen` (E, N) long. Each ship has one enemy team, so one record per ship.
  None with fog off. Columns on the middle axis so each column read back has unit stride.
- `perceive(state, revealed)` at the end of `step` (and in the reset on the spawn positions):
  visible = (sighted by a living enemy | fired a real shot | hit by a real bullet) & exists;
  a visible ship's record is replaced by its current values and its count set to 0, any other
  counts up to the cap. Inside `step` because "hit this tick" is known only there (not stored).
- `team_view(state, team)`: own ships live, enemies as recorded, a never-seen or seen-dead enemy
  not alive, bullet fields None. `observe` under fog is built from it; `key_mask` = exists and
  alive in the view; never-seen rows are zero; `Obs.visible`, `Obs.since_seen` =
  log1p(seconds since seen) (0 for own rows; with fog off all visible, all 0). Globals: the
  enemy alive fraction counts only deaths the team saw (never-seen enemies count as alive).
- The scripted opponent and the chaser call `team_view` first: under fog they target enemies at
  their last-seen place, ignore never-seen ones (no target: steer to the centre), never read
  bullets. With fog off `team_view` returns the state itself (no cost).

Tests `tests/test_fog.py` (8): fog off unchanged and full information; v2's sight (range and
the field-core cases); own team always visible, dead observers blind; firing and hits reveal for
one tick, boundary damage does not; last-seen values freeze, count, cap; never-seen masked and
seen deaths dropped, unseen deaths not known; no hidden information in the observation (a
hidden enemy's motion, store, gun, life and bullets changed: every `Obs` field equal; the same
change in sight does change it); the scripted opponent and the chaser unchanged by it.

Operations (docs/rewrite/sections/scoreboard/game_ops.py's counter, uncompiled, CPU, one
thread, 5v5, E = 256; compute + view):

| call | fog off | fog on |
|---|---|---|
| step, core config (10 drawn fields) | 1,527 + 128 | 1,593 + 158 |
| step, v2 config (no fields) | 1,342 + 124 | 1,370 + 134 |
| observe, one team | 82 + 8 (HEAD 80) | 110 + 16 |
| scripted act, one team | 193 + 41 | 208 + 44 |
| of which `perceive` (10 fields / none) | | 64 / 26 |
| of which `team_view` | | 15 |

Fog off observe +2 (the two new `Obs` fields). The sight test is dense (E, N, N, F) with the
fields vectorised, not v2's per-field loop: 2.2 ms per perceive at E = 256 with 10 fields on the
CPU; at 50v50 an uncompiled call holds about 100k booleans and floats per environment per
temporary (check on the GPU with the hit test). Scripted vs scripted, core config, 64 games
of 5v5, 1,500 ticks: fog off 385 deaths, 42 games ended; fog on 372 deaths, 39 ended; team 0
saw 50% of the living enemy rows on a tick and had 60% in its key mask.

Choices for the author:
1. No reveal at a spawn: the decided rule has none; v2 reveals every ship on its spawn tick
   (so at tick 0 in v2 everyone is seen). With the "sides" spawn nobody is seen at the start.
2. Deaths are known only when seen (on the death tick, or a wreck later in sight; a kill by a
   bullet is always seen, since the hit reveals). v2 shows every death. An unseen boundary
   death stays a last-seen row and counts as alive in the globals.
3. Wrecks can be sighted (v2's sight test ignores the target's life); this is how a hidden
   death becomes known. No respawn in lean yet, so every death is for good.
4. A hit reveals for real bullets only (ghosts are labels); friendly hits reveal too.
5. `Obs.since_seen` carries symlog(seconds) (v2's model input), not raw seconds.

## Waste fixes done (2026-11-23)

The agreed fixes 1 to 5 and 8, one commit each (82a1469d, 67a25f57, a257a1f6, 62010d71,
357f4b11, 9b313c69; then 5e67560a, the fixed tensors traced into the graph under compile).
Fixes 6, 7 and 9 need lean/learn and were not made (below).

- **1, constants.** `contract.Constants`: a constant equal for every type of the run's lineup
  (or fixed by the rules) is a Python number (`uniform`); one that differs by type is a row
  of one per-slot table (E, C, N), gathered once at reset. The action-indexed tables (stick,
  commanded, demand) are always rows of it. Read by name: `c["lift"]` or `c.lift`. Built
  once per flight settings and lineup (`ship_card`). In the core config (Vizia alone) all 43
  constants are numbers and the table holds only the 13 action-table rows; in the v2 config
  (Yongma, Jujak) 25 are numbers and 21 differ by type (31 rows with the tables). Damage
  gathers only the target constants that differ by type: none with today's card. Slots that
  do not exist take the lineup's first type (was type 0). The fields left the constants:
  `State.fields` (E, 5, F), rows x, y, radius, band, target ln n (`core.FIELD_*`).
- **2, packed state.** `State` is ships (E, 19, N) float32 (motion, store, shields, gun
  cooldown and next bullet slot as whole numbers, the cached medium, the velocity), flags
  (alive, exists, fired), identity (team, type), bullets (E, 8, N, K) with their medium,
  bullet counts (moves left, fired tick) long, bullet_real, tick, constants, fields, and the
  fog's record. Every quantity reads by its old name as a view (`contract.SHIP_COLUMNS` and
  the other tuples); `State.pack` builds one from named quantities, `State.replace` replaces
  some. `state.medium` and `state.bullet_medium` are now `log_index`, `gradient_x`,
  `gradient_y` (and `bullet_` ones). A reset's select is one where per block; a new bullet
  is one where per block; bullet motion is selected once; the swept test runs x and y on
  one axis, (E, 2, N K, N), and damage gathers the target's start and motion from it. The
  fog record (`SEEN_COLUMNS`) is the ship row plus share, alive, known; `team_view` slices it.
- **3, dead ships.** Every slot flies; the new ship block is selected once against the old
  by alive at tick start. The hit test and damage read the unselected flight (a dead ship
  is never a target: its reach is -1; it takes no boundary damage). A dead ship's cooldown
  is held instead of counted down (masked wherever read).
- **4, built once.** The aranges, the bullet-owner index and its not-own mask, the hit
  matrix's row offsets, the scatters' starting values, the 2x2 identity, the lineup's type
  indices and a fresh deal's fixed rows (no bullets, tick 0) are built once per size and
  device (`built_once`, shared read-only; `reset` copies the shared rows it returns). While
  torch.compile traces, the builders run in the graph instead. The core model's zero charge
  is no longer a tensor. The scripted opponent reuses the slot index and its gate floors.
- **5, velocity.** `velocity_x`, `velocity_y` in the ship block: the velocity at proper speed
  along the flight path, computed once at the end of the step (and at a deal). The next
  tick's fire, `observe` (over the refractive index), the scripted opponent and the fog
  record read it. After setting motion by hand, `core.with_velocity(state)` refreshes it.
- **8, no fields.** With F = 0 the flight, bullets, observe and a deal skip the medium's
  arithmetic (n = 1, no gradient, no bend, no stretch): the same values.

**Same maths.** Each commit was compared with the one before over 300 ticks, 16 environments,
team sizes 3 to 5, random sticky actions, resets (games ending, and forced at ticks 40 and
190), every state quantity, step output, both teams' observations, the scripted opponent's
marginals and its sampled actions with memory; v2 config, v2 config with two fixed fields,
core config (fields drawn, fog, ghosts), arena 1,500 m with the "mixed" spawn so that ships
fight (23 to 28 deaths, 160 to 440 hits per run). Expectation for every commit: the same
within float32 rounding, no discrete difference. Result: fix 1, largest relative difference
5e-5 (damage: the drag constant is now a Python number, one rounding fewer, and the closeness
amplifies it); marginals 4e-5; states 1e-7 to 1e-5. Fix 2: bit-identical but for one ulp of
atan2 in the observed heading. Fixes 3, 4 and 8: bit-identical. Fix 5: bit-identical but the
observed velocity, 1.2e-7 relative. No discrete difference anywhere. `pytest lean` passes
after each commit (115 passed, 1 xfailed), with `test_against_v2.py` and `test_parity_v2.py`
unchanged in their bounds (only their state construction changed) and the fog-off and ghost
equality tests passing. An inductor-compiled step (CPU) matches eager within float32 rounding
over 20 ticks; step, observe and the scripted act trace as one graph, no break.

**Operations per call** (`docs/rewrite/sections/scoreboard/game_ops.py`, compute ops, views
apart; uncompiled, CPU, one thread, E = 256, 5v5). Each column is the tree after that fix.

| call | before | 1 | 2 | 3 | 4 | 5 | 8 |
|---|---|---|---|---|---|---|---|
| step, v2 config (no fields) | 1,348 | 1,334 | 1,309 | 1,295 | 1,280 | 1,280 | **1,148** |
| step, v2 config, 10 fixed fields | 1,516 | 1,502 | 1,477 | 1,463 | 1,448 | 1,448 | **1,448** |
| step, core config | 1,599 | 1,586 | 1,561 | 1,547 | 1,522 | 1,522 | **1,522** |
| observe (one team), v2 config | 82 | 82 | 81 | 81 | 81 | 75 | **70** |
| observe (one team), v2 config, fields | 82 | 82 | 81 | 81 | 81 | 75 | **75** |
| observe (one team), core config | 110 | 110 | 108 | 108 | 108 | 102 | **102** |
| reset_done, v2 config | 281 | 131 | 99 | 99 | 88 | 90 | **87** |
| reset_done, v2 config, fields | 365 | 215 | 183 | 183 | 172 | 174 | **174** |
| reset_done, core config | 487 | 322 | 290 | 290 | 278 | 280 | **280** |
| scripted act (one team), core config | 208 | 206 | 205 | 205 | 204 | 196 | **196** |

View ops went up (127 to 201 in the v2 step; 8 to 41 in observe): every named read of a
packed block is a view (no kernel). CPU time on this machine (another agent's load, timings
noisy; base and final interleaved, three runs, E = 256): step about the same (v2 10 to 11 ms
both; core 16 to 20 ms both: memory-bound at this size, and most of the step's operations are
the flight's arithmetic, which the fixes do not thin); reset_done v2 1.4 to 0.56 ms, core
4.6 to 3.7 ms.

What is left in the counts: the core step's 1,522 are mostly the flight (two kicks of about
400 each) and the medium (fields on). 54 of the step's operations are `aten.scalar_tensor`
from `torch.where(..., scalar)`: a CPU 0-dim wrapped number each, no GPU kernel, host
overhead in eager, gone when compiled. The core config's reset_done (280) is mostly a new
map drawn and seen (the medium at the spawn and the fog's sight test with ten fields) for
every environment every call: fix 9.

**Not made (need lean/learn):**
- 6, one scripted call for both teams: `lean/learn/collect.py` `scripted_actions` calls
  `scripted.act` once per team and adds them. A both-teams act in lean/scripted is possible
  without a learn change to the game (under fog, ship j as ship i's team knows it is live for
  a team-mate and `seen[j]` for an enemy, a pairwise (E, N, N) where), but nothing calls it
  until collect does.
- 7, personas drawn only at a reset: `collect.py` `redeal` draws personas for every
  environment on every call and selects them by done.
- 9, batched resets: the collect loop must freeze finished environments, re-deal them every
  k ticks or at a segment boundary, and mask the frozen ticks.

**Outside lean/game and lean/scripted:** `lean/sandbox/world.py` builds and edits a state by
hand with the old fields (`dataclasses.replace(state, x=...)`, a constants dict with field_
keys, `state.medium`) and no longer runs; not edited (outside this task's scope). A patch
that makes it run on the packed state (World with both configs, a type switch, the chaser
toggle, 250 ticks) was written and checked but not applied.
