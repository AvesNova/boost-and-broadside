# Game audit and port map (v2 game side, for the lean core)

Date: 2026-11-22. Read-only audit of `/home/user/boost-and-broadside` (v2: the working
tree, `src/bnb/`; v1: `origin/main`, read with `git show`). Nothing in the repository was
changed. Scratch scripts and their outputs are in
`/tmp/claude-0/-home-user-boost-and-broadside/bf4c0758-f091-53dd-a269-e94d54b68e77/scratchpad/next/game-audit-scratch/`
(listed in the appendix).
The scripts were later copied to `docs/rewrite/sections/game-audit/` and removed on
2026-11-28; they are at commit ce27fe64 (`git show ce27fe64:docs/rewrite/sections/game-audit/<name>`).

Conventions in this file:

- **Read** means taken from the code or a note, with the file and function. **Measured**
  means I ran v2 on the CPU in the scratch virtualenv (Python 3.14, torch 2.15.0+cu132)
  and counted or timed it. **Inferred** means my reasoning; it is labelled.
- Operation counts use the audit's own counter
  (`docs/rewrite/sections/simplification/regions.py`): operations PyTorch dispatches in
  eager mode, views excluded. Counts do not depend on batch size or ship count.
- Paths are under `src/bnb/` unless given in full. E is environments, N is ship slots per
  environment, S is ships per team, K is bullet slots per ship, M is fields.
- "Real scale" is the only ship set in scope (`game/ships/real-scale/`): Yongma (type 0),
  Jujak (type 1), Vizia (type 2, the hull both derive from; not in the training lineup).

## 0. The findings that change the plan

1. **v2's flight alone is larger than the 1,300 tripwire.** Measured in a single-game
   elimination batch: `flight_tick` 1,470 operations, plus 70 for the ship-card lookups
   and 35 for the controls, 1,575 in all. A restructured port that computes the same
   numbers (prototype in the scratch directory) is **1,026 operations with the field
   evaluation, 930 without**. The whole minimal game step comes to about 1,500 with
   fields and about 1,250 without. So "flight lifted nearly verbatim" and "1,300
   operations per tick, uncompiled" do not both hold. The flight is one pure elementwise
   function over (E, N), the ideal case for a compiler; the tripwire needs restating
   (section 4.4, question Q1).
2. **Single-game elimination already costs much less than the audit's mixed batch.**
   Measured: move 1,861 + resolve 715 = 2,576 per tick (mixed launch configuration:
   1,861 + 1,866). Perception is 475 (mixed: 828). The mixed-batch generality is about
   1,150 operations of resolve and 350 of perception.
3. **v2 has no switch for "elimination without shrink",** but two rule values give it up
   to the last tick of the clock (section 5.2, verified). The core needs its own
   end-of-clock rule; v2's is the health tie-break (section 1.3, question Q3).
4. **A restructured float32 port agrees with v2 to rounding in 99.99% of ship-ticks.**
   The exceptions (31 of 614,400 samples) are all post-stall (angle of attack beyond the
   stall angle), where the nose oscillator amplifies rounding about 100 times. The
   tolerance design in section 5 is built on these measurements.
5. **Imugi's 2,124 operations per call break down as:** type doctrines and manoeuvres
   about 780, the Frontline layer 567 (run on every row, selected away in elimination),
   keyed random draws 396, and the rest (targeting, lead, stick, power, gun, memory) about
   380 of real work. Without doctrines it is 1,342.
6. **The flight feel is the card plus 21 shared constants.** Everything else in
   `game/handling/` (about 2,500 lines: solver, handles, estimates, charts) is design
   tooling that produced the card and is not needed to step the game.

## 1. Inventory of v2's game

### 1.1 What the rollout calls each tick

Read from `runtime/rollout.py` (`_act`, `_advance`, `_resolve`, `_perceive`):

| Stage | Calls into the game | Other per-tick work in the stage |
|---|---|---|
| `_act` | none | controllers (model, Imugi, lanes); keyed draws of the spread and tick uniforms (`streams.keyed_normal`, `keyed_uniform`, rollout.py:492-494) |
| `_advance` | `api.move` (fire, flight, bullets) | `api.launch_capture` (shot re-simulation record) |
| `_resolve` | `api.resolve` (hits, damage, mode, clocks, respawn, episode end) | `learn/rewards.reward_step_events` (159), `runtime/analytics.resolve_tick` (168) |
| `_perceive` | `api.perceive` (fog, observation, truth) | `analytics.perceive_tick` (118) |

The standalone entry points are `game/api.py`: `reset`, `move`, `resolve`, `step`
(= `move` then `resolve`), `perceive`. Randomness is an explicit input: `spread`
(E, N, 2) standard normals and `uniforms` (E, U) with U = 224 for real-scale elimination
5v5 (`Rules.tick_uniform_count`).

### 1.2 Mechanisms, constants, cost, and whether the skeleton needs them

Operation counts: **measured** on `real_scale_elimination` 5v5, 10 random fields, one
pool, eager CPU (`count_v2.py`), inclusive of callees unless marked "excl". "Mixed" is
the figure from `simplification.md` for the launch configuration (three modes, two pools
each). "Needs it" refers to step 1 of the skeleton: elimination, no shrink, one fixed
map, any team size.

**Tick and shared numbers**

| Mechanism | What it does, with constants | Where | Ops per tick | Needs it? |
|---|---|---|---|---|
| Tick | 30 Hz, physics tick = decision tick, `tick_length = 1/30` s. Phases: 1 fire, 2 ships fly, 3 bullets fly, 4 hits, 5 damage/shields/deaths, mode phase, 6 clocks, 7 respawn, 8 episode end | `game/rules.py:Rules.tick_length`; `game/physics/step.py:move`, `resolve` | step 2,576; move 1,861; resolve 715 | Yes |
| Actions | Power 3 (0 coast, 1 boost, 2 reverse), stick 7 (0 straight; 1/2 fine right/left; 3/4 sustained; 5/6 max; positive turns left), shoot 2. 42 joint actions | `game/rules.py:STICK_LEVELS`, `POWER_MODES`; `contracts/action.py` | controls 35 | Yes |
| Move/resolve split and `Sweep` | The tick is cut in two so the runtime can compile two stages; `Sweep` carries start positions and displacements between them | `game/state.py:Sweep`; `step.py:move` | about 18 glue | No (one function; keep the swept quantities as locals) |

**Flight** (detail in section 2)

| Mechanism | What it does, with constants | Where | Ops per tick | Needs it? |
|---|---|---|---|---|
| Flight tick | Half kick, drift along the exact nose arc, one field evaluation, half kick | `game/physics/flight.py:flight_tick` | 1,470 (two kicks 1,278; drift about 96; medium 96) | Yes: the thing to lift |
| Kick | Speed, angle of attack, nose rate, energy over h/2 with the pose held; nose oscillator solved exactly; two-stage speed and energy | `flight.py:kick` | 639 each | Yes |
| Setting (budgets) | Lift budget, turn priority, vertical demand, store fade, thrust, direct charge; 2 per kick | `flight.py:setting`, `_vertical_demand` | 576 for 4 calls (softmin: 288 excl for 24 calls, 12 each) | Yes |
| Rates | Drag (zero-lift with drag rise, induced with separation), thrust, exchange; 3 per kick | `flight.py:rates`, `zero_lift_drag`, `separation`, `bleed` | 246 for 6 calls | Yes |
| 2x2 exponential | Horner series of exp(A dt/2), squared, written out per matrix entry | `flight.py:exponential_2x2` | 170 for 2 calls | Yes (as a batched 2x2 product: 16) |
| Ship-card lookup | Each card value per ship as a chain of `where` over the types, rebuilt every tick | `game/ship_types.py:per_ship`; `flight.py:airframe_of` | 164 for 41 calls in the tick (airframe 70) | No: gather once at reset |
| Stick levels per type | Fine and sustained lift fractions from the card (Yongma 0.196/0.655, Jujak 0.222/0.741), max 1 | `flight.py:controls`; card `fine_stick`, `sustained_stick` | in controls | Yes (as a per-ship table) |

**Fields ("refractive medium")**

| Mechanism | What it does, with constants | Where | Ops per tick | Needs it? |
|---|---|---|---|---|
| Medium | ln n and its gradient at points: circular fields, level in {-2, -1, 1, 2}, ln n = level x ln sqrt(2) (n from 1/2 to 2), smootherstep across a band (real scale 160 m), union composition | `game/physics/medium.py:medium` | 96 per call; 3 calls (ships, bullets, spawn) = 288; mixed "about 250" | Only if the fixed map has fields (Q2) |
| Effect on ships and bullets | Proper speed kept, world speed divided by n, path bends toward higher n | `flight.py:flight_tick`, `kick`; `bullets.py:_half_bend` | inside flight and bullets (about 70 + 60) | With fields |
| Random map | M fields point-symmetric about the centre; centres uniform in 0.85 R; radius lognormal, median 400 m, log sigma 0.6, clamped 120 to 1,200 m at map scale 1 (x3 on the 7,800 m map); level sign and strength each 50/50 | `game/physics/spawning.py:new_map`, `field_radius`, `field_level`; `ships/real-scale/rules.toml` | 65 per tick (built for every environment every tick, selected by `done`) | No (fixed map: constants) |
| Fixed layout | Half the fields and team 0's spawn points written down, images by 180 degrees, optional random turn per match (`rotated`) | `contracts/config.py:MapLayout`; `game/modes/phase.py:layout_fields`; `game/modes/spawns.py:layout_placement`; preset `clover` | resolve 758 with Clover against 715 | The idea yes; the code no |
| Absent fields | `Fields.present`; an absent field becomes a fake field with band 1 m and radius -1 m so its profile is exactly 0 | `medium.py:19-22`; `spawning.py:draw_active_fields` | 2 `where` per medium call | No |

**Bullets**

| Mechanism | What it does, with constants | Where | Ops per tick | Needs it? |
|---|---|---|---|---|
| Fire | Fires if alive, shoot held, cooldown 0 and energy >= cost. Cost 575.06 J (865 shots per full store, 346 from the spawn store). Fire rate 30 Hz = 1 tick between shots. Bullet starts at the ship with the ship's velocity + muzzle 1,712 m/s along the nose + spread 12 x 0.257 = 3.08 m/s per axis. Ring buffer of K = 11 slots | `game/physics/bullets.py:fire`, `launch` | 74 | Yes |
| Bullet motion | Half drag, half bend, move, field evaluation, half bend, half drag. Drag dw/dt = -k w^2 / n with k = 5e-4 1/m | `bullets.py:bullet_motion`, `move_bullets` | 194 (medium 96) | Yes (about 85 with fields, about 20 without) |
| Bullet life | 0.3667 s = 11 moves, the first in the firing tick. A bullet fired at cruise flies 637 m (measured, `refdump.py`) | card `bullet_life`; `ship_types.py:bullet_moves` | in clocks | Yes |
| Launch capture | Stores each ship's launch state per tick so the learner can re-simulate shots | `bullets.py:Launch`, `launch_state`; `api.py:launch_capture`, `bullet_paths` | rollout side | No (belief/shot-head chain) |

**Hits and damage**

| Mechanism | What it does, with constants | Where | Ops per tick | Needs it? |
|---|---|---|---|---|
| Swept hit test | Relative swept test of every bullet against every ship but its shooter, as (E, N K, N) pairs; contact if the closest approach is inside the hit radius 12 m. Friendly fire on | `game/physics/combat.py:contacts` | 35 (about 30 pair-sized operations) | Yes |
| First contact | A bullet hits only its earliest contact (smallest tau, ties to the closer pass, then the lowest ship index), by an int64 key built from float32 bits | `combat.py:contact_order`, `first_contact` | in hits | The rule yes; the bit trick no |
| Damage | D x (x_e + (1 - x_e)(w / w_ref)^2) x closeness x aspect. D = 22.98 (Yongma), 19.19 (Jujak); explosive share 0.5 / 0; w_ref 1,761.8 / 1,745.6 m/s. Closeness: 1 inside the 5 m core, smoothstep to 0 at 12 m. Aspect: 1 - (1 - f0) exp(-ln2 (theta / theta_h)^p), f0 = 0 (no damage dead ahead), theta_h = 45 degrees, p = 3 | `combat.py:hits`, `kinetic`, `closeness` | 225 | Yes |
| Arming distance | Damage ramp over the first X metres of flight; X = 0 on every real-scale card | `combat.py:hits` (arming lines) | about 10 | No |
| Aspect statistics | Enemy damage by aspect class and the killing hit's class, for analytics | `combat.py:hits` (last block); `contracts/events.py:ASPECTS` | about 20 | No |
| Killer | The owner of the earliest hit on a ship that dies | `combat.py:hits` (`earliest`) | about 15 | Yes if rewards credit kills |
| Shields | Shield is the only health. Damage beyond the shield breaks it (overkill ignored); any damage to a shield already at 0 at tick start kills. Max 141.95 (Yongma), 79.68 (Jujak). Recharge 24 per s after 3.0 s of unbroken quiet flight (coast, stick straight or fine, no shot); any other input or a hit restarts the wait | `combat.py:apply_damage`, `quiet`; `rules.py:QUIET_STICKS`, `shield_recharge_delay` 6.0 x 0.5, `shield_recharge_rate` 12 x 2 | 41 | Yes |
| Boundary | Disc of radius R (7,800 m), no wall. Beyond it: 40 per s x smootherstep(depth / 300 m) (ramp 100 m x map scale 3) | `combat.py:boundary_damage`; rules.toml `boundary_max_damage` | 20 + `boundary_fraction` 19 | Yes, with a constant radius |
| Spawn protection | Frontline only | `modes/phase.py:protected` | 0 in elimination | No |

**Deaths, respawn, episode end**

| Mechanism | What it does, with constants | Where | Ops per tick | Needs it? |
|---|---|---|---|---|
| Death | `alive` false; speed set to the floor and nose rate to 0; the ship keeps its other values | `step.py:resolve` phase 7 | few | Yes (mask only) |
| Respawn | `respawn_wait` counts down; `respawn_open` by team (Frontline waves). Elimination: the wait is the whole episode (5,400 ticks), so nothing respawns | `step.py:resolve`; `rules.py:_pool_rules` | about 20 | No (arrives with the other modes) |
| Episode end | Done when the clock ends or the mode is won. Finished environments get a new map, new mode state, cleared bullets and all ships spawned. All of this is built for every environment every tick and selected with `where(done, ...)` | `step.py:resolve` phase 8; `_new_matches`, `_placement`; `spawning.py:spawn` | 243 (new map 65, placement 49, spawn 129 with its medium call) | Reset yes; rebuilding every tick no |
| Spawn state | Spawn speed 145.29 m/s proper (x n), angle of attack 0, store 199,025 J (40% of 497,563), full shield, cooldown 0 | `spawning.py:spawn`; rules.toml `spawn_speed` | in spawn | Yes |
| Benched slots | Unequal teams at one static size: a team's last S - active slots are held as dead ships waiting to respawn | `step.py:_bench`; `state.py:benched_of` | +13 with `active` | Replaced by an `exists` mask |

**Modes and their rules**

| Mechanism | What it does, with constants | Where | Ops per tick | Needs it? |
|---|---|---|---|---|
| Elimination | Section 1.3 | `modes/phase.py:mode_phase`, `clock_result`; `modes/rules.py` | small (score, compare) | Yes |
| Elimination spawn | Each team in a disc of 0.15 R (1,170 m) around a centre 0.8 R (6,240 m) out on a random spawn axis; nose toward the enemy area within +-15 degrees; turned the least that leaves room to turn from the edge (`spawn_turn_distance` 436.5 m) | `modes/spawns.py:side_placement`, `_edge_safe` | 49 | Yes, or fixed spawn points |
| Shrinking boundary | Holds at R for 36 s, then linear in radius past 0 to minus a ramp width, 8 s before the clock | `game/layout.py:boundary_fraction`, `environment`, `zero_share` | 19 + part of 46 | No (later addition) |
| Team deathmatch | Respawn 10 s, 5 kills per enemy ship wins, kill-lead at the clock, best-of-8 spawn candidates, 4,200 m arena | `modes/phase.py`; `modes/spawns.py:deathmatch_placement` | not run in a single-game elimination batch | No |
| Frontline | 5 zones on a ring, capture meters, ledgers (E, Z, N), capture lock, respawn waves, spawn protection | `modes/frontline.py` (350 lines); `modes/phase.py` | 0 here; mixed "about 600" for all three modes' machinery | No |
| Per-environment rules | 16 tensors (arena radius, clocks, shrink schedule, map-size sampling, field-count sampling) rebuilt from Python numbers at every call | `game/layout.py:environment`, `Environment` | 92 for 2 calls (mixed: 844) | No: constants, later set at reset |
| Per-mode slicing | Each mode's code runs on its contiguous slice, results concatenated | `layout.py:per_mode`, `slices` | 0 with one pool | Not in step 1 |
| Map-size sampling | Arena scale lognormal(0, 0.35) in [0.5, 2] per episode, with floors | `layout.py:draw_arena_scale`, `sized` | off by default | No (stage 2) |

**Perception, rewards hooks, analytics**

| Mechanism | What it does, with constants | Where | Ops per tick | Needs it? |
|---|---|---|---|---|
| Fog | A team sees an enemy within an observer's vision range (1,000 m on the cards: 3,072 x 0.3255) with a clear line; field cores block; a ship that fired or spawned this tick is seen | `game/perception/sight.py:visibility`, `sighted`, `membership` | 271 (computed in unfogged environments too) | No (addition 1) |
| Observation | Per team view: ship rows (29 columns), field rows (6), zone rows (5 x 11), globals (15); own ships first; hidden rows zeroed; accelerations as differences against the last tick | `game/perception/observe.py:perceive`; `contracts/obs.py` | 475 with sight (mixed 828) | A new, smaller contract (step 2) |
| Privileged truth | Every ship row as if seen, for labels | `contracts/obs.py:Privileged` | in perceive | The state itself serves |
| Events | Damage (E, N, N), boundary damage, died, fired, spawned, killer, done, shield lost and restored, objective amounts, result, win/loss weights, aspect statistics | `contracts/events.py:Events` | assembled in resolve | A subset: damage, died, killer, fired, done, result |
| Reward hook | Reads `Events`; keeps a ledger (E, N, N) of outstanding damage so that a recharged shield un-credits its attackers | `learn/rewards.py:reward_step_events`, `Ledger`, `attribute_recharge` | 159 | Learner's side; note the ledger exists because shields recharge |
| Analytics in the tick | Game sums, aspect shares, outcome calibration | `runtime/analytics.py:resolve_tick`, `perceive_tick` | 168 + 118 | No (compute from stored events) |

**Design tooling and live tuning (not in the tick)**

| Mechanism | What it does | Where | Needs it? |
|---|---|---|---|
| Handles, solver, estimate, charts, similarity, weapons | Solve a card from designer handles by flying the game; show derived quantities | `game/handling/` (about 2,500 lines), `game/ship_design.py` | No. Keep the three cards and the rules as data |
| Characteristics | Per-type quantities Imugi reads (cruise, top and corner speed, turn rates, gun range), flown at run time: 15.2 s on the CPU (measured), cached on disk | `game/handling/characteristics.py` | Store the numbers with the card (values in section 6.2) |
| Live tuning | Rules as 0-d tensors with every branch computed; `retune`; bullet slots sized for a 2 s life (60 slots in the play shell) | `game/tuning.py` (541 lines); `api.py:retune`; `rules.py:LONGEST_BULLET_LIFE` | No |
| Version and fingerprint | `VERSION = 20`, a hash of a short seeded match per preset | `game/__init__.py` | A version number only |

### 1.3 Exactly what "elimination" is in v2

Read from `game/modes/rules.py`, `game/modes/phase.py`, `game/rules.py:_pool_rules`,
`game/layout.py`, `game/physics/step.py:resolve`, and `docs/explanation/game/modes.md`.

- **Preset `real_scale_elimination`:** 5v5, 10 random fields, arena radius 7,800 m, clock
  180 s (5,400 ticks), real-scale ship set, default lineup alternating Yongma and Jujak
  (`Rules.team_types`: slot i has type i mod 2). The launch run used the lineup Yongma x2
  + Jujak x3 and drew elimination as 31.25% of the batch
  (`runtime/train.py:TRAINING_MIXES["real_scale_all_modes"]`).
- **Win condition:** every death scores a point for the other team, whatever its cause
  (bullets from either team, the boundary). A team wins when its score reaches the
  enemy's active ship count (`kills_per_ship` forced to 1; `mode_phase`,
  `deathmatch_targets`). So a team with no ship left loses.
- **Ties:** if both teams reach their target on the same tick the result is 0, a draw
  (`mode_phase`: `reached[:, 0] - reached[:, 1]`). Verified with a hand-built state
  (two shieldless ships each hit on the same tick: both died, `done` true, result 0).
- **Time limit:** the episode ends at tick 5,400. The result is then the sign of the
  difference in health left, where a living ship counts its shield over its type's full
  shield and a dead ship counts 0 (`clock_result`). Equal health is a draw. With the
  shrink on this decides only ships that die together.
- **The shrink:** the boundary holds at R for `shrink_hold_seconds` (36 s at real scale),
  then its radius falls linearly (`shrink_law` 0) past 0 to minus one ramp width
  (-300 m), which it reaches `shrink_end_lead_seconds` (8 s) before the clock, at 172 s.
  Boundary damage is measured from the boundary as it stands. From the zero crossing a
  ship at the centre is hurt every tick, so shields stop recharging, and from the close
  every ship takes 40 per s; the largest shield (141.95) lasts 3.5 s. So every ship dies
  before the clock.
- **Respawn:** none. It is implemented as a respawn wait equal to the episode length, so
  a dead ship is still waiting when the episode ends (`_pool_rules`).
- **Spawns:** by team side (section 1.2). The two areas are 12,480 m apart centre to
  centre.
- **Reset:** inside `resolve`, on the tick the match ends; the returned state is the new
  episode's.
- **Measured behaviour** with the shrink switched off, fog off, Imugi on both sides, 48
  games of 5v5 (`stats.py`): 43 of 48 ended by a wipe within 120 s; game length median
  45.0 s, mean 54.1 s, 10th to 90th percentile 29 to 104 s; first kill median 17.2 s
  (the fleets start 12.5 km apart); 635 shots, 6.85 deaths and 0.8 points of boundary
  damage per game; 0.40 damaging (shooter, target, tick) pairs per shot; **27% of all
  hit damage was friendly fire.**

### 1.4 How this differs from v1's game as trained

Read from `origin/main`: `profiles/rl.py`, `config/defaults.py`, `config/core.py`,
`env/env.py`, `env/physics.py`, `env/wrapper.py`, and the configuration line in
`docs/rewrite/sections/simplification/records/v1_rl_2592_512.txt`. Run 731
(`wild-universe-731`) trained with the `rl` profile.

| | v1, run 731 | v2, `real_scale_elimination` |
|---|---|---|
| Mode | One mode: last team standing. Done when a team that exists has no ship alive (`env/physics.py:_check_game_over`). No respawn | Elimination, as above |
| Time limit | 1,024 physics ticks = 17.1 s, then truncation with no winner (`max_episode_steps`; `env/wrapper.py:409-411`) | 180 s, health tie-break, shrink |
| Team size | 4v4 (`num_ships=8`); `team_sizes` option at reset for other sizes | 5v5; `active` counts for unequal teams |
| World | Torus 1,024 x 1,024, no boundary (`ShipConfig.world_size`); every distance wrapped | Disc of radius 7,800 m with boundary damage; no wrap |
| Clock | Physics 60 Hz, decisions 30 Hz: `action_repeat=2` | Physics = decisions = 30 Hz |
| Spawns | Uniformly random position and heading; team slots shuffled per environment; health and power uniform in 75 to 100% (`spawn_resource_spread=0.25`); speed 100 | By team side, full shield, 40% store |
| Ships | One type. Health 100, no shield, no recharge. Power store 100 with 10 per s passive gain | Two or three types. Rechargeable shield only. Energy store with engine charge |
| Flight | Thrust table (8 / 80 / -80), lift and drag coefficients by turn level, attitude set from the velocity plus a fixed turn offset (5 or 15 degrees), power exchanged against speed; 570 operations per tick | Section 2; 1,575 per tick |
| Guns | Speed 500, damage 10, life 1.0 s, cooldown 0.1 s (6 ticks), spread 12, cost 2, 10 slots, collision radius 10. Head-on hit lands 30% (`bullet_min_damage_frac=0.3`) | Muzzle 1,712 m/s, damage about 20, life 0.367 s, 1 shot per tick, 11 slots, hit radius 12 m with a 5 m core, no damage dead ahead |
| Hit rule | Dense (B, N K, N); two swept sub-segments per tick; **a bullet damages every ship it touches in the tick** (`_combat_damage_tensors`: `valid_hit` summed over ships) | Dense (B, N K, N); one relative swept test; a bullet hits only its first contact |
| Fields | 4 per map from a cache of 512 maps with nesting; refractive index as in v2; **interface damage** to ships crossing a field edge (10) and to bullets' damage potential | 10 random fields, no interface damage |
| Fog | None; the scripted agent and the model read the true state; the model reads bullets | Fog on by default; no bullets in the observation |
| Game length | 5.4 s at equal numbers with 4 learned ships, 7.9 s at 1.75 scripted ships each; the cap is 17.1 s (`origin/main:docs/evaluation.md`, figure 8) | Median 45 s (measured above) |
| Size-curve protocol | The learned team fixed at team 0, the scripted team's count raised until the learned win rate falls below 50%; ties count against the learned team (`modes/crossover.py`, `docs/evaluation.md`) | `league/crossover.py`, on the preset's own arena at every size (`sections/fixed-maps/progress.md`) |

Inference: "the mode closest to v1's" is right in rules (no respawn, last team standing)
but not in rhythm. v1's fights start at once from mixed random spawns and end in seconds;
v2's elimination has a 17 s approach, sides, and a boundary. If the stage 1 curve is to be
"v1's result reproduced", the spawn rule and the clock matter as much as the mode name
(question Q4).

## 2. The flight model, as the thing to lift

Sources: `game/physics/flight.py`, `game/physics/numerics.py`, `game/rules.py:FlightConstants`,
`game/ships/real-scale/rules.toml` and the three ship files; design notes in
`docs/rewrite/flight-model.md` (the 2026-10-22/23 proposal), `docs/explanation/game/physics.md`,
`docs/explanation/game/real-scale.md` and the version history in `game/__init__.py`
(versions 9 to 13). The normative statement is `docs/rewrite/sections/game-spec.md`
section 3; the code is what I read.

### 2.1 What the documents say about its design

- **Units:** per unit mass (mass 1), SI. The standard game was a P-51D with lengths x5.82
  and times x4.0; the real-scale set is "one metre is a metre", then played at **x2
  native speed** since game v13 (speeds x2, accelerations and gravity x4, times /2,
  energies x4, powers x8; lengths kept). So gravity is 39.24 m/s^2, not 9.81
  (`game/real_scale.py:EQUIVALENT_FACTOR`).
- **The store is altitude** in a 2D sky. Boost trades store for speed (a dive), reverse
  trades speed for store (a climb). The engine charges the store, less as it fills past a
  knee.
- **One lift budget,** capped by the g limit, shared between the turn and holding level
  flight; the turn has priority; what the wing cannot hold sinks the ship (store into
  speed). "Slow is deadly."
- **The nose is a damped oscillator** following the commanded angle of attack, solved
  exactly per half tick, so the gun (along the nose) lags and leads the flight path.
- **Integrator:** kick, drift, kick, one field evaluation per tick, second order
  throughout. Measured in the prototype: 0.25 m mean position error after 2 s of random
  inputs at 30 Hz (`flight-model.md`, "Accuracy").
- **Real-scale switches** (off in the standard set, on here): direct charge (v9), drag
  rise near a critical speed (v9), flow separation near maximum lift, propeller thrust
  (softmin exponent 2) and the exchange as a vertical lift demand (v11), per-type stick
  levels (v10).

### 2.2 State per ship

| Variable | Meaning | Unit |
|---|---|---|
| x, y | position, map frame, origin at the arena centre | m |
| heading_cos, heading_sin | the nose direction (unit pair, renormalised each tick) | |
| proper_speed u | speed through the medium, u = n x world speed; floor epsilon | m/s |
| angle_of_attack alpha, with its cos and sin cached | the flight path is the nose turned back by alpha | rad |
| nose_rate r | rotation rate of the nose | rad/s |
| energy E | the store, in [0, E_max] | J per kg |
| log_refractive_index, log_index_gradient_x, _y | ln n and its gradient at the ship, cached from the last field evaluation | 1, 1/m |

Not flight but in the same structure: shield, shield_recharge_wait (s), alive, fired,
spawned, respawn_wait, gun_cooldown, next_slot (ticks, ints), ship_type
(`game/state.py:Ships`).

### 2.3 Constants and where they come from

Shared (`ships/real-scale/rules.toml` `[flight]`, overriding `FlightConstants`):

| Symbol | Field | Real-scale value |
|---|---|---|
| m | `mass` | 1.0 |
| g | `gravity` | 39.24 m/s^2 |
| alpha_s0 | `stall_angle` | 0.261799 rad (15 degrees) |
| k0 | `induced_drag_factor` | 66.348 |
| e_k | `engine_knee` | 0.5 |
| beta | `store_fade_band` | 0.02 |
| a_0 | `coast_exchange` | 1.09 m/s^2 (unused under the direct charge) |
| a_c0, a_d0 | `climb_exchange`, `dive_exchange` | 9.81 m/s^2 each |
| p | `lift_softmin_exponent` | 16 |
| zeta | `nose_damping` | 0.7 |
| omega_min | `nose_frequency_floor` | 0.49971 rad/s |
| epsilon | `speed_floor` | 1.4542e-6 m/s |
| u_crit, q | `critical_speed`, `drag_rise_exponent` | 394 m/s, 12 |
| s, m_s | `separation_drag`, `separation_exponent` | 1.0, 8 |
| p_T | `thrust_softmin_exponent` | 2 |
| switches | `direct_charge`, `vertical_exchange` | true, true |
| | `fine_stick_share` | 0.3 (solver only) |

Per type (the card, `ships/real-scale/<type>.toml` `[card]`; values to 5 figures):

| Symbol | Field | Yongma | Jujak | Vizia |
|---|---|---|---|---|
| C_max | `max_lift_coefficient` (1/m) | 0.0042451 | 0.0056783 | 0.0049097 |
| a_max | `max_acceleration` (m/s^2), 9 g | 353.16 | 353.16 | 353.16 |
| c_d0 | `zero_lift_drag` (1/m) | 4.2486e-5 | 5.8070e-5 | 4.9670e-5 |
| P | `engine_power` (W) | 9,439.4 | 11,269.7 | 10,292.6 |
| P_full | `engine_power_full_store` (W) | 2,016.2 | 2,091.7 | 2,038.5 |
| T_0 | `static_thrust` (N) | 49.055 | 33.452 | 35.695 |
| E_max | `max_energy` (J) | 497,563 | 497,563 | 497,563 |
| omega_c | `corner_nose_frequency` (rad/s) | 10.925 | 11.560 | 11.231 |
| e_b | `bleed_exponent` | 2 | 2 | 2 |
| k scale | `induced_drag_scale` | 0.58313 | 0.26636 | 0.39491 |
| alpha_s scale | `stall_angle_scale` | 1.03333 (alpha_s = 0.27053 rad) | same | same |
| a_c scale | `climb_exchange_scale` | 3.4225 (33.57 m/s^2) | 3.4853 (34.19) | 3.4585 |
| a_d scale | `dive_exchange_scale` | 4.6827 (45.94 m/s^2) | 4.1438 (40.65) | 4.4064 |
| | `coast_exchange_scale` | 20.33 (unused) | 26.63 | 23.27 |
| | `engine_knee_scale` | 1.0 | 1.0 | 1.0 |
| sigma_f, sigma_s | `fine_stick`, `sustained_stick` | 0.19641, 0.65470 | 0.22232, 0.74106 | 0.21010, 0.70034 |
| u_c (derived) | sqrt(a_max / C_max) | 288.43 m/s | 249.39 | 268.20 |

What these give in flight (handles in the ship files and `measure_characteristics`,
measured): Yongma stalls at 96.1 m/s, cruises at 334.9, tops out at 449.6, and its
settled turn rates at the three stick levels are 0.246, 0.455 and 0.593 rad/s (14, 26
and 34 degrees per s); Jujak 83.1, 316.1, 433.6 and 0.356, 0.687, 0.765 rad/s (20, 39,
44 degrees per s). Spawn speed 145.3 m/s is 1.5 stall speeds.

### 2.4 The update, in order

Notation: h = 1/30 s; n = exp(ln n); softmin_q(a, b) = min(a,b) x (1 + (min/max)^q)^(-1/q)
(`numerics.py:softmin`, 0 when both are 0); cs(x) = the truncated series for (cos x,
sin x) to x^6 and x^7 (`numerics.py:small_angle_cos_sin`); sigma is the stick level
(signed lift fraction: 0, +-sigma_f, +-sigma_s, +-1; positive left); pi is the power mode
(0 coast, -1 boost, +1 reverse).

**Tick** (`flight.py:flight_tick`), for ships alive at tick start (others keep everything):

1. Kick over h/2 with the cached n and gradient.
2. Drift:
   - gamma = the gradient of ln n rotated into the nose frame (`nose_frame`);
   - world velocity in the nose frame v = (u/n)(cos alpha, -sin alpha);
   - xi = (h/2)(gamma . v); v <- v / (1 + xi (1 + xi (1/2 + xi/6))) (the index at the
     step midpoint);
   - phi = r h; A1 = (1 - phi^2 (1/6 - phi^2/120)) h; A2 = phi (1/2 - phi^2 (1/24 -
     phi^2/720)) h; body displacement = (v_x A1 - v_y A2, v_x A2 + v_y A1), rotated by the
     heading and added to the position;
   - heading <- heading rotated by cs(phi), renormalised; (cos alpha, sin alpha)
     renormalised.
3. Evaluate the fields at the new position: ln n and its gradient (`medium`).
4. Kick over h/2 with the new n and gradient.

**Kick over dt** (`flight.py:kick`), pose held:

1. u_s = max(u_0, epsilon). S1 = setting(u_s, E).
2. C_0 = C_L(alpha_0), with C_L(alpha) = C_max (2 rho - rho |rho|), rho = clamp(alpha /
   alpha_s, -2, 2) (`lift_coefficient`: peak at the stall angle, zero from twice it).
3. (du/dt)_0, (dE/dt)_0 = rates(S1, u_0, C_0, cos alpha_0, 1/u_s^2).
4. u_m = max(u_0 + (dt/2)(du/dt)_0, epsilon); S2 = setting(u_m, E + (dt/2)(dE/dt)_0).
   Write l, T for S2's lateral lift and thrust.
5. Nose oscillator, linearised about the commanded angle:
   - path rate psi = (C_0 u_m^2 + T sin alpha_0) / (u_m n) + (u_m / n)(cos alpha_0
     gamma_y + sin alpha_0 gamma_x);
   - path slope kappa = (C_L'(alpha_0) u_m^2 + T cos alpha_0) / (u_m n), with C_L'(alpha)
     = (C_max / alpha_s)(2 - 2 |alpha / alpha_s|) below twice the stall angle, else 0
     (`lift_slope`; negative between one and two stall angles);
   - frequency Omega = max(omega_c u_m / u_c, omega_min) / n;
   - commanded angle: reach = clamp(|l| / (u_m^2 C_max), 0, 1); alpha_cmd = sign(l)
     alpha_s (1 - sqrt(1 - reach)); a = alpha_cmd - alpha_0;
   - M = exp(A dt), A = [[-kappa, 1], [-Omega^2 + 2 zeta Omega kappa, -2 zeta Omega]],
     computed as the Horner series I + B(I + (B/2)(I + (B/3)(I + B/4))) with B = A dt/2,
     then squared (`exponential_2x2`);
   - y = (-a, (r - psi) - kappa a); turn e = a + M11 y1 + M12 y2; r' = psi + kappa a +
     M21 y1 + M22 y2.
6. Speed and energy, second stage: cos alpha_mid from cs(e/2); C_mid = C_L(alpha_0 +
   e/2); k1 = du/dt from rates(S2, u_0, C_mid, cos alpha_mid, 1/u_m^2); (k2, kE) =
   rates(S2, max(u_0 + (dt/2) k1, epsilon), C_mid, cos alpha_mid, 1/u_m^2).
7. Write back: u <- max(u_0 + dt k2, epsilon); alpha <- alpha_0 + e; (cos alpha, sin
   alpha) rotated by cs(e); r <- r'; E <- min(max(E + dt kE, 0), E_max).

**Setting** (`flight.py:setting`, `_vertical_demand`; the thrust and lift budgets), for a
speed u and store E, with the real-scale switches on:

1. Engine fall phi = smoothstep(clamp((E/E_max - e_k) / (1 - e_k), 0, 1)).
2. Commanded exchange a = 0 (coast), +a_c (reverse) or -a_d (boost).
3. Lift reach C_max u^2; lift budget L = softmin_16(C_max u^2, a_max), floored at 1e-12;
   lateral command l_cmd = softmin_16(|sigma| C_max u^2, a_max) with sigma's sign.
4. Store room: sink_room = clamp(E / (beta E_max), 0, 1); headroom = clamp((E_max - E) /
   (beta E_max), 0, 1).
5. Vertical demand V = smooth max(g + a, 0) = (1/2)(raw + sqrt(raw^2 + (g/32)^2)), raw =
   g + a. (Reverse asks for more lift than level flight; boost for about none.)
6. What cannot sink comes first at an empty store: held = softmin_16(V (1 - sink_room),
   L); turn budget = sqrt(max(L^2 - held^2, 0)).
7. Lateral lift |l| = softmin_16(|l_cmd|, turn budget), sign of the command.
8. Vertical lift z = softmin_16(V, sqrt(max(L^2 - l^2, 0))).
9. Exchange a' = a - (V - z) (the shortfall sinks), then faded at the store's ends:
   a' x headroom if a' > 0, else a' x sink_room.
10. Thrust T = softmin_2(P_full / u, T_0) / m (the null power only).
11. Direct charge c = (P - P_full)(1 - phi) x sqrt(1 - (|l|/L)^2) x (z / V) x headroom / m.

**Rates** (`flight.py:rates`), for a speed u, lift coefficient C, cos alpha and 1/u_ref^2:

- S = C^2 + (z / u_ref^2)^2 (total lift coefficient squared);
- drag = (c_d0 (1 + (u / u_crit)^q) + k0 k_scale S (1 + s (S / C_max^2)^(m_s/2))) u^2
  (the bleed term is exactly S at bleed exponent 2);
- du/dt = (T cos alpha - drag - a') / n; dE/dt = (m u a' + m c) / n.

### 2.5 Game content against implementation

**Game content (behaviour to preserve):**

- the equations and their order in section 2.4, including the kick-drift-kick split, the
  two-stage speed and energy update and the exact oscillator;
- the 21 shared constants and the per-type card values of section 2.3;
- the softmin exponents (16 for lift, 2 for thrust): they shape the instantaneous turn
  curve and the propeller's static thrust;
- turn priority over vertical lift, the sink, the store fade bands, the engine knee;
- the stick's three levels per type and their signs; the power modes;
- the gun along the nose (so the angle of attack is the gun's offset from the flight
  path), and bullets inheriting the flight-path velocity (`numerics.py:flight_path_velocity`);
- the field coupling, if fields are kept: flight in proper time (divide the rates by n)
  plus geodesic bending.

**Implementation (free to restructure; same numbers up to rounding):**

- `per_ship` and `lookup` as chains of `where` rebuilt each tick: gather the per-ship
  constants once at reset; build the stick table (E, N, 7) and the power tables (E, N, 3)
  once.
- Shared value times per-type scale computed inside the tick (`constants.stall_angle *
  airframe.stall_angle_scale`, and likewise the knee, the exchanges, the induced drag,
  the fade band): store the products.
- The commanded exchange and the vertical demand V depend only on the power action and
  the ship: compute once per tick, not four times.
- `softmin` at 12 operations (two `where` guards for 0/0, two scalar tensors): 7 with a
  floor on the denominator.
- `exponential_2x2` at 85 operations per call: 16 as a batched 2x2 matrix product.
- `zero_lift_drag` at 12 operations per call because the critical speed may be a live
  0-d tensor: 3 with a plain number.
- The switches (`direct_charge`, `vertical_exchange`, `separation_drag > 0`,
  `critical_speed > 0`, bleed exponent) and the unused branches behind them.
- The truncated series for cos and sin and the Horner exponential are "normative" so that
  devices and a float64 oracle agree; library `sin`, `cos` and a matrix exponential give
  the same values to about 1e-7 and are equally valid if bit-agreement is not wanted. The
  series are also cheap, so keeping them costs nothing.
- Dead ships: v2 computes the flight for every ship and selects with `where(alive, ...)`
  over all 22 ship fields; a mask on the written fields is enough.

Measured: with these changes the flight tick is 930 operations plus one field evaluation
(96), against 1,575, and agrees with v2 to rounding (`lean_flight.py`, section 5.4).

### 2.6 Where behaviour depends on tensor layout, float precision or batch composition

Layout:

- **Team-major ship order** (ship n = team x S + slot): `state.py:ship_team`, `by_team`,
  `enemy`; `phase.py:mode_phase` (`enemy(died)`), `clock_result` (`by_team`);
  `step.py:resolve` (`respawn_open.repeat_interleave(S)`); `benched_of` (a team's last
  slots); `observe.py:view_order`; `combat.py:hits` (`team.repeat_interleave(K)`). Both
  teams must have the same S. None of it touches the flight.
- **Bullets owner-major** (E, N, K), flattened to (E, N K) with bullet o K + k owned by
  ship o (`combat.py:contacts`, `hits`; `bullets.py:move_bullets`).
- **Exact ties** go to the lowest index: the lowest ship index among a bullet's contacts
  (`first_contact`), the lowest bullet index for the killer (`hits`). Team 0 holds the
  low indices. The code comments say ordering by tau and then by miss distance was added
  so that only exact ties in both depend on the index.
- **Damage sums:** `scatter_add` order matters "only when three or more of one shooter's
  bullets hit one target in the same tick" (`combat.py:hits`, comment).

Float precision:

- **`contact_order` reads float32 bits as integers** (`tau.view(int32) << 31 |
  miss.view(int32)`), with an `abs()` so that -0.0 becomes +0.0. It fails outright in
  float64 (measured: a shape error) and assumes non-negative float32. A port needs
  another way to order (two reductions, or an integer rank).
- **Post-stall kicks amplify rounding about 100 times.** Measured over 614,400
  ship-ticks: a float32 restructured port differs from v2 by more than 2e-6 rad in angle
  of attack in 31 samples, all with |alpha| between 0.27 and 0.32 rad (the stall angle is
  0.2705), where the lift slope is negative and kappa can change sign. Elsewhere the two
  agree to the last few bits (`outlier.py`).
- **Thresholds on floats:** fire needs `energy >= cost`; death needs `total > 0.0` on a
  shield `<= 0.0`; a contact needs `miss^2 < R^2`; "hurt" is `total > 0.0`. A hit dead
  ahead has aspect exactly 0, so damage exactly 0, so it does not count as a hit for the
  shield wait.
- **Guards:** speed floor 1.454e-6; `clamp_min(1e-12)` on the lift budget and on swept
  lengths; `clamp_min(1e-30)` in `bleed`; `softmin(0, 0) = 0` by a guard.
- **Magnitudes in float32:** positions to 7,800 m have an ulp of 4.9e-4 m against a move
  of 5 to 15 m per tick; energy to 497,563 J has an ulp of 0.03 J against a charge of up
  to 247 J per tick and a shot of 575 J. Measured one-step float32 error against
  float64: position 2.5e-4 m, speed 3e-5 m/s, energy 0.016 J (`tol.py`, `lean_vs64.py`).
- **Whole ticks:** the fire rate must be a whole number of ticks (checked with exact
  fractions, `ship_types.py:ticks_between_shots`); the bullet life is rounded to whole
  moves, which quantises the gun range in steps of about 55 m (section 7.2).

Batch composition:

- Nothing in the game reduces across environments. An environment's result depends on
  its index only through (a) the pool slices (`layout.py:slices`: the mode is a function
  of the environment's position in the batch), (b) the rollout's keyed draws (keyed by
  environment and episode, `runtime/rollout.py`), and (c) a per-environment card
  `(B, types)` if one is given.
- Imugi reduces over ship types, not over the batch (`characteristics.gun_range.amax()`
  in `controllers/imugi.py:mode_index`).

## 3. Chains of consequence on the game side

Each chain reads "B exists only because of A". "Goes" and "stays" are against the plan's
non-goals (no bit-exactness, no backward compatibility, two teams, no belief) and the
skeleton's scope.

### 3.1 Because several modes share one batch

1. Several games in one batch → `Rules.pools` and `PoolRules` (`game/rules.py`) → every
   rule that differs by game becomes a per-environment tensor → `layout.Environment` (16
   tensors) → **rebuilt from Python numbers at every call** (`layout.py:environment`,
   called in `resolve`, `perceive` and `reset`; 46 operations per call with one pool,
   844 per tick in the launch mix).
2. Different code per mode → `layout.per_mode` slices the batch, runs the mode's function
   and concatenates → the mode must be a static, contiguous slice → the mode shares come
   in sixteenths and pools are merged when adjacent (`rules_of`).
3. One state shape for every mode → `ModeState` carries Frontline's leaves in every
   environment (5 zone slots, two ledgers (E, Z, N), locks, wave clocks;
   `game/state.py:ModeState`) → zone rows and zone globals in every observation.
4. One uniform layout for every mode → `Rules.mode_uniform_count` is the widest mode's
   (elimination's ships read 3 of the 17 uniforms team deathmatch's best-of-8 placement
   needs per ship) → U = 224 uniforms per environment per tick, drawn every tick, used
   only at a reset.
5. Respawn differs by mode → one mechanism with parameters: a wait per environment, and
   `respawn_open` by team → **elimination's "no respawn" is a wait as long as the
   episode** (`rules.py:_pool_rules`) → a dead ship is a ship "waiting", which benching
   then reuses (3.2).
6. The scripted agent sees only observations → it infers its mode and ship set from the
   map size in gun ranges, gravity and the mode flags (`controllers/imugi.py:mode_index`,
   `real_scale_card`) → per-view parameter tables (`imugi_parameters.ModeParameters`) →
   both layers computed on every row and selected by the mode flag.

Goes in the skeleton: all of it. One game per batch; its numbers are Python constants.
Stays, for the later modes step: the idea in the brief's assumption register (modes as
contiguous slices, per-environment values set at reset). The cost in v2 comes from
rebuilding and from computing every branch, not from mixing.

### 3.2 Because shapes are static

1. Equal team size S in a team-major layout → unequal teams need **benched slots**
   (`step.py:_bench`, `state.py:benched_of`, `active` threaded through `reset`,
   `resolve`, `perceive`, `mode_phase`, `deathmatch_targets`) → a benched ship spawns at
   every reset and is benched again → a `benched` observation column (game v18) → "a
   benched slot costs about an active one" (`sections/fixed-maps/progress.md`).
2. No data-dependent shapes → spawn and reset are computed for every ship and every
   environment each tick and selected by mask (`spawning.py:spawn`: "Computed for every
   ship and selected by mask"; `step.py:resolve` phase 8: 243 operations per tick).
3. Bullet slots fixed → K is sized by the longest life over the shortest fire interval
   (`rules.py:bullet_slots_for`: 11 at real scale) → the hit test runs over every slot
   (about 5% occupied in scripted play: measured 635 shots per 10 ships per 1,350 ticks,
   so about 0.5 live bullets per ship). For a match that may be tuned live, slots are
   sized for a 2 s life (60 slots).
4. Field count fixed → field-count sampling needs **absent fields as fake fields**
   (`medium.py:19-22`) and a `present` column.
5. Zone count fixed → 5 zone slots everywhere (3.1).

Goes: the bench (replaced by an `exists` mask with a `team` array, section 4.2), fake
fields, zone slots. Stays, and is the plan's own choice: padding to a maximum with masks;
fixed K; a branch-free reset is acceptable if it is cheap (section 4.7).

### 3.3 Because of bit-exact reproducibility

1. Reproducible games → randomness as explicit inputs to `move` and `resolve` → the
   rollout draws them from **keyed counter-based streams** by (seed, environment,
   episode, tick) (`streams.py`; `runtime/rollout.py:492-494`) → `next_episode_uniforms`
   so a map is keyed by its episode alone (`step.py:resolve`).
2. Draws must not move when a feature is toggled → the uniform layout rules ("so turning
   map-size sampling on or off moves no other draw", `rules.py:game_uniform_count`,
   `tick_uniform_count`).
3. Same bits eagerly, compiled, on any device, and against a float64 oracle → truncated
   series for cos and sin and a written-out exponential ("normative",
   `numerics.py`, `flight.py:exponential_2x2`) → the `abs()` that turns -0.0 into +0.0
   and the integer contact key (`combat.py:contacts`, `contact_order`) → exact fractions
   for the fire rate (`ship_types.py:ticks_between_shots`) → a gun range "to the last
   bit" (`controllers/scripted.py:STANDARD_GUN_RANGE`).
4. Ratings keyed on the game → `VERSION` and `FINGERPRINT`
   (`game/__init__.py`, `tests/sections/test_game_version.py`) → twenty paragraphs of
   version history, several explaining why the fingerprint did or did not change.
5. Imugi's choices reproducible → purpose-keyed draws with salts kept "so the draws are
   those of every earlier run" (`controllers/imugi_combat.py:choice_uniforms`,
   `imugi.py:AGENT_FLAGS`): 396 of its 2,124 operations per call (measured).

Goes: keyed streams, the layout rules, the fingerprint, the salts, the bit tricks.
Stays: **randomness as an explicit input to the step** (or an explicit key). It costs
nothing, and it is what makes the trajectory comparison of section 5 possible: v2 and
the core can be fed the same spread and the same actions. A deterministic tie rule stays
(some rule must exist); a version number stays.

### 3.4 Because of the belief and the auxiliary heads

1. Belief extrapolation and the next-state head wanted accelerations → the observation's
   acceleration columns → `Match.previous_velocity_x/y`, `previous_nose_rate`,
   `previous_alive`, `previously_seen` (E, 2, N) in the game state → `perceive` returns a
   changed state → `api.perceived` so a runtime with persistent buffers can copy the
   changed part back (`game/state.py:Match`, `perception/observe.py`, `api.py:perceived`).
2. Shot-outcome and shot-trajectory heads → labels by re-simulating shots in the learner
   → `bullets.py:Launch`, `launch_state`, `api.launch_capture`, `api.bullet_paths`,
   `api.observed_fields`, `api.log_index` (the medium from stored observations) →
   `Rules.longest_bullet_moves` ("the static size of what follows a shot (its labels, the
   lookahead)") → the 128-tick lookahead.
3. Hidden-row labels → `Privileged.ships`, every row as if seen.
4. Fog with a belief → `ticks_since_seen`, `time_since_seen`, `Match.vision_range` kept
   in the state "as perception takes no card".
5. **"Ghost bullets": not found in the game.** The only "ghost" in `src/bnb` is the
   belief figure (`figures/belief.py`). v2's observation has no bullets at all
   (`contracts/obs.py:Obs`: "There are no bullets (author, 2026-10-31)"); v1's model read
   them.

Goes: the acceleration state and `perceived`, the launch capture and bullet paths, the
lookahead's size. Stays in a smaller form: the truth for the next-state head is the next
state; `time since seen` returns with fog. If the shot head returns, the game records
each bullet's outcome when it resolves (the brief's "shot labels recorded by the game"),
which is one write per bullet slot and needs no re-simulation.

### 3.5 Because of N-team generality

Read: **the game already has exactly two teams** ("A game has exactly two teams (author
decision)", `game/rules.py` docstring; `TEAMS = 2`, used 37 times in `game/`). The
`decisions.md` wish that "free-for-all must not require a rewrite" left no machinery in
the game that I could find. What the two-team choice did leave is the team-major layout
with equal S, and from it the bench (3.2). Nothing to remove here beyond that.

### 3.6 Because of backward compatibility and several ship sets

1. Three ship sets (standard, design-seed, real-scale) sharing one code path → every
   real-scale flight change arrived as a **switch that is off elsewhere**
   (`direct_charge`, `vertical_exchange`, `critical_speed` 0, `separation_drag` 0,
   `thrust_softmin_exponent`, `fine_stick_share`, the bleed exponent) so the other sets
   "compute the same numbers" → both branches in `flight.py:setting` and `rates`.
2. Per-type values introduced after shared ones → the card stores **scales of shared
   values** (`ship_types.py:SCALED`, 13 fields) "so the default card (every scale 1) is
   exactly the shared game" → two multiplications where one stored number would do, in
   the tick.
3. New damage rules kept old ones reachable → the classic closeness beside the cored one,
   the arming distance (0 on every card), the explosive share, the head-on sharpness.
4. Old policy files → `contracts/obs.py:COLUMNS_BEFORE_RECORDS`, schema versions, the
   mode flags' history.
5. Old searches → `controllers/imugi_populations.json` (5,170 lines, read by one test),
   three standard Imugi parameter sets, `imugi_defaults.STANDARD_DEFAULTS`.

Goes: all of it. The core holds the real-scale numbers as products, with the switches
fixed on.

### 3.7 Because rules can be tuned live in the play shell

Live tuning → rules that may be 0-d tensors → **every branch computed and selected**
(`spawning.py:field_radius` computes three distributions; `layout.py:boundary_fraction`
both shrink laws; `layout.py:zero_share` a 24-step bisection in tensors) →
`flight.py:zero_lift_drag`'s 12 operations per call → `api.retune`, `phase.refit_zones` →
slots for a 2 s bullet life → the handles' solver and instant estimate, and ship
characteristics flown at run time (15 s on the CPU) with a disk cache and a child process.

Goes from the hot path. The designer's tools can be rebuilt on the lean tick later if
they are wanted; the cards they produced are data.

### 3.8 Because of eager CUDA-graph capture

Capture forbids host-to-device copies → `numerics.py:lookup` and `ship_types.py:per_ship`
as `where` chains instead of a gather (the docstrings say so) → constants "filled on the
device" (`spawning.py:_live`, `flight.py:zero_lift_drag`, `phase.py:layout_fields`).
Goes with capture; a gather at reset replaces them.

### 3.9 Because of fog, variable maps and random maps

- Fog → the sight test (271 operations, run in unfogged environments too), the field
  membership bitmasks (M at most 64), the fired and spawned reveal rules, `fogged` per
  environment. Not in the skeleton; returns as addition 1.
- Variable map sizes → `Match.arena_scale`, `draw_arena_scale`, `sized`, the arena and
  zone floors, and `map_scale` threaded through every map length. Fixed map: constants.
  Returns in stage 2 as a per-environment radius set at reset.
- Random maps → a new map built for every environment every tick; with fields, three
  field evaluations per tick. One fixed map: a constant array.

### 3.10 Chains inside the game design itself (content, not machinery)

These are rules that exist because of an earlier rule. They are game content; changing
them needs the author.

1. **Shields recharge** → shields refilled between passes (v3, v4) → the quiet-input rule
   (v5: any boost, reverse, hard turn or shot restarts the 3 s wait) → the reward ledger
   that un-credits healed damage (`learn/rewards.py:Ledger`, an (E, N, N) state in the
   learner).
2. **No respawn** → a losing side can run out the clock → the health tie-break (v12) →
   still slow → **the shrinking boundary** (v15) → ships could outlast it → it closes 8 s
   before the clock (v16) → two more observation globals, the shrink schedule in
   `Environment`, Imugi reading the boundary's radius.
3. **A boundary instead of a torus** → spawn headings must leave room to turn
   (`spawns.py:_heading`, `_edge_safe`, `spawn_turn_distance`) → Imugi's boundary steer.
4. **Head-on passes were mutual kills** → frontal armour (f0, theta_h, sharpness) → at
   real scale f0 = 0: no damage dead ahead → Imugi's "deny" and "hold fire" elements.
5. **Friendly fire is on** → 27% of hit damage in scripted 5v5 lands on team-mates
   (measured). Neither Imugi nor v1's agent checks its line of fire.

## 4. The minimal game step for the skeleton

### 4.1 What elimination without shrink on one fixed map needs

Flight; fire and bullets; the swept hit test with first contact; damage with the
closeness, speed and aspect terms; shields with the quiet rule; the fixed boundary; death
as a mask; the win check (a team with no ship left) and a clock; a reset that places both
teams and refills them; and, only if the fixed map has fields, the medium. That is all.

Not needed: respawn, waves, zones, capture, spawn protection, the shrink, map and field
sampling, lineup draws, benched slots, fog, the observation's acceleration state, launch
capture, aspect statistics, per-environment rule tensors, per-mode slicing, the uniform
layout, live tuning, the solver.

### 4.2 State layout (framework-neutral)

E environments, N ship slots (the maximum total over both teams), K = 11 bullet slots
per ship, F fields, P about 30 per-ship constants.

| Array | Shape | Type | Notes |
|---|---|---|---|
| `x`, `y` | (E, N) | float32 | m, arena centre at 0 |
| `heading_cos`, `heading_sin` | (E, N) | float32 | the nose |
| `speed` | (E, N) | float32 | proper speed |
| `angle`, `angle_cos`, `angle_sin` | (E, N) | float32 | angle of attack |
| `nose_rate`, `energy` | (E, N) | float32 | |
| `shield`, `shield_wait` | (E, N) | float32 | wait in seconds |
| `cooldown`, `next_slot` | (E, N) | int | ticks; ring index |
| `alive`, `exists` | (E, N) | bool | `exists` is fixed for an episode |
| `team` | (E, N) | int | 0 or 1; any assignment of slots |
| `type` | (E, N) | int | index into the card |
| ship constants | (E, N, P) or P arrays (E, N) | float32 | gathered from the card at reset: section 2.3's values as products, the stick table (E, N, 7), the exchange and demand tables (E, N, 3), gun and shield values |
| bullets `x`, `y`, `dir_x`, `dir_y`, `speed` | (E, N, K) | float32 | by owner and slot |
| bullets `moves_left` | (E, N, K) | int | 0 is inactive |
| `tick` | (E,) | int | episode tick |
| fields `x`, `y`, `radius`, `band`, `level` | (F,) or (E, F) | float32 | constant for a run on one fixed map |
| with fields only: `log_index`, `grad_x`, `grad_y` | (E, N) and (E, N, K) | float32 | the cached medium |

Inputs per tick: actions (E, N, 3) ints; spread normals (E, N, 2); reset draws when an
environment ends. Outputs per tick: damage (E, N, N) shooter to target, or its two sums
if rewards need no more; boundary damage (E, N); `died`, `fired` (E, N); `killer` (E, N);
`done`, `result` (E,).

Differences from v2's layout, each for a reason:

- **`team` and `exists` arrays in place of the team-major order and the bench.** Team
  reductions become masked sums (`alive & (team == t)`), which is what v1 did
  (`env/physics.py:_check_game_over`). An environment playing 3v7 then uses 10 of N
  slots, not 2 x 7. This removes every layout dependence listed in section 2.6 and is the
  row format the brief's observation contract already names (`exists`, `team`, `visible`,
  `alive`).
- **Per-ship constants gathered at reset** in place of `per_ship` chains.
- **No `fired`, `spawned`, `respawn_wait`, `ship_type` scale lookups, no `Match` or
  `ModeState`.** The score is the count of dead ships by team, computed when needed.

### 4.3 The functions to lift, in order

| Order | Core function | From v2 | How |
|---|---|---|---|
| 0 | `constants(card, type)` at reset | `flight.py:airframe_of`, `controls`, the `per_ship` calls in `bullets.py`, `combat.py`; products of `SCALED` | New, small: one gather per constant |
| 1 | `fire` | `bullets.py:fire`, `launch` | Nearly verbatim; drop the launch contract and the per-type lookups |
| 2 | `flight_tick`, `kick`, `setting`, `rates`, `lift_coefficient`, `softmin`, the series | `flight.py`, `numerics.py` | **Verbatim equations**, restructured as in section 2.5 (prototype: `lean_flight.py`) |
| 3 | `bullet_motion` | `bullets.py:bullet_motion`, `_half_drag`, `_half_bend` | Verbatim; one field evaluation shared with the ships |
| 3b | `medium` (only with fields) | `medium.py:medium` | Verbatim without `present` |
| 4 | `contacts`, first contact | `combat.py:contacts`, `first_contact`, `contact_order` | Swept test verbatim; **rewrite the ordering** without the bit trick; mask by `exists & alive` |
| 4b | `hit_damage` | `combat.py:hits`, `kinetic`, `closeness`, `impact_squared` | Verbatim for the damage; drop arming, the classic closeness, the aspect statistics |
| 5 | `boundary_damage` | `combat.py:boundary_damage` | Verbatim with a constant radius |
| 5b | `apply_damage`, `quiet` | `combat.py:apply_damage`, `quiet` | Verbatim |
| 5c | win check and clock | `phase.py:mode_phase` (kill modes), `clock_result` | **Rewrite**: alive counts by team; the end-of-clock rule is question Q3 |
| 6 | clocks | `step.py:resolve` phase 6 | Verbatim (three lines) |
| 8 | `reset` | `spawns.py:side_placement`, `_edge_safe`; `spawning.py:spawn` | **Rewrite**: team sizes and slots drawn per environment; placement nearly verbatim or fixed spawn points |

### 4.4 Estimated operations per tick against the 1,300 tripwire

Eager, counted as the audit counts. "v2" is measured in the single-game elimination
batch; "core" is measured for the flight (the prototype) and estimated from v2's counts
for the rest, with the reasoning.

| Part | v2 | Core | Reasoning |
|---|---|---|---|
| Controls and card lookups | 105 (+ about 95 `per_ship` elsewhere) | in the flight figure | three gathers into tables built at reset |
| Flight (two kicks and the drift) | 1,374 | **930 (measured, with the control gathers)** | section 2.5: softmin 12 → 7 (24 calls), exponential 85 → 16 (2 calls), drag rise 12 → 3 (6 calls), products stored, demand once per tick |
| Field evaluation | 288 (ships, bullets, spawn) | 96, or 0 | one call for ships and bullets together; none at spawn if spawn areas lie outside every band; none at all without fields |
| Fire | 74 | about 45 | lookups gone |
| Bullet motion | 98 + medium | about 85, or about 20 without fields | bend and index terms vanish without fields |
| Hit test and damage | 225 | about 150 | no arming, no statistics, no lookups; the killer kept |
| Boundary | 39 | about 10 | constant radius |
| Shields and deaths | 41 + about 20 | about 35 | |
| Win check, clocks, done, result | about 60 | about 30 | two masked sums and comparisons |
| Reset, branch-free every tick | 243 | about 100, or 0 on most ticks | placement about 45, spawn state about 25, selects about 30 |
| Glue (`where` over state, sweep) | about 85 | about 20 | |
| **Total** | **2,576** | **about 1,500 with fields; about 1,250 without** | |

The flight is about 62% of the core's count. The count excludes the observation and the
rewards, which v1's 1,275 per tick included (v1: ship physics 570, bullets 171,
collisions 121, rewards and statistics 299, resets 73, observation 13, other; from
`records/v1_rl_2592_512.txt`, halved for its two ticks per decision).

So, as measured:

- **With fields, the game step crosses the tripwire by about 15% on flight alone,** before
  any observation or reward code. v2's flight is 1.6 times v1's whole ship physics even
  after restructuring. `simplification.md` already recorded "Flight model: about 1,300,
  real game content"; the tripwire was set from v1's count without subtracting it.
- Further cuts inside the flight would change the integrator (one `setting` per kick
  instead of two, one kick per tick), which is game content and would fail the
  trajectory check. I do not recommend them.
- The flight is 930 elementwise operations on (E, N) arrays with no reductions, no
  indexing and no data-dependent control flow. Compiled, it is one or two kernels in
  either framework. Timed on this container's CPU at batch size 1, eager: 2.5 ms for the
  lean flight tick (about 2.6 microseconds per operation), 8.2 ms for v2's whole step at
  5v5 and 13.0 ms at 50v50 (`timing.py`; not the reference laptop). Eager, the game step
  alone would take a third to a half of the 16 ms play budget, so the flight should be
  compiled for play whatever G1 decides.

Recommendation (question Q1): keep the tripwire's intent and restate it as two numbers:
operations outside the flight function (target 500), and the flight as one compiled
function of at most 1,000 traced operations, which also counts toward the
10,000-traced-operations tripwire.

### 4.5 Hard part: variable team sizes inside one tensor shape

What v2 does: one S for both teams, a team's surplus slots benched as dead ships that
never respawn; every function carries `active`; the win score reads it. A benched slot
costs as much as an active one in every operation.

What the core should do (inferred design): N is the maximum total ships; `exists` marks
the slots in use and `team` their side, both drawn at reset; `alive` starts as `exists`.
Then:

- flight, fire and shields run on every slot and are masked on write by `alive` (a ship
  that does not exist is never alive);
- the hit test masks targets by `alive` and bullets by `moves_left > 0` (no bullet is
  ever fired from a slot that does not exist);
- the win check is `sum(alive & team == 0) == 0` or the same for team 1;
- the reset places `exists` ships by team side; slots that do not exist get a harmless
  finite state (speed at cruise, position at the centre) so that no operation divides by
  zero. v2 gives dead ships the speed floor 1.45e-6 and then computes 1/u^2 on them
  (about 4.7e11, finite, then discarded by the mask); a core that feeds padded rows to
  anything unmasked should not copy that.

Things to decide inside M1 (not for the owner): whether slots are team-contiguous (own
ships first helps nothing in the game; it may help the model's layout), and whether N
for training is the largest drawn total or a fixed pad.

### 4.6 Hard part: the hit test grows as N^2 K

Pairs per environment per tick, N^2 K with K = 11: 5v5 1,100; 16v16 11,264; 32v32
45,056; 50v50 110,000; 64v64 180,224.

| | v1 | v2 |
|---|---|---|
| Shape | (B, N K, N), K = 10 | (B, N K, N), K = 11 |
| Tests per tick | two swept sub-segments (start to midpoint, midpoint to end), wrapped on the torus | one relative swept test (bullet and ship both move) |
| Per decision | twice (action repeat 2) | once |
| Pair-sized work | both distance tests, the mask, **and the aspect angle (an `atan2`) for every pair**, then sums | the swept test, the mask and one int64 `min` (about 30 pair-sized operations); the aspect angle, damage and attribution only per bullet, O(N K), by gather and `scatter_add` |
| Hit rule | a bullet damages every ship in reach | first contact only |

Cost at 100 ships (50v50), v2's form: 110,000 pairs per environment. Each pair-sized
float32 temporary is 0.44 MB per environment (the int64 key 0.88 MB). Measured on this
CPU at batch size 1: `hits` takes 1.0 ms at 5v5 and 4.6 ms at 50v50, where it is a third
of the step (`timing.py`). For an evaluation batch of 512 games at 50v50 each temporary
is 225 MB, and thirty of them move about 7 GB per tick in eager mode. Compiled, v2's
comment says the pairs are "reduced without storing" (`combat.py` module docstring),
which is the only form that scales.

Options, with what each does to behaviour:

- **Dense with masks (v2's and v1's form).** Exact. Recommended for the skeleton;
  evaluate large sizes in batches sized to a constant E x N^2 x K.
- **Compact the live bullets into A slots per environment.** Exact while at most A are
  live. Scripted play has about 0.5 live bullets per ship, but a policy that holds the
  trigger fills 86% of all slots (942 of 1,100 live in my all-shooting timing run), so A
  cannot be small without dropping bullets. Not for the skeleton.
- **Neighbour lists** (each bullet against its J nearest ships). Exact while no more than
  J ships are within one tick's reach of a bullet (about 95 m: a bullet step of up to
  68 m, a ship step of up to 15 m and the 12 m radius), which holds except in a tight
  furball. A later
  optimisation with an overflow counter as its instrument, if the profile asks.

The other N^2 structure is the damage matrix (E, N, N): 10,000 floats per environment at
100 ships. If rewards need only "damage dealt to enemies" and "damage taken" per ship,
the scatter can go straight to two (E, N) sums.

### 4.7 Other hard parts

- **Reset without a host synchronisation.** v2 rebuilds the next episode for every
  environment every tick. The core can do the same cheaply once the map is fixed (about
  100 operations), or guard it with the framework's conditional. Reading `any(done)` on
  the host each tick is the thing to avoid.
- **First contact without the bit trick.** Two reductions (the smallest tau per bullet,
  then the smallest miss among those within a tolerance) or one float key (tau plus a
  tiny multiple of the miss distance). Either changes the outcome only for near-exact
  ties.
- **One bullet per tick per ship.** The fire rate is 30 Hz, so a ship holding the trigger
  fills all 11 slots and spends 17,250 J per second, which empties a full store in about
  29 s before the engine's charge is counted. The gun's cost, not a cooldown, is the
  limiter.
- **The clock with no shrink** (question Q3).
- **Stage 2 map sizes.** A density-scaled 64v64 map is 27.9 km in radius (7,800 x
  sqrt(64/5)); float32 positions there have an ulp of 2e-3 m, still far below the 12 m
  hit radius. No offset frame is needed in the game.

## 5. The trajectory comparison

### 5.1 What I ran, and what worked

All on the CPU, in the scratch virtualenv, from the scratch directory, with
`sys.dont_write_bytecode` set and `src/` on the path. Nothing was written into the
repository (`git status --short` is empty after the runs).

| Script | What it does | Result |
|---|---|---|
| `step_v2.py` | Imports `bnb.game.api`, builds `real_scale_elimination`, resets 4 environments, steps 60 ticks with random actions | Works. K = 11, U = 224, 5,400-tick episode. 60 ticks in 4.3 s including first-call costs |
| `count_v2.py` | Counts operations per game function for one tick, at 5v5, 50v50, with `active = (3, 5)`, with no fields, and on Clover | The counts in section 1 |
| `tol.py` | One-step and 150-tick comparison of v2 in float32 against v2 in float64, 64 environments | Section 5.4. Needs `combat.contact_order` replaced in-process for float64 |
| `refdump.py` | Builds six hand-made edge cases in a 1v1 without fields, steps 24 ticks, saves every state and event leaf per tick to `ref_edge.npz` (1,959 arrays) | Works; each case behaves as the rules say (section 5.3) |
| `lean_flight.py`, `lean_vs64.py`, `outlier.py` | A restructured flight tick, counted and compared one step against v2's `flight_tick` over 614,400 ship-ticks | 1,026 operations; agreement to rounding except post-stall |
| `stats.py`, `match.py` | Imugi against Imugi (and variants, and the chaser) in elimination without shrink or fog | Sections 1.3 and 6.3. An earlier attempt at four pairings in parallel oversubscribed the four cores and was stopped; the three reported ran one at a time |
| `count_imugi.py` | Counts Imugi's operations per function on elimination views | Section 6.2 |
| `timing.py` | Batch-1 eager timings | Section 4.4 |

### 5.2 The v2 entry points

The whole tick, from a given state with given actions (verified):

```python
import dataclasses, torch
from bnb.game import api
from bnb.contracts import Action

game  = api.PRESETS["real_scale_elimination"]      # or dataclasses.replace(game, ships_per_team=S, num_fields=M)
types = api.ship_types(ship_set=game.ship_set)      # the card, float32
rules = api.rules_of(game, types=types)             # tick 1/30 s, K = 11 from the card
state = api.reset(rules, types, api.uniforms(rules, E, generator=g),
                  fogged=torch.zeros(E, dtype=torch.bool))
state, events = api.step(rules, types, state,
                         Action(power=power, stick=stick, shoot=shoot),  # int (E, N) each
                         spread,                     # (E, N, 2) standard normals, or zeros
                         uniforms)                   # (E, 224) in [0, 1); read only by a reset
```

The flight alone, from a ships structure (verified; this is what the strict flight check
calls):

```python
from bnb.game.physics import flight as F
airframe      = F.airframe_of(types, state.ships.ship_type)
stick, power  = F.controls(airframe, action.power, action.stick)
flown         = F.flight_tick(rules.flight, airframe, rules.tick_length,
                              state.ships, stick, power, state.fields)
```

The state is a tree of dataclasses of tensors (`game/state.py`). A reference state is
built with `dataclasses.replace` on its leaves, and dumped by walking
`dataclasses.fields` (`refdump.py:leaves`).

What must be switched off or supplied to run the game alone:

- **Fog, belief, analytics, rewards, lanes, keyed streams, compilation:** none is reached
  from `api.step`. `perceive` is a separate call; the comparison never needs it. The
  rollout (`runtime/rollout.py`) is not involved.
- **The shrink.** v2 has no switch. Two ways, both verified:
  - compare only within the hold: the boundary is at R for the first 36 s (1,080 ticks)
    of every episode; hand-built states start at episode tick 0;
  - or replace the mode's rules: `shrink_hold_seconds=1e9, shrink_end_lead_seconds=0.0`
    on `rules.pools[0].mode` (with `dataclasses.replace`). The boundary fraction is then
    1.0 for ticks 0 to 5,399 and jumps to -0.038 on the last tick of the clock, where
    every ship takes one tick of full boundary damage (1.33). So stop a comparison before
    the clock's last tick.
- **The reset inside `resolve`.** When an environment ends, the returned state is already
  the next episode's. Stop comparing that environment at `events.done`, and compare the
  ending through the events (`died`, `result`).
- **Spawns.** A trajectory comparison loads v2's state, so the spawn rule is not
  exercised. To compare the spawn itself, compare distributions (positions in the disc,
  headings within the jitter and the edge rule), or feed the core the same three
  uniforms per ship (the mode's columns 0, 8 and 16 of each ship's 17).
- **The map.** Dump `state.fields` (x, y, radius, band_width, level, present) and hold it
  as the core's fixed map; or run without fields (`num_fields=0` steps correctly:
  verified); or define the map in v2 as a `MapLayout` with `rotated=False`
  (`contracts/config.py:MapLayout`), which makes v2 itself play one fixed map.
- **Lineup and team sizes.** `reset(..., ship_type=(E, N) int32)` sets each ship's type;
  `active=(E, 2) int32` passed to `reset` and `step` gives unequal teams (a team's last
  slots benched).
- **Scripted actions.** `Imugi` needs the card's characteristics. Use
  `handling.characteristics.measure_characteristics(types, rules)` (15.2 s) rather than
  `characteristics()`, which writes a disk cache (to `$BNB_CACHE` or the user cache
  directory, outside the repository).
- **A float64 reference.** `move` (fire, flight, bullets) runs in float64 if the state's
  float leaves and the card (`api.ship_types(dtype=torch.float64)`) are cast. `resolve`
  does too once `combat.contact_order` is replaced by one that does not read float32
  bits.

### 5.3 Reference states and action sequences

Three sources, all dumped from v2 once and stored as generated fixtures (the dump script
lives in the core's repository and reads the v2 worktree; the files are regenerated, not
committed, unless the worktree is unavailable in CI).

1. **Scripted play.** Imugi on both sides for 60 s, states sampled every 15 ticks: the
   states a trained game visits (approach, merge, turning fights, disengagement).
2. **Random sticky actions.** Each ship redraws a uniform action with probability 0.1 per
   tick. This reaches what scripted play avoids: stalls, post-stall angles, full and
   empty stores, long boosts, flight beyond the boundary. It is the main source for the
   strict check.
3. **Hand-built edge cases** (each a 1v1 without fields; all six run in `refdump.py`):

| Case | Construction | What v2 does (measured) |
|---|---|---|
| Stall | speed 40 m/s (stall is 96), full stick | speed recovers 42.4, 44.7, ... 87.5 m/s over 24 ticks while the angle of attack creeps to 0.10 rad |
| Boundary contact | 50 m inside the edge, flying out at cruise | damage starts on tick 5 and reaches 1.16 per tick at tick 24; shield 141.95 → 132.90 |
| Simultaneous kill | both ships shieldless, each crossed by the other's bullet in the same tick | both die on tick 1, `done`, result 0 (a draw) |
| Shield recharge | shield 10, wait 0.05 s, quiet input for 12 ticks then boost | +0.4 on the tick the wait ends (a part tick), +0.8 per tick after, wait back to 3.0 s on the boost |
| Energy depletion | 3.5 shots of store, boost and shoot held | fires on ticks 1 to 4, then only on ticks where the charge has lifted the store above one shot |
| Bullet at maximum life | one shot from cruise | 11 moves, the first in the firing tick; gone after tick 11 at 637.2 m |

Add when the code exists: a graze at the hit radius' rim; a hit inside the core; a hit
dead ahead (zero damage, shield wait not reset); a bullet with two ships in reach (first
contact); a store at exactly one shot's cost; a ship at the speed floor; unequal teams;
a full ring buffer.

Each fixture holds: the state before, the action, the spread, and v2's state and events
after, for T ticks.

### 5.4 Measured baselines for the tolerances

**(a) v2 against itself in float64, one flight tick, and a restructured float32 port
against the same float64 reference** (603,248 ship-ticks, random sticky actions, 10
fields; `lean_vs64.py`). Absolute error, median / 99.9th percentile / maximum:

| Quantity | v2 float32 | Restructured float32 port |
|---|---|---|
| x (m) | 6.0e-5 / 2.4e-4 / 2.5e-4 | 6.0e-5 / 2.4e-4 / 2.7e-4 |
| heading cos | 2.1e-8 / 1.2e-7 / 1.5e-7 | 2.1e-8 / 1.2e-7 / 1.3e-5 |
| speed (m/s) | 4.4e-6 / 2.7e-5 / 3.0e-5 | 4.4e-6 / 2.7e-5 / 3.4e-4 |
| angle of attack (rad) | 3.0e-9 / 7.1e-8 / 4.2e-7 | 3.0e-9 / 7.3e-8 / 8.7e-5 |
| nose rate (rad/s) | 2.1e-8 / 1.1e-6 / 8.7e-6 | 2.1e-8 / 1.1e-6 / 8.6e-4 |
| energy (J) | 4.6e-3 / 1.6e-2 / 1.6e-2 | 4.6e-3 / 1.6e-2 / 1.6e-2 |

So a correct port sits on v2's own rounding noise up to the 99.9th percentile, and its
worst cases are about 100 times larger, all post-stall (section 2.6).

**(b) The whole step, float32 against float64, one step from identical states** (96,000
ship-ticks; `tol.py`): position 2.5e-4 m, speed 3.0e-5 m/s, nose rate 7.9e-6 rad/s,
energy 0.020 J, shield 1.4e-3; `alive` and every bullet's `moves_left` equal in every
comparison.

**(c) Free-running, open loop: the same actions fed to float32 and float64 for 150
ticks** (640 ships, zero spread; `tol.py`):

| Ticks | Position, max (m) | Position, median (m) | Speed, max (m/s) | Energy, max (J) | Shield, max | `alive` mismatches |
|---|---|---|---|---|---|---|
| 1 | 3.2e-4 | 1.4e-4 | 2.1e-5 | 0.018 | 0 | 0 |
| 10 | 1.6e-3 | 3.7e-4 | 7.8e-5 | 0.18 | 5.0e-3 | 0 |
| 30 | 2.6e-3 | 6.6e-4 | 1.9e-4 | 0.53 | 7.6e-3 | 0 |
| 60 | 3.4e-3 | 8.8e-4 | 2.6e-4 | 0.86 | 1.4e-2 | 0 |
| 150 | 1.2e-2 | 1.5e-3 | 1.6e-3 | 1.34 | 1.4e-2 | 0 (14 deaths) |

Open loop, the game is not chaotic on this horizon: rounding grows roughly linearly, to
about a centimetre in 5 s. Divergence comes from discrete events (a graze that lands in
one run and not the other) and, in closed loop, from the first action that differs.

### 5.5 The design: four checks

**Check A, strict: one step from identical states.** For each fixture state, run one core
step with v2's action and spread and compare with v2's next state.

- Continuous quantities, per ship alive at tick start (absolute, real-scale units). Two
  bounds each: one that 99.9% of samples must meet, set at about 4 times v2's own
  float32 noise, and one that every sample must meet, set at about 5 times the worst a
  correct restructured port showed.

| Quantity | 99.9% within | All within |
|---|---|---|
| x, y (m) | 1e-3 | 2e-3 |
| heading cos, sin; angle cos, sin | 5e-7 | 1e-4 |
| speed (m/s) | 1e-4 | 2e-3 |
| angle of attack (rad) | 3e-7 | 5e-4 |
| nose rate (rad/s) | 5e-6 | 5e-3 |
| energy (J) | 0.06 | 0.1 |
| shield | 5e-3 | 2e-2 |
| bullet x, y (m); bullet speed (m/s) | 1e-3; 1e-3 | 2e-3; 1e-2 |

- Discrete quantities (`alive`, `died`, `fired`, `cooldown`, `next_slot`, each bullet's
  `moves_left`, `done`, `result`, `killer`) must be equal, except in samples flagged as
  near a threshold: a closest approach within 1e-3 m of the hit radius, a store within
  0.1 J of a shot's cost, a shield within 2e-2 of 0, two contacts of one bullet within
  1e-6 in tau. Flagged samples are counted and reported; the gate is that they are under
  1 in 10,000.
- Samples with |angle of attack| above the stall angle are also reported as their own
  group, since that is where a correct port differs most.
- Size: 100,000 ship-ticks or more from the random source (the post-stall cases are 1 in
  20,000), a few thousand from scripted play, and every edge case.
- The same check runs on the flight alone (`flight_tick`) first; it localises a failure.

Why these numbers: the 99.9% bounds would catch any formula change (a wrong constant in
the fourth significant figure moves the speed by more than 1e-4 m/s in one tick: drag at
cruise is about 6 m/s^2, so 0.1% of it over 1/30 s is 2e-4 m/s). The all-samples bounds
exist only to admit the post-stall amplification.

**Check B, loose: short open-loop rollouts.** From each fixture start, feed v2's recorded
actions and spread for 150 ticks (5 s) and compare the trajectories.

| At tick | Position (m) | Heading (rad) | Speed (m/s) | Energy (J) | Shield |
|---|---|---|---|---|---|
| 30 | 0.05 | 5e-4 | 0.01 | 5 | 0.1 |
| 150 | 0.25 | 2e-3 | 0.05 | 30 | 0.3 |

- These are about 20 times the float32-against-float64 drift measured above, so a
  correct port passes with room, and they are tight enough to catch what one step can
  hide: a 1% error in zero-lift drag is 0.06 m/s^2 at cruise, which is 0.3 m/s and 0.75 m
  after 5 s; a sign error in the field bending, a wrong half-step order or a tick-length
  slip are far larger.
- An environment where a discrete event differs (a death or a hit in one run only) is
  dropped from that tick on and counted. Gate: under 1% of environments by tick 150.
- Why 150 ticks: it spans 13 bullet lives and half a sustained turn (Yongma's full circle
  is 10.6 s, Jujak's 8.2 s). A 600-tick run (20 s, two full circles) is reported with the
  bounds times 10 but does not gate, because discrete events accumulate.
- Open loop only. With a controller in the loop the runs part at the first sampled
  action that differs, and the comparison then says nothing about the physics.

**Check C, distributional: whole games.** The same controller on both sides in v2 and in
the core, 2,048 games each of elimination without shrink at 5v5, plus a smaller run at
unequal sizes. The controller must be one that is identical in both: random sticky
actions, and v2's `Chaser` (`controllers/scripted.py:Chaser`: deterministic, stateless,
about 40 lines), ported for this purpose. Not Imugi against the core's new opponent,
which would confound the game with the player.

- Statistics: game length (median and quantiles), time of first kill, deaths per game,
  win, draw and loss shares and team 0's share (a symmetry check), shots per game,
  damaging hits per shot, damage per hit, friendly share of damage, boundary damage per
  game, mean speed, store and shield over ship-ticks, share of ship-ticks post-stall,
  share of games that reach the clock.
- Gate: each within 3.5 standard errors of the difference between the two samples (about
  12 statistics, so a false failure about once in 200 runs). For scale: 48 scripted
  games gave a mean length of 54 s; if the spread is about 25 s, 2,048 games give a
  standard error of 0.55 s on each mean, so a 3% shift in game length is visible.
- Cost: v2 eager on the CPU ran 48 games of 120 s in 105 s with Imugi in the loop
  (measured). Inferred: 2,048 chaser games are minutes on a GPU, or about an hour on a
  CPU in batches.

**Check D, the designer's numbers, as a permanent property test.** The core flies each
type's standard manoeuvres and must reproduce the card's measured characteristics within
0.2%: stall speed, cruise, top speed, corner speed, the three settled turn rates, gun
range (values in sections 2.3 and 6.2, from `measure_characteristics`). This is the
check of "feel" in the units a designer uses. Unlike A to C it does not need v2 and
survives after the worktree is gone.

Order of use: D and the flight-only form of A while porting the flight; A and B for the
M1a gate; C once the opponent exists. After M1a, A and B are deleted or kept as one
small fixture (the brief's rule on pins); C's statistics and D stay.

## 6. The scripted opponent

### 6.1 v1's scripted agent

Read from `origin/main:src/boost_and_broadside/agents/stochastic_scripted.py`,
`scripted_utils.py`, `stochastic_config.py`. About 170 operations per call (the audit's
count: 852 for 5 calls per step).

- **Reads the true state** of every ship (v1 has no fog), not an observation.
- **Target:** each ship's nearest living enemy, by an N x N wrapped distance.
- **Aim:** first-order intercept with a constant bullet speed (500) and both ships'
  velocities; no drag, no fields.
- **Turn:** P(turn) is a linear ramp of the angle error from 0.03 to 0.12 rad (0.068 to
  0.95); P(sharp) a ramp from 0.24 to 0.42 rad (0.05 to 0.95); the side is the error's
  sign.
- **Power:** P(reverse) when close (10 to 50 units: 0.5 to 0.05); boost when slow (speed
  16 to 32: 0.68 to 0.04) or when the store is ample for the distance; coast otherwise.
- **Shoot:** P(aligned), by the angle over the target's angular size (ratio 1.0 to 1.5:
  0.95 to 0.05), times P(in range) (200 to 500: 0.95 to 0.05).
- **Output:** three independent marginals (3, 7, 2), sampled with ordinary random draws;
  the marginals are also the BC soft labels.
- 14 ramps, 28 numbers (`StochasticAgentConfig`), with a flat-vector interface for a
  parameter search. A team-target blend exists and is off by default. No memory, no
  hysteresis, no boundary (torus), no ship types.
- The ladder of weaker opponents is a mixture: each action scripted with probability p,
  else uniform (`agents/semi_random_scripted.py`). Other one-trick agents exist
  (`jouster`, `boom_zoom`, `jinking`, `run_away`, `spiral_evader`, `reverse_turret`,
  `abreast`, `team_jouster`).

### 6.2 v2's Imugi

Read from `controllers/imugi.py`, `imugi_combat.py`, `imugi_doctrine.py`,
`imugi_frontline.py`, `imugi_memory.py`, `imugi_parameters.py`, `imugi_defaults.py`, the
two JSON files, and `docs/explanation/imugi.md`. **Measured** on elimination views, one
call for 8 views of 5 own ships (`count_imugi.py`): **2,124 operations** (the audit's
launch figure is 2,145).

| Part | Ops per call | What it does | Matters in elimination without shrink or fog? |
|---|---|---|---|
| Context and memory (`context`, `remember`, `commit`, `switch_probability`) | about 270 | Reads own and enemy rows; remembers each enemy's last seen position and velocity; dead reckoning with a fade (0.255 s centre); committed target, turn side, goal and recovery that switch at a smooth rate | Hysteresis: yes, a little. Dead reckoning: no (nothing is hidden) |
| Target choice (`choose_target`) | 84 | Nearest known enemy by distance over weight, with switching hysteresis | Yes |
| Lead (`lead`) | 24 | First-order intercept using the drag law's time of flight, capped at 4.78 gun ranges | Yes |
| Stick (`stick_marginal`, 3 calls) | 406 | P(turn) and two level logistics in stall angles; committed side for errors near 180 degrees; each floored (0.02) | Yes (one call; the others are the "hold" case and the Frontline layer) |
| Power (`slow`, `far`, `power_marginal`) | about 60 | Boost when slow (in corner speeds) or far with store; reverse when close (0.21 gun ranges); floored (0.04) | Yes |
| Gun | about 60 | Aligned (angle over angular size, clamped 1 to 45 degrees) x in range (1.04 gun ranges) x target weight; 0 while the gun cools or the store cannot pay | Yes |
| Boundary steer (`steer`, in `context`) | about 50 | Within 2.28 g-limited turn radii of the edge and heading out, blend a steer to the centre | Yes (v1 had no boundary) |
| Type doctrines (`doctrinal`, `advantage_weight`) | 296 | Per type: extend, slash, bleed, lag, deny, weighted by an advantage logistic. Stored real-scale values: Yongma slash 0.874, lag 0.129; Jujak bleed 0.816, deny 0.806, lag 0.305; extend 0 for both | No measurable strength here: see 6.3 |
| Manoeuvres (`maneuvers`, `commanded_stick`) | 392 | Lead turn, yo-yo, brake, post-merge sense (one-circle, own type only), crowd sense, duel extension, hold fire; most weights 0 in the stored doctrines | Mostly off |
| Frontline layer (`frontline`) | 567 | Zone strength, defence assignment, separation, recovery, blend with combat | **No. Computed on every row and discarded** (`imugi.py:decide`: `torch.where(in_frontline, ...)`) |
| Keyed draws (`row_uniforms`, `action_uniforms`) | 396 | Eight hashed streams for the plan switches, the sense and the three action draws | No: one ordinary draw |
| Logistic switches (`logistic`, 43 calls x 9) | 387 (inside the rows above) | Hill functions in the quantity's log, with a temperature | The gates yes; 23 of the 43 belong to doctrines |
| Mode and parameter resolution (`mode_index`, `resolved`) | 24 | Infers the mode and ship set from the observation | No |
| Temperature and share | few | Flatten every decision (T) or mix with random play per ship per episode: the ladder's rungs | For a ladder, later |

Without doctrines: 1,342. Without doctrines or memory: 1,344 (the memory code still
runs). The real work of the combat core, taking out the Frontline layer, the keyed draws
and the mode resolution, is about 420 operations in v2's form.

The parameters are dimensionless multiples of each type's characteristics
(`measure_characteristics`, measured here for Yongma, Jujak, Vizia): cruise 334.9, 316.1,
325.4 m/s; top speed 449.6, 433.6, 441.6; corner speed 288.4, 249.4, 268.2; settled turn
rate at the three stick levels 0.246/0.455/0.593, 0.356/0.687/0.765, 0.297/0.559/0.673
rad/s; gun range 637.3, 632.3, 634.8 m; launch bullet speed 2,046.9, 2,028.1, 2,037.4
m/s; climb power 7,423, 9,178, 8,254 W; stall angle 0.2705 rad; vision 1,000 m. The
real-scale elimination parameter set is a copy of the deathmatch set "without a search"
(`controllers/imugi_real_scale.json`).

### 6.3 What the layers are worth in this mode (measured)

Elimination without shrink or fog, 5v5 (Yongma and Jujak alternating), 64 games per
pairing, seats alternated, 100 s horizon, eager CPU (`match.py`). The score is the first
player's wins plus half its draws over the games that ended; games still running at the
horizon are left out.

| First player | Second player | Ended | Win / draw / loss | Score | Median length |
|---|---|---|---|---|---|
| Imugi (stored doctrines, memory) | Imugi without doctrines | 55 | 27 / 0 / 28 | 0.49 +- 0.07 | 42.5 s |
| Imugi (stored doctrines, memory) | Imugi without doctrines or memory | 60 | 27 / 0 / 33 | 0.45 +- 0.06 | 45.7 s |
| Imugi (stored doctrines, memory) | Chaser (`controllers/scripted.py:Chaser`) | 61 | 60 / 0 / 1 | 0.98 +- 0.02 | 32.2 s |

Reading, with the sample's limits (64 games, one standard error shown, fog off):

- **In this mode the doctrines and the memory buy no strength that 64 games can see.**
  Imugi with neither scored 0.55 against full Imugi, which is within noise of even and
  agrees with `imugi.md`'s own measurement that the memory costs strength (0.374 for the
  agent against itself with memory off, across the three standard presets). A difference
  under about 0.13 either way would not show at this size.
- So the 780 operations of doctrines and manoeuvres, the 567 of the Frontline layer and
  the 396 of keyed draws can go from the stage 1 yardstick without making it measurably
  weaker here. The doctrines were tuned for mixed-type duels and Frontline; this says
  nothing about those.
- The combat core is far above the chaser (aim straight at the nearest enemy, fixed
  distances): 60 of 61. Inferred: the lead, the soft gates and the boundary steer are
  where the strength is, since those are what the chaser lacks.
- A 512-game repeat on a GPU would settle whether the simplified opponent is equal or
  slightly stronger; it is worth doing once, since it fixes the exchange rate of 6.5.

### 6.4 Side by side

| | v1 scripted | Imugi |
|---|---|---|
| Reads | true state | the team's observation (fog included) |
| Ops per call | 170 | 2,124 (1,342 without doctrines) |
| Targeting | nearest enemy | nearest known enemy, with hysteresis and dead reckoning |
| Aim | constant-speed intercept | intercept with the drag law, capped |
| Stick | 2 linear ramps | 3 logistics, committed side, floors |
| Power | 3 ramps | 3 logistics; type doctrines change it |
| Gun | 2 ramps | 2 logistics, cooldown and store gates |
| Boundary | none (torus) | steer to the centre near the edge |
| Ship types | one | per-type characteristics and doctrines |
| Memory | none | plan commitments and last-seen enemies |
| Modes | one | combat layer plus a Frontline layer, inferred per view |
| Randomness | ordinary draws | purpose-keyed streams |
| Softness | probability ramps with 0.05 ends | logistics in log space, floors, a temperature |
| Weak rungs | scripted with probability p per action | temperature, and a per-ship share drawn per episode |
| Parameters | 28 numbers, hand-set with a search interface | 29 fields per mode (about 50 numbers), searched per mode, plus 36 fields per type doctrine |
| Lines | about 630 | about 2,470, plus 5,609 lines of JSON (5,170 of them the unused populations) |

### 6.5 Proposal: the simplified opponent for the skeleton

**Behaviours to keep** (Imugi's combat core, stated in the card's own units so it follows
any ship):

1. target: the nearest living enemy, kept unless another is closer by a set factor (one
   comparison of hysteresis; no rates, no draws);
2. aim: first-order intercept with the drag law's time of flight, capped;
3. stick: P(turn) and the two level gates as logistics in stall angles, side by the
   error's sign;
4. power: boost when slow or far with store, reverse when very close, coast otherwise;
5. gun: aligned times in range, zero while the gun cannot fire;
6. boundary: steer to the centre near the edge;
7. soft outputs: three marginals with floors, sampled with one ordinary draw of (E, N, 3)
   uniforms; the marginals double as BC soft labels, as in v1 and v2.

Left out: the Frontline layer, dead reckoning (until fog), the mode inference (the game
is passed at construction), keyed draws, the manoeuvre set, and the type doctrines,
which section 6.3 could not tell from nothing in this mode.

It reads the game's arrays directly in the skeleton (the observation contract comes in
step 2 and fog later); it is written against a small "what a team may see" view so that
fog can be put in front of it without rewriting it.

**Operation budget** (inferred, eager): pairwise offsets and the masked nearest enemy 12;
hysteresis 6; lead 9; angle error 6; boundary 12; stick 25; power 18; gun 15; sampling
12; glue 15. About **130, or about 170 with a small per-type power rule** (Yongma boosts
into the attack, Jujak slows to its corner speed), against the tripwire of 170. All
per-ship parameters are (E, 1) or (E, N) arrays, so varying them costs nothing per tick.
Proposed tripwire once set: 200.

**Making it broader for behaviour cloning.** Options, cheapest first; each is a draw at
reset, so none adds operations per tick:

| Option | What varies | What the clone sees |
|---|---|---|
| A. A population of parameter settings | Each ship (or team) draws its gate centres from a spread around the tuned values: engagement range, how early it turns hard, when it reverses, trigger discipline, boundary margin | Different temperaments from one rule family |
| B. Temperature per ship | Imugi's own lever: sharper or softer decisions | More and less decisive play; more entropy in the labels |
| C. Styles as parameter extremes | A preferred stand-off distance (0: jouster; about one gun range: kiter), an extend-after-pass weight (boom and zoom), a slow-to-corner weight (turn fighter). These are v1's one-trick agents and v2's cheese agents expressed as numbers in the same formulas, not extra branches | Qualitatively different tactics |
| D. Target rule weights | Nearest, weakest shield, the one aiming at me, the team's common target (v1 had the blend) | Focus fire and self-defence, which nearest-enemy never shows |
| E. Human-like imperfection | Reaction delay (act on the target chosen k ticks ago), action hold (redraw every 1 to 6 ticks), aim noise that persists for a second | Play that is less twitchy than a per-tick sampler, closer to a person's |
| F. A line-of-fire check | Hold fire when a team-mate is nearer the line than the target | Less friendly fire (27% today); changes the opponent's strength |

Recommendation: A, C and E for the BC teacher, with the persona fixed per ship per
episode so that a recurrent student can infer it from behaviour; B only as the ladder;
D as one extra weight; F as a separate decision for the author because it changes how
the game is played.

**The yardstick and comparability.** The teacher may be a population; the yardstick must
be one fixed setting. Keep them as two named things: "the yardstick" is the centre
persona, no reaction delay, no persona draw, versioned; "the teacher" is the population
around it. The size curve and the RL requirement are measured against the yardstick only.

What keeps old results comparable, and what does not:

- **Nothing short of the old opponent in the old game keeps v1's 1.75 comparable.** It
  was measured against v1's agent in v1's game (torus, one ship type, other guns). It is
  already not comparable with v2's 1.49 (`model-audit.md`, finding 3).
- **Keeping v2's 1.49 comparable needs Imugi itself** (with its stored doctrines and
  memory, in deathmatch on v2's training arena: `model-audit.md` reads the figure off the
  deathmatch table). Porting that is about 2,470 lines and 2,124 operations per call,
  which is what the audit set out to remove.
- **A simplified opponent breaks direct comparison with both.** The cheap repair is an
  exchange rate measured in v2, where every player exists: write the simplified opponent
  as a v2 controller (about 150 lines against v2's observation), then measure (a) its
  break-even against Imugi and (b), when a GPU is free, v2's champion against it, in
  elimination without shrink. That re-bases 1.49 onto the new yardstick without porting
  Imugi, and it tells the author how much weaker the new yardstick is before any
  training.
- Using Imugi's own tuned centres for the kept gates (the `real_scale_elimination` set:
  turn 0.0106 stall angles, levels 0.0775 and 0.386, reverse inside 0.21 gun ranges,
  shoot inside 1.04, aligned within 1.01 target sizes, boundary margin 2.28 turn radii)
  makes the simplified opponent "Imugi without doctrines, memory or the Frontline layer",
  which scored 0.55 +- 0.06 against full Imugi in section 6.3. That is the choice that
  stays closest to v2, and on this evidence it costs no strength in this mode.

## 7. Open questions for the owner, and suspected defects

### 7.1 Questions, each with a recommendation

**Q1. The game's operation tripwire.** v2's flight is 930 operations after restructuring,
so the game step is about 1,500 with fields against a tripwire of 1,300 that was v1's
whole environment. Options: raise it to about 1,700; or split it. *Recommendation:*
split it: at most 500 operations outside the flight, and the flight as one compiled
function of at most 1,000 traced operations. Do not thin the integrator to fit.

**Q2. Does the stage 1 fixed map have fields?** v1 trained with 4, v2 with 10 random
ones. Fields cost one evaluation (96) and about 130 operations inside flight and bullets,
and they are part of the game's character (slow zones, bending bullets; later, sight
blocking). *Recommendation:* build the step with fields from the start, since the terms
sit inside the flight equations and adding them later means re-validating the flight;
play stage 1 on a fixed map with a few (four, point-symmetric, clear of the spawn
areas); keep a no-field map for debugging and for the cheapest benchmarks.

**Q3. What ends a game when there is no shrink?** Options: v2's rule at the clock (more
health left wins, equal is a draw); v1's (truncation, no winner); a shorter clock.
*Recommendation:* v2's health rule at 180 s. It is already the game's rule and it is
zero-sum. Report the share of games that reach the clock on the scoreboard: with
scripted play 5 of 48 had not ended at 120 s, but the shrink was added because learned
policies ran from losing fights, and that can recur.

**Q4. The arena and the spawns for stage 1.** v2's elimination starts the fleets 12.5 km
apart on a 7,800 m map: 17 s to first kill out of a 45 s median game, during which
nothing is learned. v1 scattered ships at random and fought at once. Options: v2 as it
is; the same rule on the 4,200 m deathmatch arena (about 9 s approach); v1-like mixed
spawns. *Recommendation:* the side rule on a smaller arena (4,200 m), and say so in the
stage 1 record, since "elimination without shrink" alone does not fix the rhythm.

**Q5. Friendly fire.** On in v1 and v2; 27% of hit damage in scripted 5v5 is on
team-mates, and it will grow with fleet size on a fixed map. *Recommendation:* keep the
rule (it is the game), measure the share at each size in the size curve, and decide
separately whether the yardstick opponent holds fire when a team-mate is in the line
(6.5 F).

**Q6. The lineup.** Options: fixed (the launch's Yongma x2 + Jujak x3); drawn per slot,
both teams mirrored (`modes/rules.py:DRAWN_LINEUPS`, v2's training option); drawn
independently per team. *Recommendation:* mirrored draws of Yongma and Jujak, so a
result never rests on a type advantage; Vizia stays in the card, unused.

**Q7. The scripted opponent** (section 6.5): how far to simplify, whether type doctrines
stay, and whether the teacher is a population. *Recommendation:* the 130 to 170 operation
core as the yardstick, with Imugi's tuned centres and no doctrines or memory (which
scored 0.55 +- 0.06 against full Imugi in this mode); a population around it for BC; and
the exchange-rate measurement in v2, at 512 games, before M1c.

**Q8. Which v2 physics is "the feel".** *Recommendation:* game version 20 with the
real-scale set at x2 native speed, exactly as in `ships/real-scale/` today; the standard
and design-seed sets are not ported. Confirm that x2 stays.

**Q9. What the step hands to rewards.** v2 passes the full damage matrix (E, N, N) and
the learner keeps a ledger of the same shape so that recharged shields un-credit their
attackers. *Recommendation:* decide with the learner; if damage rewards are paid when
dealt and not taken back, the game can emit two (E, N) sums and the ledger goes.

**Q10. First contact and tie rules.** *Recommendation:* keep "a bullet hits only its
first contact" (v2's rule; v1's bullets hit everything in reach); drop the bit-level
ordering.

### 7.2 Suspected defects and stale constants (unconfirmed unless marked)

1. **The gun flies 8% further than its design range** (measured). The ship files' handle
   `gun_range` is 589.5 m (Yongma), 584.3 (Jujak), 586.9 (Vizia); the card's bullet life
   is the whole number of ticks that covers it, 11 moves, in which a bullet fired at
   cruise travels 637.3, 632.3 and 634.8 m (`measure_characteristics`; 637.2 m in
   `refdump.py`). At 30 Hz a bullet moves 50 to 68 m per tick, so the range is quantised
   that coarsely. Imugi's gates use the flown value.
2. **`controllers/imugi_combat.py:44`: `STICK_MAGNITUDES = (0.0, 0.4, 0.75, 1.0)`** are
   the standard set's stick levels. Real-scale types have flown per-type levels since
   game v10 (0.196/0.655 and 0.222/0.741). `commanded_stick` (the manoeuvres' stick)
   reads the stale table.
3. **`spawn_speed = 145.288` m/s is 1.3 times the stall speed of the Hammerhead,** a type
   that left the game at v17 (`real-scale.md` lines 61 to 91), doubled at v13. For the
   ships that remain it is 1.5 stall speeds and 0.43 of cruise. `spawn_turn_distance =
   436.5` m is the standard heavy's 75 m times 5.82, not re-derived for the current
   cards (inferred: it is still ample, since the lift-limited turn radius is 1/C_max =
   236 m for Yongma and 176 m for Jujak).
4. **Imugi's real-scale elimination parameters were never searched:** a copy of the
   deathmatch set of 2026-11-12, tuned on another arena and before the boundary and
   x2-speed changes (its time parameters were rescaled by script).
5. **Imugi runs its Frontline layer on every row in every mode** (567 operations,
   measured) and selects it away outside Frontline.
6. **"Elimination without shrink" does not exist in v2.** The nearest setting applies one
   tick of full boundary damage to every ship on the clock's last tick (verified).
7. **`combat.contact_order` fails in float64** (measured): any float64 oracle of
   `resolve` has to replace it.
8. **Dead ships are held at the speed floor (1.45e-6 m/s) and still flown,** so 1/u^2 is
   about 4.7e11 before the mask discards it. Finite today; one more square would not be.
   The audit records that the one real NaN came from the game.
9. **The next-state bounds the audit suspected:** a ship at cruise moves 11.2 m per tick
   and at top speed 15.0 m (from the measured 334.9 and 449.6 m/s), both beyond the
   8.7 m position bound the audit quotes. Arithmetic only; I did not read the model code.
10. **No damage dead ahead** (f0 = 0, sharpness 3): under 1% of a hit within 10 degrees
    of the target's nose, 19% at 30 degrees, 50% at 45. A hit that deals exactly 0 still
    consumes the bullet and does not restart the shield wait. By design (v12); listed
    because it shapes every head-on pass and the author may want to see it stated.
11. **Stale documentation:** `docs/explanation/game/physics.md` gives the real-scale
    bullet life as 0.67 to 0.77 s (it is 0.367) and the critical speed as 197 m/s (394);
    `real-scale.md`'s rule table gives bullet drag 1.375e-4 (5e-4), vision 3,072 m (the
    cards scale it to 1,000) and spawn speed 72.6 (145.3); `imugi.md` quotes a 1,024 m
    vision against a 210 m gun range (1,000 against 637); `docs/rewrite/flight-model.md`
    and `ship-card.md` describe the pre-real-scale model. `Rules`' defaults
    (`bullet_slots = 6`, `longest_bullet_moves = 12`) are the standard set's.
12. **Vision range against speed, for when fog returns:** 1,000 m of sight against a
    closing speed of up to 900 m/s is about a second of warning, and the gun reaches 637
    m. Inferred; not a defect in the skeleton.

## Appendix: scratch files

Directory:
`/tmp/claude-0/-home-user-boost-and-broadside/bf4c0758-f091-53dd-a269-e94d54b68e77/scratchpad/next/game-audit-scratch/`

Run each as `PYTHONDONTWRITEBYTECODE=1 <scratchpad>/venv/bin/python <script>` from that
directory.

| File | Purpose |
|---|---|
| `step_v2.py` | the entry-point check |
| `count_v2.py` | operations per game function |
| `count_imugi.py` | operations per Imugi function |
| `tol.py` | float32 against float64, one step and 150 ticks |
| `lean_flight.py` | the restructured flight tick, its count and its one-step comparison |
| `lean_vs64.py`, `outlier.py` | error percentiles against float64; where the port differs |
| `refdump.py`, `ref_edge.npz` | hand-built edge cases and their dump |
| `stats.py`, `stats.out` | Imugi mirror statistics |
| `match.py`, `match_*.out` | pairings of Imugi variants and the chaser (64 games each) |
| `timing.py` | batch-1 eager timings |
