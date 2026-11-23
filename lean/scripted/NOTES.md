# lean/scripted: notes

What the package is today. Decisions: docs/rewrite/core-decisions.md ("Scripted opponent",
"Answers from the author review" 5.x, "Sign-off and personas"). History:
docs/rewrite/core-worklog.md.

## What it is

- `opponent.py`, the scripted opponent (the yardstick and its persona population):
  - `act(config, state, team, persona, uniforms, memory) -> (action, memory)`: one decision
    per living ship of `team`, or of both teams with `team=None` (one call; each ship
    decides as in its own team's call). `uniforms` (E, N, 3) from `draw_uniforms`: the only
    randomness (no action hold, no random switches).
  - `marginals(...)`: the three independent distributions (stick 7, power 3, fire), the BC
    soft labels; `sample` draws from them by inverse transform.
  - Memory, carried by every caller (collect's teacher and league opponent, evaluation, the
    sandbox): `fresh_memory` at a match's start, `renew_memory(memory, done, generator)` for
    re-dealt environments. "target": the kept target (switch only to an enemy closer than
    0.8 of its distance). "side": +1 left or -1 right, drawn once per ship per match: in the
    symmetric head-on pass (the wanted direction within 0.1 rad of straight behind) the ship
    turns to that side.
  - `personas(config, E, N, generator, device, spread)`: spread 0 (the default everywhere)
    is the yardstick and returns None, so no per-ship tensors are carried. 0 < spread <= 1
    draws every parameter per ship, uniform on a bounded box around the yardstick (log-uniform
    for scales): every gate centre (half-width a factor e^0.4 at spread 1), the temperature
    (e^0.4, dividing every gate's sharpness), the boundary margin (e^0.3), lead gain (+-0.2),
    weak-target weight (0 to 1), regain share (+-0.25). Hidden from the student.
  - Every ship characteristic is the game's constant, by its name in `state.constants`
    (`gun_range`, `stall`, `stall_speed`, `corner_speed`, `bullet_launch_speed`,
    `bullet_drag`, `max_acceleration`, `inverse_max_energy`, `inverse_fade`, `hit_radius`,
    `arena_radius`). `ScriptedConfig` holds only behaviour parameters.
- `chaser.py`: `chaser_act(config, state, team)`, a deterministic, memoryless floor (points
  at the nearest enemy, coasts, fires within 0.05 rad and one gun range). Used by tests and
  the sandbox's `--opponent chaser`.

## Behaviour

- Target: the nearest known enemy (its team's view: under fog, enemies at their last
  sighting), kept with factor 0.8. A weak-target preference exists for personas; at weight
  0 (the yardstick) it is not computed.
- Lead: first-order intercept with the drag law's time of flight, distance capped at 4.78 gun
  ranges, relative velocity over the map (the bullet inherits the shooter's velocity). In a
  field, velocities over the map are proper velocities over the index n, and the time of
  flight is n times longer (the shooter's n taken for the whole path).
- Stick: P(turn) and two level gates in stall angles (0.0106, 0.0775, 0.386; floors 0.02),
  the side by the error's sign (floor 0.02), the kept side in the head-on band.
- Boundary steer: within 2.28 turn radii of the game's arena radius and flying outward, a
  steer to the centre blended in (ramp 0.0554 to 0.52); without a known enemy, the centre.
- Gun: aligned (1.01 target sizes) times in range (1.04 gun ranges); never while cooling or
  with a team-mate in the line of fire.
- No retreat or extend (decision 5.7; in the backlog).

## Power (author 2026-11-23, 5.1)

Power 1 dives (store into speed; the 1 g held is freed for the turn), 2 climbs (speed into
store). The rule, each part a logistic in the ship's own characteristics:

- dive when slow: below 1.8 stall speeds, or below 0.95 corner speeds while turning hard
  (the probability of at least the sustained stick);
- a slow ship never climbs; it dives only with store to spend (the dive's probability fades
  over the game's 5% edge band of the store, as the dive's effect does), else it coasts;
- otherwise climb when above 1.2 corner speeds, or within 0.21 gun ranges of the target, or,
  on about half the ticks (regain share 0.5), when farther than 3 gun ranges from the target
  (or with no known enemy) and the store is below 0.3;
- otherwise coast. Each of coast, dive and climb keeps a probability of at least 0.02.

What it does (yardstick, `marginals`; own ship heading +x, target flying the same way; the
cases are `test_power_rule`):

| situation | speed (corner speeds) | target bearing, distance (gun ranges) | store | coast | dive | climb |
|---|---|---|---|---|---|---|
| slow, straight, target ahead | 0.55 | 0 deg, 2.0 | 0.5 | 0.07 | 0.91 | 0.02 |
| below corner in a hard turn | 0.85 | 90 deg, 1.5 | 0.5 | 0.14 | 0.84 | 0.02 |
| slow in a hard turn, store empty | 0.6 | 90 deg, 1.5 | 0.0 | 0.96 | 0.02 | 0.02 |
| at corner speed, straight | 1.0 | 0 deg, 1.5 | 0.5 | 0.93 | 0.02 | 0.04 |
| just above corner in a hard turn | 1.08 | 90 deg, 1.5 | 0.5 | 0.80 | 0.08 | 0.12 |
| well above corner speed | 1.3 | 0 deg, 1.5 | 0.5 | 0.18 | 0.02 | 0.80 |
| very close behind the target | 1.0 | 0 deg, 0.12 | 0.5 | 0.05 | 0.02 | 0.93 |
| far, store low | 1.0 | 0 deg, 6.0 | 0.1 | 0.49 | 0.02 | 0.49 |
| far, store high | 1.0 | 0 deg, 6.0 | 0.8 | 0.93 | 0.02 | 0.05 |

Scripted against scripted, before and after step S (measured 2026-11-23): the core game
(`GameConfig()`: fog, ten fields, the standard spawn), 4v4, both teams the yardstick in one
call, 256 environments x 5,400 ticks, seed 0, finished games re-dealt; shares are of living
ship-ticks. Before: core at 585dca98 (memoryless, as its callers called it, the old power
gates); after: this package (memory carried and renewed per game). Script: the step S
scratch `measure_scripted.py` (not kept). Speeds are proper speeds; the P-51D's corner speed
is 263 m/s; kills per minute are per environment, at the game's time (30 ticks a second).

| | before | after | after, spread 1 |
|---|---|---|---|
| mean speed | 184.9 m/s (0.702 corner) | 186.7 m/s (0.709) | 185.1 m/s (0.703) |
| share below 0.75 / above 1.25 corner speeds | 0.684 / 0.001 | 0.648 / 0.000 | 0.666 / 0.000 |
| mean store | 0.027 of full | 0.050 | 0.068 |
| share diving / climbing / coasting | 0.863 / 0.102 / 0.035 | 0.219 / 0.055 / 0.727 | 0.209 / 0.065 / 0.726 |
| share firing | 0.086 | 0.084 | 0.082 |
| kills per minute | 4.82 | 5.44 | 5.14 |
| games ended; mean length of an ended game | 636; 1,610 ticks | 710; 1,370 ticks | 686; 1,548 ticks |
| share of ship-ticks beyond the arena radius | 0.0000 | 0.0000 | 0.0002 |

In these brawls the fight is slow and lift-limited (two thirds of the time below 0.75
corner speeds) and the store stays nearly empty in both versions: the rule regains store only
far from every enemy, which a 4v4 brawl rarely gives. Before, 86% of ticks were "dive" and
most of those did nothing (an empty store); after, a slow ship with no store coasts.

## Checks

- `tests/test_scripted.py`: actions in range and zero on uncontrolled rows; marginals are
  distributions; spread 0 is None and uses no draw; no hold, no dependence on the tick; the
  left-turn side; the power cases; the kept target switches only below 0.8 of its distance;
  the head-on side kept through a match and redrawn only by `renew_memory`; the lead in a
  field; every number from the game (the arena radius steers at 1,500 m); no shot through
  a team-mate; one call for both teams equals the two single calls (actions and memory);
  every persona competent at spread 1 (32 x 1,800 ticks of 4v4: none past the boundary's
  ramp, at least 49 in 50 of the ships alive 20 s fire); the yardstick beats the chaser.
- `tests/test_callers.py`: collect, evaluation and the sandbox carry the memory and renew it
  per match.
- Operations (docs/rewrite/sections/scoreboard/game_ops.py, CPU, one thread, E = 256, 5v5,
  the game with fields and fog): `act` for both teams 208 compute (one team 212), against
  186 before step S without memory (196 with the old memory and hold). The medium in the
  lead is 7 of them (201 without fields).

## Open

- Basic fighter manoeuvres (one-circle against two-circle, lead turn) and the frontline
  layer: later steps (core-backlog E2).
- Retreat or extend: none (decision 5.7).
- The in-range gate uses the gun range without fields; a bullet's reach in a field of index
  n is shorter (log(1 + drag v T / n) / drag). Not modelled.
- The persona box's widths are set for competence, not tuned; the population is an
  experiment arm (`persona_spread`, default 0).
