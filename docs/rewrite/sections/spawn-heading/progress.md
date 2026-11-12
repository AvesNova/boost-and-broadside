# Spawn heading: room to turn away from the edge — progress

Branch `fix/spawn-heading` (from `integrate/next-runs` at f20e1dfc), 2026-11-12.

## Problem

The author's report: in the arena preset (2v2, real-scale set, 1950 m, a killed ship
respawns on the next tick at a random point with a random heading) a ship can spawn at the
edge heading outward and take boundary damage before it can turn.

## Where random headings were

- Team deathmatch, uniform and grouped (`modes.spawns.deathmatch_placement`): one uniform
  heading per ship, independent of the point. Uniform points reach the edge itself; grouped
  candidates are clipped to `R_a − spawn_margin`. Elimination places as team deathmatch.
- Frontline: the heading points at the enemy's defense zone, a point inside the arena; the
  straight path to it stays inside (the disc is convex). Not random, not changed.
- Fixed layouts (Clover): nose to the centre from the unjittered point, ±5° jitter, at
  0.875 R_a − 150 m at least (467 m or more from the edge, beyond the turn distance below);
  outward cosine below −0.9 in every draw (tested). Not changed.
- Sandbox formation: nose along the spawn axis toward the other team's centre. Not changed.

## Rule (done)

`modes.spawns._heading`: the uniform draws an angle `a = π(2u − 1)` off the outward normal.
Within `D = Rules.spawn_turn_distance` of the edge the angle off the normal is at least
`m = acos(clamp(d / D, 0, 1))` (outward cosine at most `d / D`, tangent at the edge), and the
draw is spread evenly over the allowed arc on its own side: `a → sign(a)(m + |a|(1 − m/π))`.

Refinement over the proposed clamp (rotate a violating heading to `c = c_max`): the clamp
puts every violating draw on the two boundary directions, so at the edge half of all spawns
would fly exactly tangent. Spreading the draw keeps the heading uniform over the allowed
directions, with the same properties otherwise: deterministic, one tensor pass, static
shapes, continuous in `d` (the cosine bound is piecewise linear; the angle has a square-root
cusp at `d = D`, still continuous), and identical to the plain uniform heading from `D`
inward. The linear cosine bound (not a smoothstep) is the chord under the exact kinematic
one: a circle of radius `D` started at `d` with outward cosine `c` reaches
`D(1 − √(1 − c²))` outward, `≤ d` when `c ≤ d / D` (the exact bound is `c ≤ √(2x − x²)`,
`x = d/D`); a smoothstep would forbid more than needed near the edge and allow more than
the bound past `D / 2`.

## D (done)

`bench/spawn_turn_excursion.py`: each type from spawn speed and spawn energy, heading
straight out, the stick full over at once, every power mode, the largest outward reach.

| set | spawn speed | largest reach | type | D |
|---|---|---|---|---|
| standard | 30 m/s | 65.4 m | Heavy (reverse) | 75 m |
| real-scale | 72.6 m/s | 404.5 m | Me 262 (reverse) | 436.5 m |

The arena's two types reach 243 m (P-51D) and 224 m (Spitfire XIV). D is the standard
value rounded up, 75 m, times the real-scale set's LENGTH (5.82): a turn radius follows the
ship's units, not the map (MAP_SCALE would give 225 m, under the real-scale set's reach). It
is a `Rules` field (`[rules] spawn_turn_distance` in the real-scale and design-seed
rules.toml, `real_scale.rule_values` writes it), a length in `similarity.RULE_UNITS`, and a
live setting in the tuning panel's spawns group (1-2000 m).

## Measurement (done)

`bench/spawn_boundary_deaths.py`, arena preset, scripted agent both sides, 128 environments,
CPU, uncompiled; respawns only, each followed for its window. "hold": ticks the spawned
ship flies straight (coast, no shot) before the agent takes over, a slow pilot.

| hold, window | build | spawns | boundary deaths | left the arena | boundary damage / spawn |
|---|---|---|---|---|---|
| 0, 5 s | before | 5137 | 1 (0.02%) | 13.7% | 3.80 |
| 0, 5 s | after | 5773 | 0 | 0.02% | 0.00 |
| 2 s, 5 s | before | 4916 | 1 (0.02%) | 23.0% | 7.18 |
| 2 s, 5 s | after | 5372 | 0 | 0.06% | 0.00 |
| 5 s, 5 s | before | 2304 | 0 | 27.1% | 8.01 |
| 5 s, 5 s | after | 2256 | 0 | 2.7% | 0.12 |
| 15 s, 15 s | before | 2476 | 1052 (42.5%) | 58.5% | 49.9 |
| 15 s, 15 s | after | 2135 | 416 (19.5%) | 39.2% | 27.9 |

Reading: within 5 s almost no spawn dies to the boundary even before the change (a 100
shield against 5 + 0.05/m per s), so the scripted agent's deaths are not the measure of
the flaw; leaving the arena is. Before, one spawn in seven left the arena within 5 s with
the agent turning at once, one in four with a slow pilot. After, essentially none do with
any reaction; a pilot who flies straight for 5 s leaves on 2.7% of spawns (a tangent line
leaves a circle) and takes 1/65 of the damage. A pilot who does nothing for 15 s (about
1500 m of straight flight, most of the arena's 3900 m width) dies to the boundary on 42% of
spawns before and 19% after: the rule buys turning room, not a path that never needs a
turn.

## Game version (done)

Version 4, fingerprint `85912f2a4ebdafa9`: team deathmatch's and elimination's spawn
headings change in every preset with random headings (training presets included:
deathmatch 75 m of 2600 m, 5.7% of the area; real_scale_deathmatch and
real_scale_elimination 436.5 m of 7800 m, 10.9%; the arena 40%).

## Tests

`tests/sections/test_spawn_heading.py`: the set's D; at the edge no heading points outward;
near it the outward cosine is at most d / D and the arc is used to its end; from D inward
the heading is the uniform draw's (and uniform by sector); near the edge it is uniform over
the allowed arc; continuous in d; the arena's placement end to end; Clover's fixed spawns
face inward beyond D. Quick tier on the touched files plus the tuning, time-scale and
mode-rule files: green; ruff clean. The compiled `_heading` matches eager to 1.2e-7.
