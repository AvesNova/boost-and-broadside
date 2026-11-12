# Mode layer tuning — progress

Branch `tune/modes` (from `integrate/next-runs` at 3d36a721), 2026-11-12. Worktree
`/home/vizia/avesnova/bnb-tune-modes` (the `bnb-modes` path holds an older `section/modes`
worktree, left alone).

## Step 1: elimination spawns by team side (done)

- Tried/built: `modes.spawns.side_placement`. Each team's area is a disc of
  `ModeRules.side_area_fraction` (0.15) arena radii around its centre at
  `Rules.spawn_offset` (0.8, the sandbox formation's) arena radii on the spawn axis; each
  ship uniform in it (the mode's existing candidate-0 uniforms, so the uniform stream's
  layout is unchanged), nose toward the enemy area's centre within `side_heading_jitter`
  (15°), then `_edge_safe`: turned the least that keeps the v4 edge rule (outward cosine at
  most d / D within D of the edge). Clover's layout spawns still take precedence; team
  deathmatch keeps its uniform spawns with the edge rule; frontline unchanged.
- Both new fields are live tuning settings (an "Elimination" group, applied at the next
  episode) and dimensionless in `similarity.RULE_UNITS`.
- Game version 7, fingerprint `2581fdc8196942e9` (only elimination's numbers change).
- Tests: `test_elimination.py` +4 (areas and sides, heading within jitter, edge rule with
  an area reaching the edge, end to end through `api.reset`). The random-play wipe test now
  plays with spawn offset 0 (`modes_cases.play(**rules)`): random actions rarely wipe a team
  across the map. Also added two missing bench docstrings (the docstring test failed on
  the base).
- Didn't / found: `test_modes.py` fails on the base already (4 tests: the respawn
  farthest-candidate test compares the pre-v4 heading; the shield-events tests see no
  recharge since v5's quiet-restart rule). Not touched (outside this task); elimination's
  shield-events case passes after this change.
- Real scale: the areas' centres are 12.5 km apart (2 x 0.8 x 7800 m), nearest points
  10.1 km: about 30 s of closing at cruise for each side. Measured below.

## Step 2a: zone travel times (done, provisional zone sizes)

`bench/zone_travel.py` (CPU, the flight tick alone: straight, stick centred, no field).
Table: `records/zone_travel.md`, JSON `records/zone_travel.json`. Geometry: ring at 0.615 R,
zone radius 330 m x map scale (R / 2600 m) = 0.127 R, so everything scales with R; the
fleet `_density` presets grow R as sqrt(n / 5). "attack entry": spawn-zone centre to the
edge of the zone it attacks (two places round the ring, chord 2 x 0.615 R sin 72° less the
zone radius). Seconds, start from spawn (spawn speed and energy) / from cruise with a full
store:

| map | R (m) | zone r (m) | Yongma coast / boost | Jujak | vizia | Zero (slowest, 131 m/s) | Firecracker (fastest, 231 m/s) |
|---|---|---|---|---|---|---|---|
| real_scale_frontline (5v5) | 7800 | 990 | 59 / 33 | 59 / 36 | 59 / 34 | 69 / 38 | 47 / 29 |
| 10v10 density | 11031 | 1400 | 80 / 43 | 81 / 46 | 80 / 44 | 95 / 52 | 63 / 38 |
| 20v20 density | 15600 | 1980 | 109 / 57 | 111 / 61 | 110 / 59 | 131 / 74 | 85 / 50 |
| 50v50 density | 24666 | 3131 | 166 / 87 | 171 / 91 | 168 / 89 | 204 / 121 | 127 / 73 |

Adjacent zones (centre to centre) take about 0.66 of these. Against the 600 s real-scale
clock: a 5v5 attack leg is 6-10% of the clock coasting; at 50v50 density it is 28-34%
coasting, 15% boosting, so a round trip (respawn, fly back) eats a large share of the
match. The `_arena` presets keep 7800 m (the 5v5 numbers).
