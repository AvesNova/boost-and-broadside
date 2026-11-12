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

## Box (gpubox4, vast.ai)

RTX 5060 Ti 16 GB (Blackwell, sm_120, 180 W, driver 595.99.02), i5-12400F with a 5.76-core
CPU quota, 31 GB. The locked env: torch 2.15.0+cu132, Triton 3.9.0, Python 3.14; sm_120 is
in torch's arch list and `torch.compile` works out of the box. No Blackwell problem so far.
Notes: Triton 3.9 warns that `TRITON_DEFAULT_FP_FUSION` is unset (fusion stays on; set it to
1 to keep today's numerics and silence it). A compile (inductor workers + ptxas) takes the
whole CPU quota: an eager loop beside it ran at 0.65 s/tick, so eager measurement loops are
out and every sim here runs on the compiled rollout (CUDA graphs). The match engine builds
a 5v5 real-scale arena in ~3 min cold, 1 s warm; a 256-game batch of 5v5 real-scale
deathmatch then takes ~5 s. Gotchas: `pkill -f "uv sync"` over ssh kills the ssh command
itself; run long jobs in tmux.

## Step 2b: weaker rung, 5v5 (done)

`bench/density_sweep.py --sizes 5 --radii fixed` (real_scale_deathmatch, no fields, 5v5,
7800 m, current tickets 25 / 600 s): the scripted agent (T = 1) against `rung_T<t>`,
512 games each, score = wins + ties / 2:

| rung | score | ± |
|---|---|---|
| T = 1.5 | 0.610 | 0.021 |
| T = 2 | 0.697 | 0.020 |
| T = 3 | 0.900 | 0.018 |

T = 2 is the sweep's weaker rung (a clear edge, far from saturation).
Sweep launched 16:01 KST: sizes 5/10/20/35/50 per side at the fixed 7800 m arena and at the
`_density` radii (7800 sqrt(n / 5)); ±0.025 or 1536 games per cell; bounded at 3 h.

## Step 2c: deathmatch kill rate, 5v5 (rough, default lineup)

`bench/mode_dynamics.py deathmatch --label dm_5v5` (real_scale_deathmatch, no fields, the
game's lineup P-51D/Heavy alternating, scripted mirror T = 1, tickets off, 600 s), 512
games; record `records/dynamics.jsonl`, report `bench/mode_dynamics_report.py`.

- Kills per minute (both teams): mean 8.60, median 8.60, quartiles 8.4-8.8, 5-95% 8.2-9.1.
  First point at a median 12.8 s.
- **Trades**: 95.5% of deaths have an enemy death within 0.5 s (97.6% within 3 s): ships
  die in pairs in head-on merges. Final |margin| <= 2 in 97% of games, a tie in 41%. A
  combat-layer matter (the gun study's trades, much stronger at 5v5); flagged to the
  coordinator. Re-measure after the combat freeze.
- Share of games a team's tickets end, by clock (columns: tickets per team):

| clock (s) | T=20 | T=25 | T=30 | T=35 | T=40 | T=45 | T for 80% |
|---|---|---|---|---|---|---|---|
| 300 | 0.97 | 0.01 | 0 | 0 | 0 | 0 | 21 |
| 400 | 1.00 | 1.00 | 0.29 | 0 | 0 | 0 | 28 |
| 500 | 1.00 | 1.00 | 1.00 | 0.90 | 0.01 | 0 | 35 |
| 600 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.21 | 42 |

  The kill count is far less spread than a Poisson count (sd ~0.3/min against ~0.9), so the
  share ended by tickets drops from 1 to 0 within a few tickets: "80% by tickets" sits on a
  knife edge and moves with any change in kill rate. Today's 25 tickets in 600 s end every
  game by tickets, at ~5.8 min.
- Next: the same with the Yongma/Jujak lineup (queued), then Frontline.

## Step 2d: deathmatch, the release lineup (2 Yongma + 3 Jujak a team; game v6/v7 cards)

`--label dm_5v5_yj --lineup jujak,yongma` (slots J, Y, J, Y, J; both teams), otherwise as 2c.
512 games. The default-lineup numbers above (P-51D/Heavy) do not carry over.

- Kills per minute (both teams): mean 7.19, median 7.20, quartiles 7.0-7.4, 5-95% 6.7-7.6.
- Trades: 14% of deaths have an enemy death within 0.5 s, 49% within 3 s (95% / 98% with
  P-51D/Heavy). Final |margin|: median 2, a tie in 12.5%.
- Tickets (per team) by clock, share of games ended by tickets:

| clock (s) | T=15 | T=20 | T=25 | T=30 | T=35 | T=40 | T for 80% |
|---|---|---|---|---|---|---|---|
| 300 | 1.00 | 0.16 | 0 | 0 | 0 | 0 | 17 |
| 400 | 1.00 | 1.00 | 0.62 | 0 | 0 | 0 | 24 |
| 500 | 1.00 | 1.00 | 1.00 | 0.82 | 0.02 | 0 | 30 |
| 600 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 0.08 | 36 |

- **The function for the freeze**: T(80% by tickets) ≈ 0.49 x k x C, k the kills per
  minute of both teams, C the clock in minutes (0.488 with P-51D/Heavy, 0.494 here; it is
  just under k C / 2 because a symmetric mirror's leader is barely ahead and the kill count
  is far tighter than Poisson). After the freeze: rerun `bench.mode_dynamics deathmatch`
  (8 s per 256 games warm on the box) and read k and T off the report.
  Today's 25 tickets in 600 s: every game ends by tickets (at about 7 min).
