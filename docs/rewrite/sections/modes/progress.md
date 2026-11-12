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

## Step 2e: Frontline capture dynamics, 5v5, release lineup (rough; provisional zone sizes)

`--label fl_5v5_yj` (real_scale_frontline, no fields, zones kept, 2 Yongma + 3 Jujak, scripted
mirror, 512 games; zone radius 330 m x map scale 3 = 990 m, capture 20 s, threshold 3,
600 s):

- Ended by the front (the final capture): **0.090 ± 0.013**; 91% run out the clock (target
  ~80% by the front).
- Captures per game: mean 10.8, median 11 (5-95%: 5-16); the first at a median 87 s
  (quartiles 80-124 s).
- **P(next capture by the same team | a capture) = 0.143 ± 0.005** (5021 pairs), P(reversal)
  0.857: strongly anti-persistent, the degenerate end, not a random walk. The September v1
  study (docs/engineering/frontline-density-and-front-dynamics.md on the v1 tree; memory
  note) found P(reversal) = 0.84 and a 300 s clock cutting ~90% of 5v5 games: unchanged in
  v2 at the real scale's 600 s. The cause it named is structural (the role rotation makes
  the zone just captured the loser's next target, with its defenders already inside), so
  clock and capture-time knobs alone may not fix it. Knob scan running (16:10 KST, ETA
  ~16:40): capture 10 s / 40 s, threshold 2, zone radius 495 / 660 m (1.5x / 2x), clock
  1200 s.
- Zone sizes in use: `ModeRules.zone_radius` 330 m at map scale 1, scaled by R / 2600 m:
  990 m on the 7800 m real-scale map (0.127 R), 1400 / 1980 / 3131 m on the 10v10 / 20v20 /
  50v50 `_density` maps; the ring at 0.615 R. A zone covers 1.6% of the arena's area.

## Step 2f: Frontline knob scan (rough, 256 games each, release lineup)

One knob changed from the 5v5 baseline (990 m zones, capture 20 s, threshold 3, 600 s):

| variant | ended by the front | captures / game | first capture median (s) | P(same team next) |
|---|---|---|---|---|
| baseline (512 games) | 0.090 ± 0.013 | 10.8 | 87 | 0.143 ± 0.005 |
| capture 10 s | 0.191 ± 0.025 | 16.3 | 78 | 0.132 ± 0.005 |
| capture 40 s | 0.004 ± 0.004 | 1.8 | 289 | 0.125 ± 0.020 |
| threshold 2 | 0.445 ± 0.031 | 9.0 | 87 | 0.171 ± 0.008 |
| zone radius 1.5x (1485 m) | 0.082 ± 0.017 | 9.9 | 88 | 0.163 ± 0.008 |
| zone radius 2x (1980 m) | 0.098 ± 0.019 | 8.3 | 89 | 0.180 ± 0.009 |
| clock 1200 s | 0.281 ± 0.028 | 21.2 | 87 | 0.141 ± 0.005 |

- Tried: capture time, threshold, zone size, clock. Worked: none reaches ~80% by the front;
  threshold 2 gets closest (45%). Didn't: P(same team next) stays 0.13-0.18 under every
  knob: the reversal is structural, set by the role rotation and the agent's play, not by
  the knobs. Bigger zones nudge it toward 0.5 (0.14 → 0.18) and slow the game.
- Why (the rotation; zone indices round the ring 0-4): after team 0 takes team 1's
  defense zone, that zone becomes team 0's defense and so team 1's next target, with team
  1's ships still in it. Team 0's own next target is the old team-1 spawn, next door. The
  scripted agent's attackers move on to the new target and the losers' survivors retake
  the zone: so the measurement is the rules and the scripted doctrine together. A trained
  policy that holds what it took could play differently.
- Proposal (the author decides; not implemented): a rule that keeps a just-captured zone
  from being taken straight back, e.g. a lock (no meter movement in the captured zone
  for L seconds after a capture, L ~ capture time to 2x), or rotate the roles so the loser's
  next target is not the zone it just lost. Either is a rule change (game version). I can
  measure a lock in an hour if wanted. With a near-random walk (P ≈ 0.5) a symmetric walk
  needs ~9 captures to reach ±3, which at ~1 capture per 55 s fits a 600 s clock.

## Step 2g: the author's per-mode quantities (release lineup; tickets and clocks dropped)

The author (16:20): no more ticket/clock calibration until the ships are final; measure the
quantities below on the tree's default cards, to be checked against real-game bands and
re-measured after the freeze. Record `records/probes.jsonl` (per-ship probe histograms in
1 s bins; `bench/mode_dynamics_report.py`).

### Deathmatch (5v5, 7800 m, no fields, 2 Yongma + 3 Jujak a team, scripted mirror, 512 games, 600 s, tickets off)

| quantity | mean | median | quartiles | 5-95% |
|---|---|---|---|---|
| kills per minute (both teams) | 7.19 | 7.20 | 7.0-7.4 | 6.7-7.6 |
| match start to the first enemy hit (s) | 24.6 | 29.8 | 6.8-38.3 | 2.2-44.8 |
| match start to the first death (s) | 27.0 | 32.6 | 10.2-40.4 | 4.4-46.2 |
| time between deaths, any team (s) | 7.95 | 4.47 | 2.2-10.1 | 0.3-28.8 |
| a ship's spawn to its first hit dealt or taken (s) | 64.9 | 59.5 | 50.5-71.5 | 37.5-113.5 |
| a ship's spawn to its death (s) | 68.8 | 62.5 | 54.5-77.5 | 41.5-116.5 |

Trade share (deaths with an enemy death within the window): 0.140 at 0.5 s, 0.203 at
1 s, 0.347 at 2 s. Readings: the start is bimodal (a quarter of games see a hit within 7 s:
deathmatch's random spawns can land enemies close; the rest meet at ~30-40 s). A ship
lives ~1 min, almost all of it before its first hit: from first contact to death is a few
seconds (life median 62.5 s against first contact 59.5 s): the fights are short and
decisive; the time is travel. Includes the respawn's time to find a fight (respawns land
farthest from enemies).
