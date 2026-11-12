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

### Frontline (5v5, 7800 m, zones 990 m, capture 20 s, threshold 3, 600 s; release lineup; 512 games; provisional zone sizes)

| quantity | mean | median | quartiles | 5-95% |
|---|---|---|---|---|
| a ship's spawn to its first zone other than its spawn zone (s) | 36.9 | 34.5 | 26.5-50.5 | 1.5-75.5 |
| leaving a zone to entering another (s), contested and not | 32.9 | 28.5 | 19.5-41.5 | 17.5-66.5 |
| contested capture: last all-zero meters to the capture (s) | 17.5 | 14.5 | 12.5-17.5 | 11.5-38.5 |
| first capture of the match (s) | 109.6 | 87.5 | 80-124 | 69-226 |
| a ship's spawn to its first hit dealt or taken (s) | 58.2 | 49.5 | 34.5-70.5 | 26.5-124.5 |
| a ship's spawn to its death (s) | 72.2 | 64.5 | 48.5-86.5 | 34.5-139.5 |

- Uncontested transit: the fastest zone-to-zone legs (5% 17.5 s, quartile 19.5 s) are the
  boosted uncontested ones, against the flight-only numbers (step 2a): adjacent zones edge
  to edge 3659 m, ~22 s at cruise, ~14-16 s boosting from the zone edge. The median 28.5 s
  includes legs broken by fights.
- Capture: a capture takes 14.5 s median once a meter starts moving (20 s for a lead of
  one; more attackers fill faster), and the first capture of a match comes at ~87 s.
- Capture dynamics: P(next capture by the same team) 0.143 ± 0.005; 9% of games end by
  the front (step 2e/2f).

### Elimination with side spawns (5v5, 7800 m, no fields, release lineup, scripted mirror, 512 games, 180 s)

`bench.mode_dynamics deathmatch --preset real_scale_elimination` (record
`records/elimination.jsonl`). With the teams' areas 12.5 km apart (spawn offset 0.8):

- First enemy hit at 36.7 s mean (median 36.6, quartiles 35.6-37.6): the fleets fly
  straight at each other and meet mid-map; first death 37.6 s median.
- 99.8% of games end by a wipe, at 47.5 s median (quartiles 46-55 s); essentially none run
  out the 180 s clock. The fight itself lasts ~10 s from first hit to wipe.
- Note: the trace misses the deaths of the tick a match ends on (that tick's state is the
  new match's), so per-game winners of a wipe are not read here; the timings are.
- If a shorter approach is wanted, `Rules.spawn_offset` (shared with the sandbox formation)
  or a separate elimination offset sets it: at 0.5 the areas are 7.8 km apart, ~22 s to
  contact. The author's call.

## Step 3a: density sweep, fixed arena (deathmatch, release lineup)

`bench/density_sweep.py --lineup jujak,yongma --temperature 3` (real_scale_deathmatch, no
fields, 7800 m, tickets 5 per ship, 600 s; the scripted agent at T = 1 against itself at
T = 3; record `/workspace/results/density_yj/density.jsonl` on the box, copied to
`records/density_yj.jsonl` at the end). Rung choice: on this lineup T = 2 scores only
0.575 ± 0.022 at 5v5, T = 3 0.708 ± 0.020 (512 games), close to T = 2's 0.70 on the old
lineup; T = 3 is the sweep's weaker rung.

| per side | ships / km² | score (T = 1 vs T = 3) | games |
|---|---|---|---|
| 5 | 0.052 | 0.699 ± 0.029 | 256 |
| 10 | 0.105 | 0.832 ± 0.023 | 256 |
| 20 | 0.209 | 0.881 ± 0.020 | 256 |
| 35 | 0.366 | 0.957 ± 0.013 | 256 |
| 50 | 0.523 | 0.961 ± 0.012 | 256 |

No collapse up to 50v50 on the 7800 m map (10x the 5v5's density): the edge grows with the
fleet, as a per-ship edge summed over more ships and more kills (the tickets grow with the
fleet) should. The v1 hypothesis is not reproduced in this range. Next: the `_density`
radii (running), then much denser cells (20 and 50 a side at 3900 m and 1950 m; 5v5 there
too) to find a critical density if there is one. ETA ~18:00 KST (worst 18:30): each new
size or radius compiles for 5-10 min on the box.

## Step 3b: density sweep at the `_density` radii (R = 7800 sqrt(n / 5))

| per side | R (m) | ships / km² | score (T = 1 vs T = 3) |
|---|---|---|---|
| 5 | 7800 | 0.052 | 0.699 ± 0.029 |
| 10 | 11031 | 0.052 | 0.785 ± 0.026 |
| 20 | 15600 | 0.052 | 0.865 ± 0.021 |
| 35 | 20637 | 0.052 | 0.902 ± 0.018 |
| 50 | 24666 | 0.052 | 0.924 ± 0.016 |

At equal density the edge also grows with the fleet, a little less than on the fixed map
(0.83 / 0.88 / 0.96 / 0.96 at 10 / 20 / 35 / 50): crowding helps the better side, if
anything. Dense cells (3900 m, 1950 m) running.

## Density record (durable; for the paper)

**Setup.** Branch `tune/modes` at 7cb29748 (game version 7; the tree's default real-scale
cards of 2026-11-12, before the combat freeze). `bench/density_sweep.py` on the league
match engine (`bnb.league.matches.Arena`, compiled; RTX 5060 Ti box). Game: the
`real_scale_deathmatch` preset (team deathmatch, respawns 10 s, tickets 5 per ship a side,
i.e. 5n for n a side, clock 600 s) with **no fields**, fog **on** (the engine's pools: fog
share 1), n ships a side, arena radius R as listed. Lineup: both teams Jujak, Yongma,
Jujak, Yongma, ... by slot (2 Yongma + 3 Jujak at 5v5). Players: the scripted agent at
T = 1 (`scripted`) against itself at T = 3 (`rung_T3`), every first game of 256
environments a cell, the scripted agent on team 0 in half. Score = (wins + ties / 2) /
games for T = 1; ± its standard error. Records: `records/density_yj.jsonl` (one line per
batch with wins/ties/losses and per-side kills, deaths, shots, hits, living ticks),
`records/density_calib_yj.jsonl` (T = 3 at 5v5, 512 games: 0.708 ± 0.020),
`records/density_calib_default_lineup.jsonl` (the P-51D/Heavy lineup's T = 1.5 / 2 / 3).

**Fixed map, R = 7800 m (the 5v5 training map):**

| n a side | ships / km² | score | games |
|---|---|---|---|
| 5 | 0.052 | 0.699 ± 0.029 | 256 |
| 10 | 0.105 | 0.832 ± 0.023 | 256 |
| 20 | 0.209 | 0.881 ± 0.020 | 256 |
| 35 | 0.366 | 0.957 ± 0.013 | 256 |
| 50 | 0.523 | 0.961 ± 0.012 | 256 |

**Constant density, R = 7800 sqrt(n / 5) m (the `_density` fleet presets):**

| n a side | R (m) | score | games |
|---|---|---|---|
| 5 | 7800 | 0.699 ± 0.029 | 256 |
| 10 | 11031 | 0.785 ± 0.026 | 256 |
| 20 | 15600 | 0.865 ± 0.021 | 256 |
| 35 | 20637 | 0.902 ± 0.018 | 256 |
| 50 | 24666 | 0.924 ± 0.016 | 256 |

**Dense maps:**

| n a side | R (m) | ships / km² | score |
|---|---|---|---|
| 5 | 3900 | 0.21 | 0.797 ± 0.025 |
| 20 | 3900 | 0.84 | 0.898 ± 0.019 |
| 50 | 3900 | 2.09 | 0.895 ± 0.019 |
| 5 | 1950 | 0.84 | 0.742 ± 0.027 |
| 20 | 1950 | 3.35 | 0.820 ± 0.024 |
| 50 | 1950 | 8.37 | 0.875 ± 0.021 |

**Reading.** No collapse toward 50% anywhere in 0.05-8.4 ships/km² (160x the training
density): the scripted agent's edge over its T = 3 rung grows with the fleet at every
radius, as a per-ship edge summed over more ships and more kills should. Density costs a
little at the extreme: at 1950 m (the arena preset's size, ~3 turn circles across) each
fleet's edge is 0.05-0.09 below its 3900 m one, and at 20 a side 7800 → 3900 → 1950 m gives
0.88 → 0.90 → 0.82. No critical density sets a floor on the map size in this range. The
v1 hypothesis (an edge collapsing to a spawn lottery past a critical density) does not
reproduce with this agent, these cards and this game.

## Step 2h: Frontline candidate mechanics (fog off, no fields, release lineup, 256 games each)

`bench/mode_dynamics.py frontline --fog 0` plus the candidate (record
`records/candidates.jsonl`). The candidates are bench-only experiments (`Experiments`:
`capture_step` and the resolve output wrapped inside the compiled tick); none is in the
game. 5v5 at 7800 m, zones 990 m, capture 20 s, threshold 3, respawn 16 s, 600 s.

| candidate | P(same team next) | ended by the front | captures / game | first capture median (s) |
|---|---|---|---|---|
| baseline, fog off | 0.139 ± 0.007 | 0.090 ± 0.018 | 9.4 | 90 |
| (a) lock 20 s | 0.218 ± 0.010 | 0.180 ± 0.024 | 7.6 | 90 |
| (a) lock 40 s | 0.701 ± 0.017 | 0.598 ± 0.031 | 4.0 | 90 |
| (a) lock 60 s | 0.781 ± 0.016 | 0.645 ± 0.030 | 3.7 | 90 |
| (b) respawn pause 10 s | 0.173 ± 0.009 | 0.090 ± 0.018 | 8.7 | 90 |
| (b) respawn pause 20 s | 0.248 ± 0.010 | 0.180 ± 0.024 | 8.9 | 90 |
| (c) rear respawn 10 s | 0.233 ± 0.009 | 0.156 ± 0.023 | 10.4 | 90 |
| (c) rear respawn 20 s | 0.254 ± 0.009 | 0.191 ± 0.025 | 10.6 | 90 |
| (d) blast 1.0 zone radius | 0.158 ± 0.008 | 0.094 ± 0.018 | 8.9 | 90 |
| (d) blast 1.5 zone radii | 0.202 ± 0.009 | 0.184 ± 0.024 | 8.0 | 90 |
| (e) respawn 32 s (x2) | 0.281 ± 0.011 | 0.266 ± 0.028 | 7.9 | 88 |
| (e) respawn 8 s (x0.5) | 0.054 ± 0.004 | 0.023 ± 0.009 | 11.4 | 127 |
| 1v1 (R 3488 m) | - | 0.000 | 0.0 | - |
| 2v2 (R 4933 m) | 0.335 ± 0.033 | 0.008 ± 0.006 | 1.4 | 271 |

(f) one contested centre zone with two neutral zones: not run (needs a new role table and
rotation: real code). The first capture is the same in every variant whose rule acts only
after a capture (same seeds, same play until then).

- **Worked:** the capture lock, and only the lock, moves the walk past 0.5: between 20 s
  (0.22) and 40 s (0.70). Refining at 25 / 30 / 35 s and lock 30 s at 1v1, 2v2 and 10v10
  (queued, ETA ~17:45 KST).
- **Partly:** a longer respawn (x2: 0.28, 27% by the front) and the pause and rear spawn
  (0.17-0.25): they slow the loser's return, but the zone is still retaken. A short
  respawn (8 s) makes it worse (0.05).
- **Small fleets:** at 1v1 the scripted agent never captures (0 captures in 256 x 600 s)
  and at 2v2 barely (1.4 a game): a lone attacker fights the defender instead (the combat
  layer takes over near an enemy) and a capture needs a strict majority in the zone.
  Frontline as it stands does not work below ~5 a side with this agent.
- **Doctrine (is the reversal the agent's?):** the default set against itself with one side
  more defensive: the default wins 0.619 ± 0.026 against defensive_floor 0.9 (from
  0.298) and 0.934 ± 0.013 against defensive_margin 1.2 (from 0.378); aggression 0.9 or
  3.6 (from 1.81) changes nothing (0.52 ± 0.03, identical captures: the attack rule does
  not bind). Holding ships back on defense loses: with these rules capturing is simply
  easier than defending, so the reversal is the rules', not a doctrine mistake.

## Step 2i: the capture lock, refined (fog off, 5v5 unless noted, 256 games each)

| lock | P(same team next) | ended by the front | captures / game |
|---|---|---|---|
| none | 0.139 ± 0.007 | 0.090 ± 0.018 | 9.4 |
| 20 s | 0.218 ± 0.010 | 0.180 ± 0.024 | 7.6 |
| 25 s | 0.389 ± 0.014 | 0.320 ± 0.029 | 5.8 |
| **30 s** | **0.546 ± 0.016** | 0.387 ± 0.030 | 4.8 |
| 35 s | 0.630 ± 0.016 | 0.520 ± 0.031 | 4.4 |
| 40 s | 0.701 ± 0.017 | 0.598 ± 0.031 | 4.0 |
| 60 s | 0.781 ± 0.016 | 0.645 ± 0.030 | 3.7 |
| 30 s, 1v1 (3488 m) | - | 0.000 | 0.0 |
| 30 s, 2v2 (4933 m) | 0.578 ± 0.044 | 0.012 ± 0.007 | 1.1 |

The author picked the lock direction (17:35). Queued on the box (ETA 18:05 KST, worst
18:25): lock 30 s with threshold 2 / clock 900 s / clock 1200 s (toward ~80% by the
front), then the two softer variants against the hard 30 s lock: (a) a decaying hold
(H = N = 30 s and 60 s: an extra H s of meter on the captured zone, decaying over N s, and
drained on top by an attacking majority at the capture rate, H_n s a second), (b) a flat
extension (E = 10 / 20 / 30 s: no decay, drained only by an attacking majority). Checked
on a synthetic zone: a lone attacker retakes a flat-10 zone in 30.0 s (10 + 20).

- Lock 30 s at 10v10 (R 11031 m, the `_density` map): P(same) **0.066 ± 0.005**, 4.7% by the
  front, 11.8 captures a game: the 5v5 lock does not carry over. The map is 1.41x larger
  (zones and travel too), and twice the ships retake faster: a lock in seconds is tied to
  one map and fleet size; it likely has to scale (with the travel time across the ring,
  i.e. R, and with the fleet). To measure if the author keeps the lock: lock 30 x 1.41 and
  60 s at 10v10.

## Step 2j: elimination spawn spread (fog off, no fields, release lineup, 256 games each)

`bench.mode_dynamics deathmatch --preset real_scale_elimination --fog 0 --set ...`
(record `records/elimination.jsonl`). The area must stay inside the arena, so the wider
areas move their centres in; `e_offset0.68` separates the two effects.

| variant | area (R) | centre (R) | first hit median (s) | first hit to wipe median (s) [quartiles] | wiped |
|---|---|---|---|---|---|
| base (v7) | 0.15 | 0.80 | 41.1 | 11.4 [10.0-15.0] | 1.000 |
| heading jitter ±30° | 0.15 | 0.80 | 41.3 | 11.6 [10.1-16.7] | 1.000 |
| centre only moved | 0.15 | 0.68 | 35.2 | 12.1 [10.5-18.9] | 1.000 |
| spread 1.5x | 0.225 | 0.75 | 36.6 | 15.9 [14.1-19.3] | 1.000 |
| spread 2x | 0.30 | 0.68 | 30.5 | 22.4 [19.5-27.0] | 1.000 |

Wider spacing makes the fight less sharp: from first hit to wipe 11.4 → 15.9 → 22.4 s
(the centre alone: 12.1 s), every game still a wipe. The ±30° jitter changes nothing
measurable (it stays inside the author's 30° look). With fog off the first hit comes ~4 s
later than with fog on (36.6 s).

## Step 2k: lock 30 s toward ~80% by the front (5v5, fog off, 256 games each)

| variant | P(same team next) | ended by the front | median length (s) | captures / game |
|---|---|---|---|---|
| lock 30, threshold 3, 600 s | 0.546 ± 0.016 | 0.387 ± 0.030 | 600 | 4.8 |
| **lock 30, threshold 2, 600 s** | **0.553 ± 0.020** | **0.793 ± 0.025** | 342 | 3.4 |
| lock 30, threshold 3, 900 s | 0.531 ± 0.014 | 0.594 ± 0.031 | 750 | 6.3 |
| lock 30, threshold 3, 1200 s | 0.527 ± 0.012 | 0.688 ± 0.029 | 750 | 7.2 |

Lock 30 s with threshold 2 meets both targets at 5v5 (a near-random walk, 79% by the
front) in the 600 s clock; threshold 3 would need a clock well past 1200 s.

## Step 2l: softer holds against the hard lock (5v5, fog off, 256 games each)

| variant | P(same team next) | ended by the front | captures / game |
|---|---|---|---|
| hard lock 30 s | 0.546 ± 0.016 | 0.387 ± 0.030 | 4.8 |
| (a) soft hold H = N = 30 s | 0.159 ± 0.008 | 0.121 ± 0.020 | 8.6 |
| (a) soft hold H = N = 60 s | 0.541 ± 0.016 | 0.465 ± 0.031 | 4.7 |
| (b) flat extension E = 10 s | 0.193 ± 0.009 | 0.121 ± 0.020 | 7.8 |
| (b) flat extension E = 20 s | 0.321 ± 0.013 | 0.262 ± 0.027 | 6.2 |
| (b) flat extension E = 30 s | 0.518 ± 0.016 | 0.406 ± 0.031 | 4.7 |

Soft hold (decaying over N with drain): H = N = 60 s ≈ the hard 30 s lock (0.54); 30 s is
far too weak (an attacking majority drains it in seconds and it decays meanwhile). The
flat extension reaches a random walk at E ≈ 30 s (0.52). Either softer form needs about
twice the hard lock's nominal time (soft) or the same (flat) for the same walk.

## Step 2m: the lock by distance (λ s/km of the adjacent zones' centre distance d)

d = 2 x 0.615 R sin 36° = 0.723 R: 5.64 km at 5v5 (7800 m), 7.98 km at 10v10 (11031 m),
11.28 km at 20v20 (15600 m). λ = 30 s / 5.64 km = 5.3 s/km; κ = 20 s / 5.64 km = 3.5 s/km
(the author's parametrisation, 17:55).

| map | lock (s) | capture (s) | P(same team next) | ended by the front | captures / game |
|---|---|---|---|---|---|
| 5v5, d 5.64 km | 30 (λ 5.3) | 20 | 0.546 ± 0.016 | 0.387 ± 0.030 | 4.8 |
| 10v10, d 7.98 km | 30 (λ 3.8) | 20 | 0.066 ± 0.005 | 0.047 ± 0.013 | 11.8 |
| 10v10 | **42 (λ 5.3)** | 20 | **0.476 ± 0.014** | 0.543 ± 0.031 | 6.0 |
| 10v10 | 60 (λ 7.5) | 20 | 0.894 ± 0.012 | 0.918 ± 0.017 | 3.7 |

One λ ≈ 5.3 s/km gives a near-random walk at both 5v5 and 10v10 (0.55 and 0.48) with the
capture time unscaled: no fleet term is needed at 10v10 within ±0.03. The walk is steep in
λ (10v10: 3.8 → 5.3 → 7.5 s/km gives 0.07 → 0.48 → 0.89), so λ must be set carefully and
re-measured after the freeze. Running (ETA 18:25 KST, worst 18:45): 10v10 with κ too
(lock 42.3 s, capture 28.2 s), 20v20 (lock 59.8 s, capture 39.5 s), and 10v10 κ-scaled with
threshold 2.

With the capture time scaled too (κ = 3.5 s/km, λ = 5.3 s/km, threshold 3, 600 s; 256 games):

| map | lock / capture (s) | P(same team next) | ended by the front | median length (s) | first capture median (s) |
|---|---|---|---|---|---|
| 5v5, d 5.64 km | 30 / 20 | 0.546 ± 0.016 | 0.387 ± 0.030 | 600 | 90 |
| 10v10, d 7.98 km | 42.3 / 28.2 | 0.585 ± 0.016 | 0.441 ± 0.031 | 600 | 103 |
| 20v20, d 11.28 km | 59.8 / 39.5 | 0.736 ± 0.016 | 0.785 ± 0.026 | 339 | 127 |
| 10v10, threshold 2 | 42.3 / 28.2 | 0.594 ± 0.021 | 0.844 ± 0.023 | 320 | 103 |

At equal λ and κ the fleet still shifts the walk toward persistence: 0.55 → 0.59 → 0.74 at
5 → 10 → 20 a side (the 10v10 point with an unscaled capture time was 0.48). So a fleet term
is needed past ~10 a side: λ (or κ) falling with the fleet, e.g. λ_n = 5.3 (5 / n)^β s/km;
two points fit β ≈ 0.3-0.4 at 20v20 by the steepness seen at 10v10 (3.8 → 5.3 s/km took
P 0.07 → 0.48), not measured. Not searched further (the author's scope: rough values now,
calibration after the freeze).

# Handoff (2026-11-12 18:30 KST)

Branch `tune/modes`, worktree `/home/vizia/avesnova/bnb-tune-modes` (the `bnb-modes` path
is an older `section/modes` worktree, untouched). All measurements: the tree's default
real-scale cards (game v7) before the combat freeze; re-measure after it.

**In the game (committed, quick-tier tests, ruff clean):**
- Game v7: elimination spawns by team side (`modes.spawns.side_placement`; area 0.15 R at
  `Rules.spawn_offset` 0.8 R on the spawn axis, nose toward the enemy area ±15°, edge
  rule kept), two live settings (`mode.side_area_fraction`, `mode.side_heading_jitter`),
  tests in `tests/sections/test_elimination.py`. Fingerprint `2581fdc8196942e9`.
- Frontline with no fields no longer crashes (`frontline.mirror_fields`; test
  `test_frontline_plays_with_no_fields`); numbers with fields unchanged.
- Known and not mine: `test_modes.py` has three failures on the base already (the respawn
  farthest-candidate test predates v4's headings; the shield-events tests see no recharge
  since v5).

**Benches (bench/):** `zone_travel.py` (flight-only zone legs), `density_sweep.py` (the
league engine, a rung against the agent by fleet and radius, lineup option),
`mode_dynamics.py` (compiled rollout with a trace stage and a per-ship probe inside the
resolve: deathmatch kills/trades/timings, Frontline captures/travel/contest, elimination;
`--fog`, `--set key=value`, `--lineup`, and the candidate Frontline rules as bench-only
wrappers: `--lock-seconds`, `--hold-seconds`/`--hold-decay-seconds`, `--pause-seconds`,
`--rear-seconds`, `--blast-radius`, `--b-parameters`), `mode_dynamics_report.py`. Records in
`records/`. On the box, a 256-game batch of 5v5 takes 8-60 s warm; a new fleet size or a
new rule recompiles (1-10 min).

**Rough values for the author's decision (to re-measure after the freeze):**
- Frontline: the missing mechanic is holding a taken zone. A hard capture lock of
  λ ≈ 5.3 s/km of the adjacent-zone distance (30 s at 5v5) gives a near-random walk
  (P(same) 0.55; was 0.14) and with threshold 2 ends 79% of 5v5 games by the front in
  600 s. Softer: a flat extension of ~30 s (0.52) or a decaying hold H = N = 60 s (0.54).
  Capture time κ ≈ 3.5 s/km. A fleet term is needed past ~10 a side (20v20: 0.74 at the
  same λ, κ). Below ~5 a side Frontline does not work with this agent (1v1: no captures).
- Deathmatch (5v5, release lineup): 7.2 kills/min, trades 14 / 20 / 35% within
  0.5 / 1 / 2 s; tickets left to the author after the ships are final (T for 80% by
  tickets ≈ 0.49 x kills/min x clock minutes, for a mirror).
- Elimination: wipes in ~100% of games; spacing sets the fight's length (first hit to wipe
  11 → 16 → 22 s at 1x / 1.5x / 2x area); the offset stays (the author).
- Density: no critical density in 0.05-8.4 ships/km²; the scripted edge grows with the
  fleet (durable record above). The map-size rule is then free of a density floor: set
  it by travel (zone legs and time to first contact scale with R) and by the position
  codes' range.

**Next (after the freeze):** re-run `mode_dynamics` (deathmatch, Frontline baseline and
the chosen hold at 5v5 / 10v10 / 20v20) and the elimination spread on the frozen cards;
implement the chosen hold as a rule (game version), with λ, κ in s/km and a fleet term.
