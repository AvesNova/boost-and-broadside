# Unified evaluator: laptop results (2026-11-17)

RTX 4070 laptop (36 SMs, 8 GB), branch `feat/unified-eval`. Policies: the dry run's
(`dry-v24-s1`, game v24), copied from the box: the champion (training exponential
average, decay 0.95, step 98,566,144), RL exponential averages at 5M and 20M, BC's last.
The box's reference is its queue on the 5090 (`/workspace/runs/dry-v24-s1-eval`, game v24):
the suite took 32 min for 6,656 games (25.7 min of batches, the rest compiles).

## Statistics: the same numbers as the queue

On a **v24 tree** (`4c6c7227` with the unified package copied in), bucket 1, the champion
against Imugi:

| condition | unified (laptop) | box queue's suite (`bnb evaluate`) |
|---|---|---|
| elimination 1v1 | 0.238 ± 0.015 (768 games) | 0.219 ± 0.018 (512) |
| deathmatch 1v1 | 0.625 ± 0.020 (512) | 0.612 ± 0.020 (512) |

Both within one standard error of the difference.

**Benched against native** (v25 tree, the same two cells): played in a 5-a-side bucket
(4 slots a side benched) against a 1-a-side engine:

| condition | bucket 5 (benched) | bucket 1 (native) | difference / SE |
|---|---|---|---|
| elimination 1v1 | 0.078 ± 0.010 | 0.070 ± 0.009 | +0.6 |
| deathmatch 1v1 | 0.250 ± 0.016 | 0.289 ± 0.017 | −1.7 |

No significant difference (the deathmatch one is worth a bigger look on the 5090).

**v25 against v24, a finding for the author:** the v25 tree's Imugi (the committed-turn
Imugi of `676bf90c`) beats the v24 champion far more often in small fights: deathmatch
1v1 0.29 (v25) against 0.62 (v24), elimination 1v1 0.07 against 0.24, elimination 2v2
0.27 against 0.30 (box). The 5v5 scores are unchanged (deathmatch 1.00, elimination
0.98, frontline 1.00). The policies were trained on v24, so this is Imugi's change.

## Throughput

Bucket 5, 1,024 environments (the 7 suite conditions of 5 a side or fewer, 512 games
each, plus a 5-player elimination Elo field to ±60):

- 37,000–45,000 games an hour once warm, the GPU at 100% on every progress line;
- the same 6,973 games (by condition) took the queue's 5090 engine 1,198 s of batches;
  the laptop played them in 675 s, so **the laptop runs 1.8x the 5090 queue's batch
  throughput on the same games**; at the measured 3.8x rollout ratio, the 5090 should
  run about 6.7x the queue's (most of it from not waiting for a batch's longest game:
  the frontline games here averaged 69 s of game time against the queue's 0.33 s a game);
- a bucket's cold compile was 3–5 min; warm from the cache, 26 s (bucket 1).

Sized from the memory (the second run, the eager probe reading high that time): 1,344
environments, 35,000–63,000 games an hour, the GPU at 100%, the host's tallying and
planning 0–1% of the wall time, the process at 5 GB of host memory. With the suite's
cells at their budgets, the elimination slice played **overtime** for the Elo field
(errors 15–25 against the goal of 60) while the deathmatch slice finished its long games,
and the frontline slice, its one cell done, parked.

Host cost of the planner (CPU, no GPU): 0.7 ms a game for the box check's 92 targets;
3.8 ms a game for three 41-player Elo fields (vectorised proposals and cached cell
hashes; 15 ms before), well under a block's time on either GPU.

**Two buckets sharing the device** (`--concurrent yes`; elimination 5v5 in bucket 5,
10v10, 3v6 and 6v9 in bucket 10, 256 games each): sized 640 and 832 environments by
demand, both played in one host loop with the GPU at 100%, each stopped when its games
were in (1,024 games, 10 minutes with both compiles). The second bucket compiles while
the first waits (a known cost: compiling one while another plays is not built).
Scores: 0.98 (5v5), 0.97 (10v10), 0.40 (3v6), 0.86 (6v9).

Bucket size costs: the same 1v1 cells played 2.7x faster in a 1-a-side engine than
benched in a 5-a-side one (89 s against 242 s for 1,280 games). The default buckets are
5, 10, 20; a spec heavy in 1v1 and 2v2 games gains from `buckets = [2, 5, 10, 20]` (one
more compile).

## Memory and sizing

Eager probes (after a throwaway one, with garbage collection between): 76 MB fixed plus
1.12 MB an environment at bucket 5, stable over three repeats (the first version, without
the throwaway probe, gave 0.8–5.3 MB and once sized 6,336 environments, which ran out of
memory). The compiled engine at 1,024 environments: 1.33 GB allocated (1.29 MB an
environment), 1.69 GB reserved (1.57 MB), so `COMPILED_FACTOR` = 1.5 over the eager
probe; at 70% of the free memory the laptop sizes bucket 5 at about 3,000 environments.

## Planner behaviour seen on the GPU

- The first version filled a slice whose goals were met with "filler" games (its last
  cell again): 3,037 games of one Elo pair. Replaced by **overtime**: such a slice plays
  the planner's values with every goal taken as unmet (budgets binding), else parks.
- A bucket stopped as soon as nothing more was wanted, abandoning the games in flight;
  it now plays until every game a target asked for has ended.
- The host's planning share of the wall time rounds to 0% on the laptop (`host planning`
  in progress.log).

## Decisions under the null (CPU simulation)

`InformationLooks` with K = 4, α = 0.05, games ending in random bursts of 1–12 (the
refill's), scored win/draw/loss: false-positive rate 0.050 ± 0.0015 without draws and
0.054 ± 0.0016 with 20% draws at a 213-game maximum (20,000 tests; the first look at 53
games, where the estimated variance runs small); 0.045 and 0.048 ± 0.002 at an 850-game
maximum (10,000 tests). Each look schedule's exact null crossing probability, from the
boundaries spent at its actual fractions, is α (the test `test_irregular_looks_...`).
A real decision (δ = 0.02) plans about 5,300 games.

## The 5090 check (Tuesday morning)

Once the box's GPU is free (the queue holds the lock until it exits):

1. A checkout of `origin/feat/unified-eval` at `/workspace/wt/ueval` (as the queue's
   `/workspace/wt/evalq`; game v25: the dry run's policies are v24, so 1v1 and 2v2
   scores against Imugi will read lower than the queue's, as on the laptop).
2. Optionally `CACHE_ARCHIVE=/workspace/compile-cache.tar.zst` (a 5090 archive) to start
   warm, and `CACHE_EXPORT=...` to keep this run's compiles for the next box.
3. `nohup setsid bash /workspace/wt/ueval/box/unified_check.sh dry-v24-s1 2 > ... &`
   (the queue's targets: suite ±0.02, maps ±45 (dropped 2026-11-17: evaluation plays the trained map sizes), curves ±60 on the queue's own field,
   break-even ±0.1 at most 2,048 games a mode, team size n v n, n v 1.5n, n v 2n ±0.05).

What to read in `$RUNS/dry-v24-s1-unified/`:

- `sizing.json`: each bucket's environments; expect about 4x the laptop's (bucket 5: the
  laptop sizes about 1,300–3,000), and the peak reserved within 70% of 32 GB;
- `progress.log`: GPU above 90%, host planning a few percent at most; games per hour per
  bucket 3.8x (the training-rollout ratio) to 5x (SMs times clock) the laptop's
  35,000–63,000 at bucket 5;
- the whole queue's equivalent took 32 (suite) + 4 (maps) + 86 (curves) + about 40
  (break-even) + team size minutes of batches and compiles on the 5090; the laptop
  measured 1.8x the queue's batch throughput on the same games, so the 5090 should
  finish the same targets in roughly a fifth of the queue's time, plus 3 compiles;
- `report.json` against the queue's tables (`dry-v24-s1-eval/suite/summary.txt`,
  `curves/table.txt`, `break-even/`, `team-size/`), allowing for v25's Imugi in small
  fights.

## Elo curves to precision (2026-11-17)

Dry run 1's curves (the queue's `bnb pool rate`, 52 snapshots, ±60, 12,288 games a mode)
ended with 11 frontline players without a rating (9 bounds, one undetermined, one never
played) and 40 deathmatch ones (31 bounds, 9 never played), none in elimination; the late
RL snapshots after one or two sweeps of far weaker
opponents (`dry-v24-curves-table.txt`, the box's `table.txt`). The planner now never takes
a bound as a result (plan.md, "Elo: no bound is a result").

**Offline** (`bench/elo_curve_planner.py`, CPU): that table as the truth (a bound placed
300 Elo past its value, the undetermined player between its neighbours), Bradley-Terry
games, 1,024 games in flight ending in bursts of 2%, from no games:

| mode | span (Elo) | games to ±60 | old budget | players without a rating | worst SE | games beyond p = 0.95 / 0.99 (after the first wave) | mean p (1 − p) |
|---|---|---|---|---|---|---|---|
| elimination | 2,500 | 4,940 | 12,288 | 0 | 59.9 | 11% / 6% | 0.137 |
| deathmatch | 5,000 | 11,776 | 12,288 | 0 | 59.7 | 32% / 5% | 0.087 |
| frontline | 5,400 | 15,408 | 12,288 | 0 | 59.9 | 38% / 7% | 0.089 |

A player without a rating has at most 8 games in flight a pair and 32 in all until results
come in (`PROBE_PAIR`, `PROBE_PLAYER`): without the cap a refill round committed dozens of
games to a guess (on the laptop, 54 deathmatch games of an early BC snapshot against one
rated far above it, every one lost); the cap took 7–14% off the games above. The near-certain
games that remain are structural: frontline's field has a 700-Elo hole above the early
BC snapshots (−1,124 to −419), and ±60 on both sides of it needs games across it (of the
games beyond p = 0.99 after the first wave, 46% were a bound below against a rated player,
42% two rated players bridging the hole). Per game, the information about a distance is
largest at hops near 400 Elo (p ≈ 0.92), so a long span is chained in such hops, not at
0.5. The fitted errors match the truth (root mean square of error over SE 0.73–1.41,
one run each).
With draws (15% of even games, fewer as the odds part, the mean score kept at p), frontline
took 14,720 games, every player rated, error over SE 0.95.


## Dry run 1's curves completed on the laptop (2026-11-17, in progress)

The v24 tree with the unified package copied in (`ueval-runs/curves-v24`), the queue's pool
(`dry-v24-s1-eval/curves/pool`, 36,672 games: 12,288 frontline, 12,288 deathmatch, 12,096
elimination) loaded as prior (`--prior`), the queue's 52 snapshots, ±60, stopping on
precision (safety: 60,000 games a mode with the prior, 3 hours a launch). Elimination met
every goal from the prior alone (0 new games). Frontline: 11 players without a rating in the
prior, all rated after about 750 new games.

**Throughput.** The first launch played 4,000–9,000 games an hour against 37,000–63,000 in
the earlier laptop checks. Where the time went: those checks seated one or a few network
players (lanes 2,730 rows a tick for 2,688 rows); 52 players cut into two runs sized by
share padded the 4,096 rows to 8,996 a tick, and the 5v5 frontline and deathmatch games are
3–6 times longer in game time than the checks' 5v5 eliminations (about 220 and 370 s).
GPU 100%, host planning 0–1%, no recompiles. In game-seconds the first launch ran 905 a
wall second at steady state (about 27,000 env-steps a second at 30 ticks a game second);
the earlier checks 1,450–1,680 at 1,024–1,344 environments.

The resident set (plan.md) seats 12 players in 12 lanes of 393 rows (4,716 a tick):
52,600–57,400 env-steps a second on the same field (progress.log now prints env-steps a
second). Its first version parked environments (222 in 10 minutes: an environment holds lane
rows for its current and its next game), so an environment's next game keeps its players
when worth half the best; at a swap ratio of 2 no swap came in 30 minutes (the values run
flat over a curve's field), so it is 1.25. A direct A/B against all 52 seated was not
measured: that layout's cold compile was killed at the 8 GB host-memory cap.

**Deathmatch caveat (dry run 1):** dry run 1 trained deathmatch with 5·N tickets, but every
evaluation played the real rule, 4·N^(4/3) rounded to 5 (35 at 5v5, against 25 in
training). The author expects little effect, mainly in close games decided by a last kill
against the clock.


## Parking on the box's 108-player field (2026-11-17 noon)

The box's two-run field (dry-v26-v24-unified: 108 players, 12 residents in lanes of 3,092
rows, 16,128 environments in bucket 5) parked more games than it played ("2395 games ...
parked 3150, swaps 0"). Cause: the residents are seated by value, and the most valuable
players are those without a rating, which the probe caps hold to 32 games in flight
(`targets.PROBE_PLAYER`). In the v24 chain every player is without a rating at first: its
pool's games connect it to imugi_v24, not to the anchor. So 10 of 12 lanes ran at about 96
of 3,092 rows, nothing else fitted, and the environments parked. The swap rule saw the
capped residents as worth little and marked 3 leaving at once. Each then waited out its
current and next 5v5 games before its lane could take anyone, which is why no swap had
come yet.

Changes (planner and host only): `Planner.choose` takes a third tier, filler
(`Target.filler`: an Elo target's pairs with the probe caps lifted, flagged overtime),
before it parks. A resident set has a spare lane, so the incoming player is seated at once
while the outgoing one drains. A leaving resident still takes short games (elimination)
while a longer one is in flight. The choice reads only the seated players' cells, and the
filler values refresh every 64 games. progress.log prints the parked share of the
environments and of the ended games' ticks.

CPU simulation (`bench/unified_park_sim.py`: the real planner, targets and engine
assignment methods; the box's field rebuilt from dry run 1's pool, a renamed copy as the
v26 chain with the laptop's unified games as its prior; static slices 36/46/18%; game
lengths from the laptop's games; 600 game seconds):

| 16,128 environments | parked (environment time) | real games | real games an environment-second | swaps | host seconds |
|---|---|---|---|---|---|
| before (f7acfbb7) | 63% (70.7% at the end) | 30,693 | 0.0032 | 0 | 131 start + 899 |
| after | 0% | 49,555 | 0.0051 | 1 | 27 start + 337 |

At 2,048 environments: 49% parked and 4,897 real games in 900 s before; 0% and 9,849
after. An eager GPU smoke of the real engine (64 environments, 5 players, 2 residents)
swapped through the spare lane, emptied and reused a drained lane, with no overflow and no
missed games.

Open: a leaving resident still drains over its current and next long games (deathmatch
about 370 s each), so swaps come about one per drain. Its not-yet-started next games are
not re-planned, because the one-block lag means one may already have begun. More spare
lanes would allow concurrent swaps, at about 8% more padded network rows each.
