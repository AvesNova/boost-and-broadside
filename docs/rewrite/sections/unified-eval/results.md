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
   (the queue's targets: suite ±0.02, maps ±45, curves ±60 on the queue's own field,
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
| elimination | 2,500 | 5,280 | 12,288 | 0 | 59.8 | 11% / 5% | 0.137 |
| deathmatch | 5,000 | 13,680 | 12,288 | 0 | 60 | 32% / 4% | 0.087 |
| frontline | 5,400 | 16,960 | 12,288 | 0 | 60 | 39% / 4% | 0.093 |

The first wave (planned before any result) holds most of the near-certain games; after it,
the games beyond p = 0.95 are mostly the long hops A-optimality picks to chain a span of
5,000 Elo (per game, the information about a distance is largest near 400 Elo a hop,
p ≈ 0.92). The fitted errors match the truth (root mean square of error over SE 0.94–1.21).
With draws (15% of even games, fewer as the odds part, the mean score kept at p), frontline
took 14,720 games, every player rated, error over SE 0.95.

**Deathmatch caveat (dry run 1):** dry run 1 trained deathmatch with 5·N tickets, but every
evaluation played the real rule, 4·N^(4/3) rounded to 5 (35 at 5v5, against 25 in
training). The author expects little effect, mainly in close games decided by a last kill
against the clock; the curves' deathmatch field carries the note.
