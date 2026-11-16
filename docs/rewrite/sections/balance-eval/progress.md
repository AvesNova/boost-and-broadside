# Balance eval: progress (2026-11-16)

Branch `feat/balance-eval` from `origin/feat/imugi-boundary` (not merged).

## 17:05 KST: the switches built, CPU checks pass

- Tried/built: `bench/balance_sweep.py` (conditions, `BALANCE` settings: no fog, no fields,
  seeded spawns, whole episodes; sweep candidates and any-card pairs; kill share pooled with
  kills and deaths a minute per side; check and speed subcommands) and `bench/balance_worker.py`
  (the balance-final worker on it, 32 variants per card table by default). `bench.game_loop`:
  static fog switch (the sight test skipped: `perceive(fog=False)`), seeded spawns
  (`seeded_uniforms`, by job game index), first-match tallies (`match_kills`,
  `match_deaths`, `match_ticks`), a pluggable controller factory, `Loop.set_table`, a scored
  batch stopping once every first match is over.
- Worked: CPU eager check on every condition tried: fogged 0/B, no fields present, every
  living row visible; spawns repeat under a seed, differ under another, follow the game index
  across batch positions. Quick tests (53) and ruff pass.
- Didn't: torch's CPU generator keeps only 32 bits of a seed (seed << 32 + game gave the same
  spawns for every seed); the spawns use numpy's seed sequence of (seed, game) instead.
- Next: the box is off (author); the GPU check on the laptop through laptop_job.sh.

## 17:10 KST: fog speed on the laptop (RTX 4070, power profile performance)

The box is off (author, ~16:55); the GPU checks run on the laptop. Deathmatch 5v5, density map,
B = 2048 games of 60 s (1800 ticks), 32 card variants, each setting its own compiled process,
two timed batches each under the GPU lock (`speed`; journals in `data/speed-*.journal.txt`):

| fog | fields | batch s | games/s | kills a game |
|---|---|---|---|---|
| off | off | 4.2, 4.2 | 486.8 | 11.7 |
| on | off | 4.4, 4.2 | 483.0 | 10.2 |
| off | on (the preset's 10) | 4.4, 4.3, 4.3 | 473.3 | 8.3 |
| on | on (the preset's 10) | 4.4, 4.4 | 470.5 | 7.7 |

- Fog costs under 1% with or without fields (487 -> 483 without, 473 -> 471 with); the
  fields cost about 3%; both together about 3.5% (the batches agree within about 2%, so the
  fog's part is at the noise). On this GPU, at this size, turning fog off buys little speed.
- The games change much more than the speed: kills a game 7.7 -> 10.2 -> 11.7.
- The only recompile is the known `carry.boundary_progress` stride guard on the second tick,
  inside the warm-up (outside the lock).

## 17:16 KST: the worker on a real job (laptop, deathmatch 5v5, 256 games, the whole 600 s)

`check_run.sh`: the worker (`bench.balance_worker`, B = 256, 32 variants) on one job of two
entries, seed 0, fog and fields off. src/ holds the v22 card on this branch (VERSION 22;
Jujak sustained turn 0.7647, Yongma shield 141.9). Warm-up 211 s, the cards measured in 9 s,
then 27 s a batch of 256 games of 600 s; no recompile inside the job (the one guard,
`carry.boundary_progress`'s stride, fires in the warm-up). Records:
`data/check-deathmatch-5v5.jsonl`, journal `data/balance-eval-worker.journal.txt`.

| entry | A kill share | A point share | kills/min A, B | deaths/min A, B | score (W/D) |
|---|---|---|---|---|---|
| Yongma team v Jujak team | **44.9 ± 0.2%** | 46.1 ± 0.2% | 5.45, 6.70 | 6.84, 5.86 | 11.7 ± 2.0% (27/6) |
| pair Jujak v Jujak (mirror) | 50.6 ± 0.3% | 50.5 ± 0.3% | 6.39, 6.24 | 6.58, 6.71 | 53.9 ± 3.1% |

- For scale (other settings, not comparable): balance-final's v22 at 300 s with fog and fields
  read a 46.2% kill share and a 24.8% score. A 600 s game with more kills turns the same small
  kill edge into a larger win edge.
- The mirror is 1.8 standard errors off 50%; the kill share's standard error is the ratio
  estimator's by game, and the ticks' randomness is not common across entries (only the
  spawns), so a mirror check at more games is worth one run before trusting 0.3-point errors.

## 17:22 KST: ready; open points for the author

1. The deathmatch 1v1 and 2v2 maps sit on the mode's 3000 m floor (`api.arena_radius_for`), so
   they are larger than pure density (1878 m and 2656 m). Kept as the game's rule; a pure
   density switch is one line in `balance_sweep.CONDITIONS` if wanted.
2. Elimination and Frontline play the preset's own map (7800 m) at 10v10 too, as the game does
   since 2026-11-15; the density map would be 11031 m. Say if they should scale.
3. The whole clock is the preset's: deathmatch is 600 s (balance-final swept 300 s), so a
   deathmatch candidate costs twice what it did.
4. Fog off changes the games (kills a game 10.2 -> 11.7 at 60 s, fields off) for under 1% of
   speed: the measurement is of a different game than the one played with fog.
5. Only the spawns are common across entries; respawns, spread and decisions draw from the
   default generator (seeded per chunk), so they differ by batch position.
6. A pair's cards fly their base ships' stored doctrines (Yongma's or Jujak's); a sampled card
   with no base ship needs one named.
7. A controller that derives settings from the card must read `loop.card` in the graph or
   define `refresh(card, types)` (`Loop.set_table` calls it after a new card table).

## 17:35 KST: round 2 (the author's decisions of 2026-11-16)

Decisions: fog back on (fields stay off); kill share primary, with a conversion to win rate;
maps as the game uses them; deathmatch 300 s (the cap 600 s; shorter only if measured to
give the same kill share at less cost); measure the error bars. `BALANCE` is now
`Settings(fog=True, fields=False, spawn_seed=0)` with `DEATHMATCH_SECONDS = 300`.

### Error bars (deathmatch 5v5, 300 s, fog on, fields off; laptop)

A Jujak and a Yongma mirror, 4 seeds x 512 games each (the worker at 1024 environments,
`round2_run.sh`; `error_bars.py`, `data/error_bars.md`, records
`data/mirrors-deathmatch-5v5-seed*.jsonl`). A mirror's truth is 50% (the same card on both
sides, seats alternating).

| metric | mirror | seeds: value ± se | mean se | sd between seeds / mean se |
|---|---|---|---|---|
| kill_share_a | Jujak mirror | 49.82 ± 0.36, 50.82 ± 0.35, 49.71 ± 0.36, 50.02 ± 0.37 | 0.36 | 1.39 |
| kill_share_a | Yongma mirror | 49.65 ± 0.23, 49.64 ± 0.22, 49.75 ± 0.24, 49.90 ± 0.23 | 0.23 | 0.52 |
| kill_share_a | both: mean z^2 against 50% over 8 | 1.58 (error scale 1.26) | | |
| point_share_a | Jujak mirror | 49.85 ± 0.34, 50.70 ± 0.33, 49.67 ± 0.34, 50.02 ± 0.34 | 0.34 | 1.34 |
| point_share_a | Yongma mirror | 49.64 ± 0.21, 49.64 ± 0.21, 49.77 ± 0.22, 49.88 ± 0.22 | 0.22 | 0.53 |
| point_share_a | both: mean z^2 against 50% over 8 | 1.59 (error scale 1.26) | | |
| kill_score | Jujak mirror | 49.96 ± 0.57, 50.69 ± 0.54, 49.32 ± 0.58, 49.99 ± 0.58 | 0.57 | 0.99 |
| kill_score | Yongma mirror | 49.57 ± 0.47, 49.72 ± 0.47, 49.44 ± 0.51, 49.77 ± 0.51 | 0.49 | 0.31 |
| kill_score | both: mean z^2 against 50% over 8 | 0.71 (error scale 0.84) | | |
| score | Jujak mirror | 49.61 ± 2.11, 55.27 ± 2.10, 48.34 ± 2.12, 48.44 ± 2.11 | 2.11 | 1.56 |
| score | Yongma mirror | 47.46 ± 2.06, 47.07 ± 2.05, 47.56 ± 2.07, 49.41 ± 2.05 | 2.06 | 0.51 |
| score | both: mean z^2 against 50% over 8 | 1.57 (error scale 1.25) | | |

- The per-game standard error holds: the spread between seeds is 1.39 and 0.52 times it
  (Jujak, Yongma: about 1 on average), and against the known 50% the mean z^2 over the 8
  values is 1.58 (kill share) and 1.57 (score), inside the 0.27-1.94 that 8 values give
  95% of the time when the error is right. Each game is an independent environment and the
  only shared draw is the spawn layout by game index, so a cluster or bootstrap error would
  equal the per-game one; not changed. 8 values cannot rule out an error up to about 1.4x too
  small; a 16-seed mirror would.
- To watch: the Yongma mirror reads under 50% on all four seeds (pooled 49.74 ± 0.12, 2.3
  sigma; Jujak 50.09 ± 0.18). Could be chance (one of two tests at 2.3 sigma), or a small
  seat effect; worth a 16-seed mirror before reading differences under half a point.
- The old per-game mean kill_score has about twice the error of the pooled kill share
  (0.57 against 0.36 points for Jujak), for the same games.
- A 1024-game job (2 mirrors x 512, 300 s) took about 18 s after the worker's 202 s warm-up.

### Conversion: score against kill share (balance-final records, card candidates)

`conversion_fit.py` on balance-final's data/ and data/box-out/ (`data/conversion_fit.md`):
logit(score) = a + b logit(kill_score), weighted, per condition; doctrine candidates left
out (with them deathmatch 5v5 fit at chi^2/dof 4.1; without, 0.31). kill_score there is the
per-game mean of the point share (deaths of any cause), not the pooled kills by hits.

| condition | cards | slope b | intercept a | R^2 | chi^2/dof | residual (score) | kill share for 45 / 50 / 55% | kill shares measured |
|---|---|---|---|---|---|---|---|---|
| deathmatch-1v1 | 3 | 0.99 | -0.034 | 0.997 | 0.00 | 0.0 pts | 45.8 / 50.9 / 55.9 | 31-32 |
| deathmatch-2v2 | 269 | 2.50 | -0.011 | 0.994 | 0.23 | 1.2 pts | 48.1 / 50.1 / 52.1 | 29-69 |
| deathmatch-5v5 | 373 | 7.62 | -0.009 | 0.992 | 0.31 | 1.3 pts | 49.4 / 50.0 / 50.7 | 41-61 |
| deathmatch-10v10 | 241 | 14.15 | -0.014 | 0.989 | 0.52 | 1.4 pts | 49.7 / 50.0 / 50.4 | 45-58 |
| elimination-2v2 | 3 | 1.43 | -0.088 | 0.939 | 0.91 | 0.9 pts | 48.0 / 51.5 / 55.1 | 28-35 |
| elimination-5v5 | 184 | 2.12 | -0.185 | 0.972 | 0.58 | 1.8 pts | 49.8 / 52.2 / 54.5 | 30-72 |
| elimination-10v10 | 75 | 3.00 | -0.109 | 0.877 | 0.82 | 2.2 pts | 49.2 / 50.9 / 52.6 | 43-56 |
| frontline-5v5 | 226 | 4.66 | -0.040 | 0.975 | 1.04 | 2.3 pts | 49.1 / 50.2 / 51.3 | 35-64 |
| frontline-10v10 | 202 | 6.73 | -0.015 | 0.948 | 2.06 | 3.4 pts | 49.3 / 50.1 / 50.8 | 39-64 |

- The win rate's 45-55% band is a narrow kill-share band, and narrower the larger the fight:
  deathmatch 2v2 48.1-52.1, 5v5 49.4-50.7, 10v10 49.7-50.4; Frontline 5v5 49.1-51.3, 10v10
  49.3-50.8; elimination 5v5 49.8-54.5, 10v10 49.2-52.6. Elimination's intercept is
  negative: at an even kill share the Yongma team loses (45% at 5v5), so its band sits above
  50.
- The 1v1 and elimination 2v2 rows have 3 cards each: not a fit.
- How far it carries over: those games ran with fog AND fields on; fields off adds kills
  (7.7 -> 10.2 a game in 60 s at 5v5 with fog), and more kills a game make a kill edge into a
  win edge faster, so the slopes here are likely a little shallow for the new settings and
  the bands a little wide; the centre (a) should carry. Refit on a few dozen cards measured
  at the new settings before using the bands to accept a card.

### Timing for compute estimates: laptop (RTX 4070 Laptop, power profile performance)

One steady batch each at the balance settings (fog on, fields off; deathmatch 300 s,
elimination and Frontline their own clocks, a batch stopping when every first match has
ended), 4096 environments, 32 variants, the base card's Yongma team v Jujak team; compiled
and warmed outside the lock (`bench.balance_sweep timing`, `data/timing.jsonl`,
`timing_table.py`). Frontline 10v10 not run on the laptop (stopped at 18:05 to keep the
laptop free for the author's flight; the 5090 has it). "game-s/s": first-match seconds
played a second (scored modes stop early, so their batch time follows the longest match).

| condition | clock s | match s (mean) | batch s | games/s | game-s/s | s a card (256 games) | GPU util | busiest threads | kill share Y v J | compile s |
|---|---|---|---|---|---|---|---|---|---|---|
| deathmatch-5v5 | 300 | 300 | 37.9 | 108.0 | 32.4k | 2.37 | 99% (71) | 100% / 0% | 44.5 ± 0.1 | 211 |
| deathmatch-10v10 | 300 | 300 | 79.6 | 51.5 | 15.4k | 4.97 | 99% (151) | 100% / 0% | 45.1 ± 0.1 | 223 |
| deathmatch-2v2 | 300 | 300 | 16.4 | 249.9 | 75.0k | 1.02 | 89% (30) | 100% / 0% | 35.5 ± 0.2 | 186 |
| deathmatch-1v1 | 300 | 300 | 15.1 | 271.0 | 81.3k | 0.94 | 80% (28) | 100% / 0% | 19.7 ± 0.6 | 177 |
| elimination-5v5 | 248 | 82 | 25.1 | 163.4 | 13.3k | 1.57 | 96% (47) | 100% / 0% | 36.2 ± 0.3 | 224 |
| elimination-10v10 | 248 | 74 | 54.8 | 74.7 | 5.5k | 3.43 | 99% (103) | 100% / 0% | 40.9 ± 0.3 | 186 |
| frontline-5v5 | 600 | 190 | 73.0 | 56.1 | 10.7k | 4.56 | 99% (138) | 100% / 0% | 38.3 ± 0.2 | 198 |

- GPU-bound in every condition: utilisation 96-99% at 5v5 and 10v10, 89% at deathmatch 2v2,
  80% at 1v1. The busiest thread at 100% is the main Python thread (launching and waiting
  on the GPU); every other thread is idle, and the GPU stays near full, so the batch is not
  launch-limited at these sizes. 1v1 is the one place a larger batch would gain.
- Seconds a card at 256 games: deathmatch 0.9 (1v1) - 5.0 (10v10), elimination 1.6 / 3.4,
  Frontline 5v5 4.6. Kill shares here (base card, fog on, fields off, seed 0) are the
  Yongma team's: 19.7 (1v1), 35.5 (2v2), 44.5 (5v5), 45.1 (10v10); elimination 36.2 / 40.9;
  Frontline 5v5 38.3.

### Timing for compute estimates: rented RTX 5090 (18:30 KST)

`ssh box5090` (vast.ai): RTX 5090 32 GB, driver 580.178.04, power limit 500 W (max 600),
max clocks SM 3105 / memory 14001 MHz, persistence on; Ryzen 7 7800X3D. After each batch
nvidia-smi read P1, SM 2542 MHz, memory 13801 MHz, 87-90 W, 32-36 C (`data/box5090/`).
The same `timing` at 4096 environments and 32 variants (the worktree at c42bb759: no
kills-by-minute tally), then deathmatch 5v5 and Frontline 10v10 at 16384:

| condition | clock s | match s (mean) | batch s | games/s | game-s/s | s a card (256 games) | GPU util | busiest threads | kill share Y v J | compile s |
|---|---|---|---|---|---|---|---|---|---|---|
| deathmatch-5v5 | 300 | 300 | 15.3 | 268.2 | 80.5k | 0.95 | 59% (28) | 100% / 0% | 44.5 ± 0.1 | 175 |
| deathmatch-10v10 | 300 | 300 | 17.6 | 233.2 | 70.0k | 1.10 | 84% (32) | 100% / 0% | 45.1 ± 0.1 | 165 |
| deathmatch-2v2 | 300 | 300 | 13.6 | 300.5 | 90.1k | 0.85 | 69% (25) | 100% / 0% | 35.5 ± 0.2 | 160 |
| deathmatch-1v1 | 300 | 300 | 13.0 | 314.6 | 94.4k | 0.81 | 35% (23) | 100% / 0% | 20.3 ± 0.6 | 150 |
| elimination-5v5 | 248 | 82 | 10.7 | 383.1 | 31.2k | 0.67 | 54% (19) | 100% / 0% | 36.2 ± 0.3 | 162 |
| elimination-10v10 | 248 | 74 | 12.4 | 330.7 | 24.6k | 0.77 | 86% (22) | 100% / 0% | 40.9 ± 0.3 | 160 |
| frontline-5v5 | 600 | 190 | 32.1 | 127.7 | 24.3k | 2.00 | 59% (60) | 100% / 0% | 38.3 ± 0.2 | 163 |
| frontline-10v10 | 600 | 175 | 34.4 | 119.1 | 20.8k | 2.15 | 86% (65) | 100% / 0% | 42.1 ± 0.1 | 162 |
| deathmatch-5v5 | 300 | 300 | 33.8 | 484.6 | 145.4k | 0.53 | 91% (57) | 100% / 0% | 44.6 ± 0.1 | 172 |
| frontline-10v10 | 600 | 175 | 110.3 | 148.5 | 25.9k | 1.72 | 96% (205) | 100% / 0% | 42.1 ± 0.1 | 174 |

- At 4096 the 5090 is launch-bound: GPU 35-60% at 1v1-5v5, 84-86% at 10v10, the main
  Python thread at 100%. Against the laptop it is 2.5x at deathmatch 5v5 and 4.5x at 10v10.
- At 16384 it fills: deathmatch 5v5 485 games/s (1.8x of 4096; GPU 91%), peak memory 1.4
  GiB; Frontline 10v10 149 games/s (1.25x; GPU 96%; a scored batch waits on its longest
  match). Workers on the 5090 should run 16384 environments (memory allows far more).
- Seconds a card at 256 games on the 5090 (4096 environments): deathmatch 0.81 (1v1) - 1.10
  (10v10), elimination 0.67 / 0.77, Frontline 2.00 / 2.15; at 16384, deathmatch 5v5 0.53 and
  Frontline 10v10 1.72. Compile 150-175 s per condition (laptop 177-224 s).
- The kill shares match the laptop's to the decimal at the same seed (deathmatch 5v5 44.5,
  10v10 45.1; 1v1 20.3 against 19.7, within its 0.6 error).
