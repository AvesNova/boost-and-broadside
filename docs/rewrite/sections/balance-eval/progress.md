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
