# Front reversal: is the frontline front a random walk?

Measured 2026-11-15 on the branch `bench/front-reversal` (from `integrate/next-runs`, game as of
`eda3fc6c`).

## Question

A September 2026 measurement on v1 found the front anti-persistent: P(reversal) = 0.84. The
capture lock (`lock_seconds_per_root_km`, about 45 s at 5v5 real scale) was set so the team
that captured is about as likely as the other to capture next. Does the current game's front
behave like a random walk, P(reversal) = 0.5?

## Definitions

- **Front:** the signed capture count (`ModeState.front`, +1 for a team 0 capture, -1 for a
  team 1 capture). **Move:** one capture, including the winning one.
- **Reversal:** a move in the direction opposite to the previous move of the same game. A
  game's first move has no predecessor and is not counted.
- **P(reversal)** = reversals / pairs of consecutive moves, pooled over games. Intervals:
  Wilson 95%, and a game-clustered bootstrap 95% (4000 resamples; moves in one game are not
  independent).
- Stopping at the win threshold does not bias the estimate: whether a pair exists depends only
  on earlier moves, so a fair coin still gives 0.5.

## Setup

`real_scale_frontline`, 5v5, front threshold 3, 600 s clock, Imugi on both seats
(T = 1, mirrored seats), fog on. Lineups mirrored per slot (both teams the same type in each
slot), alternating yongma,jujak,yongma,jujak,yongma and jujak,yongma,jujak,yongma,jujak in
equal numbers, so each ship type fills each slot half of the time. The bench runs with
`num_fields=0` (Imugi ignores fields). Each environment's first game only.
Run on the CPU (eager, no `torch.compile`: the CPU inductor path fails with a `KeyError` in
this torch build), 16 environments per batch, 32 games per lineup, about 6.3 minutes per
batch, 4 batches. The GPU was busy with a training run.

Command (script `bench/front_reversal.py`, games from `bench/mode_dynamics.py`):

    PYTHONPATH=src uv run --no-sync python -m bench.mode_dynamics frontline --device cpu --eager \
        --batch 16 --games 32 --lineup yongma,jujak --label yongma,jujak --seed 11 \
        --out docs/rewrite/sections/front-reversal/cpu.jsonl
    (likewise --lineup jujak,yongma --seed 12)
    PYTHONPATH=src uv run --no-sync python -m bench.front_reversal docs/rewrite/sections/front-reversal/cpu.jsonl

## Result (CPU, 64 games)

| quantity | value |
|---|---|
| games / moves | 64 / 272 (4.25 per game) |
| consecutive move pairs / reversals | 208 / 39 |
| **P(reversal)** | **0.188** |
| Wilson 95% | [0.140, 0.246] |
| game-clustered bootstrap 95% | [0.134, 0.238] |
| ended by capture (front reached 3) | 55 (85.9%) |
| ended on the clock (600 s) | 9 (14.1%) |
| mean game length | 296 s |

Moves per game: 3 moves in 42 games, 5 in 8, 6 in 1, 7 in 8, 8 in 1, 9 in 4. So 42 of 64
games (66%) are a straight 3-0 sweep; a fair coin gives 25% for the first three moves going
one way. Of the 9 clock-ended games, 3 were drawn at front 0 and 6 decided by the sign
of the front.

## Reading

On the current game the front is persistent, not a random walk and not anti-persistent:
P(reversal) is about 0.19, well below 0.5 and well below the v1 value of 0.84. After a
capture, the team that captured usually captures next, in spite of the capture lock. The
measurement is of Imugi against itself; the design bench that set the lock
measured the probability that the capturing team captures next as 0.5 under a different
condition, which this bench does not reproduce.

The interval half-width is 0.053 (Wilson), slightly above the 0.05 target, but the
distance from 0.5 is about six half-widths, so a larger sample would not change the
conclusion. For the clock-end tuning: 86% of games end by capture at a mean of under 5
minutes, 14% on the 600 s clock.

## Where the persistence comes from (same 64 CPU games)

P(next capture by the team that just captured) = 1 - P(reversal) = 0.812 (Wilson 95%
[0.754, 0.860]): the lock's tuning target, 0.5 in the design bench, is 0.81 here.
P(reversal) by the front's size before the move (`bench.front_reversal` prints this):

| front before the move | reversals / pairs | P(reversal) | Wilson 95% |
|---|---|---|---|
| 0 (after a return to 0) | 4 / 29 | 0.138 | [0.055, 0.306] |
| 1 (after a first capture, or a step back) | 20 / 105 | 0.190 | [0.127, 0.276] |
| 2 (one capture from winning) | 15 / 74 | 0.203 | [0.127, 0.308] |

The persistence is about the same at every front size, so it is not an effect of the win
threshold; the team that captured keeps the initiative at each step.

## GPU run with fields on: not run

The coordinator's revised follow-up (one GPU job of at most 20 minutes, the preset's fields on,
about 128 games) was prepared but skipped: the GPU still had compute processes other than PIDs
1180275 and 1180276 (2117566, 2117567, 2577381) at 01:39, the 01:40 deadline. The CPU games above ran with
`num_fields=0`; fields change the physics even though Imugi ignores them. The
bench now takes `--fields` (keeps the preset's fields) for this run:

    systemd-run --user --unit=front-reversal -p MemoryMax=12G --working-directory=$PWD \
        --setenv=PYTHONPATH=src bash -c 'for s in "yongma,jujak 31" "jujak,yongma 32"; do set -- $s; \
        timeout 1200 uv run --no-sync python -m bench.mode_dynamics frontline --fields --games 64 \
        --batch 64 --lineup $1 --label $1 --seed $2 --out docs/rewrite/sections/front-reversal/gpu_fields.jsonl; done'
