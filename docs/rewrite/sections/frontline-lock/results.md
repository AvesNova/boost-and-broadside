# Frontline lock re-tune: the front as a random walk

Branch `game/frontline-lock` (from `bench/front-reversal`). Goal (the author, 2026-11-15):
re-tune the capture lock so the front is a random walk, P(reversal) = 0.5, measured in the
training conditions (the `real_scale_frontline` preset with its fields on). Target: P(reversal)
in 0.45-0.55 with a 95% interval half-width of at most 0.05 at the chosen value, without many
more games ending on the clock.

Bench: `bench.mode_dynamics frontline` (scripted agent on both seats, T = 1, fog on, 5v5,
mirrored lineups yongma,jujak and jujak,yongma) and `bench.front_reversal` (definitions in its
docstring and in docs/rewrite/sections/front-reversal/results.md).

## Diagnosis

**What the lock does** (`frontline.locks_after`, `attacker_locked`; game v8): when a zone
falls, the team that lost it gains no progress on it for L s; its majority there does not
move the meter. After a capture that zone is the losing team's only attack target (the roles
rotate one place), so for L s the losing team can only defend. L = `lock_seconds_per_root_km`
x sqrt(d / 1 km); at 19 s/sqrt(km) and d = 5.64 km (real-scale 5v5: ring radius 0.615 x
7800 m, chord 2 r sin 36 deg) it is 45.1 s.

**What it was tuned against:** the frontline design bench (bench/frontline_design.py on the
tune/respawn-waves branch, 2026-11-13), as the hard lock that gave P(same team captures
next) = 0.5, before the final ship card and before the scripted agent's lock-aware doctrine
(`lock_press` 1, Baekho, 2026-11-13). Without a lock v1's front was anti-persistent
(P(reversal) 0.84): the losers stand on the zone they just lost and retake it as the
capturers leave for the next one.

**Why the capturing team keeps the momentum** (the 64 CPU games of the front-reversal
bench, no fields, lock 45 s):

- The geometry after a capture is symmetric: each team's new spawn is one zone from its new
  defense zone and two from its attack zone. The asymmetry is in time: the capturing team's
  survivors stand on the captured zone, one zone (5.6 km) from their next target, and the
  losing team cannot score at all for 45 s.
- The next capture by the same team comes a median 36 s after the previous one (quartiles
  28 / 36 / 59 s, n 169), mostly inside the 45 s lock; a reversal comes a median 91 s after
  (68 / 91 / 131 s, n 39), after the lock has run out. So the lock is longer than the
  capturing team needs to travel one zone and take it.
- The scripted agent plays the lock (`lock_press` 1 in its stored doctrines): while the
  enemy is locked out for longer than its travel back, it attacks; while its own team is
  locked out, it defends. The capture lock thus gives the side that just won a fight a free
  attack window against a team that just lost one (it usually has fewer living ships).
- It is not a threshold effect: P(reversal) is 0.14-0.20 at front 0, 1 and 2 alike.

So the cause is the lock's length relative to the travel-plus-capture time, and the most
direct knob is the lock's own parameter, `lock_seconds_per_root_km`; a shorter lock
shortens the window. At 0 the v1 behaviour (anti-persistence) is expected back, so the
random walk lies between 0 and 19.

Instrumentation added to `bench.mode_dynamics` (per batch, `lock_tallies`): while a team is
locked out, its living ships' time in the locked zone and in its own defense zone, and the
capturing team's time in its attack zone. `bench.front_reversal` now also prints the time to
the next capture by outcome and the mean game length.

## Progress

- 01:52: diagnosis written; instrumentation committed. GPU held by `rl-v20-seed1` until about
  02:00. Next: baseline with fields on (lock 19), 128 games, then a sweep of
  `lock_seconds_per_root_km`.

## Sweep and choice (2026-11-15, GPU, fields on, the scripted agent on both sides)

Real-scale frontline 5v5 with fog on and both mirrored lineups mixed. Each value was measured with `bench/mode_dynamics.py` and `bench/front_reversal.py`; Wilson 95% intervals.

| lock (s per sqrt km) | games | move pairs | P(reversal) | ended on the clock |
|---|---|---|---|---|
| 19 (was) | 128 | 387 | 0.292 [0.249, 0.339] | 43.0% |
| 14 | 128 | 415 | 0.378 [0.333, 0.426] | 48.4% |
| 10 | 128 | 464 | 0.418 [0.374, 0.463] | 51.6% |
| 5 | 128 | 503 | 0.511 [0.467, 0.554] | 57.8% |
| 0 | 128 | 552 | 0.574 [0.533, 0.615] | 68.0% |
| **6 (chosen)** | 512 | 1991 | **0.490 [0.468, 0.512]** (game-clustered bootstrap [0.466, 0.513]) | 56.4% |

**At 6** (about 14 s at 5v5):
- **P(reversal) by front position:** 0.528 at |0|, 0.500 at |1|, 0.435 at |2|. Near the win threshold, the leading team still keeps a little momentum.
- **The team that just captured takes the next capture** with probability 0.510.
- **Games** average 4.88 moves and last a mean of 484 s.
- **Clock ends: 56% of games end on today's 600 s clock.** That is far above the intended share, and left for the clock tuning.
- **Fields matter.** With fields off, the same 19 gave P(reversal) 0.19 and 14% clock ends (CPU, 64 games); fields on give 0.29 and 43%. Fields slow captures.

Game version 20. The fingerprint is unchanged: the 48 fingerprint ticks never reach a lock.
