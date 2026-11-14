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
