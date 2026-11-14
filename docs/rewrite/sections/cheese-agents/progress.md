# Cheese agents: progress

Branch `feat/cheese-agents` (from `origin/feat/seed-isolation`). CPU only.

## Tried / worked

- 20:15 `bnb.controllers.scripted_cheese`: six stateless agents (zone_rusher, turtle,
  evader, deathball, jouster, kiter) as one controller `Cheese`, reusing the chaser's
  bearing-to-stick rule (`stick_toward`, factored out of `Chaser`). No random draws.
- 20:30 tests: shapes per mode, hidden/dead enemies ignored, determinism, behaviours, a short
  CPU game of each per mode.
- 20:40 `scripted:<name>` players for `bnb head-to-head` and `bnb ratings` (match seat takes a
  per-row cheese kind).
- 20:50 league entries (`LeagueConfig.cheese`). Author change at 20:55: ON by default (all six),
  `cheese=()` turns them off; `bnb ratings` plays them by default (`--no-cheese`). The league
  tests' base config sets `cheese=()` so they keep testing the base pool.

## Didn't / caveats

- Sampling weights across the cheese entries use the existing gauge rule as is
  (`p (1 - p)` on the predicted win probability plus the floor); provisional. They start at the
  live rating (p = 0.5, the largest weight) until the league's games rate them.
- Not run: any GPU job, the exploit-check command, the full suite (author's rule).
- Jouster did not fire in a 12 s, 1500 m game on the CPU (fog and slow starts); behaviour is
  covered on constructed observations instead.

## Next

- Run the exploit check on a trained policy once the GPU is free; the author decides the league
  weights.
