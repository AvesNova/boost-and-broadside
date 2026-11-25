# Collisions at first touch: progress

Author decision 2026-11-26: score a ship pair once per overlap, on the tick of first touch
(start gap at least the two hit radii, the swept straight move inside them during the tick);
b the closest distance of the tick's relative line extended (the pass's predicted closest
approach); blame at first touch unchanged. Branch collision-first-touch from core 17e6cf3b,
worktree ../v2-first-touch.

- 06:59 worktree made; read core.collisions, tests, nose-cone sanity. Plan: change
  `collisions` (contact = start gap >= R and segment distance < R; b from the unclamped line),
  tests, then sanity before/after (CPU), compile check (unit bnb-first-touch), docs.
  ETA code + tests 07:40, sanity 08:30, compile check 09:00 (worst 09:30).
- 07:05 sanity before (core 17e6cf3b, scratchpad worktree) and after launched on CPU, 4 threads each, MemoryMax 8G scopes; records persist per (size, fields) to first-touch/sanity-{before,after}.json. ETA 07:15 (worst 07:30).
