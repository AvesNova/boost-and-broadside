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
- 07:10 sanity done (numbers below once written up). Compile check launched as unit bnb-first-touch (performance profile, load 3.3, GPU idle). ETA 07:15 (worst 07:25).
- 07:12 Sanity (nose-cone/sanity.py, extended: contacts are the tree's own `core.collisions`;
  overlap episodes that began inside the radii and fast ones apart), 256 games each, seed 0;
  first-touch/sanity-before.json (core 17e6cf3b) and sanity-after.json. The before numbers
  reproduce nose-cone/sanity-after.json (1v1 draws 41, 32.09 s). Worked: 0 doubles and 0 misses
  in all four conditions after (before 0-5 each: 4v4 fields 5 doubles, 5 misses). Didn't expect:
  contacts rose 12-17% (1v1 248 -> 290). Checked on the same play (episodes.py on the before
  tree, both rules on the same moves): every overlap episode holds exactly one first touch; the
  extra contacts are episodes cut by a death or the game's end between first touch and the
  closest-approach tick, which the old rule never scored (1v1 68 of 316, 4v4 103-120 of
  880-850). The old once-per-pass check left cut episodes out, so this miss did not show.
  Next: compile check (unit bnb-first-touch, running), docs.
