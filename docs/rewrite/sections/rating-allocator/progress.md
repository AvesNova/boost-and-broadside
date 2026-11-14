# Rating allocator and shared pool: progress

Branch `feat/rating-allocator` from `integrate/next-runs` (997a700e).

## 2026-11-14 19:10 to 20:30: root cause and the design

- **Tried:** reproduced the 672-of-1152 allocation in a three-player simulation (scripted,
  two arms at +1500 and +1450 Elo, 2 blocks of 96): the old `allocate_blocks` spent 960 of
  1152 games on one anchor pair and 96 on the arms' pair.
- **Root cause:** the fit's prior was one virtual game against the anchor. A virtual game's
  information is p (1 - p) at the estimate, so after a sweep the estimate grows as the log
  of the games and n p (1 - p) stays at about one game's: the strong player's standard
  error never falls (175 Elo after 96 or 960 games). The greedy trace allocator, which
  values the next block at the current estimate, saw the largest variance there and kept
  choosing that pair; the arms' difference, which the games could resolve, was not a target.
- **Worked:** `bnb.league.design`: a Gaussian prior (1000 Elo about the anchor), games
  planned at the less informative end of a one-SE interval (at most 200 Elo), a continuous
  A-optimal design rounded one block at a time under players-per-batch and lanes, targets
  per player plus decision players and contrasts, and out-of-reach / hopeless rules. Same
  simulation with the arms' contrast at 20 Elo: 480 of 576 games to the arms, then stops.
- **Didn't:** a one-standard-error margin without a cap made every unplayed pair look
  saturated (the prior's 1000 Elo); a floors-then-greedy rounding left a ladder's first batch
  with no anchor games (the greedy trace is myopic); fixed: cap and sequential rounding.
- **Next:** the shared pool, the loop check, the CLI, the probe, docs, the refit of the
  live-elo records.
