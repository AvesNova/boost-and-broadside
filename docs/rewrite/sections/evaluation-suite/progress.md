# Evaluation suite: `bnb evaluate` and its design pilot (2026-11-15)

Branch `feat/evaluation-suite`, from `origin/integrate/next-runs` (b2b57e04). The spec is
[evaluation-suite.md](../../evaluation-suite.md); the decisions behind it are in
[arms-todo.md](../../arms-todo.md) (2026-11-14 and 15).

## Tried and worked

- **The command** (982d1988): `bnb evaluate A B --out DIR` with the 13 conditions,
  the fixed-weight score (thirds, equal within a mode), the gate, the cost-aware
  allocation toward each planned look, and the group-sequential test
  (`bnb.league.sequential`: O'Brien-Fleming-type alpha and beta spending at the looks'
  actual fractions, a binding inner wedge for the tie, I_max from the planned looks).
  `bnb evaluate decide DIR` takes a look; `bnb evaluate design DIR ...` reads pilots.
- **Boundaries.** The recursive integration (Simpson nodes over the continuation region)
  matches 400k Monte Carlo paths to the third decimal at K = 1 to 5, at drifts 0, η_δ and
  2η_δ. K = 1 reproduces the fixed-sample test (u = 1.960, η_δ = 2.802). The inflation
  over one look is small: η_δ 2.853, 2.896, 2.917, 2.933 for K = 2 to 5 (I_max 4 to 10%
  above the fixed sample's).
- **v19 checkpoints in the v20 game.** rl-v20, rl-mix-kl and rl-dm at 75M load (their
  records hold observation schema 3, the current one) and play an Arena game on the CPU.
  All three are one build, so A and B share a network group and one stacked forward.
  What differs is the game: version 20 changed the frontline lock (19 to 6 s per root km),
  which these policies never trained on.
- **CPU end to end:** tiny 3-ship presets, scripted against scripted at T = 2: every
  condition's first batch, break-even until both fits cross, then the look (a tie at
  fraction 0.6 with δ = 0.2), in 30 s.

## Didn't

- A first smoke run with 4 s games ended almost every break-even game drawn; the fits
  never crossed and the suite stopped at `initial_batch_limit` with the reason, as meant.

## Next

- The GPU pilot (unit `evaluate-pilot`) once `twin-noise` ends (about 05:00), then the
  design from its variances and costs.
