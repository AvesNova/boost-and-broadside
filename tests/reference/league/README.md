# League equivalence fixtures (section 8)

Goldens computed in float64 on the CPU from v1's own rating code at tag `v1-torch-final`,
on simulated match records: the Bradley–Terry fit with standard errors. (v1's offline
game allocation, `allocate`, was retired on 2026-11-06 with its golden: the tournament now
gives whole blocks of games to pairs, `bnb.league.ratings.allocate_blocks`.)

Opponent sampling, snapshots and the pool are tested by the section's own tests.

- **Run:** `uv run --no-sync pytest tests/reference/league`. The tests skip until
  `league_arm.make_arm()` returns an implementation.
- **Fixed:** the tests, helpers, conftest and goldens. Only `league_arm.py` is written by
  the section.
- **Validated:** the tests pass on v1, and every deliberate mutant fails by a value
  comparison.
- **Tolerance:** each golden is its estimator's converged optimum. An implementation must
  reach it to 1e-6 relative (about 1e-3 rating points at the gauge's scale), whatever
  algorithm it uses.

## Conventions

- **Players** are indexed `0 … K−1`.
- **`wins[i, j]`** is player i's wins over player j. Where draws exist, the caller scores
  them as half a win to each side (`fit_ratings`).
- **Rating scale:** 400 points per factor of 10 in odds: `P(i beats j) = 1 / (1 +
  10^((r_j − r_i)/400))`.

## Interface

- **`fit_ratings(wins, anchor, prior_games)`** returns `(ratings (K,), stderr (K,))`.
  - The maximum-likelihood Bradley–Terry ratings, with the anchor pinned at 0.
  - `prior_games` virtual games per player are split evenly for and against the anchor,
    which keeps a clean sweep finite.
  - The standard errors come from the inverse Fisher information with the anchor's row
    and column removed; the anchor's is 0.
