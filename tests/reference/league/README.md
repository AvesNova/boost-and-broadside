# League equivalence fixtures (section 8)

Goldens computed in float64 on the CPU from v1's own rating code at tag `v1-torch-final`,
on simulated match records:
- the Bradley–Terry fit with standard errors;
- one player rated against known opponents;
- the ladder fit with the gauge held;
- the live rating;
- the offline A-optimal game allocation.

Opponent sampling, snapshots and the pool are tested by the section's own tests.

- **Run:** `uv run --no-sync pytest tests/reference/league`. The tests skip until
  `league_arm.make_arm()` returns an implementation.
- **Fixed:** the tests, helpers, conftest and goldens. Only `league_arm.py` is written by
  the section.
- **Validated:** the tests pass on v1, and every deliberate mutant fails by a value
  comparison.
- **Tolerance:** each golden is its estimator's converged optimum. An implementation must
  reach it to 1e-6 relative (about 1e-3 rating points at the gauge's scale), whatever
  algorithm it uses. Infinities and nan must match exactly.

## Conventions

- **Players** are indexed `0 … K−1`.
- **`wins[i, j]`** is player i's wins over player j. Where draws exist, the caller scores
  them as half a win to each side (`fit_ratings`), or passes them separately as a
  symmetric `ties` matrix (`fit_ladder`) or vector (`rate_live`).
- **Rating scale:** 400 points per factor of 10 in odds: `P(i beats j) = 1 / (1 +
  10^((r_j − r_i)/400))`.

## Interface

- **`fit_ratings(wins, anchor, prior_games)`** returns `(ratings (K,), stderr (K,))`.
  - The maximum-likelihood Bradley–Terry ratings, with the anchor pinned at 0.
  - `prior_games` virtual games per player are split evenly for and against the anchor,
    which keeps a clean sweep finite.
  - The standard errors come from the inverse Fisher information with the anchor's row
    and column removed; the anchor's is 0.
- **`rate_against(opponent_ratings, wins, losses)`** returns `(rating, stderr)`: one
  player's maximum-likelihood rating against opponents with known ratings.
  - A clean sweep is `+inf` (`−inf` if swept), with an infinite error.
  - No games gives `nan` with an infinite error.
- **`fit_ladder(wins, ties, fixed, fixed_ratings, prior_games)`** returns `ratings (K,)`.
  - The fixed players (the gauge: random 0, rung `p` at `1000 p`, scripted 1000) keep
    `fixed_ratings` exactly. Every other player is fitted.
  - Each free player gets `prior_games` virtual games split evenly for and against the
    strongest fixed rating.
  - Ties count as half a win to each side. Every player has at least one game.
  - The case includes v1's trap: one checkpoint sweeps random 15–0 and the lowest rung
    27–0. Fitted jointly, that ranks random above the rung.
- **`rate_live(opponent_ratings, wins, losses, ties)`** returns `(rating, stderr)`: one
  update's live record against known opponents, with ties as half wins and no prior. A
  clean sweep is `+inf`.
- **`allocate(pair_games, ratings, anchor, budget, minimum_per_pair)`** returns a `(K, K)`
  symmetric integer matrix: how many of `budget` games each pair plays.
  - The share is each pair's A-optimal value: how much one more game would reduce the
    summed rating variance, with the anchor's gauge imposed.
  - Shares are floored, then raised to at least `minimum_per_pair`. The remainder goes
    to the pairs with the largest shares, one each.
  - With no games yet, the budget is spread evenly over pairs.
