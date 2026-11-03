# League notes (section 8)

Branch `section/league`. Spec: `league-spec.md`. Brief: `league-brief.md`.

## Progress

- [x] Part 1: the ratings (`bnb.league.ratings`, `bnb.league.gauge`); the six equivalence
  tests pass through `league_arm.py`.
- [ ] Part 2: the pool, retention and snapshots.
- [ ] Part 3: slots, the stationary ladder, the drain and in-graph sampling.
- [ ] Part 4: resume.
- [ ] Part 5: metrics.
- [ ] Part 6: `bnb ratings` and `bnb head-to-head`.

## Spec items

### Ratings (spec 4, 6; part 1)

- `bnb.league.ratings`: `fit_ratings`, `rate_against`, `rate_live`, `fit_ladder`,
  `allocate`, float64 on the CPU, Newton's method with a backtracking line search run to a
  Newton decrement below 1e-13 (the goldens' converged optimum to well under 1e-6).
  - `allocate`'s value is the derivative of the summed variance, `p(1 − p) v'C²v`, not
    the Sherman–Morrison drop for one whole game: only the derivative reproduces the
    golden (found by comparing both against it). The remainder goes to the largest
    shares, not the largest fractional parts.
- `bnb.league.gauge.Gauge`: one mode's live gauge, run once per update:
  ladder fit (`fit_ladder`, prior 1 game at the strongest fixed rating) → live rating
  (`rate_live`, this update only) → the averages (`rate_against` over the last 200 games,
  each against its update's live rating; no games: the live rating).
  - **How checkpoints get a record (reading of the spec):** training only plays the live
    policy against entries, so the only games among stationary players are a snapshot's
    own: a snapshot is the weights that played its update's rollout, and those games
    (against stationary entries; not against the averages) are credited to it when it is
    added. It starts at that update's live rating; the next ladder fit refines it.
  - A sweep reports ±inf and an infinite error; `Rated.rating` holds the last finite
    value, and that is what sampling and the fits use.
  - A player with no stationary games is left out of the ladder fit (it keeps its start).

## Acceptance tests and hand breaks

- 8.1 equivalence: `tests/reference/league` (6 tests) through `league_arm.py`.
- 8.2 recovery: `tests/sections/test_league_ratings.py`: the live rating (200 trials,
  7 opponents × 12 games) and the offline fit (200 trials, 6 players, 30 games per pair),
  ties included with an expected score equal to the model's; coverage within 2 SE ≥ 90%.
- Hand breaks (gauge): (a) the live rating assigned even when infinite → the sweep test
  fails; (b) the live rating from an accumulated record instead of this update → the sweep
  and window tests fail. Both reverted.

## Measurements

(none yet)

## Deviations, requests and open questions

(see the spec items above; collected here at the end)
