# Break-even: the weaker side at 20 (progress)

Branch `feat/break-even-weaker-twenty`, from `feat/imugi-boundary` (game version 21). The
author's decision of 2026-11-15: break-even fixes the weaker side at 20 ships and varies
the stronger side's count downward.

## 2026-11-15: the rule, the fit and the suite

- **Tried / done.** `bnb.league.crossover` works on lineups (policy's ships, opponent's
  ships) along one axis, r = log(policy / opponent). The candidates are (n, 20) and
  (20, n) for n from the floor to 20; the fit is the old logistic, now in r, over both
  regimes at once; the first batch is all 20v20, then the bracket on the side the mean
  score points to (both at exactly one half), then the c-optimal allocation over every
  candidate. Separated bounds are lineups; halving/doubling is in ratio. `maximum_count`
  and the "count above 20" path are gone; `opponent_count` became `fixed_count`.
  Evaluation: blocks record `policy_count` and `opponent_count`; the score formula is
  unchanged in form, with r in place of log(count). `bnb crossover` applies the mode's
  minimum team per preset (it did not before; only the suite did).
- **Choices beyond the brief** (for the author):
  - `target_stderr` is now in log ratio: 0.02 (0.3 ships at 15 against 20; the old 0.3
    ships was 0.3/N* in log terms, so no single value is exactly equivalent).
  - The reported ratio is policy / opponent (below 1 for a policy stronger than Imugi).
    The old `ratio` was opponent / N*, the inverse; results and docs that quote "beats
    1.4x its number" read as a ratio of about 0.7 now.
  - The record key carries `fixed_count` and `minimum_count` (the old key had the
    opponent's count): a different floor is another measurement, so a resume never plays
    below the current floor.
  - The break-even lineup is reported with the non-fixed side fractional (e.g. 20v16.7).
- **Worked.** Quick tier on the touched files: test_crossover 28 passed; test_evaluation
  27 passed (4 slow deselected) with test_modes' break-even floor test; docs references
  and docstrings 607 passed. A synthetic test runs a
  stronger, a weaker, a near-parity and a far-stronger policy through allocate/fit from
  the first parity batch: each finds its ratio within 3 SE, and no lineup exceeds 20 a
  side or goes below the floor.
- **Not done.** No GPU run; the default target and budget are untested on a real policy.
- **Next.** The author reviews; merge into `feat/imugi-boundary`.

Also noted, not changed: docs/explanation/league.md's decision-suite paragraph still says
frontline break-even is left out of the score until the map-size arm reports, which
contradicts evaluation-suite.md and the default `unscored=()`.

## 2026-11-15: no floor, ratio as held, suspicious flag (author's follow-up)

- **Done.** Break-even has no floor: the stronger side goes down to 1 ship in every mode,
  Frontline included. `minimum_count` is gone from `CrossoverConfig`, `EvaluationConfig`
  and `bnb crossover`; `crossover_config()` takes no mode. `MINIMUM_SHIPS_PER_TEAM` stays,
  for equal-size teams. The "bound" stop is all won at 1v20 or all lost at 20v1. The
  record key drops `minimum_count`.
- The reported ratio is the opponent's ships per one of the policy's, exp(-r*) (above 1
  for a policy stronger than Imugi: "holds X times its number"); the fit's covariate stays
  r = log(policy / opponent).
- A break-even with the stronger side at 2 ships or fewer against 20 (either side) sets
  `suspicious` in the result, prints a WARNING line suggesting a bug check, and marks the
  CLI's summary row. The evaluation suite does not flag it (its records hold the fits;
  the flag is in `Fit.suspicious` if wanted there).
- league.md's stale sentence on frontline break-even being unscored is fixed.
- First batch and stop target unchanged, as asked.
- **Bug found and fixed** (present in the old count-based code too): with separated
  outcomes and a mixed lineup beyond the last all-lost one (e.g. 20v20 and 20v10 all lost,
  20v5 mixed), the "double the ratio" step started from the all-lost bound and landed on the
  mixed lineup again, forever. It now halves/doubles from the lowest/highest lineup played.
  The simulated test gained a far-weaker policy (20v3) that caught it.
- **Tests.** test_crossover 35 passed; test_evaluation + test_modes' break-even test 27
  passed (4 slow deselected); docs references/docstrings 607 passed.

## 2026-11-15: lineup per game; whole-axis first batch (author's second task)

- **Done.** `Arena` tallies per game (`MatchSeat.tally` by game; `play` still returns per
  block, `game_outcomes()` the last batch's (games, 3)); `play(active=...)` takes (games, 2)
  per-game counts. Break-even allocates per game, in pairs of games (one per seat, so each
  lineup stays seat-balanced): first batch `first_lineups` (13 ratios evenly in r, 1v20 to
  20v1), then bracket / separated / c-optimal as before at pair granularity. Records hold
  one entry per (block, lineup) with a `block` index (`block_tallies`); the fit only needs
  the per-lineup tallies, so a resume is exact. Evaluation's Engine does the same per side.
- **Choice.** The allocation unit is a pair of environments rather than a single one, to
  keep every lineup's games split evenly between the seats (block environments are even).
- `CrossoverConfig.parity` went (no longer used); `first_ratios` (13) added.
