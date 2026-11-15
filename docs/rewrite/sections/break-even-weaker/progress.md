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
- **Worked.** Quick-tier tests on the touched files (see below). A synthetic test runs a
  stronger, a weaker, a near-parity and a far-stronger policy through allocate/fit from
  the first parity batch: each finds its ratio within 3 SE, and no lineup exceeds 20 a
  side or goes below the floor.
- **Not done.** No GPU run; the default target and budget are untested on a real policy.
- **Next.** The author reviews; merge into `feat/imugi-boundary`.
