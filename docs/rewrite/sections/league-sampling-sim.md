# League sampling: why the league played beaten opponents, and the samplers compared

Brief: `league-sampling-sim-brief.md`. Run: `2026-11-06-mixed-calm-fern` (96 updates,
100.7M steps, read only). Simulation: `bench/league_sampling.py`, run as

```bash
uv run --no-sync python -m bench.league_sampling ../runs-v2/2026-11-06-mixed-calm-fern --seeds 400 --json out.json
```

It reads `metrics.jsonl`, `config.json` and the training checkpoints' league state, and
plays no games. Tests: `tests/sections/test_league_sampling_sim.py`. No league behaviour
changed.

**The run's code is not v2's head.** The run ran commit 724153e7. Its gauge rated snapshots
with a separate ladder fit (`fit_ladder`). Commit 6845fa73 replaced that with the run's one
fit. The reproduction uses the run's rule: `fit_ladder_724153e7` in the script is a copy of
the old function, and the Newton solver it calls (`bnb.league.ratings._maximise`) has not
changed.

## 1. Reproduction

| What | Result |
|---|---|
| Sampling shares from the logged ratings (`sampling_weights`), every update, mode and entry | max abs difference 0.0 where the live rating is logged; 7e-16 where it is solved for from the shares (updates 0–12, before the fit rated live) |
| Host draws replayed with `counter_uniform` (seed 0, `league.swap`) and the weights logged the update before | the 6 drain draws reproduced; the logged slot column matched at 89/89 updates after the first checkpoint; `held` and `swaps` matched at 9/9 checkpoints |
| Snapshot ratings replayed with the run's `fit_ladder` on each snapshot's credited games and the logged ladder ratings | 420/420 fits exact (max difference 0.0) |
| Game share by kind from the reconstructed residency, with a fitted game rate per mode and class | see below |

Game share by kind (predicted / logged), each window pooled:

| Updates | scripted | rungs | averages | snapshots |
|---|---|---|---|---|
| 0–31 | 0.117 / 0.166 | 0.332 / 0.282 | 0.552 / 0.552 | 0 / 0 |
| 32–63 | 0.044 / 0.044 | 0.345 / 0.370 | 0.612 / 0.587 | 0 / 0 |
| 64–95 | 0.098 / 0.089 | 0.491 / 0.495 | 0.408 / 0.409 | 0.003 / 0.007 |

The per-update mean absolute error is 0.043. Each mode logs about 50 league games per
update, so most of that error is binomial noise on single updates.

## 2. The cause

There were three causes, in order of size. Neither of the brief's two candidates is the
main one.

**(a) Snapshot ratings ran away: the main cause, and specific to the run's code.**
- The old ladder fit started each snapshot at rating 0, about 1000 below its optimum. Its
  first Newton step overshoots to where every win probability is exactly 1 in float64.
  There the Hessian is 0, the step is 0, and the fit stops.
- The logged snapshot ratings were therefore 7,000–65,000, median 13,867, with infinite
  error. On the same games and the same likelihood the true maximum is 1,280–1,920, median
  1,642.
- At those ratings p(1 − p) = 0, so every snapshot drew only its floor, 0.05/18 = 0.003.
  The one exception is the frontline rating of `snapshot_56623104`, whose fit happened to
  converge to about 1,640.
- Late-run draw weight (updates 48–95, modes mixed):

| | ladder | averages | snapshots |
|---|---|---|---|
| As run | 0.44 | 0.38 | 0.18 |
| Snapshot ratings repaired | 0.25 | 0.19 | **0.56** |

- **v2's head does not do this.** The one fit, given the same credited games with the rungs
  pinned to their logged ratings, rates these snapshots at 1,440–1,750. The solver is the
  same, though, so a guard is worth having (recommendation 4).

**(b) Slot residency and turnover: the brief's candidate 2, and the cause that remains at
head.**
- A drain ends only when the last of a slot's 256 rows has reset, about one game clock:
  exactly 14 updates in this run.
- Drains run one at a time, round robin. The whole run therefore made 11 draws: 4 at the
  start and 7 drains. Each draw holds a quarter of the rows for about 56 updates.
- So late game shares follow weights drawn up to 56 updates earlier, before most snapshots
  existed, and they vary widely from one seed to the next.
- With the run's own weights, the late snapshot game share is 0.067 (10th–90th percentile
  0–0.22) under the slots, against 0.17 for a draw per row. The run drew 0.004.
- From update 55 on, two slots held the ladder and two held the exponential average. The
  ladder was `scripted` and `rung_T2`, each playing the rungs by the ladder weights; the
  average was drawn twice.

**(c) The ladder is 11 entries.**
- Each rung gets its own floor share and its own p(1 − p). At the gauge's predicted p of
  0.93–0.97 that is about 0.03–0.06 each, even though the live policy actually scored
  0.98–1.0.
- With snapshots repaired, the ladder still gets 25% of draws. With the observed scores in
  place of the gauge's p it still gets 21%: the count matters more than the prediction.
- A drawn rung turns the whole slot into a ladder slot.
- The floor is only 5% in all, of which 3.1 points fall on the ladder. **The brief's
  candidate 1, the floor dominating, is ruled out.**

Two smaller effects:
- Ladder games end sooner than network games. The fitted mean game lengths are 158 s
  against 212 s in frontline, and 184 s against 202 s in deathmatch. So the ladder's share
  of games is about 1.1–1.3 times its share of rows. The learner trains on steps, which is
  why the table below gives both.
- A new network entry's first games are tallied one game length after its rows switch. That
  is about 10 updates of a 14-update drain, which is why `snapshot_75497472` logged no
  deathmatch games in the 13 updates after it was drawn.

## 3. Samplers

All rows use the same reconstructed ratings and the same seeds: 400 Monte Carlo runs of the
slots and drains. Scores cover the late updates 48–95, as mean and 10th–90th percentile.
"Repaired" replaces the snapshot ratings with the likelihood's true maximum, which is close
to what head's fit gives.

Columns:
- **Open games** and **open steps:** the share of games, and of steps, against entries the
  live policy truly beats less than 90% of the time.
- **Signal:** the sum over steps of share × p(1 − p), at most 0.25.
- **Rating info:** the training games' information on live's rating, in even games against
  a known opponent, per update and mode (definition in section 4).
- **Snap.** and **ladder:** those entries' shares of the games.

| Sampler (repaired unless said) | Open games | Open steps | Signal | Rating info | Snap. games | Ladder games |
|---|---|---|---|---|---|---|
| Logged (as run) | 0.225 | 0.266 | 0.058 | 4.4 | 0.004 | 0.566 |
| Today, as run | 0.214 [0.07, 0.37] | 0.236 | 0.056 | 5.2 | 0.067 | 0.610 |
| **Today** | 0.368 [0.16, 0.55] | 0.396 | 0.090 | 7.0 | 0.278 | 0.494 |
| Today, 8 slots | 0.230 [0.11, 0.35] | 0.254 | 0.062 | 6.8 | 0.132 | 0.579 |
| Today, 4 slots draining together | 0.555 [0.40, 0.71] | 0.586 | 0.130 | 8.8 | 0.467 | 0.354 |
| Today, a draw per row (the limit) | 0.607 | 0.641 | 0.143 | 11.8 | 0.528 | 0.292 |
| No floor over beaten entries | 0.384 [0.19, 0.56] | 0.412 | 0.093 | 6.9 | 0.289 | 0.475 |
| (1 − p)² | 0.687 [0.50, 0.81] | 0.705 | 0.150 | 8.0 | 0.602 | 0.191 |
| (1 − p)², as run | 0.721 | 0.738 | 0.158 | 8.1 | 0.663 | 0.173 |
| Fixed 40%: newest 2 snapshots + 2 averages | 0.255 [0.07, 0.47] | 0.277 | 0.067 | 6.0 | 0.199 | 0.560 |
| Ladder as one entry | 0.532 [0.35, 0.71] | 0.551 | 0.120 | 7.8 | 0.385 | 0.233 |
| **Ladder as one entry, draining together** | 0.749 [0.62, 0.86] | 0.761 | 0.164 | 9.6 | 0.625 | 0.104 |
| Nash over the pool | 0.768 [0.73, 0.80] | 0.787 | 0.166 | 7.0 | 0.739 | 0.180 |

Cost and failure modes:

- **No floor over beaten entries:** a one-line change with almost no effect (+1.6 points),
  because the floor is small. It never retests an entry the gauge calls beaten, which hurts
  if that rating is wrong.
- **(1 − p)²:** a one-line change. It chases rating errors. On the run's broken ratings it
  put 66% on the snapshots by accident, because it rewards an entry rated far above live
  rather than near it. A truly stronger entry, or a fluke rating, takes almost everything.
  Its p(1 − p) is lower than its share suggests.
- **Fixed share:** simple, and it does not depend on the ratings. But the share it fixes
  includes the uniform average, which live beats 97% of the time, and it caps the snapshots
  below what today's rule gives them once the ratings are right. It is worse here.
- **Ladder as one entry:** a small change in `_weigh`/`_draw`; the ladder rows already draw
  per reset in the graph. It removes cause (c). The ladder keeps a share as large as its
  best rung's p(1 − p), so the scripted anchor still gets games.
- **Nash over the pool:** in a transitive pool, Nash is the single strongest entry, here
  the newest or strongest snapshot. That gives the highest signal but no diversity, and the
  choice is an argmax of noisy ratings. Its rating information is lower than draining
  together, because the newest snapshot is the least well rated. It needs the payoff matrix
  between entries, which the run records only through the lanes. The script uses the payoffs
  the ratings imply, which the ratings were fitted to.
- **8 slots with serial drains:** worse in a run of this length. Each slot turns over every
  8 × 14 = 112 updates, longer than the run, so the draws made at the start dominate. The
  stacked weights cost 9 sets instead of 5 and there are more set calls per tick.
- **Draining all slots together:** each draw is resident 14 updates instead of 56, close to
  the per-row limit.
  - Cost: one incoming set per slot, so 2 × slots stacked sets (8 instead of 5 at 4 slots).
    That means more opponent-parameter memory and more per-set calls per tick. The rows
    played stay the same.
  - Complexity: the drain bookkeeping becomes per slot.
  - Failure mode: every slot changes at once, so the opponent mix jumps every 14 updates
    instead of a quarter at a time.
  - A middle option is 2 incoming sets.

## 4. Overlap with the rating games

How informative a game is about live's rating depends on the same quantity as the learning
signal, w = p(1 − p), discounted by the opponent's own rating error σ. Per opponent the
information is n w / (1 + n w σ²) for n games. This is the expression behind the "rating
info" column; σ for a snapshot comes from its credited games.

- **The agents are largely the same.** Across the samplers, rating information rises with
  the signal (5.2 to 11.8 as the signal goes from 0.056 to 0.143). The exception is
  concentration on poorly rated entries: Nash has the highest signal and middling
  information.
- **The training games already outweigh the lanes' games on live.** The lanes' live games
  are against the scripted agent, the exponential average, and snapshot slots; the run's
  lanes left other slots unrated. They gave 0.65 even games per update and mode. The run's
  training games gave 4.4, and today's sampler with repaired ratings would give 7.0.
- A sampler that fixes residency (draining together, 8.8) gives the rating about 13 times
  the lanes' information. Head's fit already reads the training games (6845fa73).

## 5. Recommendation

These are proposals; the author decides.

1. **Fix turnover first.** Drain all slots together, or use 2 incoming sets, so each draw is
   resident about one game clock rather than slots × one game clock. This is the largest
   gain at head: open games from 0.37 to 0.56, snapshot games from 0.28 to 0.47, and a far
   narrower spread.
2. **Make the ladder one entry for the draw.** Weight it by its best rung's p(1 − p) and let
   its rows keep drawing rungs in the graph. Open games go from 0.37 to 0.53 on its own.
   Combined with (1) they reach 0.75 [0.62, 0.86]: Nash's signal (0.164 against 0.166)
   without its concentration, and with more rating information (9.6 against 7.0). The
   ladder then gets about 10% of the games. If that is too little for the scripted
   anchor's chain, the rating lanes already play it.
3. **Keep p(1 − p).** Leave the floor as it is: its effect is within noise. Do not adopt
   (1 − p)² or Nash. Their gains here come from concentrating on one or two entries, and
   they amplify any rating error.
4. **Guard the gauge.** Add a test that a near-sweep, for example 28 of 29 against the
   ladder, rates finite and within a few hundred points of its maximum under the one fit
   (the test file has the run's case). Add an alert when an entry's gauge rating is more
   than about 1,500 from live's, or its error is infinite after it has played. Do not add
   more slots without (1): 8 serial slots are worse.

## Assumptions

- **The first five draws.** Their weights were never logged (live's starting rating is the
  parent policy's, outside the run). The replay starts from the first checkpoint (update 6:
  `held` and `swaps` = 5). A scan finds the five draws consistent with a range of starting
  ratings.
- **Drains.**
  - Draining rows are assumed to switch linearly over the drain.
  - One update is taken as 300 s / 14 = 21.4 s of game time; the game clock caps a game
    at 300 s.
  - A switched row's games are tallied one mean game length later, by mode and class, with
    lengths taken from the fitted game rates (512 opponent rows per mode).
  - The four game rates (per mode, ladder or network) are fitted over the whole run.
- **"True" p:**
  - a rung: its observed score over the last 20 updates (the whole run when that holds
    fewer than 10 games), with half a game each way;
  - an average: its observed score over the last 10 updates when that holds at least 10
    games, else the gauge;
  - a snapshot: the live rating at its update against live's now, since it was the live
    weights.
- **The Monte Carlo** uses the reconstructed per-update ratings as given. It ignores that a
  different sampler would have trained a different policy.
  - It draws from the modes' mixed weights, as the league does. The per-row limit uses each
    mode's own weights.
  - It takes the drain as 14 updates for any slot count.
  - It does not model the tally lag, so it scores what was played, not when it was counted.
- **Rating information** counts one update's games, not the fit's 10-update window, and
  uses the logged errors for the rungs and averages.
- **The brief** says 4 snapshots; the pool held 4 for updates 71–88 and 5 at the end. Its
  "about 50% rungs" matches the rung kind, 0.49 for updates 64–95; with the scripted agent
  the ladder took 0.57.
