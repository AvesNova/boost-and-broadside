---
covers: [src/bnb/league/, src/bnb/runtime/opponents.py]
---

# The league

<!-- generated: section-status -->

The league chooses the learner's opponents during training, rates the learner as it trains,
and rates finished policies afterwards. The training side is `bnb.league.league.League`,
attached to a trainer as its hooks; inside the rollout it acts through one controller, the
league's seat (`bnb.runtime.opponents.OpponentSlots`). The offline side is
`bnb.league.tournament`, behind `bnb ratings` and `bnb head-to-head`.

## The pool

The default lineup (`bnb.runtime.train.LINEUP`) gives each game of the training mix two
pools: self-play and the league's seat, half each. Self-play is not a pool entry; it keeps
its fixed share. The entries (`bnb.league.pool.Entry`) are:

- **snapshots** of the live policy, taken every `LeagueConfig.snapshot_every` environment
  steps. Without a set value the league measures it once, about 30 minutes of the run's
  throughput, and records it in `run.json`{.plain}. A snapshot is the weights that played the
  update it was taken at;
- **the averages:** the uniform average and the exponential moving average of the live
  policy's weights, folded in after every update (`bnb.league.pool.average_into`, the decay
  `LeagueConfig.exponential_decay` per update);
- **the stationary ladder:** random, the rungs and the scripted agent
  (`LeagueConfig.ladder`, the scripted probabilities; see
  [The scripted agent](scripted-agent.md)).

**Retention** (`bnb.league.pool.retained`) keeps the newest 8 snapshots and, of the older
ones, the one nearest each age of 2^k snapshots for k from 3. The run's `league/` directory
holds each retained snapshot as `step_<steps>/checkpoint`, with the two averages at the same
step beside it (`uniform_average`, `exponential_average`), all policy files.

## Sampling

An entry's sampling weight is `p (1 − p)` (`bnb.league.pool.sampling_weights`), where `p`
is the predicted probability that the live policy beats it under that mode's live gauge,
plus a floor (`LeagueConfig.floor`) spread evenly over the entries. Each mode samples by
its own gauge.

## Slots, the ladder and the drain

The league's seat plays every opponent row of the batch in the captured rollout graph:

- **Network slots** (`LeagueConfig.slots`): their parameters are stacked, with one extra set
  for the entry coming in. Slot k plays rows k, k + slots, …, so every slot plays every
  mode. Each set runs on its own rows inside the one compiled graph, and each row keeps its
  network's carry.
- **The ladder:** one scripted pass and one uniform draw over every row; a row's scripted
  probability decides between the two (scripted spec A6). A slot can hold the ladder; each
  of its rows then draws a rung at every reset from its mode's weights over the ladder.
- **The drain:** after each segment, if no slot is draining, the next slot (round robin)
  receives an entry drawn from the modes' weights, mixed by the modes' shares of the
  opponent rows. Its weights are copied into the extra set in place. Each row of the slot
  switches at its own next reset, so no opponent changes mid-episode; when every row has
  switched, the extra set becomes the slot's and the drain ends.
- **Draws in the graph** use counter-based uniforms (`bnb.runtime.opponents.counter_uniform`,
  a hash of the stream's key, the counter and the row), so they need no generator state and
  repeat exactly; the host draws a slot's entry the same way.
- **The results:** the rollout tallies every ended game by what the opponent row played and
  by mode, from the learner's side; the league reads the tally once per segment.

## The rating lanes

The live and averaged ratings come from the rating lanes ([The training
loop](training-loop.md#the-rating-lanes)): games of rated matchups played beside the training
games, in the same rollout. `bnb.league.lanes.LaneRatings` keeps their record by mode and
fits, once per update and per mode, one Bradley–Terry model of every lane player with the
scripted agent pinned at 1000 (`bnb.league.ratings.fit_ratings`, draws as half a win, one
prior game against the scripted agent, half won, which keeps a sweep finite).

- **The record:** games are kept by mode and by who played them. A game of two stationary
  players (the rungs, the frozen snapshots) is kept for the whole run; one with a moving
  player (the live policy, an average) for its last `LaneConfig.window` updates (10).
- **The snapshots:** every `LaneConfig.snapshot_every` updates (8) the live weights are
  frozen into the next of the lanes' snapshot slots, round robin (the first holds the
  starting weights). A slot's games are credited to the snapshot it holds,
  `lane_snapshot_<steps>`, so a retired snapshot keeps its games and its rating, and a new
  one joins the fit as soon as its games connect it. A game in progress at a swap was
  played partly by the old weights, so a slot's games count again only once every game
  since the swap began after it (`settle`: the longest episode over the ticks an update
  plays, 8 updates at the 300 s limit). The slots' weights are in the training
  checkpoint.
- **Who is rated:** a player whose games connect it to the scripted agent; none is reported
  before its matchups have played.
- **The ladder:** the rungs, chained from the scripted agent down, are rated from their own
  lane games. Measured against the scripted agent the old fixed scale, rung `p` at
  `1000 p`, was far off (rung_0.95 scores 0.065 against it, about 530 points below, not 50).
- **The series:** per mode `rating/live/<mode>`{.plain} and `rating/average/<mode>`{.plain} (the
  exponential average), their standard errors, and every rated player's
  `lanes/rating/<mode>/<player>`{.plain}. A BC run's lanes give its live rating the same way
  (`bnb.league.lanes.LaneHooks`), and the clone's policy file keeps it per mode
  (`PolicyRecord.ratings`): a warm start's live rating starts from its parent's, not 0.

Against the scripted agent alone a policy 1000 points above it would be rated blind: its
games there carry 1/60 of an even game's information. Through the snapshot chain, whose
stationary links gather games over the run, the same window of live games rates it several
times more precisely
(`tests/sections/test_lanes.py::test_a_strong_player_is_rated_far_better_through_a_snapshot_chain`).
The measured game counts and errors are in `docs/rewrite/sections/rollout-lanes-notes.md`.

## The live gauge

During training each mode has a gauge (`bnb.league.gauge.Gauge`) that steers sampling. The
ladder's ratings are held by the fits: the scripted agent 1000 and every other rung the
lanes' calibration (`Gauge.calibrate`); before the lanes have rated a rung it sits at the
prior `1000 p`. Once per update:

1. **The ladder fit:** the snapshots' ratings from the record among stationary players,
   the ladder held (`bnb.league.ratings.fit_ladder`). The only games among stationary
   players are each snapshot's own update, credited when it is taken.
2. **The live rating:** the lanes'; without lanes, from this update's training games alone,
   against the entries' current ratings (`bnb.league.ratings.rate_live`, ties as half
   wins). A new snapshot starts at it.
3. **The averages:** the lanes' rating of an average they rate; otherwise each rated on its
   last 200 games against the live rating of each game's update
   (`bnb.league.ratings.rate_against`).

A sweep gives an infinite estimate; the gauge reports it and keeps the last finite value,
which sampling uses. Without lanes the gauge's live and averaged estimates are the
`rating/live/<mode>`{.plain} and `rating/average/<mode>`{.plain} series; they are never
published.

## Resuming

A training checkpoint holds the opponent slots' stacked weights and buffers, the rollout
carry (every row's carry, flags and rung) and the league's state: its entries, gauges,
sampling state, the lanes' record, and the averages' and snapshots' weights. A resumed run draws the same
opponents and reports the same ratings as one run straight through
(`tests/runs/test_resume.py`). A fork starts a fresh league from its model.

## Metrics

`League.updated` returns, under `league`: per mode the live and averaged ratings and their
standard errors (the lanes'), every lane player's rating, and per entry its rating,
sampling share and score against the live policy this update;
the pool's entries, snapshots, the slots' entries, the draining slot and the incoming
entry; and the games of the update by opponent kind.

## Published ratings

`bnb ratings RUN` plays an offline tournament over the run's snapshots and averages, the
ladder and the players `--with` names, game preset by preset
(`bnb.league.tournament.rate`). The match engine (`bnb.league.matches.play`) seats each
pair's first player on team 0 in half its games and on team 1 in the other half; policies
of any architecture play.

- The first batch spreads its games evenly over the pairs; each later batch splits its
  games by each pair's A-optimal value (`bnb.league.ratings.allocate`), at least one game
  per pair, rounded up to an even count.
- After each batch the ratings are the Bradley–Terry fit with standard errors
  (`bnb.league.ratings.fit_ratings`, draws as half a win, one prior game), shifted so the
  scripted agent is at 1000. Only the scripted agent is pinned; random is fitted.
- The tournament stops when every standard error is under `--target-stderr`, or when
  `--budget` games are spent.
- The leaderboard rows (`calibrated_elo`) are keyed by game version, mode, map and fleet
  size, with the standard error and the games behind each. They are written to the run's
  `artifacts/ratings/` with the inputs' hashes.

<!-- not-runnable: needs CUDA and a run -->
```bash
uv run --no-sync bnb ratings 2026-11-02-mixed-purple-pumpkin --target-stderr 25
```

`bnb head-to-head A B [C D ...] --games N` plays only the named pairs on the same engine.
A player is a policy file, a run (its newest checkpoint), `scripted`, `random` or a rung
such as `rung_0.5`. Per pair it reports the wins, losses and draws, the score with its
standard error, and the split by preset, and records them under `artifacts/head_to_head/`.

<!-- not-runnable: needs CUDA and a run -->
```bash
uv run --no-sync bnb head-to-head 2026-11-02-mixed-purple-pumpkin scripted --games 64
```
