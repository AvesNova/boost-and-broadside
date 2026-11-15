---
covers: [src/bnb/league/, src/bnb/runtime/opponents.py]
---

# The league

<!-- generated: section-status -->

The league chooses the learner's opponents during training, rates the learner as it trains,
and rates finished policies afterwards. The training side is `bnb.league.league.League`,
attached to a trainer as its hooks; inside the rollout it acts through one controller, the
league's seat (`bnb.runtime.opponents.OpponentSlots`). The offline side is
`bnb.league.tournament`, behind `bnb ratings` and `bnb head-to-head`, `bnb.league.rating_pool`,
behind `bnb pool`, `bnb.league.crossover`, behind `bnb crossover`, and
`bnb.league.evaluation`, behind `bnb evaluate`.

## The pool

Beside the entries below, the six cheese agents (`cheese:zone_rusher` and the others,
[the Imugi page](imugi.md#cheese-agents)) are entries by default,
`LeagueConfig.cheese` (`()` turns them off); they are sampled by the same rule.

The default lineup (`bnb.runtime.train.LINEUP`) gives each game of the training mix two
pools: self-play and the league's seat, half each. Self-play is not a pool entry; it keeps
its fixed share. The entries (`bnb.league.pool.Entry`) are:

- **snapshots** of the live policy, taken on the schedule `LeagueConfig.snapshots`
  (`bnb.runtime.snapshots.LEAGUE_SNAPSHOTS`, decided 2026-11-14: at 5, 10, 15, 20, 25, 30,
  40, 50, 60, 75, 90, 110, 130 and 160M environment steps, then every 30M, about 100 to
  200 Elo apart on rl-v20's curve; to be re-derived after the game freeze). With
  `snapshots=None` the league measures a period once, about 30 minutes at the pace of the
  slowest of the second to fourth updates, and records it in `run.json`{.plain}. A
  snapshot is the weights that played the update it was taken at;
- **the averages:** the uniform average and the exponential moving average of the live
  policy's weights, folded in after every update (`bnb.league.pool.Averages`, through `average_into`, the
  decay `LeagueConfig.exponential_decay` per update; behaviour cloning keeps the same
  two with the same settings). The uniform average starts after
  the run's learning-rate warm-up (`LearnerConfig.warmup_steps`, 10M environment steps by
  default, or `LeagueConfig.uniform_from` when set: `TrainConfig.uniform_start`), so a
  warm start's optimizer, critic and advantages settle first; until then it is the live weights;
- **Imugi** at full strength (temperature 1, `scripted`;
  `bnb.league.pool.IMUGI_ENTRY`), the pool's only scripted entry. The pool holds no
  weaker rung and no random player (the author, 2026-11-12). The stationary ladder
  (`bnb.runtime.opponents.LADDER`: random, the rungs and Imugi) is still
  played by the offline tournament ([Published ratings](#published-ratings)), and the rating lanes can play its
  rungs (`bnb.runtime.lanes.ladder_chain`); it is not part of training.

**Retention** (`bnb.league.pool.retained`) keeps the newest 8 snapshots and, of the older
ones, the one nearest each age of 2^k snapshots for k from 3. The run's `league/` directory
holds each retained snapshot as `step_<steps>/checkpoint`, with the two averages at the same
step beside it (`uniform_average`, `exponential_average`), all policy files.

## Sampling

Half of each mode's batch is self-play, a fixed share (`bnb.runtime.train.LINEUP`); the
league's games are the other half. An entry's sampling weight is `(1 − p)²`
(`bnb.league.pool.sampling_weights`, `LeagueConfig.sampling = "hard"`), where `p` is the
predicted probability that the live policy beats it under that mode's live gauge: the
entries the live policy loses to most weigh most, as in AlphaStar's prioritised fictitious
self-play (decided 2026-11-14). Every entry is weighed alike: the snapshots, the two
averages, Imugi and the cheese agents, with no uniform floor
(`LeagueConfig.floor`, 0), so an entry the live policy always beats is not drawn. When every
weight is zero the draw is even. Each mode samples by its own gauge. For an ablation,
`sampling = "variance"` weighs `p (1 − p)`, the entries whose result is least certain,
and `floor` spreads a share of the weight evenly.

## Slots, Imugi and the cut

The league's seat plays every opponent row of the batch in the captured rollout graph:

- **Network slots** (`LeagueConfig.slots`, 4): their parameters are stacked, one set per
  slot. Slot k plays rows k, k + slots, …, so every slot plays every mode. Each set runs on
  its own rows inside the one compiled graph, and each row keeps its network's carry.
- **Imugi:** one full-strength scripted pass over every row, with its carry
  for every row in the seat's carry. A slot can hold Imugi; its rows then
  take the scripted action (selected per row with `where`) and tally to the column after
  the slots'. The pass runs every tick whether or not a slot holds the agent: the captured
  graph has fixed shapes.
- **The clean cut** (`League._cut`{.plain}, `OpponentSlots.cut`): every `cut_every`
  updates (2), at the update boundary, `cut_slots` slots (1, round robin) each draw an
  entry from the modes' weights, mixed by the modes' shares of the opponent rows. A slot
  whose draw differs from what it holds takes it in place: its weights are copied into the
  slot's set. The games in progress on its rows play on against the new entry. Their
  ticks still train the learner (they are on-policy data); their results are credited to
  no one: the rows' `whole` flags are cleared, so the tally skips each row's game until its
  next reset. Drawing the entry a slot already holds changes nothing.
- **The opponent's carry at a cut** (`carry_at_cut`): `"keep"` (default) leaves the old
  occupant's carry when the old and the new entry are both networks
  (`bnb.league.league.same_build`: every network of the training league is the live
  policy's build, in its one stacked template), else resets; `"reset"` zeroes the
  network's trunk carry on the cut rows (Yemong's state and convolution buffer,
  `YemongController.forget`) and keeps the belief part (observations, last sightings,
  forecasts), the previous actions and the position offset. Imugi's carry
  runs on every row throughout.
- **The hand-over test** (`bench/memory_handover.py`, CPU, 2026-11-06): 16 fresh games of
  calm-fern's 73M live policy A against itself, the opponent switched at 60 s to B: B
  from A's carry ("keep") and from a reset trunk ("reset"), scored by KL over the 42 joint
  actions per living ship against B warmed from the game's start, on the same
  observations; "gap" is A playing on (no hand-over). Near: B is the 88M policy; far: the
  7.3M one. Every model draws the same position offset at a reset, so the test sees the
  memory, not the frame.

  | Seconds after | Mode, pair | Gap | Keep | Reset |
  |---|---|---|---|---|
  | 0-0.25 | frontline, near | 0.150 | 0.043 | 0.579 |
  | | frontline, far | 1.14 | 0.138 | 0.451 |
  | | deathmatch, near | 0.075 | 0.029 | 0.622 |
  | | deathmatch, far | 1.63 | 0.173 | 0.490 |
  | 0.5-1 | frontline, near | 0.135 | 0.007 | 0.075 |
  | | frontline, far | 1.17 | 0.021 | 0.059 |
  | | deathmatch, near | 0.071 | 0.003 | 0.032 |
  | | deathmatch, far | 1.63 | 0.027 | 0.054 |
  | 2-4 | frontline, near | 0.150 | 0.002 | 0.044 |
  | | frontline, far | 1.24 | 0.002 | 0.004 |
  | | deathmatch, near | 0.076 | 0.001 | 0.004 |
  | | deathmatch, far | 1.60 | 0.002 | 0.004 |

  "Keep" is closer at every horizon until both reach the floor, about 0.001, from 4-8 s
  ("reset" is still 0.005 there for frontline's near pair). A reset opponent plays less
  like B than A does for its first quarter second. The full table, by window to 32 s, is
  in `docs/rewrite/sections/rl-round-five-notes.md`.

<!-- not-runnable: needs the calm-fern policies -->
```bash
P=../runs-v2/players-calm-fern
uv run --no-sync python -m bench.memory_handover $P/step_73400320-live \
    $P/step_88080384-live $P/step_7340032-live --mode frontline --threads 8
```

- **Why a cut, not a drain.** The drain it replaced switched each row of one slot at its
  next reset, one slot at a time, so a draw stayed resident for slots × one game clock
  (about 56 updates in run 2026-11-06-mixed-calm-fern) and the late opponent mix followed
  weights drawn long before. The simulations
  (`docs/rewrite/sections/league-sampling-sim.md`, rounds 1 and 2) put the RL case (ladder
  out, 4 slots) at 0.62 open games under the serial drain and 0.84 under a cut of one
  slot every update, at one network pass per slot instead of slots + 1; the author chose
  one slot every 2 to 4 updates. The cost is the cut games' results and the new
  opponent's first seconds of play away from its own (the hand-over test above).
- **The draws** use keyed uniforms (`bnb.streams.keyed_uniform` of the run's swap key
  and the swap count), so they need no generator state and repeat exactly.
- **The results:** the rollout tallies every ended game but each row's first by what the
  opponent row played and by mode, from the learner's side; the league reads the tally
  once per segment.

## The ratings

Every rating during training comes from one fit per mode, `bnb.league.lanes.RunRatings`:
one Bradley–Terry model of every rated player with Imugi pinned at 1000
(`bnb.league.ratings.fit_ratings`, draws as half a win, one prior game against the scripted
agent, half won, which keeps a sweep finite), refitted once per update. Its games are every
game a rated player played:

- **The rating lanes' games** ([The training loop](training-loop.md#the-rating-lanes)):
  rated matchups played beside the training games, in the same rollout.
- **The league's training games:** per mode, the live policy's games against each entry
  it played this update (`League.record`), as games of the live policy against that entry.
  Self-play is not an entry and stays out. A snapshot taken at an update is credited with
  that update's games as well, since its weights played them. The uniform average's games
  are left out until it has started: before then it is the live weights.

Without lanes the fit reads the league's games alone.

- **Names:** the fit names players by the league's entry names (`snapshot_<steps>`,
  `exponential_average`, `uniform_average`, `scripted`, and any rungs the lanes play)
  and `live`. A lane plays
  the entry a league slot holds as `league_<k>`, which `RunRatings.resolved` maps to that
  entry, the one mapping of lane players and entries onto the fit.
- **The record:** games are kept by mode and by who played them. A game of two stationary
  players (the rungs, the league's snapshots) is kept for the whole run; one with a moving
  player (the live policy, an average) for its last `LaneConfig.window` updates (10).
- **Attribution:** the rollout tallies each ended training game under the column its row
  played as it ended (a slot's set or Imugi), and the league reads the tally once per
  segment; a cut comes after the update's last read and clears its rows' `whole` flags, so
  each tallied game was played whole by the entry it is credited to and a cut game by
  no one (`tests/sections/test_league_slots.py::test_a_cut_game_is_credited_to_no_one`).
  A training game spanning an update is credited, like a lane game of the live policy, at
  the update it ends in. Each row's first game is left out, as in the lanes: the rollout's
  clocks are staggered, so it is cut short and leans to ties
  (`bnb.runtime.opponents.whole_game_begins`, the one rule for both).
- **The averages move:** a slot holding an average, and the lanes' `league_<k>` holding
  one, take its weights after every update, as the lanes' own average players do, so an
  average is one moving player wherever it plays.
- **The league's entries in the lanes:** the lanes play the entries the league's network
  slots hold, copied in when a slot takes another (`League._load_lanes`). A snapshot's
  games are credited to it by name, so it keeps its games and its rating after it cycles
  out, and a new one joins the fit as soon as its games connect it; a slot holding the
  Imugi plays unrated. A lane game in progress when a slot's lane player changes is
  left out, as is each lane row's first game (cut short by the staggered clocks).
- **Who is rated:** a player whose games connect it to Imugi; none is reported
  before its matchups have played.
- **The ladder:** rungs the lanes play (chained from Imugi down,
  `ladder_chain`) are rated from their games; training plays none. A fixed scale would be
  far off: the earlier probability rungs' `1000 p` put rung_0.95 50 points below the scripted
  agent, where it scored 0.065 against it, about 530 points below.
- **The series:** per mode `rating/live/<mode>`{.plain} and `rating/average/<mode>`{.plain} (the
  exponential average), their standard errors, and every rated player's
  `lanes/rating/<mode>/<player>`{.plain}. A BC run's lanes rate the live clone and both averages the same way
  (`bnb.league.lanes.LaneHooks`), the uniform average as
  `rating/uniform_average/<mode>`{.plain}, and each of its policy files keeps its
  player's per mode (`PolicyRecord.ratings`): a warm start's live rating starts from its
  parent's, not 0; a mode the parent has no rating for (a clone trained without lanes)
  starts at Imugi's 1000.

The league's games are about half of the information. Refitted offline on run
2026-11-06-mixed-calm-fern's first 29 updates (its per-update lane games and `pool/entries`
table), the lanes alone left the live rating's standard error at 95–112 and the
exponential average's at 82–106; with the training games they are 59–66 and 63–71, and the
live policy is rated from the first update instead of the twelfth. The synthetic case is
`tests/sections/test_lanes.py::test_the_leagues_games_shrink_the_ratings_errors`.

Against Imugi alone a policy 1000 points above it would be rated blind: its
games there carry 1/60 of an even game's information. Through a chain of snapshots, whose
stationary links gather games over the run, the same window of live games rates it several
times more precisely
(`tests/sections/test_lanes.py::test_a_strong_player_is_rated_far_better_through_a_snapshot_chain`).
The measured game counts and errors are in `docs/rewrite/sections/rollout-lanes-notes.md`.

**Without lanes** the league's own games against Imugi connect the live
policy to it, and through the live policy the entries it plays. A run that wants every
entry rated from its first updates keeps its rating lanes.

## The live gauge

During training each mode has a gauge (`bnb.league.gauge.Gauge`) that steers sampling. It
reads every rating from the fit once per update; an entry the fit does not rate this update
holds its last rating:

- Imugi its fixed 1000 until the fit first rates it;
- a new snapshot the live rating of its update, until its games rate it;
- an average the live rating until the fit first rates it;
- the live policy its last fitted rating, from a warm start's parent's at the start.

**The guard.** A near-sweep rates finite and near its likelihood's maximum
(`tests/sections/test_league_ratings.py::test_a_near_sweep_against_the_ladder_rates_finite_near_its_maximum`,
the run's case: 28 of 29 against the rungs, which the old ladder fit put past 10,000).
Each update counts, per mode, the entries that have played (credited training games) but
are rated more than 1,500 from the live policy or still have an infinite error
(`League.gauge_anomalies`{.plain}, `pool/gauge_anomalies/<mode>`{.plain}); the trigger
rule `gauge_runaway` alerts when any is.

## Resuming

A training checkpoint holds the opponent slots' stacked weights and buffers, the rollout
carry (every row's carry and flags) and the league's state: its entries, gauges,
sampling state, the ratings' record, and the averages' and snapshots' weights. A resumed run draws the same
opponents and reports the same ratings as one run straight through
(`tests/runs/test_resume.py`). A fork starts a fresh league from its model.

## Metrics

`League.updated` returns, under `league`: per mode the live and averaged ratings and their
standard errors, every rated player's rating, and per entry its rating,
sampling share and score against the live policy this update;
the pool's entries, snapshots and the slots' entries; the update's cut
(`pool/cut/<series>`{.plain}: the slots drawn, the slots that took another entry, their
rows, the games left uncredited and the rows whose opponent carry was reset) and
the slots' mix after it (`pool/slot_share/<kind>`{.plain}, `pool/distinct_held`{.plain});
the gauge's anomalies; and the games of the update by opponent kind
(`pool/game_share/<kind>`{.plain}, `pool/score/<kind>`{.plain}).

## Published ratings

`bnb ratings RUN` plays an offline tournament over the run's snapshots and averages, the
ladder and the players `--with` names, game preset by preset
(`bnb.league.tournament.rate`). `--no-ladder` leaves out the rungs below Imugi,
which always plays: it anchors the scale. `--no-league` leaves out the run's `league/`
policies, so the field is `--with`'s players and the ladder. `--imugi-plays A B ...`
keeps Imugi to the named players; the others reach the anchor through their
games with them. `--out DIR` writes the result under `DIR/artifacts/ratings/` instead of
the run's.

**The match engine** (`bnb.league.matches.Arena`) is one rollout per preset, built and
compiled once. Its layout is fixed: `--blocks` matchup blocks of `--block-environments`
environments each. Each block is two slots, its first seat and its second, both played by
one controller (`bnb.league.matches.MatchSeat`):

- A **network slot** plays on one **lane** of its player's **group**. The networks of
  one build (the same model and build record) form a group, with a template and a forward
  of its own; a field may mix builds, and a block's two seats may come from different
  groups (two Yemong arms that differ in a build setting, for one). Each group has a fixed
  number of lanes: a lane holds one slot's weights and the slot it plays, both copied in
  place before a batch, and the group's lanes run as one forward over their stacked
  weights, each lane on its slot's rows (`bnb.runtime.stacked`: `torch.vmap` over
  `torch.func.functional_call`, as the rating lanes and the league's slots do), writing
  back each slot's actions and carry; an idle lane runs on a slot of its own and writes
  nothing. The compiled stage therefore holds one copy of each group's network, however
  many lanes it has. The lanes split the slots over the groups in proportion to their
  players, but a group gets no more lanes than a batch can fill (one slot per block for a
  group with one player, two with more: `bnb.league.matches.default_lanes`), so **a batch
  costs one network pass over every row however many builds the field mixes**. A batch may hold at
  most a group's lanes of its slots: the tournament's allocation keeps to them, and
  head-to-head gives each group the most of its slots any of its batches holds. One
  build is one group whose lanes are every slot, and plays as before.
- A **scripted slot** (a rung or a teacher) takes the scripted pass, one over every row,
  at the slot's temperature, share and floor, written in place.
- **A batch** copies the chosen players into the slots, in place, so the captured graph
  stays valid; every environment starts a fresh game (`Rollout.restart`). Environment i of
  a block seats the block's first player on team i mod 2, so a block's games split evenly
  between the seats. Each environment's first game is played to its end and counted once.

**Mixed builds.** On frontline, a field of four builds (BC arms A, R, C and D, with an
average of A's build) plays a batch in about the time one build takes: 20.9 s against
21.1 s at 2 blocks of 16 environments, 76.5 s against 79.0 s at 8 blocks of 16, and
148 s at the default 8 blocks of 128, where a single-build field takes about 145 s
(2026-11-07).

**The tournament:**

- **Targets.** Every player's standard error should be under `--target-stderr`; the
  players named by `--decision` under the stricter `--decision-stderr`; and each
  `--contrast A:B`, the difference between two players' ratings (two arms' final
  checkpoints, say), under `--contrast-stderr`. The errors are relative to the scripted
  agent, so a contrast is the target to name when what matters is how two strong players
  compare: their difference can be resolved by their games with each other even when
  both beat Imugi every game.
- **The plan of a batch** (`bnb.league.design.plan_batch`). The batch's blocks minimise the
  weighted A-optimal criterion, the sum over the unmet targets of each one's variance
  after the batch over its target squared. The variance after the batch counts every game
  so far, so a pair that has had many games is worth less. A game is valued
  at its information p (1 − p) at the far end of a one-standard-error interval of the
  pair's rating difference (at most 200 Elo wide), so a pair whose result is near certain
  is discounted. The plan first finds the continuous design, the share of the batch's
  games each allowed pair should get (the multiplicative algorithm for A-optimality), then
  rounds it to whole blocks one at a time: each block goes to the design's heaviest pair,
  and the design is solved again for the games left over the pairs that still fit the
  batch, at most `--players-per-batch` distinct players (the engine loads each player a
  batch holds; default as many as its slots) and each network group's lanes.
- **The fit** (`bnb.league.design.fit_posterior`). After each batch the ratings are the
  Bradley–Terry maximum-likelihood fit on every game so far, draws as half a win, with no
  prior, shifted so Imugi is at 1000. Only Imugi is pinned;
  random is fitted. The standard errors are the inverse of the Fisher information.
- **Bounds.** The likelihood has no finite maximum for a player who won every game it
  played, or lost every one, or for a group of players joined to the rest only by such
  sweeps. Such a player is reported as a bound, not a number: `bnb pool table` shows
  "> 2400.0 (bound)" or "< -216.1 (bound)", and the `curves` export gives the bound as its
  `calibrated_elo`, no `stderr` and the status `above` or `below`. The bound is one-sided at
  5%: the rating at which the sweeps it made (or suffered) against players with a value
  had a 5% probability, those players held at their values (one who beat the scripted
  agent in all of 96 games: > 1600). Within a group the differences keep their maximum
  likelihood and standard errors, so a contrast between two arms that both sweep the
  Imugi is still resolved by their games with each other. A target on a bound is
  left out of the stopping check; a player with games but no route to Imugi
  is shown as undetermined, and its target stays unmet.
- **Planning around bounds and new players.** For choosing games only, a bounded player is
  placed 200 Elo beyond its bound on the open side, so its next games go to stronger (or
  weaker) opponents than the ones it swept. A new checkpoint in a rating pool, with no
  games, is placed at the mean of the other runs' checkpoints of its weights kind at the
  same steps (interpolated between the nearest steps, or the nearest steps' mean outside
  them); with no other run in the pool, at its run's previous checkpoint; else at the
  Imugi. After its first batch the plan uses the fit only. None of this placement
  enters a reported rating.
- **Why no prior.** A prior moves the reported ratings. The first fit's prior was one
  virtual game against Imugi, half won. A virtual game's information is
  p (1 − p) at the estimate, which vanishes as a clean sweep drives the estimate out: after
  a sweep the estimate grows as the log of the games and the information stays about one
  game's, so the player's standard error never falls (about 175 Elo after 96 games or after
  960). The earlier greedy allocation valued the next block at the current estimate, saw
  the largest variance there and kept choosing that pair: in one evaluation it spent 672
  of 1152 games on a policy against Imugi (576 wins, no losses) and 96 on the
  pair that mattered. `tests/sections/test_rating_design.py` reproduces both allocations. A
  Gaussian prior of 1000 Elo replaced it for a day (2026-11-14); it pulled a sparse field
  of strong players toward 1000 (the win/loss deathmatch re-ratings sit 60 to 120 Elo
  higher without it) and was dropped.
- **Out of reach.** A target the remaining budget cannot bring in is left out of the
  criterion while other targets can be met (its batches needed at the better of the plan's
  rate for it and its best single pair's). When every unmet target needs more than
  `--hopeless` times the batches left (default 10), the tournament stops and logs which
  targets and why, rather than spend its budget on games that hardly move them;
  `--hopeless inf` plays the budget.
- **Scoring:** every game counts as a win, a draw or a loss; a team deathmatch game by
  the sign of the final kill difference (the kill lead, decided 2026-11-14).
  `--deathmatch-score kill_share` scores a team deathmatch game instead as its kill share:
  the first player's kills over both sides' (0.5 with none), taken by the fit as a draw's
  half is (`bnb.league.tournament.kill_share_scored`). `bnb head-to-head` and
  `bnb pool create` take the same option. The scoring is applied when
  the games are fitted, not when they are recorded (see the record below), so the option
  chooses only how the fit reads the games.
- The tournament stops when every target is met, when the next batch would pass `--budget`
  games, or when the targets left are out of reach (above).
- The leaderboard rows (`calibrated_elo`) are keyed by game version, mode, map and fleet
  size, with the standard error and the games behind each. They are written to the run's
  `artifacts/ratings/` with the inputs' hashes.
- **The record:** after each batch one JSON line is appended, and synced to disk, to
  `<preset>.batches.jsonl` in `--out` (else the run's `artifacts/ratings/`): the batch's
  index, its blocks' matchups and each block's full outcome, the games so far, every
  player's rating and standard error, the worst error, the batch's seconds and the wall
  time, the games reused (`reused_games`), the targets the plan left out as out of reach
  (`out_of_reach`), each player's identity (`identities`, name to identity) and the scoring the batch's ratings were fitted under (`scoring`: `kill_share`
  or `win_loss`). A block's full outcome is kept in every mode, whatever the scoring
  (`bnb.league.tournament.outcomes`): `results`, the first player's wins, draws and losses
  by the game's own rule (in team deathmatch the sign of the final kill difference);
  `kill_share`, the sum of its games' kill shares; and `kills`, the first player's and the
  second player's kills summed over the block's games. A batch line looks like

  ```json
  {"batch": 0, "game_version": 19, "scoring": "win_loss",
   "matchups": [["scripted", "a"], ["a", "b"]],
   "results": [[31, 4, 93], [70, 9, 49]], "kill_share": [41.2, 72.9],
   "kills": [[210, 402], [377, 241]], "games": 256, ...}
  ```

  The record keeps sums per block, not per game: the fit uses only each pair's sums, and
  a batch of 8 blocks of 128 games would otherwise carry 1024 rows a line. The kills are
  read once after the batch, from the counts the games keep on the device, as the
  kill-share scoring always read them; nothing is added per game.
- **Scoring at fit time** (`bnb.league.tournament.recorded_results`): a batch with its
  full outcome is read under either scoring, as its wins, draws and losses or as (the
  summed shares, 0, the games less them). A record written before the full outcome was
  kept holds only the results of the scoring it names (a batch that names none was
  scored by kill share in team deathmatch, the default then, and by a win, draw or loss
  elsewhere: `bnb.league.tournament.UNNAMED_SCORE`),
  and is read only under that scoring; in a record that mixes the two, each batch is
  judged on its own.
- **Resuming:** a tournament that finds its preset's record rebuilds the games from it and
  goes on (the record's field must be its own: same names, same order, same identities,
  else it stops and asks for another `--out`; an older batch of the other scoring, or a
  batch of another game version, stops it too); a finished one plays nothing.
- **Refitting without playing:** `--budget 0` plays no games and builds no engine. The
  field is fitted from its record in `--out` and the `--reuse` records alone, under
  `--deathmatch-score`, and the leaderboard is written as usual. A tournament is
  re-rated under the other scoring with

  <!-- not-runnable: needs a run and its record -->
  ```bash
  bnb ratings RUN --out DIR --budget 0 --deathmatch-score win_loss
  ```

  and with players added or left out by naming the new field, a fresh `--out` and the old
  directory as `--reuse DIR` (only games whose two players are both in the field count,
  so a newcomer needs a budget to be rated).
- **Reusing earlier games** (`--reuse DIR [DIR ...]`, `bnb.league.tournament.reuse_record`):
  every `<preset>.batches.jsonl` below the directories (as `--out` writes them; the run's
  own record is left out) adds each recorded game whose two players are both in the field
  to the starting games, whatever its field or arena layout. The fit starts from them, and
  batches are played only while an error is above the target (none when the reused games
  meet it); `--budget` counts only the games played here. Players are matched by identity
  (`bnb.league.tournament.identity`): a policy's is the SHA-256 of its
  weights file as its checkpoint.json records it (with its ablations), so a moved
  policy still matches and other weights at the same path do not; a scripted player's
  (the ladder's, a teacher's) is its name. Each record's reused games, matched players and
  skipped names are logged. Older batches of the other deathmatch scoring are left out
  and counted in the log; a record made only of them is reported with every player
  skipped and the reason. A **legacy record** (written before identities were recorded)
  holds no hashes, so it trusts its paths: each name is resolved as it would be now, and
  is skipped when it no longer resolves, or when its policy's checkpoint.json `time` is
  after the record's first batch ended. That catches a rewind, which moves a policy to
  `policies/rewound_<n>/` and later writes other weights at the old path: the old path's
  games are skipped, and the moved policy cannot be matched to them either (nothing in the
  record says which weights played). Each batch also logs a line with the local clock times it expects to finish
  (the errors falling as one over the root of the games) and, at the latest, when the
  budget is spent, at the mean batch's time.
- **A guard:** a batch whose games run past the game clock's ticks (plus two segments)
  raises rather than playing on.
- **Other fields.** `rate` takes another anchor (rated 1000), the balanced allocation
  (`TournamentConfig.allocation = "balanced"`, `bnb.league.tournament.balanced_blocks`:
  each block to the allowed pair with the fewest games, so a budget of every pair's games
  is a full round robin) and, with `statistics`, keeps each block's first-game statistics
  in the record (`bnb.league.matches.STATISTICS`: kills, deaths, shots, hits, living
  ship-ticks, their summed speed and a network's summed entropy, per side). A player may
  fly one ship type with every ship of its team (`Player.ship_type`;
  `bnb.league.matches.ship_type_players` gives Imugi flying each type of a
  set); the engine then sets each environment's ship types per team (`bnb.game.api.reset`'s `ship_type`). The [ship-type
  tournament](game/real-scale-tournament.md) is such a field.

<!-- not-runnable: needs CUDA and a run -->
```bash
uv run --no-sync bnb ratings 2026-11-02-mixed-purple-pumpkin --target-stderr 25
uv run --no-sync bnb ratings RUN --with ARM_A ARM_B --contrast ARM_A:ARM_B --contrast-stderr 20
```

`bnb head-to-head A B [C D ...] --games N` plays only the named pairs on the same engine,
in blocks that divide `N` (`bnb.league.tournament.head_to_head_layout`).
A player is a policy file, a run (its newest checkpoint), `scripted`, `random`, a rung such
as `rung_T2`, or a teacher, `scripted@T=<t>,floor=<f>` (the agent at its own temperature
`t`, then a uniform joint action with probability `f`; either key optional, T = 1 and
floor 0 by default, so `scripted@T=2` plays as `rung_T2`). Per pair it reports the wins,
losses and draws, the score with its standard error, and the split by preset, and records
them under `artifacts/head_to_head/`.

<!-- not-runnable: needs CUDA and a run -->
```bash
uv run --no-sync bnb head-to-head 2026-11-02-mixed-purple-pumpkin scripted --games 64
```

**The break-even ship count.** `bnb crossover POLICY` (`bnb.league.crossover`) measures
how many of a policy's ships hold 20 of Imugi's, per mode. The opponent always
fields `--opponent-count` ships (20); the policy fields a varying number, from
`--minimum-count` to `--maximum-count` (1 to 30: above the opponent's count, so a policy
weaker than Imugi at parity still crosses inside the range rather than by extrapolation).
Each mode's game (by default the real-scale frontline, deathmatch and elimination presets)
is built once at the larger fleet's size (30 a side by default) on the preset's own arena
(`--map-rule preset`; every map is fixed-size until an agent has trained successfully on
variable-size maps, the author's rule of 2026-11-15). `--map-rule density` grows the arena
by the density rule for the opponent's fleet (`bnb.game.api.arena_radius_for`), the
map-size arm's opt-in. Each team's unused slots are benched
([benched ships](game/modes.md#benched-ships)): they stay dead and out of play, and team
deathmatch's winning score counts the enemy's active ships. Every game draws its own
lineup, each slot Yongma or Jujak with equal chance (`--lineup-types`), both teams flying
it slot for slot, keyed by the seed, the batch and the game; a team's benched ships are its
lineup's last slots. A game is scored by its
own rule: a win, a draw (half a win) or a loss.

The measurement runs in batches of the engine's layout, `--blocks` blocks of
`--block-environments` games, each block at one count, its games split evenly between the
seats; a network's blocks run as one stacked forward, as in head-to-head.

- **Fit.** One logistic regression of the policy's score on log(count) over every game so
  far, with a binomial likelihood on the score. The break-even count N\* is where the
  fitted curve crosses one half, and its standard error comes from the fit's covariance by
  the delta method; the ratio 20 / N\* carries the same relative error. Draws make the
  binomial standard error a slight overestimate. While the outcomes are separated (every
  count below some point lost outright, every count above it won outright), the fit has no
  maximum, and the result is a bound on N\* instead.
- **Allocation.** The first batch spreads its blocks over `--bracket` (6, 10, 14, 20).
  After that, with a crossing, each block goes in turn to the count that most reduces the
  predicted variance of log N\*, averaged over N\* at the estimate and one standard error
  either side, so the games gather at the counts next to the crossing. With separated
  outcomes the blocks go inside the bound, or halve the lowest count played (every game
  won) or double the highest (every game lost) to extend the range.
- **Stop.** A mode stops when N\*'s standard error reaches `--target-stderr` ships (0.3),
  when its games reach `--budget-games` (8192; the batch that reaches it is the last), or
  when the bound is final: every game won at the minimum count, or lost at the maximum.

Every batch is appended to `artifacts/crossover/<preset>.batches.jsonl`{.plain} of `--out`
(else the policy's run, else the runs directory) as it ends: each block's count and its
wins, draws and losses, and the fit after it. A line per batch prints the counts, the games,
N\* with its standard error (or the bound) and an ETA as a clock time, with the budget's as
the worst case. A rerun reuses the record's batches of the same measurement (the game
version, the players' identities, the game, the seed, the opponent's count and the lineup
types) and continues past them, so an interrupted measurement resumes and a finished one only reports. The
result (per mode N\*, the ratio, the stop reason and the games per count, with the command
and the policy's SHA-256) goes to `artifacts/crossover/<time>/result.json`{.plain}.

A benched slot costs about what an active one does: the game, Imugi and the
network run at fixed shapes over every slot, so a benched ship is a dead ship carried
through the same kernels. A 20v20 tick on the CPU (128 games, eager) took the same time
with the policy at 4, 10 or 20 ships, within the measurement's noise of about 10%, and a
12v12 game about two thirds of it. At a count N the default 30-a-side game therefore spends
50 - N of its 60 slots on benched ships, about 1.5 times a 20-a-side game's cost per game
by slot count; `--maximum-count 20` plays the 20-a-side game where a policy is known to
cross below 20.

<!-- not-runnable: needs CUDA and a policy -->
```bash
# N* against 20 scripted ships in the three real-scale modes
uv run --no-sync bnb crossover RUN/policies/step_<steps>/exponential_average
# one mode, a tighter target
uv run --no-sync bnb crossover POLICY --presets real_scale_deathmatch --target-stderr 0.15
```

**Ablated players.** A Yemong policy may play ablated, in `--with` or in head-to-head:
`PATH:rotary`, `PATH:forecast`, `PATH:absent_zones` or a combination such as
`PATH:rotary+forecast` (a run's name works as the policy too). An ablated player is the same weights and build, so it shares its policy's
group and stacked weights; the seat holds each row's ablations and passes them to every
Yemong group's network (`bnb.models.yemong.model.Ablations`), so only that player's rows
are ablated and the captured graph stays one graph. A tournament without an ablated player calls the networks
without them. `rotary` makes the attention's rotary rotation the identity (every angle 0,
token and map tables); `forecast` stores nothing back to the belief, as
`yemong.forecast_feedback=false` does; `absent_zones` takes the absent zone slots (the
learned absent token of a mode without zones) out of every attention as keys and values,
so no head can attend to them (through the fused kernel's key mask; a tournament without
an `absent_zones` player attends as before). All are inference-time ablations of a model
trained with the feature on: they measure how much the trained policy relies on it, not
what a model trained without it would reach.

<!-- not-runnable: needs CUDA and a run -->
```bash
uv run --no-sync bnb ratings RUN --with POLICY POLICY:rotary POLICY:forecast
```

**Players at a temperature.** A network may play at a sampling temperature T, in
`--with` or in head-to-head: `PATH@T=0.7`, after any ablations (`PATH:rotary@T=0.7`).
Its joint action is drawn from the policy's logits divided by T, the whole 42-way joint
at once, so power, stick and shoot are tempered alike; T below 1 sharpens the policy
toward its most likely action, above 1 flattens it toward uniform. Like an ablated
player it is the same weights and group; the seat holds each row's temperature in place
(1 on every other row, which samples exactly as without one), so the captured graph stays
one graph. A tempered player's identity is its weights' and its temperature
(`sha256:<hash>@T=0.7`), so reused games keep the temperatures apart. Each block's
statistics also sum the network's entropy (nats, of the distribution it sampled from)
over its living ship-ticks, so `entropy / ship_ticks` is the mean entropy it played at.
Imugi's own temperature is the teacher's syntax, `scripted@T=<temperature>`.

<!-- not-runnable: needs CUDA and a run -->
```bash
uv run --no-sync bnb ratings RUN --with POLICY@T=0.7 POLICY POLICY@T=1.4
```

## The decision suite

`bnb evaluate A B --out DIR` (`bnb.league.evaluation`) decides between two policies,
normally two arms' final checkpoints, by win probability over a fixed set of conditions:
head-to-head A against B in each mode and at several team sizes, and each policy's
break-even curve against 20 scripted ships (the crossover's fit). Every condition plays
its preset's own arena at every team size (`--map-rule preset`; every map is fixed-size
until an agent has trained successfully on variable-size maps, the author's rule of
2026-11-15), and the density rule's arenas are the map-size arm's opt-in
(`--map-rule density`). Frontline break-even is left out of the score until the map-size
arm reports (`--unscored`). A group-sequential test
(`bnb.league.sequential`) on the overall score stops at a planned look with A wins, B
wins or a tie, and a gate holds a winner that is clearly worse in one mode to a tie.
Batches go to the conditions where they most reduce the overall score's variance per
engine second; the weights stay fixed. Each process appends its batches and looks to its
own files in DIR, so the conditions may run as separate processes, and
`bnb evaluate decide DIR` takes a look at any time. The conditions, the score, the test's
settings and the pilot that set them are in
[the evaluation suite](../rewrite/evaluation-suite.md).

<!-- not-runnable: needs CUDA and two policies -->
```bash
uv run --no-sync bnb evaluate RUN_A/policies/step_N/exponential_average \
    RUN_B/policies/step_N/exponential_average --out evaluations/a-against-b
uv run --no-sync bnb evaluate decide evaluations/a-against-b --peek
```

## The shared rating pool

`bnb pool` keeps one pool per directory: every rated checkpoint of many runs and arms, and
every game played among them, rated per game preset against Imugi at 1000
(`bnb.league.rating_pool`). It is meant for a shared volume that more than one machine
writes at once.

- **Players** have stable names: a policy's is `<run>/step_<steps>/<weights kind>`, read
  from its directory (`<run>/policies/step_<n>/<kind>`, or a run's `league/` or
  `checkpoints/`), not the machine's absolute path; a scripted player's is its own name.
  Each is registered with its identity, the SHA-256 of its weights from its checkpoint.json
  after every file's hash is checked, and the paths where writers found it. A name
  registered again with other weights is refused, and games recorded under other weights
  are not counted.
- **Storage:** `pool.json`{.plain} (the conditions every game shares: the lineup, the arena scale
  and the default team-deathmatch scoring, written once), `players/<writer>.jsonl`{.plain} and
  `games/<preset>/<writer>.jsonl`{.plain}. A writer is one process (host, process id and start
  time) and only it appends to its files; a reader lists the directories and merges every
  file. No file is written by two processes, so no lock is needed: file locks are
  unreliable across machines on network and synchronised volumes, and appends from two
  machines to one file can interleave. A last line without its newline (a write in
  progress) is skipped; every batch carries an id, so a file copied in twice counts once.
- **Batches** keep each block's full outcome, as `bnb ratings` records do, with the
  players' identities, the game version and the seed; the scoring is applied when the
  pool is fitted.
- **Rating** (`bnb pool rate`): every stored game of the preset is fitted; a session then
  plays among at most `--session-players` players that can be played on this machine
  (those the continuous design over every playable pair weighs most, Imugi
  always among them), each batch planned on the posterior's marginal over them, so games
  elsewhere in the pool count. Before each batch the pool is read again, so another
  machine's games count as they arrive. The targets and limits are `bnb ratings`'
  (`--target-stderr`, default 45 for a ladder of checkpoints; `--decision`,
  `--contrast`, `--players-per-batch`, `--hopeless`).
- **The table** (`bnb pool table`) prints each preset's leaderboard and how well one rating
  per player fits the games (`bnb.league.goodness_of_fit`): the deviance of the directly
  played pairs' observed scores against the predicted ones (a pair across a bound, which
  the fit matches exactly, left out), with its p-value and the
  pairs with the largest deviance residuals, and a likelihood-ratio test of Bradley–Terry
  against mElo-2, the multidimensional Elo with one cyclic component (Balduzzi et al.
  2018): a small p-value is evidence of a rock-paper-scissors structure among the
  played pairs. With kill-share games both tests are conservative (a kill share varies
  less than a win or a loss). `--triangle-share` reserves a share of each batch for the
  least-played pairs among the decision players, so their triangles are played.
- **Curves** (`bnb pool curves --out FILE.csv`): each run's Elo against steps, per preset
  and weights kind, with the standard error and the games.
- **Importing** (`bnb pool import POOL DIR ...`): every `<preset>.batches.jsonl` below the
  directories is brought in once (its batches' ids come from their content), its players
  registered under their pool names; a policy the pool already holds under another name,
  by its weights, keeps the pool's name. The records are read, not changed.

<!-- not-runnable: needs CUDA and runs -->
```bash
uv run --no-sync bnb pool create /shared/rating-pool --lineup Yongma Yongma Jujak Jujak Jujak
uv run --no-sync bnb pool import /shared/rating-pool ~/avesnova/live-elo
uv run --no-sync bnb pool add /shared/rating-pool runs/RUN/policies/step_75497472/exponential_average
uv run --no-sync bnb pool rate /shared/rating-pool --presets real_scale_frontline --runs-directory runs
uv run --no-sync bnb pool table /shared/rating-pool --presets real_scale_frontline
```

Refitted from the live-Elo records of seven runs (bc-v19, bc-v20, rl-v20, rl-dm, rl-kl,
rl-mix-kl and the win/loss re-ratings, 377 batches), the pool's kill-share and win/loss
curves agree with each run's own last table to a median of 1 to 10 Elo (0.03 to 0.3 of the
earlier standard errors). Without a prior, three bc-v19 frontline snapshots, which lost
every frontline game they played, are upper bounds (< -216, < -238 and < -20), and three
rl-kl checkpoints are lower bounds in the win/loss deathmatch (> 1741, > 2179, > 2421);
the kill-share curves are unchanged (under 1 Elo), the other frontline and elimination
curves move by a median of 2 to 5 Elo, and the win/loss deathmatch curves rise by a median
of 72 Elo (0.76 of a standard error), the prior's pull on a few hundred games each
(docs/rewrite/sections/rating-allocator/progress.md).
