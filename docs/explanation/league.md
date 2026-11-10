---
covers: [src/bnb/league/, src/bnb/runtime/opponents.py]
---

# The league

<!-- generated: section-status -->

The league chooses the learner's opponents during training, rates the learner as it trains,
and rates finished policies afterwards. The training side is `bnb.league.league.League`,
attached to a trainer as its hooks; inside the rollout it acts through one controller, the
league's seat (`bnb.runtime.opponents.OpponentSlots`). The offline side is
`bnb.league.tournament`, behind `bnb ratings` and `bnb head-to-head`, and
`bnb.league.crossover`, behind `bnb crossover`.

## The pool

The default lineup (`bnb.runtime.train.LINEUP`) gives each game of the training mix two
pools: self-play and the league's seat, half each. Self-play is not a pool entry; it keeps
its fixed share. The entries (`bnb.league.pool.Entry`) are:

- **snapshots** of the live policy, taken every `LeagueConfig.snapshot_every` environment
  steps. Without a set value the league measures it once, about 30 minutes at the pace of the slowest of
  the second to fourth updates, and records it in `run.json`{.plain}. A snapshot is the weights that played the
  update it was taken at;
- **the averages:** the uniform average and the exponential moving average of the live
  policy's weights, folded in after every update (`bnb.league.pool.Averages`, through `average_into`, the
  decay `LeagueConfig.exponential_decay` per update; behaviour cloning keeps the same
  two with the same settings). The uniform average starts after
  the run's learning-rate warm-up (`LearnerConfig.warmup_steps`, 10M environment steps by
  default, or `LeagueConfig.uniform_from` when set: `TrainConfig.uniform_start`), so a
  warm start's optimizer, critic and advantages settle first; until then it is the live weights;
- **the stationary ladder**, only with `LeagueConfig.ladder_in_pool` (default false: the
  RL training pool is the snapshots and the averages; the rating lanes play the scripted
  agent and their own ladder either way): random, the rungs and the scripted agent
  (`LeagueConfig.ladder`, `bnb.runtime.opponents.Rung`s of the scripted agent's
  temperature and share, weakest first: `+inf` is `random`; a share p below 1 at T is
  `rung_T<T>_p<p>`, each ship the agent at T with probability p per game, else random (the
  mixture rungs, the ladder's bottom); T > 1 at share 1 is `rung_T<T>`; T = 1 is
  `scripted`; see
  [The scripted agent](scripted-agent.md#softness-logistics-the-floor-and-the-temperature)).
  The scripted agent gained a memory and per-mode parameters on 2026-11-05, so ladder
  ratings from before then do not carry over.

**Retention** (`bnb.league.pool.retained`) keeps the newest 8 snapshots and, of the older
ones, the one nearest each age of 2^k snapshots for k from 3. The run's `league/` directory
holds each retained snapshot as `step_<steps>/checkpoint`, with the two averages at the same
step beside it (`uniform_average`, `exponential_average`), all policy files.

## Sampling

An entry's sampling weight is `p (1 − p)` (`bnb.league.pool.sampling_weights`), where `p`
is the predicted probability that the live policy beats it under that mode's live gauge,
plus a floor (`LeagueConfig.floor`) spread evenly over the entries. Each mode samples by
its own gauge.

## Slots, the ladder and the cut

The league's seat plays every opponent row of the batch in the captured rollout graph:

- **Network slots** (`LeagueConfig.slots`, 4): their parameters are stacked, one set per
  slot. Slot k plays rows k, k + slots, …, so every slot plays every mode. Each set runs on
  its own rows inside the one compiled graph, and each row keeps its network's carry.
- **The ladder**, when it is in the pool: one scripted pass over every row, each row at its
  rung's temperature and share (written in place; `+inf` plays uniformly), with the
  scripted agent's carry for every row in the seat's carry. A slot can hold the ladder;
  each of its rows then draws a rung at every reset from its mode's weights over the
  ladder. Out of the pool (the default) the seat runs no scripted pass.
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
  forecasts), the previous actions and the position offset. A ladder row's scripted carry
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
- **Draws in the graph** use counter-based uniforms (`bnb.runtime.opponents.counter_uniform`,
  a hash of the stream's key, the counter and the row), so they need no generator state and
  repeat exactly; the host draws a slot's entry the same way.
- **The results:** the rollout tallies every ended game but each row's first by what the
  opponent row played and by mode, from the learner's side; the league reads the tally
  once per segment.

## The ratings

Every rating during training comes from one fit per mode, `bnb.league.lanes.RunRatings`:
one Bradley–Terry model of every rated player with the scripted agent pinned at 1000
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
  `exponential_average`, `uniform_average`, the ladder's rungs) and `live`. A lane plays
  the entry a league slot holds as `league_<k>`, which `RunRatings.resolved` maps to that
  entry, the one mapping of lane players and entries onto the fit.
- **The record:** games are kept by mode and by who played them. A game of two stationary
  players (the rungs, the league's snapshots) is kept for the whole run; one with a moving
  player (the live policy, an average) for its last `LaneConfig.window` updates (10).
- **Attribution:** the rollout tallies each ended training game under the column its row
  played as it ended (a slot's set or a rung), and the league reads the tally once per
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
  out, and a new one joins the fit as soon as its games connect it; a slot holding a
  ladder entry plays unrated. A lane game in progress when a slot's lane player changes is
  left out, as is each lane row's first game (cut short by the staggered clocks).
- **Who is rated:** a player whose games connect it to the scripted agent; none is reported
  before its matchups have played.
- **The ladder:** the rungs, chained from the scripted agent down by the lanes and played
  by the live policy in training, are rated from their games. A fixed scale would be far
  off: the earlier probability rungs' `1000 p` put rung_0.95 50 points below the scripted
  agent, where it scored 0.065 against it, about 530 points below.
- **The series:** per mode `rating/live/<mode>`{.plain} and `rating/average/<mode>`{.plain} (the
  exponential average), their standard errors, and every rated player's
  `lanes/rating/<mode>/<player>`{.plain}. A BC run's lanes rate the live clone and both averages the same way
  (`bnb.league.lanes.LaneHooks`), the uniform average as
  `rating/uniform_average/<mode>`{.plain}, and each of its policy files keeps its
  player's per mode (`PolicyRecord.ratings`): a warm start's live rating starts from its
  parent's, not 0.

The league's games are about half of the information. Refitted offline on run
2026-11-06-mixed-calm-fern's first 29 updates (its per-update lane games and `pool/entries`
table), the lanes alone left the live rating's standard error at 95–112 and the
exponential average's at 82–106; with the training games they are 59–66 and 63–71, and the
live policy is rated from the first update instead of the twelfth. The synthetic case is
`tests/sections/test_lanes.py::test_the_leagues_games_shrink_the_ratings_errors`.

Against the scripted agent alone a policy 1000 points above it would be rated blind: its
games there carry 1/60 of an even game's information. Through a chain of snapshots, whose
stationary links gather games over the run, the same window of live games rates it several
times more precisely
(`tests/sections/test_lanes.py::test_a_strong_player_is_rated_far_better_through_a_snapshot_chain`).
The measured game counts and errors are in `docs/rewrite/sections/rollout-lanes-notes.md`.

**Without lanes and without the ladder in the pool** nothing connects the league's entries
to the scripted agent, so the fit rates no one and the gauge keeps its starting ratings.
A run that wants league sampling by rating keeps its rating lanes.

## The live gauge

During training each mode has a gauge (`bnb.league.gauge.Gauge`) that steers sampling. It
reads every rating from the fit once per update; an entry the fit does not rate this update
holds its last rating:

- a rung its prior `1000 / T` (random 0), a placeholder, until the fit first rates it;
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
carry (every row's carry, flags and rung) and the league's state: its entries, gauges,
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
(`bnb.league.tournament.rate`). `--no-ladder` leaves out the rungs below the scripted agent,
which always plays: it anchors the scale. `--no-league` leaves out the run's `league/`
policies, so the field is `--with`'s players and the ladder. `--scripted-plays A B ...`
keeps the scripted agent to the named players; the others reach the anchor through their
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

- Before each batch every block is given a pair (`bnb.league.ratings.allocate_blocks`),
  greedily: the allowed pair whose block of games most lowers the summed rating variance
  at the current ratings, given every game so far and the blocks already chosen, among
  the pairs that still fit the engine's lanes. The information includes the fit's prior
  games, so the first batch is chosen the same way.
- After each batch the ratings are the Bradley–Terry fit on every game so far
  (`bnb.league.ratings.fit_ratings`, draws as half a win, one prior game), shifted so the
  scripted agent is at 1000. Only the scripted agent is pinned; random is fitted.
- The tournament stops when every standard error is under `--target-stderr`, or when the
  next batch would pass `--budget` games.
- The leaderboard rows (`calibrated_elo`) are keyed by game version, mode, map and fleet
  size, with the standard error and the games behind each. They are written to the run's
  `artifacts/ratings/` with the inputs' hashes.
- **The record:** after each batch one JSON line is appended, and synced to disk, to
  `<preset>.batches.jsonl` in `--out` (else the run's `artifacts/ratings/`): the batch's
  index, its blocks' matchups and results, the games so far, every player's rating and
  standard error, the worst error, the batch's seconds and the wall time, the games reused
  (`reused_games`) and each player's identity (`identities`, name to identity). A
  tournament that finds its preset's record rebuilds the games from it and goes on (the
  record's field must be its own: same names, same order, same identities); a finished one
  plays nothing.
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
  skipped names are logged. A **legacy record** (written before identities were recorded)
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
  ship-ticks and their summed speed, per side). A player may fly one ship type with every
  ship of its team (`Player.ship_type`; `bnb.league.matches.ship_type_players` gives the
  scripted agent flying each type of a set); the engine then sets each environment's ship
  types per team (`bnb.game.api.reset`'s `ship_type`). The [ship-type
  tournament](game/real-scale-tournament.md) is such a field.

<!-- not-runnable: needs CUDA and a run -->
```bash
uv run --no-sync bnb ratings 2026-11-02-mixed-purple-pumpkin --target-stderr 25
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

**Crossover fights.** `bnb crossover A [B] --pairs a:b ...` (`bnb.league.crossover`) plays
A (B is `scripted` by default) at unequal ship counts, to find the handicap at which A's
score crosses one half: each pair `a:b` gives A `a` active ships and B `b`, the rest of
each team's slots benched ([benched ships](game/modes.md#benched-ships)), in the game of
`--preset` built at `--ships-per-team` ships per team (default the preset's own; it must
hold the largest count). The engine is built with benching and gives every block its
pair's counts on the right sides whatever the seat, so every pair plays in the same
batches, `--games` games each, split evenly between the seats, from `--seed`. Per pair it
prints and records A's wins, draws and losses and its score (a draw half a win) with a 95%
Wilson interval, and where the score crosses one half between neighbouring pairs, in the
order given, by linear interpolation; `pairs.jsonl`{.plain} and `result.json`{.plain} (with the command and
the players' SHA-256) go to `artifacts/crossover/<time>/`{.plain}. Refining is explicit: play a
finer grid, or more games, around the crossing.

<!-- not-runnable: needs CUDA and a policy -->
```bash
# A with 5 ships against the scripted agent with 5 to 8, in an 8v8 deathmatch game
uv run --no-sync bnb crossover POLICY --preset deathmatch --ships-per-team 8 --pairs 5:5 5:6 5:7 5:8
# A with 5 down to 1 ship against 5, in the 5v5 training deathmatch
uv run --no-sync bnb crossover POLICY --preset deathmatch --pairs 5:5 4:5 3:5 2:5 1:5
```

**Ablated players.** A Yemong policy may play ablated, in `--with` or in head-to-head:
`PATH:rotary`, `PATH:forecast` or `PATH:rotary+forecast` (a run's name works as the
policy too). An ablated player is the same weights and build, so it shares its policy's
group and stacked weights; the seat holds each row's ablations and passes them to every
Yemong group's network (`bnb.models.yemong.model.Ablations`), so only that player's rows
are ablated and the captured graph stays one graph. A tournament without an ablated player calls the networks
without them. `rotary` makes the attention's rotary rotation the identity (every angle 0,
token and map tables); `forecast` stores nothing back to the belief, as
`yemong.forecast_feedback=false` does. Both are inference-time ablations of a model
trained with the feature on: they measure how much the trained policy relies on it, not
what a model trained without it would reach.

<!-- not-runnable: needs CUDA and a run -->
```bash
uv run --no-sync bnb ratings RUN --with POLICY POLICY:rotary POLICY:forecast
```
