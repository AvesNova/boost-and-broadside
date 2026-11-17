# Unified evaluator: design as built

Branch `feat/unified-eval` (from `feat/imugi-boundary`, with `feat/hero-clip` and
`fix/average-sweep` merged; `feat/v26-imugi-y046` merged 2026-11-17, so it plays game v26, dry run 3's). Code: `src/bnb/league/unified/`; command: `bnb unified`;
tests: `tests/sections/test_unified_eval.py`; box check: `box/unified_check.sh`.

## Why

The dry run's evaluation queue (`box/eval_queue.sh`) took about 4 h on the 5090 against
2.7 h of training: every team size or condition was its own compiled engine (5 min cold
each), batches were small and launch-bound, and every batch waited for its longest game
(30 s eliminations beside 600 s Frontline timeouts).

## The engine

- **Size buckets** (`cells.BUCKETS` = 5, 10, 20 ships a side): one compiled rollout per
  bucket. A game smaller than its bucket is benched (`api`'s `active` counts, as training's
  team-size sampling and `bnb crossover` bench). A cell goes to the smallest bucket that
  holds its larger team.
- **Modes are static slices** of a bucket's batch (one pool per preset). A mode's rules are
  the graph's Python constants (`bnb.game.rules.PoolRules`), so an environment keeps its
  mode; the split follows the planner's demand at the start of each bucket's run (value
  times expected game seconds per preset). Every other part of a game is data:
- **Pending buffers** (`engine.BucketRollout`): each environment's next game's seat, active
  counts, ship lineup and arena-scale uniform; the seat's next players
  (`seat.RefillSeat.pending`). The rollout reads them where it would otherwise draw
  (`_seats`, `_team_sizes`, `_lineups`, `_episode_uniforms`), so when a game ends the
  environment starts the planned next game inside the graph, at once.
- **Players as data** (`seat.RefillSeat`): each row holds its player index, switched at
  a new game's first tick. Scripted players run one Imugi pass over every row at each
  row's temperature/share/floor (read from per-player tables). Network players have a
  lane each in their build's group, weights stacked once (`bnb.runtime.stacked`); every
  tick the rows of each lane's player are compacted into a fixed-capacity table by a sort
  (static shapes), the group runs one vmapped forward, and results are written back
  (padding writes a spare row). The host keeps each player's rows within its lane's
  capacity (current and next game counted); a device counter reports any overflow and
  the run stops on one.
- **Resident set** (a field of more than `engine.RESIDENTS` = 12 network players a build):
  only 12 are seated, in one run of even lanes (1.15 x the rows over 12) plus one spare
  lane (`engine.SPARE_LANES`); the planner's `fits` admits only residents. Every 4 blocks
  the engine values each player by the planner's values of its cells (those against an
  outsider at half). While the run has an empty lane, the best outsider is seated in it
  at once when it is worth 1.25 times the least valuable staying resident (values run
  flat over a curve's field: at 2x no swap came in 30 minutes), and that resident is
  marked leaving (`RefillSeat.swap`: weights copied in place, tables updated; nothing
  recompiles). A leaving resident takes only short games (expected at most
  `engine.SHORT_SECONDS` = 120 game seconds: elimination's, in that slice) while one of its
  longer games is still in flight; once its games are done its lane is emptied and is the
  next spare. So no environment waits on a swap. An environment's next game keeps its
  current players when worth half the best (`Planner.choose`'s prefer), since its current
  and next games both hold lane rows.
  Why: each run's lanes are padded to capacity every tick, and 52 players cut into runs
  sized by share padded 4,096 rows to 8,996.
- **Map size**: a slice with scaled cells is built with map-size sampling, and each game's
  uniform is the inverse of the draw at the cell's scale (`cells.arena_uniform`).
- **Fog** follows the preset (the pool's fog share), as the queue played it.

## The host loop (`engine.BucketEngine.step`)

Replay block k; queue a copy of the first-role rows' game logs (finished count, result,
both teams' deaths, ticks) to pinned memory behind it; wait for block k − 1's copy (long
done); tally the games that ended; ask the planner for each of those environments' game
after next; queue the pending writes (pinned → device, non-blocking) behind block k. The
GPU always has the next block queued while the host plans. A game lasts thousands of
ticks, so the one-block lag never meets an environment twice (counted as `missed` if it
did). Buckets run in turn, or share the device when each gets at least 512 environments
(`run.MINIMUM_CONCURRENT`, `--concurrent yes|no|auto`).

## Sizing

`engine.footprint`: after a throwaway build, two eager probes (64 and 256 environments, a
few ticks each, garbage collected between) give the fixed and per-environment bytes;
`size_environments` takes 70% of the free memory (`--memory-share`) at the eager
footprint times `COMPILED_FACTOR` (1.5, measured on the laptop) and rounds down to a
multiple of 64. After the warm-up the run measures the compiled engine's reserved memory
and remembers the size it allows for this machine (`bnb.cache.ROOT/unified-sizes.json`,
keyed by GPU, memory, bucket, slices and shares), and later runs use it, so a machine
plays the same shapes run after run (one compile per bucket per machine, which the box's
`cache.sh` archive carries to a box of the same GPU model, the sizes file with it).
`--environments` overrides both.

## The planner (`planner.Planner`)

When an environment needs its next game, the planner takes every target's proposals
(`Target.proposals`: cell → the predicted drop of each unmet goal's variance over the
goal's variance, games in flight counted at their planned information), sums them per
cell, and gives the environment the cell of its bucket and mode slice with the most value
per expected game second whose players fit their lanes. It counts the game in flight with
every target, so one round's refills spread over the cells. Each cell's games alternate
seats. Expected seconds are each condition's running mean (a prior of half the clock).

Targets (`targets.py`):

| target | estimate | value of a game | stops |
|---|---|---|---|
| `EloTarget` | Bradley-Terry fit (`design.fit_posterior`) per condition | A-optimal drop on the planning precision (p (1 − p) at one SE past the planning difference) | each player and contrast at its SE (a bound or an undetermined player never) |
| `BreakEvenTarget` | crossover logistic fit (`crossover.fit`) | crossover's own allocation (`crossover.allocate`) picks the lineup; c-optimal drop of var(r*) | crossover's `stop_reason` (target, bound, budget) |
| `WinRateTarget` | the cell's mean score | drop of 0.25 / n | observed SE at the goal (16 games at least) |
| `DecisionTarget` | score − ½, group-sequential | drop of 0.25 / n toward its value at the planned maximum: **counts only** | a look's decision |

## Elo: no bound is a result (2026-11-17)

Dry run 1's curves (`bnb pool rate`, 12,288 games a mode) ended with late RL snapshots as
one-sided bounds after one or two sweeps of far weaker opponents: a bound was accepted
once its player had the games an even pair would need, and the unified `EloTarget` had the
same rule. Now (the author's decisions):

- **A bound or an undetermined player is never a result.** The planning covariance counts
  a finished game's information only between players of one fitted component (a game
  across components went one way and says nothing about the gap), so such a player keeps
  the prior's variance (`PLANNING_STDERR`, 1000 Elo): the least precise of the field, it
  takes games first.
- **Planning ratings for every player** (`targets.planning_ratings`): the Bradley-Terry
  likelihood of every game with a prior of `NEIGHBOUR_STEP` (400 Elo) on each neighbour
  pair's difference and 1000 Elo on each rating, so a bound's games go to the opponents
  just past it, not to the anchor. Neighbours (`run.field_neighbours`) are a cross: a
  run's consecutive checkpoints; other runs' checkpoints at the nearest steps; a chain's
  first RL checkpoint and its last BC one. The fit's estimates take over as games come in.
- **Stop on precision.** A `budget` and `--hours` are safety bounds; a target stopped
  short is named in a `WARNING` line (`run.short_of_goal`) and its report's `stopped`.
- **Prior games** (`run.load_prior`, `--prior POOL`): a rating pool's games of each Elo
  target's condition (`prior_keys`), this game version, players matched by weights.

The A-optimal value still picks the pairs; over a long chain it prefers hops of
p ≈ 0.8–0.95 (the information about a distance D over k hops is best near 400 Elo a
hop, where x² p (1 − p) peaks), so "near 0.5" is not the optimum for a 5,000-Elo span.

## Elo curves: the static design (the author, 2026-11-17 afternoon)

The adaptive planner is replaced for learning curves by a design fixed in advance
(`static_design.py`, `static_target.StaticEloTarget`; spec `design = "static"`, the
default `box/unified_spec.py` writes for curves; `--curve-design adaptive` restores the
planner). Win-rate, team-size and break-even targets keep their planning.

- **Players:** the anchor (this tree's Imugi, 1000 in the fit) and each run's snapshots.
  A champion with a snapshot's weights is that snapshot, never a second player. No old
  Imugi in the field, and no other scripted player.
- **Design prior** (counts only, never the fit): adjacent checkpoints 100 Elo apart along
  BC then RL; Imugi at the first RL checkpoint, 100 above BC's last.
- **Pairs:** each checkpoint against its +-1 .. +-`reach` neighbours on its run (default 2,
  BC continuing into RL). The anchor plays the checkpoints within 3 places of the first RL
  one, plus every second one out to 8 places (the links).
- **Counts:** A-optimal under the prior. A greedy pass adds games by the variance drop over
  the unmet goals per game (rank-one updates). A trim pass then takes back every game the
  goals do not need. Existing games count at their information and reduce the counts. The
  information matrix must be full rank (every checkpoint connected to the anchor), checked.
- **Two stages:** stage 1 plays the prior design (`pilot` > 1 designs it to pilot x the
  goal). Once its last result is in: the fit, then the A-optimal top-up at the fitted
  ratings, to 0.95 x the goal (`static_target.MARGIN`: the fit moves with the new games).
  A player outside the anchor's fitted component counts no information toward it. Then
  the target stops, and every player still above the goal gets a `WARNING` line
  (`run.short_of_goal`). A rerun designs again from every game in hand.
- **Engine:** every cell short of its count is proposed (no probe caps, no overtime, no
  filler). A resident set streams along the fixed training order (`BucketEngine._stream`,
  no values). A resident without proposed cells drains and leaves, and the next player in
  order with cells takes the free lane. If nothing is playable among the seated players
  and no lane can free up, the earliest resident leaves. `static_design.streaming_schedule`
  prints the idealised window steps.
- **Each run on its own first.** Several runs in one field are each resolved against the
  anchor (no cross-run pairs). The linking stage comes after the single-run stages, as its
  own stage or command: each checkpoint of run A against run B's checkpoint nearest in
  fitted rating, counts A-optimal for the run-to-run offset, then a joint fit. Not built.

Simulated on dry run 1's curves (results.md, "Static design"): elimination meets +-60 in
~15,600 games with the spec's pairs (the adaptive planner took 4,940). reach 4 with pilot 2
takes ~7,000. Fields with large fitted gaps (deathmatch, frontline) need pairs chosen by
fitted rating, not by place (open).

## Two runs in one field (the author, 2026-11-17)

Dry run 1 (game v24) and dry run 3 (v26) share the ship card and the game rules; only
Imugi differs. So one field rates both runs' snapshots and champions with both Imugis as
players: the anchor "scripted" (this tree's Imugi, 1000) and "imugi_v24" (`[players]`
value `"imugi_v24"`, `run.IMUGI_VARIANTS`: the turn commitment and the stick by energy
off). Each scripted player's rules are per row (`Imugi(rules=...)`, `imugi_combat.Context`),
so both play in one Imugi pass; the process flags stay the default. v24 games are admitted
into a v26 tree by an explicit table (`run.ADMITTED`: v24's "scripted" stands for
"imugi_v24"; every other version mismatch is still left out, and the log counts both).
`--prior` also takes a unified run's directory (games.jsonl, players by its spec.toml, by
identity). `box/unified_spec.py` builds the spec for several chains (names prefixed with
the chain); matched steps across two chains are cross neighbours stage with stage.

The planner over a 108-player field took 68 ms a choice; now about 8 ms (pair products
gathered from the covariance's rows instead of multiplied, the inverse on one thread, the
value sums kept incrementally).

## Map sizes and the fan (the author, 2026-11-17)

Evaluation plays only the map sizes the run trained on (elimination's shrinking map
included); the map-scale Elo targets are gone from `box/unified_check.sh`. The break-even
fan holds the policy at the trained 5 ships and varies Imugi's count, 1 to 20
(`crossover.CrossoverConfig.policy_count`; spec `policy_count = 5`); 20 a side stays in
the team-size lineups.

## Decisions at information fractions (`targets.InformationLooks`)

The design (`bnb.league.sequential`: K = 4 looks, O'Brien-Fleming spending, α = 0.05,
β = 0.2, δ) gives I_max; information is **counted**, n / 0.25, so n_max = I_max / 4. Look k
comes at the first tally after n reaches k/K of n_max; its boundary is spent at the
fraction actually reached, given the earlier looks' actual fractions (Lan-DeMets), and a
look past the last planned fraction is final. A burst that passes two planned fractions
takes one look there. The decision's value to the planner reads only n, never the
observed difference, so the allocation cannot bias the test; games that finish after the
decision are not the test's.

## Records

`games.jsonl`: one line per flush (every 10 s), every game finished since (cell, seat,
bucket, overtime flag, score, kills, seconds), written whole and synced. A rerun with the
same `--out` replays it into the targets and continues. `report.json` holds every
target's estimate, rewritten with each progress line; `progress.log` a line every 30 s
with games, games per hour, GPU utilisation, each bucket's environments and the ETA as a
clock time (from the rise of the mean information fraction, goal² / SE² per target).

## Not done (open)

- Rating-pool export: Elo games are in `games.jsonl`, not yet appended to a
  `bnb.league.rating_pool` pool (so `bnb pool table` does not read them).
- A mode's slice is fixed per bucket run: when its goals are all met first, its
  environments play **overtime** (the planner's values with every goal taken as unmet,
  budgets still binding; the games are tallied and flagged), then **filler** (an Elo
  target's pairs with the probe caps lifted, `Target.filler`; also flagged overtime), and
  park (Imugi against itself at 1v1, untallied) only when none of these fits, until the
  bucket's other slices finish. Rebuilding the bucket
  with a new split costs a compile.
- Kill-share scoring for deathmatch Elo is an option (`kill_share = true`), not the
  default (the pools' default is win/loss).
- Decisions are direct head-to-heads in one condition; a decision on a Bradley-Terry
  contrast through common opponents is not built.
