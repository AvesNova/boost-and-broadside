# lean/learn and lean/eval: notes

## Implemented

- `config.py`: `LearnConfig`, `LEVEL_NAMES`, `bc_weight`, `learning_rate`.
- `rewards.py`: v2's five levels: `reward_step(config, ledger, events, team) -> ((E, N, LEVELS),
  ledger)`, `tick_events` (lean's StepOut to v2's events), `Ledger`. (Since 2026-11-23; see
  "Reward levels and the hit head" at the end. The four-level `rewards` with shaping is gone.)
- `collect.py`: `start`, `collect`, `abandon`, `Opponent`, `CollectCarry`, `Rollout`, team-size draws.
- `returns.py`: `value_centres` (the learner's support, built with lean.model.value),
  `generalised_advantages`, `normalise_advantages`. HL-Gauss targets and decoding are
  lean.model.value's `hl_gauss_targets` and `decode_value`.
- `update.py`: `make_optimiser`, `update`.
- `pool.py`: `League` (the scripted agent and snapshots, PFSP draws, decaying tallies).
- `lean/eval/matches.py`: `play`, `head_to_head`, `crossover`, `wilson_interval`, `games_for`.
- `lean/train.py`: command-line entry with `--smoke`.

## What was run

- `pytest lean/learn/tests lean/eval/tests` on the CPU: 10 passed. Nine use stand-ins
  (`lean/learn/tests/standins.py`); `test_real_packages_smoke` ran against the real
  lean.game, lean.scripted and lean.model (4 environments, 2v2, 8 ticks, one update).
- `python -m lean.train --smoke`: see the hand-back report for whether it completed.
- Not run: anything on a GPU, `torch.compile`, a snapshot opponent with the real model,
  unequal team sizes with the real game, shaping values against the real state, any
  learning run long enough to move the win rate, crossover with the real game beyond
  the smoke setting.

## Followed from v1 (run 731)

(The reward weights, discounts, single lambda, shaping and team-mean mixing below are
superseded by v2's levels, 2026-11-23: see "Reward levels and the hit head" at the end.)

- Discounts per level 0.998, 0.990, 0.982, 0.951: v1's per-decision values. v1 decided
  at 30 Hz and this game ticks and decides at 30 Hz, so the conversion is the identity.
- Reward weights: win 1, death -0.283, kill +0.283, friendly kill -0.283, damage taken
  -0.274, dealt +0.548, friendly damage -0.548, facing 0.09, closing 0.08; shaping at
  full scale to 100M steps then exponential decay to 0.05 at 400M.
- BC inside PPO: cross-entropy to the scripted action, coefficient
  2.0 * max(0, 1 - win_rate / 0.45).
- 4v4 by default, 4 epochs, 128-tick segments, gradient clip 1.0, Adam eps 1e-5.
- (Superseded: "half of the environments always face the scripted agent" was wrong by the
  author's account of v1; see "The league" below.)
- Crossover: network team 0, ties as non-wins, walk from the previous crossover,
  `beats_up_to` / `crossover` / ratio, 320-ship cap, `games_for` is v1's games formula.

## Departures from v1, and why

- One lambda 0.95 (v1: one per component), clip 0.2 (0.15), learning rate 3e-4 constant
  after a 10-update linear warm-up (4.5e-4 with decay), entropy 0.003 (0.005), value
  coefficient 0.05 (1.0), next-state 0.1 (0.2 plus drift loss): as the task set them.
- Categorical critic with HL-Gauss targets instead of a Huber loss on normalised returns.
- Advantages: summed over levels with weight 1 and divided by the standard deviation
  over the update (v1: per-component running RMS, then weights, then buffer RMS). The
  mean is not subtracted.
- Reward normalisation: v1 divided every reward by the total ship count. Here outcome is
  one value per ship; kill and damage are individual_share * own + (1 - share) * team
  mean; so targets do not shrink or grow with fleet size.
- Outcome goes to every ship that existed, dead included (v1: surviving team-mates).
  Dead rows have value zero and no loss; the reward reaches the ship's living ticks
  through GAE, which is cut only at the end of the episode.
- No kill assist, no credit for boundary ("field") deaths or damage.
- BC target is the scripted agent's sampled action, not its probabilities; the gate does
  not latch off after 3 updates; no KL early stop; no weight averaging.
- (Superseded 2026-11-23 by the author's league, below.) The league was a uniform pool of
  the last 8 snapshots with a fixed scripted half and one trained team per environment.
- Crossover's first size walks from parity instead of doubling and bisecting (same
  answer under the monotonic assumption, more matches when far from parity).

## Choices the plan left open

- `individual_share` 0.5; `damage_unit` 1.0 (damage assumed to be a fraction of full
  health: check against lean.game's damage units and set it); `speed_unit` 100 m/s.
- Value supports: half-widths 1.5, 2, 2, 4 per level; sigma 0.75 bin widths; bins read
  from the model's output. Targets beyond the support are clamped.
- Facing uses the nose (`angle_cos/sin`), closing the velocity (`heading_cos/sin * speed`).
- Scripted win rate for the BC gate: since 2026-11-23 the league's tally against the
  scripted member (see below).
- Minibatches are random sets of views (one seat's whole segment); 32 per update, run as
  micro-batches with gradient accumulation (see "Rollout store, streaming and micro-batches").
- Advantages and value targets once per update from the values stored at collection
  (`targets_every_epoch=False`, the author's decision); True recomputes them from the current
  critic at the start of every epoch.
- `collect`, `update`, `crossover` keep the contract's positional signature and add
  keyword-only options (`shaping`, `game`, `scripted`, `scripted_win_rate`,
  `update_index`, `generator`, `match`). `opponent` is an `Opponent`, `carry` a `CollectCarry`.

## Assumptions about the sibling packages

- The carry's environment axis is 1 when the carry has 4 dimensions (lean.model:
  blocks, E, 1 + N, width) and 0 otherwise (`carry_axis`). The learner zeroes the carry
  itself on reset and does not use `Policy.step`'s optional `reset` or `position_offset`.
- `game.reset` sizes N from the largest total dealt, so `start` forces environment 0 to
  the maximum size; `game.deal` is called at that N.
- `scripted.act` is called once per tick for both teams (`team=None`), without its
  optional memory argument and with uniforms (E, N, 3) from `torch.rand`, not
  `scripted.draw_uniforms`; personas are drawn with the segment's deals (every persona
  field is assumed to have a leading E).
- Next-state loss: the target here is the change in `obs.ship`, used only if the head's
  width equals SHIP_FEATURES. lean.model's head predicts scaled changes of State fields
  (`model.next_state_target(config, state, next_state)`), so with the real model the
  next-state loss is currently ZERO. To fix: record that target in `collect`.
- The value helpers are local copies; lean.model has `hl_gauss_targets` and its own
  centres, which may differ from `value_centres` here. Unify before trusting values.
- Stepping continues after `done` only through `take_deal` (collect) or `reset_done` (eval).

## Known weaknesses

- `collect` drew the deal, personas, team sizes and learner teams for all environments on
  every tick: since 2026-11-23 once per segment (below, "Waste fixes 6, 7 and 9").
- `team_view` calls `observe` twice per tick (three or four times with a snapshot).
- `play` reads one flag back every 32 ticks.

## First on the GPU

1. `python -m lean.train --smoke` with the generator and model on the device (train.py
   builds a CPU generator; device selection is not wired yet).
2. Steps per second of `collect` alone and of `update` alone at 256 and 2,592 environments.
3. Peak memory of a 128-tick rollout: done (the rollout store, below).
4. `torch.compile` of `policy.step` and of the tick body; check that the varying
   `faces_scripted` and snapshot branch do not retrace.
5. The scripted win rate gauge against `head_to_head` on 1,000 games.

## Second pass: what changed, what ran, and the profile

Changed:

- Next-state loss is live. `collect` records `lean.model.next_state_target(policy.config,
  state, next_state)` around each game step (before the re-deal) as `Rollout.next_state`,
  with `next_valid = own & next_state.alive & ~done`. `update` regresses the head on it
  and raises ValueError when the head and the target differ in shape; there is no skip.
- Value helpers come from `lean.model.value`. The support is `LearnConfig.value_low` and
  `value_high` (outermost bin centres per level): outcome +-1.5, kill +-2, damage +-2,
  shaping +-4. With 77 bins three bins are 0.12, 0.16, 0.16 and 0.32. Bounds used:
  outcome returns lie in +-1; kill and death at 4v4 in about -0.28 to +1.13; shaping at
  most (0.09 + 0.08) / (1 - 0.951) = 3.47, so 3.79 with margin. The damage bound depends
  on the game's damage per tick against `damage_unit` and was not measured: +-2 is
  unverified. Larger teams need the kill level widened. The model's own `value_low` and
  `value_high` buffer is not used by the learner.
- `--device` in `lean/train.py` (cuda when available, else cpu) places the model and the
  generator; the state, carries and draws follow the generator's device. Not run on a
  GPU: there is none here. `--threads` sets torch's CPU threads.
- Position offset: drawn per environment in `start`, redrawn where an episode ends,
  passed to the learner's and the snapshot's `step`, stored as `Rollout.position_offset`
  (T, E, 2) and passed to `sequence` in `update`. `lean/eval` passes none.
- The stand-in policy has a `config` (position periods, next-state fields) and accepts
  `position_offset`. `lean.learn` now imports `lean.model` at import time.

Ran (CPU, OMP_NUM_THREADS=1): `python -m pytest lean/learn lean/eval -q`, 12 passed;
`python -m lean.train --smoke` completed (two updates, one with a snapshot opponent, then
the crossover). Not run: a GPU, a width mismatch with the real model, unequal teams with
the real game.

Profile of `python -m lean.train --smoke` with a wall-clock timer around each call, one
thread, 6.8 s in all (4 environments, 2v2, 8 ticks, 2 updates; crossover 2 games of 1v1):

| Part | Seconds | Share |
|---|---|---|
| crossover (576 match ticks of 2 games) | 5.11 | 75.5% |
| collect, both rollouts | 0.24 | 3.6% |
| update, both | 0.13 | 1.9% |
| imports and start-up | the rest | about 19% |

By function over the whole run (592 game ticks, nearly all in the crossover):

| Function | Seconds | Share | Per call |
|---|---|---|---|
| `lean.game.step` | 2.11 | 31.3% | 3.57 ms |
| `lean.model.Policy.step` | 1.40 | 20.7% | 2.32 ms |
| `lean.scripted.act` (two calls per tick) | 0.78 | 11.5% | 0.66 ms |
| `lean.game.reset_done` | 0.71 | 10.4% | 1.19 ms |
| `lean.game.observe` | 0.21 | 3.0% | 0.33 ms |
| `lean.model.Policy.sequence` | 0.05 | 0.7% | 6.2 ms |
| `lean.learn.rewards` | 0.008 | 0.1% | 0.50 ms |

Collect and update together ran at 150 to 180 steps per second with one thread and the
same with four. The earlier 3 to 5 steps per second did not reproduce; nothing in these
files was slow, and nothing was changed for speed. The smoke run's wall time is the
crossover: 1v1 games that run hundreds of ticks at about 8.9 ms a tick, which is fixed
per-call overhead at this batch size (game step, then model step). `game.reset_done`
costs a third of a game step on every tick, though almost no tick ends a game; skipping
it would need a read-back of `done` in the loop or a cheaper path inside lean.game.

Found on the way, not in these files: the energy column of the next-state target has a
mean square of about 1470 in the smoke rollout (the other columns 0.001 to 0.14, shield
0), because energy changes by hundreds of units in a tick against the model's scale of
10. The loss reads 209.8 and, at coefficient 0.1, outweighs every other term until the
scale in `ModelConfig.next_state_fields` is set from measured changes.

## The league: the author's design (2026-11-22)

Half the environments self-play (the current policy on both sides; both sides' data
trains). Half play league games: the current policy against an opponent drawn from the
league, which holds past snapshots and the scripted agent as one member among them; only
the live side trains (PPO is on-policy). This replaces the fixed scripted half and the
snapshot half above. The statement under "Followed from v1" that half of the environments
always face the scripted agent is wrong by the author's account of v1.

### Implemented (2026-11-23)

What the code does now:

- **Seats.** A seat is one team of one environment that the live policy plays. The first
  E_s = round(self_play_share * E) environments (0.5, fixed by index) are self-play and
  have two seats; the rest are league games and have one. The live policy runs once per
  tick on all S = E + E_s seats (one batched `step`). The rollout's second axis is the
  seat axis, which `update` treats as its environment axis, so both self-play teams'
  rows are in the loss (each seat's `own` is its team's living ships) and a league game's
  opponent rows are in no seat.
- **League** (`pool.py`). Member 0 is the scripted agent; members 1..8 are snapshot slots
  (one added every `snapshot_every` = 10 updates, oldest replaced). Per member, tallies
  of games, wins and draws on the device; each game decays its member's tallies by 0.99
  and then counts; p = (wins + draws / 2) / games, 0.5 unplayed. Probabilities: 7/8 by
  (1 - p)^2 normalised over the members present (uniform if the weights sum below 1e-10,
  AlphaStar's rule), 1/8 uniform over the members present. Only league games are
  tallied; self-play games are not.
- **Draws.** A league environment's member is drawn at `start` and redrawn in the same
  step that re-deals a finished environment (`redeal`), from the tallies as they stand
  after that tick's games. Nothing else changes it.
- **Opponent actions** (`league_actions`). The scripted agent's action for every ship is
  computed every tick anyway (it is the BC target for every live ship, both self-play
  teams included); scripted-member environments take it. Each snapshot member present
  runs its own frozen copy once per tick on just its environments: the league environments
  are grouped by member (`argsort`), which needs one read-back per tick (`bincount` of
  the members, at most 9 numbers), and the loop runs over the members present (at most 8
  model calls, each on its own subset, so the total rows equal the league environments).
  Before the first snapshot there is no read-back and no loop.
- **Replacement.** `League.maybe_add` returns the slot it wrote. `abandon` re-deals the
  league environments that were playing that slot (their weights changed mid-game), draws
  them new opponents and marks them reset; their unfinished games are not tallied. The
  previous rollout already bootstrapped those seats from the old state, as at any segment
  end. `lean/train.py` calls it after each addition.
- **Observation.** `observe` is called twice per tick (team 0 and team 1) and every seat
  and snapshot view is selected from those two (was two to four calls).
- `LearnConfig`: `scripted_share` and `win_rate_momentum` removed; `self_play_share`,
  `league_floor` (0.125), `league_decay` (0.99) added. `Opponent(scripted_config, league,
  self_play_share, persona_spread)`; `league=None` means every league game is against the
  scripted agent with no tally (tests and simple callers).

Choices the author should check:

1. **BC gate input.** The gate reads the league's tally against the scripted member, with
   draws as half (AlphaStar's win rate; the old gauge counted ties as non-wins) and **0
   until a scripted game has been played**, not the sampler's 0.5 prior (0.5 is above the
   0.45 gate and would switch BC off at the start). The tally's window is about 100 games
   against the scripted agent (decay 0.99), much shorter than the old per-rollout EMA.
2. **Simultaneous games.** n games of one member ending on the same tick are counted as
   every order averaged: games * 0.99^n + (1 - 0.99^n) / 0.01, each win weighted
   (1 - 0.99^n) / (0.01 n). Exact for n = 1, which is the usual case.
3. **Replaced member's games are abandoned** (re-dealt), not allowed to finish against
   the new weights. Cost: a few cut episodes every 10 updates once the pool is full.
4. **Fixed split by index.** Environments 0..E_s-1 are always self-play; which team is
   the "learner team" there is irrelevant. The draw is per environment, the split is not
   (keeps the seat count and every shape static).
5. **The floor and the weights range over the members present**: before the first
   snapshot every league game is against the scripted agent.
6. **Self-play games update no tally**, and the scripted member's p counts only league
   games against it.
7. Snapshot opponents sample their actions (Gumbel-max) as before; temperature 1.

Cost and memory:

- The update now trains 1.5 seats per environment: at E = 256 that is 384 seats, so a
  minibatch holds 1.5x the rows it did. The 8 GB figures (4 minibatches out of memory, 8
  at 4.8 GB, 16 at 2.5 GB) were measured before this change; 12 to 16 minibatches give
  the measured row counts. `minibatches` stays 4 (recipe unchanged); the default run will
  not fit 8 GB.
- Collection runs the live model on 1.5x rows and the snapshot models on the league half
  in up to 8 smaller calls; the read-back is one small `bincount` per tick. Not measured
  on the GPU beyond one run without snapshots (below).

Ran: `pytest lean -q` on the CPU (OMP_NUM_THREADS=1): 89 passed, 1 xfailed (the xfail
is lean/game's v2 parity test; 8 of the passes are new in `lean/learn/tests/test_league.py`;
`test_learn.py` updated to seats).
`python -m lean.train --smoke --device cpu` completed. GPU (8 GB laptop, performance
profile): `python -m lean.train --device cuda --envs 64 --updates 3` completed at about
1,500 environment ticks per second with peak 4.2 GB in use; no snapshot was in the league
in those three updates (first addition after update 10). The snapshot path with the real
model ran only on the CPU (`test_snapshot_members_run_with_the_real_packages`).

## Reward levels and the hit head (2026-11-23)

The author's decisions: lean follows v2's mathematics unless a deliberate change says
otherwise; every reward level is the same kind of target (HL-Gauss, linear bins, 64 per
level, a scale per level); no shaping level. Source: v2's release commit 4cfd1e86
(`src/bnb/learn/rewards.py`, `src/bnb/learn/ppo.py:172-173`).

### What changed

- **Levels** (`config.py`): `LEVEL_NAMES` = damage, kill_death, objective_progress,
  objective, outcome (v2's order; `DAMAGE` .. `OUTCOME` = 0 .. 4). Weights v2's: 2.74 per
  100 damage points, 2.83, 20, 20, 10 (`reward_weights`). Discounts 0.982, 0.990, 0.995,
  0.999, 0.9997 and GAE lambdas 0.81, 0.90, 0.95, 0.97, 0.97 per level (`gae_lambdas`;
  `generalised_advantages` takes a lambda per level). Gone: `win_reward`, `kill_reward`,
  `death_penalty`, the damage rewards, `individual_share`, facing and closing, `speed_unit`,
  the shaping schedule and `shaping_scale`, and `collect`'s `shaping` argument.
- **Rewards** (`rewards.py`): v2's components with v2's mathematics, summed per level
  (v2's offence and defence parts are not kept apart: their sum is the level). Damage:
  taken, dealt, friendly fire (the shooter blamed, the other team paid twice), recharge
  refunded to the target and charged back to each attacker pro rata through the ledger,
  boundary damage and its unattributed healing to the other team. Kill/death: death -1; a
  kill shared half by the final tick's raw hits and half by the ledger's outstanding damage
  (assists); an ally's share blamed and paid twice to the other team; a boundary death pays
  the other team. Outcome: +1 / -1 to every ship of the winning / losing team on the last
  tick, the dead included, 0 for a draw. The objective levels are zero columns. A slot out
  of the match (`exists` False) gets nothing. Damage and kill/death are zero-sum; the
  outcome is zero-sum between equal teams only (each ship +-1, as v2).
- **Ledger**: `CollectCarry.ledger` (E, N, N) and (E, N), carried across ticks and
  segments; a death clears the dead ship's column, a finished game the environment;
  `redeal` clears re-dealt environments (so `abandon` does too).
- **Hit labels** (`collect.py`): `Rollout.hit_label` (T, S, N) long. Each tick, every
  bullet that ended (`StepOut.shot_outcome` >= 0) writes its outcome to row
  `label_row - (state.tick - shot_tick)` (episode ticks and rows advance together within
  an episode). The labels hold 2T rows: the previous segment's T ticks, then this
  segment's; in a store, the first half is merged into the previous segment's labels
  (below, "Chained returns and hit labels across segments"). A bullet fired earlier goes
  to a spare entry; rows whose bullet has not ended by the update's last tick stay -1. One
  scatter per tick, nothing read back. With ghost bullets every living ship's every row is
  labelled except the update's last segment's tail.
- **Hit loss** (`update.py`): `hit_loss` is the cross-entropy in nats over the rows that
  are `own` and labelled; other rows add nothing to the loss or the gradient (a `where`,
  not a multiply, so a meaningless masked logit cannot leak a NaN). `hit_coefficient`
  0.05. Metrics `hit` and `hit_labelled` (labelled share of own rows).
- **Range monitoring** (`returns.support_usage`, called by `fresh_targets` every epoch):
  per level `range_clamped_<level>` (share of valued targets beyond the outermost
  centres), `range_bins_<level>` (bins holding at least 0.1% of the targets) and flags
  `range_warning_clamped_<level>` (clamped share > 1%) and `range_warning_narrow_<level>`
  (fewer than 8 of 64 bins occupied, unless the level paid no reward in the rollout, which
  keeps the objective levels quiet in elimination). Averaged over the epochs in `update`'s
  metrics; `lean/train.py` prints a `WARNING value support:` line naming each raised flag.
- `lean/train.py`: `--game v2` (default: `GameConfig(ghost_bullets=True)`, so the hit head
  has labels) or `--game core` (`GameConfig.core()`).

### Supports, from measured returns

Discounted returns per living ship-tick over whole episodes, scripted against scripted
(CPU, 64 environments each; v2 game with ghost bullets, and the core game):

| Game, size | damage min / q.001 / q.999 / max | kill_death min / q.001 / q.999 / max | outcome |
|---|---|---|---|
| v2 4v4 | -4.72 / -2.74 / 2.67 / 4.47 | -3.07 / -2.80 / 2.81 / 4.68 | +-10 |
| v2 5v5 | -3.85 / -2.93 / 2.89 / 5.84 | -3.42 / -2.80 / 3.10 / 5.49 | +-10 |
| v2 8v8 | -4.85 / -2.77 / 2.62 / 5.48 | -3.31 / -2.80 / 2.90 / 5.36 | +-10 |
| core 4v4 | -3.35 / -2.36 / 3.04 / 5.22 | -3.69 / -2.80 / 4.01 / 7.00 | +-10 |
| core 8v8 | -4.72 / -2.45 / 2.85 / 5.92 | -5.70 / -2.83 / 4.07 / 7.88 | +-10 |

The learner's own first rollouts (random weights, 16 environments 4v4, four 128-tick
segments, returns over the concatenated segments): damage -4.27 to 2.16, kill_death -4.79
to 2.76 (ally kills by a random policy), outcome -10 to 0.

Supports (outermost centres): damage +-7, kill_death +-8, outcome +-12, objective levels
+-1 (placeholders). Each holds the extremes above with about three bin widths of margin,
where HL-Gauss's expectation is still unbiased, except kill_death's 7.88 (one rare
multi-kill, half a bin inside the outermost centre, so slightly pulled in). Symmetric so that an untrained critic (a uniform softmax)
reads 0. Bin widths: 0.22, 0.25, 0.03, 0.03, 0.38. In the same runs the damage and
kill/death levels summed over all ships to at most 1e-6 per tick (zero-sum through the
real game).

### Ran

CPU, OMP_NUM_THREADS=1: `pytest lean/learn lean/model -q`, 56 passed; `pytest lean/eval
-q`, 3 passed; `python -m lean.train --smoke --device cpu` completed. In the smoke run
`hit_labelled` was 0: its 8-tick segments are shorter than a bullet's 11-tick life, so
only same-segment hits labelled a row and none occurred (since the labels chain across
segments, 2026-11-23, the smoke run's first segment is labelled from its second:
`hit_labelled` 0.375). The value-support flags were quiet
there (no level paid a reward in 8 ticks); with 128-tick segments early training is
expected to raise the narrow flag on damage and kill/death until ships fight.

New tests: `test_rewards.py` (zero-sum with the ledger and a benched slot; outcome; kill
shared with an assist; last hit takes the kill; ally kill; boundary death; recharge
charge-back; friendly fire; objective columns; zero-sum through the real game; range
monitoring on synthetic clamped and narrow targets) and `test_hit_head.py` (labels land
on the firing row, not the ending row; the loss reads only labelled own rows and gives
no gradient elsewhere; an update with no labels leaves the hit head's weights unchanged;
real-game labels with and without ghost bullets).

### Choices the author should check

1. **The events bridge (`tick_events`) is interim and partly reconstructed.** v2's rewards
   need the attacker-by-target raw damage, the boundary damage apart from the hits, and
   the shield's restoration. lean's StepOut has per-ship totals only. Exact: shield lost
   and restored (from the two states), deaths, done, result. Mirrored from lean.game: the
   boundary damage (its formula on the post-move position), so a change to the game's
   boundary (a moving boundary in another mode) must be mirrored here. Reconstructed: the
   raw matrix, by splitting each target's hit damage over the attackers in proportion to
   their enemy or friendly damage; exact unless an attacker hits two targets or a target
   is hit by two attackers on the same tick (column totals always exact). **Contract
   change wanted** (not made: another agent is editing the contract):
   `StepOut.hit_damage` (E, N, N) [attacker, target] raw damage of real bullets (the game
   already has per-bullet damage and target: one scatter_add), `StepOut.boundary_damage`
   (E, N), `StepOut.shield_restored` (E, N). Then `tick_events` reads fields and the
   mirror goes.
2. **Five critic levels, not v2's ten parts** (offence and defence apart): the decision
   says one target per level. The parts' sum is the level, so the rewards are v2's.
3. **No team mixing.** v2's `team_spirit` is 0 on every level; lean's own 0.5 team-mean
   mixing is dropped. Rewards are additive per ship, as the decision on team size says.
4. **The outcome is a level** (HL-Gauss over +-12), not v2's 4-class outcome head on
   lambda targets: the decision.
5. **Supports** as above; the objective levels' +-1 is a placeholder until a mode pays them.
6. **Range thresholds**: clamped above 1%; narrow below 8 bins occupied (an eighth of 64);
   a bin counts from 0.1% of the targets. Arbitrary; the author may want others.
7. **`hit_coefficient` 0.05**, equal to the value coefficient: both are cross-entropies in
   nats per row (the value one averaged over levels), and the head is an auxiliary signal
   of the same kind. v2's shot-outcome head had 0.019 against a value coefficient of 0.007
   summed over nine heads; v2's coefficients sit against a policy term of 27 and are not
   comparable to lean's directly.
8. **Hit classes** follow `StepOut.shot_outcome` (0 miss, 1 enemy, 2 team-mate). Class
   balance in scripted play: about 90% misses; enemy 7% and team-mate 3% at 4v4; at 8v8
   team-mate hits (63k) outnumber enemy hits (61k) (ghost bullets of ships aiming through
   a crowd). A random policy hits team-mates more than enemies.
9. **The training entry now plays ghost bullets** (`--game v2` default), so the hit head
   is trained by default; ghosts change no play.
10. v2 values ships alive or waiting to respawn; lean values `own` (living own ships).
    Unchanged here; it matters once respawn modes exist.

### Open for the author (not adopted)

- v2's separate BC stage (100M steps cloning Imugi's full distribution, then RL from the
  clone's EMA weights with no BC term and no entropy bonus); lean keeps the gated BC term.
- Weight averaging (v2's in-run EMA 0.95 per update and the uniform average, both league
  members and rated).
- The KL early stop (v2: 0.02 after each epoch but the last).
- (Decided 2026-11-23: the update size is v2's, 2,048 environments x 4 segments x 128 ticks,
  32 minibatches; see "Rollout store, streaming and micro-batches".)
- Also differing from v2 and untouched here: advantage scaling (v2 a slow EMA of the
  returns' 5-95% gap, floor 1, once per update; lean the advantage std every epoch), the
  policy coefficient (27) against value (0.007 summed), the 128-tick lookahead before the
  bootstrap, the entropy settings, Adam's epsilon.

## Rollout store, streaming and micro-batches (2026-11-23)

The update is at parity with v2 (the author's decision): `LearnConfig.environments` 2,048 x
`segments_per_update` 4 x `segment_ticks` 128 = 1,048,576 environment ticks, 32 minibatches,
4 epochs; with half self-play that is 12,288 views (one seat's 128-tick segment each), 384
per minibatch. Measurements: docs/rewrite/scoreboard.md, "Update at parity with v2".

- **Store** (`store.py`, `RolloutStore`). `lean/train.py` collects the segments of an update
  into one store (`collect(..., store=, segment=)`), then `update` reads it. Records are
  (T, V + 1, ...), view v = segment x S + seat; the extra view is an empty spare that pads the
  last micro-batch of a minibatch so every micro-batch has one shape (nothing owned, no
  recompiles). Large records (the observation's per-tick fields, actions, old
  log-probabilities, stored values, rewards, next-state targets) and the initial carries
  live in pinned host memory; masks and labels (`own`, `done`, `reset`, `next_valid`,
  `hit_label`, the deal row) stay on the device, as do the bootstrap values, advantages and
  value targets. `stream_rollout=False` (or the CPU) keeps everything on the device.
- **Fields once per episode.** The field observations (`field`, `field_pos`, `field_mask`)
  and the position offset are constant within an episode: `collect` builds a deal table per
  segment (rows: the environments as the segment starts, then the segment's deals; two
  `observe` calls per segment) and each tick records only each seat's row. The core game
  (10 fields) saves 228 bytes per environment tick; the v2 game has no fields.
- **Compact types** (author's decision): `pos`, `heading`, `field_pos`, the position offset,
  old log-probabilities, values and rewards float32 (advantages and targets are computed in
  float32 on the device); `ship`, `globals`, `since_seen` (symlog seconds), `field` and the
  next-state targets float16; actions, teacher actions and hit labels int8; masks bool; the
  deal row int32. Cast back (float32, long) on the device after the copy and before the
  model, so the position code and the rotary angles see float32 positions. Ranges in
  scripted rollouts of both games: every float16 field within 4.6 in magnitude (ship 1.34,
  globals 1, field 4, since_seen 2.6, next-state 4.5); none near float16's limit. `heading`
  is kept float32 because it is a rotary angle (position-like). The ship row's x/R and y/R
  columns are float16: the first linear layer reads them in bf16 anyway (coarser). The
  initial carries stay float32 (432 bytes per environment tick, a fifth of the store).
- **Streaming out.** Each tick the copy stream waits for the compute stream (a stream wait,
  no host wait) and copies the tick's large records into the pinned arrays;
  `record_stream` keeps each source alive until its copy ran. Nothing is read back.
- **Streaming in, double-buffered.** Per micro-batch the host gathers the views from the
  pinned arrays (CPU index_select into one of two pinned staging buffers) and the copy
  stream sends it to the device while the compute stream works on the previous one
  (events: the copy waits for the compute stream to release that slot; compute waits for
  the copy). The next micro-batch is gathered before the current one is handed over.
- **Micro-batches** (`micro_batch_views`, default 64 views; 6 per minibatch): every loss
  term (surrogate, entropy, BC, value, next-state, hit) and every measure is a sum over the
  micro-batch divided by the whole minibatch's count (own tokens, next-valid tokens,
  labelled own tokens, from `RolloutStore.counts` once per update), so the accumulated
  gradient is the minibatch's; one clipped Adam step per minibatch. Checked: the step from
  micro-batches of 2 views equals one pass of every view within float32 rounding
  (`test_store.py`). 64 was the first size tried and fits with room (peak 5.2 GiB at
  parity, v2 game); larger sizes were not measured.
- **Targets.** `targets_every_epoch=False` (default, decided): one pass over the stored
  rewards and values (streamed) computes GAE per level (`returns.generalised_advantages`,
  a doubling scan, no loop over ticks) over each seat's segments chained in time, and the
  normalisation over the update. True: the critic's values are recomputed per epoch
  through the same streamed micro-batches. (Until 2026-11-23 GAE ran per segment, a bug;
  see "Chained returns and hit labels across segments".)
- **Host waits that remain:** at the update's start, the wait for the last copies
  (`ready`) and the read of the epochs' data orders; per micro-batch, the wait for the
  staging buffer's copy of two micro-batches before (finished by then: 4 to 7 ms over a
  whole update); the measures once at the end. The "percentile" normaliser reads a count.
- **Compact against full precision** (`test_store.py`, float32 on the CPU, initial
  weights, 6 environments, 24 ticks): largest change of a sampled action's
  log-probability 1.9e-6, mean KL 4e-9 (core game) and 8e-9 (v2), against a clip of 0.2.
  Not checked on trained weights (there are none yet) or under bf16 (whose own rounding is
  larger than float16 storage's).
- **Not measured:** micro-batch sizes other than 64; the store across a league abandon on
  the GPU (abandon runs between updates, the CPU smoke run covers it); expandable segments
  for the core game's reserved memory (7.27 GiB reserved at peak).

## Compiled training on the GPU (2026-11-23)

Measurements: docs/rewrite/scoreboard.md, "Training on the GPU" (script
docs/rewrite/sections/scoreboard/train_gpu.py). Compiled with bf16 at 256 environments, the
v2 game: collect 26k, update 6.4k, loop 5,160 environment steps per second (eager 2,110).

### What changed

- `--compile` in lean/train.py (default on; `--no-compile` eager; the smoke run stays
  eager). `collect(..., compiled=True)` and `update(..., compiled=True)` run the graphs of
  `lean/learn/compiled.py`: `act` (both views, the live policy step, sampling, value decode,
  the scripted agent), `advance` (actions, game step, rewards, hit labels, counters, league
  tallies, re-deal), `snapshot_actions` (every league snapshot), `minibatch_values` (the
  critic's target pass) and `minibatch_loss` (forward; its backward is compiled by
  AOTAutograd). Adam and clipping stay eager. All with `dynamic=True` and duck sizing off
  (set on importing lean.learn.compiled): no recompile across team sizes, environment
  counts or minibatch sizes.
- **Random draws under compile** come from torch's default generator of the device (seeded
  by `--seed` through `torch.manual_seed`), not the explicit generator: a graph cannot take
  a generator object. Eager keeps the generator. `draw_offset` replaces
  lean.model.draw_position_offset inside collect (that one draws on the generator's device,
  the CPU without one).
- **League snapshots in one call** (`snapshot_actions`): `League.weights` holds every
  snapshot slot's parameters stacked (capacity on a leading axis; a slot not yet filled
  repeats the first snapshot and is never selected), `League.template` a weightless
  `Stepper`; torch.func.vmap runs all slots on all league environments and each keeps its
  own member's output. Fixed work (capacity x L rows), **no read-back, no loop**. This
  replaced the per-member grouping with its per-tick `bincount().tolist()` (measured: the
  read-back cost 5% of collect, the eight small model calls most of the rest; collect went
  from 20,546 to 25,974 steps/s). `League.frozen` is gone. Eagerly the vmapped attention
  uses the math kernel (the fused ones reject vmap's mask layout on the GPU).
- **bf16 autocast** for the policy's forward passes, rollout and update
  (`LearnConfig.bf16`, default True, GPU only; the losses in float32). The update runs 1.8x
  faster and in two thirds of the memory. The model keeps position phases and the
  recurrence in float32 (lean.model). A choice for the author to confirm: the earlier
  lean runs were float32; v2 trained in bf16.
- (Superseded by the rollout store below: memory now follows the micro-batch, not the
  minibatch; 32 minibatches; targets once per update by default.) At the time: 8 minibatches
  fitted 8 GB at 256 environments, 4 did not; targets once per update measured 6,457 steps/s
  against 5,160.
- Host reads: the update's measures stay on the device and are read once per update (were
  eleven `float()` per minibatch); `support_usage` reads its twenty numbers in one go.
  The bootstrap value comes from `act` (one graph fewer to compile).

### Graph breaks and host synchronisations found

1. `torch.rand/randn/randint(..., generator=None)` with a symbolic size fails to trace in
   PyTorch 2.15 (fake-tensor error); without the keyword it traces. Fixed at the source
   (2026-11-23): lean.game, lean.scripted and lean.learn.collect draw through
   `lean.contract.random_keywords(generator, device)`, which leaves `generator` out when it
   is None. (Was a process-wide wrap of the three factories in lean.learn.compiled.)
2. The ship card (a functools.cache whose builder ends in `.item()`): dynamo traced through
   the cache and broke the graph; the break inside `_deal` then threw the whole deal back to
   eager (dynamo fails to resume the `uniform` closure). Fixed at the source (2026-11-23):
   `lean.game.core.ship_card` keeps built cards in a plain dict; the first deal of a run
   (`reset`, eager, before any compiled tick) builds the card and a traced deal only looks
   it up. (Was a runtime replacement of `_ship_card` plus `compiled.prepare`, both gone.)
   Before the rebase on 681eb492 the same happened with `derived_characteristics` (float
   `.tolist()`).
3. `reset_done` iterated `dataclasses.fields(State)` on the class, which dynamo cannot
   trace (gb0180). Fixed at the source (2026-11-23): it iterates the instance's fields.
   (Was a swapped `dataclasses` module inside lean.game.core.)
4. Stride recompiles: a fresh state holds broadcast tensors (stride 0, also the empty
   `fields` when F = 0) where a re-dealt one holds whole ones; collect makes them whole
   before the graphs. The opponent carry is kept contiguous for the same reason.
5. `tick` as a Python int would be a new constant every tick: it is a 0-d device tensor.
6. A snapshot group of one environment specialised a size of 1 (a second policy compile):
   gone with the batched snapshots.
7. Host syncs in the per-tick loop: the member-count read-back (gone, item above). None
   left. Per update: see "Rollout store, streaming and micro-batches" (host waits); the BC
   gate's win rate.

lean.learn.compiled no longer patches anything: it sets duck sizing off and compiles. A
compiled collect on the CPU (6 environments, 2v2 to 3v3, a snapshot in the league, two
segments, v2 and core games) after the fixes: 0 graph breaks, 3 frames (act, advance,
snapshot_actions), 0 recompiles.

### Ran

CPU, OMP_NUM_THREADS=1: `pytest lean -q -m "not slow"` passes (new: the stored-targets
test, the batched-snapshot test). `lean/learn/tests/test_compiled.py` (slow, CPU): the
compiled collect and update run with the real packages and a snapshot, and the compiled
loss and gradients equal eager. GPU: the scoreboard rows; 0 graph breaks and 0
recompiles in every row.

### Open

- The floor is met with the v2 game only, by 3%; the core game (fields, fog) runs at 4,640.
  The update dominates (5.1 of 6.3 s). Levers, all the author's: targets once per update
  (6,457), fewer epochs, a smaller update.
- Compile start-up: 199 s cold, 72 s warm (dynamo tracing alone is about 60 s).
- Snapshots still cost 3.7 ms a tick (8 slots x 128 environments of policy step); with a
  league smaller than its capacity the empty slots are computed too.

## Waste fixes 6, 7 and 9 (2026-11-23)

The learn-side items of the agreed waste fixes (lean/game/NOTES.md, "Agreed waste fixes").

- **6, one scripted call for both teams.** `lean.scripted.act(config, state, None, ...)` (and
  `marginals`, `engage`) decide for every ship of both teams in one call: own ships and
  team-mates from the state, enemies from `lean.game.core.enemy_view` (every ship as its enemy
  team knows it: under fog its last-seen row; with fog off the state itself). `scripted_actions`
  (collect's teacher and scripted member, and eval's scripted opponent) makes that one call.
  Same behaviour: bit-identical to the two single-team calls added, checked over 300 ticks of
  fights, 16 environments, 3v3 to 5v5, personas at spread 0 and 1, v2 and core configs
  (fog, fields), both spawns, with resets: marginals, targets and sampled actions all equal
  (`test_one_call_for_both_teams_equals_the_two_single_team_calls`). A single-team call is
  bit-identical to the tree before too.
- **7 and 9, deals drawn once per segment.** `collect` draws `NextEpisodes` at the start of each
  segment: E game deals (`lean.game.deal`, team sizes drawn as before), E sets of personas, E
  learner teams and position offsets. A finished environment takes the next unused deal in
  order (`lean.game.take_deal` with an index: the k-th environment to finish in the segment
  takes deal k), still with no host read; its league member is drawn as it finishes, as
  before, so the PFSP draw sees the tallies as they stand. Before, every tick dealt every
  environment afresh (the game's deal, the personas, the sizes, the learner team, the offset)
  and kept the deals of the finished ones. This is item 9 in the form that keeps behaviour:
  finished environments are not frozen; they restart on the tick after they finish, as before.
  `abandon` draws its own deals (rare, between segments).
- **Draws.** 6 changes none. 7 and 9 change which random numbers each deal and each tick
  receive (the deals are drawn at the segment start, and the per-tick deal draws are gone, so
  every later draw from the generator shifts), not their distribution: each episode still
  starts from an independent deal drawn by the same functions. One exception: a segment in
  which more than E games end reuses deals (modulo E), counted in `Rollout.repeated_deals`.
  The shortest episodes measured (128 environments, 3,000 ticks; scripted and uniform random
  actions; 1v1 and 4v4; v2 and core games) were 403 to 920 ticks against 128-tick segments,
  so a reuse needs every environment to finish more than once in a segment and was never
  seen; `test_a_segment_counts_deals_used_twice` forces it with a 4-tick clock. With
  `persona_spread` 0 (the default) personas use no random numbers at all.
- **Operations per tick** (eager, CPU, one thread, 5v5, regions counter, inclusive compute
  ops; scratch count of a 4-tick collect with self-play and the scripted league side):

  | | v2 game before | after | core game before | after |
  |---|---|---|---|---|
  | scripted teacher (`scripted_actions`) | 370 (2 x 184 + 2) | 185 | 394 (2 x 196 + 2) | 189 |
  | re-deal (`redeal`) | 149 | 61 | 342 | 65 |
  | of which the game's (`reset_done` / `take_deal`) | 87 | 18 | 280 | 22 |
  | of which personas | 29 | 0 | 29 | 0 |
  | segment deals (`draw_next_episodes`), once per segment | | 115 | | 306 |
  | first half of a tick (`act`) | 1,077 | 892 | 1,165 | 960 |
  | second half (`advance`) | 1,541 | 1,453 | 2,108 | 1,831 |

  Per tick (both halves): v2 2,618 to 2,345 (-10%), core 3,273 to 2,791 (-15%), plus the
  segment deals over 128 ticks (about 1 and 2 a tick). game_ops.py: the scripted act for one
  team 196 to 190 (the enemy view is cheaper than the team view), for both teams in one
  call 188; per tick the teacher was 392 and is 188 (the tripwire of docs/questions.md
  item 11 is 170 per call).
- **Compiled.** A compiled collect on the CPU (6 environments, 2v2 to 3v3, a snapshot in the
  league, two 3-tick segments; v2 and core games; and the core game with a 4-tick clock so
  that games end and take deals: 6 in the second segment, none reused): 0 graph breaks,
  3 frames, 0 recompiles, every value finite.
- **GPU, one row** (docs/rewrite/sections/scoreboard/train_gpu.py, compiled, bf16, 256
  environments, v2 game, 8 snapshots, 3 updates; laptop RTX 4070, profile "performance", no
  other GPU job; 2026-11-23 06:27): steady collect 28,620 steps/s (scoreboard before: 25,974),
  update 6,440 (6,439), loop 5,257 (5,160); peak 3.68 / 4.35 GiB; first update 98.1 + 92.9 s
  (compile cache cold for the changed graphs). One run each, so the collect gain (+10%) is the
  only difference larger than run-to-run noise is likely to be; the core game was not run.
  `python -m lean.train --smoke --device cuda` ran.

## Batched crossover (D6, 2026-11-23)

### What changed

- `lean.eval.play_results` plays environments of different team sizes in one batch (the
  game pads each to the largest total) and returns each environment's first result.
  `play` is that at one size, as before. `play_matches` plays a list of (network size,
  scripted size, games) in one batch and returns the wins per match (`maximum_games`
  splits a long list into consecutive batches; a match is never split).
- `crossover` runs every network size's walk at once, in rounds. A round plays, in one
  batch, the scripted size each unfinished walk needs next with its neighbours: `spread`
  (2) either side of the start, then 4 more in the walk's direction. The walk itself is
  v1's (`walk`), read from the matches played so far, so beats_up_to, crossover, the
  win rates and ratio mean what they did. The report adds `rounds`, and `matches` holds
  the neighbours too.
- **Start of the walk.** Each size walks from round(network_size * `start_ratio`):
  parity by default, or a dict of ratios by size. v1 started each size from the previous
  size's crossover, which needs that size finished first. `lean/train.py` passes the
  previous evaluation's ratios, so a training run's first crossover starts at parity and
  the later ones at the last answer. Under the walk's assumption (the win rate falls as
  scripted ships are added) the start changes which matches are played, not the answer.

### Equivalence (CPU, OMP_NUM_THREADS=4)

The scripted agent played team 0 through the policy interface (so the rates are neither
0 nor 1), against the scripted agent, v2 game with ghost bullets. Old: `play` one size at
a time (the code before this change); new: all sizes in one `play_matches` batch, padded
to 10 slots. Same number of games per size; z = difference over its standard error.

| Match | seed 1, 128 games: old / new / z | seed 2, 128 games: old / new / z |
|---|---|---|
| 2v1 | 1.000 / 0.992 / +1.00 | 1.000 / 1.000 / 0 |
| 2v2 | 0.484 / 0.445 / +0.63 | 0.523 / 0.477 / +0.75 |
| 2v3 | 0.102 / 0.141 / -0.96 | 0.148 / 0.109 / +0.93 |
| 4v3 | 0.859 / 0.883 / -0.56 | 0.859 / 0.820 / +0.85 |
| 4v4 | 0.359 / 0.422 / -1.03 | 0.531 / 0.492 / +0.63 |
| 4v5 | 0.141 / 0.250 / -2.23 | 0.234 / 0.250 / -0.29 |
| 4v6 | 0.062 / 0.086 / -0.72 | 0.039 / 0.094 / -1.77 |

Sum of z squared 14.9 over 14 comparisons (expected 14 if the two agree). Because 4v5
and 4v6 leaned the same way, a third run of 384 games each (2v1, 4v5, 4v6 in one batch,
seed 3): 4v5 0.221 / 0.237 (z -0.52), 4v6 0.099 / 0.073 (z +1.29). Pooled over the three
runs (640 games a side) 4v6 is 51 wins to 51. A first run of 16 games (seed 0) agreed
too. Script: kept outside the repository; the adapter wraps `lean.game.observe` to hand
the state to `scripted_actions` and returns one-hot logits.

### Timing (untrained lean.model policy against the scripted agent, v2 game, eager)

| Where | Sizes, games | Old | New from parity | New from the previous ratios |
|---|---|---|---|---|
| CPU, 4 threads | [4], 64 | 56.0 s, 4 matches | 102.0 s, 6 matches, 2 rounds | 40.4 s, 3 matches, 1 round |
| CPU, 4 threads | [4, 8], 32 | 47.6 s, 5 matches | 267.4 s, 16 matches, 2 rounds | 58.1 s, 6 matches, 1 round |
| GPU (RTX 4070 Laptop, profile "performance") | [4, 8], 128 | 67.6 s, 5 matches | 83.0 s, 16 matches, 2 rounds | not run |

One run each; the CPU machine had other load (load average 1.7 to 3.9). An untrained
policy beats no scripted count, the worst case for a parity start: the walk runs to 1,
and the old protocol's chained start made size 8 a single match. A pass lasts about the
same number of ticks whatever is in it (about 1,000 here: 960 to 1,152 ticks on the CPU
for 16 to 160 environments), but a tick's cost grows with environments times slots on the
CPU (7.1 ms at 16 environments, 71 ms at 160 with 18 slots), and on the GPU eagerly at
these sizes too (about 13.5 s per old pass of 128 environments, 41 s per new pass of up to
1,280). So batching pays only where a tick of the larger batch costs little more than a
tick of the smaller one, and the extra neighbour matches and the padding to the largest
team cost real work. Where every match is needed anyway it pays even on the CPU: the
equivalence runs' seven matches of 128 games took 202.8 and 200.9 s one at a time, 134.9
and 135.7 s in one batch. The GPU check covered the parity start only.

### Open

- Whether the batch is cheaper than the sequential walk on the GPU depends on the
  environment count at which the eager (or compiled) tick stops being flat; measure
  `start_ratio` from the previous ratios on the GPU, and with `games_for` at larger sizes.
- `spread` 2 is a guess: smaller means fewer wasted matches and more rounds.
- Mixing team sizes pads every environment to the largest total; grouping sizes by
  total into separate batches would cut that at the price of more passes.

## Chained returns and hit labels across segments (2026-11-23)

The bug (docs/rewrite/literature-recipe-2.md, section 1): an update stores 4 consecutive
128-tick segments per environment, but GAE ran per segment and bootstrapped from the
critic at every segment's end, so returns were cut at ticks 128, 256 and 384 although
the rewards after each cut were in the store; and a segment's last ticks (a bullet's
life) had no hit label. v2 chained the returns over the whole update and bootstrapped only
at its end (`src/bnb/learn/ppo.py:6-8, 1015-1029`). Lean now does the same (without v2's
lookahead).

### What changed

- **Chained GAE** (`update.target_pass`, `RolloutStore.chain_index`,
  `returns.join_segments` / `split_segments`). A seat is the same environment's same team
  in every segment (`collect.seats`: seat s < E is environment s's learner team, E + e the
  other team of self-play environment e; the team changes only on a re-deal, after an
  episode end, which cuts anyway), and its carry runs on across segments, so seat s's
  segments are views s, S + s, ..., (U - 1) S + s. The target pass's micro-batches are
  whole seats: k = micro_batch_views // U seats per row (16 at parity, so 64 views as
  before), segment-major, the last row padded with the spare view. A row's (T, U k)
  records are reshaped to (U T, k), GAE runs once over the 512 ticks (the same reverse
  doubling scan, log2 512 = 9 rounds instead of 7), and the results are reshaped back. The
  value after a segment's last tick is the next segment's first stored (or recomputed)
  value; only the last segment's stored bootstrap is read. `done` cuts as before,
  including on a segment's last tick. All five levels, both `targets_every_epoch` modes
  (recomputed values come per view from each view's stored carry, as before). Self-play
  and league seats chain alike (each seat its own column).
- **Reading the pinned store.** The chained pass uses the same `RolloutStore.batches`
  double-buffered stream: each row is one host index_select of U k views from the pinned
  arrays into a staging buffer, copied asynchronously while the device computes the
  previous row; every view is read once per pass, nothing is read back, and the micro-batch
  shape is unchanged at parity (64 views), so the compiled critic graph is too.
- **Hit labels.** `collect`'s flat labels have 2T rows: a bullet that ends in segment u
  but left the gun in segment u - 1 lands in the first T rows, and
  `RolloutStore.end_segment(..., previous_hit_label)` merges them into segment u - 1's
  labels (elementwise max: a row's bullet ends once, so the other label is -1). Segment
  0's earlier rows belong to the previous update and are dropped. A Rollout collected
  without a store keeps only its own rows (unchanged). `advance` takes the tick's label
  row (`label_row`, starting at T) instead of the segment tick.
- Training sequences are unchanged: one view (one seat's 128-tick segment) from its stored
  carry; minibatches and micro-batches as before.

### Tests (CPU)

`test_store.py`: `test_chained_targets_equal_one_gae_pass_over_the_concatenated_ticks`
(4 segments of 12 ticks, stand-in game, random rewards and values on every level, an
episode end inside segment 2 of one seat and one on segment 1's last tick of another;
the target pass equals one GAE pass over each seat's 48 concatenated ticks within float32
rounding, stored and recomputed values; fails with the per-segment pass);
`test_hit_labels_near_a_segment_boundary_come_from_the_next_segment` (three 8-tick
segments, 2-tick bullets: every row labelled as its bullet's outcome except those whose
bullet outlives its episode or the update). The streamed-store test now checks that the
store agrees with the unstreamed rollouts on every row those label, and that it labels
the first segment's tail in addition.

