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
- Minibatches are whole environments with the full segment; default 4.
- Advantages and value targets are recomputed from the current critic at the start of
  every epoch (one extra no-grad pass over the rollout per epoch).
- `collect`, `update`, `crossover` keep the contract's positional signature and add
  keyword-only options (`shaping`, `game`, `scripted`, `scripted_win_rate`,
  `update_index`, `generator`, `match`). `opponent` is an `Opponent`, `carry` a `CollectCarry`.

## Assumptions about the sibling packages

- The carry's environment axis is 1 when the carry has 4 dimensions (lean.model:
  blocks, E, 1 + N, width) and 0 otherwise (`carry_axis`). The learner zeroes the carry
  itself on reset and does not use `Policy.step`'s optional `reset` or `position_offset`.
- `game.reset` sizes N from the largest total dealt, so `start` forces environment 0 to
  the maximum size; `reset_done` is assumed to keep N.
- `scripted.act` is called without its optional memory argument and with uniforms
  (E, N, 3) from `torch.rand`, not `scripted.draw_uniforms`; personas are redrawn for
  the whole batch every tick and selected per finished environment (every persona
  field is assumed to have a leading E).
- Next-state loss: the target here is the change in `obs.ship`, used only if the head's
  width equals SHIP_FEATURES. lean.model's head predicts scaled changes of State fields
  (`model.next_state_target(config, state, next_state)`), so with the real model the
  next-state loss is currently ZERO. To fix: record that target in `collect`.
- The value helpers are local copies; lean.model has `hl_gauss_targets` and its own
  centres, which may differ from `value_centres` here. Unify before trusting values.
- Stepping continues after `done` only through `reset_done`.

## Known weaknesses

- `collect` draws personas, team sizes and learner teams for all environments on every
  tick to avoid a host branch; cheap, but worth moving into a reset-only path if it shows up.
- `team_view` calls `observe` twice per tick (three or four times with a snapshot).
- `play` reads one flag back every 32 ticks.

## First on the GPU

1. `python -m lean.train --smoke` with the generator and model on the device (train.py
   builds a CPU generator; device selection is not wired yet).
2. Steps per second of `collect` alone and of `update` alone at 256 and 2,592 environments.
3. Peak memory of a 128-tick rollout: observations are stored whole, and the
   per-epoch target pass holds value logits for a minibatch of environments.
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
  `tick - (state.tick - shot_tick)` (episode ticks and rows advance together within an
  episode); a bullet fired before the segment goes to a spare entry; rows whose bullet has
  not ended by the segment's end stay -1. One scatter per tick, nothing read back. With
  ghost bullets every living ship's every row is labelled except the segment's tail.
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
`hit_labelled` is 0: its 8-tick segments are shorter than a bullet's 11-tick life, so
only same-segment hits label a row and none occurred. The value-support flags were quiet
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
- Update size (v2: 2,048 environments x 4 shards x 128 ticks, 32 minibatches; lean 256 x
  128, 4 minibatches).
- Also differing from v2 and untouched here: advantage scaling (v2 a slow EMA of the
  returns' 5-95% gap, floor 1, once per update; lean the advantage std every epoch), the
  policy coefficient (27) against value (0.007 summed), the 128-tick lookahead before the
  bootstrap, the entropy settings, Adam's epsilon.
