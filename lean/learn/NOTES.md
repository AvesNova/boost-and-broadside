# lean/learn and lean/eval: notes

## Implemented

- `config.py`: `LearnConfig`, `bc_weight`, `shaping_scale`, `learning_rate`.
- `rewards.py`: `rewards(step_out, state, config, shaping) -> (E, N, LEVELS)`.
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
