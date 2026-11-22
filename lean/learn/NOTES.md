# lean/learn and lean/eval: notes

## Implemented

- `config.py`: `LearnConfig`, `bc_weight`, `shaping_scale`, `learning_rate`.
- `rewards.py`: `rewards(step_out, state, config, shaping) -> (E, N, LEVELS)`.
- `collect.py`: `start`, `collect`, `Opponent`, `CollectCarry`, `Rollout`, team-size draws.
- `returns.py`: categorical value helpers (`value_centres`, `expected_value`, `hl_gauss`),
  `generalised_advantages`, `normalise_advantages`.
- `update.py`: `make_optimiser`, `update`.
- `pool.py`: `SnapshotPool`.
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
- Half of the environments always face the scripted agent.
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
- The league is a uniform pool of the last 8 snapshots (v1: rating-distance sampling,
  half live self-play). There is no live self-play on both sides: the non-scripted half
  plays a frozen snapshot, and only one team per environment trains.
- Crossover's first size walks from parity instead of doubling and bisecting (same
  answer under the monotonic assumption, more matches when far from parity).

## Choices the plan left open

- `individual_share` 0.5; `damage_unit` 1.0 (damage assumed to be a fraction of full
  health: check against lean.game's damage units and set it); `speed_unit` 100 m/s.
- Value supports: half-widths 1.5, 2, 2, 4 per level; sigma 0.75 bin widths; bins read
  from the model's output. Targets beyond the support are clamped.
- Facing uses the nose (`angle_cos/sin`), closing the velocity (`heading_cos/sin * speed`).
- Environments with index below scripted_share * E face the scripted agent; fixed, not redrawn.
- Scripted win rate: exponential average (momentum 0.9) over rollouts with at least one
  finished scripted game, ties as non-wins, starting at 0.
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
