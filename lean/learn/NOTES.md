# lean/learn and lean/eval: notes

What is, and what is open. History: `docs/rewrite/core-worklog.md` and this file's git log
(the long notes before 2026-11-23's step L1 are there). Decisions: `docs/rewrite/core-decisions.md`.

## Modules

- `config.py`: `LearnConfig` (every number, with its reason), `learning_rate(config, step)`.
- `rollout.py`: `start`, `collect`, `abandon`, `Opponent`, `CollectCarry`, `Rollout`; seats,
  league opponents, hit labels, the teacher's distribution (`teacher_marginals`, `teacher_joint`).
- `store.py`: `RolloutStore`, an update's segments, compact, streamed to pinned host memory on a GPU.
- `returns.py`: GAE per level, chaining across segments, advantage scaling and shares, the
  value supports' range monitoring.
- `training.py`: `update` (both stages), `TeacherTerm`, `make_optimiser`, the losses.
- `weights.py`: `WeightAverages` (EMA and uniform), checkpoints.
- `pool.py`: `League` (the scripted agent and snapshots, PFSP draws, decaying tallies).
- `rewards.py`: v2's five reward levels and the damage ledger.
- `compiled.py`: `graphs()`, the compiled tick halves, snapshots, target pass and loss.
- `lean/train.py`: the stages. `lean/eval`: matches, `crossover`; `python -m lean.eval`.

## Commands

    python -m lean.train                                   # BC (96 updates), then RL (--updates, 1000)
    python -m lean.train --stage bc --out DIR              # BC only: DIR/bc.pt
    python -m lean.train --stage rl --start DIR/bc.pt --out DIR   # RL from the BC hand-off: DIR/rl.pt
    python -m lean.train --smoke --device cpu [--stage bc|rl]     # tiny, eager
    python -m lean.eval DIR/rl.pt [--weights ema|uniform|live] [--sizes 4 8] [--games 64]

## The recipe (decided 2026-11-23)

- **BC stage** (`update(..., cloning=True)`): the student plays its own games (half self-play,
  half against the scripted agent; no snapshots join the league). Loss: the cross-entropy of
  the 42-way policy against the scripted teacher's full distribution (the product of
  `lean.scripted.Marginals`' stick, power and fire, recorded per ship each tick) at
  `bc_coefficient` 1.0, plus value, next-state and hit as in RL; no surrogate, no entropy
  bonus. The critics learn the stage's own returns. 96 updates by default: 100.7M ticks at
  parity, v2's 100M-tick stage. `bc_kl` is the cross-entropy less the teacher's entropy.
- **Hand-off**: RL starts from the BC stage's EMA (`handoff`), with a fresh optimiser, warm-up,
  league and averages.
- **Teacher term in RL** (`TeacherTerm`): the same cross-entropy at 0.5 x 0.95^k at RL update k,
  off for good once the league's win rate against the scripted agent reaches 0.45 (it reads 0
  until a game against it has ended) or the weight falls below 0.005 (from RL update 90).
- **Weight averages** (`weights.py`): an EMA (0.95 per update) and a uniform mean of the weights
  after every update from the end of the warm-up; side copies, never read by training; saved
  with the live weights in each stage's checkpoint (`DIR/bc.pt`, `DIR/rl.pt` at the stage's end).
- **Optimiser**: Adam, no weight decay, epsilon 1e-5, fused (on the CPU too), global clip 1.0;
  a step whose gradient norm is not finite is skipped through the fused optimiser's
  `found_inf` (no host read) and counted (`skipped_steps`, a warning line). Learning rate 3e-4
  after a linear warm-up of 2,000 Adam steps (16 updates of 128 steps), counted per stage.
- **KL**: no early stop; `approximate_kl_epoch_<k>` per epoch, `kl_warning` above 0.02.
- **Advantages**: summed over levels, divided by their standard deviation over the update,
  mean kept. `advantage_share_<level>` = Cov(A_level, A) / Var(A) (the shares add up to 1),
  `advantage_std` the scale before the division.
- **Losses**: fixed coefficients on scale-free terms: surrogate 1, value 0.05 (nats per level),
  next-state 0.1 (nats per column), hit 0.05 (nats), entropy 0.003 on the joint 42-way
  entropy in RL; `entropy_fraction` = H / ln 42. No gradient-share tuning (lean never had it).
- **Crossover**: evaluation only (`python -m lean.eval`), not in the training loop.

## Rollouts and the update (as built before L1, unchanged)

- Seats: the first half of the environments are self-play (two live seats each), the rest
  league games (one live seat against a league member: the scripted agent or a snapshot,
  vmapped in one call). Deals are drawn once per segment.
- An update is 2,048 environments x 4 segments x 128 ticks; returns chained over the 512 ticks
  of each seat, bootstrapped only at the end; hit labels chained across segments; 32
  minibatches of views, micro-batches of 64 views with gradient accumulation; targets once per
  update from the stored values (`targets_every_epoch=False`).
- Rewards: v2's five levels with the ledger (`rewards.py`); supports damage +-7, kill_death +-8,
  outcome +-12, objective levels +-1 (placeholders); range warnings per level.
- The store keeps positions, PPO numbers float32, other floats (the teacher's 11 probabilities
  included) float16, actions int8; the teacher adds 22 bytes per ship row.
- Compiled: `dynamic=True`, duck sizing off from the first compiled call; 0 graph breaks and 0
  recompiles recorded before L1 (scoreboard); not re-counted in L1.

## Tests

All learner, evaluation and caller tests run the real game, scripted agent and model at tiny
sizes (`tests/setting.py`; the stand-ins are gone). `StepRecorder` records the game steps
(hit-label and self-play checks against a search over them). `test_recipe.py`: the teacher's
joint, the soft-target cross-entropy, the BC stage training every head and ignoring the
advantages, the teacher term's decay, gate and latch, the averages' formulas and that they are
not fed back, checkpoints, the optimiser settings and the non-finite skip, the warm-up, the KL
monitor, the advantage shares. `test_train.py`: `lean.train` BC then RL from the checkpoint.

## Ran (step L1, 2026-11-23, CPU, OMP_NUM_THREADS=4)

`pytest lean -m "not slow"`: 185 passed. Slow: 3 passed (compiled collect and update, compiled
loss equals eager). `python -m lean.train --smoke --device cpu` with `--stage bc`, `--stage rl
--start`, and the default (both). `python -m lean.eval` on the smoke checkpoint.

GPU: see "GPU check" below.

## Open

- For L2: the averages and the cheese agents in the league, capacity routing with retiring
  slots, v2's snapshot schedule, checkpoints every 10M steps (now only at a stage's end), a
  resume (the optimiser's state is not saved).
- The teacher term's numbers (0.5, 0.95 per update, cut-off 0.005) and the BC length are
  defaults with reasons, not measured here; v2's BC ran 256 minibatches (8x the Adam steps per
  update); lean's BC uses the RL minibatches (32).
- Entropy stays on during the teacher phase (VPT dropped it there); the decision is a constant 0.003.
- The uniform average in BC starts after the warm-up and includes the stage's early weights;
  the hand-off uses the EMA.
- The events bridge (`rewards.tick_events`) and the supports: as before (objective supports are
  placeholders).
