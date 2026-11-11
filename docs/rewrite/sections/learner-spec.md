# Learner spec (section 6): PPO and labels

Status: spec, 2026-11-01. Decisions: `learner.md`. Fixtures: `tests/reference/learner/`
(README there); the critic distributions, codes and Poisson terms are already pinned by
`tests/reference/yemong/`.

The learner trains a model through PPO with v1's math: per-level returns, the categorical
critic, the return scale, and the auxiliary losses with their labels. It replaces the
skeleton's `bnb/learn/ppo.py`. **(v2)** marks a decided change from v1. **(measure)**
marks something this section must measure and report.

## 1. The model protocol

The learner trains any model exposing:
- `step` through its controller (rollout);
- `unroll(carry₀, obs, replay, reset)`, which returns an `Unrolled`.

Heads a model lacks give `None`, and their losses are skipped. Yemong provides all of
them; the toy learner (section 9) provides the policy and the critic.

## 2. The data path (v2: through pinned CPU RAM)

- **A segment** is `L = 128` ticks of the learner's `R` views. An update consumes
  `shards` segments, collected one after another, each starting from the carry the last
  one ended with.
- **Stored per segment, for the learner's views:**
  - the observation parts the model reads, the actions, the replay, the rewards
    `(L, R, S, 5)`, the values, the outcome probabilities, `done`, and the initial carry;
  - the label inputs: the fog-off privileged rows of every view (`Privileged.ships`, in
    the view's row order) and every row's action (own and enemy, in view order);
  - the shot capture (section 6).
- **Streaming:**
  - each tick's records go into a small GPU staging ring, then to pinned host memory on
    a copy stream, asynchronously;
  - the update stages micro-batches back the same way, prefetching the next while the
    current one computes;
  - everything derived from the stored data (returns, labels, aggregates) is computed on
    the GPU one shard at a time.
- **Bootstrap:** after a segment's last tick, the learner's controller evaluates the next
  observation without acting, for the values and outcome probabilities `p(s_L)`. That
  observation is the next segment's first.
- **Compaction (v2):** the replay stores the belief for hidden rows only. A visible row's
  means are the observation's shifted truth and its spreads are zero, so `unroll` rebuilds
  them. **(measure)** the bytes per view-tick before and after.
- **Budget:** streaming costs under 5% of rollout and update throughput together
  **(measure)**, against an all-GPU path at a size that fits.

## 3. Returns (v1; fixtures `learner_gae`, `learner_team_aggregate`)

- **Levels** in `REWARD_LEVELS` order, each with its own γ and λ:

  | level | γ | λ |
  |---|---:|---:|
  | damage | 0.991 | 0.90 |
  | kill_death | 0.995 | 0.95 |
  | objective_progress | 0.995 | 0.95 |
  | objective | 0.999 | 0.97 |
  | outcome | 0.9997 | 0.97 |

- **Values in reward units:** a per-ship level's value is its bin expectation. The outcome
  level's is `P(win) − P(loss)` times the outcome weight. The views are team-relative, so
  that is already each own ship's frame.
- **GAE per level** (v1's `compute_gae`): `done[t]` cuts the trace after `t`, and a timeout
  is a real end. The last step bootstraps from section 2's values.
- **The outcome's categorical targets:** v1's `outcome_targets` with the outcome's γ and
  λ, from the stored `p(s_t)` and the bootstrap `p(s_L)`.
- **Team mixing and the level sum:** `A_i = Σ_k (A_ik + s_k · mean_{living teammates j ≠ i}
  A_jk)`, and the same of the returns. Team spirit `s` is 0 on every level by default.
- **The return scale:** EMAs (decay 0.99) of the 5th and 95th percentiles of the summed
  returns of living own ships, floor 1. Advantages are divided by it, with no mean
  subtracted.

## 4. Losses (v1; fixtures `learner_policy`, `learner_critic`, `learner_shot_cross_entropy`)

Every term is a mean in nats per token. Each divides by its **minibatch-total**
denominator (the mask summed over every micro-batch of the minibatch), so accumulating
micro-batches equals one unsplit minibatch exactly.

| term | coefficient | tokens (mask) | definition |
|---|---:|---|---|
| policy | 27 | own ships alive at `t` | clipped surrogate, clip 0.2, on the scaled summed advantage |
| entropy | −0.0081 | own ships alive | the joint action distribution's entropy |
| value | 0.007 | own ships alive or waiting to respawn; the outcome per view-step | per-level cross-entropy: per-ship levels against the two-hot of the realised return, the outcome against its targets; level weights 1 |
| next state | 0.0041 | every row whose belief is valid at `t`, where `t` did not end an episode | the code's cross-entropy summed over its groups, against the sharp code of the true state at `t + 1` |
| enemy action | 0.11 | enemy rows alive at `t` | the cross-entropy of the enemy's actual joint action at `t` |
| density | 0.021 | per view-step, divided by the valued-ship count | Poisson terms summed over cells, against the hex counts of living ships at `t` |
| shot trajectory | 0.007 **(v2)** | valid shots of own ships and of visible enemies | the cross-entropy summed over parts, **averaged over 4 queried ages** (the last, a uniform one, the closest pass, the first bend; v1 summed two ages at 0.0035) |
| shot outcome | 0.019 | the same shots | the outcome code's cross-entropy |

- **The next-state target** is in the model's frame: the true position plus the view's
  position offset from the replay.
- **The density target** is in world coordinates, relative to the arena centre and scaled
  by each environment's arena radius.
- **Optimizer:**
  - Adam at 3e-4, with a linear warmup from 1e-7 over the first 1M environment steps;
  - gradient clip 1.0;
  - 4 epochs of 32 minibatches over the views (a view keeps its whole segment, which
    `unroll` needs);
  - early stop when the approximate KL exceeds 0.02.
- **Old log-probabilities:** either the rollout's, or recomputed by `unroll` before the
  first epoch. It is a setting, measured in section 8.

## 5. Next-state, enemy-action and density labels

All are built on the GPU per shard, from the stored privileged rows and actions. Next
state and enemy action are row-aligned with the view, as the model's rows are.

## 6. Shot labels (v2: every age, the game's own code)

- **Capture:** every tick, the game exposes each ship's launch state, as the game reads it
  when it fires: after the tick's kinematics, before its hit test. The rollout stores it
  with the tick's ship sweeps and the field layout.
- **Re-simulation, per shard on the GPU:** for every living ship on every tick, fire the
  bullet the game would fire, fly it and test it against the stored ships of the
  following ticks.
  - It uses the game's own launch, flight and swept-hit functions (`bnb.game`, imported by
    `learn` as the layering allows). Nothing is reimplemented.
  - The spread is drawn from the learner's own generator.
- **Each shot records,** at every age `a = 1 … 12`, the snapshot: the displacement and
  velocity in the launch attitude's frame, and the refractive index. It also records its
  first hit (enemy or ally of the shooter), or else its closest pass and that ship's side.
  - Every age of a valid shot is labelled: past a hit, the bullet's straight-on flight
    (the trajectory code has no hit component; the outcome code keeps the first hit).
  - A shot is invalid if its whole flight crosses an episode end or the window's end.
- **Targets** are Yemong's shot codes: the trajectory code at 4 ages per shot, and the
  outcome code. The ages: the last move, one uniform over the flight, the closest pass to
  another living ship and the first move into a field's band (each of the last two a
  uniform age of its own when there is none), drawn by the learner's generator.

## 7. Mixed batches and metrics

The batch is section 3c's: pools with their own modes, maps and fog in one batch. Losses
are computed over the whole batch. The progress metrics per pool are the learner reward
per level and the episode results, plus the losses, KL, clip fraction and explained
variance per level.

## 8. Precision (decided by measurement)

The deliverable is a matrix. Each row is one configuration:
- **precision:** float32, TF32 or bf16 matmuls;
- **old log-probabilities:** the rollout's, or recomputed.

For each row, measure:
- rollout and update time;
- the step/unroll ratio drift;
- the clip fraction and KL at the first epoch;
- a short seeded learning comparison: the toy learner's curve, three seeds each.

In every configuration, recurrent state, norm statistics, scan accumulations and every
loss reduction stay float32. The default stays float32 until the author chooses from the
matrix.

## 9. The toy learner

- **The model:** a fixed-size relational MLP for 2v2 team deathmatch, in
  `bnb/models/relational.py`. For each own ship it reads its own state and, for each of
  the other three ships:
  - position and velocity relative to it, rotated into its heading frame;
  - distance;
  - relative heading (cos, sin);
  - team (ally or enemy), `alive` and `visible`.

  It has two hidden layers, the joint 42-way action, and the categorical critic with the
  outcome.
- **Stateless:** its carry is empty, and its `unroll` is the per-step forward.
- **Acceptance:** trained by this learner from scratch on 2v2 team deathmatch against the
  chaser (the scripted controller), it must beat the chaser, with a win rate of at least
  60% over 256 evaluation games, within a fixed budget the section measures and records
  (minutes on the dev GPU). It replaces the skeleton's toy learning test.

## 10. Acceptance

1. **Equivalence:** `tests/reference/learner` passes through `learner_arm.py`.
2. **Loss terms:** each matches an independent computation on a real stored segment,
   masks included.
3. **Micro-batching:** the gradient accumulated over micro-batches equals the unsplit
   minibatch's gradient, to float32 tolerance.
4. **Shot labels:** a shot re-simulated from captured state equals the bullet the game
   actually fires from that state (position, velocity and hit, tick by tick), including
   friendly fire, fields and a shot that ends at an episode end.
5. **Labels:** next-state and enemy-action labels are row-aligned and in the model's frame;
   the density target sums to the living ships per side.
6. **Streaming:**
   - the update through pinned CPU memory gives the same losses as the all-GPU path,
     exactly;
   - the cost budget of section 2 holds;
   - the 26-ship bench cell (1,576 environments) now fits in 8 GB, or the report says what
     still does not.
7. **The toy learner beats the chaser** (section 9).
8. **The precision matrix** (section 8).
9. **Throughput:**
   - environment steps per second end to end (rollout and update);
   - the 5v5 training cells (`deathmatch`, `deathmatch_wide`, `frontline`, `mixed`) with
     Yemong as the learner, against the rollout alone.
