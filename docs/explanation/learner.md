---
covers: [src/bnb/learn/ppo.py, src/bnb/learn/returns.py, src/bnb/learn/losses.py, src/bnb/learn/labels.py, src/bnb/learn/shots.py, src/bnb/runtime/store.py, src/bnb/models/relational.py]
---

# The learner

<!-- generated: section-status -->

The learner is PPO with v1's math over any model that has an `unroll`: per-level returns,
the categorical critic, the return scale and the auxiliary losses with their labels. It is
`bnb.learn.ppo.Learner`, configured by `bnb.learn.ppo.LearnerConfig`. The training loop
around it is in [The training loop](training-loop.md); the design record is
`docs/rewrite/sections/learner-spec.md`.

## The model protocol

The learner trains a model that

- acts through its controller in the rollout, and
- has `unroll(carry, obs, replay, reset)`, which recomputes a stored segment from the
  [carry](glossary.md#carry) at its start and returns every head's output over it.

`reset` is the flag a controller's `step` received: the episode ended before this
observation. The learner reads it off the stored observation's `reset` column, so
`reset[t]` is the stored `done[t - 1]`
(`tests/sections/test_learner.py::test_unroll_receives_the_reset_before_each_step`).

A head the model lacks is a missing field of its output, and its loss is skipped. Yemong
(`bnb.models.yemong`) has every head. The toy model, `bnb.models.relational.Relational`,
has the policy and the critic only.

## Where the segments live

An update consumes `shards` segments of `length` ticks each, collected one after another,
each continuing from the carry the last one ended with. The rollout records the learner's
views into a staging ring of two graph blocks on the device, and after each block copies the
block into a `bnb.runtime.store.SegmentStore` on a copy stream. The store keeps the segments
in pinned host memory (`host_store`, the default) or on the device. The update stages
micro-batches of views back the same way, gathering and copying the next one while the
current one computes. A rollout from the same seed writes the same records into either
store, with one shard or several, and the update's counts, losses and gradients are the
same bit for bit
(`tests/sections/test_learner_streaming.py::test_the_update_through_pinned_memory_equals_the_device_path`).
Readers wait on the store's last write: the GPU waits where the data stays on the device,
and the host only where it reads pinned memory itself. An update synchronises with the
host once per epoch, for the early stop, and once at its end, for its results
(`tests/sections/test_learner_streaming.py::test_an_update_synchronises_only_to_read_its_kl_and_its_results`).

A segment holds, per view and tick:

- the observation parts the model reads, every row's action in view order, the
  log-probability, value, values per level, outcome probabilities, rewards and `done`, and
  the model's replay record;
- the label inputs: the enemy rows of the fog-off privileged view (`truth`; an own ship is
  always visible, so its row is the observation's), every row's `alive` flag and each own
  ship's `valued` flag;
- the shot capture (`bnb.learn.shots.ShotCapture`).

Per segment it holds the learner's carry at the start and the privileged rows of the
observation after the last tick. After a segment's last tick, `bnb.runtime.rollout.Rollout.bootstrap`
evaluates the learner's controller on the next observation without acting, for the values
and outcome probabilities the returns bootstrap from.

## Returns

For each shard, on the device:

- **Values in reward units.** A per-ship level's value is its bin expectation; the outcome
  level's is `P(win) - P(loss)` times the outcome's reward weight.
- **Per-level GAE** (`bnb.learn.returns.per_level_gae`), each level with its own discount and
  trace decay (`LearnerConfig.discount`, `LearnerConfig.trace_decay`). `done[t]` cuts the
  trace after `t`; a timeout is a real end.
- **Team mixing and the level sum** (`bnb.learn.returns.team_aggregate`): a ship's advantage
  is the sum over levels of its own advantage plus each level's team spirit times the mean
  over its living teammates. Team spirit is 0 on every level by default.
- **The outcome's targets:** lambda targets of the outcome distribution
  (`bnb.models.yemong.critic.outcome_targets`), with the view's result at an episode end.

The return scale is the gap between exponential moving averages of the 5th and 95th
percentiles of the summed returns of living own ships, at least 1, updated once per update
(`bnb.models.yemong.critic.ReturnScale`). Advantages are divided by it, with no mean
subtracted.

## Losses

Every term is a mean in nats per token. Each divides its masked sum by the mask summed over
the whole minibatch, so the micro-batches of a minibatch add up to the unsplit minibatch
(`tests/sections/test_learner.py::test_micro_batches_accumulate_to_the_unsplit_minibatch`).
The coefficients are the fields of `bnb.learn.ppo.LearnerConfig` named after the terms.

| Term | Tokens | Definition |
|---|---|---|
| `policy` | own ships alive at `t` | the clipped surrogate on the scaled advantage |
| `entropy` | the same | the joint action distribution's entropy (a negative coefficient rewards it) |
| `value` | own ships alive or waiting to respawn; each view-step for the outcome | per level, the cross-entropy against the two-hot of the return, the outcome against its targets |
| `next_state` | rows whose belief is valid at `t`, where `t` did not end an episode | the state code's cross-entropy against the sharp code of the true state at `t + 1`, in the model's shifted frame |
| `enemy_action` | enemy rows alive at `t` | the cross-entropy of the enemy's actual joint action |
| `density` | each view-step, over its count of valued own ships | Poisson terms over the hex cells, against the counts of living ships at `t` |
| `shot_trajectory` | valid shots of own ships and of enemies visible at `t` | the trajectory code's cross-entropy, averaged over the shot's valid ages |
| `shot_outcome` | the same shots | the outcome code's cross-entropy |

Each term matches an independent computation on a stored segment
(`tests/sections/test_learner.py::test_each_loss_term_matches_an_independent_computation`).

An update makes `epochs` passes of `minibatches` minibatches over the views; a view keeps
its whole segment, which `unroll` needs. It clips the gradient norm, steps Adam with a
linear warmup of the learning rate over the first environment steps, and stops early after
an epoch whose mean approximate KL divergence exceeds `target_kl`. The minibatch order and
the shot spread come from the learner's own random streams (`bnb.learn.ppo.random_stream`,
seeded from `LearnerConfig.seed`), so the global generator does not change an update. The old log-probabilities are
the rollout's, or recomputed by `unroll` before the first epoch
(`LearnerConfig.old_log_probabilities`).

## Shot labels

Every tick, each living ship's shot is labelled at every age of its flight, by firing it
again with the game's own code (`bnb.learn.shots.resimulate`):

- **Capture.** The rollout stores each ship's launch state as `fire` reads it
  (`bnb.game.physics.bullets.launch_state`), its move that tick, and whether it was alive.
- **Scan.** One bullet per row and age, newest first. Each tick every row fires a fresh
  bullet with `bnb.game.physics.bullets.launch`, every bullet moves with
  `bnb.game.physics.bullets.move_bullets`, and the game's swept test
  (`bnb.game.physics.combat.contacts`, `bnb.game.physics.combat.first_contact`) runs it against
  the tick's stored ship motion. The spread is drawn from the learner's own generator, once
  per update. The re-simulation runs once per update, before the epochs, and its labels go
  into the store beside the segments; every epoch reads them back.
- **Each shot records** at every age its displacement and velocity in the launch heading's
  frame and the refractive index, and its first hit (an enemy or an ally of the shooter), or
  else its closest pass and that ship's side. Ages after a hit are invalid. A shot whose
  flight crosses an episode end is invalid, and so is every shot launched within the
  longest flight of the segment's end, whatever it hit: keeping the ones that hit in time
  would bias the labels toward hits.

A re-simulated shot is the bullet the game fires from that state, position, velocity and
hit, tick by tick, including friendly fire and fields (`tests/sections/test_learner_shots.py`).
The targets are Yemong's shot codes (`bnb.models.yemong.shots`), and the trajectory head is
queried at every age at once.

## The other labels

Built per micro-batch on the device, from the stored rows (`bnb.learn.labels`), row-aligned
with the view (as are the shot labels' codes; expanded, the codes would be many times the
size of the segment):

- **Next state:** the sharp code of each row's truth at `t + 1`, its position shifted by the
  view's offset; the last step's truth is the observation after the segment.
- **Enemy action:** each enemy row's joint action at `t`.
- **Density:** the hex counts of the living ships of each side at `t`, in world coordinates
  scaled by each environment's arena radius.

## Compilation and precision

Section code does not compile itself. The learner takes a `stage` function from the runtime
(`bnb.runtime.train.compiler`), which compiles the micro-batch loss and the
re-simulation's tick (`bnb.learn.shots.tick`). Matmuls are float32 by default.
`TrainConfig.precision` selects TF32 (a process-wide setting the entry point applies with
`bnb.runtime.train.set_precision`) or bf16 matmuls inside the model; recurrent state, norms,
scans and every loss reduction stay float32 in all of them.

## The toy learner

`bnb.models.relational.Relational` is a fixed-size relational model for small team
deathmatch. Each own ship reads its own state and, for every other row of the view, that
ship relative to it in its heading frame: position, velocity, distance and relative heading
where it is visible, and its team, `alive` and `visible` flags. Two hidden layers feed the
joint 42-way policy, the per-ship categorical critic, and the outcome from the mean of the
own ships' hidden states. It is stateless, so its `unroll` is its forward over every step.
Trained from scratch against the chaser at 2v2, it must win at least 60% of 256 games after
the budget in `tests/loop/test_learning.py`; the [training guide](../guides/training.md#the-toy-learner)
shows how to run it.
