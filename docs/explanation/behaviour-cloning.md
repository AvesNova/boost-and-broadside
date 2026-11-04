---
covers: [src/bnb/learn/bc.py, src/bnb/runtime/cloning.py]
reviewed: 2026-11-04 12:00
---

# Behaviour cloning

<!-- generated: section-status -->

Behaviour cloning (BC) gives every head of a model a head start before reinforcement
learning: the policy, the critic, next state and the rest. The student flies the
learner's views, and the teacher, the [scripted agent](scripted-agent.md), labels the
states it visits (on-policy imitation). BC runs alone; reinforcement learning then starts
from its weights and return scale with a fresh optimizer and no cloning term. How to run it
is in the [behaviour cloning guide](../guides/behaviour-cloning.md).

## The learner

`bnb.learn.bc.BehaviourCloning` is [the learner](learner.md) with one change: the
policy's loss is the cross-entropy of the student's joint 42-way distribution against the
teacher's labels, in nats per token, over own ships alive, with coefficient 0.025
(`bnb.learn.bc.config`). Everything else is the learner's: the segments, streaming,
`unroll`, micro-batching, mixed batches, the critic on the student's own returns, the
auxiliary heads at their coefficients, and the return scale, which runs during BC and is
handed on. There is no policy gradient, no entropy bonus and no KL early stop.

## The labels

The teacher labels the stored observations at update time, on the device, in the compiled
loss: the scripted agent is memoryless and reads only the observation, so the rollout
stores nothing for it beyond the observation parts it reads (`Rollout` records them with
`extra_reads`). On the same observation the labels are the distribution the scripted agent
plays (`tests/sections/test_bc.py`).

A temperature and a uniform floor set the labels' entropy, so the clone starts
reinforcement learning at a chosen entropy without an entropy bonus
(`bnb.learn.bc.TeacherConfig`): `q_T ∝ q^(1/T)` over the joint, then `(1 − ε) q_T + ε / 42`.
One global temperature keeps the teacher's relative confidence between states, which the
clone learns. A structural zero (turning away from the target) stays zero under the
temperature alone; the floor lifts every zero and reaches entropies the temperature cannot.
In BC's own games the scripted opponent plays the labels' distribution
(`bnb.learn.bc.TeacherOpponent`: each marginal tempered, then a uniform joint action with
probability ε), so the live score is against the teacher the clone copies. At T = 1 and
ε = 0 it draws exactly the scripted agent's actions. Everywhere else the scripted agent
plays its natural distribution.

The floor defaults to ε = 0.01 (`bnb.learn.bc.FLOOR`), spread over all 42 joint actions,
so no label gives an action probability 0. The teacher itself puts weight on about 2.4
joint actions per living ship-step on the training mix ([measured](#measurements)); a
clone trained on its distribution unfloored would give the rest probabilities near 0, and
PPO never samples an action of probability 0. With the floor, the clone can sample every
action in reinforcement learning.

`bnb.learn.bc.calibrate` bisects the temperature to a target mean entropy, in nats per
living ship-step, over the scripted agent's own distributions in its own games of the
training mix (`bnb.runtime.cloning.teacher_distributions`).

## Rollouts

`bnb.runtime.cloning.LINEUP`: half self-play, half against the scripted agent playing the
labels' distribution, crossed with the training mix in a mixed batch. The live scores per
opponent come from the training games themselves; there are no evaluation rollouts.

## When it stops

`bnb.learn.bc.StopRule`, fed once per update by `bnb.runtime.cloning.update`, names the
first rule to fire (`bnb.learn.bc.StopConfig`, the run's `TrainConfig.stop`). Its state is
part of the training checkpoint, so a resumed run continues the same window:

- **score:** the score against the scripted agent (wins plus half the draws) over the last
  1000 finished training games, less two of its standard errors, reaches 0.40, even if
  the KL is still falling. The standard error is the window's own: the sample deviation of
  its per-game scores (1, 0.5 or 0) over the square root of 1000. The rule is checked
  every update, so the bound keeps a window that is high by chance from stopping BC. The
  window keeps each update's games, points and squared points, drops the oldest updates
  whole and counts the oldest one it keeps by the share of its games the window needs, so
  trimming favours no result;
- **KL plateau:** the BC KL, `KL(labels ‖ student) = CE − H(labels)` per token, as an
  exponential average over updates, improved by less than 1% over the last 20M
  environment steps;
- **budget:** 500M environment steps.

## Metrics

Each update reports the BC KL, the labels' and the student's entropy, overall and per
pool (`bc/kl`{.plain}, `bc/kl_by_pool/<pool>`{.plain} and so on); the student's agreement
with the labels' likeliest action per factor (`bc/agreement/<factor>`{.plain}); the
explained variance per level; the auxiliary losses; and the live score per opponent kind
(`bc/score/<kind>`{.plain}).

## The hand-off

`bnb train --objective bc` saves the clone, when it stops, as the run's policy file
(`bnb.runs.session.save_clone`, through `bnb.runs.checkpoint.save_policy`): the weights, the
return scale's state and the step counts. `bnb train --init-from` starts reinforcement
learning from it (`bnb.runs.checkpoint.warm_start`): the weights and return scale, a fresh
optimizer, fresh step counts and no cloning term ([Runs and checkpoints](runs.md)). In
Python, `bnb.runtime.cloning.warm_start` does the same in memory, copying into a trainer in
place so the rollout's captured graphs keep reading the weights.

## Measurements

Measured on 2026-11-02 on the development laptop (RTX 4070 Laptop GPU):

- **The teacher's natural entropy:** 0.396 nats per living ship-step, over 1.27M ship-steps
  of the scripted agent's own games of the 5v5 training mix (256 environments, four
  segments of 128 ticks). Its distribution puts weight on 2.4 joint actions on average,
  so the temperature alone reaches at most 0.75 nats; a floor of 0.1 gives 1.03 to 1.34.
- **The toy test** (`tests/sections/test_bc_toy.py`, at the default floor 0.01): the
  relational model cloned at 2v2 team deathmatch, 256 environments, stopped by the score
  rule after 65.6M environment steps (13 minutes), with a window score of 0.432 and a
  bound of 0.401. Over 512 fresh games it scores 0.418 against its teacher (the toy
  test's bar is 0.356), and over 4096 games 0.425; against the chaser it scores 0.994,
  the teacher 1.000.
- **Throughput:** Yemong on the 5v5 training mix at 1024 environments, 1923 environment
  steps per second end to end, against 2596 for PPO with its lineup, measured while other
  processes shared the machine (provisional). The teacher's labels take 0.7 ms per
  micro-batch of 16 views x 128 ticks.
