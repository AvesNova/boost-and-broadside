---
covers: [src/bnb/learn/bc.py, src/bnb/runtime/cloning.py]
reviewed: 2026-11-02
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
temperature alone; the floor exists for a target entropy the temperature cannot reach. The
transformation applies to the labels only: as an opponent, the scripted agent plays its
natural distribution.

`bnb.learn.bc.calibrate` bisects the temperature to a target mean entropy, in nats per
living ship-step, over the scripted agent's own distributions in its own games of the
training mix (`bnb.runtime.cloning.teacher_distributions`).

## Rollouts

`bnb.runtime.cloning.LINEUP`: half self-play, half against the scripted agent at its
natural distribution, crossed with the training mix in a mixed batch. The live scores per
opponent come from the training games themselves; there are no evaluation rollouts.

## When it stops

`bnb.learn.bc.StopRule`, fed once per update by `bnb.runtime.cloning.clone`, names the
first rule to fire (`bnb.learn.bc.StopConfig`):

- **score:** the score against the scripted agent (wins plus half the draws) over the last
  1000 finished training games reaches 0.45, even if the KL is still falling;
- **KL plateau:** the BC KL, `KL(labels ‖ student) = CE − H(labels)` per token, as an
  exponential average over updates, improved by less than 1% over the last 20M
  environment steps;
- **budget:** 500M environment steps.

## Metrics

Each update reports the BC KL, the labels' and the student's entropy, overall and per
pool (`bc_kl/<pool>`{.plain} and so on); the explained variance per level; the auxiliary
losses; and the live score per opponent kind.

## The hand-off

`bnb.runtime.cloning.warm_start` copies the student's weights and return scale into a
reinforcement-learning trainer, in place, so the rollout's captured graphs keep reading
them; the optimizer stays fresh. Saving the clone to a file waits for the run directory
and its policy files (in progress, [Runs and checkpoints](in-progress/runs.md)).

## Measurements

<!-- measured: bc-results -->
