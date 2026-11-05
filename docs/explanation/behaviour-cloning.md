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

The teacher is the scripted agent at its own temperature T, its labels optionally mixed
with uniform by a floor φ (`bnb.learn.bc.TeacherConfig`): `(1 − φ) q_T + φ / 42`, with q_T
the agent's distribution at T ([the scripted agent's
softness](scripted-agent.md#softness-logistics-the-floor-and-the-temperature): every
logistic's scale times T). The default is T = 1 and φ = 0, the scripted agent itself: it
gives no living ship's action probability 0 (its own floors on its decisions) except a
shot it cannot or need not fire (no target, a gun cooling), where P(shoot) is 0; the KL
needs no floor (a zero label contributes nothing to the cross-entropy).
Raising T sets a higher entropy for the clone to start reinforcement learning at, without
an entropy bonus. In BC's own games the scripted opponent plays the labels' distribution
(`bnb.learn.bc.TeacherOpponent`: the agent at T, then a uniform joint action with
probability φ), so the live score is against the teacher the clone copies; at the default
it is the scripted agent, so BC's score stop (parity against the scripted agent in the
lanes) is a target the teacher itself meets.

`bnb.learn.bc.calibrate` bisects the agent's temperature to a target mean entropy, in nats
per living ship-step, over the scripted agent's distributions on the observations of its
own games of the training mix (`bnb.runtime.cloning.teacher_distributions`, relabelled at
each temperature tried).

## Rollouts

`bnb.runtime.cloning.LINEUP`: half self-play, half against the scripted agent playing the
labels' distribution, crossed with the training mix in a mixed batch. Beside them the
rating lanes (`bnb.runtime.cloning.LANES`) play the live policy against the raw scripted
agent in every mode, in the same rollout ([The training
loop](training-loop.md#the-rating-lanes)): the score stop and the live rating read those
games (at the default teacher, the same player). There are no separate evaluation rollouts.

## When it stops

`bnb.learn.bc.StopRule`, fed once per update by `bnb.runtime.cloning.update`, names the
first rule to fire (`bnb.learn.bc.StopConfig`, the run's `TrainConfig.stop`). Its state is
part of the training checkpoint, so a resumed run continues the same window:

- **score:** the live policy's score in its lanes against the raw scripted agent (wins
  plus half the draws) over the last 1000 finished lane games reaches parity, 0.5, even if
  the KL is still falling. It is the plain score; its standard error (the sample deviation
  of the per-game scores over the square root of 1000) is reported beside it
  (`bc/stop/window_stderr`{.plain}). The window keeps each update's games, points and
  squared points, drops the oldest updates whole and counts the oldest one it keeps by the
  share of its games the window needs, so trimming favours no result;
- **KL plateau:** the BC KL, `KL(labels ‖ student) = CE − H(labels)` per token, as an
  exponential average over updates, improved by less than 1% over the last 20M
  environment steps;
- **budget:** 50M environment steps, v1's time to parity. A run stopped by its budget ends
  with a checkpoint and its clone; `bnb train --resume RUN --bc-budget-steps N` with a
  larger budget continues it.

## Metrics

Each update reports the BC KL, the labels' and the student's entropy, overall and per
pool (`bc/kl`{.plain}, `bc/kl_by_pool/<pool>`{.plain} and so on); the student's agreement
with the labels' likeliest action per factor (`bc/agreement/<factor>`{.plain}); the
explained variance per level; the auxiliary losses; the training games' score per
opponent kind (`bc/score/<kind>`{.plain}); and the lanes' games, scores and live rating
(`lanes/score/<mode>/live_vs_scripted`{.plain}, `rating/live/<mode>`{.plain}).

## The hand-off

`bnb train --objective bc` saves the clone, when it stops, as the run's policy file
(`bnb.runs.session.save_clone`, through `bnb.runs.checkpoint.save_policy`): the weights, the
return scale's state, the step counts and its lane rating per mode, where the RL run's live
rating starts. `bnb train --init-from` starts reinforcement
learning from it (`bnb.runs.checkpoint.warm_start`): the weights and return scale, a fresh
optimizer, fresh step counts and no cloning term ([Runs and checkpoints](runs.md)). In
Python, `bnb.runtime.cloning.warm_start` does the same in memory, copying into a trainer in
place so the rollout's captured graphs keep reading the weights.

## Measurements

Measured on 2026-11-02 on the development laptop (RTX 4070 Laptop GPU), with the earlier
scripted agent (linear ramps ending at exactly 0 and 1, no floor of its own) and its
teacher at floor 0.01; the current agent's entropy is in
[the scripted agent's measurements](scripted-agent.md#measurements):

- **The teacher's natural entropy:** 0.396 nats per living ship-step, over 1.27M ship-steps
  of the scripted agent's own games of the 5v5 training mix (256 environments, four
  segments of 128 ticks). Its distribution puts weight on 2.4 joint actions on average,
  so the temperature alone reaches at most 0.75 nats; a floor of 0.1 gives 1.03 to 1.34.
- **The toy test** (`tests/sections/test_bc_toy.py`, at the default floor 0.01; measured
  before the stop read the lanes, with the then score rule: 0.40 less two standard errors,
  against the teacher): the relational model cloned at 2v2 team deathmatch, 256
  environments, stopped by the score rule after 65.6M environment steps (13 minutes), with
  a window score of 0.432 and a bound of 0.401. Over 512 fresh games it scores 0.418 against its teacher (the toy
  test's bar is 0.356), and over 4096 games 0.425; against the chaser it scores 0.994,
  the teacher 1.000.
- **Throughput:** Yemong on the 5v5 training mix at 1024 environments, 1923 environment
  steps per second end to end, against 2596 for PPO with its lineup, measured while other
  processes shared the machine (provisional). The teacher's labels take 0.7 ms per
  micro-batch of 16 views x 128 ticks.
