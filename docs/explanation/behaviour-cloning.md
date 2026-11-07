---
covers: [src/bnb/learn/bc.py, src/bnb/runtime/cloning.py]
reviewed: 2026-11-06 12:00
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
loss. The scripted agent reads only the observation, so the rollout stores nothing for it
beyond the observation parts it reads (`Rollout` records them with `extra_reads`). It has a
memory, its carry ([the scripted agent's
memory](scripted-agent.md#memory-soft-hysteresis-and-dead-reckoning)): before each update
the learner runs the teacher over the stored segments in time order, its carry going on
from the last update's (`bnb.learn.bc.teacher_carries`), and keeps its carry at every tick;
a label is the teacher's distribution given its carry there. So the student is asked to
remember what the teacher remembers. With its memory off the labels are the distribution the
memoryless agent plays on the same observation (`tests/sections/test_bc.py`).

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
rating lanes (`bnb.runtime.cloning.LANES`) play three players against the raw scripted
agent in every mode, in the same rollout ([The training
loop](training-loop.md#the-rating-lanes)): the live clone, the uniform average and the
exponential average of its weights. Each mode's lanes go 1 : 2 : 2 to them (the
matchups' weights 0.5, 1 and 1; of the default 1,024 lanes in one mode, 205, 410 and
409), since the score stop reads the averages and the live clone is only logged. The
score stop and the ratings read those games (at the default teacher, the scripted agent
is the teacher). There are no separate evaluation rollouts.

## The averages

The averaged agents are the product of BC, as in reinforcement learning (decisions,
"Behaviour cloning, round three"). BC keeps the league's two averages of the live weights
(`bnb.league.pool.Averages`, the same class and rule [the league](league.md#the-pool)
uses), folded in after every update by BC's hooks (`bnb.league.lanes.LaneHooks`):

- **the uniform average** of the live weights over every update after the run's
  learning-rate warm-up (`LearnerConfig.warmup_steps`, 10M environment steps by default; an
  explicit `LeagueConfig.uniform_from` overrides it: `TrainConfig.uniform_start`); until
  then it is the live weights;
- **the exponential moving average**, decay 0.95 per update
  (`LeagueConfig.exponential_decay`), from the first update.

BC reads these settings from the run's `TrainConfig.league`, the league's own fields,
rather than fields of its own: one warm-up and one decay for both objectives. The
uniform average's start follows the learning rate's warm-up, so a fine-tune with a
shorter warm-up starts averaging when its warm-up ends; `--set league.uniform_from=...`
sets it apart from the warm-up. Both averages go into the training checkpoint (the
league state's tensors, and the uniform average's update count) and come back on a
resume. After each update the hooks copy both into the lanes that play them, so an
update's lane games are those of the averages as the previous update left them.

The last BC run before the averages (run `2026-11-05-mixed-eager-acorn`, 150M steps)
shows why: its live clone scored 0.272 against the scripted agent, a uniform average of
its retained checkpoints from 66M to 151M steps, built offline, 0.350.

## When it stops

`bnb.learn.bc.StopRule`, fed once per update by `bnb.runtime.cloning.update`, names the
first rule to fire (`bnb.learn.bc.StopConfig`, the run's `TrainConfig.stop`). Its state is
part of the training checkpoint, so a resumed run continues the same window:

- **score:** the score against the raw scripted agent in the lanes (wins plus half the
  draws) over the last 1000 finished lane games of the better of the two averages reaches
  parity, 0.5, even if the KL is still falling. Each player keeps its own window
  (`bnb.learn.bc.ScoreWindow`): the live clone's too, which is logged and never read. The
  rule reads, of the averages `StopConfig.reads` names, the one whose full window scores
  higher (`StopRule.read`). Against one opponent a rating is a monotone function of the
  score, so this is the better-rated average over the stop's own window, not the lanes'
  ten-update rating window. An equal score goes to the exponential average, the first in
  `reads`. The uniform average's games count only from the first update that began with
  it past its warm-up: before that it is the live clone, and the rule reads the
  exponential average alone. It is the plain score; its standard error (the sample
  deviation of the per-game scores over the square root of 1000) is reported beside it
  (`bc/stop/window_stderr`{.plain}). A window keeps each update's games, points and
  squared points, drops the oldest updates whole and counts the oldest one it keeps by the
  share of its games the window needs, so trimming favours no result;
- **KL plateau:** the BC KL, `KL(labels ‖ student) = CE − H(labels)` per token, as an
  exponential average over updates, improved by less than 1% over the last 20M
  environment steps;
- **budget:** 50M environment steps, v1's time to parity. A run stopped by its budget ends
  with a checkpoint and its policy files; `bnb train --resume RUN --bc-budget-steps N`
  with a larger budget continues it.

## Metrics

Each update reports the BC KL, the labels' and the student's entropy, overall and per
pool (`bc/kl`{.plain}, `bc/kl_by_pool/<pool>`{.plain} and so on); the student's agreement
with the labels' likeliest action per factor (`bc/agreement/<factor>`{.plain}); the
explained variance per level; the auxiliary losses; the training games' score per
opponent kind (`bc/score/<kind>`{.plain}); the lanes' games, scores and ratings
(`lanes/score/<mode>/<player>_vs_scripted`{.plain}; `rating/live/<mode>`{.plain},
`rating/average/<mode>`{.plain} for the exponential average as in reinforcement learning,
and `rating/uniform_average/<mode>`{.plain}); and the stop's windows: the read average's
score and error (`bc/stop/window_score`{.plain}, `bc/stop/window_stderr`{.plain}), every
full window's (`bc/stop/window_score/<player>`{.plain}) and which average the stop reads
(`bc/stop/reads/<average>`{.plain}, 1 for the read one).

## The hand-off

`bnb train --objective bc`, when it stops, saves four policy files
(`bnb.runs.session.save_policies`, through `bnb.runs.checkpoint.save_policy`), each with
its weights, the return scale's state, the step counts and its own lane rating per mode,
where the RL run's live rating starts:

- `policy/`{.plain}: the run's policy, the average the score stop read last (before any
  window filled, the exponential average); `run.json`{.plain} names it as
  `policy_player`;
- `policy-uniform/`{.plain} and `policy-exponential/`{.plain}: each average;
- `policy-live/`{.plain}: the live clone.

`policy/` duplicates one of the averages' files, so that the hand-off has one path
whichever average won. A resumed run replaces these four at its next end; the three
players of every 10M environment steps and of each end stay under
`policies/step_<steps>/` (`TrainConfig.policy_snapshot_every`,
[Runs and checkpoints](runs.md#every-updates-policy-files)). The return scale is the learner's in all four. `bnb train
--init-from` starts reinforcement learning from any of them (`bnb.runs.checkpoint.warm_start`): the weights and return scale, a fresh
optimizer, fresh step counts and no cloning term ([Runs and checkpoints](runs.md)). In
Python, `bnb.runtime.cloning.warm_start` does the same in memory, copying into a trainer in
place so the rollout's captured graphs keep reading the weights.

## Checkpoint retention

The run's retention of training checkpoints
([Runs and checkpoints](runs.md)) is unchanged: the newest few, then one per doubling of
age. It thins the older checkpoints, so an offline uniform average of them would weigh a
few far-apart weights. That is no longer needed: the learner's uniform average folds in
every update, and every training checkpoint keeps it with its update count. Two retained
checkpoints also give the uniform average over the updates between them, up to rounding:
`(n_b ū_b − n_a ū_a) / (n_b − n_a)`, with `ū` the average and `n` its count
(`averaged` in the checkpoint's league record).

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
