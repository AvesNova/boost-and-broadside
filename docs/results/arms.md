---
covers: [src/bnb/league/evaluation.py, src/bnb/league/sequential.py]
---

# The arms

**The question:** which changes to the model and to the training make the policy stronger,
each measured on its own?

## The program

An arm is one change, trained as a full run and compared with the incumbent: the
champion's configuration when the arm starts. The pair plays the decision suite, and a
sequential test decides between them ([How it is measured](evaluation.md#arm-decisions)).
The winner becomes the incumbent for the next arm, so a change that lost never carries into
a later arm.

- **Equal budgets.** Every arm trains 100M steps of behaviour cloning and 100M of
  reinforcement learning, unless its question is about steps (training from scratch, a
  wider network at equal compute).
- **Shared clones.** An arm that changes only reinforcement learning forks from the
  incumbent's behaviour-cloning run, so the two differ in that change alone.
- **Final checkpoints.** Each decision rates the two runs' final weights, not the best
  snapshot of either.
- **Seed noise.** Two pairs of runs that differ only in their seed set the indifference
  zone: a difference smaller than the seed-to-seed spread is a tie.

## The decisions

{{figure: docs/results/arms_forest.png | One row per arm: its score against the incumbent with a 95% interval, the indifference zone shaded, the decision marked, and a row for seed-to-seed noise}}

The score is the arm's expected score against the incumbent less one half, averaged over
the suite's conditions: 0 is even, positive favours the arm.

{{pending: arms_table | The table of arms in the order they ran: the change, the class (behaviour cloning and reinforcement learning, or a reinforcement-learning fork), the score with its interval, the decision, and the steps and GPU hours. From the arm list and each decision's record.}}

{{pending: arms_reading | What the arms showed: which changes were kept, which made no difference, and anything that surprised.}}

## Why sequential tests

A full run costs hours of GPU time, so each comparison should stop as soon as the evidence
decides it and not before. A group-sequential test fixes its looks and thresholds in
advance, so stopping early does not inflate the rate of false decisions, and a tie is a
decision, not a failure to finish.
