---
covers: []
---

# Results

The results of version 2: one network trained to command a fleet in three modes, measured
against Imugi, the scripted agent, and against its own earlier configurations.

{{pending: results_overview | Two or three sentences: the champion's run and steps, its break-even ratio per mode, and its Elo per mode, written after the results.}}

{{figure: docs/results/win_rate_fan.png | Win rate against Imugi over training steps, one line per lineup from 5 against 5 to 5 against 10, one panel per mode}}

## The pages

| Page | The question |
|---|---|
| [The crossover](crossover.md) | How many of Imugi's ships does the policy beat per one of its own, and when in training does it pass even odds? |
| [Training curves](training.md) | How does the rating rise through behaviour cloning and reinforcement learning, mode by mode? |
| [The arms](arms.md) | Which changes to the model and the training made it stronger, decided one at a time? |
| [Team sizes](generalisation.md) | Trained at one team size, how does the policy play at the others? |
| [Compute](compute.md) | What does a training run cost, on a laptop GPU and on an RTX 5090? |
| [Behaviour and failures](behaviour.md) | What does the policy do in a game, and where does it lose? |
| [How it is measured](evaluation.md) | The ratings, the crossover fit, the decision test, and the limits of each. |

## The policy measured

Every result names the policy it measured: the run, its environment steps, and whether it
is the run's live weights or an average of them. Unless a page says otherwise, the measured
policy is the champion: the average of the run's weights that the
[averaging sweep](training.md#which-weights-the-averaging-sweep) keeps, at the end of
reinforcement learning:

| | |
|---|---|
| Run | {{champion_run}} |
| Environment steps of reinforcement learning | {{champion_steps:,.0f}} (after {{chain_steps_bc:,.0f}} of behaviour cloning) |
| Weights | {{champion_weights}} |
| Game version | {{champion_game_version}} |

The numbers on these pages come from one values file, `docs/results/values.json`{.plain},
which the chart commands write ([below](#where-the-numbers-come-from)).

## Where the numbers come from

Each figure is drawn by `bnb figures` from the records the runs and the measurements
write, and carries its provenance (the commit, the command line and the sources it read)
in its PNG text chunks. Each figure's numbers are merged into the values file, and the
pages quote them from there, so a number that appears on several pages is changed in one
place. How the placeholders work is in [Writing docs](../contributing/writing-docs.md#results-placeholders).
