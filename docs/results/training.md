---
covers: [src/bnb/runs/chain.py, src/bnb/league/rating_pool.py]
---

# Training curves

**The question:** how does the policy's strength rise through behaviour cloning and then
reinforcement learning, mode by mode, and is the run healthy while it does?

## The run

A run is a chain of two stages on one GPU, as one command (`bnb train --objective chain`):

1. **Behaviour cloning** for 100M environment steps: the student flies, and Imugi labels the
   states it visits ([Behaviour cloning](../explanation/behaviour-cloning.md)). Every head
   is trained, the critic and the auxiliary heads included.
2. **Reinforcement learning** for 100M more: PPO from the clone's exponential average, with
   a fresh optimizer, against itself and a league of its snapshots, Imugi and the cheese
   agents ([The league](../explanation/league.md)).

Each batch mixes the three real-scale modes, 6/16 Frontline, 5/16 deathmatch and 5/16
elimination, at 5 ships a side.

## Elo over training

{{figure: docs/results/elo_over_training.png | Elo against environment steps, one line per mode, Imugi at 1000, the hand-off from behaviour cloning to reinforcement learning marked}}

Each point is one snapshot of the run, rated against Imugi and the run's other snapshots in
the shared rating pool ([How it is measured](evaluation.md#elo)). The snapshots are taken
every 5M steps to 50M, then at an interval that doubles each time the steps double.

| Mode | After behaviour cloning | After reinforcement learning |
|---|---|---|
| Deathmatch | {{elo_deathmatch_bc_final:.0f}} | {{elo_deathmatch_final:.0f}} ± {{elo_deathmatch_final_stderr:.0f}} |
| Elimination | {{elo_elimination_bc_final:.0f}} | {{elo_elimination_final:.0f}} ± {{elo_elimination_final_stderr:.0f}} |
| Frontline | {{elo_frontline_bc_final:.0f}} | {{elo_frontline_final:.0f}} ± {{elo_frontline_final_stderr:.0f}} |

{{pending: training_reading | What the curves show: how close behaviour cloning comes to Imugi's 1000, how fast reinforcement learning climbs past it, and whether any mode is still climbing at 200M.}}

## Run health

{{figure: docs/results/run_health.png | Small multiples against steps: clip fraction, approximate KL, explained variance and value loss per reward part, entropy per action factor, behaviour cloning's KL to its teacher, and each auxiliary head's loss}}

These are the learner's own records ([the learner](../explanation/learner.md)), drawn as a
faint raw line and a centred moving mean. They show whether an update did what it should:
a clip fraction and a KL that stay bounded, a critic that explains its returns, and an
entropy that falls without collapsing.

{{pending: health_reading | One paragraph: anything the health curves show that bears on the results, such as the entropy at the hand-off or a head whose loss rises.}}
