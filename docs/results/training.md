---
covers: [src/bnb/runs/chain.py, src/bnb/league/rating_pool.py]
---

# Training curves

**The question:** how does the policy's strength rise through behaviour cloning and then
reinforcement learning, mode by mode, and is the run healthy while it does?

## The run

The curves on this page are the two dry runs': dry run 3 (`dry-v26-s1`{.plain}, game
version 26) and, where the two are compared, dry run 1 (`dry-v24-s1`{.plain}, game version
24, the earlier Imugi). Version 26 kept version 24's ships, gave Imugi committed turns and fixed
deathmatch's tickets, so the two runs' teachers differ.

{{pending: release_run_training | The release run's curves and table: elimination only, in place of the dry runs' three modes.}}

A run is a chain of two stages on one GPU, as one command (`bnb train --objective chain`):

1. **Behaviour cloning** for 100M environment steps: the student flies, and Imugi labels the
   states it visits ([Behaviour cloning](../explanation/behaviour-cloning.md)). Every head
   is trained, the critic and the auxiliary heads included.
2. **Reinforcement learning** for 100M more (dry run 3 was extended to
   {{chain_steps_rl:,.0f}} steps, then stopped): PPO from the clone's exponential average, with
   a fresh optimizer, against itself and a league of its snapshots, Imugi and the cheese
   agents ([The league](../explanation/league.md)).

In the dry runs each batch mixes the three real-scale modes, 6/16 Frontline, 5/16
deathmatch and 5/16 elimination, at 5 ships a side.

## Elo over training

{{figure: docs/results/elo_over_training.png | Elo against environment steps, one line per mode, Imugi at 1000, the hand-off from behaviour cloning to reinforcement learning marked}}

The snapshots are taken at 1, 2, 3, 4, 6, 8, 10, 12 and 16M steps into each stage, then
every 5M; the rating is described on [How it is measured](evaluation.md#elo). Both runs'
snapshots, both champions and both Imugis are rated in one field of
{{elo_report_games:,.0f}} games so far, so the two runs share one scale. Elimination, the
release mode, is drawn first; the other modes follow once their games are played.

Each run's last rated snapshot of each stage, and its champion (Imugi of game v26 at 1000):

| Mode | Run | After behaviour cloning | After reinforcement learning | Champion |
|---|---|---|---|---|
| Elimination | dry run 3 | {{elo_dry_v26_s1_elimination_bc_final}} | {{elo_dry_v26_s1_elimination_final}} | {{elo_dry_v26_s1_elimination_champion}} |
| Elimination | dry run 1 | {{elo_dry_v24_s1_elimination_bc_final}} | {{elo_dry_v24_s1_elimination_final}} | {{elo_dry_v24_s1_elimination_champion}} |
| Frontline | dry run 3 | {{elo_dry_v26_s1_frontline_bc_final}} | {{elo_dry_v26_s1_frontline_final}} | {{elo_dry_v26_s1_frontline_champion}} |
| Frontline | dry run 1 | {{elo_dry_v24_s1_frontline_bc_final}} | {{elo_dry_v24_s1_frontline_final}} | {{elo_dry_v24_s1_frontline_champion}} |
| Deathmatch | dry run 3 | {{elo_dry_v26_s1_deathmatch_bc_final}} | {{elo_dry_v26_s1_deathmatch_final}} | {{elo_dry_v26_s1_deathmatch_champion}} |
| Deathmatch | dry run 1 | {{elo_dry_v24_s1_deathmatch_bc_final}} | {{elo_dry_v24_s1_deathmatch_final}} | {{elo_dry_v24_s1_deathmatch_champion}} |

Imugi of game v24 rates {{elo_imugi_v24_elimination}} in elimination on the same scale.
A "(a bound)" is a snapshot that won or lost nearly every game, so only a bound is known.

### Cloning against the earlier teacher

Both runs' final cloning checkpoints (the exponential average at 100.7M steps) against
the same opponent, Imugi of game v24, 5 against 5, 256 games per mode; the score is wins
plus half the draws, with its standard error:

| Mode | Dry run 3 (taught by Imugi v26) | Dry run 1 (taught by Imugi v24) |
|---|---|---|
| Frontline | {{matchup_dry_v26_s1_bc_frontline_vs_imugi_v24_score:.2f}} ± {{matchup_dry_v26_s1_bc_frontline_vs_imugi_v24_stderr:.2f}} | {{matchup_dry_v24_s1_bc_frontline_vs_imugi_v24_score:.2f}} ± {{matchup_dry_v24_s1_bc_frontline_vs_imugi_v24_stderr:.2f}} |
| Deathmatch | {{matchup_dry_v26_s1_bc_deathmatch_vs_imugi_v24_score:.2f}} ± {{matchup_dry_v26_s1_bc_deathmatch_vs_imugi_v24_stderr:.2f}} | {{matchup_dry_v24_s1_bc_deathmatch_vs_imugi_v24_score:.2f}} ± {{matchup_dry_v24_s1_bc_deathmatch_vs_imugi_v24_stderr:.2f}} |
| Elimination | {{matchup_dry_v26_s1_bc_elimination_vs_imugi_v24_score:.2f}} ± {{matchup_dry_v26_s1_bc_elimination_vs_imugi_v24_stderr:.2f}} | {{matchup_dry_v24_s1_bc_elimination_vs_imugi_v24_score:.2f}} ± {{matchup_dry_v24_s1_bc_elimination_vs_imugi_v24_stderr:.2f}} |

{{pending: training_reading | What the curves show: how close behaviour cloning comes to Imugi's 1000, how fast reinforcement learning climbs past it, and whether any mode is still climbing at 200M.}}

The training also rates itself as it goes (dry run 3 below): its lanes play the live weights and their
exponential average against Imugi and its snapshots, a few games each update. These live
ratings are noisier than the pool's, and the deathmatch lanes play fewest games:

{{figure: docs/results/training_ratings.png | The training lanes' Elo against Imugi during reinforcement learning, one panel per mode, the live weights solid and their exponential average dashed with its standard error}}

## Which weights: the averaging sweep

The run keeps every update's weights, so after training any average of them can be built
and rated. The sweep rates each candidate average against the training's own exponential
average (the incumbent) in all three modes at 5 against 5, and keeps the incumbent unless a
candidate's mean Elo difference over the modes exceeds two standard errors.

{{figure: docs/results/averaging_sweep.png | Each weight average's Elo difference from the training's exponential average, the mean over the modes with a 95% interval and a tick per mode, the champion marked}}

{{pending: averaging_reading | Which average won and by how much, or that the incumbent stayed; whether the modes agree.}}

## The champion on every condition

The suite below is dry run 1's champion against Imugi of game v24, its own game. Both
champions in elimination at 5 against 5, from the unified field (score with its standard
error):

| Champion | Against Imugi v26 | Against Imugi v24 |
|---|---|---|
| Dry run 3 | {{h2h_dry_v26_s1_elimination_5v5_vs_scripted_score:.0%}} ± {{h2h_dry_v26_s1_elimination_5v5_vs_scripted_stderr:.0%}} | {{h2h_dry_v26_s1_elimination_5v5_vs_imugi_v24_score:.0%}} ± {{h2h_dry_v26_s1_elimination_5v5_vs_imugi_v24_stderr:.0%}} |
| Dry run 1 | {{h2h_dry_v24_s1_elimination_5v5_vs_scripted_score:.0%}} ± {{h2h_dry_v24_s1_elimination_5v5_vs_scripted_stderr:.0%}} | {{h2h_dry_v24_s1_elimination_5v5_vs_imugi_v24_score:.0%}} ± {{h2h_dry_v24_s1_elimination_5v5_vs_imugi_v24_stderr:.0%}} |

{{figure: docs/results/suite_conditions.png | The champion's score against Imugi on each head-to-head condition of the decision suite, one panel per mode, one point per team size, with 95% intervals}}

The decision suite's head-to-head conditions, each played to {{suite_games_per_condition:.0f}}
games: the champion and Imugi on the same lineup of ship types, the champion's score being
its wins and half its draws, with its 95% interval:

| Team size | Frontline | Deathmatch | Elimination |
|---|---|---|---|
| 1 against 1 | not in the suite | {{suite_deathmatch_1v1_score:.0%}} ({{suite_deathmatch_1v1_low:.0%}}–{{suite_deathmatch_1v1_high:.0%}}) | {{suite_elimination_1v1_score:.0%}} ({{suite_elimination_1v1_low:.0%}}–{{suite_elimination_1v1_high:.0%}}) |
| 2 against 2 | not in the suite | {{suite_deathmatch_2v2_score:.0%}} ({{suite_deathmatch_2v2_low:.0%}}–{{suite_deathmatch_2v2_high:.0%}}) | {{suite_elimination_2v2_score:.0%}} ({{suite_elimination_2v2_low:.0%}}–{{suite_elimination_2v2_high:.0%}}) |
| 5 against 5 | {{suite_frontline_5v5_score:.0%}} ({{suite_frontline_5v5_low:.0%}}–{{suite_frontline_5v5_high:.0%}}) | {{suite_deathmatch_5v5_score:.0%}} ({{suite_deathmatch_5v5_low:.0%}}–{{suite_deathmatch_5v5_high:.0%}}) | {{suite_elimination_5v5_score:.0%}} ({{suite_elimination_5v5_low:.0%}}–{{suite_elimination_5v5_high:.0%}}) |
| 10 against 10 | {{suite_frontline_10v10_score:.0%}} ({{suite_frontline_10v10_low:.0%}}–{{suite_frontline_10v10_high:.0%}}) | {{suite_deathmatch_10v10_score:.0%}} ({{suite_deathmatch_10v10_low:.0%}}–{{suite_deathmatch_10v10_high:.0%}}) | {{suite_elimination_10v10_score:.0%}} ({{suite_elimination_10v10_low:.0%}}–{{suite_elimination_10v10_high:.0%}}) |

{{pending: suite_reading | What the conditions show: where the champion is far ahead of Imugi, and the small elimination games it loses.}}

## Run health

Dry run 3's learner records:

{{figure: docs/results/run_health.png | Small multiples against steps: clip fraction, approximate KL, explained variance and value loss per reward part, entropy per action factor, behaviour cloning's KL to its teacher, and each auxiliary head's loss}}

These are the learner's own records ([the learner](../explanation/learner.md)), drawn as a
faint raw line and a centred moving mean. They show whether an update did what it should:
a clip fraction and a KL that stay bounded, a critic that explains its returns, and an
entropy that falls without collapsing.

{{pending: health_reading | One paragraph: anything the health curves show that bears on the results, such as the entropy at the hand-off or a head whose loss rises.}}
