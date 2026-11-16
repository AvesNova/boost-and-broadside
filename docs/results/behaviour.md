---
covers: []
---

# Behaviour and failures

**The question:** what does the trained policy do in a game, and where does it still lose?

The aggregate numbers say how often the policy wins; the clips say how. Each clip names the
policy, the mode, the lineup and the seed, so it can be replayed with `bnb play`
([Playing and spectating](../guides/playing.md)).

## Behaviour over training

{{figure: docs/results/behaviour_over_training.png | Against environment steps, one line per mode: kills per ship-minute, game length, the share of deaths to the boundary, the hit rate, the front's moves per minute in Frontline, and the mean speed; the hand-off from behaviour cloning to reinforcement learning dashed}}

These are the training's own games, from the run's metrics: during behaviour cloning the
student's games against its teacher, during reinforcement learning its games against itself
and its league. They show how the play changed, not how strong it is; strength is on
[Training curves](training.md).

## Game length

Over the last five updates of reinforcement learning, in the training's own games:

| Mode | Mean game | Games that ran the clock out |
|---|---|---|
| Frontline | {{game_length_frontline_training_mean_seconds:.0f}} s | {{game_length_frontline_training_clock_out_share:.0%}} |
| Deathmatch | {{game_length_deathmatch_training_mean_seconds:.0f}} s | {{game_length_deathmatch_training_clock_out_share:.0%}} |
| Elimination | {{game_length_elimination_training_mean_seconds:.0f}} s | {{game_length_elimination_training_clock_out_share:.0%}} |

The clock is 600 s in Frontline and deathmatch and 248 s in elimination.

{{pending: game_length_reading | Whether the trained policy ends games faster or slower than Imugi does against itself, and in which modes.}}

## Gallery

{{figure: docs/results/media/outnumbered_win.gif | The champion winning while outnumbered, at the break-even ratio}}

{{pending: gallery_outnumbered | Caption: mode, lineup, run and steps, seed; and what the clip shows the fleet doing.}}

{{figure: docs/results/media/frontline_push.gif | A Frontline game: the policy taking and holding zones under fog of war}}

{{pending: gallery_frontline | Caption, as above.}}

{{figure: docs/results/media/elimination_boundary.gif | An elimination game as the boundary closes}}

{{pending: gallery_elimination | Caption, as above.}}

## Failures

{{pending: failure_gallery | One clip or figure per failure seen in the evaluation games, for example a strategy of a cheese agent the policy has no answer to, or a mode or team size where it loses to Imugi, each with its frequency in the measured games.}}
