---
covers: []
---

# Behaviour and failures

**The question:** what does the trained policy do in a game, and where does it still lose?

The aggregate numbers say how often the policy wins; the clips say how. Each clip names the
policy, the mode, the lineup and the seed, so it can be replayed with `bnb play`
([Playing and spectating](../guides/playing.md)).

## Game length

{{figure: docs/results/game_length.png | The distribution of game lengths per mode, the medians dotted and the clocks dashed, with the share of games that ran the clock out}}

| Mode | Median game | Games that ran the clock out |
|---|---|---|
| Frontline | {{game_length_frontline_median_seconds:.0f}} s | {{game_length_frontline_clock_out_share:.0%}} |
| Deathmatch | {{game_length_deathmatch_median_seconds:.0f}} s | {{game_length_deathmatch_clock_out_share:.0%}} |
| Elimination | {{game_length_elimination_median_seconds:.0f}} s | {{game_length_elimination_clock_out_share:.0%}} |

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
