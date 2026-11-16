---
covers: [src/bnb/league/crossover.py]
---

# The crossover

**The question:** how many of Imugi's ships does the trained policy beat per one of its
own, and at what point in training does it first win while outnumbered?

{{pending: crossover_headline | The headline sentence, chosen after the results; the same text as the README's.}}

## Win rate against training steps

{{figure: docs/results/win_rate_fan.png | Win rate against Imugi over training steps, one line per ratio of Imugi's ships to ours (1, 1.2, 1.6 and 2), one panel per mode}}

Each line is one ratio of ships, Imugi's 1, 1.2, 1.6 or 2 per one of the policy's
(written 5 against 5, 6, 8 and 10 below). The games are played with 20 ships on the larger
side ([How it is measured](evaluation.md#the-crossover)), and at every training snapshot
the crossover's logistic fit is read at each ratio; the band is the fit's 95% interval, and
the open points are the games played at that ratio. The dashed line marks the hand-off
from behaviour cloning to reinforcement learning. A line above one half at 5 against 8
means the policy wins more often than not while outnumbered 8 to 5.

The last snapshot, by mode and lineup (win rate, with its 95% interval):

| Lineup | Frontline | Deathmatch | Elimination |
|---|---|---|---|
| 5 against 5 | {{fan_frontline_5v5_win_rate:.0%}} ({{fan_frontline_5v5_low:.0%}}–{{fan_frontline_5v5_high:.0%}}) | {{fan_deathmatch_5v5_win_rate:.0%}} ({{fan_deathmatch_5v5_low:.0%}}–{{fan_deathmatch_5v5_high:.0%}}) | {{fan_elimination_5v5_win_rate:.0%}} ({{fan_elimination_5v5_low:.0%}}–{{fan_elimination_5v5_high:.0%}}) |
| 5 against 6 | {{fan_frontline_5v6_win_rate:.0%}} ({{fan_frontline_5v6_low:.0%}}–{{fan_frontline_5v6_high:.0%}}) | {{fan_deathmatch_5v6_win_rate:.0%}} ({{fan_deathmatch_5v6_low:.0%}}–{{fan_deathmatch_5v6_high:.0%}}) | {{fan_elimination_5v6_win_rate:.0%}} ({{fan_elimination_5v6_low:.0%}}–{{fan_elimination_5v6_high:.0%}}) |
| 5 against 8 | {{fan_frontline_5v8_win_rate:.0%}} ({{fan_frontline_5v8_low:.0%}}–{{fan_frontline_5v8_high:.0%}}) | {{fan_deathmatch_5v8_win_rate:.0%}} ({{fan_deathmatch_5v8_low:.0%}}–{{fan_deathmatch_5v8_high:.0%}}) | {{fan_elimination_5v8_win_rate:.0%}} ({{fan_elimination_5v8_low:.0%}}–{{fan_elimination_5v8_high:.0%}}) |
| 5 against 10 | {{fan_frontline_5v10_win_rate:.0%}} ({{fan_frontline_5v10_low:.0%}}–{{fan_frontline_5v10_high:.0%}}) | {{fan_deathmatch_5v10_win_rate:.0%}} ({{fan_deathmatch_5v10_low:.0%}}–{{fan_deathmatch_5v10_high:.0%}}) | {{fan_elimination_5v10_win_rate:.0%}} ({{fan_elimination_5v10_low:.0%}}–{{fan_elimination_5v10_high:.0%}}) |

## The break-even ratio against steps

{{figure: docs/results/break_even_ratio.png | Imugi's ships per one of ours at even odds, against environment steps, one line per mode, on a log axis}}

The break-even ratio is the number of Imugi's ships per one of the policy's at which each
side wins half the games. Above 1, the policy holds more than its own number; below 1, it
needs more ships than Imugi. The axis is logarithmic, so holding twice its number and
needing twice Imugi's are the same distance from 1.

| Mode | Break-even ratio | Standard error | At steps |
|---|---|---|---|
| Frontline | {{crossover_ratio_frontline:.2f}} | {{crossover_ratio_frontline_stderr:.2f}} | {{crossover_ratio_frontline_steps:,.0f}} |
| Deathmatch | {{crossover_ratio_deathmatch:.2f}} | {{crossover_ratio_deathmatch_stderr:.2f}} | {{crossover_ratio_deathmatch_steps:,.0f}} |
| Elimination | {{crossover_ratio_elimination:.2f}} | {{crossover_ratio_elimination_stderr:.2f}} | {{crossover_ratio_elimination_steps:,.0f}} |

{{pending: crossover_reading | What the curves show: when each mode passes a ratio of 1, whether behaviour cloning alone reaches it, and how the modes differ.}}

## Why this measurement

A rating against Imugi saturates: once the policy wins nearly every even game, more wins
say little about how much stronger it is. Taking ships away from the policy keeps the games
close, so every game informs the estimate, and the ratio of ships is a unit that needs no
rating scale to read. How the games are allocated and fitted, and the caveats, are on
[How it is measured](evaluation.md#the-crossover).
