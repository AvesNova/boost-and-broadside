---
covers: [src/bnb/league/crossover.py]
---

# The crossover

**The question:** how many of Imugi's ships does the trained policy beat per one of its
own, and at what point in training does it first win while outnumbered?

{{pending: crossover_headline | The headline sentence, chosen after the results; the same text as the README's.}}

{{pending: release_run_crossover | The release run's break-even, elimination only, in place of the dry runs' charts below.}}

The fan and the break-even ratio over training below are dry run 1's (`dry-v24-s1`{.plain},
game v24, against its own Imugi), every lineup with 20 ships on its larger side. The two
dry runs' champions at 5 ships against the current Imugi are in
[the last section](#both-champions-at-five-ships).

## Win rate against training steps

{{figure: docs/results/win_rate_fan.png | Win rate against Imugi over training steps, one line per ratio of Imugi's ships to ours (1, 1.2, 1.6 and 2), one panel per mode}}

The table writes the ratios as lineups, 5 against 5, 6, 8 and 10. A line above one half
at 5 against 8 means the policy wins more often than not while outnumbered 8 to 5. How the
games are allocated and fitted is on [How it is measured](evaluation.md#the-crossover).

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

## Both champions at five ships

Each dry run's champion with 5 of its ships against 1 to 20 of Imugi's (game v26), in
elimination, fitted on those lineups alone:

{{figure: docs/results/champion_break_even.png | The score against Imugi at each lineup of 5 of ours against N of Imugi's, with each champion's logistic fit and break-even ratio}}

| Champion | Break-even ratio | Standard error | Games |
|---|---|---|---|
| Dry run 3 | {{break_even_dry_v26_s1_elimination_ratio:.2f}} | {{break_even_dry_v26_s1_elimination_ratio_stderr:.2f}} | {{break_even_dry_v26_s1_elimination_games:,.0f}} |
| Dry run 1 | {{break_even_dry_v24_s1_elimination_ratio:.2f}} | {{break_even_dry_v24_s1_elimination_ratio_stderr:.2f}} | {{break_even_dry_v24_s1_elimination_games:,.0f}} |

## Why this measurement

A rating against Imugi saturates: once the policy wins nearly every even game, more wins
say little about how much stronger it is. Taking ships away from the policy keeps the games
close, so every game informs the estimate, and the ratio of ships is a unit that needs no
rating scale to read. How the games are allocated and fitted, and the caveats, are on
[How it is measured](evaluation.md#the-crossover).
