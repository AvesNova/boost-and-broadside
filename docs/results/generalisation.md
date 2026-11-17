---
covers: [src/bnb/league/crossover.py, src/bnb/models/yemong/model.py]
---

# Team sizes

**The question:** a policy trained at one team size, how does it play at the others, from
1 against 1 to 10 against 10?

Yemong reads the fleet as a set of tokens, one per ship, field and zone, so no weight
depends on the number of ships ([The model](../explanation/yemong.md)). One set of weights
plays every team size without retraining; what it learned at the training size may or may
not carry over.

The measured policy trained at 5 against 5 alone, in all three modes.

## Win rate by team size

{{figure: docs/results/team_size_generalization.png | Win rate against Imugi at each team size, one line per ratio of Imugi's ships to ours (1, 1.5 and 2), one panel per mode, the training size marked}}

Even numbers, the policy's win rate against Imugi:

| Team size | Frontline | Deathmatch | Elimination |
|---|---|---|---|
| 1 against 1 | {{team_size_frontline_ratio_1_size_1_win_rate:.0%}} | {{team_size_deathmatch_ratio_1_size_1_win_rate:.0%}} | {{team_size_elimination_ratio_1_size_1_win_rate:.0%}} |
| 2 against 2 | {{team_size_frontline_ratio_1_size_2_win_rate:.0%}} | {{team_size_deathmatch_ratio_1_size_2_win_rate:.0%}} | {{team_size_elimination_ratio_1_size_2_win_rate:.0%}} |
| 5 against 5 | {{team_size_frontline_ratio_1_size_5_win_rate:.0%}} | {{team_size_deathmatch_ratio_1_size_5_win_rate:.0%}} | {{team_size_elimination_ratio_1_size_5_win_rate:.0%}} |
| 10 against 10 | {{team_size_frontline_ratio_1_size_10_win_rate:.0%}} | {{team_size_deathmatch_ratio_1_size_10_win_rate:.0%}} | {{team_size_elimination_ratio_1_size_10_win_rate:.0%}} |

{{pending: generalisation_reading | What the chart shows: whether the win rate holds away from the training size, where it falls, and whether small games (1 against 1, 2 against 2) differ in kind, for example by running the clock out.}}

## Why it matters

Small games and large games ask different things: at 1 against 1 a fight is a duel of
flying; at 10 against 10 it is more a matter of where the fleet concentrates. A policy that
plays well at both from one training size has learned something about the game rather than
about one lineup.
