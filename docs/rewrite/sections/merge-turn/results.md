# Merge turn: how long the scripted pilot flies straight after a pass

Script: `docs/rewrite/sections/merge-turn/merge_turn.py` (run per setup with `--only`, then `--summarise`).
Matches: `game.reset` with team sizes (S, S), seed 0, `ScriptedConfig()` on both teams, fog on,
256 envs for 1v1 and 64 for 4v4, 1800 ticks (60 s). Pilot code and game code are unchanged.

Pass event: a local minimum of the pair distance between opposite-team aircraft, with each nose
within 60 degrees of the other aircraft one second earlier. Events with a closest approach under
10 m (collisions) are included; they are 164, 98 and 61 of the events.

## Time from closest approach to a heading change

Median / 75th / 90th percentile, seconds. Aircraft that never reach the threshold are counted at
the 10 s cap, so the upper percentiles are lower bounds where the cap is hit.

| setup | pass events | aircraft | removed / match ended before 30 deg | time to 30 deg | time to 90 deg |
|---|---|---|---|---|---|
| 1v1, no fields | 689 | 1378 | 45 / 43 | 0.80 / 1.03 / 1.14 | 2.40 / 2.63 / 2.80 |
| 1v1, fields | 487 | 974 | 26 / 34 | 0.77 / 1.07 / 1.27 | 2.30 / 2.67 / 3.33 |
| 4v4, fields | 441 | 882 | 100 / 9 | 0.73 / 1.10 / 1.99 | 2.27 / 3.48 / 6.26 |

Slow aircraft (time to 30 deg above 1.5 s, or not reached in 10 s):
- 1v1, no fields: 0 of 1290 (none).
- 1v1, fields: 42 of 914 (4.6%).
- 4v4, fields: 112 of 773 (14.5%).

So the slow case is a fields-and-fog effect in these runs: with no fields it does not occur in 1v1.

## What the slow cases share (first 1.5 s of the window, majority of rows)

Counts are slow aircraft / fast aircraft (fast = time to 30 deg at most 1.5 s).

| feature | 1v1 fields: slow (42) | 1v1 fields: fast (872) | 4v4 fields: slow (112) | 4v4 fields: fast (661) |
|---|---|---|---|---|
| no committed turn sense (`sense` = 0) | 9 | 19 | 66 | 128 |
| counterpart not visible (fog) | 8 | 27 | 12 | 58 |
| counterpart not visible at closest approach | 10 | 15 | 13 | 62 |
| chosen counterpart is the pass partner at closest approach | 38 | 867 | 75 | 540 |
| chosen counterpart changed during window | 0 | 1 | 21 | 108 |
| dominant tactic disengaged | 8 | 9 | 27 | 33 |
| dominant tactic turning | 32 | 837 | 42 | 470 |
| dominant tactic merge or offensive (counterpart ahead) | 1 | 13 | 40 | 108 |
| stick cap folds full into middle (cap > 0.5) | 26 | 575 | 25 | 247 |
| boundary steer active | 8 | 9 | 0 | 10 |
| speed above 1.5 x corner speed | 0 | 0 | 0 | 0 |
| aim within 2 deg of nose (stick error) | 6 | 0 | 15 | 0 |

Combined signature (aim, counterpart visible, committed sense):
- 1v1 fields, slow: 6 "aim within 2 deg, counterpart hidden, no sense"; 0 of those in fast.
- 4v4 fields, slow: 3 "aim within 2 deg, counterpart hidden, no sense" and 12 "aim within 2 deg, counterpart seen, no sense"; 0 of those in fast.

## Likely cause

The straight stretch needs two things together: the stick aims near the nose (within 2 deg,
so the pilot's turn gates take the small levels only), and no turn sense has been committed.
The aim comes from the target's lead point, which is ahead when the target is still ahead.
The sense is committed only when the target is aft of the 3/9 line:

- `lean/scripted/opponent.py` line 421: `aft = has_target & (bearing_size > line)`
- line 424: `held = torch.where(same_target & ~forward, held, 0.0)` (a target change clears the sense)
- line 426: `commit = aft & (held == 0.0)`

So after a pass, while the chosen counterpart is still ahead (a stale remembered enemy under fog,
or another enemy ahead), no sense is committed, the stick follows the sign of the error, and the
pilot flies at a point straight ahead (lines 300-302 and 408 set the aim and the stick error).
This is consistent with the 1v1 fields slow cases (counterpart hidden at closest approach in
10 of 42, against 15 of 872 fast) and the 4v4 ones (no sense in 59% of slow, against 19% of fast;
the chosen counterpart is not the pass partner at closest approach in 33%, against 18% of fast).

The stick cap (line 378, folding full into middle) is not the distinguishing rule: the cap is
above 0.5 in 62% of slow and 66% of fast aircraft in 1v1 fields, and 22% of slow and 37% of fast
in 4v4. The boundary steer is rare in both groups.

This is a reading of the counts, not a tested mechanism: I did not change the pilot to check that
removing the `aft` gate for the stale or ahead counterpart removes the slow cases.
