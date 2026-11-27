# Scripted agent: the one-tick action delay (queue 8b, 2026-11-28)

Correctness review 2026-11-28, finding 4 (branch `correctness-review`): the game applies the
action chosen on tick t at t + 1, after the pending action; the trigger aimed from tick t's pose
and the candidate paths started each stick at once. Author: a bug (2026-11-28). Branch
`scripted-delay`, scripts in `sections/scripted-delay/`.

## Log

- 04:35 fix committed (lean/scripted only). `opponent.after_pending`: every ship one tick on
  under its pending flight action through the game's `core.flight_tick` (the firing tick's
  thrust cut left out: it needs GameConfig). The trigger aims (new `opponent.aim`), ranges,
  checks the line and launches from that pose against the enemies one tick on (straight at
  their map velocity), and reads the gun at the shot's tick (with the default gun, one shot a
  tick, fire_interval is 1 and this changes nothing). The own candidate paths start from that
  pose with the first segment one tick shorter (`paths.predict(start=)`), the points at the
  same times; the enemies' paths are unchanged (their pending action is not observed).
  New tests: the pending tick equals the game's step; the trigger holds fire when the nose
  turns off the lead point over the pending tick and fires when it turns onto it; the gun read
  at the shot's tick.
- 04:40 first CPU check (yardstick 1v1, 32 x 450 ticks, old agent, 103 decisions): from the
  decision's pose 28% of fire decisions are more than the half width off at the shot (review:
  22% of 263); from the pose after the pending tick 0%.
- 04:41 started: the aim statistic and hit rate (CPU, 128 envs x 1800 ticks, old and new x
  yardstick and robust centre 3; `measure.sh`), Elo new against old (GPU, `elo.py`).
- 04:48 aim + hit rate done (`aim-*.json`). Error between the decision's prediction and the
  shot, in target half widths (side-on), share above 1:

  | agent (self-play 1v1) | decisions | from decision pose | from pose after pending | shots / ship-min | hits / shot | hits |
  |---|---|---|---|---|---|---|
  | yardstick, old | 3,927 | 20.9% (median 0.36) | 0.8% (median 0.06) | 16.1 | 0.313 | 1,291 |
  | yardstick, new | 5,014 | 17.7% | 0.4% | 20.6 | 0.294 | 1,546 |
  | centre 3, old | 10,283 | 15.4% | 0.2% | 39.0 | 0.182 | 1,816 |
  | centre 3, new | 12,281 | 14.6% | 0.2% | 46.9 | 0.195 | 2,339 |

  Worked: the old agent's statistic reproduces the review (21% against 22%); the new
  trigger's own prediction (pose after pending) is within the half width on 99.6-99.8% of
  decisions (the rest presumably the enemy's own unobserved pending action; not checked). The new agent fires 20-28%
  more and lands 20-29% more hits in the same play time; hits per shot down 0.019 for the
  yardstick (about 2 binomial SE, shots correlated), up 0.013 for the centre.
- 04:53 Elo (`elo.json`, new against old at 1000, both centre 3, fields and fog, the study
  clocks): 1v1 +34 +- 8 (2,048 games), 4v4 +18 +- 5 (4,096), 16v16 +30 +- 8 (2,048). The fix
  wins at every size; the centre's parameters are not re-tuned for it (fire_tolerance in
  particular was tuned against the late aim).
- 04:55 speed (`speed.json`, 4v4, 4096 envs, compiled static, laptop 4070, performance
  profile, GPU otherwise idle): `act` 6.95 ms -> 7.13 ms (+2.6%); first call (compile)
  33 s -> 40 s.

## Next

- Merge decision for the author. A small re-check of `fire_tolerance` (and `lead_error`,
  questions.md 14) on the fixed agent may move them; not done here.
