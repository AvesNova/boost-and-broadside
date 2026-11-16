# Lethality + sticky Imugi: progress

## 16:50-17:05 step 1, lethality (game v23)
- Tried: time to kill from the cards (time_to_kill.py: target max_shield / (shooter
  hit_damage x tail aspect x fire_rate), core hits at the reference distance). Before:
  0.12-0.25 s by pairing (Y on J 0.12, J on J 0.14, Y on Y 0.21, J on Y 0.25).
- Worked: shields and recharge rates x 5.4 (ratio kept): 0.62 / 0.75 / 1.11 / 1.33 s.
  The ratio Y/J (1.78 shield, 1.20 damage) spreads the four pairings by 2.1x, so "about
  1 s for every pairing" cannot hold with the ratio kept; 5.4 is the centre of the
  0.6-1.4 s band in log. Real-scale boundary_max_damage x 5.4 too (40 -> 216 /s) so
  elimination's closing boundary still kills before the clock.
- Next: baseline measurements running (sticky-before unit), Imugi stickiness.

## 16:52-17:20 measurements and step 2, sticky Imugi
- Baseline (game v22, c5b59a8d) and lethality-only (v23, 4b356e85) measured (measure.py,
  512 games per lineup, 5v5 deathmatch, fields on): data/before.json, data/lethality.json.
  Lethality alone halves kills/min (0.86 -> 0.49 per ship-minute), passes per kill
  0.39 -> 0.57, and Jujak dominates: Yongma's team score 0.22/0.25 -> 0.07/0.08.
- Tried: deterministic target hysteresis (`choose_target`: dead, unseen 3 s, or far /
  outscored x2 for 2 s), a hidden committed target kept known 3 s (`remember`), an
  extend-and-return rule from the card, Frontline recovery off.
- Didn't work: the first extension trigger (target behind the beam) fired with Jujak on
  Yongma's tail: Yongma's score fell to 0.05 (data/after1.json); off it was 0.15/0.12
  (data/noextend.json).
- Worked: gating the start on the two opening after a pass: Yongma 0.17/0.17
  (data/after2.json). Target changes 18.7 -> 7.7 per ship-minute, 11.6 -> 4.1 with the old
  target within 2 gun ranges; ships with a target 73% -> 93% (probe.py).
- Didn't move: passes per kill (0.48) and no-merge kills (66%) stay near the baseline: most
  5v5 kills come without a head-on pass of the killer's pair (bounces), whatever the target
  rule does.
- Next: Frontline and elimination check (modes.py), then hand over for the 18:30 flight.

## 17:15-17:25 Frontline and elimination still work (modes.py, 256 games each, 5v5)
| run | Frontline by capture | Frontline median s | Elim. median s | Elim. deaths by boundary | Elim. last death by boundary |
|---|---|---|---|---|---|
| v22, old Imugi | 0.77 | 290 | 91 | 0.09 | 0.17 |
| v23, old Imugi | 0.54 | 531 | 143 | 0.44 | 0.46 |
| v23, sticky Imugi | 0.67 | 368 | 146 | 0.40 | 0.48 |
- Frontline: the sticky Imugi recovers most of the captures lethality cost.
- Elimination: no game reaches the clock, but with 5.4x shields about 40% of deaths are
  the closing boundary's (was 9%): the fights are slower than the shrink. Lethality's,
  not Imugi's. For the author: lengthen elimination's clock / shrink hold, or accept.
- Next: the author flies (18:30).

## 17:45-18:15 round 2: circular lead, offset approach, lead turn, the circle choice
- Measure: measure.py now adds the hit rate (bench.game_loop's new shots / damaging enemy
  hits tally) and two 1v1 duels (turn mirror, Yongma v Jujak; 120 s, 512 games).
- Worked (build 1, c4ede6c8): circular lead + fire on the predicted miss in hit radii.
  Hit rate 5v5 0.32 -> 0.36, 1v1 mirror 0.45 -> 0.50, 1v1 Y v J 0.32 -> 0.49. Yongma's
  5v5 score 0.16 -> 0.07 (aim helps the turner). The author flew it: a good start.
- Cleanups merged: target_switch_in_log_distance and Frontline recovery removed (26a4664a);
  the sense's draw, mirror and lift-ratio gates and ImugiCarry.sense_draw removed
  (95a2321b).
- Didn't work: the first circle rule (two circles unless my radius is smaller) made
  mirrors settle into even two-circle fights: 1v1 mirror kills/min 0.17 -> 0.05 (fb7dd03d,
  data/r2approach.json, measured on box5090).
- Tried next: two circles only with a turn rate now 5% above the target's, else one
  circle (1c57c928): measuring.
