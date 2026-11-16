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
