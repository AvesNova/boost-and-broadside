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
