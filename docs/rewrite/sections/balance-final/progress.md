# Balance final: progress (2026-11-16)

Goal (author, 2026-11-15): Yongma team v Jujak team in deathmatch 5v5 within 55/45 (soft), from
21.8 ± 1.8% on game v22; doctrines first, cards only if needed; keep the kill-aspect mix,
mutual head-on near zero, Imugi's strength (mixed lineups v the before doctrines), Frontline
as a check. Power profile `performance`.

## 04:15 start

- Merged origin/feat/imugi-boundary (83404bbf) into tune/balance-final.
- Reading of the before doctrines: both types' doctrine weight (`strength` x `in_advantage`)
  is near zero in every matchup. Yongma's advantage is (turn ratio)^1.86: 0.62 against Jujak,
  1 in the mirror, against a centre of 1.3; Jujak's is (turn ratio)^-0.385:
  0.91 against Yongma, 1 in the mirror, against a centre of 1.68.
  So the before Imugi flies only the maneuvers (sense, the crowd extension, lock press) and
  the combat layer's base pursuit; the energy fighter never extends or slashes.
- Round 1 (deathmatch 5v5, 256 games E v T, 128 + 128 mixed): turn Yongma's doctrine on
  against Jujak only (turn_exponent -1.86 keeps the mirror off), with and without extend,
  deny, the duel extension gated on the lift ratio, and the sense against every type.

## 04:20 rounds 1-2 (deathmatch 5v5, 256 games E v T; a batch of 16 candidates takes ~2 min)

- Before: 0.26 ± 0.03 and 0.19 ± 0.02 (two repeats). Mixed before-v-before 0.51 / 0.47.
- **Turning Yongma's doctrine on against Jujak makes Yongma worse**: on 0.05, + extend 0.00
  (kills 0.20, tail 77%), + deny 0.04, slash only 0.20, lag only 0.07. The Baekho elements
  as tuned do not help Yongma against Jujak. The crowd extension 0.04-0.07; the sense
  against every type 0.14 (one-circle) / 0.20 (two-circle). Nothing beat the before.
- **Turning Jujak's doctrine on against Yongma flips the matchup**: 0.87 (bleed, deny, lag
  all on; without bleed 0.89, without deny 0.84). But mixed v before falls to 0.25-0.33, so
  that is a weaker Jujak, not a better Yongma: rejected (Imugi strength).
- The duel extension in the mirror (lift-gated, Yongma) leaves E v T unchanged (0.20) and is
  0.56 ± 0.03 against the before on mixed lineups: a possible strength gain, noted.
- Next: a random search of Yongma's doctrine (round 3, 64 candidates, 128 + 128 games),
  and a card round ready (sweep.py now takes a candidate "card" of field factors).
