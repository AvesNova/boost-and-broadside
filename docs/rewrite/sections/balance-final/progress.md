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

## 04:27 round 3: random Yongma doctrines (64, 128 + 128 games)

- Best E v T 0.34 ± 0.04 (r3-041), 0.34 (r3-032), 0.31 (r3-007); the median about 0.2. With
  64 draws at ±0.04, the top is mostly selection: a re-check at 512 games is round 5.
  Common to the top three: `lag` 0. No doctrine came near 0.45.
- The GPU lock is shared with another agent's timing jobs (04:28-04:40 waiting).

## 04:59 round 4: card field factors (stored doctrines, 256 + 256 games; the lock was busy 04:27-04:59)

Card fields (not handles), E v T: base 0.26; Yongma damage x1.15 0.27, x1.3 0.37, x1.5 0.51;
Yongma shield x1.2 0.33, x1.4 0.47; Jujak shield x0.85 0.40; Jujak damage x0.85 0.31; Jujak
induced drag x1.3 0.31, x1.6 0.49 (the sustained turn); Yongma engine power x1.2 0.43;
Yongma induced x0.8 0.34; Yongma explosive 1 0.30; max lift (instantaneous turn) Yongma x1.1
0.22, Jujak x0.93 0.20 (no effect). Mixed v before 0.46-0.54 everywhere (strength unchanged).
Aspects unchanged (mutual 3-4%). So the balance moves with the sustained turn, the power, and
damage/shield; not with the instantaneous turn.

Author's go-ahead (via the coordinator, ~04:45): cards may change, modestly, every value inside
the real types' range; prefer handles moving one ratio each. The real-type spread
(`src/bnb/reference/real-plane-types`, the 16 real WWII types, Me 262/163 excluded): sustained
turn 0.503-0.785 rad/s, low-speed acceleration 8.87-18.97, damage per bullet ratio 1.57, dps
ratio 1.42, shield ratio 1.44. Yongma's shield over Jujak's is already 1.78 (outside the real
spread), so a Yongma shield rise is out. Round 6 is handle candidates re-solved
(`handle_cards.py`).
