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

## 05:09 round 5: the top random doctrines re-checked (512 E v T + 256 mixed)

Before 0.242 ± 0.018; r3-007 0.285 ± 0.019, r3-033 0.275 (mixed 0.40: weaker), r3-032 0.271,
r3-041 0.269 ± 0.019 with mixed 0.580 ± 0.030 (a stronger Imugi), the rest 0.20-0.24. So the
random search's top was mostly selection: doctrines move E v T by +3-4 points at most.
**Doctrines alone cannot reach 45%; the card is needed** (the author's go-ahead came in at
~04:45). r3-041 is the doctrine to carry: E v T +2.7 and stronger on mixed lineups (extend
0.93 gated on the energy ratio, lag 0, the duel extension in every matchup, wider fight range).

## 05:21 rounds 6-7: handle solves; r3-041 (d041) with card factors

- Handle solves (`handle_cards.py`, 7 processes, 14 min): Jujak sustained turn x0.95 and x0.91
  solve (the solver moves the induced drag x1.39 / x1.79, the static thrust, the sticks);
  x0.87 does not. **Yongma's low-speed acceleration does not solve at x1.25 or x1.5** (misses
  by 8-23%; ill-conditioned with the sustained turn and the cruise), and x1.15 moves the static
  thrust 49 -> 474: dropped. Yongma climb time x0.88 (333.6 -> 293.6 s, the Tempest's 290 the
  real floor) is the engine power x1.107. Damage is direct.
- Round 7 (d041 + card fields, 256 games): base 0.36, Jujak induced x1.3 0.43, x1.45 0.47;
  Yongma power x1.2 0.44; damage x1.3 0.43; induced x1.3 + power x1.1 0.52; induced x1.3 +
  damage x1.15 0.49 (mixed 0.59). The d041 base here (0.36 ± 0.03) is above its 512-game
  check (0.27 ± 0.02): noise between seeds is larger than I like; final numbers at 1024+.
- Round 8: d041 on the solved handle cards (Jujak turn, Yongma climb, Yongma damage, combos),
  512 games.

## 05:40 import check (coordinator's question)

Every sweep unit (rounds 1-8 and the finalists) runs `sweep.sh`: `systemd-run ...
-E PYTHONPATH=src:. --working-directory=/home/vizia/avesnova/bnb-balance-final`. Each unit's
journal shows its tracebacks/warnings from `/home/vizia/avesnova/bnb-balance-final/src/bnb`
(rounds 1, 3, 4, 5, 7, 8 checked), and `PYTHONPATH=src:. python -c "import bnb"` gives this
worktree's file. The card candidates are passed to the bench as explicit field values
(`{"set": ...}` or factors on the loaded card), and the handle solves ran with
`PYTHONPATH=src:.`. No round ran against the main worktree's code. From here on the scripts
use `PYTHONPATH=$PWD/src:$PWD`.

## 06:15 the d041 doctrine is out; finalists again on the before doctrine

- Finalists (d041 + handle cards, 1024 games, deathmatch 5v5): before 0.219 ± 0.012 (the
  freeze's 21.8 reproduced); d041 + Jujak turn x0.95 0.452 ± 0.015; + Yongma damage x1.08
  0.519 ± 0.015; + Yongma climb x0.88 0.557 ± 0.015. At 10v10 (300 s, no tickets) before
  0.41, d041 + Jujak turn x0.95 0.73: the card change overshoots at 10v10.
- **The quick tests caught it**: `test_play_doctrines` holds the author's decision of
  2026-11-14: Baekho's running-away elements (extend, the crowd extension, the duel
  extension) stay off (they ran ships out of bounds and taught the BC student to flee).
  d041's gain comes from exactly those (extend 0.93, crowd_sense 0.09, duel_extend 0.22; its
  turn_exponent 1.86 leaves its doctrine weight off against Jujak). So d041 is not allowed;
  the d041 finalist runs on the box (elimination, Frontline, 2v2) were stopped. Of the
  allowed elements (slash, bleed, lag, deny, the sense), rounds 1-2 found nothing that helps.
- Now: finalists2 = the before doctrine on handle cards (Jujak turn x0.97-0.91, Yongma damage
  x1.08/1.18, Yongma climb x0.88), on the rented 3070 (dm 5v5 1024 games, then 10v10 and
  2v2); a constrained random doctrine search (round 10: extend, crowd and duel extension held
  at 0) on the laptop when the lock frees.
- `test_ship_sets.py::test_training_takes_its_games_set_card` fails on the base tree too (a
  'trial' set has no drawn lineup types): not this branch's.

## 06:40 finalists2 (before doctrine, handle cards, deathmatch 5v5, 1024 games, the 3070)

| card | Yongma team | 
|---|---|
| before | 0.205 ± 0.012 |
| Jujak turn x0.97 + Yongma damage x1.18 | 0.417 ± 0.015 |
| Jujak turn x0.95 | 0.397 ± 0.015 |
| Jujak turn x0.95 + Yongma damage x1.08 | 0.440 ± 0.015 |
| Yongma damage x1.18 + climb x0.88 (Jujak untouched) | 0.457 ± 0.015 |
| Jujak turn x0.95 + Yongma damage x1.18 | 0.481 ± 0.015 |
| **Jujak turn x0.93** (one handle) | **0.485 ± 0.015** |
| **Jujak turn x0.95 + Yongma climb x0.88** | **0.494 ± 0.015** |
| Jujak turn x0.93 + Yongma damage x1.08 | 0.558 ± 0.015 |
| Jujak turn x0.91 | 0.597 ± 0.015 |

- Round 10 (64 random Yongma doctrines with extend, the crowd and duel extension held at 0,
  128 games): the top 0.31 ± 0.04, the rest at the before's level: the allowed doctrine
  elements do not move the matchup.
- The two finalists (A: Jujak turn x0.93; B: Jujak turn x0.95 + Yongma climb x0.88) go to the
  final measurement (`candidates/final.json`: deathmatch 5v5 2048 games, elimination and
  Frontline 5v5 1024, on the 3070; deathmatch 1v1 and elimination 2v2 on the laptop). More
  handle variants solving (round 11: Jujak low-speed acceleration, climb, drag, cruise, stall;
  Yongma climb x0.93).
- The box's job script now takes `flock /tmp/gpu.lock` (shared with the third-ship agent).

## 07:20 the after is in (v23), results.md written; round 11; more checks running

- **Final, deathmatch 5v5, 2048 games (3070, seed 100): before 24.8 ± 0.9%, A (Jujak turn
  x0.93) 47.7 ± 1.1%, B (Jujak turn x0.95 + Yongma climb x0.88) 51.2 ± 1.1%.** 1v1 30-31%
  for all three (mostly draws); elimination 2v2 19.0 / 24.4 / 28.1%. A is applied to src/
  (game v23, the reference re-recorded, quick tests on the touched files pass) and committed:
  c875bb84. results.md has the tables, the range check and how to test.
- sweep.py now plays before/'s card by default (`--base before`), so a sweep means the same
  thing whatever src/ holds; laptop jobs started before that ran from a detached worktree at
  the base (/home/vizia/avesnova/bnb-balance-sweep).
- Round 11 (512 games, other Jujak/Yongma handles alone): Jujak low-speed acceleration x0.85
  24.6%, x0.75 18.2% (the solver lowers its induced drag with it); Jujak climb x1.05 20.2%;
  drag x1.1 24.3%; cruise x0.97 22.7%; stall x1.05 20.7%; Yongma climb x0.93 30.0%; Jujak
  turn x0.94 45.9%; turn x0.95 + Yongma climb x0.93 50.5%; turn x0.94 + climb x0.93 48.0%.
  **Only the sustained-turn ratio (and a little Yongma power) moves the matchup.**
- Running: Frontline and elimination 10v10 (laptop), elimination and Frontline 5v5 (3070,
  waiting on the shared lock), round 12 (the size spread: Jujak turn x0.91-0.93 with Yongma
  shield x0.8-0.9) at 5v5, 2v2, 10v10 on the laptop.

## 07:50 round 12: the size spread (deathmatch 2v2 / 5v5 / 10v10, 512 games, laptop)

| card | 2v2 | 5v5 | 10v10 |
|---|---|---|---|
| before | 13.9 | 22.8 | 40.8 |
| Jujak turn x0.93 (the v23 after) | 28.6 | 51.8 | 71.3 |
| Jujak turn x0.91 | 34.1 | 54.6 | 81.0 |
| Jujak turn x0.91 + Yongma shield x0.8 | 27.7 | 42.4 | 58.3 |
| Jujak turn x0.91 + Yongma shield x0.85 | 29.5 | 48.3 | 69.9 |
| Jujak turn x0.91 + Yongma shield x0.9 | 30.7 | 48.9 | 73.9 |
| Jujak turn x0.93 + Yongma shield x0.9 | 26.9 | 38.6 | 62.0 |
| Jujak turn x0.93 + Yongma shield x0.85 | 19.7 | 36.8 | 55.2 |

(SE 1.4-2.1 points.) A Yongma shield cut takes more off the 10v10 than the 2v2, so more
Jujak turn with less Yongma shield flattens the spread: x0.91 + shield x0.8 spans 28-58
(30 points) against x0.93's 29-71 (43). Shield x0.8 is 113.6, 1.43 times Jujak's: inside the
real 1.44, which the before's 1.78 is not. Next: Jujak turn x0.88-0.90 (solving) with Yongma
shield x0.75-0.8, and these in elimination and Frontline 5v5.
- handle_cards.py now reads before/'s ship files (src/ holds v23 since c875bb84; a solve
  started at 07:50 from src/ was stopped before it finished, none of its numbers used).

## 08:12 elimination 5v5 final; round 12 in elimination and Frontline; Jujak turn's solver limit

- Final elimination 5v5 (1024, 3070): before 20.4 ± 1.3%, A 38.1 ± 1.5%, B 38.4 ± 1.5%.
- Round 12 in the other modes (512 games, laptop): elimination 5v5 before 26.0, x0.93 37.9,
  x0.91 + shield x0.8 36.1, x0.91 + shield x0.85 39.3; Frontline 5v5 before 8.4 (!), x0.93
  33.3, x0.91 + shield x0.8 33.2, x0.91 + shield x0.85 38.1. Frontline is the most lopsided
  mode and stays Jujak-favoured with every card so far.
- Jujak turn x0.88, 0.89, 0.90 do not solve (the low-speed acceleration or the nose reversal
  missed by 1-9%): x0.91 is the limit with the other handles held. Round 13 (3070): x0.91 with
  Yongma shield x0.75-0.85 and Yongma climb x0.88/0.93 or damage x1.08. Round 14 (laptop):
  Yongma sustained turn x1.08 with them.
- An elimination run on the laptop failed once on "Disk quota exceeded" in /tmp (the tmpfs
  was full for a moment; 8.9 of 23 GB used after); re-run, fine.

## 08:40 rounds 14-15: Yongma's turn and a deep shield cut (deathmatch 2v2 / 5v5 / 10v10, 512)

| card | 2v2 | 5v5 | 10v10 |
|---|---|---|---|
| Yongma turn x1.08 | 36.3 | 58.0 | 84.7 |
| Jujak turn x0.95 + Yongma turn x1.08 | 52.1 | 80.3 | 94.4 |
| Jujak x0.91 + Yongma turn x1.08 + shield x0.8 | 62.3 | 82.0 | 92.6 |
| Yongma turn x1.08 + shield x0.8 | 28.1 | 39.4 | 60.1 |
| Yongma turn x1.08 + shield x0.7 | 25.9 | 28.2 | 43.5 |
| Yongma turn x1.08 + shield x0.6 | 21.5 | 18.2 | 20.7 |
| Jujak x0.95 + Yongma turn x1.08 + shield x0.6 | 31.0 | 41.4 | 45.4 |
| **Jujak x0.91 + Yongma turn x1.08 + shield x0.6** | **50.4** | **54.4** | **62.6** |

The shield sets the slope over sizes (a shield cut takes most off the 10v10; at x0.6 the
spread even inverts) and the turn ratio sets the level. Jujak x0.91 + Yongma turn x1.08 +
shield x0.6 is the first card within about 13 points from 2v2 to 10v10 and centred. It is a
larger change: the turn ratio 1.29 -> 1.09, the shields nearly equal (85.2 against 79.7);
every value inside the real spread. Round 16: smaller steps around it (Yongma turn x1.04-1.08
solved, shield x0.6-0.7).
- The 3070 has been held since ~08:00 by another agent's compile-memory measure (the lock
  held, the GPU at 0%); my Frontline 5v5 final and round 13 wait behind it.
