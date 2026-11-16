# Third ship: Bonghwang, the power-and-roll fighter (draft, 2026-11-16)

Branch `tune/third-ship` (never merged into feat/imugi-boundary). Game v22 rules with the
balance agent's v23 Jujak (sustained turn x0.93, tune/balance-final c875bb84). Every number is
Imugi on both sides, fog on, the preset's fields and map, seats alternating; "C v J" is a team
of five Bonghwang against five Jujak, the score Bonghwang's (a deathmatch game is won on
points over 300 s), ± one standard error. The working log is [progress.md](progress.md).

**Status:** the card and doctrine are in the game on this branch (d1.2/s115). Open: the
re-check against the balance agent's final Yongma/Jujak cards (its option C is running).

## The ship

**Bonghwang** (봉황, the phoenix). Archetype: **the power-and-roll fighter** (Fw 190A-8 roll,
Bf 109K-4 low-speed acceleration, Tempest V climb), the late-war high-power fighters that
neither out-turned the Spitfire nor out-dove the P-47 but won by rolling and accelerating out
of a fight faster than either.

| | Yongma (energy) | Jujak v23 (turn) | **Bonghwang** |
|---|---|---|---|
| roll (nose reversal, s) | 0.394 | 0.394 | **0.249** (the fastest) |
| low-speed acceleration (m/s²) | 8.79 | 13.54 | **18.97** (the strongest) |
| top speed, a boost from cruise (m/s) | 450 | 434 | **428** (the lowest) |
| zero-lift drag (1/m) | 4.25e-5 | 5.81e-5 | **7.0e-5** (the draggiest) |
| dive acceleration (m/s²) | 46.5 | 40.9 | 43.6 |
| g limit | 9 | 9 | **7** (the lowest) |
| corner speed (m/s) | 288 | 249 | 250 |
| turn, fine / sustained / max (°/s) | 14.1 / 26.1 / 34.0 | 18.5 / 33.9 / 40.7 | 16.0 / 30.8 / 36.1 |
| cruise (m/s) | 335 | 316 | 330 |
| store / climb power (s) | 67 | 54 | 58 |
| damage per bullet, rate | 22.98, 30 Hz, explosive 0.5 | 19.19, 30 Hz | 23.03, 30 Hz |
| shield | 142 | 80 | 115 |

(The turn, top, corner, climb rows are the game's own flights, `flown.py`.)

**How it flies.** It is the middle ship in turn and speed, and the extreme in three things:
it reverses its turn fastest, accelerates hardest from low speed, and bleeds speed fastest
when it stops working the engine. It cannot dive away from Yongma (the lowest boost top
speed) and cannot sit in a sustained turn with Jujak (8% slower), so its fight is the
**scissors**: force overshoots by reversing, accelerate back out, and take the snap shot at the
reversal. Measured: Bonghwang's mirror is half scissors (0.49-0.52 of post-merge fights with
the base doctrine; 0.38 with its own) against 0.15 (Yongma's mirror) and 0.20 (Jujak's), and
its cross fights 0.18-0.23 against 0.12 in Yongma v Jujak. Without the roll the same airframe
(screen 4) fights like the others (mirror 0.21) and scores 10 points lower.

**Its doctrine** (Imugi): Jujak's stored doctrine with the post-merge sense off (`sense` 0):
Bonghwang never commits to a one-circle after a merge; it keeps the pursuit and its rolls. On
mixed lineups (Bonghwang + Jujak against Bonghwang + Jujak) this doctrine scores **0.542 ±
0.015** (2048 games) against the same ship flying Jujak's doctrine; the release types fly one
circle in their mirrors. A random search of the allowed elements (slash, bleed, lag, deny,
the sense; extend and the extensions off by the author's 2026-11-14 decision) found nothing
better: every element at full weight costs strength (lag most), and the best searched
doctrines' gains were the same sense change.

## One ship, a trim per balance option

The Yongma-Jujak balance is not final (tune/balance-final offers A, in src as v23, and the
flatter C', D and E for the author), and Bonghwang's balance moves with it. Its identity (the
roll, the acceleration, the drag, g 7, the in-range dive, its doctrine) stays; three numbers
are trimmed per option: the sustained turn (each option moves the pair's turns toward each
other, so Bonghwang's follows), damage and shield (the options bring the shields to 78-85, and
a 115-shield Bonghwang is then the tank that wins the big fights).

| balance option (Yongma, Jujak) | Bonghwang: sustained turn, damage per bullet, shield | DM 5v5, 1536 games: B v J / B v Y / Y v J |
|---|---|---|
| **A** (v23, in src): Jujak turn x0.93 | 0.630, 23.03 (x1.2), 115 | 0.483 / 0.562 / 0.474 |
| **C'**: Jujak x0.91, Yongma turn x1.08, shield x0.55 | 0.660, 21.11 (x1.1), 90 | 0.544 / 0.567 / 0.527 |
| **D**: Jujak x0.93, Yongma turn x1.08, shield x0.6 | 0.660, 21.11 (x1.1), 95 | 0.465 / 0.520 / 0.479 |
| **E**: Jujak x0.95, Yongma turn x1.12, shield x0.55 | 0.700, 21.11 (x1.1), 90 | 0.498 / 0.503 / 0.531 |

(± 0.012 each. Damage is x Jujak's 19.19 a bullet, all at 30 Hz. Under E Yongma's turn is
0.664 and Jujak's 0.726, so Bonghwang's 0.70 sits between them, as 0.63 does under A.)

## Matchups (option A)

Bonghwang at d1.2/s115 with its doctrine, against the release Yongma and the v23 Jujak. The
release pair's own score is from the same run (its own seed), so each row is a self-contained
triangle.

| mode, size (games a pairing) | Bonghwang v Jujak | Bonghwang v Yongma | Yongma v Jujak |
|---|---|---|---|
| **deathmatch 5v5 (1536)** | **0.490 ± 0.012** | **0.567 ± 0.012** | 0.495 ± 0.012 |
| deathmatch 5v5, other runs (512, 640) | 0.510, 0.473 | 0.597, 0.601 | 0.461 |
| deathmatch 1v1 (1024; mostly draws) | 0.333 ± 0.009 | 0.565 ± 0.009 | 0.310 ± 0.009 |
| deathmatch 2v2 (1024) | 0.311 ± 0.013 | 0.674 ± 0.013 | 0.279 ± 0.013 |
| deathmatch 10v10 (384) | 0.600 ± 0.024 | 0.464 ± 0.025 | 0.732 ± 0.022 |
| elimination 5v5 (512) | 0.396 ± 0.022 | 0.682 ± 0.021 | 0.356 ± 0.021 |
| Frontline 5v5 (512) | 0.326 ± 0.020 | 0.635 ± 0.020 | 0.335 ± 0.023 |
| mirrors | 0.5 by symmetry | | |

Seed-to-seed spread at 5v5 is about ±0.03 (three runs of the same card: 0.47-0.51 against
Jujak, 0.57-0.60 against Yongma), larger than one run's standard error.

**Reading.** At 5v5 deathmatch, the balance target, Bonghwang is level with Jujak and a little
ahead of Yongma. Off that point it moves less than the release pair does: from 2v2 to 10v10
Yongma v Jujak runs 0.28 -> 0.73, Bonghwang v Jujak 0.31 -> 0.60 and Bonghwang v Yongma
0.67 -> 0.46. In every condition Bonghwang lands between the other two (Jujak strongest in
the small fights, elimination and Frontline; Yongma at 10v10), so where the release pair is
far apart no third ship can be within 40-60% of both; the size and mode spread is the release
pair's, and the balance agent's flatter option (C: Jujak turn x0.91, Yongma turn x1.08,
Yongma shield x0.6) is being checked against Bonghwang now.

**Pace and aspects** (kills per minute, mean; kill aspect shares mutual head-on / offset
head-on / flank / tail; passes per kill; trades):

| | kills/min | aspects | passes/kill | trades |
|---|---|---|---|---|
| Yongma v Jujak | 8.37 | 0.03 / 0.13 / 0.62 / 0.22 | 0.37 | 0.11 |
| Bonghwang v Jujak | 7.95 | 0.04 / 0.14 / 0.62 / 0.21 | 0.38 | 0.11 |
| Bonghwang v Yongma | 7.26 | 0.03 / 0.13 / 0.61 / 0.23 | 0.37 | 0.09 |
| Yongma mirror | 6.82 | 0.02 / 0.11 / 0.55 / 0.31 | 0.33 | 0.09 |
| Jujak mirror | 7.86 | 0.04 / 0.14 / 0.58 / 0.25 | 0.40 | 0.12 |
| Bonghwang mirror | 7.11 | 0.03 / 0.15 / 0.63 / 0.19 | 0.38 | 0.09 |

Mutual head-on kills stay at 3-4%; first-merge kills 12-15% as in the release pair.

## Range check

Real range: the 16 real WWII types of `src/bnb/reference/real-plane-types` (Me 262 and Me 163
excluded), handles at game v16 (the set's native x2 units). The combat values are on the set's
own scale since game v12's lethality, so they are checked as ratios against the release pair.

| Handle | Bonghwang | Real range | |
|---|---|---|---|
| stall speed | 94.4 | 75 - 105.6 | Bf 109K-4's |
| g limit | 7 | 6 - 8 | (Yongma and Jujak are 9, outside) |
| cruise | 330 | 261 - 336 | |
| climb time | 290 | 290 - 555 | Tempest V's (Jujak's 270 is outside) |
| low-speed acceleration | 18.97 | 8.87 - 18.97 | Bf 109K-4's |
| sustained turn | 0.63 | 0.503 - 0.785 | |
| nose reversal | 0.249 | 0.249 - 0.623 | Fw 190A-8's |
| dive acceleration | 43.6 | 43.6 - 44.3 | A6M5's (Jujak 40.9 and Yongma 46.5 are outside) |
| zoom acceleration | -34.87 | -34.87 - -34.20 | |
| zero-lift drag | 7.0e-5 | 4.97e-5 - 1.06e-4 | |
| ceiling (store) | 497,563 | 356,420 - 526,210 | the set's |
| fire rate | 30 Hz | 15 - 30 | |
| muzzle speed, spread, gun range, bullet drag, shots per store | 1712, 3.08, 584, 5e-4, 865 | 1040-1760, 0.75-3.08, 400-600, 5e-4-9e-4, 354-920 | the set's gun |
| damage per bullet | 23.03 (1.20 x Jujak's, 1.00 x Yongma's) | real spread 1.57 | inside the release pair's 1.20 |
| damage per second | 691 (1.20 x Jujak's) | real spread 1.42 | |
| shield | 115 (between 80 and 142) | real spread 1.44 | the release pair is 1.78 apart already |
| hit radius, core | 12, 5 | the set's rule | |

## How to fly it

On this branch the real-scale set has three types (Yongma 0, Jujak 1, Bonghwang 2):

```bash
uv run --no-sync bnb play --preset bonghwang-mirror   # five Bonghwang a side
uv run --no-sync bnb play --preset bonghwang-jujak    # Bonghwang and Jujak alternating
uv run --no-sync bnb play --preset bonghwang-yongma
uv run --no-sync bnb play --preset three-ships        # all three
```

`apply.py` (`docs/rewrite/sections/third-ship/apply.py candidates/trim1.json B-d1.2-s115
candidates/trim1.json B-d1.2-s115`) wrote the ship file, the doctrine, Jujak's v23 card and the
maps; in the play shell's tuning panel Bonghwang is the third type.

## What did not work

- **The radius fighter** (A6M5/Ki-43: stall 75, 45°/s max turn, cruise 275, fragile): 0.06-0.13
  against Jujak, 0.09-0.20 against Yongma, out-turning Jujak at every stick level. A slow ship
  that charges slowly loses the 5v5 melee; shield is the strongest repair (80 -> 110: 0.28 /
  0.46). Doctrines (one- or two-circle against all, bleed, deny, slash) did not help.
  Radius-not-rate variants (stall 75 with a Jujak-level sustained turn) do not solve.
- **The heavy boom-and-zoom** (P-47/Tempest: sustained 0.52): beats Yongma (up to 0.68) and
  loses to Jujak (0.07-0.17); its card needs a low-speed acceleration of 6, below the real 8.87.
- **Short-range cannon** (400 m, Bf 109K's MK 108-like): 0.01-0.24. Gun reach is the strongest
  axis under Imugi: the real 400-600 m spread is worth more than 30% damage, and the set's guns
  sit at its top.
- **15 Hz cannon** at a real per-bullet ratio (1.5 x a 30 Hz gun's) has 0.75 of its damage per
  second; at balance it needs a per-bullet damage outside the real spread.
