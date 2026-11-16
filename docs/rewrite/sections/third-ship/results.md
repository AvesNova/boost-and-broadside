# Third ship: Bonghwang, the power-and-roll fighter (draft, 2026-11-16)

Branch `tune/third-ship` (never merged into feat/imugi-boundary). Game v22 rules with the
balance agent's v23 Jujak (sustained turn x0.93, tune/balance-final c875bb84). Every number is
Imugi on both sides, fog on, the preset's fields and map, seats alternating; "C v J" is a team
of five Bonghwang against five Jujak, the score Bonghwang's (a deathmatch game is won on
points over 300 s), ± one standard error. The working log is [progress.md](progress.md).

**Status (13:15):** the ship, its doctrine and a trim for each Yongma-Jujak balance option are
measured; option A's trim is in the game on this branch. The flat options' trims (g 8):

| balance option | Bonghwang trim | 2v2 | 5v5 | 10v10 | elimination 5v5 | Frontline 5v5 |
|---|---|---|---|---|---|---|
| **E** (Jujak turn x0.95, Yongma turn x1.12, shield x0.55) | t0.70 g8 d0.95 s80 | 0.46 / 0.50 | 0.49 / 0.47 | 0.49 / 0.47 | 0.49 / 0.53 | 0.55 / 0.50 |
| **C'** (Jujak x0.91, Yongma x1.08, shield x0.55) | t0.66 g8 d0.95 s85 | 0.43 / 0.50 | 0.51 / 0.51 | 0.52 / 0.56 | 0.48 / 0.49 | 0.55 / 0.43 |
| A (v23, in src) | t0.63 g7 d1.2 s115 | 0.31 / 0.66 | 0.48 / 0.56 | 0.65 / 0.47 | 0.43 / 0.67 | 0.28 / 0.55 |

(Bonghwang v Jujak / Bonghwang v Yongma, deathmatch unless named; 384-1536 games a cell, SE
0.012-0.025. Under A the release pair itself runs 0.27 -> 0.73 over sizes, so no third ship
fits both there.) With E's trim every measured cell is within 0.46-0.55 of both ships.

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

| balance option (Yongma, Jujak) | Bonghwang: sustained turn, g, damage per bullet, shield | DM 5v5: B v J / B v Y / Y v J |
|---|---|---|
| **A** (v23, in src): Jujak turn x0.93 | 0.630, 7, 23.03 (x1.2), 115 | 0.483 / 0.562 / 0.474 (1536) |
| **C'**: Jujak x0.91, Yongma turn x1.08, shield x0.55 | 0.660, 8, 18.23 (x0.95), 85 | 0.513 / 0.515 / 0.52 (1024) |
| **D**: Jujak x0.93, Yongma turn x1.08, shield x0.6 | 0.660, 7, 21.11 (x1.1), 95 | 0.465 / 0.520 / 0.479 (1536) |
| **E**: Jujak x0.95, Yongma turn x1.12, shield x0.55 | 0.700, 8, 18.23 (x0.95), 80 | 0.491 / 0.468 / 0.53 (1024) |

(Damage is x Jujak's 19.19 a bullet, all at 30 Hz. Under E Yongma's turn is 0.664 and
Jujak's 0.726, so Bonghwang's 0.70 sits between them, as 0.63 does under A. The flat options
equalise the shields at 78-85; at 115 Bonghwang would be the tank that wins the big fights,
so its shield follows theirs. With g 7 the flat options leave Bonghwang weak in the small
fights (E, 2v2 against Jujak: 0.30-0.39 at any turn, damage and shield tried): the duel uses
the instantaneous turn, so these trims take the real maximum, g 8. D's trim is g 7 from the
first round; its flat g 8 version is not searched.)

**The first-round trims** (g 7, before the slope rounds; still in `finals1.json`): C' t0.66 d1.1
s90, D t0.66 d1.1 s95, E t0.70 d1.1 s90; their full tables follow.

## Matchups: the flat options' trims (finals2)

`finals2.json`, Bonghwang flying its doctrine against each option's pair, the option's own
Yongma v Jujak from the same run. 2v2 and 5v5 at 1536 games on the 3070; 10v10 512,
elimination and Frontline 768, 1v1 1024 on the laptop.

**C': Bg8t66-d0.95-s85**

| condition (games) | Bonghwang v Jujak | Bonghwang v Yongma | Yongma v Jujak |
|---|---|---|---|
| deathmatch 1v1 (1024) | 0.386 ± 0.009 | 0.474 ± 0.007 | 0.412 ± 0.009 |
| deathmatch 2v2 (1536) | 0.461 ± 0.011 | 0.447 ± 0.011 | 0.488 ± 0.012 |
| deathmatch 5v5 (1536) | 0.516 ± 0.012 | 0.495 ± 0.012 | 0.516 ± 0.012 |
| deathmatch 10v10 (512) | 0.516 ± 0.021 | 0.555 ± 0.021 | 0.498 ± 0.021 |
| elimination 5v5 (768) | 0.477 ± 0.018 | 0.487 ± 0.018 | 0.473 ± 0.018 |
| Frontline 5v5 (768) | 0.553 ± 0.017 | 0.430 ± 0.017 | 0.596 ± 0.017 |

**E: Bg8t70-d0.95-s80**

| condition (games) | Bonghwang v Jujak | Bonghwang v Yongma | Yongma v Jujak |
|---|---|---|---|
| deathmatch 1v1 (1024) | 0.414 ± 0.009 | 0.495 ± 0.008 | 0.422 ± 0.009 |
| deathmatch 2v2 (1536) | 0.470 ± 0.011 | 0.498 ± 0.011 | 0.472 ± 0.012 |
| deathmatch 5v5 (1536) | 0.496 ± 0.012 | 0.500 ± 0.012 | 0.492 ± 0.012 |
| deathmatch 10v10 (512) | 0.493 ± 0.021 | 0.466 ± 0.021 | 0.530 ± 0.021 |
| elimination 5v5 (768) | 0.493 ± 0.018 | 0.526 ± 0.018 | 0.448 ± 0.018 |
| Frontline 5v5 (768) | 0.554 ± 0.017 | 0.501 ± 0.017 | 0.577 ± 0.017 |

(Frontline 10v10 and elimination 10v10, the balance agent's hardest conditions, are queued on
the box; `data/finals2-fl10-*.jsonl`, `-el10-*` when they land.)

## Matchups: the first-round trims, every option (finals1)

`finals1.json` (the per-option cards above, Bonghwang flying its doctrine), each option's own
Yongma v Jujak from the same run (an entry whose card is the option's Yongma, flying Yongma's
doctrine). Deathmatch 300 s, no early win (Bonghwang's points over the other team's); the
1v1 is mostly draws. Seed-to-seed spread at 5v5 is about ±0.03, larger than one run's SE.

**A** (Bonghwang B-d1.2-s115)

| condition (games) | Bonghwang v Jujak | Bonghwang v Yongma | Yongma v Jujak |
|---|---|---|---|
| deathmatch 1v1 (1024) | 0.336 ± 0.008 | 0.587 ± 0.008 | 0.296 ± 0.009 |
| deathmatch 2v2 (1024) | 0.314 ± 0.013 | 0.656 ± 0.013 | 0.267 ± 0.012 |
| deathmatch 5v5 (1536) | 0.483 ± 0.012 | 0.562 ± 0.012 | 0.474 ± 0.012 |
| deathmatch 10v10 (384) | 0.646 ± 0.024 | 0.469 ± 0.025 | 0.732 ± 0.022 |
| elimination 5v5 (768) | 0.425 ± 0.018 | 0.665 ± 0.017 | 0.378 ± 0.018 |
| Frontline 5v5 (768) | 0.275 ± 0.015 | 0.549 ± 0.017 | 0.355 ± 0.016 |

**C'** (Bonghwang Bt66-d1.1-s90)

| condition (games) | Bonghwang v Jujak | Bonghwang v Yongma | Yongma v Jujak |
|---|---|---|---|
| deathmatch 1v1 (1024) | 0.383 ± 0.008 | 0.428 ± 0.008 | 0.425 ± 0.009 |
| deathmatch 2v2 (1024) | 0.419 ± 0.014 | 0.499 ± 0.014 | 0.458 ± 0.014 |
| deathmatch 5v5 (1536) | 0.544 ± 0.012 | 0.567 ± 0.012 | 0.527 ± 0.012 |
| deathmatch 10v10 (384) | 0.566 ± 0.025 | 0.603 ± 0.024 | 0.501 ± 0.025 |
| elimination 5v5 (768) | 0.533 ± 0.018 | 0.605 ± 0.018 | 0.458 ± 0.018 |
| Frontline 5v5 (768) | 0.490 ± 0.017 | 0.471 ± 0.017 | 0.592 ± 0.017 |

**D** (Bonghwang Bt66-d1.1-s95)

| condition (games) | Bonghwang v Jujak | Bonghwang v Yongma | Yongma v Jujak |
|---|---|---|---|
| deathmatch 1v1 (1024) | 0.362 ± 0.008 | 0.425 ± 0.008 | 0.398 ± 0.009 |
| deathmatch 2v2 (1024) | 0.366 ± 0.013 | 0.465 ± 0.014 | 0.407 ± 0.014 |
| deathmatch 5v5 (1536) | 0.465 ± 0.012 | 0.520 ± 0.012 | 0.479 ± 0.012 |
| deathmatch 10v10 (384) | 0.526 ± 0.025 | 0.562 ± 0.025 | 0.492 ± 0.024 |
| elimination 5v5 (768) | 0.483 ± 0.018 | 0.561 ± 0.018 | 0.404 ± 0.018 |
| Frontline 5v5 (768) | 0.368 ± 0.017 | 0.449 ± 0.017 | 0.493 ± 0.017 |

**E** (Bonghwang Bt70-d1.1-s90)

| condition (games) | Bonghwang v Jujak | Bonghwang v Yongma | Yongma v Jujak |
|---|---|---|---|
| deathmatch 1v1 (1024) | 0.369 ± 0.008 | 0.414 ± 0.008 | 0.413 ± 0.009 |
| deathmatch 2v2 (1024) | 0.376 ± 0.014 | 0.422 ± 0.014 | 0.460 ± 0.014 |
| deathmatch 5v5 (1536) | 0.498 ± 0.012 | 0.503 ± 0.012 | 0.531 ± 0.012 |
| deathmatch 10v10 (384) | 0.574 ± 0.024 | 0.592 ± 0.024 | 0.514 ± 0.025 |
| elimination 5v5 (768) | 0.460 ± 0.018 | 0.574 ± 0.018 | 0.457 ± 0.018 |
| Frontline 5v5 (768) | 0.495 ± 0.018 | 0.480 ± 0.017 | 0.590 ± 0.017 |


**Reading.** At the balance target (deathmatch 5v5) every trim is within 0.46-0.57 of both
ships, E's at 0.50 / 0.50. Off that point Bonghwang keeps a slope of its own under the flat
options (weaker in 2v2 and 1v1, a little stronger at 10v10: E 0.38 -> 0.50 -> 0.57 against
Jujak); a lower shield with more turn or damage should flatten it (the slope round, running).
Under A the release pair itself spans 0.27-0.73 over sizes, and Bonghwang lands between them
in every condition; there no third ship can be within 40-60% of both away from 5v5.

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

The branch holds option A (v23's Jujak with Bonghwang d1.2/s115), written by `apply.py`. To fly
another option's pair with its Bonghwang trim (the option's Yongma and Jujak files are copied
from tune/balance-final into `candidates/options/`):

```bash
# from the worktree root; the candidate paths are relative to docs/rewrite/sections/third-ship
A=docs/rewrite/sections/third-ship/apply.py
# E (Jujak turn x0.95, Yongma turn x1.12, Yongma shield x0.55) with its Bonghwang:
PYTHONPATH=$PWD/src:$PWD .venv/bin/python $A candidates/finals2.json "E:Bg8t70-d0.95-s80" \
    candidates/finals2.json "E:Bg8t70-d0.95-s80" --option jt0.95-yt1.12-ys0.55
# C' (Jujak turn x0.91, Yongma turn x1.08, Yongma shield x0.55):
PYTHONPATH=$PWD/src:$PWD .venv/bin/python $A candidates/finals2.json "Cp:Bg8t66-d0.95-s85" \
    candidates/finals2.json "Cp:Bg8t66-d0.95-s85" --option jt0.91-yt1.08-ys0.55
# back to A:
PYTHONPATH=$PWD/src:$PWD .venv/bin/python $A candidates/trim1.json B-d1.2-s115 \
    candidates/trim1.json B-d1.2-s115
```

A changed card needs the game reference re-recorded before the quick
tests pass (`python -m tests.sections.test_game_version --write`).

**Flying it.** Stay fast and work the engine: Bonghwang's drag punishes coasting, and its
low-speed acceleration recovers speed faster than anything else. At the merge, do not commit
to a circle: reverse (the roll is 0.25 s against 0.39) when the bandit commits, make it
overshoot, and take the snap shot at the crossing; accelerate out rather than extend in a dive
(it cannot out-dive Yongma) and re-engage. Against Jujak, never settle into a sustained
turn; against Yongma, never chase it in a straight line.

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

## Fight shape and aspects with the flat trims (deathmatch 5v5, 1536 + 128 mirror games)

Post-merge fights one-circle / two-circle / scissors, kill aspects mutual head-on / offset
head-on / flank / tail, kills a minute, trades:

| | fights | aspects | kills/min | trades |
|---|---|---|---|---|
| E: Yongma v Jujak | 0.31 / 0.57 / 0.13 | 0.04 / 0.15 / 0.62 / 0.19 | 8.21 | 0.11 |
| E: Bonghwang v Jujak | 0.27 / 0.51 / 0.22 | 0.04 / 0.15 / 0.62 / 0.19 | 8.02 | 0.11 |
| E: Bonghwang v Yongma | 0.27 / 0.51 / 0.22 | 0.04 / 0.15 / 0.62 / 0.19 | 7.59 | 0.10 |
| E: Yongma mirror | 0.37 / 0.41 / 0.22 | 0.05 / 0.13 / 0.54 / 0.28 | 7.67 | 0.11 |
| E: Bonghwang mirror | 0.20 / 0.39 / **0.41** | 0.04 / 0.15 / 0.63 / 0.18 | 7.32 | 0.09 |
| C': Bonghwang mirror | 0.19 / 0.43 / **0.38** | 0.04 / 0.15 / 0.62 / 0.20 | 7.47 | 0.10 |

The scissors signature survives every trim: about twice the release pair's share wherever
Bonghwang flies, and its mirror is the only one where scissors are the most common fight.
Mutual head-on kills stay at 4%; kills a minute 7.3-8.2 as in the release pair.

## Open for the author

1. **Bonghwang must be re-trimmed when the Yongma-Jujak card is decided.** Its balance moves
   with the pair's: the three trim numbers (sustained turn, damage, shield; g 7 or 8) were
   tuned against A, C', D and E, and the balance agent's newest round (probe3: Yongma paid in
   store, shot cost and vision) will likely be none of them. `retrim.py grid` builds the
   64-card grid (g 7 and 8) against any pair's ship files; one run per size (2v2, 5v5, 10v10) and
   `retrim.py pick` choose the trim, then elimination and Frontline as checks. On the box
   workers that is about an hour.
2. **Is it different enough?** Its fights are different (scissors, measured), its card is the
   extreme on roll, low-speed acceleration and drag, and its doctrine is its own (no
   post-merge circle). In turn and speed it sits between the other two by construction, and
   under the flat options all three turn alike (0.66-0.73), so the turn no longer separates
   them; fly it and say whether it feels like a third way to fly.
3. **The name**: Bonghwang (봉황, the phoenix) follows the rule (Korean, flying, no "eo"/"eu"). An
   alternative is Samjogo (삼족오, the three-legged sun crow).
4. **No cycle**: every edge is near 50%; nothing here makes an RPS cycle. A cycle would need a
   mechanism the three do not have (gun reach was the strongest axis found, and the set's
   guns already sit at the real maximum range).
5. **g limit**: the flat trims use g 8 (the real maximum); Yongma and Jujak are at 9, outside
   the real range, as are Jujak's climb time and both ships' dive accelerations.
