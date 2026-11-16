# Imugi's turning fight: commitment at the merge, initiative, stick by energy

Branch feat/imugi-initiative (from feat/imugi-boundary 47604cd0: game v24, the Y0.46 card with
head-on damage 0.05 and the boundary fix), 2026-11-16/17, laptop RTX 4070, power profile
`performance`. Every number is the scripted agent on both sides, eager, the balance conditions'
settings (`bench.balance_sweep.BALANCE`: deathmatch 300 s without fog, elimination and Frontline
with fog, seeded spawns). Before: the branch's base (`TREE=base`); after: both rules on. Raw
lines in data/.

## The rules (`src/bnb/controllers/imugi_combat.py`)

- `TURN_COMMIT` (env `BNB_TURN_COMMIT`, default 1; 0 is the old per-tick sense): the turning
  fight's sense is committed once and held (`turn_commitment`; carry `fight_sense`,
  `fight_clock`). Released by the next head-on merge once held `COMMIT_HOLD_SECONDS` (2.5 s),
  a new or lost target, an extension, or the boundary steer (BOUNDARY_OVER_MANEUVERS kept).
  - Initiative: the ship whose full-stick rate now (`turn_now`) is above the target's by
    TWO_CIRCLE_RATE_MARGIN, or whose energy is above by ENERGY_ADVANTAGE_RATIO (1.1), commits on
    a head-on approach at the latest tick whose time to the closest approach is within its turn
    onset (`turn_onset`: 3.3 nose periods, 0.30 s at corner speed on both ships; measured by
    turn_onset.py), so its turn is at full rate at the crossing. The nose turns at most
    rate x onset (0.36 rad Yongma, 0.41 Jujak at corner) before the crossing, inside the
    head-on armour's half-strength angle of 0.785 rad (a test).
  - At the merge (target behind the beam, within 4 gun ranges): with an advantage on either side
    and the target turning, the card's choice against the observed sense (two circles where the
    own rate is clearly higher, else one circle): the outmatched ship reacts once; an even pair,
    or a target not turning, by the geometry (the side of the target's tail point).
  - Tie (lateral offset within 1 m): toward the map centre, then left. No random draw.
  - `ENERGY_BREAKS_RATE_TIES` (env, default 0): rate first, energy only between even rates.
    Measured worse for Yongma in 1v1 (0.036 against 0.068); left off.
- `STICK_BY_ENERGY` (env `BNB_STICK_BY_ENERGY`, default 1): in a two-circle turn (the target's
  own sense) below 0.95 of the corner speed, the sustained stick level (0.75: the card's
  `sustained_stick` lift fraction, the turn that holds speed); full stick otherwise, with the
  existing bleed reverse above the corner speed.
- `Characteristics.nose_frequency` (the card's corner_nose_frequency) added for the onset.

## Deathmatch 1v1 (`bench/turn_fight.py`, 128 games)

| metric | before | commit only | both rules |
|---|---|---|---|
| deaths a game | 1.66 | 1.83 | 2.47 |
| merges a game (post-merge windows) | 21.5 | 11.5 | 15.7 |
| merges a kill | 13.3 | 9.0 | 9.2 |
| Yongma kill / point share | 0.591 / 0.577 | 0.012 / 0.056 | 0.068 / 0.079 |
| post-merge windows: scissors / one circle / two circle | 2752 / 3 / 0 | 29 / 808 / 625 | 33 / 1096 / 883 |
| both ships >= 2 reversals (at 15 deg/s) in a window | 99.6% | 1.5% | 1.0% |
| reversals a window, Yongma / Jujak (15 deg/s, hysteresis) | 36.8 / 36.8 | 1.07 / 0.76 | 1.04 / 0.79 |
| sustained turn (> 15 deg/s held 1 s), Yongma / Jujak | 13.9% / 13.9% | 68.3% / 61.9% | 54.2% / 49.8% |
| old definition (>= half the sustained rate, 3 s) | 0.4% / 1.2% | 49.9% / 21.2% | 27.5% / 14.2% |
| stick side flips a minute, Yongma / Jujak | 414 / 459 | 66 / 123 | 95 / 145 |

The oscillation is gone, and the duels are now decided: a held turning fight goes to the turn
fighter (Jujak), so the energy fighter's 1v1 collapses (Yongma 205 deaths by hits and 86 at the
boundary in 128 games, Jujak 25). Stick by energy adds merges and kills.

## Balance spot check (`bench/action_mix.py`, 512 games), Yongma kill share

| condition | band | before | commit only | stick only | both |
|---|---|---|---|---|---|
| deathmatch-2v2 | .481-.521 | 0.421 ± 0.005 | 0.471 ± 0.005 | 0.415 ± 0.005 | 0.466 ± 0.006 |
| deathmatch-5v5 | .494-.507 | 0.489 ± 0.003 | 0.509 ± 0.003 | 0.493 ± 0.003 | 0.514 ± 0.003 |
| elimination-5v5 | .498-.545 | 0.544 ± 0.013 | 0.592 ± 0.014 | 0.539 ± 0.013 | 0.592 ± 0.014 |
| frontline-5v5 | .491-.513 | 0.481 ± 0.004 | 0.528 ± 0.004 | 0.483 ± 0.004 | 0.507 ± 0.004 |

Point shares follow within 0.01 (data/). The commitment carries the shift; stick by energy alone
is neutral.

## Action mix

Step 0 (before, 128 games) in progress.md; per condition before/after in data/bal-*.json (stick,
power and shoot shares per ship). In the 5v5 modes the mix barely moves (the sense manoeuvre is
seldom on in a crowd); in 2v2 the sustained level rises (Yongma 0.75: 8.7% -> 14.0%) and in 1v1
too (6.9% -> 20.4%).

## Stills

/home/vizia/avesnova/review/stills/initiative/: {before,after}_{duel,5v5}_{game,opening,busiest}.png
(`bnb figures trails --scripted`, real-scale deathmatch; duel 1v1 120 s, 5v5 60 s, seed 0). The
duel after: clean two-circle loops and repeated merges; before: weaving S-curves.
