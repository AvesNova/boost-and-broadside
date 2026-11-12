# Maneuvers: progress

Branch `tune/maneuvers` (from `origin/tune/tank-armour` 010a0931). The task: give the scripted
agent air-combat maneuvers that break the symmetric stalemates (the Jujak mirror's
side-by-side two circles under the tank-armour cone) and beat the joust. Hard stop 08:00
KST. Power profile: **performance** (read 00:13). The laptop GPU (RTX 4070 laptop, 8 GB) is
this job's alone; every run a `systemd-run --user` unit with `MemoryMax=8G`, one at a time.

## Code

Three maneuvers as doctrine elements (`scripted_doctrine.Doctrine`, the combat layer's
`scripted_combat.maneuvers`), every value a doctrine-table column, so the game loop and
the ES sweep them per group without a recompile. Unlike the older elements they are not
weighted by `strength` x `in_advantage`: r9's advantage gate is shut in every mirror (Jujak
centre 1.68, Yongma 1.3 at an advantage of 1), so r9 flies the bare combat layer there.

- **Lead turn** (`lead_turn`, `lead_turn_in_seconds`, `lead_turn_in_crossing`,
  `lead_turn_stick`, `lead_turn_one_circle`): on a head-on approach (the flight paths within
  `lead_turn_in_crossing` rad of opposed) with the merge `lead_turn_in_seconds` away (range
  over closure), turn at `lead_turn_stick` of full stick toward the side the bandit will
  pass on (its offset at the closest approach): the two-circle sense; with
  `lead_turn_one_circle` > 1/2 away from it (one-circle). The committed turn side follows.
- **Yo-yo** (`yoyo`, `yoyo_after_seconds`, `yoyo_phase`, `yoyo_climb_seconds`,
  `yoyo_ease`, `yoyo_dive_seconds`): a stalemate clock in the carry
  (`ScriptedCarry.stalemate_seconds`) runs while the committed target is within 4 gun
  ranges and no angle is gained (nose within 1 stall angle of the aim point while the
  target's nose is more than 2 of its stall angles off the ship; the joust does not count);
  past `yoyo_after_seconds`, when the target comes within `yoyo_phase` rad of the nose,
  the yo-yo's clock (`yoyo_seconds`) starts: ease the stick to `yoyo_ease` and reverse
  (speed into the store) for `yoyo_climb_seconds`, then the combat layer's stick and boost
  (the store into speed) for `yoyo_dive_seconds`; then both clocks start over.
- **Brake** (`brake`, `brake_in_stall_angle`, `brake_in_gun_range`): overshoot forcing,
  the store as a speed brake: a closing target in the rear hemisphere with its nose on
  the ship inside the range: reverse, while the combat layer turns toward it.

The clocks advance only on a switching decision (a re-read carry gives the same
marginals) and reset on a fresh ship or a lost target. The doctrine table is 14 columns
wider (the play shell's compiled tick needs a rebuild). Tests:
`test_a_lead_turn_turns_before_the_merge_to_the_passing_side_or_away`,
`test_a_yoyo_climbs_with_the_stick_eased_after_a_stalemate_then_dives`,
`test_a_brake_reverses_with_a_closing_target_on_the_tail`.

## Tools

`sizes.py`, `table.py`, `launch.sh` (tank-armour's), `make.py` (the candidates: each card x
each maneuver set, the set on both types over r9). Cards: frozen; D2 (`balance-d2`); D2 with
the 12-degree p8 cone at f0 0.085 and damage x1.5 (tank-armour round 3's pick, "cone
0.085"); the same at f0 0.01 (the near-immune nose of the side-by-side stalemate, "cone
0.01"). Against r9 at 1v1 the `mixed` lineup is the Yongma mirror and `mixed reversed` the
Jujak mirror (`teams_of` at size 1); at 5v5 they are the mixed lineups.

## 00:13-00:20 KST: round 1 (`round1/`): first settings, 1v1, every card

512 games a job; the maneuver on both types over r9. Self-play mirrors and the candidate
against r9 in the 1v1 mirrors (score, ±0.02). `mtable.py round1/sizes.json` has every
column; here the Jujak mirror (mutual head-on kill share, life; one / two circle / scissors;
side-by-side share of the two-circle fights) and the strength:

| Card: maneuver | JJ mutual, life | JJ 1c / 2c / sc, side by side | YY mutual, life | v r9 JJ / YY |
|---|---|---|---|---|
| frozen: r9 | 0.80, 148 s | 0.16 / 0.80 / 0.04, 0.64 | 0.33, 956 s | 0.503 / 0.508 |
| frozen: lead 1 s | 0.80, 199 s | 0.02 / 0.78 / 0.20, 0.51 | 0.11, 1349 s | 0.409 / 0.481 |
| frozen: lead 2 s | 0.61, 396 s | 0.01 / 0.62 / 0.37, 0.30 | 0.01, 1314 s | 0.157 / 0.454 |
| frozen: lead 5 s | 0.49, 523 s | 0.01 / 0.41 / 0.58, 0.28 | 0.04, 1549 s | 0.082 / 0.422 |
| frozen: lead 4 s one-circle | 0.00, 11804 s | 0.01 / 0.98 / 0.01, 0.02 | 0.00, 19188 s | 0.372 / 0.490 |
| frozen: high yo-yo 20 s | 0.34, 267 s | 0.11 / 0.85 / 0.04, 0.70 | 0.01, 1303 s | 0.241 / 0.423 |
| frozen: brake | 0.79, 153 s | 0.15 / 0.81 / 0.04, 0.62 | 0.32, 913 s | 0.479 / 0.510 |
| D2: r9 | 0.64, 255 s | 0.15 / 0.81 / 0.04, 0.74 | 0.33, 281 s | 0.490 / 0.486 |
| D2: lead 1 s | 0.55, 333 s | 0.02 / 0.62 / 0.36, 0.60 | 0.19, 398 s | 0.501 / 0.453 |
| D2: lead 2 s | 0.19, 1021 s | 0.01 / 0.46 / 0.53, 0.27 | 0.05, 632 s | 0.359 / 0.381 |
| D2: lead 5 s | 0.15, 1607 s | 0.00 / 0.32 / 0.67, 0.18 | 0.02, 723 s | 0.207 / 0.332 |
| D2: high yo-yo 20 s | 0.01, 1303 s | 0.06 / 0.90 / 0.04, 0.55 | 0.06, 665 s | 0.062 / 0.340 |
| D2: brake | 0.64, 257 s | 0.15 / 0.81 / 0.04, 0.74 | 0.35, 285 s | 0.499 / 0.492 |
| cone 0.085: r9 | 0.45, 286 s | 0.16 / 0.81 / 0.04, 0.84 | 0.08, 391 s | 0.507 / 0.492 |
| cone 0.085: lead 1 s | 0.27, 483 s | 0.02 / 0.59 / 0.39, 0.63 | 0.04, 323 s | 0.483 / 0.493 |
| cone 0.085: lead 2 s | 0.11, 545 s | 0.01 / 0.48 / 0.52, 0.29 | 0.03, 393 s | 0.301 / 0.395 |
| cone 0.085: high yo-yo 20 s | 0.03, 1119 s | 0.07 / 0.91 / 0.03, 0.57 | 0.02, 659 s | 0.071 / 0.267 |
| cone 0.01: r9 | 0.03, 1021 s | 0.14 / 0.82 / 0.05, 0.86 | 0.02, 445 s | 0.483 / 0.505 |
| cone 0.01: lead 1 s | 0.04, 1181 s | 0.02 / 0.55 / 0.43, 0.61 | 0.02, 461 s | 0.503 / 0.493 |
| cone 0.01: lead 2 s | 0.05, 718 s | 0.01 / 0.45 / 0.54, 0.29 | 0.01, 410 s | 0.409 / 0.453 |

- **The lead turn loses to the joust.** Against r9 (which holds the head-on line and
  turns after the pass) every lead turn started 2 s or more before the merge loses, the
  earlier the worse (D2 JJ: 0.36 at 2 s, 0.21 at 5 s; frozen 0.16, 0.08). The one-circle
  sense loses too (0.17-0.37). At 1 s it is even on D2 and both cones (0.48-0.50) and loses
  on the frozen card (0.41), where the frontal hit is full.
- **It changes the fight's shape, though**: the two-circle share falls and the scissors
  rise (D2 JJ: scissors 0.04 to 0.36 at 1 s, 0.53 at 2 s; side by side 0.74 to 0.60 and
  0.27), and in self-play the lead turn shortens the cone 0.01 stalemate (1021 s to 718 s
  at 2 s).
- **The one-circle lead turn is a stalemate of its own** (self-play life 3,000-19,000 s,
  0.98 two-circle by the bench's sense test: both ships turning the one way round).
- **The (high) yo-yo loses badly** (0.03-0.26) and takes the joust out of self-play
  (D2 JJ mutual 0.64 to 0.01) only by not finishing fights. Why (`regime.py`, r9's Jujak
  mirror on cone 0.085, 90 s, 512 games): inside one gun range the ships fly at **0.39 of
  the corner speed** (p10-p90 0.35-0.69) with the store at 0.21-0.35. The turning fight is
  slow and lift-limited, where the turn rate is the lift coefficient times the speed: a
  faster ship turns faster, up to the corner speed (0.71 rad/s there against 0.31 sustained).
  The high yo-yo's climb (speed into the store) is the wrong trade in that regime; the
  turn radius (1 / C_max) does not depend on the speed at all, so the one-circle is a
  stalemate too. **Round 2 tries the low yo-yo**: unload and dive (store into speed) toward
  the corner speed, then pull.
- **The brake is inert** (0.48-0.51; the fights barely change): a closing bandit on the
  tail is rare in a slow two-circle fight.

## 00:23-00:29 KST: round 2 (`round2/`): the low yo-yo; the lead turn at 0.5-1.5 s

512 games, 1v1. The low yo-yo: after `after` s of stalemate, with the target within `ph`
rad of the nose and its nose off the ship, unload (stick `ease`, 0 = straight) and dive
for up to `s` seconds or until `to` corner speeds (1.0 unless named), then pull. Columns:
the candidate against r9 in the 1v1 mirrors, and the damage a game taken in the Jujak duel
by the candidate / by r9 (the bench's new `damage_by_side`).

| Maneuver | frozen JJ / YY; dmg | D2 JJ / YY; dmg | cone 0.085 JJ / YY; dmg | cone 0.01 JJ / YY; dmg |
|---|---|---|---|---|
| r9 | 0.506 / 0.488; 176 / 178 | 0.505 / 0.493; 120 / 120 | 0.525 / 0.503; 107 / 110 | 0.509 / 0.508; 38 / 39 |
| lead 0.5 s | **0.569** / 0.496; 165 / 173 | 0.503 / 0.515; 113 / 118 | 0.532 / 0.490; 104 / 106 | 0.505 / 0.511; 37 / 39 |
| lead 1 s | 0.419 / 0.502; 173 / 155 | 0.519 / 0.473; 110 / 113 | 0.501 / 0.482; 101 / 100 | 0.506 / 0.492; 39 / 39 |
| lead 1 s stick 0.6 | 0.306 / 0.486 | 0.461 / 0.457 | 0.465 / 0.490 | 0.496 / 0.500 |
| lead 1.5 s | 0.230 / 0.473; 180 / 123 | 0.413 / 0.433 | 0.399 / 0.437 | 0.489 / 0.485 |
| low 10 s ph1.5 4 s | 0.158 / 0.480; 171 / 86 | 0.264 / 0.356; 92 / 41 | 0.212 / 0.362 | 0.228 / 0.315 |
| low 10 s ph3.1 4 s | 0.032 / 0.308 | 0.080 / 0.287 | 0.064 / 0.263 | 0.030 / 0.313 |
| low 10 s ph1.5 3 s ease 0.4 to 0.8 | 0.476 / 0.500 | 0.472 / 0.504 | 0.442 / 0.483 | 0.445 / 0.483 |
| low 10 s ph0.5 4 s | 0.314 / 0.474 | 0.373 / 0.409 | 0.312 / 0.404 | 0.355 / 0.404 |
| high 20 s (no yo-yo under its guns) | 0.266 / 0.446 | 0.074 / 0.350 | 0.076 / 0.328 | 0.086 / 0.339 |

- **The low yo-yo fails as clearly as the high one** (0.02-0.37 against r9; the mildest,
  eased to 0.4 and ended at 0.8 corner speeds, 0.44-0.50). Unloading gives the bandit,
  still at full stick, the angles: the yo-yo ship takes about twice the damage it deals
  (D2 92 / 41 a game). Its self-play shows the same thing from both sides: the mirrors stop
  ending (lives of 2,000-10,000 s with the 5-10 s triggers). In a slow two-circle fight
  where both ships already fly the best sustained turn, any phase off full stick is
  a loss. **The yo-yo is not a stalemate breaker in this flight model**; the vertical's
  geometry (the turn continuing out of plane while the speed trades) is what makes it work
  in a real fight, and the store has no geometry.
- **The lead turn's only winning setting is short: 0.5 s** (frozen JJ 0.569, the first
  candidate to beat r9 in a mirror; D2 0.503 / 0.515, cone 0.085 0.532 / 0.490, cone 0.01
  0.505 / 0.511), and it takes less damage than it deals on every card. Round 3 maps it
  around 0.5 s and adds the 5v5 strength.

## 00:30-00:45 KST: round 3 (`round3/`): the lead turn at 0.25-1 s; 1v1 and 5v5

512 games. `cross` is `lead_turn_in_crossing`'s centre (rad from opposed; 0.5 unless
named). Against r9 in the 1v1 mirrors (JJ / YY) and the damage a game in the Jujak duel
(the candidate's / r9's):

| Lead turn | frozen | D2 | cone 0.085 | cone 0.01 |
|---|---|---|---|---|
| r9 (control) | 0.512 / 0.496; 182 / 182 | 0.497 / 0.511; 119 / 120 | 0.464 / 0.498; 110 / 106 | 0.490 / 0.515; 39 / 38 |
| 0.25 s | 0.535 / 0.503 | 0.518 / 0.486 | 0.521 / 0.492 | 0.499 / 0.513 |
| **0.5 s** | **0.566 / 0.521**; 165 / 176 | **0.535 / 0.517**; 110 / 120 | **0.543 / 0.503**; 102 / 107 | 0.507 / 0.482; 37 / 40 |
| 0.75 s | 0.501 / 0.506 | 0.525 / 0.475 | 0.529 / 0.507 | 0.512 / 0.486 |
| 0.5 s, cross 1.0 | 0.562 / 0.501 | 0.512 / 0.515 | 0.545 / 0.498 | **0.539 / 0.508**; 34 / 41 |
| 0.5 s, cross 1.5 | 0.560 / 0.509 | 0.526 / 0.501 | 0.530 / 0.486 | 0.523 / 0.505 |
| 0.75 s, cross 1.0 | 0.521 / 0.498 | 0.530 / 0.493 | 0.518 / 0.476 | 0.507 / 0.505 |
| 1 s, cross 1.0 | 0.396 / 0.485 | 0.486 / 0.470 | 0.474 / 0.492 | 0.522 / 0.511 |
| 0.5 s, stick 0.75 | 0.511 / 0.493 | 0.500 / 0.494 | 0.516 / 0.493 | 0.515 / 0.489 |
| 0.5 s, one-circle | 0.292 / 0.451 | 0.459 / 0.441 | 0.464 / 0.414 | 0.467 / 0.447 |
| 0.5 s + brake | 0.569 / 0.494 | 0.541 / 0.502 | 0.516 / 0.507 | 0.505 / 0.494 |

- **The lead turn 0.5 s before the merge beats the joust in the Jujak mirror on three
  cards** (0.535-0.566, ±0.02) and holds the Yongma mirror (0.48-0.52), dealing more damage
  than it takes. The window is narrow: 0.25 s gains less, 1 s is even or worse. Full stick
  matters (0.75 of it: even), and the two-circle sense: the one-circle lead turn loses
  everywhere. On the near-immune cone 0.01 it needs the wider crossing gate (cross 1.0:
  0.539), since there the approaches are offset (the kills are offset head-on passes).
- **Self-play mirrors** (`mtable.py round3/sizes.json`): the Jujak mirror's mutual share
  falls on the cones (cone 0.085: 0.48 to 0.35; D2 0.67 to 0.63; frozen unchanged 0.78),
  life rises (cone 0.085 297 to 391 s); merges a kill 3.2-3.6. The lead turn shifts D2's
  energy v turn duel toward the Yongma (1v1 0.49 to 0.54-0.59).

The 5v5 strength against r9 (mixed / mixed reversed, ±0.022; r9's own control row shows the
noise, 0.469-0.521):

| Lead turn | frozen | D2 | cone 0.085 | cone 0.01 |
|---|---|---|---|---|
| r9 (control) | 0.489 / 0.469 | 0.492 / 0.508 | 0.498 / 0.510 | 0.495 / 0.521 |
| 0.25 s | 0.507 / 0.464 | 0.493 / 0.503 | 0.505 / 0.484 | 0.502 / 0.487 |
| **0.5 s** | **0.543 / 0.490** | **0.526 / 0.512** | 0.496 / 0.495 | **0.514 / 0.521** |
| 0.75 s | 0.440 / 0.469 | 0.507 / 0.487 | 0.479 / 0.492 | 0.444 / 0.467 |
| 0.5 s, cross 1.0 | 0.481 / 0.506 | 0.530 / 0.482 | 0.497 / 0.514 | 0.475 / 0.511 |
| 0.5 s, cross 1.5 | 0.519 / 0.479 | 0.508 / 0.545 | 0.474 / 0.493 | 0.508 / 0.526 |
| 1 s, cross 1.0 | 0.445 / 0.417 | 0.427 / 0.463 | 0.444 / 0.449 | 0.444 / 0.468 |
| 0.5 s, one-circle | 0.447 / 0.441 | 0.466 / 0.498 | 0.479 / 0.462 | 0.461 / 0.458 |
| 0.5 s + brake | 0.483 / 0.459 | 0.465 / 0.446 | 0.488 / 0.458 | 0.469 / 0.478 |

- **The 0.5 s lead turn holds its strength in the crowd** (0.496-0.543 on every card: at
  or above 0.50 within the noise), where the deny failed (0.43-0.47). From 0.75 s on it
  loses at 5v5 too (0.42-0.51). The brake costs a little at 5v5 (0.45-0.49): out of it.
- Energy v turn 5v5 is unchanged by the lead turn (within 0.03 of r9's on every card).

## The sense: the post-merge choice (code)

A fourth maneuver, from the research list (nose-to-nose against nose-to-tail at the merge):
**sense** (`sense`, `sense_one_circle`, `sense_in_gun_range`; later `sense_in_lift_ratio`,
`sense_mirror_only`). With the target behind the ship's 3-9 line within
`sense_in_gun_range` and turning (its observed nose rate above 0.05 rad/s), turn at full
stick in its rotation sense (two-circle) or against it (one-circle). The lift-ratio gate
rises in the own maximum lift coefficient (`max_acceleration / corner_speed^2`) over the
target's: in the slow fight the turn radius is 1 / C_max, and the one-circle goes to the
smaller radius. Tests: `test_the_sense_turns_with_or_against_the_targets_rotation_after_the_merge`,
`test_the_sense_gates_by_the_lift_ratio_and_the_mirror`.

## 00:46-01:01 KST: round 4 (`round4/`): the sense

512 games, on both types; `gr` is `sense_in_gun_range`'s centre (2 unless named). Against
r9 in the 1v1 mirrors (JJ / YY), and the self-play mirrors (mutual share, life; JJ one /
two circle / scissors):

| Sense | frozen v r9 | D2 v r9 | cone 0.085 v r9 | cone 0.01 v r9 |
|---|---|---|---|---|
| r9 (control) | 0.475 / 0.498 | 0.519 / 0.480 | 0.491 / 0.498 | 0.504 / 0.496 |
| two-circle | 0.501 / 0.496 | 0.497 / 0.505 | 0.524 / 0.500 | 0.537 / 0.522 |
| **one-circle** | **0.645 / 0.610** | **0.585 / 0.635** | **0.547 / 0.597** | 0.514 / 0.496 |
| one-circle 1 gr | 0.624 / 0.619 | 0.570 / 0.546 | 0.498 / 0.517 | 0.490 / 0.433 |
| one-circle 4 gr | 0.666 / 0.613 | 0.585 / 0.624 | 0.530 / 0.592 | 0.509 / 0.474 |
| one-circle at weight 0.5 | 0.379 / 0.458 | 0.301 / 0.409 | 0.355 / 0.395 | 0.472 / 0.384 |
| lead 0.5 s + one-circle | 0.622 / 0.583 | 0.569 / 0.593 | 0.535 / 0.572 | 0.510 / 0.481 |

| Self-play (JJ; YY) | frozen | D2 | cone 0.085 | cone 0.01 |
|---|---|---|---|---|
| r9 | 0.81, 154 s, 0.15 / 0.81 / 0.04; 0.37, 1057 s | 0.64, 262 s, 0.15 / 0.81 / 0.04; 0.39, 273 s | 0.49, 305 s; 0.08, 348 s | 0.06, 1111 s; 0.02, 430 s |
| two-circle | 0.10, 2065 s, 0.01 / 0.99 / 0.00; 0.00, 5109 s | 0.00, 2390 s; 0.02, 929 s | 0.03, 1412 s; 0.02, 796 s | 0.00, 1641 s; 0.00, 839 s |
| one-circle | 0.87, 109 s, 0.62 / 0.05 / 0.33; 0.78, 201 s | 0.81, 208 s, 0.59 / 0.03 / 0.39; 0.74, 133 s | 0.72, 246 s, 0.58 / 0.02 / 0.40; 0.54, 181 s | 0.12, 994 s, 0.46 / 0.01 / 0.53; 0.04, 760 s |

- **The one-circle choice after the merge beats r9 in both mirrors on three cards**
  (0.55-0.67), the largest edge of any maneuver tried, and it is even on cone 0.01. r9 turns
  the short way round after a pass, which is the same sense as the bandit's about 80% of
  the time (the two-circle); a ship that turns the other way puts both on one circle, where
  it has the bandit's tail rather than its nose as the pair comes round.
- **In self-play it ends the side-by-side stalemate**: no two-circle fights are left
  (side by side 0.00), the scissors share rises to 0.33-0.53, and the duels shorten (YY
  frozen 1057 to 201 s, D2 273 to 133 s; JJ D2 262 to 208 s, cone 0.085 305 to 246 s).
  But the one-circle flow meets nose to nose every lap, so **the mutual head-on share of
  the kills rises** (D2 JJ 0.64 to 0.81, YY 0.39 to 0.74). The near-immune cone 0.01 stays a
  stalemate (994 s; the scissors' head-on passes do not kill there).
- **The two-circle sense is a stalemate** (life 1400-5100 s): both ships held to the same
  sense never meet nose to tail. The weight must be all or nothing (0.5 dithers between
  the sides and loses: 0.30-0.47).
- **It moves the type balance**: on both types the energy v turn 1v1 falls to 0.01-0.03 on
  every card (D2 0.49 to 0.03): the one-circle goes to the smaller turn radius, and the
  Jujak's maximum lift coefficient is 1.34 times the Yongma's (radius 176 against 235 m).
  Round 5 gates it by that ratio and by the mirror.

**Round 4's 5v5** undoes it: against r9 in the crowd the one-circle sense loses on every
card (frozen 0.23-0.33, D2 0.40-0.44, cone 0.085 0.39-0.50, cone 0.01 0.42-0.47); so does
the two-circle sense (0.34-0.51). Turning away from the target after the pass, with other
enemies about, turns the ship into them. And the sense at weight 0.5 (which loses at 1v1)
**wins the crowd big** (0.55-0.69 on every card): it mixes full stick to the one-circle
side with the combat layer's stick, so a ship whose target has passed behind it while
turning flies nearly straight (the two sides' draws cancel tick by tick): an extension
away from a turning bandit. Round 6 separates the two: a crowd gate (`sense_in_crowd`,
the other known enemies within `sense_crowd_in_gun_range`) gives the duel to the
one-circle and the crowd to `crowd_sense` (a set stick and a boost).

## 00:57- KST: round 5 (`round5/`): gating the one-circle by type

512 games. Against r9 in the 1v1 mirrors (JJ / YY) and energy v turn 1v1 (Yongma's score):

| Candidate | frozen v r9; E v T | D2 v r9; E v T | cone 0.085 v r9; E v T | cone 0.01 v r9; E v T |
|---|---|---|---|---|
| r9 | 0.496 / 0.517; 0.01 | 0.488 / 0.482; 0.50 | 0.506 / 0.481; 0.54 | 0.474 / 0.493; 0.52 |
| lead 0.5 s | 0.570 / 0.507; 0.01 | 0.521 / 0.499; 0.56 | 0.538 / 0.494; 0.62 | 0.526 / 0.505; 0.58 |
| one-circle (4 gr) | 0.673 / 0.630; 0.01 | 0.585 / 0.641; 0.02 | 0.529 / 0.601; 0.02 | 0.500 / 0.494; 0.02 |
| **one-circle, mirror only** | 0.614 / 0.604; 0.01 | **0.571 / 0.615; 0.53** | 0.515 / 0.590; 0.53 | 0.512 / 0.489; 0.52 |
| one-circle, lift ratio > 0.95 | 0.640 / 0.607; 0.06 | 0.608 / 0.604; 0.14 | 0.521 / 0.586; 0.27 | 0.494 / 0.484; 0.41 |
| lead + one-circle mirror | 0.660 / 0.555; 0.01 | 0.560 / 0.609; 0.55 | 0.521 / 0.554; 0.63 | 0.500 / 0.484; 0.60 |
| Jujak one-circle, Yongma lead | 0.641 / 0.509; 0.04 | 0.559 / 0.504; 0.18 | 0.532 / 0.509; 0.30 | 0.532 / 0.483; 0.42 |

- **The mirror-only gate keeps the type balance** (E v T within 0.03 of r9's on every card)
  and the mirror edge (D2 0.57 / 0.62).
- **The lift-ratio gate does not**: the Yongma no longer one-circles a Jujak, but the Jujak
  one-circling the Yongma still takes the duel (D2 0.50 to 0.14). One-circle is the turn
  fighter's move (the archetype), and on a card balanced for r9 it is too strong. Gating it
  to the mirror is the choice that leaves the cards' balance alone.

Round 5's 5v5 against r9 (mixed / reversed): the type gates do not save the crowd. One-circle
mirror-only 0.35-0.40 (frozen), 0.43-0.47 (D2), 0.45-0.48 (cone 0.085), 0.47-0.48 (cone
0.01); ungated 0.21-0.45. The lead turn alone: 0.47-0.52 (frozen, D2 0.45-0.51, cones
0.46-0.52; about 0.49 averaged with round 3's). Energy v turn 2v2 / 5v5 stays near r9's
with the mirror gate (D2 0.44 / 0.56 against 0.46 / 0.54); the lift gate tips the 5v5 to
the Yongma on frozen (0.76) while the 1v1 goes to the Jujak.

## 01:05-01:24 KST: round 6 (`round6/`): the crowd gate; the extension in the crowd

512 games. "Duel" = the one-circle sense, mirror only, 4 gun ranges, open only with no other
known enemy within 3 own gun ranges (`sense_in_crowd` [0.5, 0.1]); "crowd straight boost" =
the crowd sense: with the target passed behind the 3-9 line and turning and another enemy
near, fly straight (`crowd_sense_stick` 0) and boost. Against r9: 1v1 JJ / YY, then 5v5
mixed / reversed; energy v turn 5v5 and its life:

| Candidate | frozen | D2 | cone 0.085 | cone 0.01 |
|---|---|---|---|---|
| r9 | 0.485 / 0.501; 0.481 / 0.489; 0.50, 110 s | 0.515 / 0.503; 0.475 / 0.516; 0.53, 102 s | 0.503 / 0.495; 0.493 / 0.478; 0.66, 92 s | 0.501 / 0.512; 0.495 / 0.494; 0.65, 93 s |
| duel | 0.645 / 0.630; 0.461 / 0.526; 0.53 | 0.573 / 0.598; 0.505 / 0.509; 0.57 | 0.519 / 0.611; 0.510 / 0.479; 0.68 | 0.519 / 0.485; 0.500 / 0.489; 0.68 |
| **duel + crowd straight boost** | 0.683 / 0.597; **0.724 / 0.796**; 0.35, 148 s | 0.575 / 0.615; **0.772 / 0.832**; 0.56, 122 s | 0.508 / 0.571; **0.808 / 0.812**; 0.67, 109 s | 0.503 / 0.476; **0.794 / 0.822**; 0.67, 111 s |
| duel + crowd straight (no boost) | 0.649 / 0.630; 0.714 / 0.819 | 0.586 / 0.607; 0.771 / 0.803 | 0.524 / 0.579; 0.793 / 0.801 | 0.521 / 0.470; 0.762 / 0.803 |
| duel, crowd 5 gr, straight boost | 0.658 / 0.606; 0.699 / 0.801 | 0.564 / 0.628; 0.784 / 0.804 | 0.528 / 0.559; 0.799 / 0.800 | 0.502 / 0.493; 0.825 / 0.821 |
| duel + crowd half one-circle | 0.640 / 0.647; 0.486 / 0.616 | 0.561 / 0.604; 0.598 / 0.604 | 0.524 / 0.618; 0.618 / 0.630 | 0.535 / 0.467; 0.610 / 0.613 |
| crowd straight boost only | 0.490 / 0.502; 0.720 / 0.832; 0.34, 145 s | 0.500 / 0.507; 0.804 / 0.814; 0.54, 123 s | 0.495 / 0.509; 0.782 / 0.797; 0.67, 108 s | 0.506 / 0.510; 0.791 / 0.818; 0.67, 112 s |
| lead + duel + crowd straight boost | 0.646 / 0.562; 0.692 / 0.785 | 0.556 / 0.603; 0.748 / 0.854 | 0.511 / 0.568; 0.810 / 0.821 | 0.526 / 0.476; 0.771 / 0.813 |

- **The crowd gate gives the duel its edge back without the crowd's loss**: the duel arm
  alone is even at 5v5 (0.46-0.53) and keeps 0.52-0.65 at 1v1.
- **The extension in the crowd is the strongest maneuver found: 0.70-0.85 against r9 at
  5v5 on every card**, by itself (the duel arm adds nothing at 5v5). r9 turns after any
  target that passes it, and in a crowd a slow turning fight is where it is shot by the
  others; the extension keeps the ship's speed and leaves the furball, to come back on a
  new target with its speed. The boost barely matters (straight alone 0.71-0.82). Round
  4's sense at weight 0.5 was a weaker version of it (its 50/50 stick draws averaged to
  nearly straight).
- **Costs**: on the frozen card the crowd extension moves energy v turn 5v5 from 0.50 to
  0.34-0.35 (the Jujaks extend better? to be checked) and lengthens the 5v5 life (110 to
  145-148 s; D2 102 to 122 s, cones 92 to 108-111 s). On D2 and the cones energy v turn
  5v5 is unchanged.

## 01:24-01:47 KST: confirm (`confirm/`): 1024 games, 1v1 / 2v2 / 5v5

1024 games a job (SE ±0.015 at 1v1, ±0.016 at 2v2 and 5v5). Maneuvers on both types over
r9: "lead" the 0.5 s lead turn; "crowd extend" the crowd sense (straight and boost, the
target passed behind and turning, another enemy within 3 gun ranges); "duel one-circle" the
one-circle sense, mirror only, alone; "p0.7" the one-circle drawn with probability 0.7 at
each merge. `ctable.py confirm/sizes.json` prints all three tables.

**Strength and balance**: against r9 (1v1 JJ / YY; 2v2 mixed / reversed; 5v5 mixed /
reversed), and energy v turn in self-play (1v1 / 2v2 / 5v5):

| Card: candidate | v r9 1v1 | v r9 2v2 | v r9 5v5 | E v T |
|---|---|---|---|---|
| frozen: r9 | 0.494 / 0.499 | 0.500 / 0.504 | 0.512 / 0.515 | 0.01 / 0.39 / 0.53 |
| frozen: lead | 0.562 / 0.507 | 0.484 / 0.486 | 0.489 / 0.498 | 0.00 / 0.40 / 0.53 |
| frozen: crowd extend | 0.498 / 0.502 | 0.527 / 0.511 | 0.717 / 0.792 | 0.00 / 0.42 / 0.37 |
| **frozen: duel one-circle + crowd extend** | **0.656 / 0.609** | 0.523 / 0.507 | **0.699 / 0.834** | 0.00 / 0.41 / 0.34 |
| frozen: duel + crowd + lead | 0.636 / 0.580 | 0.533 / 0.523 | 0.710 / 0.826 | 0.00 / 0.45 / 0.34 |
| frozen: duel p0.7 + crowd + lead | 0.569 / 0.507 | 0.467 / 0.473 | 0.685 / 0.779 | 0.01 / 0.43 / 0.33 |
| D2: r9 | 0.499 / 0.508 | 0.514 / 0.495 | 0.497 / 0.507 | 0.50 / 0.44 / 0.56 |
| D2: lead | 0.527 / 0.506 | 0.485 / 0.500 | 0.463 / 0.469 | 0.55 / 0.45 / 0.53 |
| D2: crowd extend | 0.504 / 0.493 | 0.507 / 0.536 | 0.760 / 0.816 | 0.49 / 0.45 / 0.53 |
| **D2: duel one-circle + crowd extend** | **0.580 / 0.613** | 0.518 / 0.492 | **0.753 / 0.811** | 0.49 / 0.45 / 0.52 |
| D2: duel + crowd + lead | 0.560 / 0.598 | 0.534 / 0.496 | 0.770 / 0.824 | 0.56 / 0.43 / 0.53 |
| D2: duel p0.7 + crowd + lead | 0.528 / 0.504 | 0.499 / 0.497 | 0.777 / 0.825 | 0.56 / 0.43 / 0.51 |
| cone 0.085: r9 | 0.483 / 0.494 | 0.475 / 0.497 | 0.484 / 0.498 | 0.53 / 0.52 / 0.68 |
| cone 0.085: lead | 0.520 / 0.492 | 0.457 / 0.504 | 0.495 / 0.503 | 0.60 / 0.54 / 0.66 |
| cone 0.085: crowd extend | 0.487 / 0.504 | 0.508 / 0.519 | 0.760 / 0.813 | 0.54 / 0.49 / 0.68 |
| **cone 0.085: duel one-circle + crowd extend** | **0.524 / 0.598** | 0.531 / 0.531 | **0.792 / 0.819** | 0.53 / 0.50 / 0.69 |
| cone 0.085: duel + crowd + lead | 0.522 / 0.555 | 0.536 / 0.550 | 0.777 / 0.788 | 0.60 / 0.52 / 0.66 |
| cone 0.01: r9 | 0.507 / 0.516 | 0.488 / 0.473 | 0.481 / 0.486 | 0.53 / 0.48 / 0.65 |
| cone 0.01: lead | 0.523 / 0.500 | 0.512 / 0.504 | 0.477 / 0.504 | 0.60 / 0.49 / 0.65 |
| cone 0.01: crowd extend | 0.489 / 0.506 | 0.535 / 0.535 | 0.778 / 0.815 | 0.52 / 0.48 / 0.64 |
| **cone 0.01: duel one-circle + crowd extend** | 0.523 / 0.475 | 0.528 / 0.519 | **0.778 / 0.843** | 0.53 / 0.52 / 0.66 |
| cone 0.01: duel + crowd + lead | 0.511 / 0.489 | 0.517 / 0.524 | 0.777 / 0.815 | 0.61 / 0.50 / 0.66 |

**The duels** (self-play 1v1: mutual head-on kill share, life, merges a kill mean / median
/ p90; one / two circle / scissors; side by side; post-merge same / opposite / extend):

| Card: candidate | JJ | YY |
|---|---|---|
| frozen: r9 | 0.79, 153 s, 2.2/2/4; 0.15/0.81/0.04; 0.63; 0.78/0.22/0.00 | 0.35, 1053 s, 3.5/3/8; 0.04/0.92/0.03; 0.35; 0.92/0.08/0.00 |
| frozen: duel + crowd | 0.86, 110 s, 2.3/2/5; 0.63/0.05/0.32; 0.00; 0.00/0.97/0.03 | 0.78, 199 s, 4.0/5/8; 0.66/0.04/0.30; 0.00; 0.00/0.99/0.01 |
| frozen: p0.7 + crowd + lead | 0.82, 130 s, 2.3/2/6; 0.56/0.29/0.15; 0.39; 0.24/0.71/0.05 | 0.63, 393 s, 4.2/3/11; 0.51/0.28/0.21; 0.31; 0.22/0.77/0.01 |
| D2: r9 | 0.65, 257 s, 3.3/3/7; 0.15/0.81/0.04; 0.75; 0.80/0.20/0.00 | 0.37, 275 s, 2.1/2/4; 0.09/0.88/0.03; 0.43; 0.84/0.16/0.00 |
| D2: lead | 0.60, 282 s, 3.5/3/7; 0.02/0.89/0.10; 0.70; 0.95/0.05/0.00 | 0.25, 373 s, 2.0/2/4; 0.01/0.94/0.05; 0.36 |
| D2: duel + crowd | 0.84, 205 s, 4.3/5/8; 0.60/0.02/0.38; 0.00; 0.00/0.99/0.01 | 0.74, 133 s, 2.5/3/4; 0.60/0.05/0.36; 0.00 |
| D2: p0.7 + crowd + lead | 0.75, 284 s, 4.3/4/10; 0.56/0.29/0.15; 0.37; 0.22/0.77/0.02 | 0.61, 180 s, 2.6/2/5; 0.50/0.32/0.18; 0.33 |
| cone 0.085: r9 | 0.49, 299 s, 3.7/2/9; 0.15/0.81/0.04; 0.84 | 0.07, 379 s, 1.9/2/4; 0.08/0.89/0.03; 0.43 |
| cone 0.085: lead | 0.32, 416 s, 3.3/2/8; 0.02/0.89/0.09; 0.79 | 0.06, 370 s, 1.7/2/4 |
| cone 0.085: duel + crowd | 0.72, 247 s, 4.7/3/11; 0.58/0.02/0.40; 0.00 | 0.53, 180 s, 3.6/3/7; 0.69/0.04/0.27; 0.00 |
| cone 0.085: p0.7 + crowd + lead | 0.53, 383 s, 4.8/3/13; 0.54/0.29/0.17; 0.37 | 0.32, 257 s, 3.2/3/7 |
| cone 0.01: r9 | 0.05, 1141 s, 1.7/1/6; 0.14/0.82/0.05; 0.85 | 0.01, 477 s, 2.0/2/4 |
| cone 0.01: duel + crowd | 0.14, 1059 s, 1.9/1/7; 0.46/0.01/0.53; 0.00 | 0.06, 741 s, 2.1/1/5 |

**The crowd** (self-play energy v turn 5v5: life, kills a minute a ship; post-merge same /
opposite / extend, Yongma's and Jujak's):

| Card | r9 | duel one-circle + crowd extend |
|---|---|---|
| frozen | 109 s, 0.50; 0.64/0.36/0.00 | 146 s, 0.38; 0.11/0.10/0.48 (Y), 0.31 (J) |
| D2 | 102 s, 0.53; 0.66/0.34/0.00 | 124 s, 0.45; 0.13/0.10/0.42 (Y), 0.35 (J) |
| cone 0.085 | 92 s, 0.59; 0.67/0.33/0.00 | 109 s, 0.51; 0.13/0.10/0.41 (Y), 0.36 (J) |
| cone 0.01 | 93 s, 0.58; 0.67/0.33/0.00 | 112 s, 0.49; 0.13/0.11/0.41 (Y), 0.36 (J) |

- **Confirmed: "duel one-circle + crowd extend" (`champion.json`, r9m) beats r9 at 1v1 and
  5v5 on the frozen card, D2 and cone 0.085** (1v1 0.52-0.66, 5v5 0.70-0.83), even at 2v2
  (0.49-0.53). On cone 0.01 the 1v1 is even (0.523 / 0.475: the Yongma mirror 1.6 SE under)
  and the 5v5 0.78-0.84.
- It keeps D2's and the cones' type balance at every size (D2 0.49 / 0.45 / 0.52 against
  r9's 0.50 / 0.44 / 0.56). **On the frozen card the 5v5 goes to the Jujak** (0.53 to 0.34).
- **The crowd gets more decisions**: extensions after a merge go from none to 0.31-0.48 of
  the choices; the 5v5 life lengthens 18-37 s (still inside 45-300 s), the pace falls about
  15%.
- **The duels**: the side-by-side stalemate is gone (0.00), the scissors appear
  (0.30-0.40), the Yongma mirror shortens (frozen 1053 to 199 s, D2 275 to 133 s), the D2
  and cone 0.085 Jujak mirrors too (257 to 205 s, 299 to 247 s), with more merges a kill
  (3.3 to 4.3 on D2, spread median 5, p90 8). **But the mutual head-on share rises**
  (D2 JJ 0.65 to 0.84): the one-circle meets nose to nose every lap, so the head-on is the
  fight's shot. The p0.7 draw spreads the post-merge choice (0.22 / 0.77 instead of 1.00
  opposite) and lowers the joust (D2 0.75, cone 0.085 0.53) at most of the 1v1 edge (D2
  0.528 / 0.504).
- **The lead turn on top adds nothing** at 5v5 and little at 1v1, and moves D2's and the
  cones' energy v turn 1v1 toward the Yongma (0.50 to 0.55-0.61). Alone it beats r9 in the
  Jujak mirror (0.52-0.56) and is even elsewhere.

## 01:46- KST: round 7 (`round7/`): tuning the crowd extension; 2v2, 5v5, 10v10

512 games. The crowd extension alone (no duel arm) with one parameter moved from the base
(another enemy within 3 gun ranges, the target within 4, straight, boost, weight 1).
Against r9 (2v2; 5v5, mixed / reversed) and energy v turn 5v5 with its life:

| Variant | frozen 2v2; 5v5; E v T | D2 | cone 0.085 | cone 0.01 |
|---|---|---|---|---|
| r9 | 0.50 / 0.52; 0.48 / 0.52; 0.50, 109 s | 0.48 / 0.47; 0.50 / 0.52; 0.57, 103 s | 0.49 / 0.48; 0.48 / 0.48; 0.70, 92 s | 0.52 / 0.49; 0.49 / 0.47; 0.68, 94 s |
| base | 0.50 / 0.52; 0.71 / 0.80; 0.38, 147 s | 0.51 / 0.52; 0.77 / 0.80; 0.55, 123 s | 0.56 / 0.48; 0.77 / 0.82; 0.67, 107 s | 0.54 / 0.50; 0.75 / 0.79; 0.64, 112 s |
| crowd 2 gr | 0.52 / 0.54; 0.76 / 0.80; 0.41, 137 s | 0.52 / 0.52; 0.73 / 0.79; 0.54 | 0.53 / 0.53; 0.74 / 0.76; 0.65 | 0.51 / 0.52; 0.75 / 0.79; 0.66 |
| crowd 8 gr | 0.49 / 0.51; 0.70 / 0.80; 0.27, 165 s | 0.51 / 0.52; 0.79 / 0.80; 0.53, 137 s | 0.53 / 0.53; 0.80 / 0.82; 0.67, 118 s | 0.54 / 0.50; 0.79 / 0.85; 0.66, 121 s |
| two others | 0.52 / 0.49; 0.68 / 0.70 | 0.53 / 0.46; 0.64 / 0.67 | 0.49 / 0.50; 0.61 / 0.61 | 0.49 / 0.48; 0.66 / 0.64 |
| target 2 gr | 0.46 / 0.50; 0.74 / 0.79 | 0.50 / 0.53; 0.76 / 0.80 | 0.49 / 0.51; 0.74 / 0.81 | 0.51 / 0.54; 0.77 / 0.81 |
| target 8 gr | 0.51 / 0.47; 0.69 / 0.84 | 0.51 / 0.49; 0.75 / 0.82 | 0.52 / 0.50; 0.78 / 0.81 | 0.50 / 0.52; 0.78 / 0.81 |
| stick 0.4 (not straight) | 0.36 / 0.37; 0.46 / 0.54 | 0.45 / 0.40; 0.64 / 0.61 | 0.44 / 0.48; 0.61 / 0.61 | 0.41 / 0.43; 0.59 / 0.63 |
| weight 0.7 | 0.49 / 0.50; 0.70 / 0.78 | 0.47 / 0.53; 0.72 / 0.72 | 0.51 / 0.47; 0.71 / 0.73 | 0.54 / 0.50; 0.75 / 0.71 |

- **The extension is robust**: the crowd's radius (2-8 gun ranges) and the target's range
  (2-8) barely move it (0.70-0.85 at 5v5). What matters is **flying straight**: at 0.4 of
  full stick toward the one-circle side it loses at 2v2 (0.36-0.48) and keeps only
  0.46-0.64 at 5v5. Waiting for two other enemies (0.61-0.70) or a weight of 0.7
  (0.70-0.78) give some of it away. The base setting stays.
- A wider crowd radius lengthens the 5v5 (8 gun ranges: frozen 165 s) and on the frozen card
  tips energy v turn further to the Jujak (0.27).

10v10 (512 games against r9, 256 energy v turn; the arena by the density rule): the
extension grows with the crowd. Against r9 (mixed / reversed) and energy v turn, life:

| Variant | frozen | D2 | cone 0.085 | cone 0.01 |
|---|---|---|---|---|
| r9 | 0.53 / 0.50; 0.48, 126 s | 0.53 / 0.50; 0.42, 117 s | 0.47 / 0.48; 0.61, 104 s | 0.51 / 0.50; 0.61, 104 s |
| base | 0.86 / 0.89; 0.22, 168 s | 0.89 / 0.89; 0.43, 147 s | 0.89 / 0.93; 0.62, 126 s | 0.89 / 0.92; 0.58, 129 s |
| crowd 8 gr | 0.89 / 0.91; 0.21, 186 s | 0.92 / 0.94; 0.47, 161 s | 0.92 / 0.93; 0.69, 136 s | 0.95 / 0.95; 0.59, 142 s |
| stick 0.4 | 0.59 / 0.59 | 0.67 / 0.68 | 0.70 / 0.73 | 0.73 / 0.71 |

The type balance holds at 10v10 on D2 and the cones (within 0.07 of r9's); on the frozen
card the Jujak takes the crowd (0.48 to 0.22).

## 02:10-02:24 KST: robustness (`robust/`): r9m against the league

1024 games. r9 and r9m (the confirmed champion: duel one-circle + crowd extension) against
the league's r9, r7, r4, `hand` and `off` (1v1 JJ / YY; 5v5 mixed / reversed):

| Card: player | r9 | r7 | r4 | hand | off |
|---|---|---|---|---|---|
| frozen: r9 | 0.51 / 0.50; 0.50 / 0.48 | 0.57 / 0.53; 0.55 / 0.51 | 0.56 / 0.53; 0.55 / 0.55 | 0.48 / 0.51; 0.51 / 0.52 | 0.49 / 0.50; 0.53 / 0.49 |
| frozen: r9m | 0.66 / 0.61; 0.73 / 0.82 | 0.68 / 0.64; 0.72 / 0.84 | 0.67 / 0.63; 0.73 / 0.82 | 0.66 / 0.62; 0.73 / 0.82 | 0.66 / 0.60; 0.74 / 0.81 |
| D2: r9 | 0.52 / 0.51; 0.51 / 0.51 | 0.54 / 0.51; 0.57 / 0.55 | 0.52 / 0.53; 0.60 / 0.58 | 0.50 / 0.49; 0.52 / 0.50 | 0.50 / 0.50; 0.48 / 0.49 |
| D2: r9m | 0.59 / 0.62; 0.76 / 0.83 | 0.58 / 0.61; 0.80 / 0.86 | 0.57 / 0.62; 0.84 / 0.86 | 0.56 / 0.62; 0.77 / 0.81 | 0.57 / 0.60; 0.76 / 0.83 |
| cone 0.085: r9 | 0.48 / 0.51; 0.51 / 0.50 | 0.52 / 0.50; 0.56 / 0.53 | 0.51 / 0.52; 0.58 / 0.55 | 0.49 / 0.49; 0.51 / 0.52 | 0.49 / 0.51; 0.51 / 0.50 |
| cone 0.085: r9m | 0.52 / 0.57; 0.79 / 0.81 | 0.53 / 0.59; 0.79 / 0.83 | 0.53 / 0.61; 0.81 / 0.84 | 0.53 / 0.58; 0.79 / 0.80 | 0.53 / 0.57; 0.80 / 0.81 |

**r9m beats every league player** at 1v1 (0.52-0.68) and 5v5 (0.72-0.86), on every card:
the edge is not a reply to r9 alone. (In the mirrors every league player flies the bare
combat layer, the doctrines' advantage gates being shut, so the 1v1 columns differ by noise.)
On the frozen card's energy v turn, the extension turns the crowd into duels, which the
frozen card's Jujak wins (its 1v1 is 0.01): that is why the frozen 5v5 and 10v10 tip to the
Jujak while D2, balanced at 1v1, holds.
