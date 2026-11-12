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
