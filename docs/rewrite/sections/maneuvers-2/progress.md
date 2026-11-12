# Maneuvers 2: progress

Branch `tune/maneuvers-2` (from `origin/tune/maneuvers` 0483a4be). The task: make the
post-merge choice (extend / one-circle / two-circle) depend on the situation in a way that
wins, so the choice entropy rises above r9m's at equal or better strength; and take the
mutual head-on out of r9m's one-circle duels. Hard stop 07:45 KST. Power profile:
**performance** (read 02:40). The laptop GPU (RTX 4070 laptop, 8 GB) is this job's alone;
every run a `systemd-run --user` unit with `MemoryMax=8G`, one at a time.

Strength rule: every candidate at least 0.50 against r9m at 1v1, 2v2 and 5v5, both lineup
orders. Cards: D2, D2 with the 12-degree p8 cone at f0 0.085 and damage x1.5 ("cone
0.085"), and the archetypes branch's big v small pair (confirm 1's, on hull M; its "JJ" is
the small mirror, "YY" the big one).

## Code

New doctrine fields (`scripted_doctrine.Doctrine`; all open or off by default, so r9m is
unchanged within rounding):

- **Press gates on the crowd extension**: `crowd_sense_in_energy_ratio` and
  `crowd_sense_in_shield_ratio` (falling): the crowd extension holds only below an own over
  target specific energy (speed^2 / 2 plus the store) and shield ratio. Where a gate shuts,
  the ship presses: the sense's turn (with the sense's own gates: mirror only, lift ratio).
- **Who counts as the crowd**: `sense_crowd_in_nose_angle` (falling): an other enemy counts
  only with its track within that angle of the line to the ship (a threat);
  `sense_crowd_ally_weight`: the count less that times the own ships within the radius.
- **Duel extension** (`duel_extend`): alone, extend (the crowd extension's stick and boost)
  instead of the one-circle when the energy, shield or lift ratio is below its gate
  (`duel_extend_in_energy_ratio`, `_in_shield_ratio`, `_in_lift_ratio`; shut by default).
- **Hold fire** (`hold_fire`, `hold_fire_in_stall_angle`, `hold_fire_in_gun_range`): on a
  closing approach with the target's nose on the ship inside the range, the shot's
  probability falls by the weight. The doctrine's first element on the shot.

Tests: `test_the_crowd_extension_presses_by_the_situation`,
`test_the_duel_extension_flies_straight_behind_on_the_lift_ratio`,
`test_hold_fire_withholds_the_shot_at_a_nose_to_nose_meeting`.

## Tools

`make.py` (candidates over r9m), `sizes.py` (the maneuvers section's, unchanged),
`table.py` (strength against a versus player at 1v1 / 2v2 / 5v5, the 5v5 self-play's
choice entropy, choice shares and net outcome a choice, the 1v1 mirrors), `launch.sh`.
H is the post-merge choice entropy in bits over the bench's four observed choices (same
sense, opposite sense, extend, turn v extender), pooled over both types; "net" is the
chooser's kills less its deaths per choice made, from the engagement that followed.

## 02:40-02:57 KST: round 1 (`round1/`): one situational gate at a time over r9m

512 games (SE about ±0.02). Against r9m: the worst of 1v1 (YY / JJ), 2v2 and 5v5 (mixed /
reversed); the 5v5 against r9m (both orders); the 5v5 self-play choice entropy H; the 1v1
mirrors' mutual head-on share and H (JJ; YY). `table.py round1/sizes.json --versus r9m`
has every column.

| Card: candidate | worst v r9m | 5v5 v r9m | 5v5 H | JJ mutual, H; YY mutual, H |
|---|---|---|---|---|
| D2: r9m | 0.488 | 0.502 / 0.498 | 1.78 | 0.83, 0.10; 0.75, 0.15 |
| D2: press energy 1.3 / 1.0 | 0.31 / 0.24 | 0.31 / 0.33; 0.30 / 0.24 | 1.98 / 1.96 | unchanged |
| D2: press shield 1.5 / 1.0 | 0.39 / 0.27 | 0.39 / 0.42; 0.35 / 0.27 | 1.96 / 1.95 | unchanged |
| D2: threat 0.8 / 0.4 rad | 0.36 / 0.31 | 0.36 / 0.36; 0.37 / 0.31 | 1.93 / 1.79 | unchanged |
| D2: allies 1 | 0.28 | 0.32 / 0.28 | 1.98 | unchanged |
| **D2: duel extend shield 0.6** | **0.519** | **0.567 / 0.555** | 1.73 | **0.57, 0.56; 0.56, 1.07** |
| D2: duel extend energy 0.8 | 0.408 (JJ 1v1) | 0.571 / 0.563 | 1.66 | 0.70, 0.70; 0.64, 1.40 |
| D2: duel extend lift 0.95, any type | 0.351 (YY 1v1) | 0.531 / 0.606 | 1.55 | 0.83, 0.01; 0.73, 0.00 |
| D2: hold fire 1 / 0.5 gr | 0.09 / 0.10 (JJ 1v1) | 0.42 / 0.41; 0.46 / 0.44 | 1.77 | 0.13-0.14 at 1163-1209 s |
| cone 0.085: r9m | 0.481 | 0.501 / 0.518 | 1.80 | 0.72, 0.09; 0.53, 0.09 |
| cone 0.085: press / threat / allies (6 rows) | 0.25-0.42 | 0.25-0.43 | 1.80-1.98 | unchanged |
| **cone 0.085: duel extend shield 0.6** | 0.485 (JJ 1v1) | **0.546 / 0.540** | 1.76 | **0.46, 0.52; 0.34, 0.99** |
| cone 0.085: duel extend energy 0.8 | 0.454 (JJ 1v1) | 0.578 / 0.582 | 1.66 | 0.50, 0.77; 0.39, 1.27 |
| cone 0.085: hold fire 1 / 0.5 gr | 0.19 / 0.17 | 0.43 / 0.48; 0.50 / 0.48 | 1.78 | 0.13-0.16 at 941-997 s |
| big v small: r9m | 0.485 | 0.491 / 0.505 | 1.82 | 0.64, 0.12; 0.52, 0.15 |
| big v small: press / threat / allies (6 rows) | 0.24-0.42 | 0.24-0.43 | 1.76-1.97 | unchanged |
| **big v small: duel extend shield 0.6** | **0.532** | **0.546 / 0.554** | 1.77 | **0.33, 0.81; 0.31, 0.90** |
| big v small: duel extend energy 0.8 | 0.479 | 0.588 / 0.598 | 1.71 | 0.47, 1.20; 0.35, 1.34 |
| big v small: hold fire 1 / 0.5 gr | 0.21 / 0.18 | 0.41 / 0.43; 0.47 / 0.48 | 1.81 | 0.11-0.14 at 498-739 s |

- **r9m's narrow choice is in the duel, not the crowd.** Its 1v1 mirrors make one choice
  (H 0.09-0.15: the one-circle, "opposite sense", every time); its 5v5 already spreads over
  the four observed choices (H 1.78-1.82: same 0.13, opposite 0.10, extend 0.38, turn v
  extender 0.38). (The archetypes branch's 0.1 bits were its 1v1 mirrors.)
- **Every crowd press loses the crowd** (0.24-0.43 against r9m at 5v5 on every card): turning
  back into a crowd with more energy, more shield, fewer threats pointed at the ship or an
  ally near is still the turning fight the extension wins against. They raise the 5v5 H
  (to 1.9-2.0 bits) by making the losing choice more often. The extension is right in the
  crowd whatever the energy, shield or threat.
- **The duel extension on the shield ratio (hurt: extend) is the first situational choice
  that wins**: alone, with the own shield below 0.6 of the target's, extend instead of the
  one-circle. It beats r9m at 5v5 (0.54-0.57) and 2v2 (0.51-0.55), holds 1v1 on D2 and big
  v small (0.52-0.58; cone 0.085's Jujak mirror 0.485, -1 SE), and **it is item 2's fix
  too**: the mutual head-on share falls (D2 JJ 0.83 to 0.57, cone 0.72 to 0.46, big v small
  0.64 to 0.33) at about the same mirror life, and the duel H rises to 0.5-1.1 bits.
- The energy-gated duel extension (below 0.8 of the target's specific energy) spreads the
  duel choice further (H 0.7-1.4) and wins the crowd more (0.56-0.60) but loses the Jujak
  mirror (0.41-0.48). The lift-gated one (the worse turner extends, the better one
  one-circles across types) loses the Yongma mirror (0.31-0.35).
- **Hold fire fails**: not shooting the nose-to-nose meeting loses it to a ship that shoots
  (1v1 0.09-0.37) and makes the self-play mirrors 500-1200 s stalemates.

## 02:59-03:14 KST: round 2 (`round2/`): the duel extension's shield gate

512 games. "DE shield x": alone, extend when the own shield is below x of the target's
(scale 0.1 unless "soft", 0.3). Against r9m: 1v1 YY / JJ, 2v2, 5v5 (mixed / reversed);
against r9 at 5v5; the 1v1 mirrors' mutual share and H (JJ; YY); 5v5 self-play E v T:

| Card: candidate | v r9m 1v1 | v r9m 2v2 | v r9m 5v5 | v r9 5v5 | JJ; YY mutual, H | E v T 5v5 |
|---|---|---|---|---|---|---|
| D2: r9m | 0.496 / 0.493 | 0.494 / 0.497 | 0.508 / 0.484 | 0.76 / 0.83 | 0.83, 0.11; 0.73, 0.14 | 0.52 |
| **D2: DE 0.4** | **0.538 / 0.554** | 0.524 / 0.499 | 0.519 / 0.510 | 0.78 / 0.83 | **0.63, 0.43; 0.59, 0.85** | 0.57 |
| D2: DE 0.6 | 0.557 / 0.501 | 0.517 / 0.538 | 0.545 / 0.534 | 0.81 / 0.86 | 0.60, 0.56; 0.57, 1.09 | 0.57 |
| D2: DE 0.8 | 0.553 / 0.451 | 0.567 / 0.561 | 0.590 / 0.577 | 0.81 / 0.84 | 0.46, 1.20; 0.51, 1.47 | 0.56 |
| D2: DE 1.0 | 0.552 / 0.460 | 0.602 / 0.639 | 0.640 / 0.619 | 0.84 / 0.89 | 0.44, 1.59; 0.61, 1.60 | 0.62 |
| D2: DE 0.6 soft | 0.453 / 0.369 | 0.544 / 0.524 | 0.561 / 0.562 | 0.81 / 0.85 | 0.40, 0.52; 0.37, 0.88 | 0.55 |
| D2: DE 0.6 + energy 0.6 | 0.562 / 0.493 | 0.563 / 0.566 | 0.581 / 0.605 | 0.82 / 0.89 | 0.55, 0.93; 0.56, 1.56 | 0.61 |
| D2: DE 0.6, Yongma only | 0.576 / 0.478 | 0.514 / 0.543 | 0.566 / 0.521 | 0.80 / 0.85 | 0.82, 0.10; 0.55, 1.10 | 0.64 |
| cone 0.085: r9m | 0.488 / 0.497 | 0.503 / 0.482 | 0.503 / 0.519 | 0.78 / 0.81 | 0.73, 0.10; 0.53, 0.09 | 0.65 |
| **cone 0.085: DE 0.4** | **0.576 / 0.523** | 0.502 / 0.529 | 0.511 / 0.528 | 0.79 / 0.84 | **0.52, 0.41; 0.40, 0.75** | 0.68 |
| cone 0.085: DE 0.6 | 0.569 / 0.475 | 0.518 / 0.529 | 0.536 / 0.547 | 0.81 / 0.83 | 0.46, 0.54; 0.34, 1.00 | 0.68 |
| cone 0.085: DE 0.8 | 0.521 / 0.396 | 0.574 / 0.558 | 0.584 / 0.500 | 0.83 / 0.87 | 0.27, 1.00; 0.28, 1.40 | 0.68 |
| cone 0.085: DE 1.0 | 0.528 / 0.375 | 0.636 / 0.606 | 0.652 / 0.657 | 0.85 / 0.89 | 0.30, 1.58; 0.31, 1.61 | 0.73 |
| cone 0.085: DE 0.6 + energy 0.6 | 0.573 / 0.441 | 0.559 / 0.562 | 0.567 / 0.547 | 0.86 / 0.86 | 0.43, 0.89; 0.32, 1.48 | 0.74 |
| big v small: r9m | 0.511 / 0.497 | 0.505 / 0.502 | 0.487 / 0.472 | 0.82 / 0.82 | 0.62, 0.12; 0.52, 0.15 | 0.47 |
| **big v small: DE 0.4** | **0.577 / 0.594** | 0.525 / 0.499 | 0.535 / 0.521 | 0.82 / 0.84 | **0.43, 0.57; 0.40, 0.73** | 0.53 |
| big v small: DE 0.6 | 0.578 / 0.562 | 0.502 / 0.495 | 0.571 / 0.526 | 0.86 / 0.82 | 0.34, 0.81; 0.34, 0.92 | 0.51 |
| big v small: DE 0.8 | 0.533 / 0.502 | 0.566 / 0.561 | 0.582 / 0.564 | 0.85 / 0.86 | 0.26, 1.28; 0.27, 1.37 | 0.50 |
| big v small: DE 1.0 | 0.553 / 0.543 | 0.608 / 0.616 | 0.642 / 0.703 | 0.87 / 0.92 | 0.34, 1.60; 0.31, 1.60 | 0.50 |

- **A trade along the gate**: the higher the shield gate (the earlier a hurt ship leaves a
  duel), the stronger in the 2v2 and 5v5 (DE 1.0: 0.60-0.70 against r9m) and the more varied
  the duel (H up to 1.6 bits, near the 2-bit ceiling of four choices), but the Jujak mirror
  1v1 falls (D2 0.55 at 0.4 to 0.46 at 1.0; cone 0.52 to 0.38). The Jujak is the turn fighter:
  leaving its mirror's turning fight early gives it away. The Yongma mirror and big v small
  gain at every gate (0.52-0.59).
- **DE 0.4 is the gate that passes the strength rule on all three cards** (worst 0.499-0.502
  against r9m, at or above r9m everywhere within the noise) and beats the 1v1 mirrors
  (0.52-0.59), with the duel H at 0.41-0.85 bits (r9m 0.09-0.15) and the mutual head-on
  share down (D2 JJ 0.83 to 0.63, cone 0.73 to 0.52, big v small 0.62 to 0.43).
- The soft gate (scale 0.3: a hurt ship sometimes stays, a healthy one sometimes leaves)
  loses the 1v1 (0.29-0.45): the extension must follow the shield, not dither.
- The duel extension moves the self-play energy v turn 5v5 a little toward the Yongma (D2
  0.52 to 0.57, cone 0.65 to 0.68, big v small 0.47 to 0.53); the Yongma-only variant more
  (0.64).
- Round 3: per type (the Yongma's gate higher, the Jujak's at 0.4) and the frozen card.

## 03:16-03:34 KST: round 3 (`round3/`): the gate per type; the frozen card

512 games, four cards. Worst score against r9m (1v1 / 2v2 / 5v5, both orders); the 5v5
against r9m and r9; the duel H (JJ; YY); self-play energy v turn 1v1 / 2v2 / 5v5:

| Card: candidate | worst v r9m | 5v5 v r9m | 5v5 v r9 | JJ; YY H | E v T 1/2/5 |
|---|---|---|---|---|---|
| D2: r9m | 0.475 | 0.518 / 0.505 | 0.80 / 0.81 | 0.10; 0.15 | 0.50 / 0.42 / 0.58 |
| D2: DE 0.4 | 0.514 | 0.514 / 0.515 | 0.81 / 0.82 | 0.41; 0.86 | 0.43 / 0.42 / 0.58 |
| D2: DE 0.5 | 0.521 | 0.547 / 0.550 | 0.77 / 0.84 | 0.45; 0.99 | 0.38 / 0.43 / 0.49 |
| D2: DE Y 0.6 / J 0.4 | 0.529 | 0.533 / 0.541 | 0.81 / 0.83 | 0.40; 1.11 | 0.36 / 0.50 / 0.62 |
| D2: DE Y 1.0 / J 0.4 | 0.515 | 0.604 / 0.592 | 0.80 / 0.85 | 0.37; 1.60 | 0.34 / 0.55 / 0.61 |
| cone 0.085: r9m | 0.480 | 0.480 / 0.521 | 0.77 / 0.77 | 0.09; 0.09 | 0.55 / 0.50 / 0.69 |
| cone 0.085: DE 0.4 | 0.505 | 0.551 / 0.506 | 0.78 / 0.81 | 0.42; 0.72 | 0.51 / 0.52 / 0.69 |
| cone 0.085: DE 0.5 | 0.500 | 0.520 / 0.528 | 0.80 / 0.84 | 0.48; 0.88 | 0.47 / 0.47 / 0.66 |
| cone 0.085: DE Y 0.6 / J 0.4 | 0.515 | 0.532 / 0.545 | 0.82 / 0.80 | 0.42; 0.99 | 0.50 / 0.51 / 0.70 |
| cone 0.085: DE Y 1.0 / J 0.4 | 0.499 | 0.611 / 0.583 | 0.82 / 0.87 | 0.41; 1.60 | 0.43 / 0.60 / 0.77 |
| big v small: r9m | 0.467 | 0.469 / 0.491 | 0.81 / 0.83 | 0.15; 0.17 | 0.56 / 0.51 / 0.46 |
| big v small: DE 0.4 | 0.479 | 0.540 / 0.529 | 0.80 / 0.84 | 0.60; 0.71 | 0.60 / 0.52 / 0.54 |
| big v small: DE 0.5 | 0.496 | 0.520 / 0.551 | 0.81 / 0.83 | 0.65; 0.79 | 0.59 / 0.52 / 0.52 |
| big v small: DE Y 1.0 / J 0.4 | 0.553 | 0.611 / 0.588 | 0.87 / 0.88 | 0.63; 1.57 | 0.61 / 0.61 / 0.62 |
| frozen: r9m | 0.478 | 0.509 / 0.522 | 0.73 / 0.80 | 0.21; 0.08 | 0.01 / 0.44 / 0.35 |
| frozen: DE 0.4 | 0.524 | 0.550 / 0.524 | 0.72 / 0.83 | 0.77; 0.72 | 0.36 / 0.38 / 0.26 |
| frozen: DE 0.5 | 0.547 | 0.570 / 0.550 | 0.74 / 0.83 | 0.89; 0.89 | 0.38 / 0.33 / 0.25 |
| frozen: DE Y 1.0 / J 0.4 | 0.535 | 0.536 / 0.550 | 0.75 / 0.82 | 0.81; 1.59 | 0.43 / 0.44 / 0.32 |

(r9m's own row shows the noise: its worst against itself is 0.47-0.48 at 512 games.)

- **The per-type gate (the Yongma's high, the Jujak's at 0.4) holds the Jujak mirror and
  keeps the crowd gain**: DE Y 1.0 / J 0.4 beats r9m 0.58-0.61 at 5v5 on every card and its
  Yongma mirror makes the most varied choice (H 1.6), with the worst at 0.499-0.553.
- **But the duel extension moves the cross-type duel**: it applies against any type, and on
  D2 the energy v turn 1v1 falls from 0.50 to 0.34-0.43 (a hurt Yongma leaving a Jujak is
  not the Yongma's win); on the frozen card it rises from 0.01 to 0.36-0.43 (the frozen
  Jujak's duel edge shrinks), and the frozen 5v5 tips further to the Jujak (0.35 to
  0.25-0.32): **item 3 does not fall out of it**. Round 4 gates the duel extension to the
  mirror, as the duel one-circle is.

## 03:43-03:52 KST: round 4 (`round4/`): the duel extension in the mirror only

512 games, four cards. "mirror": the duel extension only against the own type
(`duel_extend_mirror_only`). Worst against r9m; the 5v5 against r9m; E v T 1v1 / 2v2 / 5v5;
the duel H (JJ; YY):

| Card: candidate | worst v r9m | 5v5 v r9m | E v T 1/2/5 | JJ; YY mutual, H |
|---|---|---|---|---|
| D2: r9m | 0.491 | 0.529 / 0.492 | 0.51 / 0.44 / 0.56 | 0.84, 0.09; 0.75, 0.15 |
| D2: DE 0.4 | 0.518 | 0.537 / 0.518 | 0.43 / 0.45 / 0.53 | 0.64, 0.43; 0.59, 0.85 |
| D2: DE 0.4 mirror | 0.511 | 0.511 / 0.519 | 0.50 / 0.43 / 0.51 | 0.66, 0.40; 0.58, 0.85 |
| D2: DE 0.5 mirror | 0.480 | 0.515 / 0.542 | 0.48 / 0.45 / 0.57 | 0.62, 0.46 |
| D2: DE Y 1.0 / J 0.4 mirror | 0.489 | 0.507 / 0.489 | 0.51 / 0.46 / 0.57 | 0.65, 0.42 |
| cone: r9m | 0.466 | 0.498 / 0.466 | 0.54 / 0.52 / 0.66 | 0.71, 0.09 |
| cone: DE 0.4 | 0.509 | 0.552 / 0.530 | 0.50 / 0.49 / 0.66 | 0.55, 0.43 |
| cone: DE 0.4 mirror | 0.499 | 0.502 / 0.536 | 0.53 / 0.53 / 0.70 | 0.53, 0.44 |
| cone: DE 0.5 mirror | 0.488 | 0.521 / 0.507 | 0.51 / 0.48 / 0.64 | 0.50, 0.48 |
| big v small: r9m | 0.475 | 0.475 / 0.488 | 0.59 / 0.55 / 0.52 | 0.62, 0.14 |
| big v small: DE 0.4 | 0.493 | 0.505 / 0.519 | 0.56 / 0.54 / 0.50 | 0.44, 0.63 |
| big v small: DE 0.4 mirror | 0.489 | 0.562 / 0.523 | 0.58 / 0.50 / 0.51 | 0.43, 0.62 |
| big v small: DE Y 1.0 / J 0.4 mirror | 0.507 | 0.547 / 0.507 | 0.60 / 0.52 / 0.52 | 0.43, 0.58 |
| frozen: r9m | 0.485 | 0.511 / 0.496 | 0.01 / 0.44 / 0.32 | 0.87, 0.22 |
| frozen: DE 0.4 | 0.535 | 0.542 / 0.552 | 0.31 / 0.40 / 0.30 | 0.78, 0.83 |
| frozen: DE 0.4 mirror | 0.474 | 0.532 / 0.512 | 0.01 / 0.42 / 0.33 | 0.79, 0.81 |
| frozen: DE 0.5 mirror | 0.504 | 0.504 / 0.517 | 0.01 / 0.46 / 0.37 | 0.76, 0.91 |

- **In the mirror only, the duel extension leaves the type balance as r9m's** (D2 E v T
  0.50 / 0.43 / 0.51 against r9m's 0.51 / 0.44 / 0.56; frozen 1v1 0.01) and keeps the duel's
  gains (the mirrors' H 0.40-0.91, the mutual share down, 1v1 mirrors 0.50-0.59 against r9m).
  Its 2v2 / 5v5 against r9m is even (0.47-0.56).
- **The crowd edge of the high Yongma gate came from the cross-type duels**: mirror-only, DE
  Y 1.0 / J 0.4 is even at 5v5 (0.49-0.55) where round 3's any-type version took 0.58-0.61.
  A hurt Yongma leaving a Jujak in a crowd wins the crowd, and loses the lone duel (D2 E v T
  1v1 0.34).

## 03:55-04:26 KST: confirm (`confirm/`): 1024 games, 1v1 / 2v2 / 5v5 (10v10 self-play)

1024 games a job (SE about ±0.015; 10v10 self-play 512). `table.py confirm/sizes.json
--versus r9m` (or `r9`) has every column.

**Strength** against r9m (1v1 YY / JJ; 2v2 and 5v5 mixed / reversed; the worst) and r9 (5v5):

| Card: candidate | v r9m 1v1 | v r9m 2v2 | v r9m 5v5 | worst | v r9 5v5 |
|---|---|---|---|---|---|
| D2: r9m | 0.489 / 0.513 | 0.506 / 0.503 | 0.479 / 0.492 | 0.479 | 0.78 / 0.83 |
| D2: DE 0.4 | 0.551 / 0.565 | 0.478 / 0.523 | 0.540 / 0.543 | 0.478 | 0.78 / 0.83 |
| D2: DE 0.4 mirror | 0.541 / 0.520 | 0.515 / 0.506 | 0.537 / 0.499 | 0.499 | 0.77 / 0.83 |
| D2: DE 0.5 mirror | 0.562 / 0.522 | 0.526 / 0.507 | 0.503 / 0.502 | 0.502 | 0.78 / 0.84 |
| **D2: DE Y 1.0 / J 0.4** | **0.601 / 0.523** | **0.556 / 0.587** | **0.588 / 0.579** | **0.523** | **0.83 / 0.85** |
| cone: r9m | 0.503 / 0.513 | 0.493 / 0.478 | 0.530 / 0.469 | 0.469 | 0.78 / 0.83 |
| cone: DE 0.4 | 0.566 / 0.532 | 0.520 / 0.545 | 0.524 / 0.524 | 0.520 | 0.80 / 0.84 |
| cone: DE 0.4 mirror | 0.571 / 0.503 | 0.492 / 0.513 | 0.504 / 0.507 | 0.492 | 0.78 / 0.81 |
| cone: DE 0.5 mirror | 0.564 / 0.499 | 0.503 / 0.514 | 0.487 / 0.525 | 0.487 | 0.78 / 0.83 |
| **cone: DE Y 1.0 / J 0.4** | **0.535 / 0.534** | **0.569 / 0.565** | **0.586 / 0.588** | **0.534** | **0.82 / 0.86** |
| big v small: r9m | 0.501 / 0.498 | 0.503 / 0.495 | 0.481 / 0.515 | 0.481 | 0.80 / 0.80 |
| big v small: DE 0.4 | 0.585 / 0.562 | 0.512 / 0.511 | 0.536 / 0.523 | 0.511 | 0.80 / 0.84 |
| big v small: DE 0.4 mirror | 0.571 / 0.575 | 0.495 / 0.519 | 0.526 / 0.485 | 0.485 | 0.81 / 0.82 |
| big v small: DE 0.5 mirror | 0.564 / 0.557 | 0.504 / 0.479 | 0.519 / 0.499 | 0.479 | 0.81 / 0.81 |
| **big v small: DE Y 1.0 / J 0.4** | **0.549 / 0.550** | **0.544 / 0.559** | **0.621 / 0.563** | **0.544** | **0.84 / 0.86** |
| frozen: r9m | 0.507 / 0.507 | 0.486 / 0.509 | 0.517 / 0.521 | 0.486 | 0.75 / 0.81 |
| frozen: DE 0.4 | 0.565 / 0.557 | 0.533 / 0.561 | 0.515 / 0.539 | 0.515 | 0.73 / 0.83 |
| frozen: DE 0.4 mirror | 0.552 / 0.557 | 0.512 / 0.515 | 0.498 / 0.510 | 0.498 | 0.76 / 0.83 |
| frozen: DE 0.5 mirror | 0.565 / 0.548 | 0.499 / 0.499 | 0.506 / 0.527 | 0.499 | 0.70 / 0.83 |
| **frozen: DE Y 1.0 / J 0.4** | **0.596 / 0.560** | **0.552 / 0.566** | **0.570 / 0.600** | **0.552** | **0.76 / 0.84** |

**The duels and the balance**: the 1v1 mirrors (mutual head-on share, life, merges a kill,
H; JJ then YY), the 5v5 mirrors' H (JJ; YY), self-play energy v turn by size (1 / 2 / 5 /
10v10):

| Card: candidate | JJ 1v1 | YY 1v1 | 5v5 mirrors H | E v T 1/2/5/10 |
|---|---|---|---|---|
| D2: r9m | 0.85, 210 s, 4.3, 0.10 | 0.75, 133 s, 2.4, 0.14 | 1.78; 1.80 | 0.51 / 0.45 / 0.52 / 0.44 |
| D2: DE 0.4 | 0.65, 224 s, 4.0, 0.41 | 0.62, 152 s, 2.2, 0.83 | 1.76; 1.77 | 0.42 / 0.43 / 0.56 / 0.43 |
| D2: DE 0.4 mirror | 0.66, 223 s, 4.1, 0.41 | 0.60, 153 s, 2.3, 0.86 | 1.76; 1.78 | 0.49 / 0.44 / 0.56 / 0.42 |
| D2: DE 0.5 mirror | 0.64, 224 s, 3.9, 0.44 | 0.58, 151 s, 2.2, 0.96 | 1.75; 1.77 | 0.48 / 0.44 / 0.56 / 0.42 |
| D2: DE Y 1.0 / J 0.4 | 0.66, 223 s, 4.1, 0.40 | 0.61, 185 s, 2.0, 1.59 | 1.77; 1.69 | 0.35 / 0.54 / 0.64 / 0.61 |
| cone: r9m | 0.71, 246 s, 4.6, 0.10 | 0.53, 184 s, 3.7, 0.09 | 1.76; 1.77 | 0.52 / 0.53 / 0.68 / 0.60 |
| cone: DE 0.4 | 0.54, 249 s, 4.3, 0.42 | 0.38, 192 s, 3.1, 0.72 | 1.74; 1.78 | 0.48 / 0.49 / 0.72 / 0.62 |
| cone: DE 0.4 mirror | 0.55, 253 s, 4.4, 0.42 | 0.41, 199 s, 3.1, 0.74 | 1.76; 1.78 | 0.53 / 0.47 / 0.66 / 0.60 |
| cone: DE 0.5 mirror | 0.50, 242 s, 4.1, 0.49 | 0.37, 189 s, 2.9, 0.86 | 1.74; 1.77 | 0.52 / 0.50 / 0.68 / 0.62 |
| cone: DE Y 1.0 / J 0.4 | 0.53, 255 s, 4.4, 0.41 | 0.31, 228 s, 2.2, 1.60 | 1.74; 1.73 | 0.42 / 0.62 / 0.72 / 0.69 |
| big v small: r9m | 0.63, 168 s, 3.8, 0.14 | 0.52, 134 s, 2.5, 0.15 | 1.79; 1.76 | 0.60 / 0.50 / 0.50 / 0.53 |
| big v small: DE 0.4 | 0.43, 187 s, 3.4, 0.59 | 0.39, 144 s, 2.4, 0.72 | 1.76; 1.75 | 0.59 / 0.51 / 0.50 / 0.53 |
| big v small: DE 0.4 mirror | 0.42, 191 s, 3.5, 0.60 | 0.41, 145 s, 2.4, 0.70 | 1.77; 1.75 | 0.61 / 0.51 / 0.49 / 0.50 |
| big v small: DE Y 1.0 / J 0.4 | 0.41, 187 s, 3.5, 0.62 | 0.29, 166 s, 1.8, 1.58 | 1.77; 1.70 | 0.61 / 0.61 / 0.64 / 0.60 |
| frozen: r9m | 0.86, 109 s, 2.3, 0.23 | 0.79, 198 s, 4.0, 0.08 | 1.78; 1.78 | 0.01 / 0.44 / 0.37 / 0.23 |
| frozen: DE 0.4 | 0.79, 123 s, 2.2, 0.81 | 0.59, 221 s, 3.5, 0.72 | 1.76; 1.76 | 0.32 / 0.37 / 0.28 / 0.20 |
| frozen: DE 0.4 mirror | 0.78, 122 s, 2.2, 0.78 | 0.61, 217 s, 3.6, 0.72 | 1.77; 1.76 | 0.01 / 0.44 / 0.36 / 0.21 |
| frozen: DE Y 1.0 / J 0.4 | 0.79, 121 s, 2.2, 0.80 | 0.62, 277 s, 3.0, 1.59 | 1.77; 1.69 | 0.43 / 0.43 / 0.34 / 0.22 |

(In big v small "JJ" is the small mirror and "YY" the big one; "Y" in a candidate's name is
the first slot, the big ship. The 5v5 choice shares barely move: r9m's 5v5 mirrors are
already at 1.7-1.8 bits.)

- **Only "DE Y 1.0 / J 0.4" clears the strength rule on every card at 1024 games** (worst
  0.523-0.552 against r9m; 5v5 0.56-0.62, 2v2 0.54-0.59), and it beats r9 more than r9m does
  (5v5 0.76-0.86). The mirror-only gates are r9m's equal in the crowd (worst 0.479-0.502, as
  r9m's own control row, 0.469-0.486) and better in the 1v1 mirrors (0.50-0.58).
- **The duel's choice opens up under every candidate**: 1v1 mirror H from 0.08-0.23 to
  0.40-0.96 (the Yongma mirror at a full-shield gate 1.6, near the 2-bit ceiling); the
  mutual head-on share falls in every mirror (D2 JJ 0.85 to 0.64-0.66, cone 0.71 to
  0.50-0.55, big v small 0.63 to 0.41-0.43; YY D2 0.75 to 0.58-0.62, cone 0.53 to 0.31-0.41).
  The frozen Jujak mirror moves least (0.86 to 0.78-0.79). The Yongma mirrors lengthen
  (D2 133 to 151-185 s; frozen 198 to 217-277 s) and need fewer merges a kill.
- **The cost of the strongest**: the any-type duel extension moves energy v turn, toward the
  Jujak at 1v1 (D2 0.51 to 0.35, cone 0.52 to 0.42) and toward the Yongma from 2v2 up (D2
  5v5 0.52 to 0.64, 10v10 0.44 to 0.61; cone 10v10 0.60 to 0.69; big v small 0.50-0.53 to
  0.60-0.64). On the frozen card the 1v1 improves (0.01 to 0.43) but the crowd still tips to
  the Jujak (5v5 0.34, 10v10 0.22): item 3 does not fall out of the duel extension.
- **The mirror-only gate keeps every card's balance** (within 0.04 of r9m's at every size).
- **Net outcome a choice**: in self-play the bench's per-pair outcome is symmetric for same
  and opposite sense (both ships make the choice), so only extend v chase is informative:
  the extender's engagement nets -0.01 to -0.03 a choice (it does not win the engagement
  it leaves; it wins the game by not being in the turning fight when hurt or crowded).

## 04:28-04:39 KST: round 5 (`round5/`): the gates per type for the balance

512 games. Worst against r9m (1v1 / 2v2 / 5v5); E v T 1 / 2 / 5 / 10v10; the 1v1 mirrors'
H (JJ; YY):

| Candidate | D2 worst; E v T | cone 0.085 worst; E v T | big v small worst; E v T | frozen worst; E v T | H (D2) |
|---|---|---|---|---|---|
| r9m | 0.489; 0.49/0.44/0.56/0.46 | 0.477; 0.55/0.51/0.68/0.54 | 0.456; 0.57/0.51/0.50/0.54 | 0.466; 0.00/0.44/0.36/0.21 | 0.11; 0.13 |
| DE Y 1.0 / J 0.4 | 0.531; 0.35/0.58/0.65/0.59 | 0.515; 0.40/0.64/0.75/0.71 | 0.570; 0.61/0.61/0.66/0.63 | 0.549; 0.46/0.41/0.31/0.23 | 0.38; 1.59 |
| DE Y 1.0 / J 0.6 | 0.521; 0.34/0.54/0.60/0.59 | 0.502; 0.40/0.58/0.73/0.71 | 0.562; 0.60/0.58/0.63/0.66 | 0.552; 0.49/0.35/0.22/0.13 | 0.57; 1.58 |
| DE Y 0.7 / J 0.4 | 0.509; 0.32/0.48/0.55/0.50 | 0.505; 0.45/0.54/0.71/0.63 | 0.514; 0.57/0.55/0.53/0.52 | 0.519; 0.38/0.39/0.28/0.23 | 0.40; 1.30 |
| DE Y 1.0 / J 0.4 mirror | 0.535; 0.36/0.57/0.65/0.60 | 0.511; 0.43/0.61/0.77/0.73 | 0.553; 0.65/0.66/0.62/0.63 | 0.533; 0.23/0.49/0.39/0.30 | 0.41; 1.60 |
| DE Y 0.7 / J 0.6 | 0.507; 0.30/0.48/0.54/0.47 | 0.483; 0.38/0.52/0.69/0.69 | 0.522; 0.60/0.52/0.52/0.56 | 0.544; 0.45/0.32/0.22/0.16 | 0.55; 1.30 |

- The Yongma's cross-type duel extension is what moves the balance: at 1v1 it hands the
  duel to the Jujak (D2 0.49 to 0.30-0.36, whatever the Jujak's gate), from 2v2 up it gives
  the crowd to the Yongma, the more the higher its gate (D2 5v5: 0.55 at 0.7, 0.65 at 1.0).
  The Jujak's gate barely moves it (0.4 to 0.6: the frozen crowd tips further to the Jujak).
- So the cross-type extension wins in a team game and loses in a lone duel. Round 6 gates
  it on another known enemy anywhere (`duel_extend_cross_in_others`, centre 1/2: never in a
  lone 1v1), to keep the 1v1 balance and the crowd gain.

## 04:41-04:54 KST: round 6 (`round6/`): the cross-type extension in team games only

512 games. "team": the Yongma's duel extension against a Jujak only with another known
enemy anywhere (`duel_extend_cross_in_others` [0.5, 0.1]). Worst against r9m; E v T 1 / 2 /
5 / 10v10; the 1v1 mirrors' H (JJ; YY):

| Candidate | D2 worst; E v T | cone 0.085 worst; E v T | big v small worst; E v T | frozen worst; E v T | H (D2) |
|---|---|---|---|---|---|
| r9m | 0.465; 0.52/0.44/0.57/0.42 | 0.476; 0.52/0.51/0.68/0.60 | 0.467; 0.57/0.53/0.50/0.51 | 0.492; 0.02/0.40/0.33/0.24 | 0.10; 0.12 |
| DE Y 0.7 / J 0.4 | 0.516; 0.33/0.47/0.58/0.48 | 0.505; 0.44/0.53/0.71/0.60 | 0.482; 0.58/0.54/0.58/0.56 | 0.526; 0.41/0.39/0.30/0.21 | 0.41; 1.28 |
| DE Y 0.7 team / J 0.4 | 0.525; 0.46/0.42/0.57/0.46 | 0.482; 0.51/0.52/0.69/0.64 | 0.495; 0.57/0.51/0.47/0.49 | 0.513; 0.21/0.37/0.30/0.22 | 0.39; 1.30 |
| DE Y 0.85 team / J 0.4 | 0.485; 0.50/0.40/0.55/0.48 | 0.489; 0.52/0.51/0.67/0.67 | 0.510; 0.56/0.52/0.52/0.48 | 0.503; 0.20/0.37/0.30/0.21 | 0.41; 1.53 |
| **DE Y 1.0 team / J 0.4** | **0.521; 0.49/0.43/0.58/0.56** | 0.486; 0.54/0.50/0.70/0.63 | **0.513; 0.58/0.50/0.48/0.55** | **0.528; 0.21/0.37/0.31/0.23** | 0.42; 1.61 |
| DE Y 0.7 team / J 0.4 mirror | 0.501; 0.51/0.48/0.59/0.50 | 0.483; 0.53/0.53/0.69/0.58 | 0.500; 0.57/0.51/0.52/0.52 | 0.516; 0.00/0.44/0.36/0.23 | 0.41; 1.27 |

- **The team gate gives the 1v1 balance back** (D2 E v T 1v1 0.33-0.36 to 0.46-0.50; cone
  0.40-0.45 to 0.51-0.54) and keeps the rest within about 0.05 of r9m's at every size; D2's
  10v10 moves toward even (0.42 to 0.46-0.56).
- It costs part of the crowd edge (5v5 against r9m 0.52-0.58 where round 5's ungated
  version took 0.57-0.62); the high Yongma gate (1.0) keeps the most.
- The frozen card's 1v1 rises from 0.02 to 0.20-0.21 with the Jujak's any-type gate (a
  hurt frozen Jujak leaves a Yongma); its crowd is unchanged (item 3 stays open).

## 04:56-05:16 KST: confirm 2 (`confirm2/`): the team-gated duel extension at 1024 games

1024 games (SE about ±0.015; 10v10 self-play 512). "team / team": both types' cross-type
extension gated on another known enemy anywhere. Against r9m (1v1 YY / JJ; 2v2; 5v5;
worst) and r9 (5v5):

| Card: candidate | v r9m 1v1 | 2v2 | 5v5 | worst | v r9 5v5 |
|---|---|---|---|---|---|
| D2: r9m | 0.486 / 0.502 | 0.513 / 0.523 | 0.484 / 0.480 | 0.480 | 0.76 / 0.83 |
| D2: DE Y 1.0 team / J 0.4 | 0.582 / 0.537 | 0.555 / 0.555 | 0.582 / 0.559 | 0.537 | 0.83 / 0.84 |
| **D2: DE Y 1.0 team / J 0.4 team** | **0.562 / 0.532** | **0.533 / 0.539** | **0.586 / 0.532** | **0.532** | **0.81 / 0.85** |
| D2: DE Y 0.85 team / J 0.4 | 0.518 / 0.535 | 0.511 / 0.503 | 0.545 / 0.544 | 0.503 | 0.79 / 0.86 |
| cone: r9m | 0.498 / 0.497 | 0.473 / 0.499 | 0.486 / 0.496 | 0.473 | 0.78 / 0.82 |
| cone: DE Y 1.0 team / J 0.4 | 0.538 / 0.510 | 0.522 / 0.524 | 0.576 / 0.534 | 0.510 | 0.82 / 0.85 |
| **cone: DE Y 1.0 team / J 0.4 team** | **0.555 / 0.531** | **0.534 / 0.541** | **0.591 / 0.554** | **0.531** | **0.81 / 0.84** |
| cone: DE Y 0.85 team / J 0.4 | 0.507 / 0.532 | 0.530 / 0.525 | 0.547 / 0.532 | 0.507 | 0.80 / 0.82 |
| big v small: r9m | 0.493 / 0.511 | 0.494 / 0.482 | 0.499 / 0.512 | 0.482 | 0.80 / 0.81 |
| big v small: DE Y 1.0 team / J 0.4 | 0.585 / 0.560 | 0.516 / 0.500 | 0.581 / 0.549 | 0.500 | 0.83 / 0.83 |
| **big v small: DE Y 1.0 team / J 0.4 team** | **0.563 / 0.577** | **0.530 / 0.519** | **0.554 / 0.556** | **0.519** | **0.84 / 0.86** |
| big v small: DE Y 0.85 team / J 0.4 | 0.523 / 0.556 | 0.528 / 0.531 | 0.550 / 0.548 | 0.523 | 0.82 / 0.85 |
| frozen: r9m | 0.510 / 0.501 | 0.494 / 0.489 | 0.507 / 0.486 | 0.486 | 0.72 / 0.82 |
| frozen: DE Y 1.0 team / J 0.4 | 0.572 / 0.562 | 0.569 / 0.552 | 0.576 / 0.533 | 0.533 | 0.73 / 0.82 |
| **frozen: DE Y 1.0 team / J 0.4 team** | **0.591 / 0.571** | **0.552 / 0.519** | **0.560 / 0.517** | **0.517** | **0.75 / 0.84** |
| frozen: DE Y 0.85 team / J 0.4 | 0.521 / 0.536 | 0.530 / 0.532 | 0.535 / 0.529 | 0.521 | 0.76 / 0.84 |

The duels (1v1 mirrors: mutual head-on share, life, merges a kill, H; JJ then YY) and the
balance (self-play energy v turn 1 / 2 / 5 / 10v10):

| Card: candidate | JJ 1v1 | YY 1v1 | E v T 1/2/5/10 |
|---|---|---|---|
| D2: r9m | 0.84, 213 s, 4.3, 0.09 | 0.74, 132 s, 2.4, 0.15 | 0.50 / 0.44 / 0.53 / 0.43 |
| D2: DE Y 1.0 team / J 0.4 | 0.66, 221 s, 4.1, 0.40 | 0.62, 184 s, 2.0, 1.60 | 0.47 / 0.42 / 0.55 / 0.53 |
| **D2: team / team** | **0.66, 221 s, 4.1, 0.40** | **0.62, 183 s, 2.0, 1.59** | **0.50 / 0.44 / 0.59 / 0.50** |
| cone: r9m | 0.70, 246 s, 4.6, 0.10 | 0.56, 183 s, 3.6, 0.09 | 0.52 / 0.52 / 0.69 / 0.59 |
| **cone: team / team** | **0.52, 249 s, 4.2, 0.43** | **0.32, 232 s, 2.2, 1.60** | **0.53 / 0.51 / 0.71 / 0.68** |
| big v small: r9m | 0.62, 168 s, 3.9, 0.12 | 0.53, 133 s, 2.5, 0.15 | 0.60 / 0.51 / 0.50 / 0.47 |
| **big v small: team / team** | **0.44, 189 s, 3.5, 0.61** | **0.31, 159 s, 1.8, 1.60** | **0.57 / 0.51 / 0.53 / 0.55** |
| frozen: r9m | 0.86, 109 s, 2.3, 0.22 | 0.79, 197 s, 4.0, 0.09 | 0.01 / 0.43 / 0.36 / 0.26 |
| **frozen: team / team** | **0.78, 121 s, 2.2, 0.78** | **0.63, 276 s, 3.1, 1.58** | **0.01 / 0.45 / 0.33 / 0.18** |

- **"DE Y 1.0 team / J 0.4 team" (r9m2, `champion2.json`) passes the strength rule on every
  card at 1024 games**: worst against r9m 0.517-0.532 (about 1-2 SE above even), 5v5
  0.53-0.59, 1v1 0.53-0.59; against r9 it beats r9m's margins (5v5 0.75-0.86).
- **The type balance stays r9m's at 1v1-5v5** (within 0.06 at every size on every card;
  D2 0.50 / 0.44 / 0.59 against 0.50 / 0.44 / 0.53). At 10v10 it moves toward the Yongma
  (D2 0.43 to 0.50, big v small 0.47 to 0.55, cone 0.59 to 0.68) and the frozen card's to the
  Jujak (0.26 to 0.18).
- **The duel choice**: H 0.40-0.78 in the Jujak mirror, 1.58-1.60 in the Yongma mirror
  (r9m 0.09-0.22): the Yongma one-circles at a shield lead and extends behind, the
  Jujak extends only when well behind (below 0.4 of the target's shield).
- **The joust in the duels**: the mutual head-on share falls in every mirror (JJ D2 0.84 to
  0.66, cone 0.70 to 0.52, big v small 0.62 to 0.44, frozen 0.86 to 0.78; YY D2 0.74 to 0.62,
  cone 0.56 to 0.32, big v small 0.53 to 0.31, frozen 0.79 to 0.63) with fewer merges a kill
  in the Yongma mirror (D2 2.4 to 2.0) and a longer Yongma mirror (D2 132 to 183 s, cone 183
  to 232 s, frozen 197 to 276 s).

## 05:18-05:33 KST: robustness (`robust/`): r9m2 against the league

1024 games. r9m and r9m2 against the league (1v1 YY / JJ; 5v5 mixed / reversed):

| Card: player | r9 | r7 | r4 | hand | off |
|---|---|---|---|---|---|
| D2: r9m | 0.60 / 0.57; 0.81 / 0.81 | 0.60 / 0.58; 0.84 / 0.86 | 0.61 / 0.58; 0.81 / 0.87 | 0.61 / 0.57; 0.78 / 0.85 | 0.62 / 0.56; 0.78 / 0.82 |
| D2: r9m2 | 0.62 / 0.60; 0.81 / 0.82 | 0.61 / 0.60; 0.82 / 0.88 | 0.62 / 0.60; 0.83 / 0.88 | 0.61 / 0.61; 0.78 / 0.85 | 0.61 / 0.58; 0.83 / 0.85 |
| cone 0.085: r9m | 0.61 / 0.52; 0.81 / 0.82 | 0.59 / 0.53; 0.82 / 0.83 | 0.61 / 0.54; 0.81 / 0.85 | 0.57 / 0.53; 0.79 / 0.80 | 0.59 / 0.52; 0.77 / 0.81 |
| cone 0.085: r9m2 | 0.57 / 0.53; 0.80 / 0.83 | 0.59 / 0.56; 0.84 / 0.83 | 0.57 / 0.54; 0.83 / 0.85 | 0.55 / 0.54; 0.83 / 0.82 | 0.57 / 0.53; 0.81 / 0.85 |
| big v small: r9m | 0.57 / 0.54; 0.79 / 0.84 | 0.56 / 0.59; 0.79 / 0.81 | 0.55 / 0.58; 0.82 / 0.83 | 0.58 / 0.56; 0.79 / 0.82 | 0.55 / 0.55; 0.80 / 0.83 |
| big v small: r9m2 | 0.56 / 0.64; 0.84 / 0.84 | 0.56 / 0.63; 0.84 / 0.84 | 0.59 / 0.64; 0.83 / 0.85 | 0.58 / 0.64; 0.85 / 0.85 | 0.57 / 0.64; 0.84 / 0.85 |
| frozen: r9m | 0.61 / 0.65; 0.73 / 0.80 | 0.62 / 0.68; 0.75 / 0.84 | 0.65 / 0.67; 0.69 / 0.81 | 0.61 / 0.68; 0.72 / 0.84 | 0.61 / 0.66; 0.71 / 0.80 |
| frozen: r9m2 | 0.59 / 0.72; 0.74 / 0.82 | 0.60 / 0.75; 0.77 / 0.85 | 0.61 / 0.74; 0.74 / 0.84 | 0.62 / 0.74; 0.74 / 0.82 | 0.59 / 0.73; 0.74 / 0.81 |

- **r9m2 beats every league player on every card** (1v1 0.53-0.75, 5v5 0.74-0.88), at or above
  r9m's margins except the cone card's Yongma mirror (0.55-0.59 against 0.57-0.61, about one
  SE). Its edge over r9m is not a reply to r9m alone.

## 05:35-05:36 KST: 10v10 (`ten/`): r9m2 against r9m in the big crowd

512 games (±0.021). Against r9m (mixed / reversed), r9m's own control row first; self-play
energy v turn and life:

| Card | r9m v r9m | r9m2 v r9m | E v T r9m; r9m2 |
|---|---|---|---|
| D2 | 0.520 / 0.460 | **0.558 / 0.607** | 0.43, 148 s; 0.55, 156 s |
| cone 0.085 | 0.519 / 0.544 | **0.552 / 0.592** | 0.62, 125 s; 0.68, 133 s |
| big v small | 0.518 / 0.498 | **0.559 / 0.565** | 0.54, 111 s; 0.59, 113 s |
| frozen | 0.512 / 0.479 | **0.575 / 0.606** | 0.26, 168 s; 0.21, 175 s |

- r9m2 beats r9m at 10v10 too (0.55-0.61 on every card). Its 10v10 balance moves toward
  the Yongma (D2 0.43 to 0.55, cone 0.62 to 0.68, big v small 0.54 to 0.59) and on the
  frozen card further to the Jujak (0.26 to 0.21); lives lengthen 2-8 s.

## Final state (05:40 KST)

**The best doctrine: r9m2** (`champion2.json`): r9m with the duel extension on both types,
gated on the shield ratio per type and, against the other type, on a team game:

```json
"yongma": {"duel_extend": 1.0, "duel_extend_in_shield_ratio": [1.0, 0.1],
           "duel_extend_cross_in_others": [0.5, 0.1]},
"jujak":  {"duel_extend": 1.0, "duel_extend_in_shield_ratio": [0.4, 0.1],
           "duel_extend_cross_in_others": [0.5, 0.1]}
```

After a merge, alone with the target: a Yongma behind on shield (below the target's) extends,
level or ahead it one-circles (in the mirror; the combat layer's turn otherwise); a Jujak
extends only when well behind (below 0.4 of the target's shield). Against the other type it
extends only in a team game (another known enemy anywhere), never in a lone 1v1. In a crowd
r9m's extension is unchanged.

- **Strength** (1024 games): against r9m worst 0.517-0.532 over 1v1 / 2v2 / 5v5 and both
  orders on D2, cone 0.085, big v small and frozen; 5v5 0.53-0.59, 10v10 0.55-0.61 (512).
  Against r9 5v5 0.75-0.86; against every league player 1v1 0.53-0.75, 5v5 0.74-0.88.
- **Choice entropy** (the post-merge choice, bits over the bench's four observed choices):
  the 1v1 Jujak mirror 0.40-0.78, the Yongma mirror 1.58-1.60 (r9m: 0.09-0.22). The 5v5
  self-play stays at 1.7-1.8 bits (r9m's crowd was already varied: about 0.13 same sense,
  0.10 opposite, 0.38 extend, 0.38 turn v extender).
- **Mirror shapes** (mutual head-on share, life): JJ D2 0.84 to 0.66 (213 to 221 s), cone
  0.70 to 0.52, big v small 0.62 to 0.44, frozen 0.86 to 0.78; YY D2 0.74 to 0.62 (132 to
  183 s), cone 0.56 to 0.32 (183 to 232 s), big v small 0.53 to 0.31, frozen 0.79 to 0.63
  (197 to 276 s). Merges a kill: about the same in the Jujak mirror, fewer in the Yongma's.
- **Balance per size** (energy v turn 1 / 2 / 5 / 10v10, r9m then r9m2): D2 0.50 / 0.44 /
  0.53 / 0.43, 0.50 / 0.44 / 0.59 / 0.50; cone 0.52 / 0.52 / 0.69 / 0.59, 0.53 / 0.51 / 0.71
  / 0.68; big v small 0.60 / 0.51 / 0.50 / 0.47, 0.57 / 0.51 / 0.53 / 0.55; frozen 0.01 /
  0.43 / 0.36 / 0.26, 0.01 / 0.45 / 0.33 / 0.18.

**What worked**: the duel extension on the shield ratio (a hurt ship leaves the turning
fight): the first situational post-merge choice that wins, and the fix for the one-circle's
joust (a ship behind no longer takes the nose-to-nose meeting); a per-type gate (the
energy fighter leaves early, the turn fighter late); the team gate on the cross-type
extension (a Yongma leaving a Jujak wins a team game and loses a lone duel).

**What failed**: pressing instead of extending in the crowd, on any signal (energy, shield,
threats only, allies near): 0.24-0.43 against r9m at 5v5; the extension is right in the
crowd whatever the situation. Hold fire at the nose-to-nose meeting (0.09-0.37 at 1v1; the
mirrors become 500-1200 s stalemates). The soft (dithering) shield gate. The duel extension
on the lift ratio or the energy ratio alone (they lose a mirror). A high Jujak gate (loses
the Jujak mirror).

**Open for the author**:
1. Store r9m2 as the scripted agent's doctrine? The doctrine table grows from 66 to 88
   columns since `tune/maneuvers` (the play shell's compiled tick needs a rebuild), and
   r9m itself is not stored yet. The mirror-only variant ("DE 0.4 mirror") is the option that
   leaves every card's balance exactly as r9m's, at r9m's crowd strength (worst 0.485-0.499,
   within the noise) with the same duel gains.
2. Item 3 (the frozen card's crowd tipping to the Jujak) does not fall out: r9m2 tips it a
   little further at 10v10 (0.26 to 0.18-0.21). It is the frozen card's (its Jujak wins
   duels). On D2, the cone and big v small r9m2 holds r9m's balance within 0.06 from 1v1 to
   5v5; at 10v10 it moves toward the Yongma by 0.05-0.12 (D2 0.43 to 0.50-0.55, toward even;
   cone 0.59-0.62 to 0.68, away from it).
3. The bench's per-choice outcome is pair-symmetric in self-play, so "how often each choice
   wins" is measurable only for extend against the chase (the extender nets -0.01 to -0.03 a
   choice: it wins the game, not the engagement it leaves). A per-player choice tally in
   the versus jobs would answer it properly.
