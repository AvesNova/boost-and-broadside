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

## 02:59- KST: round 2 (`round2/`): the duel extension's shield gate

512 games against r9m and r9 at 1v1 / 2v2 / 5v5. ETA 03:25 KST (worst 03:35).
