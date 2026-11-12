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

## 02:40- KST: round 1 (`round1/`): one situational gate at a time over r9m

512 games, 1v1 / 2v2 / 5v5 against r9m, self-play at 5v5 (energy v turn) and the 1v1
mirrors. ETA 03:10 KST (worst 03:20).
