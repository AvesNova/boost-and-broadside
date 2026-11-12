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
