# Elimination and the Clover map: progress

## 2026-11-12, step 1 (the mode, fixed layouts, Clover, the mode input)

- Tried: elimination as team deathmatch's variant: every death scores for the other team,
  one kill per enemy ship wins (`kills_per_ship` 1: a wipe), and no respawn, as a rule of
  the clock: a dead ship waits the episode's whole length (`CLOCKS[elimination]` respawn
  None), so it is still waiting when the episode ends, and benching needs nothing new. At
  the clock: more ships alive wins, then more total shield (the game's only health) over
  the living ships, then a draw (`modes.phase.clock_result`).
- Fixed layouts (`contracts.config.MapLayout`): half the fields (centre, radius, level
  choices) with their 180-degree images, team 0's spawn points (position, heading or
  toward the centre, jitter) with team 1's images, and a random turn of the whole layout
  per match (the spawn axis's draw). Part of `ModeRules`, so a layout is its own slice of
  the batch. Random maps stay the default; the existing presets compute the same numbers
  (their fingerprint without the new observation column is the recorded one).
- Clover: 2v2, real-scale set, arena 4933.2 m (the density rule's 2v2 size of the
  7800 m map), fields of radius 1233 m at 0.45 R on the diagonals, spawns at 0.875 R at
  1/4 and 7/10 o'clock, 150 m and 5 degrees of jitter, 180 s.
- real_scale_elimination: 5v5, 10 random fields, 7800 m, 180 s; not in the training mix
  yet (the share is the author's call: options A/B/C asked).
- Model: a `mode_elimination` observation column; Yemong's `mode_inputs` build option
  (old records read the two older flags; new runs read all three).
- Next: tests of the rules, docs, an offscreen render of Clover.
