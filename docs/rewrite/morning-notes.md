# Morning notes

Judgement calls made overnight for the author to review, newest last. One line each, with
where to look.

- G1 removed Vizia, Yongma and Jujak (v2's solved rows, not built by the tuning route); only the P-51D is left. The P-51D's guns, shield and nose values are Vizia's, kept as `tuning.GAME_FIELDS`. Vizia can come back as a tuning-route row if you want it as a reference.
- The sandbox now flies the real game, fog and fields included, so the chaser sees you only within 1,000 m (or when you fire or are hit). For the old pure-flight feel: `--set fog=False --set field_count=0`.
- G2 spawn edge rule: within 1,000 m of the boundary (the boundary feature's clip) an outward spawn heading is mirrored in the tangent; it replaces the old minimum-turn rule (acos(d / 436.5 m)). Both spawn rules use it (lean/game/core.py `_edge_safe`).
- G2 "mixed" keeps v2's spawn disc of 0.85 R; "sides" is a half disc of radius 0.80 R per team with its flat edge 0.05 R (210 m) from the dividing line, noses along the axis toward the other side within 15 degrees (card.SIDE_GAP_FRACTION, SIDE_HEADING_JITTER).
- G2 training spawn draws proper speed (stall 93 m/s to top level 335 m/s) regardless of fields; the standard spawn keeps 145.3 m/s over the map. One eighth standard is a per-environment coin (share 1/8 in expectation, not exact).
- G2 clock tie-break compares total shield in points (not fractions of full); a lead within 1e-5 of the sums is equal (rounding of the two sums).
- G2 pending action lives in State.pending (E, N, 3) and Obs.pending (joint index, 42 = none), read by the model as a 42-way one-hot appended to the ship input (zero for none); stored int8. A dead ship's given action is stored as pending but never applied.
- G2 lean.eval forces training_spawn off inside play_results, so a training GameConfig passed to the crossover still plays the standard spawn; lean.train turns it on.
- G2 feature scales: speed and velocity symlog over 200 m/s, nose rate symlog over 1 rad/s, shield over 100 (was 150), gun cooldown in whole ticks (always 0 at one shot a tick); angle-of-attack cos and sin kept unscaled though they repeat the angle (lean/game/NOTES.md "Feature scales").
- S power thresholds (untuned, in the ship's own numbers): dive below 1.8 stall speeds (167 m/s) or below 0.95 corner speeds in a hard turn (P of at least the sustained stick); climb above 1.2 corner speeds, within 0.21 gun ranges (Imugi's reverse centre), or on half the ticks when farther than 3 gun ranges with the store below 0.3; speed gates sharp (scale 0.05). lean/scripted/NOTES.md "Power".
- S a slow ship never climbs, and dives only with store to spend (the dive's probability fades over the game's 5% edge band, as its effect does), else it coasts: no "dive" label that does nothing (before, 86% of ticks were dives, mostly on an empty store).
- S power floor 0.02 on each of coast, dive and climb of the final distribution, replacing the old 0.04 per power gate (which compounded); the stick floors are unchanged.
- S "far from every enemy" is the kept target's distance (at most 1/0.8 of the nearest); with no known enemy (fog) the ship counts as far.
- S head-on side lives in the memory (drawn by fresh_memory, kept by act, redrawn by renew_memory), so the yardstick has it too; it applies when the wanted direction is within 0.1 rad of straight behind.
- S memory is required and functional: act(..., memory) returns (action, memory); no memoryless path is left. Collect keeps one memory for every ship (the one scripted call serves the BC teacher and the league opponent).
- S persona population: independent uniform draws on a box (log-uniform for centres, temperature e^+-0.4, boundary margin e^+-0.3; lead gain +-0.2, weak-target weight 0 to 1, regain share +-0.25); spread above 1 is refused; v2's stand-off field dropped (the close-climb centre's spread covers it); keep factor and head-on band are not per persona.
- S competence test: at spread 1, no ship past the end of the boundary ramp and at least 98% of ships alive 20 s fire (not 100%: the yardstick itself had 1 of 236 survivors that never got a firing solution).
- S the sandbox's opponent is now the scripted opponent by default (memory carried, fresh at each respawn); `--opponent chaser` gives the old one; the sandbox's "chaser" names became "opponent" (`--no-opponent`, C toggles it).
- S the chaser stays memoryless (a deterministic floor used only by tests and `--opponent chaser`).
- S lead in a field: map velocities are proper velocities over n, and the time of flight is the shooter's n times longer; the gun's shorter reach inside a field is not modelled (NOTES "Open").
- S operations 186 -> 208 per scripted call for both teams, over the 188 signed off: the kept target (the old optional memory with its hold was 196), the medium in the lead (7), the power rule and the head-on side.
- S docs/rewrite/sections/imugi/measure_lean.py left on the old API (it records the 2026-11-22 measurement); game_ops.py and feature_ranges.py moved to the new one.
