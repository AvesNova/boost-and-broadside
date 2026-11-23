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
