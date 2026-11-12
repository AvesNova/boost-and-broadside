# Map panel: progress

Done
- Game rules (defaults = today's numbers, fingerprint unchanged): `field_radius_min` 30 m,
  `field_radius_max` 490 m (map scale 1, scaled by the map scale like the other map lengths),
  `field_disc_fraction` 0.85, `field_high_share` 0.5 (P(level > 0)), `field_strong_share` 0.5
  (P(|level| = 2)). Sign and strength are independent draws; point-symmetric pairs share a
  level, so any shares leave the map fair. Radius law: `field_radius_distribution` (0 uniform,
  the default; 1 triangular, densest at the smallest radius, `min + span * (1 - sqrt(1 - u))`;
  2 lognormal, `median * exp(sigma * sqrt(2) * erfinv(2u - 1))` clamped to [min, max], with
  `field_radius_median` (default the geometric mean of min and max, 121 m) and
  `field_radius_log_sigma` 0.6). All three read the same one uniform per field, so the draw
  schedule and the default's numbers are unchanged. Choose it in the Map / mode panel's
  Map group (slider 0-2); the panel's last line shows the mean and 10th/90th percentile
  radius at map scale 1. Tuning: MAP group, from the next episode, live.
- Play shell: Map / mode panel (F3, top bar "Map / mode") with New map, Restart match, Save
  map, the mode buttons and the map's rule groups; the old tuning panel is now "Ships and
  flight" (F2, top bar "Ships"). New map and Restart are also in the top bar.

To do (not built)
- Save only the current ship type.
- Undo / revert a ship's unsaved edits (the P-51D was edited by accident, thinking it was vizia).
- An "Edit this ship" button on a selected ship's card.
- Move the selection card to the left under the fleet status list, leaving the right side for
  the ship and map panels.
