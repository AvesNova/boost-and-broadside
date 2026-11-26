# Bullet deflector and the hit-model workbench (branch `bullet-deflector`, overnight 2026-11-27)

Stream B of docs/rewrite/overnight-2026-11-27.md. Proposal, unmerged: waits for the author.
Milestone notes: tried / worked / didn't / next; measured numbers only; ETAs as clock times.

## 01:45 start

- Read the plan, core's bullet move / hit test / damage, the card plumbing (`Constants`:
  numbers equal for every type are Python floats in `uniform` and bake into the compiled
  graph; only per-type rows are tensors).
- Plan: (1) hit-model numbers always rows of the per-slot table (runtime tensors); (2) the
  deflector in the hit test, damage and a bend after the hit test; (3) presets in
  `card.DEFLECTOR_PRESETS`, `GameConfig.deflector`; (4) workbench; (5) god-mode measurements;
  (6) sandbox sliders; (7) tests and docs. ETA mechanism + runtime constants 03:15 (worst 04:00).
