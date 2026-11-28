# Item 20: the scripted trigger redesign (progress)

Branch `trigger-20` (from `core` 5a88e603). Design (author 2026-11-29 ~00:00): queue item 20.

- 00:20 built: `_expected_value_trigger` (lean/scripted/opponent.py) beside the old gated rule,
  selected per ship by the row `fire_rule` (1 new, 0 old; for this measurement). New rows
  `fire_uncertainty` (k) and `fire_gate_scale` (s, 0.25 log units, shared by the worth gate
  and the team-mate gate). Quick tests on lean/scripted pass (two old tests loosened to the
  soft gates, two new). Variants in variants/*.json; mirror-game statistics behaviour.py.
- Waiting for the GPU: bnb-17-rl runs to about 02:40-03:15. CPU timing of one mirror run
  under way to see whether the CPU can do the step-1 games before then.
