# Item 20: the scripted trigger redesign (progress)

Branch `trigger-20` (from `core` 5a88e603). Design (author 2026-11-29 ~00:00): queue item 20.

- 00:05 built: `_expected_value_trigger` (lean/scripted/opponent.py) beside the old gated rule,
  selected per ship by the row `fire_rule` (1 new, 0 old; for this measurement). New rows
  `fire_uncertainty` (k) and `fire_gate_scale` (s, 0.25 log units, shared by the worth gate
  and the team-mate gate). Quick tests on lean/scripted pass (two old tests loosened to the
  soft gates, two new). Variants in variants/*.json; mirror-game statistics behaviour.py.
- 00:10 the CPU is too slow for step 1 (a 256-game 4v4 mirror did not reach 300 ticks in 3 min at 4 threads). Step 1 waits for the GPU: a watcher launches `bnb-20-step1` (step1.sh, under the GPU lock) once bnb-17-rl is inactive (ETA 02:40, worst 03:15). Step 1 ETA: start + ~1.5 h (first GPU run of these scripts; revised once it runs). CPU smoke tests of the eval path and the mirror script meanwhile.
