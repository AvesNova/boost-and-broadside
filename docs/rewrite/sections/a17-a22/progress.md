# Arms A17 (raw inputs) and A22 (shot damage head): progress

Branch `feat/a17-a22-options`, from `feat/imugi-boundary` (83404bbf). Brief: both arms launch
from settings alone; defaults unchanged; CPU tests only.

## 2026-11-16

- Started. Read arm-list (rows 12, 22), arm-review (A17, A22), ideas 35 and 74, arms-todo
  (HL-Gauss decision), yemong.md.
- **A17 built:** `YemongConfig.input_encoding` ("codes" default, "raw"). Raw: ship state 25
  scalars for the 580-wide code (`state_code.RAW_STATE`, `ShipStateCode.raw_inputs`),
  accelerations symlog scalars, ellipse features raw, field/zone centres from the map centre,
  no global centre code, arena log as one column. Needs next_state "delta" (as Fourier).
  New-run widths: ship 89/644, global 31/136, field 5/84, zone 30/109; 2.74M params for 2.95M.
  Tests `test_yemong_raw_inputs.py` (9) pass; ONNX export case `a17_raw` passes.
- **A22 built:** `shot_head="damage"`: a pairwise head (own shooter x every row) predicting the
  damage share (HL-Gauss, 21 bins 0..2 of a full hit). Game damage factored into
  `combat.hit_damage` (game unchanged); the re-simulation records each shot's target and
  damage, bit-equal to the game's on the shots fixture (106/106 hits, target heading from the
  next tick's capture). Learner term `shot_damage` (coefficient 0.019, sum over targets).
  Smoke update: loss 29.9 nats/shot at init (9 targets x ln 25), gradient share 0.2%.
- Machine heavily loaded (load ~30): the full touched test files time out at 50 min; ran
  targeted subsets.
