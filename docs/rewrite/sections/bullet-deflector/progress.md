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

## 01:55 mechanism, runtime hit model, workbench

- Worked: the deflector in `core` (hit test contact on b + D, `_damage` via `core.hit_fraction`,
  `_deflect` bends a missing bullet once on the tick holding its closest approach); presets in
  `card.DEFLECTOR_PRESETS`; every hit-model number is a row of the per-slot table
  (`core.RUNTIME_CONSTANTS`, `core.with_hit_model`). `lean/game/tests/test_deflector.py` 12
  passed, including a `torch.compile(core.step, fullgraph=True)` run where every hit-model input
  moves between calls: 1 frame compiled, no recompile (error_on_recompile on), the change took
  effect (CPU, CompileCounter backend).
- Side effect measured: random-action 20v20 (test_invariants, 30 ticks) deaths off 2 /
  symmetric 0 / frontal 0, damage dealt 2,118 / 1,490 / 1,537; the test now runs 45 ticks
  (7 / 2 / 3 deaths). The scripted two-call test needed 240 ticks for a death.
- Workbench `lean/tool/hitbench.py`: one step on ~13k synthetic bullets per preset; all three
  presets measured in 0.6 s, table + PNG in about 2 s wall. At 300 m: effective hit radius
  off 8.50 m (56.7 mrad), symmetric and frontal 5.50 m (36.7 mrad) off the nose; head-on 0 for
  all three (nose cone or frontal glance). Time to kill from astern at 10 mrad aim error:
  0.37 s off, 0.41 s symmetric/frontal at 300 m; 0.63 / 0.85 s at 650 m.
- Next: god-mode measurements, cost (compile check off vs on), sandbox sliders, rendering.
