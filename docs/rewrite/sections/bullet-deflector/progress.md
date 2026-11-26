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

## 02:05 god-mode play, sandbox sliders, the bend still

- Agent: the rule-based scripted agent on core (lean.scripted, ScriptedConfig(), both sides;
  its fire-worth now reads `core.hit_fraction`). `godmode.py`, CPU, 4 seeds x 128 games x 60 s
  each (`godmode*.json`; `summary.py` prints the table). God mode: 1v1, fog off, no fields,
  shields refilled each tick. Kills: GameConfig() otherwise, re-dealt games. Under god mode the
  flight does not depend on damage, so off and symmetric fly the same games (same passes and
  control); frontal differs only where the agent's fire-worth differs (30-45 deg).
4 seeds x 128 games x 60 s each; mean (min-max over seeds)
  
  | | off | symmetric | frontal |
  |---|---|---|---|
  | damaging hits / min / game | 50.9 (47.8-54.5) | 38.4 (35.7-41.2) | 37.4 (34.8-40.5) |
  | bullet damage / min / game | 290 (267-313) | 187 (169-204) | 186 (168-207) |
  | damaging hits per pass (mean) | 12.18 (11.64-12.79) | 9.19 (8.75-9.70) | 9.00 (8.41-9.57) |
  | passes with no damaging hit | 0.295 (0.273-0.306) | 0.303 (0.283-0.319) | 0.311 (0.285-0.330) |
  | passes with 11+ hits (a kill's worth) | 0.372 (0.353-0.395) | 0.319 (0.306-0.344) | 0.308 (0.280-0.335) |
  | share of hits in control | 0.319 (0.295-0.352) | 0.319 (0.298-0.350) | 0.316 (0.282-0.380) |
  | share of ship-time in control | 0.044 (0.040-0.046) | 0.044 (0.040-0.046) | 0.042 (0.035-0.045) |
  | contacts (real bullets that touched) | 6722 (6337-7132) | 5075 (4740-5400) | 4782 (4458-5187) |
  | contacts within 30 deg of the nose | 205 (161-231) | 158 (122-183) | 0 (0-0) |
  | damaging hits within 30 deg of the nose | 0 (0-0) | 0 (0-0) | 0 (0-0) |
  | pass interval median, s | 10.95 (9.73-11.93) | 10.95 (9.73-11.93) | 11.36 (9.75-12.68) |
  | kills / min, 1v1 | 0.795 (0.734-0.914) | 0.609 (0.523-0.719) | 0.615 (0.578-0.641) |
  | 1v1 draws | 12 (10-14) | 17 (14-19) | 16 (12-18) |
  | kills / min, 4v4 (all 8 ships) | 4.68 (4.48-4.80) | 4.01 (3.91-4.21) | 3.94 (3.80-4.05) |

- Frontal: 0 contacts and 0 damaging hits within 30 deg of the nose (every frontal bullet
  glances). Symmetric and off: 158 / 205 frontal contacts per seed, all doing 0 (the nose cone).
- The side D (3 m) is what moves play: damaging hits -25% (symmetric -25%, frontal -27%), bullet
  damage -36%, kills/min 1v1 -23% and 4v4 -15% against off; symmetric and frontal are within
  the seed range of each other on every row. Share of hits in control unchanged (about 0.32;
  my control rule: within gun range, nose within 30 deg of the line, behind the target's beam).
- Sandbox: "Hit model (live)" panel (sliders from `core.HIT_MODEL`, the deflector presets from
  `card.DEFLECTOR_PRESETS`), each change sent to the running World (`World.set_hit_model`), no
  restart; bullets drawn joined tick to tick by identity with a spark where a tick turned one by
  more than a degree. Tests: socket round trip, kept across a reset. Not flown by eye.
- Bend still (`bend_still.py` -> bend_still.png): frontal head-on fan glances 14 deg off either
  side; one bullet at offset 0 passes straight through (the smooth sign at b = 0; no damage);
  side passes show the 3 m push as a kink.
- GPU tick (first version, 4070 laptop, performance profile, E 4096, 4v4, compiled): off 1.77
  ms, symmetric 2.93, frontal 2.94 (+66%); CPU eager E 256: 16.1 / 15.7 / 15.9 ms (noise);
  eager dispatch calls 2,423 / 2,580 (+6.5%). Moved the pair work into the hit test's
  reductions (bitwise-equal play over 300 ticks); re-measuring (bench2), compile checks queued.
