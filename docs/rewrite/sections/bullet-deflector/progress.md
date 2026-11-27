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

## 02:40 cost: profile, cut work

GPU tick, compiled `core.step`, 4070 laptop, performance profile, E 4096, 4v4, fields and fog
(`bench_tick.py`, median of 3 blocks of 300 ticks); CUDA time from `profile_tick.py` (40 ticks).

| version | off, ms | frontal, ms | deflector's cost |
|---|---|---|---|
| first (pair tensors D and tau out of the hit test) | 1.77 | 2.94 | +1.17 ms (+66%) |
| D and the bend's ship found inside the hit test's reductions | 1.76 | 2.06 | +0.30 (+17%) |
| + the flight's 2x2 matrix exponential elementwise (not the deflector) | 1.07 | 1.71 | (merged reduction, below) |
| merged contact + bend reduction (stacked keys): reverted | 1.07 | 1.71 | +0.64: worse |
| reverted to separate reductions | 1.07 | 1.37 | +0.31 (+29%) |
| + no (E, 2, N K, N) pair blocks out of the hit test (`_pass_against`) | 0.97 | 1.21 | +0.24 (+24%) |

- Worked: keeping the pair work inside the hit test's reductions; recomputing one bullet's pass
  per bullet. Each step bitwise-equal play (300 ticks random 4v4, off and frontal).
- Didn't: merging the two reductions by stacking their keys (the stack is materialised).
- Found while profiling (not the deflector): the flight's 2x2 matrix exponential ran as four
  cuBLAS gemv calls, 43% of the step's CUDA time (35.7 of 82.7 ms / 40 ticks). Written out
  entry by entry it fuses: the whole step 1.76 -> 1.07 ms with the deflector off (-39%); within
  8e-6 of the old on entries up to 53; flight tests pass. Its own commit (f583eb0c), to
  cherry-pick or drop.
- Left: the bend's reduction is its own kernel (0.17 ms of the 0.24); a further cut would fold
  it into the contact reduction without materialising keys (a custom reduction), not tried.
- CPU eager E 256: about 11-13 ms either way (noise ~1 ms); eager dispatch calls 2,495 off /
  2,701 frontal (+8%).

## 02:55 checks, wrap-up

- Compile check (cold, laptop 4070, performance; `compile_check.py --deflector`): `advance`
  core baseline 2,856 traced operations; branch off 2,998 (+22 runtime rows, +120 the
  elementwise matrix exponential); frontal 3,219 (+221 the deflector). Frontal FAILS the
  tripwire (+13% > 10%); off within. 4 graphs, 0 recompiles, 0 graph breaks; cold 237.0 s off,
  237.4 s frontal (baseline 231.7 s). Baseline not updated.
- Quick tier (CPU) on lean/game, lean/scripted, lean/tool: 215 passed, 1 xfailed (the scripted
  competence test, as on core), 1 xpassed; lean/learn, lean/eval, lean/model 152 passed (before
  the last two core changes, both bitwise-equal or rounding-only). ruff clean.
- The god-mode tables were measured before the matrix-exponential commit (float rounding only).
- Open for the author: docs/rewrite/questions.md (bend angle rule, field reach, dead-centre
  frontal bullets, the side D's cost, default preset, collision blame angles).

# Branch deflector-lookahead (2026-11-27 morning): the egg

## 05:30 step 1: one branch

- Merged origin/bullet-deflector into deflector-lookahead (from scripted-lookahead 56f70453).
  One hit-worth function: `core.hit_damage` (the tick's damage and the agent's fire-worth);
  the deflector branch's `hit_fraction` folded in. The rule-based agent stays deleted (its
  deflector edits dropped); godmode.py switched to the look-ahead agent. Kept: the matrix
  exponential (f583eb0c, author: decided), the runtime hit rows. Quick tier lean/game,
  lean/scripted, lean/tool, lean/sandbox: 190 passed, 1 failed (the pair kernel's twin, the
  kernel still on the old aspect; fixed in step 2). Commit f323531e.

## 05:50 step 2: the egg (author decision 2026-11-27)

- Tried / worked: `core.egg_entry` (the first inward crossing of the egg by the straight
  in-tick relative path, two half-ellipse quadratics), `core.egg_turn` (the early hump turn,
  the turned remainder's swept test), `core._deflect` (the turned bullet: end shifted by the
  turned rest of the relative move; map velocity = the ship's tick-average velocity + the
  turned relative velocity, so the relative speed is kept). Old push, presets
  symmetric/frontal, `deflection`, `hit_fraction`, `DEFLECTOR_SIGN_SOFTNESS` removed.
- First version rotated the world direction (world speed kept): the bench's egg hit width
  astern at 100-200 m came out 3.7-3.9 m against the still-ship 4.75 m, because the ship's
  velocity rotated out of the relative frame ((R - I) v_ship, about 130 m/s at 45 deg). The
  relative-frame velocity fixed it (4.7 m at every range).
- Hit widths (`core.hit_widths`, a 17-point runtime column by aspect cosine) equal the
  design sketch (egg 24 m, early hump, 45 deg): head-on 2.05 / side 4.75 / astern 4.75 m
  against the sketch's 2.05 / 4.75 / 4.75; 24.0 m with no turn. In the step (target at
  cruise, 300 m): 2.1 / 4.7 / 4.7 m (30 deg: 3.1 / 6.8 / 7.1; 60 deg: 1.5 / 3.5 / 3.5).
  The ship's motion: within 0.1-0.2 m; the core taper does not change the width (any contact
  damages: graze is positive up to the hit radius) but the damage near the edge.
- Agent: target size = half the hit width at the bullet's aspect (gun solution, pair kernel,
  trigger); trigger's worth = hit damage x kinetic x hit chance, chance = erf(half width /
  (sqrt 2 sigma)), sigma = flight time x sqrt(spread^2 + (lead_error x 0.5 x his sustained
  turn acceleration x flight)^2); lead_error 0.25 (a reasoned choice, not tuned). Nose cone
  off by default (nose_cone 0; the code kept as a hit-model row).
- Hit radius 12, core 10 m (author). Workbench and views moved to the egg
  (workbench/hitbench.md, apparent_size.png, bullet_paths.png).

## 06:05 step 3: re-measure (laptop 4070, performance profile)

- Cost, compiled `core.step`, E 4096, 4v4, fields and fog (bench_tick.py, median of 3 blocks
  of 300 ticks): off 0.98 ms, egg 1.58 ms (+0.60 ms, +62%; the old push was +0.24 ms). Eager
  dispatch calls 2,511 off / 2,792 egg (+11%). (The off timing ran beside a CPU job of mine;
  the egg timing on a quiet machine.)
- God mode (shields refilled every tick), the look-ahead agent both sides, 512 games 1v1 (fog
  off, no fields) and 256 games 4v4 (fields, fog), 60 s each, today's spread:

  | | hit chance / real bullet | hits / pass | passes with no hit | hits in control | hits head-on / side / rear | merge turn 90 deg, median s | friendly-fire share of hits (4v4) |
  |---|---|---|---|---|---|---|---|
  | off | 0.752 | 51.7 | 0.18 | 0.80 | 0.09 / 0.52 / 0.39 | 2.57 | 0.045 |
  | egg | 0.275 | 15.3 | 0.26 | 0.95 | 0.01 / 0.42 / 0.57 | 2.23 | 0.049 |

  Firing share of ship-ticks 1v1 0.15 off / 0.13 egg; 4v4 0.25 / 0.10. A pass is a local
  minimum of the distance under the gun range, 1 s apart; hits go to the nearest pass.
- Next: the spread grid (running), compile checks (queued after it).
- Compile check (cold, laptop 4070, performance; baseline not updated): `advance` 3,022 traced
  operations off / 3,302 egg (+280, +9%; +16% over the core baseline 2,856); `act` 4,552 both
  (look-ahead agent; the core baseline 1,821 predates it; round 3 recorded 4,494, so the hit
  widths and the hit chance add 58); 4 graphs, 0 graph breaks, no recompiles; cold 287.8 s off,
  291.6 s egg (budget 260 s; the author raised it provisionally). FAILS the tripwire on act
  (pre-existing), advance (egg) and the budget; reported, not a blocker (author).

## 06:25 step 4: experiments (round 1 grid, round 2)

- Grid: deflector off / egg x spread today (1.5 mrad), 5, 10, 20 mrad x 1v1 (fog off, no
  fields, 512 games) and 4v4 (fields, fog, 256 games), 60 s each, re-dealt. Tables in
  egg-experiments.md. Spread barely moves play numbers: the agent fires at about 200 m
  (median, egg) and holds fire more at a wide spread (59 -> 50 real bullets per kill in 1v1);
  its misses are mostly aim, not spread.
- Didn't: round 2's extra cells (damage 7/14, then nose cone, egg front 24/80) shared the
  cached ship card, so a cell's rows carried into the next (nose cone and front cells ran with
  hit damage 14). Fixed (each cell from the game's own card) and those cells re-run; the
  contaminated records kept apart (data/egg-2026-11-27/round2_contaminated).
- Replays (mp4) in data/replays-2026-11-27-egg/: 1v1 egg and off (2 seeds), 1v1 egg 20 mrad,
  4v4 egg and off. 1v1 off seed 0: a head-on joust from the spawn, both firing nose to nose,
  over in 2.7 s.
- Next: store_crowd duels (queued), the egg's tick cost (profile).

## 07:20 step 4 done; the cut; final checks

- Experiments written up in egg-experiments.md (grid, largest turn, damage, egg front, the
  old rule, god mode, seed 1, lead_error, fire_tolerance, store_crowd duels, TTK band).
  Worked: the egg ends the head-on joust (head-on kills 0.45 -> 0.00 in 1v1), hits come from
  control (0.05 -> 0.70), kills per minute halve, nothing degenerate measured. Didn't matter:
  spread, hit damage, egg front, lead_error, for this agent's kill rate.
- Cut (speed order: profile, cut work): a ship whose egg a bullet enters this tick has no
  straight contact, so the entry and contact reductions run as one pass: egg tick 1.58 ->
  1.37 ms (off 0.98; +0.39 ms, +40%). Play re-checked after it: within the seed spread.
- Final: quick tier lean/game, scripted, tool, sandbox, learn, eval, model: 342 passed; ruff
  clean. Compile check egg (after the cut): advance 3,296 traced operations (core baseline
  2,856, +15%), act 4,590 (store_crowd adds 38), 4 graphs, 0 breaks, cold 293.4 s (budget
  260 s; author: provisionally raised). Baseline not updated.
- Replays (11 mp4) in data/replays-2026-11-27-egg/: 1v1 egg, off, old rule (nose cone), egg
  60 deg, egg 20 mrad; 4v4 egg, off, egg 20 mrad.

## 2026-11-27 13:50 egg speed, step 1: where the egg's time goes (branch egg-speed)

Laptop 4070, performance profile, compiled `core.step`, E 4096, 4v4, fields and fog. The video
player (2.3 GB, ~20% GPU) ran during the first bench, closed before the second; the desktop
(gnome-shell on this GPU) shows ~17% utilisation at idle throughout.

- bench_tick (median of 3 x 300 ticks), off / egg back to back: 1.277 / 1.765 ms (player on),
  1.270 / 1.772 ms (player closed): **+0.50 ms (+39%)**. (The older 0.98 ms off did not
  reproduce: 1.27 ms today in both conditions.)
- Kernel profile (torch.profiler over 40 compiled ticks, 5 alternating off/egg rounds, per
  kernel the minimum round): off 27 launches, 984 us of kernel time per tick; egg 31 launches,
  1,408 us (**+424 us**). Hardware counters are not readable here (ncu: ERR_NVGPUCTRPERM), so
  traffic is estimated from the generated kernels: a per-bullet f32 tensor (E, N K) is 1.6 MB,
  a pair tensor (E, N K, N) 12.6 MB, the bullet block 12.6 MB; at ~250 GB/s one pass over a
  per-bullet tensor is ~6 us.

  | kernel (what it does) | off us | egg us | extra | launches added |
  |---|---|---|---|---|
  | pair reduction: swept test (+ egg entry, both halves) + first contact, (E, NK, N) in registers | 76 | 107 | +31 | 0 |
  | egg_turn on the bend ship's pass + bent_hit + the fog's real_hits scatter: 23 loads, 10 per-bullet stores | - | 112 | +112 | 1 |
  | gathers: the pass against bend ship / target (2 kernels in egg) | 11 | 10 + 48 | +48 | 1 |
  | _damage + hit_taken / hit_damage scatters (recomputes the turn's cos; 33 loads) | 60 | 168 | +108 | 0 |
  | _deflect: the turned head (E, 5, N, K), each row recomputing its part from ~8 per-bullet inputs | - | 156 | +156 | 1 |
  | the cat: copy of the 3 tail columns into the new block | - | 28 | +28 | 1 |
  | rest (fire, move, flight, fields, fog, ...) | 837 | 779 | -58 (L2 noise: the fire kernel 92 -> 39 us, the block hot in L2) | 0 |

  Plainly: the egg's pair work is cheap (+31 us, it stays in registers); the cost is the
  per-bullet chain after the reduction: ~45 per-bullet tensors materialised between six
  kernels (turn, gathers, damage, deflect, cat), each pass ~6 us, plus a second full write of
  the bullet block (head 156 + tail 28 us). Not atomics, not launches (+4).
