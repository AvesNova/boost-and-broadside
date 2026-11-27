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
- Open for the author: docs/questions.md (question 19) (bend angle rule, field reach, dead-centre
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
  per-bullet chain after the reduction: per-bullet tensors written and re-read between six
  kernels (turn, gathers, damage, deflect, cat; the turn kernel alone 23 loads and 10 stores
  per bullet), at ~6 us a pass, plus a second full write of the bullet block (head 156 + tail
  28 us). Not atomics, not launches (+4).

## 14:16 egg speed, step 2: cuts tried

Each against the same tree's off (bench_tick, back to back). Until ~14:10 the machine had a
second CPU job (another worktree's python, ~20% CPU) and a browser at ~80% CPU: off read
1.25-1.29 ms then, 0.98-1.04 ms after on the same code. So the step-2 rows below compare
within a row, and the final before/after is re-measured on the quiet machine.

| try | what | egg ms (off) | kept |
|---|---|---|---|
| base | 180d16a8 | 1.772 (1.270) | - |
| A | the turn of every pair's pass read at the bend ship by a masked sum over the ships inside the pair reduction (`_at`), same for the straight impact at the target: no per-bullet gathers, the turn/gather/damage kernels fold into the pair kernel (263 us) | 1.523 (1.267) | yes |
| B | A, but select only the bend pass's inputs and run egg_turn per bullet after the reduction | 1.703 (1.248) | no: inductor split it into a reduction + a per-bullet kernel with 25 per-bullet loads (268 us) |
| C | A + the moved block assembled column by column, stacked once at the end of the tick | 1.543 (1.188) | no for the egg: the pair kernel recomputes where(flying, moved, start) per pair (391 us); off gains 0.06 ms |
| D0 | A + `_deflect` stacks the 8 columns once (no where over the head + cat of the tail) | 1.430 (1.287) | yes: the deflect folds into the pair kernel, egg 31 -> 28 launches |
| E | D0 with the pair test unrolled over the 8 ships ((E, N K) tensors, running minima) to avoid the persistent reduction's replicated per-bullet work | 1.730 (0.986) | no: slower, and eager dispatch calls 2,802 -> 3,921 |

Quiet machine (load ~1.9, GPU idle but the desktop), base vs D0 alternating, 2 rounds:
base off 0.983 / 1.040, egg 1.383 / 1.423; D0 off 1.043 / 1.029, egg 1.092 / 1.123 ms.
**Egg 1.40 -> 1.11 ms (the egg's extra over off 0.39 -> 0.07 ms).** Kernel time (profile,
min of 5): egg 1,408 -> 1,096 us per tick against off 972-984.

Why A works (reading of the generated code, not a counter measurement): a gather by an index
that a reduction produced is its own kernel in inductor, so every per-bullet value after the
gather went to memory; a masked sum over the 8 ships is another reduction over the same axis
and stays in the pair kernel's registers. Why B and E lost: in B inductor split the turn into a
second kernel that re-reads 25 per-bullet values; in E the unrolled pairs were materialised
per ship. Next: the whole-tick breakdown by feature; the ghost-only bullet prototype.

## 14:50 egg speed: after the reboot (14:20); hardware counters; whole tick by feature

Re-measured after the reboot on a quiet machine (load < 2, GPU idle but the desktop, laptop
4070, performance), bench_tick, 2 rounds each, trees alternating:

| tree | off ms | egg ms |
|---|---|---|
| base 180d16a8 | 0.980 / 0.975 | 1.392 / 1.387 |
| egg-speed D0 (00e3c654) | 0.974 / 0.976 | 1.091 / 1.093 |
| ghost-only prototype (egg-speed-ghostonly b3a16e9d) | 0.757 / 0.785 | 1.062 / 1.108 |

ncu (counters now enabled; one tick, `--cache-control none`, kernels serialised at ncu's
clocks, so its us run ~20% above the profiler's): DRAM per tick off 131.5 MB (70 read, 61
write), D0 egg 143 MB, base egg 225 MB. At the 4070's ~256 GB/s that is ~0.5 ms of the off
tick's 1.2 ms: **the tick is mostly not DRAM-bound**. The egg's pair kernel (D0) is
compute-bound: SM throughput 65%, issue slots busy 66%, DRAM 26%, 127 registers per thread,
occupancy 33% (limited by registers), 358 us under ncu.

Whole tick by feature (kernels grouped by their FX source-node names; ncu us and DRAM MB of
one tick; launches per tick):

| feature | off: launches / us / MB | D0 egg: launches / us / MB | base egg: us / MB |
|---|---|---|---|
| ship flight (fields on ships; 4 kernels) | 4 / 371 / 24.4 | 4 / 371 / 24.3 | 372 / 24.9 |
| bullet move (fields on bullets + the frozen-if-not-flying block) | 2 / 339 / 37.0 | 2 / 337 / 41.3 | 328 / 38.1 |
| hit test + damage (+ egg turn, deflect) | 6 / 182 / 24.9 | 6 / 363 / 33.0 | 670 / 122.0 |
| fog / sight (sight lines through fields, seen rows, bullet share) | 7 / 148 / 12.3 | 7 / 152 / 13.1 | 151 / 14.9 |
| fire (incl. ghosts: every living ship writes a bullet a tick) | 2 / 90 / 20.2 | 3 / 81 / 17.4 | 57 / 12.2 |
| bullet bookkeeping (moves-left / fired-tick counts) | 2 / 44 / 9.6 | 2 / 44 / 10.0 | 43 / 9.4 |
| collisions + boundary + shields | 1 / 16 / 2.6 | 1 / 19 / 3.5 | 18 / 3.6 |
| end check | 1 / 4 / 0.4 | 1 / 4 / 0.4 | 3 / 0.1 |
| misc (slot ring, alive select) | 2 / 5 / 0.2 | 2 / 5 / 0.1 | 5 / 0.1 |
| total | 27 / 1,200 / 131.5 | 28 / 1,374 / 143.0 | 1,648 / 225.3 |

Bullets in total (fire, move, hit test and damage, bookkeeping; ghosts included, every bullet
slot is computed whether ghost, real or idle): off 655 us of 1,200 (55%), 92 MB; egg 825 us
(60%). Ships (flight, collisions, shields, end check): 391 us (33%), 27 MB. Fog/sight 148 us
(12%). The fields' share is inside the flight and bullet-move kernels (fused); the ablation
below separates it. Observations are not built in the tick (only the fog's seen rows).

Ghost-only prototype (branch egg-speed-ghostonly, off D0; measurement only, not for merge):
12-slot ring, every ship alive or dead writes one bullet a tick, life exactly 12 ticks, no
moves-left countdown, one uint8 state (0 ghost, 1 real, 2 contacted: no more contact or
damage, keeps flying), fired tick from the ring age, slots start as ghosts parked on their
owner. Off 0.975 -> 0.77 ms (-21%); egg 1.09 -> 1.06-1.11 ms (no clear change). Kernels
(ncu): off 27 -> 23 launches, 1,200 -> 1,012 us, 131.5 -> 91.6 MB: the frozen-if-not-flying
block write is gone (bullet move 339 -> 231 us, 37 -> 19 MB), the counts kernels are gone
(44 -> 8 us), the bullet share is one kernel less, the hit test 182 -> 170 us. Egg: the
pair kernel grows 330 -> 431 us and a new 124 us per-bullet kernel appears (the move is no
longer materialised, so the pair kernel recomputes the moved columns, as cut C showed), so
the egg keeps 589 us of hit test. Tests: lean/game quick tier 116 passed, 1 failed
(test_every_living_ship_emits_a_bullet_every_tick: dead ships now write too and states are
uint8: the intended rule change). Parked ghosts at the owner can give contact labels to ships
passing within the hit radius in the first 12 ticks (a side effect of the rule as specified).

Egg cuts tried after D0 (against D0 in the same session):

| try | what | egg ms (D0) | kept |
|---|---|---|---|
| F | select the bend pass's inputs, egg_turn once per bullet (B again, on D0) | 1.301 (1.090) | no: inductor splits the damage into a separate per-bullet kernel |
| G | only the straight impact from the selected pass (one line impact per bullet, not per pair) | 1.239 (1.094) | no: same split |

Why: the fused pair kernel is compute-bound and register-limited; any extra post-reduction
selection makes inductor cut the damage (and its ~20 per-bullet inputs) into its own kernel,
which costs more than the per-pair arithmetic it saves.

Ablation (D0 tree, one process, each config compiled and timed like bench_tick; 14:45):

| config | off ms | egg ms |
|---|---|---|
| default (fields, fog, ghosts) | 1.016 | 1.133 |
| no fields | 0.728 | 0.932 |
| no fog | 0.880 | 1.041 |
| no ghost bullets | 1.031 | 1.102 |
| no fields, fog or ghosts | 0.705 | 0.842 |

So fields cost ~0.29 ms off (28% of the tick: in flight, bullet motion and sight lines), fog
~0.14 ms, ghost bullets nothing measurable (every slot is computed densely either way; ghosts
change which slots are live, not the work). The egg's extra over off: 0.12 ms with
everything on, 0.20 ms without fields (the fields' kernels hide part of it).
Next: final checks (tests, ruff, compile check).

## 15:10 egg speed: H kept, I dropped; final checks

| try | what | measured | kept |
|---|---|---|---|
| H | the egg's semi-axes as derived runtime rows of inverses (`inverse_egg_front`, `inverse_egg_rear`, `inverse_egg_half_width`, written by `hit_model_constants` with the semi-axes, so `with_hit_model` changes them with no recompile); `egg_entry` and `egg_turn` multiply | pair kernel (ncu) 329.7 / 329.3 -> 315.7 us; tick egg 1.079 / 1.080 vs D0 1.093 / 1.142 ms | yes |
| I | H + one inverse of the turned remainder's squared length (two divides -> one) and copysign for the turn's side | pair kernel 315.7 -> 314.8 us, tick total up 4 us | no (not measurable) |

Final, quiet machine (15:07, load ~1.3), base 180d16a8 vs egg-speed 177b12ac, 2 rounds:
off 1.084 / 0.976 -> 0.973 / 0.972 ms; **egg 1.425 / 1.385 -> 1.126 / 1.081 ms** (the egg's
extra over off ~0.41 -> ~0.13 ms). CPU eager equivalence against base: bitwise for A and D0;
with H within rounding (over 100 resynced ticks, 512 envs: bullet block 3.7e-4 m, damage
1.1e-5, the same 185 hit pairs).

Checks: lean/game quick tier 117 passed (the egg tests in test_deflector.py among them);
lean/scripted, tool, sandbox quick tier 73 passed; ruff clean. Compile check (cold, baseline
not updated): advance 3,296 traced operations (unchanged; core baseline 2,856, +15%), act
4,590, 4 graphs, 0 graph breaks, no recompiles, cold 295.0 s (budget 260 s): the same three
tripwire failures as before this branch.

Open: (1) the egg's pair kernel is compute-bound and register-limited (127 registers, 33%
occupancy); fewer per-pair operations would need inductor to keep post-reduction work fused
(F and G show it splits the damage out) or a hand-written kernel (last in the speed order).
(2) Off gains 0.06 ms from assembling the moved block column by column (cut C) but the egg
loses; the ghost-only prototype gets the same off gain (-0.2 ms) for a rule change.
(3) Fields are the largest single cost (~0.29 ms of ~1.0 off).

## 2026-11-27 16:00 ghost-only bullets, merge-ready (branch ghost-only, off deflector-lookahead b70dcd18)

Author go-ahead ~15:20. Done: the prototype (b3a16e9d) cherry-picked and finished. The ring is
the clock: the slot written on tick t is t mod 12 for every ship (the per-ship `next_slot`
column is gone, so a dead ship's ring advances too: in the prototype a dead ship kept rewriting
one slot while the other eleven flew on and reported a miss every tick). `State.bullet_state`
uint8 (GHOST / REAL / CONTACTED, in lean.contract) replaces `bullet_real` and the
`bullet_counts` block; `GameConfig.ghost_bullets`, `bullet_moves` and `_bullet_ends`' countdown
removed. Fresh slots start CONTACTED at the origin (the prototype parked ghosts on the owner,
which labelled passers-by). Bullet life: asserted at constants build (`_per_type`), and
`tuning.ring_gun_range` for designs moving muzzle speed or drag.

Speed (laptop 4070, performance profile, load 1.1-1.6, both trees in one locked job,
alternating, bench_tick median of 3 x 300 ticks):

| tree | off ms | egg ms |
|---|---|---|
| base b70dcd18 | 0.973 / 0.973 | 1.078 / 1.079 |
| ghost-only | 0.753 / 0.753 | 0.880 / 0.925 |

Kernel time per tick (torch.profiler, 40 ticks, best of 3; two passes alternating): off
1,003 / 1,087 -> 807 / 884 us, 27 -> 24 launches; egg 1,206 / 1,212 -> 1,023 / 1,038 us,
28 -> 24 launches. Gone: the frozen-if-not-flying block write (~125 us), the counts kernels
(~37 + 7 us), one bullet-share kernel (~22 us). The fire kernel 93 -> 85 (off) / 77 -> 71 (egg).
The egg's pair kernel 264 -> 284 us (+20 us, not the prototype's +100 and no new per-bullet
kernel): the generated code reads the moved block from memory (inductor stores it once here;
in the prototype it recomputed the move inside the pair kernel), so the off path's gain
carries to the egg (-0.17 ms) with no further change.

ncu (one tick, `--cache-control none`, kernels serialised at ncu's clocks; 16:01-16:02, load
~2-4 from the other agent's job just before; both trees back to back):

| | base b70dcd18 | ghost-only |
|---|---|---|
| off: kernels / us / DRAM MB | 27 / 1,218 / 135.5 | 24 / 1,009 / 84.4 |
| egg: kernels / us / DRAM MB | 28 / 1,357 / 137.7 | 24 / 1,198 / 93.4 |

Changed kernels (us / MB): the frozen-if-not-flying block write (base off 107.3 / 24.4, egg
104.5 / 23.8) and the counts kernels (36.3 / 8.6 and ~6) are gone, and the bullet share's
mean kernel 20.7 / 4.5 -> 4.0 us (one uint8 compare instead of the counts and the bool). The fire kernel off 107.1 / 22.6 -> 79.5 /
18.6, egg 71.3 / 16.6 -> 65.3 / 15.2. The off pair kernel (`_first_contact`) 84.7 / 15.6 ->
63.6 / 7.9. The egg pair kernel (with `_bullet_ends`, damage and deflect fused) 316.6 / 27.7
-> 339.5 / 16.7: +23 us, less traffic (compute-bound, as before); the bullet move kernel 230.7
/ 12.9 -> 230.4 / 19.9 (it now writes the moved block that the pair kernel reads).

Checks: quick tier (`-m "not slow"`) on lean/game, scripted, tool, sandbox, learn, eval, model:
343 passed (the first run caught two tuning tests whose designs asked for an 11- and a 24-tick
life; fixed with `tuning.ring_gun_range` and a 12-tick row). Ruff clean. Compile check (cold,
baseline not updated, 16:03, performance, load ~1.2): advance 3,283 traced operations (3,296
before, core baseline 2,856), act 4,586 (4,590), 4 graphs, 0 graph breaks, no recompiles, cold
282.3 s (budget 260 s): the same three tripwire failures as before this branch.

Open: (1) the egg pair kernel is +23 us (ncu) over the base's, compute-bound as before; not
chased. (2) The tool's gun-range design input can ask for a bullet life other than 12 ticks
(the core then refuses with a clear error); it could become a read-out of muzzle speed and drag
(author). (3) Dead and absent ships' ghosts label rows the hit loss masks (not `own`); absent
ships sit at the centre, so their ghosts can touch ships there (labels only, no damage).
