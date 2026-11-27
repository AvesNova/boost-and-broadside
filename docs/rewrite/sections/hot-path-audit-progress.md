# Hot-path audit: the approved fixes (branch tick-audit, 2026-11-27 evening)

The author's decisions on the audit (hot-path-audit.md, ~16:00): items 1 (compile shapes), 2
(hit labels), 5 (constants per type), 3b (reset only finished environments, cap D = 16), deal
creation measured, and the small items 4, 8, 9, 10. Branch `tick-audit` from ghost-only
23a4271c plus deflector-lookahead (egg_pass.py ported to `bullet_state`).

Method: every timing is one locked GPU job (flock on data/gpu.lock, systemd-run, MemoryMax
12G), laptop RTX 4070, performance profile; the load average is recorded per run. Before and
after run back to back in the same job, each from an exported tree of a fixed commit
(`git archive`, PYTHONPATH), so edits in the worktree never leak into a measurement. Scripts:
`bench_tick.py` (core.step alone, E 4096, 4v4; now compiled through lean.contract.compile_graph
as the trainer compiles), `tick-audit/bench_advance.py` (the training tick: collect's segments
through lean.learn.compiled, E 2048, 128 ticks, the scripted agent the only league member,
streamed store; CUDA events around each act and advance call), `tick-audit/bench_deal.py`;
checks `tick-audit/static_check.py` (GPU) and `tick-audit/trajectory.py` + `compare.py` (CPU,
between trees).

## Results

Laptop RTX 4070, performance profile, load average 1.0-3.1 (each row's in the jobs' JSON).

**The step alone** (core.step, E 4096, 4v4, fields and fog; job 1, ghost-only tree with
compile_graph, 300-tick blocks, median of 3):

| deflector | old trainer path (`compile_dynamic`) | static (`compile_graph(..., "static")`) |
|---|---|---|
| off | 1.713 ms | **0.572 ms** |
| egg | 1.777 ms | **0.766 ms** |

(The audit measured the old path at 1.92 ms egg on b70dcd18, before ghost-only.) Compile
wall time with 20 warm ticks: static 25 s, dynamic 51-60 s.

**The training tick** (bench_advance, E 2048, 128-tick segments, 4 timed segments after 2
warm; ms per tick; act and advance by CUDA events around each call):

| tree | job | segment per tick | act | advance |
|---|---|---|---|---|
| base cf8a41db (dynamic, scatter in advance) | 1 | 31.78-31.84 | 27.79 | 3.671 |
| + hit labels per segment, dynamic | 1 | 31.14-31.19 | 27.76 | 3.064 |
| + static compile | 1 | 25.23-25.24 | 22.96 | 1.963 |
| same tree again (before item 5) | 2 | 25.22-25.25 | 22.98 | 1.940 |
| + constants per type | 2 | 25.12-25.21 | 23.02 | 1.837 |
| measurement only: no re-deal at all | 2 | 25.03-25.10 | 22.99 | 1.761 |

So the collect loop goes from 31.8 to 25.2 ms per tick (-21%, 64.4k to 81.4k collect steps/s
at E 2048); the static compile also speeds the policy's act graph (27.8 -> 23.0 ms). The game
half (advance) is now 1.84 ms of the 25.2 ms tick; act (both views, the policy, the scripted
agent) is the rest.

## Items

### 1. Compile shapes per run: kept

`LearnConfig.compile_shapes` = "static" (default) | "dynamic", read by
`lean.learn.compiled.graphs(shapes)`, the one place the training graphs compile, through
`lean.contract.compile_graph`. Static: `torch.compile(dynamic=False)` (every size, stride and
run-uniform Python number of the first call specialised: fully static, the environment count
included), then every later call under `torch.compiler.set_stance("fail_on_recompile")`
(StaticGraph; 0.85 us per call): a stray dynamic shape raises at once
(test_a_static_graph_refuses_to_compile_again). Dynamic: `compile_dynamic`, unchanged, for the
evaluation runner (crossover sizes). act, advance, minibatch_values and minibatch_loss follow
the setting; **the league routing (slot_actions) stays dynamic**: its table width W is the
league's capacity ceil(capacity_factor L / present), which changes as members join, so in
training it is a dynamic shape (open point below). bench_tick.py and profile_tick.py compile
through compile_graph (`--shapes`). Numbers: the tables above (step 1.777 -> 0.766 ms egg,
1.713 -> 0.572 off; training tick 31.1 -> 25.2 ms with item 2 in both).

Correctness (job 3, `static_check.py`, 50 states of an eager 4v4 game at E 2048, one step
each from the same state): static against dynamic, ships and seen within 2.8e-5 relative
(to max(|v|, 1)), bullets 5.9e-7, every bullet state, flag and shot outcome identical. A
with_hit_model change (hit radius, spread) on the static graph runs without a recompile and
changes bullets and damage: the hit model and sliders stay runtime rows.

CUDA graphs: possible in principle now that the shapes are static, not trivially safe, not
enabled. advance mutates the league's tallies in place (League.record), the per-tick records
are still read (async copies to pinned memory) when the next tick would replay over them, and
the graphs draw random numbers. Each needs a change first (return the tallies; clone the
records or let the store own the buffers). Gain unknown: act is 23 ms of the 25 ms tick and
looks GPU-bound, so launch overhead may be small; measure before doing it.

### 2. Hit labels once per segment: kept

advance returns each ship's bullet outcomes by age (E, N, K) int8 (`hit_outcomes_by_age`, a
gather in the graph); collect stacks them and `assemble_hit_labels` builds the (2, T, E, N)
labels at the segment's end: for each age a (12), the row block T - a .. 2T - a takes the
elementwise maximum (a bullet ends once, so every other entry is -1). No buffer is written in
the tick. Same labels: test_a_label_lands_on_the_row_its_bullet_left_the_gun (an independent
search over the recorded steps) and the store's segment-boundary test pass unchanged.
Measured: advance 3.671 -> 3.064 ms (-0.61 ms per tick at E 2048, the audit's 0.60), segment
31.8 -> 31.2 ms per tick. Extra memory: T E N K int8 per segment (25 MB at E 2048).

### 5 (audit). Constants per ship type: kept

`Constants.table` is the card's (C, T) table (shared, never copied per slot) and
`Constants.ship_type` (E, N) the slots' types; a row reads `table[row][ship_type]`. take_deal
selects only the types. Eager CPU trajectory (64 environments, 1-4 a side, 300 ticks with
re-deals every 60): bitwise equal to the previous commit. Measured (job 2, back to back):
advance 1.940 -> 1.837 ms (-0.10 ms), segment 25.24 -> 25.17 ms per tick; State 21 MB smaller
at E 4096. The step alone went 0.766 -> 0.692 ms egg and 0.572 -> 0.538 off between job 1 and
job 3 (different jobs: provisional).

### 3b (audit). Reset only finished environments (cap D = 16): dropped

Measured the whole re-deal first: with take_deal removed (measurement only), advance is 1.761
ms against 1.837 ms: **0.076 ms per tick** (0.3% of the tick) is all a re-deal costs now
that the constants table is gone. In the compiled advance the where-select fuses into the
step's output kernels for every block the step writes, and the fresh rows it gathers are a
few distinct rows (the deal index is constant between finished environments), so they come
from cache. A capped path cannot beat that inside a functional graph: the scattered D rows
still go into a new full buffer (an out-of-place index_copy writes every row, as the where
does), and the overflow rule as given (more than D finished: the extra environments take the
full select that tick) without a data-dependent branch means the full select runs every tick
anyway, so the cap only adds a sort and a scatter. Skipping the select on most ticks needs a
branch (torch.cond reads its predicate on the host, a sync per tick) or in-place state
buffers outside the graph. Not worth 0.076 ms; no overflow metric added.

### Deal creation: measured, no pool

bench_deal (E 2048, 4v4 slots, job 2): draw_next_episodes (E deals, personas, learner team,
offsets; once per segment, eager) 1.57 ms, game.deal alone 1.48 ms: **0.012 ms per tick**
over a 128-tick segment. It is launch-bound (269 kernel launches, ~0.7 ms of CPU in
cudaLaunchKernel). A pre-built deal pool would save nothing measurable.

### Small items

- **Item 4, run-uniform floats specialised: kept, by item 1.** The static compile folds every
  run-uniform Python number (the card's `uniform`) into the graph; the correctness check
  above is within rounding and the hit model stays a runtime row. The dynamic path
  (evaluation) still passes them as fp64 scalars: unchanged, a crossover run may see other
  numbers.
- **Item 10, medium: the redundant where dropped (kept: bitwise, 300-tick trajectory; the
  slope is already 0 outside the band).** The stored inverse band was not done: it needs a sixth field row for
  two divides per (point, field), which inductor emits per element either way.
- **Item 9, exp2/log2 powers: dropped.** Within rounding (30 ticks eager: ships within 1e-3 m,
  every event the same) but no gain: core.step static, two rounds back to back (job 3), egg
  0.692 / 0.692 -> 0.698 / 0.706 ms, off 0.538 / 0.556 -> 0.536 / 0.552. Reverted.
- **Item 8, the bullet share once a tick: dropped, not measured.** Since ghost-only it is a
  uint8 compare over 0.2 MB at E 2048 per call; storing it needs a ship column (State layout)
  for an estimated few us.
- **Items 6 (sight on unordered pairs) and 7 (fire's block rewrite): not done.** M effort for
  an estimated 25 us and 40-75 us at E 4096 in a step that is now ~7% of the training tick.

## Open points

- The league routing is a dynamic shape in training: its table width W follows the league's
  capacity. To make it static: W fixed for the run (the capacity at the smallest present
  count, or the league's environments), at the cost of padded rows per slot. For the author.
- The biggest part of the training tick is act (23 ms of 25 ms at E 2048): the policy, both
  views and the scripted agent. The training-path audit covers it.
- CUDA graphs (item 1, above): three blockers listed; measure the launch share first.
- egg_play.py reused `slots` for the bullet slot count inside its tick loop, so the next tick
  drew noise for 12 ship slots (fixed on this branch; its earlier pass numbers ran with it).
