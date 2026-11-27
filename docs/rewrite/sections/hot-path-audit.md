# Hot-path audit: easy simplifications (2026-11-27 afternoon)

Read-only audit of the game's hot path at deflector-lookahead b70dcd18 (lean/game/core.py
step, the rollout's tick halves, the observation), after the egg and bullet clean-ups. The
question (author 2026-11-27): where do we move or rewrite data we do not need to, compute the
same thing twice, or keep bookkeeping the rules do not need. Proposals only; nothing here is
changed. The ghost-only bullet ring (moves-left countdown, the frozen-if-not-flying block, the
counts kernels) is out of scope: branch ghost-only has it.

Method: the compiled step's generated Triton code (TORCH_LOGS=output_code) and torch.profiler
kernel times on the laptop 4070 (performance profile, desktop on the GPU, another agent's
small smoke run holding the GPU lock between my jobs), E 4096, 4v4, the default GameConfig
(fields, fog, ghosts, egg). Four short GPU jobs (scripts and raw output in the scratchpad,
not committed; the code is reproduced in the items where it matters). Kernel times are the
profiler's (about 20% below ncu's). "Measured" means one of these jobs; "estimate" is from
bytes or op counts and says so.

## Ranked list

| # | what | gain | effort | risk |
|---|---|---|---|---|
| 1 | The training and evaluation compile (`compile_dynamic`, duck sizing off) turns every reduction over the ship axis into a looped reduction: the egg's pair kernel materialises nine (E, N K, N) pair tensors (12.6 MB each) | measured: step 1.92 ms vs 1.08 ms with plain `dynamic=True`; pair kernel 966 vs 219 us | S-M (static ship axis: one compile per N; a `torch._check` bound does not work) | none (same graph) |
| 2 | `hit_labels.label.scatter_` inside the compiled `advance` copies the whole segment label buffer in and out every tick (33.5 MB at E 2048, T 128, 4v4) | measured: 0.60 ms per tick at E 2048 for the scatter alone (0.04 ms with a 1 MB buffer) | S | none |
| 3 | `take_deal` rewrites the whole state every tick, the (E, 162, N) constants table (21 MB at E 4096) included, though about two environments in 4096 finish per tick | measured standalone: 0.44 ms per call at E 4096 (constants kernel ~170 us) | S for the constants, M for the rest | none |
| 4 | Run-uniform Python floats enter the compiled step as fp64 scalar arguments, so the scalar arithmetic on them (1 / lift squared, stall speed x (1 + band), ...) runs per thread in fp64, including fp64 divides and `pow` in all four flight kernels | measured with `specialize_float`: 1.078 -> 0.953 ms (-12%; compile_dynamic 1.92 -> 1.82) | S | bitwise or within rounding |
| 5 | Constants per slot (E, 162, N) where they are per type (T, 162): 97 rows are the sustain table, 17 the hit widths, 34 action tables | memory 21 MB -> 1 KB; removes item 3's biggest part; tick reads become a cached gather | M | bitwise |
| 6 | Sight lines: `field_blocks` is symmetric in its two ends but runs on all N x N ordered pairs (diagonal and both orders) | estimate: sight kernel 52-85 us -> about half | M | within rounding |
| 7 | Fire writes the whole bullet block (E, 8, N, K) to change one slot per ship | estimate: up to the fire kernel's 74-89 us, if its select folds into the move | M (needs an inductor experiment) | bitwise |
| 8 | The bullet share (real bullets in flight per ship) is computed three times a tick: fog's snapshot in the step and each of the two `observe` calls in `act` | estimate: ~2 x 20 us | S | bitwise |
| 9 | `softmin`'s fractional powers (`(1 + r^q)^(-1/q)`, the thrust softness 2.83): ~39 `libdevice.pow` per ship per tick in the flight kernels | estimate: 20-60 us of the flight's ~270 us | S | within rounding |
| 10 | `medium`: a division per (point, field) by the band that a stored inverse removes; one redundant `where` | estimate: 5-15 us | S | within rounding |

Below: each item, then things that look wasteful but are cheap, and what was not audited.

## 1. Duck sizing off makes the ship-axis reductions looped (lean/contract.py:93)

`compile_dynamic` (training's `lean.learn.compiled.graphs`, the evaluation runner) sets
`use_duck_shape = False` and compiles with `dynamic=True`. bench_tick.py and profile_tick.py
(today's egg-speed numbers) use plain `torch.compile(core.step, dynamic=True)`, duck sizing on.
Same tree, same machine, back to back (ab.py: compile, 20 warm ticks, 3 x 300 timed ticks,
then a 20-tick profile):

| compile | ms per tick | egg pair kernel | collisions | sight |
|---|---|---|---|---|
| `torch.compile(dynamic=True)` (bench_tick) | 1.079 / 1.079 / 1.080 | 219 us, `triton_per_` (persistent) | not in the top 12 | 52 us |
| `compile_dynamic` (training, evaluation) | 1.920 / 1.916 / 1.927 | 966 us, `triton_red_` (looped) | 74 us | 85 us |

So the trainer does not get the egg-speed win: in the looped form the pair kernel's 5 loops over
the ships each recompute or re-read the pairs, and inductor materialises nine (E, N K, N) pair
buffers (buf7, 67, 136, 138, 144, 148, 149, 153, 156 in the generated `call`: 12.6 MB each at
fp32) to carry values between the loops. A persistent reduction needs a reduction size it can
bound; with duck sizing off the ship count is a free symbol.

Tried first, a bound with the sizes kept dynamic: `torch._check(slots <= 64)` (32v32) at the
top of the step (ab2.py, wrapping `core.step`). It does not take: the pair kernel stays looped
(1.962 ms; with floats specialised 1.88 ms).

Why the bench gets a persistent kernel: with duck sizing on,
the first compile specialises the ship axis to 8 (checked on the CPU, recompiles logged: the
4v4 graph's guard is `noise['spread'] size at index 1 expected 8`; 3v3 then recompiles). So the
bench's 1.08 ms is a graph with N static, and a persistent reduction over the ships needs N
static. Simpler version: compile the game graphs with the ship axis static and the
environment axis dynamic (`torch._dynamo.mark_static` on the N dims, or `mark_dynamic` on E
only). A run's N is fixed (team sizes vary per environment inside the same slots), so that is
one compile per run, plus one per evaluation size: the cost duck sizing off was meant to avoid,
and the reason to decide it rather than just do it. Not yet tried as such (the duck graph is the
evidence). Gain about 0.84 ms per tick at E 4096 on
the training path (measured, the two compiles above); effort S-M (the mark in
`compile_dynamic`'s game callers; the policy graphs keep the current setting). Risk none (the
same graph). Also: bench_tick.py and profile_tick.py should compile the way the trainer does,
so speed work measures what the trainer runs.

## 2. The hit-label scatter copies the whole label buffer every tick (lean/learn/rollout.py:590)

`advance` (compiled) writes `hit_labels.label.scatter_(0, index, outcome)` into a buffer of
2 T E N + 1 int64 (rollout.py:702: 33.5 MB at E 2048, T 128, 4v4) that is a graph input. The
mutation is functionalised and not re-inplaced: on CPU the generated kernel copies the whole
input to a new buffer, scatters into it, and copies the whole buffer back (checked with a
small compiled function under `compile_dynamic`). On the GPU the same: the generated `call`
allocates `buf0` of the label's size, scatters into it and ends with `copy_` back into the
input. Timed (scat_gpu.py: `hit_label_entries` and the `scatter_` as `advance` does them,
E 2048, 4v4, 3% of bullets ending): **0.600 ms per tick** with the segment's 33.6 MB buffer,
0.038 ms with a 1 MB one: almost all of it is the two whole-buffer copies. For scale, the game
step at E 2048 is roughly half of the 1.08-1.92 ms measured at E 4096. Also, every one of the E N K bullets is scattered each tick, about 97% of them to
the one spare entry (same-address stores).

Simpler version: keep the scatter out of the compiled graph: return the per-tick (index,
outcome) (E N K each, 3 MB) from `advance` and do the `scatter_` eagerly after it (in place,
one small kernel), or return the outcomes as a per-tick record and build the labels once per
segment. Effort S. Risk none (the same labels).

## 3. take_deal rewrites the whole state every tick (lean/learn/rollout.py:438, core.py:2790)

`redeal` runs every tick inside `advance`: for every State field, `fresh.index_select(0, index)`
then `where(done, new, old)`. Sizes at E 4096, 4v4 (deal.py): constants table 21.2 MB, bullets
12.6, bullet counts 6.3, seen 2.9, ships 2.5, fields 0.8, pending 0.8, identity 0.5, the rest
under 0.5: 48 MB, each read twice (old, the gathered fresh row) and written once. About 2 of
4096 environments finish in a tick (1,800-tick episodes). Standalone, compiled with
`compile_dynamic`: **0.44 ms per call**, 12 kernels, the largest 173 us (the constants table,
by size), 102 us (bullets), 48 us (counts). In `advance` the select may fold into the step's
output kernels for the blocks the step writes anyway (that still reads every fresh row), but
the constants table has no producer in the tick: its 21 MB are read twice and written once
every tick to change nothing in 99.95% of the rows.

Simpler version: (a) S: the constants depend only on the slot's type, which the deal already
selects (`identity`): take the constants by type (`ship_constants(config, type)` after the
select, or item 5) instead of selecting the table. (b) M: select only the finished rows: a
fixed number D of finished environments per tick (D like 64, a static size; the k-th finished
environment takes deal k already) gathered with `topk`/`sort` on `done` and written with
`index_copy`, so the traffic is D rows, not E. Risk none (the same states). Gain: (a) about
the constants kernel (~170 us standalone at E 4096; half at training's E 2048, estimate);
(b) most of the remaining ~0.27 ms.

## 4. Run-uniform floats are fp64 runtime scalars, not constants (lean/contract.py:136, core.py)

The design says a constant equal for every ship is "a Python number in `uniform`, compiled into
the graph". Under `dynamic=True` torch does not specialise Python floats: 44 of them arrive
as 0-d fp64 CPU tensors (`get_args` of the generated module), passed to the kernels as fp64
scalar arguments (`.item()` per launch, kernel 4 alone takes 24). Arithmetic between them is
emitted per thread, in fp64, in the kernel: e.g. `_fire`'s `stall_speed * (1 + band)`
(kernel 4: `tmp14 = tmp10 * (1.0 + tmp11)` in fp64), `_kick`'s `inverse_lift * inverse_lift`
(fp64 divide, then `libdevice.pow(x, 2.0)` in fp64). Per thread, per flight kernel: kernel 4
10 fp64 arithmetic, 1 divide, 1 pow; kernel 5 12, 2, 2; kernel 13 8, 1, 1; kernel 14 12, 2, 2.
fp64 runs at 1/64 of fp32 on this GPU, and a double `pow` is a long software sequence, so
these few ops per thread can be a real share of the flight kernels (estimate; the flight is
~270 us in 4 kernels for 32,768 ships). Measured with `torch._dynamo.config.specialize_float
= True` (ab2.py, same tree, back to back):

| compile | floats as fp64 inputs | floats specialised |
|---|---|---|
| `torch.compile(dynamic=True)` | 1.078 ms, 28 launches | **0.953 ms**, 26 launches |
| `compile_dynamic` | 1.92 ms | 1.82 ms |

The flight's reduction kernel (medium + second kick) goes 140 -> 57 us; the bullet move
155 -> 125 us. Correctness of the specialised graph was not checked here (expected bitwise:
the host folds the same fp64 expressions).

Simpler version: specialise the floats of the game graphs (they never change within a run;
`with_hit_model`'s values are table rows, not floats), e.g. `specialize_float` patched while
`compile_dynamic` traces the game functions only (the learner's graphs may take floats that do
change), or precompute every derived scalar (inverse lift squared, 2 nose damping, the store
band's speed) on the card. Effort S. Risk: bitwise or within rounding (the same fp64
arithmetic, done once on the host).

## 5. Constants per slot where they are per type (core.py:827, lean/contract.py:125)

`ship_constants` gathers the card (1, C, T) into (E, C, N) at every deal: C = 162 rows with
the default lineup (`sustain_table` 97, `hit_widths` 17, seven action tables 34, the hit
model 14), every row equal for every slot of one type. With one type in the lineup they are
equal for every slot. The tick reads a few of them (an action table at the slot's stick, two
sustain entries, the hit model's rows at the target); the deal and item 3 move all 162.

Simpler version: keep the card's (T, C) table on the device and read `table[type, column]`
(a gather from a ~1 KB table that stays in L1/L2), as `per_type` already does for the steady
table (`PER_TYPE_TABLES`). State shrinks by 21 MB at E 4096, the deal and take_deal stop moving
it, and the tick's per-slot constant reads become cache hits. `with_hit_model` writes the
(T, C) table instead. Effort M (Constants' accessors, the deal, with_hit_model, tests). Risk
bitwise (the same numbers).

## 6. Sight lines on ordered pairs (core.py:2836, 2881)

`sight_pairs` calls `field_blocks` on the full (E, N, N) pair grid (target x observer), F 10
fields each, and masks to cross-team pairs afterwards. Whether the fields' cores block the
segment between two ships does not depend on which end is the observer (the "exactly one end
in a core" and the "segment passes within the core" tests are symmetric in exact
arithmetic), and the diagonal is never needed. Kernel: 52 us (duck) / 85 us (compile_dynamic),
plus a 13 us per-point kernel.

Simpler version: run `field_blocks` on the N (N - 1) / 2 unordered pairs (a fixed pair list
built once per N, like `slot_indices`), then read it back for both orders by a fixed index
map. 28 of 64 pairs at 4v4 (estimate: the sight kernel about halves). Effort M. Risk within
rounding (the swapped formula's rounding differs at an exact tangency).

## 7. Fire rewrites the whole bullet block (core.py:1685)

`torch.where(written[:, None], launched[..., None], state.bullets)` writes all E x 8 x N x K
floats (12.6 MB) to change one slot per living ship; kernel 1, 74-89 us. Its consumers (the
move, the hit test's start positions, the bookkeeping) each read it back. After ghost-only
lands, this is the one remaining extra full pass over the block (the move writes it, the
deflect stacks it once). The select is two loads and a compare per element, so inlining it
into its readers costs nothing, but inductor realises it because it has several consumer
kernels; cut C (progress.md 14:16) shows that making the pair kernel recompute bullet columns
can lose. Simpler version to try: feed the move the select's inputs directly (the move kernel
already reads every column), and give the hit test only the two start columns as their own
small select. Estimate: up to the kernel's 74-89 us, if inductor keeps it fused. Effort M.
Risk bitwise.

## 8. The bullet share three times a tick (core.py:2923, 3031)

`_snapshot` (the fog's seen rows, in the step) and `observe` (twice in `act`, once per team)
each compute `((bullet_moves_left > 0) & bullet_real).float().mean(-1)` over the bullet
counts (kernel 23, 23 us in the step; the two in `act` are in another graph, estimate ~20 us
each). Simpler version: the step stores it once (a ship column or a step output the next
`act` reads), and `observe` reads it. With ghost-only's state byte the computation changes but
the triple count does not. Effort S. Risk bitwise.

## 9. softmin's fractional powers (core.py:882)

`softmin(a, b, q) = low (1 + (low / high)^q)^(-1/q)`: six per `_setting`, `_setting` twice
per kick, two kicks, plus `_path_lift` and the fire's thrust. With q = 16 the inner power
becomes multiplications; the outer `^(-1/16)` and the thrust softness 2.83 (both powers) stay
`libdevice.pow` (accurate powf: special-case handling and a long dependency chain): 39 per ship
per tick across the flight kernels (15 + 5 + 14 + 5). The flight is ~270 us for 32,768 ships:
far from the GPU's arithmetic peak, so it is latency-bound (few threads, long dependent
chains, high register use); shorter chains help directly. Simpler version: for x > 0,
`exp2(y * log2(x))` (two MUFU ops) in `softmin` and in `_rates`' drag rise and separation
powers. Estimate 20-60 us. Effort S. Risk within rounding (not bitwise).

## 10. medium per (point, field) (core.py:904-948)

Per bullet and ship, per field: `(distance - radius) / band` and `/ (band * distance)` (two
divides), and `slope / where(in_band, complement, 1.0)` inside a `where(in_band, ..., 0.0)`
whose outer `where` is redundant (`slope` is already 0 outside the band). Simpler version: an
`inverse_band` row on State.fields written at the deal (fields are fixed in an episode), one
divide by the distance, the inner `where` only. 4.3 M (point, field) pairs a tick at E 4096
(estimate 5-15 us of the bullet move's 155 us and the flight's medium). Effort S. Risk within
rounding.

## Looks wasteful, is cheap (or already handled)

- Identical expressions computed twice in one function pair (the fire's thrust softmin and the
  first kick's, `_path_lift`'s `most` and `_setting`'s budget at the same speed): the FX graph's
  common-subexpression pass merges them; the generated kernels compute each once.
- `collisions` returns six (E, N, N) tensors and the step reads only `damage`: dead outputs are
  dropped by the compiler.
- `hit_taken`, `hit_damage` and the fog's `real_hits` are three scatters over the same targets
  (core.py:1536, 1606, 1616): they run as atomics inside the pair kernel; `hit_taken` could be
  `hit_damage`'s column sum, but the saving is a few us.
- The ship block `where(alive0, stack([...]), state.ships)` (core.py:1551): one write of
  2.5 MB, the right shape already. The fog's `seen` select (19 us) likewise.
- Stored derivables (`velocity_x/y`, `angle_cos/sin`, the bullets' cached end medium): each
  saves more work than its row costs; the cached medium is what keeps fields to one medium per
  bullet per tick.
- int64 `identity`, `pending`, `tick`: under 1 MB together; only item 3 moves them per tick.
- `observe` builds a full Obs per team per tick: (E, N, ~30) floats, small next to the step.

## Not audited

The scripted agent (`lean.scripted.opponent.explain`, its custom pair kernel) beyond a skim;
the policy step; `deal` itself (once per segment). Numbers here are provisional: one quiet
laptop, another agent's GPU jobs queued between mine (never concurrent: the GPU lock).
