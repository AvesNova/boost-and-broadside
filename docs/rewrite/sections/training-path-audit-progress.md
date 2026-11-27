# Training-path audit: the approved fixes (branch train-audit, 2026-11-27 evening)

The author's decisions on the training-path audit (training-path-audit.md on
deflector-lookahead e88cf849, ~17:30): the league table at a fixed width (its dynamic-shape
bug 2), reset_carry, the host syncs, the linear scan, the norm outputs. Branch `train-audit`
from tick-audit 996fdc10 (which made the five learner graphs static except the league table
width W).

Method: every timing is one locked GPU job (flock on data/gpu.lock, systemd-run, MemoryMax
12G), laptop RTX 4070, performance profile, load average recorded per row (1.3 to 3.4 at the
starts). Before and after run back to back in one job, each from an exported tree of a fixed
commit (`git archive`, PYTHONPATH). All on the trainer's path: `lean.learn.compiled` graphs
(compile_graph static, fail_on_recompile), real collect and update, LearnConfig defaults (E
2,048, 4 x 128 ticks, 32 minibatches, 64-view micro-batches, streamed store). Scripts in the
scratchpad (not committed): `bench_collect.py` (RL collect with 1 or 12 snapshots: the EMA and
snapshots drawn into the league games by the league's own draw; ms per tick over two timed
128-tick segments), the tick-audit's `bench_advance.py` (collect without league networks: act
and advance by CUDA events), `htod.py` (pageable host-to-device copies and stream
synchronisations per tick: the profile of a 16-tick collect minus an 8-tick one),
`bench_step.py` (the update's micro-batch step, minibatch_loss forward and backward, static, on
a saved real batch: 64 seats x 128 ticks of an eager collect; 3 warm, 20 timed; weights at
initialisation and after 60 BC Adam steps; every gradient saved and compared between trees).

## Results

| item | what | before | after | kept |
|---|---|---|---|---|
| 1 | league table fixed for the run | W and K follow the members; mid-run compile | one graph from the RL stage's first tick | kept |
| 1 | RL collect, 1 snapshot (ms per tick) | 30.07 / 30.16 (2 calls x 427 rows) | 38.56 / 38.58 (18 calls x 86) | (cost) |
| 1 | RL collect, 12 snapshots (ms per tick) | 36.92 / 36.96 (11 calls x 107) | 38.78 / 38.79 (18 calls x 86) | (cost) |
| 1b | routing merge once (part of 1) | 42.80 / 44.50 | 40.30 / 40.49 | kept |
| 1c | all slots in one vmap call | 38.78 / 38.79 | 38.92 / 39.01 | dropped |
| 2 | step `reset` instead of reset_carry: advance | 1.838 ms | 0.554 ms | kept |
| 2 | same: act / whole tick (no league networks) | 22.98 / 25.13 ms | 23.45 / 24.31 ms | kept |
| 3 | pageable H2D copies in the collect loop | 1.0 a tick (1 sync) | 0 (0 syncs) | kept |
| 3 | same: tick | 24.31 ms | 24.19 ms | kept |
| 4 | linear scan: micro-batch step (init weights) | 170.03 ms, 5.12 GiB | 146.79 ms, 4.17 GiB | kept |
| 4 | same, trained weights | 169.90 ms | 146.88 ms | kept |
| 5A | norm outputs bf16, residual float32 | 146.79 ms | 146.70 ms, the same gradients | dropped |
| 5B | bf16 residual stream (measured only) | 146.88 ms, 4.17 GiB | 139.06 ms, 3.98 GiB | not adopted (decided) |

The whole branch, final tree: see "Final" below.

### 1. The league table at a fixed width: kept

The decision (core-decisions.md "League routing", 2026-11-23): each opponent a fixed row
capacity (about 1.25 x an even share), draws within the capacities, rows sorted by opponent on
the GPU each tick, a replaced snapshot finishing its running games in a retiring slot. Done:

- Capacity fixed for the run: every network member C = ceil(1.25 L / members), members = 3 +
  snapshot_members = 15, so C = 86 at L 1,024; the scripted agent's capacity is L. The table
  width W = C. Before, the capacity was ceil(1.25 L / members present), so W and the slot list
  changed whenever a member joined (W 427 with 3 present, 86 with 15).
- All K = 2 + 12 + 4 slots exist from the RL stage's start (`League.open`, called by
  run_stage for the rl stage, and by after_update and load): an empty slot runs a placeholder
  (a copy of the policy's weights at open) on no rows. The tick makes K slot calls, every tick
  of the stage; `maybe_mark_dynamic` on W is gone (slot_actions is a plain static graph).
  The BC stage never opens the league (no snapshots there), so it runs no network.
- The slot calls return only their W rows; one `merge_league` graph puts all K slots' rows
  back (before, each call rewrote the whole opponent carry, 37.7 MB, with a where: 42.8 ->
  40.4 ms per tick).
- Test: test_the_league_routing_is_static_for_the_whole_run (the training static compile,
  fail_on_recompile: one compile at the open league before any snapshot, the same graph after
  2 and after 4 snapshots, table (18, 2) throughout).

Where the code departs from the decision, for the author:
1. The scripted agent is not capped (capacity L). With C fixed at 1.25 x L / 15, the network
   members present early in RL hold at most C each, so the rest of the league games go to the
   scripted agent: with the EMA and one snapshot, 852 of 1,024 league games are scripted
   (before: about a third each). The share evens out as snapshots join (12 snapshots: 96
   scripted). If the even share is meant over the members present, W cannot be fixed without
   running every slot at the largest W (427 rows x 18 slots).
2. "Rows sorted by opponent": each slot's rows are taken in environment order by a cumulative
   count per slot (route_rows, merge_routed), which is the order a stable sort by slot gives;
   no explicit sort.
3. A new snapshot with no free slot still re-deals the games of the retiring slot with the
   fewest (League.abandoned counts them; unchanged, should stay 0).

The cost: the work is the rows run, 18 x 86 = 1,548 every tick, against 677 (1 snapshot) and
about 1,177 (12) before. Early in RL the tick is 8.5 ms slower (30.1 -> 38.6 ms), late in RL
1.8 ms (36.9 -> 38.8); per update (512 ticks) 4.4 s and 0.9 s. A single batched call (vmap over
the stacked weights, now possible since K and W are fixed; the NOTES' 10.5 against 12.75 ms
at 8 slots was before) measured no gain (38.78 -> 38.92 ms, 12 snapshots; 38.56 -> 38.67, 1):
the calls are compute-bound, so the placeholder slots' rows are real work. Dropped (reverted).
The lever left is not running the empty slots (open point).

### 2. reset_carry: kept

redeal no longer multiplies both carries (151 MB) by ~done every tick; the rollout passes
carry.reset to the step (Policy.step's `reset`, as the update's sequence reads the stored
reset), the league's networks too (routed_step's `reset`). Tests:
test_a_step_with_reset_equals_a_step_from_a_zero_carry (outputs and new carry within 1e-6),
the league routing test and collect/update tests unchanged. Measured (bench_advance, job 3):
advance 1.838 -> 0.554 ms (more than the audit's 1 ms estimate), act 22.98 -> 23.45 ms (the
step's `fresh` is now a tensor, not a folded constant), the tick 25.13 -> 24.31 ms (-3.3%).

### 3. Host syncs: kept

The critic's support (`returns.value_support`: centres and knees) is built once per collect and
per update outside the graphs and passed in (act, minibatch_values, minibatch_loss take
`support`); the next-state support is two non-persistent policy buffers; the scripted agent's
index and time tables are made on the device (`lean.contract.device_constant`, a fill per
entry: the same values and dtype as torch.tensor). Under the static compile most of the
audit's 18 copies a tick (counted under compile_dynamic) were already gone: measured 1 copy and
1 stream sync per tick in the collect loop on tick-audit, 0 and 0 after. The value centres are
bitwise equal to the old in-graph build (compared on the GPU). The tick 24.31 -> 24.19 ms. The
micro-batch step still shows 1 pageable copy per micro-batch, before and after (not found).

### 4. The linear scan: kept

`lean.model.scan.linear_scan`: a custom operator (torch.library) with a Triton kernel forward
(one thread per channel, sequential over T, float32 state) and backward (the reversed
recurrence g_t = grad_t + decay_(t+1) g_(t+1)), a loop on the CPU; one tick (the rollout's
step) is the plain product and sum, fused by the graph. It replaces the doubling scan in the
RG-LRU and in GAE; the doubling scan is gone (kept only in the test as a reference). Under
compile it is one opaque node at any shape, static or dynamic, so it does not depend on the
prototype associative_scan; bf16 autocast leaves it float32 (inputs cast to float32).

Measured (bench_step, jobs 2-3): micro-batch step 170.03 -> 146.79 ms at initial weights,
169.90 -> 146.88 ms with trained weights (-13.6%); peak allocated 5.12 -> 4.17 GiB (autograd no
longer keeps the doubling rounds). Against the doubling scan on the same real batch: loss
1.9369606 -> 1.9369595 (trained), the gradients' relative norm difference 2.1e-3 (8.5e-4 at
initial weights), largest difference 1.2e-4 against a largest gradient of 1.7e-2, every
tensor's cosine at least 0.999986; the audit measured the same largest difference between
the static and dynamic compiles of one graph (bf16 kernel order). Tests:
test_linear_scan_equals_the_loop_and_the_doubling_scan (forward and every input's gradient
against a float64 loop and the doubling scan, T 1, 2, 3, 5, 128, resets inside the sequence
and at its first tick), test_the_scan_kernels_equal_the_loop (the same on the GPU), and
test_the_scan_compiles_static_and_dynamic (fullgraph, eager-equal states and gradients).
Options: the audit's associative_scan Function measured 144.4 ms on a 168.0 ms base
(-14.1%) against -13.6% here (different bases; see "associative_scan" below); a chunked
scan was not tried (more passes over the tensor than one sequential pass).

### 5. Norm outputs: step A dropped, step B measured only

Step A (norm outputs that feed a Linear cast to bf16 after the float32 reduction; the
residual stream and the map memory float32): 146.79 -> 146.70 ms, and the loss and gradient
norm equal to every printed digit: every such norm feeds a Linear (or a gelu then a Linear)
that cast to bf16 anyway, and inductor had already fused that cast into the norm kernel.
Reverted (no gain, one more class).

Step B (measured only, not adopted: the residual stream bf16 too, the encoder's and map
memory's norms bf16): 146.88 -> 139.06 ms (-5.3%), peak 4.17 -> 3.98 GiB. Numerics against
float32 residuals (trained weights, same batch): loss 1.9369595 -> 1.9370025 (4.2e-5, the scan's
change was 1.1e-6), gradients' relative norm difference 1.3e-2 (the scan's 2.1e-3), smallest
per-tensor cosine 0.99948. A training-scale A/B would be the test; the decision (2026-11-23)
keeps the residual float32.

## Final

(job 4, filled below)

## Tests and compile check

(filled below)

## Open points

- The empty league slots: the fixed table runs 18 x 86 rows every tick, 8.5 ms a tick early in
  RL and 1.8 ms late against tick-audit. Not calling the slots that hold no weights (the host
  knows them; the same graph, no recompile; compiled at the stage's start by one call) would
  take back most of it, but makes the number of calls vary: the author's call.
- The scripted agent's uncapped capacity and the early-RL league mostly scripted (item 1,
  departure 1).
- One pageable host-to-device copy per micro-batch remains (item 3).
- Step B (bf16 residual): 5.3% of the step, a numerics decision.
