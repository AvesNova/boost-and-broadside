# Second training hot-path review (queue item 4, 2026-11-28)

Measurements and proposals on core 679a5ada (items 8a-8c merged: dead ships option C, the
visible-ship auxiliary rows, the geometry heads at K = 8). Companion to the first review
(`training-path-audit.md`, whose items 1-3 and part of 9 landed as the training-path fixes)
and the tick audit (`hot-path-audit.md`); their findings are not repeated. Nothing in lean/ is
changed; item 5 implements. Scripts, the job list and the queue runner are in
`training-review-2/`; raw JSON lines and traces in `data/review4/` (not committed); the
progress note is `training-review-2-progress.md`.

Method. Laptop RTX 4070 (8 GB, 36 SMs), performance profile, every job its own `bnb-4-*`
unit under the GPU lock, one at a time (`queue.sh`), the profile, load average and other
units recorded per job (load 1.9-3.8 at the job starts: the jobs' own compile workers; no other
bnb unit active). The update: as `heads-8c/cost.py` (128 environments, one 128-tick segment
collected eagerly, 192 views = three micro-batches of the recipe's 64 views x 128 ticks, 4v4, 10
fields, bf16, compiled static, one epoch of one minibatch; median of 3-6 steady updates after the
compiling one), so "per micro-batch" is the steady update over 3 and includes the target pass
and one Adam step (both under 0.5% of it). The rollout: the trainer's own stage set-up
(`lean.train.new_stage`: BC with its yardstick-only league, RL with the league opened, the EMA
playing in two slot calls a tick), 2,048 environments, compiled collect streamed into a store,
one compiling then one timed 128-tick segment. Profiles: torch.profiler over one steady update
(3 micro-batches) and one 16-tick segment (17 act calls: the bootstrap act included). "Measured"
means one of these jobs; an estimate says what it is computed from.

## Profile

### The update (measured)

| | geometry heads off | on (K = 8, the default) |
|---|---|---|
| per micro-batch (steady) | 150.2 ms | 165.6 ms (+15.4, +10.3%) |
| GPU busy in the profiled window | 98.1% | 98.2% |
| kernels per micro-batch | 811 | 1,001 |
| peak allocated | 4.25 GiB | 4.60 GiB |
| host floor: the same graph at 1 view (B = 1), wall per micro-batch | | 6.0 ms |

Kernel time per micro-batch by family (kernel names; a fused kernel counts once, under the
first family that matches, so the split is rough):

| family | off (ms) | on (ms) | share (on) |
|---|---|---|---|
| GEMMs (cuBLAS / CUTLASS) | 42.0 | 46.3 | 28.2% |
| other pointwise and reductions (rotary cats and slices, casts, residual adds, their backward) | 28.4 | 29.3 | 17.8% |
| RMSNorm forward and backward (fused_rms_norm) | 20.1 | 23.5 | 14.3% |
| GELU / gated MLP | 16.6 | 18.1 | 11.0% |
| sigmoid gates (RG-LRU, attention output gate, the pair head's gates) | 12.3 | 13.8 | 8.4% |
| attention (memory-efficient SDPA, forward and backward) | 11.5 | 11.5 | 7.0% |
| losses (log-softmax, HL-Gauss erf) | 6.3 | 7.3 | 4.4% |
| linear scan (Triton forward and backward) | 5.5 | 5.4 | 3.3% |
| rotary / position code (cos, sin, atan2) | 3.8 | 3.9 | 2.4% |
| top-k, gather, scatter | 0.3 | 3.0 | 1.8% |
| copies (the streamed store's H2D on the copy stream, overlapped) | 2.2 | 2.2 | 1.4% |
| Adam (fused) | 0.1 | 0.1 | 0.1% |

What changed since the first review: the doubling scan's 12.1% is now 3.3% (the Triton linear
scan), the step is static, and the geometry heads add 15.4 ms. The model is still memory-bound:
GEMMs are 28% of the step; pointwise, norm, gate and GELU kernels together are about half.

Where the geometry heads' 15.4 ms goes (on minus off, by family): GEMMs +4.3 (the two gated
per-ship projections of every slot, the field projection, the last layers), norms +3.4, top-k /
gather / scatter +2.7 (the `take` gathers and their backward scatter-adds, an aten fallback,
1.6 ms a micro-batch; the top-k fallback 0.15 ms), sigmoid +1.5, GELU +1.4, losses +1.0, other
+0.9; 190 more kernels.

### The rollout (measured)

| | BC tick | RL tick (EMA playing, 2 slot calls) |
|---|---|---|
| steady segment, ms a tick (2,048 environments) | 25.34 (80.8k steps/s) | 27.70 (73.9k steps/s) |
| GPU busy in the profiled window | 98.1% | 97.3% |
| kernels a tick | 466 | 828 |
| host floor: the same graphs at 16 environments, ms a tick | 2.94 | 5.23 |
| first (compiling) segment, cold / warm cache | 94.3 s / 14.9 s | 112.1 s / not run |
| peak allocated | 0.97 GiB | 1.16 GiB |

Store writes: 241 device-to-pinned copies per 16 ticks, 12.9 MB a tick, 1.25 ms a tick on the
copy stream (overlapped). Pageable copies: 3 (BC) and 5 (RL) host-to-device per segment (the
segment's set-up), none in the tick; the update has 5 per update (the critic's support built
twice by `value_support` with `torch.tensor`, the order's read-back, the final `.tolist()`)
and none per micro-batch.

### Host versus GPU (the launch-overhead share)

Both loops are GPU-bound on the laptop. The host floor (the same compiled graphs at a size where
the GPU work is small: wall time is then the host's launch and Python time, an upper bound on
it) is 6.0 ms per micro-batch against 165.6 ms of GPU time (3.6%), and 2.94 / 5.23 ms a tick
against 25.3 / 27.7 (12% / 19%). The GPU's idle share in the windows (1.9% BC, 2.7% RL, 1.8%
update) is the most CUDA graphs can return on this GPU. The balance changes on a faster GPU: the
host floor stays, the GPU time shrinks. The RL tick's host floor grows with the slot calls
(5.23 - 2.94 = 2.3 ms for two slot calls and the merge, about 1.1 ms a call): with 12 snapshots
(11 calls, the first review's 33.1 ms tick) it is about 13 ms (estimate), which a GPU about 3x
the laptop's would no longer hide (estimate, not measured on the 5090).

## 1. The loss rows against the decisions (correctness read)

Read: `lean/learn/training.py` (minibatch_loss), `geometry.py`, `returns.py`,
`rollout.py` (act, advance: the records), `store.py` (counts), `lean/model/geometry.py`,
`lean/game/core.py` (observe, team_view, perceive, geometry_record, pair_geometry,
_bullet_ends).

| head / term | rows | decision | |
|---|---|---|---|
| surrogate, entropy, teacher cross-entropy, value | `own` = same team & alive (the seat's living ships) | living ships; dead ships take no gradient | matches |
| value targets of fallen ships | `fallen_team_values`: living team-mates' mean at the tick and level (post-death ticks and the bootstrap); fallen rows are not trained (value loss on `own`) | option C | matches |
| next state | `observed & alive after the step & visible to the seat after the step & ~done` | visible ships, both ticks | matches |
| hit | `observed & label >= 0` (denominator: the store's labelled observed rows, same rule) | visible ships, hidden masked | matches, but see the label bias below |
| map edge | `observed` | visible ships | matches |
| ship pairs | both ships `observed`, upper triangle; symmetric columns one order, train angle and gun error both orders averaged; weight c / min(K, c), divided by the minibatch's valid pairs (c (c - 1) / 2 per tick) | pairs with a hidden or dead ship masked | matches |
| ship-field pairs | ship `observed` x every field (fields always exist in the training game); denominator c F | | matches |

Checked along the way: the observation's slots are the game's slots (no reordering), so the
geometry record, the next-state targets and the hit labels index the same rows; an enemy
`observed` this tick was sighted this tick (`ticks_since_seen == 0`), and the sighting is taken
at the end of the step (`perceive`), so its observed position, heading, velocity and store are
its true ones and the record's true values add nothing the team cannot see; the spare view's
zero record gives 0 / 0 in the flight time, replaced by `where` before the targets (no gradient
flows through targets); `observed` in the store equals `observed_rows` recomputed in the loss
(same fields).

**One issue: a selection bias in the hit labels near ends (small, systematic).** A bullet's
outcome is reported once: at its first contact (hit) or on its last tick (missed, age K - 1,
K = 12) (`_bullet_ends`). Two kinds of rows never receive their miss: (a) the last 11 ticks
before an episode ends (a fresh deal's slots never report, so bullets in flight at the end are
dropped), (b) the last 11 ticks of an update's last segment (collect drops segment 0's
`previous_hit_label`: those rows belong to the previous update, already trained). In both
windows a bullet that hits within its flight is labelled and one that misses is not, so the
labelled rows there are enriched for hits. Size (estimate): window (b) is 11 of 512 seat-ticks
(2.1%) and window (a) about 11 per episode; the hit head sees a hit share inflated by roughly
2-4% relative (1.10% read as about 1.13%), concentrated near episode ends where kills
cluster. Fix (S, no recipe change in spirit): mask every row within K - 1 ticks of an episode
end or the update's end from the hit loss, labelled or not (a `resolvable` record), or label
in-flight bullets at an episode end as missed if that is what the game means. For the author:
which of the two.

Nothing else found wrong in the rows. Two observations, not bugs: the auxiliary and pair
heads now train on visible enemies, so the first review's item 7 (heads on own rows only)
applies only to the policy and value heads; the pair head's per-ship projections run on all
8 slots, also the masked ones (needed for the gather at fixed shapes).

## 2. Activation checkpointing and the micro-batch curve

TBD

## 3. Compiler modes per graph

TBD

## 4. The CUDA-graph rewrite

**What the grid shows (measured).** `mode="reduce-overhead"` (inductor's CUDA-graph trees) fails
on both kinds of graph as the code stands. The rollout: "accessing tensor output of CUDAGraphs
that has been overwritten by a subsequent run" in `act` (rollout.py:616) during the first
segment: a tick keeps outputs of earlier replays alive (the state and carry chained from tick
to tick, the hit outcomes in a Python list until the segment ends, the store's asynchronous
copies of the tick's records, the league's tallies reassigned as attributes). The update:
"accessing gradient tensor output of CUDAGraphs that has been overwritten": with gradient
accumulation over a minibatch's micro-batches, the first micro-batch's gradients are the
graph's outputs and the next replay overwrites them. With one micro-batch per minibatch (no
accumulation) the loss graph runs: see section 3.

**Is it worth it?** The upper bound on the laptop is the GPU's idle share: 1.9% of the BC tick,
2.7% of the RL tick, 1.8% of the update (measured windows), so about 0.5-0.75 ms a tick, 0.3-0.4
s of a ~130 s update (estimate). The host floor says where it would matter: 2.94 ms (BC) and
5.23 ms (RL, two slot calls) a tick, and about 1.1 ms more per network slot call; 6.0 ms per
micro-batch for the update. The update stays GPU-bound on any GPU up to ~25x the laptop's
(165.6 / 6.0; estimate), so **no CUDA graphs for the update**. The rollout becomes host-bound
once the GPU's tick time falls below the host floor: on a GPU 3-4x the laptop's (a 5090-class
estimate: 25.3 / 3.5 = 7 ms a BC tick) the BC tick is still GPU-bound, the RL tick with 2 slots
is near the edge, and with 12 snapshots (about 13 ms of host a tick, estimate) host-bound.
Measure the tick on the 5090 (BC, RL with 12 snapshots, `profile_collect.py` and its
`--envs 16` host floor) before building it.

**What it takes, file by file** (a manual `torch.cuda.CUDAGraph` capture of the whole tick,
around the inductor graphs compiled without inductor's own CUDA graphs; simpler to reason
about than CUDA-graph trees, and every address fixed):

- `lean/learn/rollout.py` (most of the change):
  - the tick's state (`State`, `CollectCarry`) in static buffers, two sets used ping-pong
    (tick t reads set A and writes set B, tick t + 1 the reverse: two captured graphs, no
    per-tick copy of the 151 MB of recurrent carries);
  - the hit outcomes written into a preallocated (T, E, N, K) int8 buffer at the tick's index
    (a device counter advanced in the graph) instead of a Python list;
  - the tick's store records written into a device staging slot (two slots, ping-pong) inside
    the graph; the device-to-pinned copies stay eager on the copy stream after each replay
    (about 20 `cudaMemcpyAsync` a tick: cheap on the host), since a captured copy cannot move
    its destination slice from tick to tick;
  - the league side: the slot calls are a fixed list per segment (League.prepare), so the
    graph is captured once per table layout (at most once per update, when a load or unload
    changes the list; a capture costs about one tick's host time plus instantiation); or,
    better for the host, the first review's item 9: the network slots as one batched call
    over stacked weights, one graph for every layout.
- `lean/learn/pool.py`: `League.record` and `League.draw` update fixed device buffers in place
  (`games`, `wins`, `draws`, `dealt`, `overflow_games`: `mul_` / `index_add_`) instead of
  assigning new tensors to attributes (a side effect replayed after a compiled graph, which a
  captured graph never replays); the host-side readers (`allocate`, `summary`, checkpoints)
  read them between segments as now. "League tallies out of the graph" in the spec's words:
  either this, or the tick returns the finished games' (member, won, drawn) and the tally runs
  eagerly once per segment from a (T, L) record.
- `lean/learn/rollout.py` `redeal` / `NextEpisodes`: `taken` a static counter updated in place.
- Random draws: the compiled graphs draw from the default CUDA generator (`random_keywords`
  without a generator), which is capture-safe (the generator's offset advances per replay;
  torch.cuda.graph registers the default generator). The large per-tick draw (sample_joint's
  (S, N, 42) uniforms, 4 MB a tick) should stay in the graph; drawing a segment's uniforms as
  inputs up front would be 0.5 GB a segment. "Random draws as inputs" is needed only where a
  user generator is passed (eager and test paths), which the captured path does not use.
- `lean/learn/store.py`: `write_tick` from the staging slot after the replay (no
  `record_stream`: the sources are static); otherwise unchanged.
- `lean/learn/compiled.py`: act, advance, slot_actions, merge_league compiled as now (static,
  default mode), plus a `CapturedTick` (warm-up on a side stream, capture, replay); the
  recompile tripwire stays.
- The update: no change (not worth it). If ever wanted: gradients accumulated in place into
  preallocated `.grad` (`zero_grad(set_to_none=False)`), the micro-batch's inputs from the
  store's two device staging slots (two captured graphs), the measures accumulated in place.
- Tests: the captured collect equals the compiled collect (same draws: the same default
  generator seed) over a segment with episode ends, league loads and a retention drop; the
  league tallies equal the uncaptured ones.

Effort M-L (the state flattening and the ping-pong are most of it). Risk: within rounding
(bitwise with the same draws, since the kernels are the same). Gain on the laptop under 1%
(measured bound); on a faster GPU it depends on the host floor above (not measured there).

## 5. Host copies per micro-batch, and what else the profile shows

**Host copies (measured).** No pageable copy and no stream synchronisation remains per
micro-batch: the update's 5 pageable host-to-device copies (2,568 bytes) and 2 read-backs are
per update (the critic's support made twice with `torch.tensor` in `value_support`, the data
order's `.cpu()`, the measures' `.tolist()`), and its stream synchronisations are the store's
`ready()`, the order read and the final read. What remains per micro-batch is the streamed
store's own work: the host's `index_select` of the micro-batch's views into a pinned staging
slot (about 20 MB a micro-batch: 19 records and the carry) and its host-to-device copy on the
copy stream (2.2 ms of copy-stream time a micro-batch, 1.4% of the window, overlapped), plus one
event wait for the staging slot's copy of two micro-batches before (6 `cudaEventSynchronize`
per 3 micro-batches; the compute stream never waits, the first review measured 0.00 s). The
first review's host gather (2.2-3.1 s an update) runs beside 127 s of GPU work. Nothing to do
on the laptop. On a GPU where the micro-batch takes less than the host's gather (about 3-4 ms,
the first review's number over 768 micro-batches), keep the store on the device
(`stream_rollout=False`: 3 GB more device memory) or gather on the device from a pinned
mapped buffer. The per-update pageable copies: build the support once on the device (S, ~0.1 ms).

**Small eager work around the loss graph (measured, per micro-batch).** About 93 `add` kernels
(the gradient accumulation into `.grad`, one per parameter tensor, 0.17 ms), the store's widening
casts and resident `index_select`s, the deal-table look-ups, `draw_keys`' two `rand`, the
carry's `movedim(...).contiguous()` (2.4 MB): together under 1 ms of GPU time and a few hundred
microseconds of host time. Not worth a change.

**New in the profile:**

- The geometry heads' fixed cost (15.4 ms a micro-batch, 10.3%) is the largest new item. See the
  fold prototype in the ranked list (geometry_fold.py) and the gather backward (an aten
  scatter-add fallback, 1.6 ms).
- Rotary and its concatenations: the spatial sublayer's `torch.cat([rotate(q, k, v), gate])`
  and the keys' and values' `cat` with the map memory materialise copies of the projections;
  the kernels named for them (cat / slice / unbind / stack, forward and backward) total about
  15 ms a micro-batch (9%; from the top-kernel list, rough).
- Attention: memory-efficient SDPA on 9 queries x 19 keys a head (11.5 ms, 7%), plus the
  additive mask's expand / pad kernels in the rollout. See the math-attention prototype.
- The rollout's tick: 466 kernels (BC), 828 (RL); the scripted agent's `_pairs` Triton kernel
  1.4 ms a tick and its large fused gather 1.4 ms (the lookahead agent; not reviewed here).
- max-autotune's GEMM autotuning is off on this GPU ("Not enough SMs to use
  max_autotune_gemm mode": 36 SMs); the grid's max-autotune numbers here are pointwise and
  reduction autotuning only. On the 5090 (170 SMs) GEMM autotuning would apply: re-run the
  grid there before choosing.

## Ranked proposals for item 5

TBD
