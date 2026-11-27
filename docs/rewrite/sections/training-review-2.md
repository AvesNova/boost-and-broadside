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

## 2. Activation checkpointing, selective recompute, and the micro-batch curve

`memory_curve.py`: the compiled update over 384 views (256 environments, one segment), one
epoch of one minibatch, at micro-batch B views, three ways: plain (the trainer's graph);
**checkpoint** (every residual sublayer, 2 temporal and 4 spatial, under non-reentrant
`torch.utils.checkpoint`: per-block checkpointing at sublayer grain, only each sublayer's input
kept); **budget 0.5** (`torch._functorch.config.activation_memory_budget = 0.5`: the
partitioner recomputes activations, cheapest element-wise ones first, to keep at most half the
saved bytes: inductor's selective recompute). Each its own process and empty cache; the
losses agree to 6-7 digits across all of them (the same minibatch; different pair draws).

| B (views) | plain: ms per 64 views | plain: peak allocated | checkpoint | budget 0.5 |
|---|---|---|---|---|
| 8 | **140.9** (-14.6%) | | | |
| 16 | **147.8** (-10.4%) | 1.21 GiB | | |
| 24 | 154.1 (-6.6%) | | | |
| 32 | 158.8 (-3.8%) | 2.35 GiB | | 185.0, 1.79 GiB |
| 48 | 163.0 (-1.2%) | 3.50 GiB | | |
| 64 (recipe) | 165.0 | 4.61 GiB | 198.7 (+20.4%), 2.98 GiB | 190.6 (+15.5%), 3.46 GiB |
| 128 | out of memory | | 201.8 per 64 (+22%), 5.89 GiB | out of memory |
| 256 | out of memory | | out of memory | out of memory |

(B 4 at 8 environments: 159.6 ms per 64 views, 10.0 ms per micro-batch against the 6.0 ms host
floor at B 1: host-bound from about B 4 down.)

**The curve runs the other way from the usual one: on this GPU smaller micro-batches are
faster per view**, down to at least B 8. The profile at B 16 (`profile_update.py --b 16`, 12
micro-batches) against B 64, kernel time per 64 views: GEMMs 37.6 against 46.3 ms (-19%),
"other pointwise" 23.8 against 29.3, norms 22.0 / 23.5, GELU 16.7 / 18.1, sigmoid 12.1 / 13.8,
attention and losses about equal; the kernel count per micro-batch is the same (about 1,000) at
every B, so B 16 launches 4x as many per view, and the GPU's busy share falls from 98.2% to
95.0%. The likely cause is the L2 cache (32 MB on this GPU): at B 64 a float32 (tokens x width)
activation is 37.7 MB, at B 16 9.4 MB, so a norm's output is still in L2 when the next GEMM or
pointwise kernel reads it. (Not proven: no L2 counters read.) Below B 8 the host's fixed ~6 ms a
micro-batch takes over.

Checkpointing and the memory budget work (memory down 35% and 25% at B 64) but cost 20% and
15% of the step, and buy nothing here: B is already limited by speed, not memory, and the faster
small micro-batches use less memory too (1.2 GiB at B 16). They would matter only to run a large
B on a small GPU, or to keep the whole rollout on the device (`stream_rollout=False`) on the
laptop. Not proposed.

On the 5090 the optimum moves (its L2 is 96 MB, and its faster kernels make the ~6 ms of host
per micro-batch binding at a larger B): measure the same curve there (`memory_curve.py plain B`)
and set `micro_batch_views` per machine. The recipe's numbers do not change with B (a
minibatch's gradient is the sum over its micro-batches; within rounding), apart from which
geometry pairs are drawn (a fresh draw per micro-batch either way).

## 3. Compiler modes per graph

`grid_loss.py` (the update's graph: minibatch_loss forward and backward, as the update section)
and `profile_collect.py bc MODE` (act and advance together, the BC tick at 2,048
environments). Each mode its own cache directory: "cold" empties it first, "warm" is the next
process on the cache the cold one left.

| graph | mode | cold first call | warm first call | steady | vs default |
|---|---|---|---|---|---|
| minibatch_loss | default (the trainer's) | 49.6 s | 3.4 s | 165.85 ms / micro-batch | |
| minibatch_loss | max-autotune-no-cudagraphs | 498.8 s | 3.8 s | 163.4-163.8 | -1.3% |
| minibatch_loss | reduce-overhead | fails (gradient accumulation: section 4) | | | |
| minibatch_loss, 1 micro-batch per minibatch | default | | 3.5 s | 166.3 | |
| minibatch_loss, 1 micro-batch per minibatch | reduce-overhead | 50.1 s | 3.5 s | 165.4-165.5 | -0.5% |
| minibatch_loss, 1 micro-batch per minibatch | max-autotune | 469.8 s | 3.9 s | 163.75-163.77 | -1.5% |
| act + advance (BC tick) | default | 94.3 s (first segment) | 14.9 s | 25.34 ms / tick | |
| act + advance | max-autotune-no-cudagraphs | 305.7 s | 35.2 s | 25.04-25.06 | -1.2% |
| act + advance | reduce-overhead | fails (outputs overwritten: section 4) | | | |
| act + advance | max-autotune | not run: it includes inductor's CUDA graphs, which fail on act as reduce-overhead does; its other half is the row above | | | |

The first call includes the eager work before it (the collect's own set-up for the tick rows);
the "cold" loss row's 49.6 s is almost all compile. max-autotune on this GPU autotunes only
pointwise and reduction kernels ("Not enough SMs to use max_autotune_gemm mode": 36 SMs, the
threshold is 68): -1.2% to -1.3% for 6-10x the cold compile, which breaks the 150 s compile
budget (the warm cache recovers it). reduce-overhead gains 0.5% where it runs at all (the host
is hidden). slot_actions and merge_league (RL only) were not put through the grid: they are
the same model step on 107 rows and a merge; minibatch_values runs only with
`targets_every_epoch`. The losses agree across modes to 6-7 digits (first_loss in grid.jsonl).

Verdict on the laptop: keep default. On the 5090 GEMM autotuning applies and the host share
is larger: run the grid there (both scripts take the mode) before deciding.

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

## The live policy's step in seat chunks (rollout, measured; found while profiling)

The micro-batch curve suggested the same effect in the rollout, whose live-policy step runs
3,072 seats x 9 tokens (27,648 tokens, between the update's B 32 and B 48) in one call.
Prototype (`profile_collect.py --chunks C`): `Policy.step` patched to run the seats in C
equal chunks inside the compiled `act` (one graph, the chunks unrolled), the outputs
concatenated. On the CPU the chunked step equals the whole one to 3e-8 (logits; carry 5e-6 of
4.3; `chunk_check.py`).

| chunks (seats each) | BC tick | vs one call | first (compiling) segment, cold |
|---|---|---|---|
| 1 (the trainer) | 25.34 ms | | 94.3 s |
| 3 (1,024) | 23.60 | -6.9% | 107.2 s |
| 6 (512) | **22.27** | **-12.1%** | 127.4 s |
| 12 (256) | 23.17 | -8.6% | 171.1 s |
| 24 (128) | 26.26 | +3.6% | 259.9 s |
| RL tick, one call (EMA playing, 2 slot calls) | 27.70 | | 112.1 s |
| RL tick, the live policy in 6 chunks, the slots whole | **24.70** | **-10.8%** | 146.9 s |
| RL tick, every step in 6 chunks (the slots' 107 rows too) | 30.71 | +10.9% | 185.1 s |

Six chunks save about 3 ms a tick (BC and RL alike), 1.5-1.6 s of an update's 13-14 s of
collect (estimate from the tick), about 1.1% of the loop. Chunking the league networks' small
slot tables too (107 rows into chunks of 18) makes the RL tick slower: chunk only the large
seat batch. The compile grows with the unrolled chunks (+33 s cold at 6):
against the 150 s budget, the chunking wants a loop the compiler does not unroll (one
compiled chunk function called C times from `act`'s caller, or `torch._higher_order_ops`'s
scan) or the warm cache. The league networks' `slot_actions` (107 rows) need no chunking.

## Ranked proposals for item 5

Gains on the laptop, per BC update of about 140 s (768 micro-batches x 165.6 ms = 127.2 s of
learning plus 512 ticks x 25.34 ms = 13.0 s of collect, from the measured parts; the loop
itself was not timed end to end here). "Within rounding": the recipe's numbers do not change.

| # | what | gain (laptop) | size | risk |
|---|---|---|---|---|
| 1 | `micro_batch_views` 64 -> 16 (or 8), a per-machine setting | measured -10.4% (B 16) / -14.6% (B 8) of the micro-batch step: about -13.7 / -19.0 s an update, ~-10% / -14% of the loop (estimate); peak memory 4.6 -> 1.2 GiB at B 16 | S (a config number; the store and target pass already take any B) | within rounding (a minibatch's gradient is the sum of its micro-batches'); re-measure the curve on the 5090 (larger L2, host floor binding sooner) |
| 2 | The hit-label selection bias near episode and update ends (section 1) | correctness, not speed | S | changes the hit loss's rows (a fix) |
| 3 | The live policy's step in 6 seat chunks in `act` | measured -12.1% of the BC tick, -10.8% of the RL tick (live policy only), about -1.5 s an update (~1.1%) | S-M (a loop; keep the compile in budget: section above) | within rounding (checked on the CPU) |
| 4 | max-autotune-no-cudagraphs for minibatch_loss and act / advance | measured -1.3% / -1.2% | S | none; but cold compile 6-10x (499 s / 306 s; warm 3.8 / 35 s): not on the laptop; re-run the grid on the 5090 where GEMM autotuning applies |
| 5 | The critic's support built on the device once (two pageable copies per update in `value_support`) | negligible (per update) | S | bitwise |
| 6 | CUDA graphs for the rollout tick (section 4's plan) | at most the GPU's idle 1.9% (BC) / 2.7% (RL) here; on a faster GPU up to the host floor (2.94 / 5.23 ms a tick, ~1.1 ms more per slot call) | M-L | bitwise with the same draws; only if the 5090's RL tick measures host-bound; batched league slots (first review item 9) first |
| 7 | The geometry heads' fixed cost (15.4 ms a micro-batch) | folded columns: measured -0.3% (165.1 against 165.6) and -4 s of cold compile; the gathers' backward (aten scatter-add fallback, 1.6 ms) as an index_select / index_add on flattened rows: estimate up to 1% | S each | within rounding (fold: equal losses, gradients 3.8e-6 of 0.024) |
| 8 | A fused rotary: the spatial sublayer's `cat` of rotated and unrotated projections and the keys' and values' `cat` with the map memory materialise copies (kernels named for them about 15 ms a micro-batch, 9%, rough) | estimate 3-5% of the step, not prototyped | M | within rounding |

Not proposed (measured): activation checkpointing per sublayer (+20.4% at B 64, memory -35%),
the activation memory budget 0.5 (+15.5%, memory -25%), reduce-overhead for the loss (-0.5%,
and it needs the gradient accumulation reworked), attention as explicit matmuls instead of
SDPA (+2.8%: 170.2 ms), max-autotune for the loss with CUDA graphs (-1.5% at one micro-batch
per minibatch, cold 470 s).

Still open from the first review (not re-measured): the scripted agent on the league slice
only once the teacher term is off (its 5.5 ms a tick), the compact teacher record (queue item
6), the field memory once per deal row (estimate 3-6%), the policy and value heads on own
rows only (the auxiliary and pair heads now need the visible enemies).

Order for item 5: 1 (and the same curve on the 5090), 2, 3, 5; then 4 and 6 only from the
5090's numbers; 7 and 8 if the step still matters after 1.
