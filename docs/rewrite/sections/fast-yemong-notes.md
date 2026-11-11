# Fast Yemong: notes

Brief: `fast-yemong-brief.md`. Audit: `../audits/2026-11-02-fast-yemong.md`. Kernel source:
`race/yemong-blind`, `src/bnb/models/swift/` (read-only).

## Piece map

Both models are time-major over a segment: v2's `unroll` runs tokens (L, R, Q, D), map
memory (L, R, M, Dm) and the temporal carry (R, Q, D); the blind model's (T, B, Q, D). The
kernels flatten every leading axis but time (and, for attention, time too: one view-step
is one program). The table names the v2 piece each kernel replaces and what differs.

| Kernel (blind file) | v2 piece it replaces | Layout and parameters |
|---|---|---|
| scan (`scan.py`) | `temporal.parallel_scan` (the chunked scan in `RGLRU.sequence`) | Same layout: decay and contribution (L, ..., D), initial (..., D), flattened to (L, C). v2 cuts the recurrence at a start by zeroing that step's decay, which the kernel takes as given. No parameters. |
| attention with rotary (`attention.py`) | `SpatialBlock.forward` from the rotations to the output projection's input: `apply_rotary` on queries, keys and map keys, the two `cat`s, the transposes and SDPA | v2's `query_key_value` (3D, D) is (q, k, v) by rows then (heads, head width): q, k and v are strided views of its output, which the kernel reads by stride (no copy); `map_key_value` (2D, Dm) likewise. Rotary: both rotate interleaved (even, odd) pairs of a head's first 2R dims (R = 20 of 32 lanes at head width 64); v2's tables are (..., tokens, 1, R), the kernel's (views, tokens, R): a reshape. SDPA's default scale is the kernel's W^-1/2. |
| gated MLP (`gated.py`) | `GatedMLP.forward`'s `gate_up` GEMM and `gelu(gate) * up` (spatial and temporal sublayers) | v2's `gate_up.weight` (2H, D), gate rows first, is the kernel's (gate, up) pair as two row blocks of one matrix: the port takes the one matrix, so its weight gradient is one GEMM into it. `down` stays cuBLAS. 16-bit only (strict fp32 lost to cuBLAS in the blind model). |
| temporal mix (`temporal.py`) | `TemporalBlock.sequence` from `RGLRU._terms` to `recurrent * gelu(gate_branch(normed))`: the gates, the start mask, the scan and the output gate | v2's `recurrence.gates` (2D, D), recurrence rows first, is the kernel's pre-activation layout (p_r, then p_i). v2's `gate_branch` is its own (D, D) Linear: its output is the kernel's branch (unit stride, contiguous). keep = 1 - (a start after the first step). `log_decay` (D,) is shared. The convolution and the three projections stay plain. Segments only (L > 1); `step` keeps the plain path. |
| heads (`head.py`) | `HeadGroup.forward` per head (`last(gelu(norm(first x)))`), `ShotTrajectoryHead.query`'s tail (`last(gelu(middle_norm(middle h)))`; at ages shared by every row, the whole query, h = gelu(first_norm(latent + age)) made in the kernel too: the staged head, added by the learner-speed work, `compile-overhaul-notes.md`), and the encoder's `KindProjection` + `shared` (`shared(gelu(first_norm(first x)))`, before `shared_norm`) | v2's `HeadGroup.first` is one matrix for the group: head i is its row block i (a view). The encoder's queries are `shared(cat(ship, global))`: row-wise, so `cat(head(ship), head(global))`. 16-bit only. |

What the blind kernels compute differently from v2, and how the port keeps v2's function:
- **Norm epsilon under bf16 (heads, encoder): no difference.** v2's `norm` is
  `RMSNorm(eps=None)`, and under autocast a head's norm reads the bf16 output of its first
  projection. Measured: torch upcasts and uses float32's epsilon there too (an input of
  0.001 gives 0.945 for both dtypes, where bf16's epsilon would give 0.011), which is the
  blind head kernel's constant.
- **Autotuning (gated MLP).** The blind kernel autotunes over four tile configurations,
  which differ in BLOCK_K and so in accumulation order: the configuration a process picks
  by timing decides the bits. The port fixes one configuration (determinism).
- **Separate gate-branch projection (temporal mix).** The blind model computes the gate
  branch inside a fused projection and passes a strided view; v2's `gate_branch` is its own
  Linear, so the port passes its contiguous output. Same function.
- **The encoder's shared projection, kernel path only.** v2 projects the ships and the
  global token with one `shared` product over their concatenation; the head kernel runs it
  per kind (row-wise, the same function). The plain path is unchanged.

## The kernels, one commit each

Every number in this section is provisional: taken on the dev machine (RTX 4070 Laptop,
8 GB) under the benchmark lock, with the load average noted (1.1-2.2 for the kernel rows,
1.9-4.0 for the baseline rows, which match the coordinator's quiet-machine baseline within
0.5 ms). The coordinator re-measures on a quiet machine.

**Micro-benchmarks** (`/tmp/bnb-scratch-fast-yemong/micro.py`: the piece alone, compiled
with `fullgraph=True`, plain against kernel, median of 30, ms; training shape L = 128,
V = 16, 11 tokens, D = 128, forward and backward; rollout shape 2048 views, one step,
forward; load 1.0-1.4):

| Commit | Piece | Shape | fp32 plain / kernel | bf16 plain / kernel |
|---|---|---|---:|---:|
| `aa9e12cd` scan | `RGLRU.sequence` | training | 3.26 / 2.39 | 2.88 / 1.70 |
| `66cdd7ca` attention | `SpatialBlock` (whole sublayer) | training | 10.46 / 8.91 | 6.45 / 4.31 |
| | | rollout | 3.53 / 2.38 | 1.87 / 1.13 |
| `b14de1a8` gated MLP | `GatedMLP` | training | 4.84 (plain both) | 2.27 / 1.81 |
| | | rollout | 1.49 (plain both) | 0.64 / 0.34 |
| `de77abe2` temporal mix | `TemporalBlock.sequence` (plain = PyTorch scan) | training | 10.44 / 9.22 | 6.22 / 4.85 |
| `1d6bfeba` heads | `HeadGroup` (42 + 153 outputs) | training | 1.71 (plain both) | 1.15 / 0.98 |
| | | rollout | 0.40 (plain both) | 0.22 / 0.19 |

**The race benchmark after each commit** (`bench/yemong_speed.py`, ms, the better of
compiled and compiled + CUDA graphs; peak memory in GB; the kernels switched on
cumulatively with `--kernels`, which is the tree after each commit; raw rows in
`/tmp/bnb-scratch-fast-yemong/speed.jsonl`):

| Cell | v2 before | + scan | + attention | + gated MLP | + temporal mix | + heads (all) | blind |
|---|---:|---:|---:|---:|---:|---:|---:|
| Training V=16, fp32 | 85.7 (2.27) | 84.2 | 77.2 | 77.2 | 76.4 | 76.3 (2.24) | 73.7 |
| Training V=16, bf16 | 51.5 (1.54) | 49.8 | 40.8 | 38.8 | 37.9 | 37.7 (0.97) | 32.7 |
| Training V=64, bf16 | 226.2 (6.02) | 218.9 | 179.0 | 172.5 | 167.7 | 165.4 (3.77) | 141 (3.1) |
| Training V=64, fp32 | out of memory | oom | oom | oom | oom | out of memory | 329 (recomputes its MLPs) |
| Rollout, bf16 | 16.5 (0.33) | 16.3 | 13.5 | 11.8 | 11.8 | 11.5 (0.33) | 12.6 |
| Rollout, fp32 | 28.5 (0.44) | 28.4 | 25.4 | 25.3 | 25.2 | 25.4 (0.41) | 26.6 |

Speed-ups with every kernel: training bf16 1.37x (V = 16) and 1.37x (V = 64, memory
6.0 -> 3.8 GB); training fp32 1.12x; rollout bf16 1.43x, fp32 1.12x. The rollout beats the
blind model's; training bf16 is 5 ms (V = 16) and 24 ms (V = 64) short of it. The V = 64
fp32 cell still does not fit: the blind model fits it by recomputing its MLPs in the
backward, a memory feature outside this brief.

## Training throughput

`bnb train`'s trainer at 5v5 with 1024 environments, the default micro-batch (160 rows),
through `bench.cell`'s train cell (compile, one episode of rollouts, then the best of three
collect + update; `/tmp/bnb-scratch-fast-yemong/cell.py` sets the precision and the
switches; raw rows in `throughput.jsonl`). Remeasured after autotuning (2026-11-03, the
only GPU job, one cell after another; load average at the start of each cell, the 1-minute
figure first, raised by the cells before it). Provisional.

| Precision | Kernels | Rollout (s) | Update (s) | Train env-ticks/s | Peak (GB) | Load |
|---|---|---:|---:|---:|---:|---|
| fp32 | off (v2 before) | 3.42 | 44.38 | 2 742 | 3.6 | 1.10 / 1.45 / 2.17 |
| fp32 | on | 3.03 | 40.30 | 3 025 | 3.5 | 1.03 / 1.84 / 2.67 |
| bf16 | off | 1.90 | 30.98 | 3 986 | 2.8 | 1.09 / 2.40 / 3.13 |
| bf16 | on (the new default) | 1.39 | 25.43 | 4 886 | 2.2 | 2.41 / 3.34 / 3.46 |

The new default trains 1.78x faster than the old one (fp32 without kernels); the kernels
alone give 1.23x in bf16 and 1.10x in fp32. The update is 93-95% of training time in every
row. Every cell is within 1% of the same cell before autotuning (load 1.0-1.3 then).
(The audit's estimate for the blind model in bf16 was 1.8x against fp32 today.)

## Memory

The largest micro-batch that fits at 5v5 x 1024, by the memory probe (`bnb train --probe`:
setup and one update; fits = no out-of-memory and the reserved peak under the 7.12 GB
budget), doubling from 320 rows:

| Precision | Kernels off | Kernels on |
|---|---|---|
| fp32 | 160 (320 runs out of memory) | 160 (320 runs out of memory) |
| bf16 | 320 (640 runs out of memory) | 1280 (2560: reserved 7.24 GB, over the budget) |

The 26-ship cell (13 per team, 54 fields, 1576 environments, the default 160 rows), by the
probe: fp32 does not fit with or without the kernels; bf16 without kernels runs out of
memory; **bf16 with the kernels fits** (allocated 5.07 GB, reserved 6.24 GB).

After autotuning (2026-11-03, three probes of that cell, bf16 with the kernels,
`--arena-radius 4950`): with the autotune caches warm it fits (allocated 4.82 GB, reserved
5.55 GB); with both caches emptied (`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`) the
first update reserved 7.30 GB, over the budget, because the autotuner's benchmarking runs
inside the probed update; and one probe (Inductor's cache cold for these shapes, Triton's
warm) died with a device-side assert (open question 8). So on a new machine the probe
reads high until the run's shapes have been tuned once.

## Autotuning (the author's addition, 2026-11-02)

The training run will be on a rented, larger GPU, so no kernel keeps tiles chosen for this
one. Every kernel is a `triton.autotune` over a few configurations, keyed on the shapes
that change; a configuration a GPU cannot run (too much shared memory) is pruned by the
autotuner. Nothing reads the SM count or the shared-memory size; every kernel uses only
what compute capability 8.0 has (bf16 `tl.dot`, IEEE float32 dots). The configurations,
and each one forced alone at a training shape (L = 128, V = 16; eager forward + backward,
median of 20, ms, load about 1.3; `/tmp/bnb-scratch-fast-yemong/configs.py`, log
`configs.log`), with whether its bits equal the first configuration's:

| Kernel | Key | Configuration | ms | Same bits |
|---|---|---|---:|---|
| scan | channels | BLOCK 128, 4 warps | 0.546 | (first) |
| | | BLOCK 256, 4 warps | 0.541 | yes |
| | | BLOCK 512, 4 warps | 0.591 | yes |
| | | BLOCK 1024, 8 warps | 0.594 | yes |
| attention (fp32 / bf16) | tokens, maps, heads, width, IEEE | 2 warps | 1.862 / 1.159 | (first) |
| | | 4 warps | 1.892 / 1.197 | yes / no |
| | | 8 warps | 1.849 / 1.572 | no / no |
| gated MLP (bf16) | M, N, K | 64 x 64 x 64, 4 warps, 3 stages | 3.013 | (first) |
| | | 128 x 64 x 64, 4 warps, 2 stages | 3.037 | yes |
| | | 64 x 128 x 64, 4 warps, 2 stages | 3.035 | yes |
| | | 128 x 128 x 32, 8 warps, 2 stages | 3.109 | yes |
| temporal mix (fp32 / bf16) | steps, tokens, width | BLOCK 32, 2 warps | 1.433 / 1.183 | (first) |
| | | BLOCK 64, 2 warps | 1.418 / 1.166 | no |
| | | BLOCK 64, 4 warps | 1.443 / 1.206 | no |
| | | BLOCK 128, 4 warps | 1.426 / 1.173 | no |
| heads (bf16) | M, K, N, hidden | 32 rows, BLOCK_K 64, 4 warps | 1.239 | (first) |
| | | 64 rows, BLOCK_K 64, 4 warps | 1.539 | yes |
| | | 64 rows, BLOCK_K 64, 8 warps | 1.224 | no |
| | | 128 rows, BLOCK_K 32, 8 warps | pruned here (needs 112 KB of shared memory; 99 KB on this GPU) | |

The temporal mix keeps CHUNK = 16 fixed: it sets the in-chunk scan's order of operations.
The attention's tiles are set by the token and key counts (padded to powers of two of at
least 16), so only its warps are tuned.

**Cost.** The first call of a new shape tunes it: 0.9 s per kernel pair here (1.5 s for the
heads), most of it compiling each configuration; a later shape costs only the timing, a
few tens of ms. The shapes of a run are few (the rollout's, and at most three micro-batch
shapes in the update), so tuning is a one-off cost of seconds at the start of a process.

**Determinism.** Configurations that split the work differently but sum in the same order
give the same bits (the scan, the gated MLP: checked by
`test_every_configuration_of_an_order_free_kernel_gives_the_same_bits`). For the
attention, the temporal mix and the heads, the warps or the block change the layout of a
reduction or of the in-chunk scan, so configurations differ in their last bits (all agree
with the plain path: `test_every_autotune_configuration_matches_the_plain_path` forces each
in turn). Within a process the choice is fixed once made, so identical inputs give
identical bits. Across processes the choice is cached on disk (`cache_results=True` for
Triton's autotuner in eager mode; Inductor's local autotune cache, on by default, for the
compiled path, which routes user kernels through it), so a resumed run on the same machine
repeats the bits. A resume on a machine whose caches were emptied may tune differently and
then differ in the last bits from the run it continues: see the open questions.

**A bug autotuning found.** The heads' backward wrote one norm-scale partial sum per
program into a buffer sized for the smallest block; while autotuning timed the
configurations, a smaller block left stale partial sums that a larger one did not
overwrite, and the scale's gradient came out wrong (the whole-model bf16 test caught it,
73% off). The partial sums are now over fixed groups of 32 rows, whatever the block, so
every configuration writes the same rows.

## Deviations and open questions

No kernel computes a different function from v2's: the blind kernels' maths is v2's (the
piece map lists the layout differences the port bridges, and the one constant that looked
different, the norm epsilon, is not). The fixtures run the plain path and pass.

1. **Determinism across machines.** With autotuning, the attention, the temporal mix and
   the heads may pick a different configuration on a machine whose autotune caches were
   emptied, and so differ in the last bits from an earlier process (same function, same
   tolerance). A resume on the same machine repeats the bits. If exact resumes across
   machines matter, the choices could be saved in the training checkpoint, or those three
   kernels restricted to configurations that sum in one order (the attention at 2 warps,
   the temporal mix at one block, the heads at 4 warps), which gives up their tuning.
2. **The shot trajectory head runs in fp32 under bf16.** `Yemong.shot_trajectory` is called
   outside the model's autocast (`_matmuls`), so its four products run in fp32 and the head
   kernel does not apply to its tail, though `bfloat16_matmuls` is documented as covering
   the heads. It is the largest head (81 920 rows at V = 16); the profile of the bf16
   training cell shows about 1.7 ms of fp32 GEMMs, and the blind model measured that head at
   3.5 ms in bf16 against 7.0 ms in fp32. Moving it under autocast changes what bf16 rounds,
   so it is the author's call; it is likely most of the remaining gap below.
3. **The remaining gap to the blind model in bf16 training** (37.7 against 32.7 ms at
   V = 16). GPU time per segment, every kernel on (`gpu_profile.py`, before autotuning):
   GEMMs 11.8 ms, RMSNorm (aten) 7.1, Inductor's fused kernels 7.1, gated MLP 3.9, heads
   3.2, attention 2.8, temporal mix 1.2; 37.2 ms in all. The blind model's: GEMMs 11.0,
   RMSNorm 6.8, fused 5.4, gated 3.5, attention 2.7, scan 1.0; 32.8 ms.
4. **Training at V = 64 in fp32 still runs out of memory.** The blind model fits it by
   recomputing its MLPs in the backward, a memory feature outside this brief.
5. **Boundary.** `src/bnb/cli.py` is outside the brief's list: its `--precision` default
   (`"float32"`) would have overridden `TrainConfig`'s, so it changed to `"bfloat16"`, and its
   printed update line now shows the drift and the non-finite counts.
6. **A CUDA error in one probe.** One 26-ship bf16 probe with the kernels failed with a
   sticky CUDA error while the kernel sources were being edited under the running chain;
   the same probe passed when rerun (see Progress for the rerun after autotuning).
8. **Two intermittent failures, each seen once and not reproduced (2026-11-03).**
   - The first full suite after autotuning (fresh `/tmp`, so Inductor's cache was cold and
     Triton's, in `~/.triton`, warm) failed one test,
     `tests/runs/test_resume.py::test_a_resumed_run_equals_one_straight_through_on_the_gpu[yemong]`:
     the first update's pool metrics (the rollout's reward rates) differed between the
     straight process and the first. Rerun alone four times it passed (warm caches; both
     caches emptied; Inductor's emptied and Triton's warm, the suite's state; all of
     `tests/runs` and the kernel tests from emptied caches), and the second full suite
     passed.
   - One 26-ship bf16 probe died with a device-side assert in the learner's loss
     (`index out of bounds: 0 <= tmp32 < 42`): a stored action's joint index, which
     Gumbel-max sampling keeps in range, was out of range, so the update read actions that
     were never written or were overwritten. It was raised while Inductor benchmarked that
     kernel's configurations. Rerun twice (warm caches; emptied caches) it did not recur.
     The CUDA error of item 6 may be the same failure.

   Both happened in processes that were autotuning, which changes the timing around the
   segment store's asynchronous copies (pinned host memory, a copy stream and a worker
   thread: `src/bnb/runtime/store.py`, outside this brief). A race there, exposed by the
   timing rather than caused by a kernel, would explain both; nothing pins it down yet. A
   kernel reading memory that one configuration leaves unwritten would explain them too,
   but each kernel's buffers are written in full by every configuration (the heads'
   partial sums were the one exception, fixed). If it recurs: the resume test or the probe
   in a loop from emptied caches, `host_store=false` against the default.

   **Answered (2026-11-03, `fix/intermittent`, `1f5558ac`): a race in the segment store,
   not in a kernel or in autotuning.** `SegmentStore.batches` set up the staging buffers of
   a micro-batch size it had not seen on its worker thread, while the caller computed on the
   previous micro-batch, and recorded their `released` event before allocating them. The
   caching allocator can give such a buffer memory the compute stream has just freed with
   kernels still queued that write it; the first copy into the buffer runs on the copy
   stream, waited only for that earlier event, and could be overwritten by those kernels. The
   learner then read a corrupted micro-batch: in the probe, the trailing 2-view micro-batch
   of a minibatch, whose own-action indices came out of range (the assert); in the resume
   test, `prepare`'s trailing 8-view micro-batch in a process's first update. With the store
   as it was, the 26-ship probe with warm caches fails 6 times in 6 (3 times the original
   assert in the same kernel, 3 times a loss off by 12%, by 5e26, or NaN); with the fix it
   gives the same update 6 times in 6. The buffers are now allocated on the caller's thread
   at the start of `batches`, and released after the allocation; a test forces the
   interleaving (`tests/sections/test_store_staging.py`). Autotuning only made the window
   likelier. Item 6's sticky CUDA error had the edited kernel sources as an explanation and
   is not known to be this.
9. **Noise in the plain path's compile logs.** Compiling the plain path logs
   "Attempting to permute a tensor of rank 5" from the fake-tensor pass; it is not fatal and
   is not new (the plain path is v2's code).

## Progress (2026-11-03, done)

- **Done and pushed:** the piece map; the five kernels, one commit each (`aa9e12cd`,
  `66cdd7ca`, `b14de1a8`, `de77abe2`, `1d6bfeba`); bf16 by default with the precision
  signals (`dac8a4b0`); every kernel autotuned (the author's addition), with the heads'
  partial-sum fix and the tests over every configuration (`11e54dce`, committed as WIP;
  the commit after it records that the suite passed).
- **Full suite (GPU, under the lock), 2026-11-03:** first run 1 failed, 1064 passed (the
  GPU exact-resume test for Yemong, open question 8); second run **1065 passed, 7
  skipped, 1 xpassed** (49 min). The skips are six league-equivalence tests (the league
  section's `make_arm` is not implemented yet) and one ship-card chart test (needs the
  tools extra); none is a hardware skip, none is new.
- **Hand breaks after autotuning** (`breaks.py`, a 1% error in each kernel's forward
  store): every kernel's own tests fail (scan, attention, gated MLP, temporal mix,
  heads), and the sources were restored.
- **Measured after autotuning:** the race benchmark (label "6 autotuned", load 0.7-1.8):
  training V=16 fp32 76.5 ms, bf16 37.7 (0.97 GB); V=64 bf16 165.5 (3.77 GB), fp32 out of
  memory; rollout fp32 25.4, bf16 12.2, the same as before autotuning within noise. The
  training throughput cells and the 26-ship probe: see Training throughput and Memory.
- **Scratch** (`/tmp/bnb-scratch-fast-yemong`, recreated 2026-11-03 after a reboot wiped
  it): `locked.py` (a command under `bench.lock.bench_lock`), `cell.py` (bench.cell's
  train cell at 5v5 x 1024 with a precision, the switches on or off and optional
  micro-batch rows; appends to `throughput.jsonl`), `breaks.py`; logs `suite.log`,
  `suite2.log`, `breaks.log`, `probe26*.log`, `resume*.log`. Not recreated: `micro.py`,
  `configs.py`, `train_speed.py`, `gpu_profile.py` (never name a script `profile.py`: it
  shadows the standard library's).
- **Open:** questions 1-3 and 8 are the author's: determinism across machines, the shot
  trajectory head outside autocast, the remaining gap to the blind model, and the two
  intermittent failures.

## The intermittent failures (open question 8): progress

Working notes of the hunt (`docs/rewrite/sections/intermittent-failure-brief.md`); scratch in
`/tmp/bnb-scratch-intermittent` (`loop.py`, `worker.py` with switches, `idempotent.py`,
`stress*.py`, `retune_loop.sh`, `chunks.sh`; logs `*.log`).

- **The evidence on disk** (`/tmp/bnb-scratch-fast-yemong`): the suite's failure printed only
  the first update's dict (`At index 0 diff`, truncated), so which metric differed is not
  known, not necessarily a pool's. The probe's assert came from
  `triton_per_fused__log_softmax_..._gather_..._89` (the own-action log-probability of the
  loss: `power * 14 + stick * 2 + shoot`, loaded from the staged `trajectory.action`), at
  3328 rows = 128 steps x 2 views x 13 ships, the trailing micro-batch's shape, in one
  program (`block [90]`) only; it surfaced at the `synchronize` that starts a benchmark.
- **Hypothesis 1 ruled out.** Each kernel, forward and backward, eager (Triton's autotuner)
  and compiled (Inductor's), autotuned on fresh inputs with emptied caches and memory dirtied
  with NaN first, then called again with the configuration cached: bitwise equal and finite
  (`idempotent.py`, `idempotent.log`). Every kernel writes only its outputs, in full.
- **Store reads of unwritten memory ruled out (resume configuration):** every store and
  staging buffer allocated as NaN / -77 / True (`BNB_POISON=1`), and every copy-stream copy
  delayed by ~10 ms (`BNB_DELAY_COPY`): the metrics equal the plain warm run's.
- **Autotuning choice is consistent across processes** in the runs logged: 908 Inductor
  autotuners per process, 16 files shared by two instances; the second instance reads the
  first's saved choice, and process B runs exactly A's configurations (`BNB_LOG_TUNING`).
- **Loops:** cold A against warm B, `first` mode: 1 both-cold, 7 Inductor-cold (2 updates
  each from the 4th), 7 re-tune-only (Inductor's `.best_config` files deleted, compiled code
  kept): all equal. Neither original symptom recurred end to end.
- **The race found (fixed in `1f5558ac`).** `SegmentStore.batches` set up a new micro-batch
  size's staging slot on its worker thread while the caller computed on the previous
  micro-batch: it recorded the slot's `released` event on the compute stream, then
  allocated the pinned and device buffers. The caching allocator can hand the device buffer
  memory the compute stream freed after that record, with kernels still queued that write
  it (stream-ordered reuse); the first copy into the buffer, on the copy stream, waited only
  for `released`, so those kernels could land after it and overwrite the batch. Measured on
  this GPU: a pinned host-to-device copy on a side stream does not wait for queued
  default-stream work (`order2.py`, `prim.py`). Evidence: forced interleaving (the caller
  frees a still-written temporary just before the worker allocates) overwrites the batch
  every time before the fix (`tests/sections/test_store_staging.py`); natural timing
  (`stress.py 2000 2000 5000 50000000`, the real store and threads, no hooks): 12 of 2000
  batches overwritten before, 0 of 2000 after. Lazy module loading, `cudaMalloc` and a
  first `cudaHostAlloc` can wait for the device and hide it (the test warms them first).
- **Why it matches the failures.** The probe's asserting kernel ran on the trailing
  micro-batch of a minibatch (2 views of 6), a size first seen in that update, so its slot
  was set up on the worker while the caller ran the previous micro-batch's loss; the bad
  rows sat in one program. In the resume test (24 views, micro-batches of 16) `prepare`'s
  trailing 8-view slot is set up the same way, in every process's first update. Autotuning
  makes the window likelier (the caller allocates and frees benchmark clones and buffers
  with kernels queued) but is not the cause.
- **End to end (the 26-ship probe, `probe_run.py`, `probe_warm.sh`, warm caches; one
  update; logs `runs/probe/*.log`, `probe_compare.py`):** the store before the fix
  (`BNB_OLD_STORE=1`, `old_store.py` = the store at `d6fb28fd`) 6 of 6 runs corrupted: the
  original assert in the same Inductor kernel (`cgugzpcgik...py:66`) 3 times, otherwise the
  next-state loss off by 12% or 5e26, or NaN. The fixed store: 6 of 6 runs agree to within
  5e-7 relative and the rollout's pool metrics are identical. That residual is a separate
  finding: at 26 ships the update is not bitwise repeatable between processes with warm
  caches (policy, value and enemy-action losses differ in their last bits; the 2v2 exact
  resume is unaffected). Not investigated; an atomic add in a compiled backward is the
  likely kind.
- **Also fixed:** `Learner.drift` took one batch with `next(iter(...))` and abandoned the
  generator, so that slot's `released` was never recorded after drift's compute; a later
  batch of the same size could be copied into it while drift still read it (the relational
  model's sizes can meet this; Yemong's gradient batches carry the shot labels, another
  key). It now runs the generator out.
- **Done (2026-11-03):** full suite under the lock after the fix, **1071 passed, 7 skipped**
  (37 min; the same seven skips as before). Committed and pushed on `fix/intermittent`:
  `1f5558ac` (the fix, its test, the learner page), the notes. Scratch and logs in
  `/tmp/bnb-scratch-intermittent` (`suite.log`, `probe-warm*.log`, `runs/probe/`).
