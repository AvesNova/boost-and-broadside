# Brief: every kernel configuration gives the same bits

Yemong has five Triton kernels, each autotuned per shape and GPU
(`src/bnb/models/yemong/kernels/`). In the scan and the gated MLP, every configuration
gives the same bits, and a test checks it
(`test_every_configuration_of_an_order_free_kernel_gives_the_same_bits` in
`tests/sections/test_yemong_kernels.py`). In attention, the temporal mix and the heads,
configurations differ in their last bits. The first tuning of a shape picks one by
timing, and only the on-disk cache makes a later process repeat it.

That has two consequences:
- **A flaky test:** the GPU exact-resume test (`tests/runs/test_resume.py`) can fail
  when its processes tune from cold. The league's agent saw it fail once on new shapes,
  and it passed when rerun.
- **A fragile resume:** a resume on the same machine is exact only while the caches
  survive.

## The goal

Every configuration of every kernel gives the same bits: the tuning changes how the work
is split, never the order of any sum. Then a resume on one machine is exact whatever the
cache holds, and the test above covers all five kernels.

The usual way: fix the size of every reduction block (the key block in attention's online
softmax, the reduction tile in the heads and the temporal mix, the partial-sum grouping)
as a constant of the kernel, and autotune only what splits independent work (rows,
columns, warps, stages). The heads already fix their partial-sum grouping (`GROUP`).

## Speed

The author's rule is repeatable where it is free, never at the cost of throughput.
- **Before and after:** measure each kernel's micro-benchmark and the training cell
  (`bnb train`, 5v5, 1024 environments, bf16, kernels on: env-ticks/s and the update's
  time). Hold the lock, record the load average, and repeat a cell when its numbers look
  contended.
- **Within noise (about 2%) in the cell:** keep the change.
- **Costs more:** keep that kernel as it is, and instead make the resume test robust. Its
  processes read a cache tuned once before them. Write up the cost in your notes for the
  author.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-kernel-order`, branch `fix/kernel-order` (from
  v2). Work only here. Scratch in `/tmp/bnb-scratch-kernel-order`.
- **Do not read** any other directory under `/home/vizia/avesnova`.
- **Python:** `uv sync`, then always `uv run --no-sync ...`.
- **GPU:** every GPU run holds the benchmark lock (`bench.lock.bench_lock()`; never plain
  `flock`). The wrapper `/tmp/bnb-scratch-fast-yemong/locked.py` runs a command under it.
  The play shell's agent takes the lock for short GL runs.
- **Read first:** the kernels' module docstring
  (`src/bnb/models/yemong/kernels/__init__.py`), `docs/rewrite/sections/fast-yemong-notes.md`
  (the piece map and the measurements), and the kernel tests.

## Tests

- **The configuration test** covers all five kernels, forward and backward, in fp32 and
  bf16.
- **The existing tests keep passing:** each kernel against its plain twin, bitwise
  repeatability, `step` equals `unroll`, the reference fixtures, and the exact resume.
- **The resume test from cold caches:** empty Triton's and Inductor's caches, then run it
  ten times in a row. It must pass every time.
- **Break by hand:** make one kernel's reduction block depend on the configuration again,
  watch the configuration test fail, and revert.
- **The full suite** passes under the lock, once at the end.

## Boundaries

- **You may change:** `src/bnb/models/yemong/kernels/**`, their tests,
  `tests/runs/test_resume.py` only for the fallback above, the Yemong explanation page,
  and the notes.
- **The maths does not change.** The plain twins and the reference fixtures are fixed.

## Commits and hand back

- Commit to `fix/kernel-order` and push (`git push -u origin fix/kernel-order`). Author
  Julian Kizanis <julianzia@gmail.com>; end every commit message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Append a section to `docs/rewrite/sections/fast-yemong-notes.md`: what changed per
  kernel, the before and after numbers with load, the cold-cache loop, and a Progress
  list.
- Your final message summarizes it in half a page.
