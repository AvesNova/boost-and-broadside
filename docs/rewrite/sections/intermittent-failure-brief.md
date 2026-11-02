# Brief: find the intermittent failure seen while kernels autotune

v2 now has five Triton kernels in Yemong, autotuned per shape (`perf/fast-yemong`, merged).
Its verification saw two failures that it could not reproduce or explain. Find the cause and
fix it. No training run starts until this is understood, so a fix without a cause is not
enough.

## What was seen

Both failures happened in processes that were autotuning a shape for the first time. Neither
recurred once the caches were warm.

1. **The full suite, first run:** `tests/runs/test_resume.py::test_a_resumed_run_equals_one_straight_through_on_the_gpu[yemong]`
   failed. The first update's pool metrics (the rollout's reward rates) differed between the
   "straight" and "first" processes. It passed four times alone afterwards: with warm caches,
   with both autotune caches emptied, with only Inductor's emptied, and as part of
   `tests/runs` plus the kernel tests from emptied caches.
2. **The 26-ship memory probe** (bf16, kernels on, 1576 environments, radius 4950), with
   Inductor cold for those shapes and Triton warm: a device-side assert in the learner's
   loss, `index out of bounds: 0 <= tmp32 < 42`. A stored joint action index was out of
   range. Sampling cannot produce one, so the update read actions that were unwritten or
   overwritten. An earlier "sticky CUDA error" in `docs/rewrite/sections/fast-yemong-notes.md`
   may be the same thing.

## Hypotheses, to confirm or rule out with evidence

- **Autotuning reruns a kernel that is not idempotent.** `triton.autotune` benchmarks each
  configuration by launching the kernel several times on the real arguments. A kernel that
  accumulates into an output, writes in place into an input, or writes a buffer that aliases
  an input gives a wrong result after the benchmark, unless it declares `restore_value` /
  `reset_to_zero`. None of the kernels declares either (`src/bnb/models/yemong/kernels/`).
  The heads' backward already had one bug of this kind (stale norm-scale partial sums).
  Check every kernel, forward and backward, for outputs it reads or accumulates into, and
  for outputs that can alias inputs (including through `torch.library` / Inductor buffer
  reuse).
- **Autotuning changes timing or memory, and exposes a stream race.** The segment store
  (`src/bnb/runtime/store.py`) copies on its own stream, from pinned host memory, with a
  worker thread. The benchmark's extra launches, synchronizations and allocations, such as
  its L2-flush buffer, change when memory is reused. Check every buffer that crosses streams
  or threads for a missing `record_stream` or event wait. That includes allocations made on
  the worker thread, where the current stream is not the compute stream.
- **Autotuning during the rollout's warm-up or CUDA-graph capture** (`_capture` in
  `src/bnb/runtime/rollout.py`), or under `torch.compile`, leaving state that a replay
  reads.

## How

- **Reproduce first.** Write a loop that empties the caches it needs (Triton's autotune
  cache, Inductor's, or both) and runs the resume test, or a smaller case that checks the
  same thing. Run it until it fails, and measure the failure rate. Then add switches one at a
  time: kernels off, `host_store` off, autotune off (one fixed configuration). Each
  hypothesis predicts a different pattern.
- A deterministic check beats a rate. For the first hypothesis: call each kernel with
  autotuning forced on fresh inputs, then compare its result with a second call made with
  the chosen configuration cached. They must be bitwise equal.
- **The fix comes with a test** that fails before it, and runs in the quick tier when it is
  cheap.
- If the cause is in the store, the rollout or the learner rather than the kernels, fix it
  there. Same rules.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-intermittent`, branch `fix/intermittent` (from
  v2). Work only here. Scratch in `/tmp/bnb-scratch-intermittent`.
- Do not read any other directory under `/home/vizia/avesnova`.
- **Python:** `uv sync`, then always `uv run --no-sync ...`.
- **GPU:** every GPU run holds the benchmark lock (`bench.lock.bench_lock()`; never plain
  `flock`). Another agent may queue on it. Keep each locked run short, so the loop does not
  hold the GPU for hours.
- **Keep your context small:** send long output to files and read their tails.

## Done

- The cause, with the evidence that shows it.
- The fix and its test.
- After the fix, the reproduction loop runs clean for at least as many iterations as it
  needed to fail several times before.
- The full suite passes under the lock.
- `docs/rewrite/sections/fast-yemong-notes.md`: open question 8 is answered.
- Commit to `fix/intermittent` and push (`git push -u origin fix/intermittent`). Author
  Julian Kizanis <julianzia@gmail.com>; end every commit message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

Your final message summarizes the cause, the evidence and the fix in a page or less. If you
cannot find the cause, say exactly what you ruled out and how.
