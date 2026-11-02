# Brief: the blind race's kernels in v2's Yemong, and bf16 by default

You are bringing the fast kernels of a second Yemong implementation into v2's Yemong. The
model's mathematics does not change. Read `docs/rewrite/audits/2026-11-02-fast-yemong.md`
first: it says what the kernels are, what they buy, and what the author decided.

**Author's decisions (2026-11-02):**
- **All five kernels come in:** attention with rotary inside, the fused gated MLP, the
  sequential scan, the fused temporal mix (training segments only), and the fused heads.
- **Each kernel stays disposable.** v2's current PyTorch code is its plain twin, behind a
  switch, with a test that the two agree. Do not delete or fork the plain path.
- **Training runs in bf16 by default:** `TrainConfig.precision` defaults to `"bfloat16"`.
  fp32 stays one setting away.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-fast-yemong`, branch `perf/fast-yemong` (from
  v2). Work only here. Scratch in `/tmp/bnb-scratch-fast-yemong`.
- **The source of the kernels:** the branch `race/yemong-blind`, in
  `/home/vizia/avesnova/bnb-yemong-blind`, read-only. Its `NOTES.md` explains each kernel
  and its measurements, and its `tests/test_swift_kernels.py` its tests. You may read that
  directory. Do not read any other directory under `/home/vizia/avesnova` (no v1, no
  `physics-race*`, no other worktree).
- **Do not merge or read** the `perf/update` branch. It changes the same attention code
  and is redone after you.
- **Python:** `uv sync`, then always `uv run --no-sync ...`.
- **GPU:** every GPU run (tests and timings) holds the benchmark lock
  (`bench.lock.bench_lock()`; never plain `flock`). Another agent or the coordinator may
  queue on it.
- **Keep your context small:** read only what a step needs; send long output to files and
  read their tails. Run targeted tests while you work, and the full suite once at the end.

## What to do

1. **Map the pieces.** For each kernel, find the v2 module it replaces (`network.py`,
   `temporal.py`, `heads.py`, the encoder's projections) and how the blind model's tensor
   layout and parameter shapes differ from v2's. Write the map into your notes before
   coding.
2. **Port in this order, one commit each, measuring as you go:**
   1. scan;
   2. attention with rotary;
   3. gated MLP;
   4. temporal mix;
   5. heads.

   Each kernel goes in as a `torch.library.triton_op` with autograd, as the blind model
   does, so it compiles under `fullgraph=True` and captures in the rollout's CUDA graph.
3. **Keep the weights.** Keep v2's parameter names and shapes, so existing policy files
   and the compatibility set (`tests/runs/checkpoints`) still load and give the same
   outputs to kernel tolerance. If a kernel needs another layout, convert at load time
   inside the module. Do not raise the model's `VERSION`.
4. **bf16 by default.** Set `TrainConfig.precision` to `"bfloat16"`. Check where the
   default reaches: the rollout, the update, BC and the docs. Every part listed as staying
   fp32 (the residual stream, norms, softmax, the recurrent state, codes, belief, critic
   expectations, every loss, the weights and the optimizer) stays fp32. Make sure the
   training logs carry the precision signals: the first epoch's ratio drift, non-finite
   counts and the clip fraction. If any is missing, add it.

## Tests

- **Fixed:** `tests/reference/` and the goldens. The 43 Yemong fixtures run on the plain
  path (CPU, float64) and must still pass.
- **Add for each kernel:** kernel against plain, forward and backward. Use fp32 tolerances
  like the blind model's tests, plus a bf16 case.
- **Determinism:** the exact-resume tests (`tests/runs/test_resume.py`) are bitwise. A
  kernel whose backward uses atomics, or whose result depends on launch order, breaks
  them. Every kernel must give bitwise-identical results on identical inputs; add a test.
- **Already there, must keep passing:** `step` equals `unroll` (to bf16 rounding in bf16),
  the learner's tests, the runs tests and the compatibility set.
- **Break each kernel's switch by hand** (wrong result in the kernel), watch its test fail,
  and revert.
- **Everything passes:** `uv run --no-sync pytest` under the lock, nothing skipped but
  what needs hardware you lack.

## Measure

- **The race benchmark:** copy `bench/comparison/v2_speed.py` and `blind_synthetic.py`
  from the `race/yemong-blind` branch (`/home/vizia/avesnova/bnb-yemong-blind`) into
  `bench/yemong_speed.py` (and a helper module), and run it before you start and after each kernel. The coordinator's quiet-machine
  baseline and the blind model's numbers are in the audit; aim for the blind model's.
- **Training throughput:** `bnb train` at 5v5 with 1024 environments, env-ticks/s and
  the update's time, before and after, in fp32 and bf16. Record the load average with
  every number and call numbers taken under load provisional; the coordinator re-measures
  on a quiet machine.
- **Memory:** the largest micro-batch that fits, before and after, and whether the
  26-ship cell now fits.

## Docs

- Update the Yemong explanation page and any page that states the default precision.
- Document the kernel switches: where they are, and that comparisons between model types
  run both models plain or both with kernels.
- The docs tests run in the suite; follow `docs/contributing/writing-docs.md`.

## Boundaries

- **You may change:**
  - `src/bnb/models/yemong/**`;
  - `src/bnb/runtime/train.py` (the precision default and the logged signals);
  - `src/bnb/learn/**`, only for the precision signals;
  - `bench/` (the new benchmark only);
  - tests outside `tests/reference/`;
  - the user docs and your notes.
- **No legacy:** no shims. The switches are deliberate (the author's rule), not legacy.
- **If something in the blind model is not the same function as v2's** (a deviation the
  fixtures did not catch), stop and write it up in your notes. Do not change v2's maths.

## Commits

Commit to `perf/fast-yemong` and push (`git push -u origin perf/fast-yemong`). Author
Julian Kizanis <julianzia@gmail.com>; end every commit message with
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Hand back

Write `docs/rewrite/sections/fast-yemong-notes.md` with:
- the piece map;
- each kernel's commit with its micro-benchmark and the race-benchmark cell after it;
- training throughput before and after;
- memory;
- deviations and open questions;
- a Progress section kept current, so a fresh agent can take over.

Stop at a clean commit if you have been working long. Your final message summarises it in
a page or less.
