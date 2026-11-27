# Speed 5 progress (queue item 5, branch `speed-5`)

The second training review's measured wins that keep results within rounding
(training-review-2.md, ranked proposals 1, 3, 5). Scripts in `speed-5/`; logs in
`data/speed5/logs/` (not committed). Every GPU job a `bnb-5-*` unit under the GPU lock, one at
a time; "core" numbers from a detached worktree at 2b7bc58e (`v2-speed5-base`).

## 2026-11-28

- 06:55 GPU gradient check (`grad_check.py`, eager bf16, every geometry pair, 192 views, one
  minibatch, SGD at rate 1): the losses at micro-batches 64 / 16 / 8 agree to 7 digits
  (loss 5.018908 / 5.018908 / 5.018908); the parameter step differs from 64's by 1.1e-3 (16) and
  1.1e-3 (8) of its norm, 2.4e-3 / 2.0e-3 of its largest entry: bf16 autocast rounding (bf16's
  epsilon is 7.8e-3). Worked.
- 06:56-07:24 whole BC updates on the real trainer (`trainer.py`: lean.train, 2,048
  environments, the streamed store, 4 epochs x 32 minibatches; 3 BC updates each, the first
  compiling, cold cache of its own), core 2b7bc58e, performance profile, no other bnb unit,
  load 2.1-3.6 at the starts (the previous job's compile workers):

  | micro-batch views | update alone, updates 1 / 2 | loop steps/s, updates 1 / 2 | peak allocated |
  |---|---|---|---|
  | 64 (core) | 124.86 / 124.81 s | 7,605 / 7,608 | 5.13 GiB |
  | 16 | 111.18 / 112.32 s (-10.5%) | 8,441 / 8,362 | 1.73 GiB |
  | 8 | 107.36 / 107.80 s (-13.8%) | 8,711 / 8,675 (+14.3%) | 1.32 GiB |

  The whole update follows the review's per-micro-batch curve (-10.4% / -14.6%): the extra
  streamed-store gathers and copies of 8x the micro-batches stay hidden. 32 was stopped
  unmeasured (64, 16, 8 already monotone; the review's curve has 32 between). Not tried below
  8 (the review: B 4 slower, host-bound). Chosen: 8, a default with a note in LearnConfig (no
  per-machine settings place exists); the 5090 re-measures. A first attempt of the curve was
  stopped: it ran from the branch worktree while I edited it; then rerun from a detached
  core worktree.
- 07:00-07:43 seat chunks (`rollout.chunked_step`, LearnConfig.policy_step_chunks = 6). As a
  plain unrolled loop the review measured +33 s of cold compile at 6 chunks. Instead each
  chunk is a `torch.compiler.nested_compile_region` (`rollout._policy_chunk`, decorated at
  module level: decorating inside the traced act is a graph break, the first try's 2 breaks
  and 2 act graphs): the step is traced once and called per chunk. On the CPU chunked equals
  one call to 4e-8 (logits) / 5e-6 (carry), test_the_chunked_policy_step_equals_one_call.
  Ticks (`training-review-2/profile_collect.py bc|rl default cold --no-profile`, 2,048
  environments, one compiling and one timed 128-tick segment; core from the core worktree):

  | | core | 6 chunks | |
  |---|---|---|---|
  | BC tick | 25.34 ms | 20.87 ms | -17.6% |
  | RL tick (EMA, 2 slot calls; the slots stay whole) | 27.70 ms | 23.22 ms | -16.2% |
  | first (compiling) segment, cold, BC / RL | 95.6 / 113.1 s | 97.5 / 115.1 s | +2 s |
  | peak allocated, BC / RL | 0.97 / 1.16 GiB | 0.72 / 0.91 GiB | |

  More than the review's unrolled prototype (-12.1% / -10.8%). The compile check:
  **core itself fails it** (2b7bc58e, run here 07:34): act 6,325 traced operations against the
  baseline's 4,613 (+37%) and cold 160.3 s, over the 150 s budget, before any change of mine
  (the baseline, 522e25f3 on `heads-8c`, predates the scripted-delay merge dc0d5480 in
  lean/, and act holds the scripted agent: the likely cause, not bisected). The branch at 6 chunks and 8-view micro-batches: act 5,788 operations
  (fewer than core: the step is traced once), 5 graphs, no break, no recompile, cold 163.1 s
  (+2.8 s over core). Over the budget, as core is: reported, budget not raised.
- 07:35 the critic's support cached on the device per process (`returns.value_support`):
  bitwise the same values; gain not measured (the review: negligible, two pageable copies per
  update).
- 07:46-08:21 final checks. Compile check on the branch: 5 graphs, no break, no recompile,
  act 5,788 operations, cold 164.3 s (163.1 at 07:32): over the 150 s budget as core is;
  baseline reset with the reason in the commit, budget kept. Smoke runs on the real trainer
  (`trainer.py`, 2,048 environments, 3 updates each, the first compiling; no recompile, which
  the static graphs would raise):

  | | core 2b7bc58e | branch | |
  |---|---|---|---|
  | BC, steps/s at updates 1 / 2 | 7,605 / 7,608 | 8,933 / 8,747 | +17.5% / +15.0% |
  | BC, update alone | 124.86 / 124.81 s | 106.65 / 108.84 s | |
  | BC, a collect segment (steady) | 3.25 s | 2.76 s | |
  | RL from fresh weights, steps/s at updates 1 / 2 | 7,545 / 7,546 | 8,678 / 8,786 | +15.0% / +16.4% |
  | RL, update alone | 124.74 / 124.74 s | 108.74 / 107.08 s | |
  | RL, a collect segment (steady) | 3.55 s | 3.06 s | |
  | peak allocated BC / RL | 5.13 / 5.24 GiB | 1.16 / 1.23 GiB | |

  The losses at RL update 2: 2.354 (core) / 2.355 (branch).
- 08:11 the slow lean/learn tests: test_the_league_tick_compiles_fullgraph failed (the
  league tick compiled with dynamic=True: Dynamo cannot trace the nested compile region over
  unspecialised sizes and numbers, "Freevar has no source"). A check for symbolic sizes did
  not catch it (the free variable is an unspecialised number). Fix: act chunks only under
  compile_shapes "static" (a training run; eager too), one call under "dynamic", and the test
  sets compile_shapes "dynamic" (it already built `graphs("dynamic")`). The static traced
  graph is unchanged by the fix (the compile check above stands). Slow lean/learn tests 08:32:
  6 passed. Quick tier (lean/learn, lean/model, lean/eval) 179 passed before the fix; ruff
  clean.

## Next

- The 5090 re-measures both per-GPU settings (micro_batch_views: the review's
  `memory_curve.py plain B` or `trainer.py`; policy_step_chunks: `profile_collect.py` with the
  setting changed).
- The cold compile is over the 150 s budget on core already (160.3 s): for the author.
