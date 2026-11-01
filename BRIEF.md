# Brief: a runtime-fast Yemong, from the spec alone

Yemong is the policy and value model of Boost and Broadside, a team space-dogfight game
trained with PPO. An implementation exists elsewhere. You will not see it: you get only the
model's specification, in mathematics and prose, and float64 equivalence fixtures generated
from an earlier version's code. Build the model so it **runs as fast as possible** on a GPU
(an RTX 4070 Laptop, 8 GB), in training (a segment forward and backward) and in rollouts
(one decision step for many views). Being correct is the price of entry; the fixtures
decide it. Beyond that, speed is the goal. You are free to choose any structure, kernel
strategy or library that computes the specified function.

## Where you work

- **This directory:** `/home/vizia/avesnova/bnb-yemong-blind`, branch
  `race/yemong-blind`. Work only here.
- **Scratch:** `/tmp/bnb-scratch-yemong-blind`.
- **Do not read** any other directory under `/home/vizia/avesnova`, any other git branch
  (no `git log --all`, `git show <other branch>`), or any site-packages `bnb` but this
  one. Being blind is the point of the exercise.
- **Python:** `uv run --no-sync ...` (the venv is set up; `uv sync` if you add a library).
  Libraries are allowed (for example Triton, which ships with torch).
- **The GPU may be shared** with other work. Hold the benchmark lock
  (`from bench.lock import bench_lock`; `with bench_lock(): ...`) for every timing, and
  never time without it. A GPU test failing by out-of-memory while another process is on
  the GPU is not your bug: rerun it.

## What you have

- `docs/yemong-spec.md`: **the specification**. Sections 3–9 are the model; section 10
  the contract additions; section 12 the deliberate changes from v1.
- `docs/yemong.md`: the decisions behind it.
- `src/bnb/contracts/`: the observation, action and other contracts the model reads (types
  only).
- `tests/reference/yemong/`: the equivalence fixtures. `README.md` defines the interface
  and the conventions; `yemong_arm.py` is your adapter (the only fixture file you may
  edit). The goldens are fixed.

## Phase 1 (this agent): correct, then first speed

1. **Implement the model** in `src/bnb/models/swift/`: every piece the fixtures test, plus
   the whole model with the spec's `step(carry, obs, reset)` and
   `unroll(carry, obs, replay, reset)` and every head the spec lists.
2. **Pass every fixture** (`uv run --no-sync pytest tests/reference/yemong`): float64 to
   1e-9 and the float32 network tests to 2e-4.
3. **Your own tests** in `tests/`: `unroll` over a segment equals a loop of `step`
   (including a reset mid-segment and a reset at the first step); shapes and dtypes; finite
   outputs; float32 and bfloat16 autocast both run.
4. **The benchmark** `bench/speed.py`, with the lock held, steady state (warm-up excluded),
   median of at least 20 repetitions, CUDA-synchronised timing:
   - **training:** `unroll` forward and backward of a loss (the mean of every head's
     output, summed) on `V` views × `L = 128` steps, `V ∈ {16, 64}`, 5v5 (`N = 10` ships,
     the spec's default fields `M = 10` and zones `Z = 5`), the spec's default
     configuration, starting from a nonzero carry with one reset mid-segment. Report ms
     per segment and view-ticks per second;
   - **rollouts:** `step` forward only for `V = 2048` views, under `torch.no_grad`. Report
     ms per step and view-steps per second;
   - each in float32 and in bfloat16 autocast; and each eager and compiled
     (`torch.compile(fullgraph=True)`; with CUDA graphs where they help). Report every
     cell.
   - Write the results to `bench/results.jsonl` (one JSON line per cell, with the commit).
5. **Then make it fast,** while the fixtures keep passing. Profile first
   (`torch.profiler`), and say what each change bought.

Stop at a clean point when phase 1 is done, or earlier if you have been working long: a
fresh agent continues from your notes.

## Notes and commits

- Keep `NOTES.md` current: design, how each spec piece maps to code, the benchmark table,
  what each speed change bought, what you would try next, and a **Progress** section
  (done, in flight, next) so a fresh agent can take over.
- **Keep your context small:** read only what a step needs; send long output to files and
  read their tails.
- Commit often to `race/yemong-blind` and push (`git push -u origin race/yemong-blind`).
  Author Julian Kizanis <julianzia@gmail.com>; end every commit message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Full words in names; docstrings with shapes and units.

## Hand back

Your final message: what is built, the fixture result, the benchmark table, the biggest
speed levers you found, and what is next. A page or less.
