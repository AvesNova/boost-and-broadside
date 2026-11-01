---
covers: [bench/]
---

# Benchmarking

The harness lives in `bench/`, outside the package. Its rules come from the physics race:
size the measurement first, benchmark at steady state and at sizes nobody tuned for, and
run one benchmark at a time on the machine.

## The lock

Only one benchmark runs on the machine at a time. `bench.lock.bench_lock` is a context
manager that takes an exclusive `flock` on the file `bench.lock.LOCK_PATH`; a second
holder waits until the first releases it.

The lock passes through to child processes. While a process holds it, it sets the
environment variable named by `bench.lock.HELD_ENV` to `1`, and `bench_lock` in any
process that sees that value proceeds without waiting. `bench.run` sets it for each cell
it starts, so cells run under their parent's lock instead of waiting on it.

Measure your own change under the lock:

<!-- not-runnable: needs CUDA and the benchmark lock -->
```python
from bench.lock import bench_lock
from bench.run import run_cell

with bench_lock():
    print(run_cell({"kind": "train", "preset": "mixed", "envs": 4096}))
```

Several agents share one GPU. A measurement taken without the lock can overlap another
agent's benchmark, and then both numbers are wrong.

## Cells

A **cell** is one measurement in a fresh process (`bench.cell`). It prints one line,
`RESULT {json}`, which `bench.run.run_cell` parses.

- **`train`:** builds the trainer, runs one update (compilation, the CUDA-graph capture
  and the learner's compile), then one episode of rollouts so that the staggered clocks
  and combat reach a steady state. It then times three rollouts and three PPO updates and
  reports the fastest of each: rollout and training throughput in environment-ticks per
  second, the rollout and learner times, peak VRAM, compile time, and the rollout's
  event counters. A train cell gives either a `size` (`ships_per_team`, `num_fields`,
  `arena_radius`) or a `preset`: a game preset name, or `mixed`.
- **`realtime`:** one match on the CPU at batch size 1 (`bnb.runtime.realtime.Realtime`),
  compiled. It reports the median and 99th-percentile latency of one tick over 1000 ticks
  after 300 warm-up ticks.

One cell alone, without `bench.run` (this does not take the lock):

<!-- not-runnable: needs CUDA -->
```bash
uv run --no-sync python -m bench.cell '{"kind": "train", "preset": "deathmatch", "envs": 4096}'
```

## The skeleton benchmark

`uv run --no-sync python -m bench.run [--seed N]` runs three training cells at random
sizes and one realtime cell, under the lock, and appends every result to
`bench/results.jsonl` with the commit, the seed and a time stamp. It is sized to finish in
about five minutes on the development laptop, most of it compilation.

The sizes come from `bench.sizes.draw`, which draws ship, field and arena sizes the way the
physics race did. Benchmarking at sizes nobody tuned for keeps optimizations general:

<!-- generated-table: bench-sizes -->

`bench.sizes.envs_for` sets each cell's batch so that environments times ships is the
ship budget, rounded to a whole multiple of what the pools need.

<!-- runnable -->
```python
from bench import sizes

for size in sizes.draw(sizes.SKELETON_SEED, 3):
    print(size, sizes.envs_for(size))
```

The skeleton seed is public. A section's acceptance run uses a seed the section's agent is
not given, so no one can tune for the sizes it draws.

## Reading results

`bench/results.jsonl` has one JSON object per cell and run, appended in order. Compare a
change with earlier lines at the same seed and cell. A crashed or timed-out cell is
recorded with `status` `crash` or `timeout` and the end of its error output.

Every number quoted in these docs or in a section's notes names where it was measured:
the commit, the cell and the machine.
