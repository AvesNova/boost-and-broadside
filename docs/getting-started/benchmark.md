---
covers: [bench/]
reviewed: 2026-11-03 09:50
---

# A benchmark

The skeleton benchmark measures the training loop at random sizes and the CPU realtime
path at one:

<!-- not-runnable: needs CUDA and the benchmark lock; takes minutes -->
```bash
uv run --no-sync python -m bench.run
```

It takes the machine-wide benchmark lock first, so it waits while another benchmark runs.
Each cell runs in a fresh process and prints one line:

- a training cell: rollout and training throughput in environment-ticks per second, the
  learner's time per update, peak VRAM and compile time;
- the realtime cell: the median (p50) and 99th-percentile (p99) latency of one CPU tick.

Every result is appended to `bench/results.jsonl` with the commit and the seed. The
[benchmarking guide](../guides/benchmarking.md) explains the lock, the sizes and the cells,
and how to measure a change.
