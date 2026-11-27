# Queue item 4: second training hot-path review (progress)

Branch `review-4` (from core 679a5ada, items 8a-8c merged). Spec: docs/rewrite/queue.md
"4. Second training hot-path review". Report: training-review-2.md; scripts and raw output:
training-review-2/. Laptop RTX 4070 (8 GB); every GPU job a `bnb-4-*` unit under the GPU lock,
power profile and load recorded per job in the raw output.

## Plan (started 05:25; ETA report 09:00, worst 09:30)

1. Correctness read of the loss rows (CPU, read-only) - 05:25-05:50.
2. Profile: a steady update (3 micro-batches) and a steady rollout segment (BC and RL ticks) -
   ETA 06:20.
3. Compiler-mode grid per graph (act + advance, minibatch_loss), cold and warm - ETA 07:30.
4. Checkpointing / memory budget x micro-batch size curve - ETA 08:15.
5. CUDA-graph plan, host copies, report - ETA 09:00.

## Tried / worked / didn't

- 05:25-05:35 correctness read of the loss rows: every head's rows match the decisions; one
  small label-selection bias found (hit labels near an episode's or the update's end: misses
  never report, hits do). Details in the report.
- 05:27-05:36 profiles (worked): update 165.6 ms per micro-batch (K = 8; off 150.2), GPU busy
  98.2%, 1,001 kernels per micro-batch; BC tick 25.34 ms (GPU busy 98.1%), RL tick (EMA
  playing) 27.7 ms (97.3%). Host floor (same graphs at tiny sizes): update 6.0 ms per
  micro-batch at 1 view, BC tick 2.94 ms, RL tick 5.23 ms at 16 environments.
- 05:36 grid and memory curve queued (jobs2.txt, one unit at a time): ETA 07:00 (worst 07:30).
- 05:40-05:53 grid, first half: loss default cold 49.6 s / warm 3.4 s, 165.9 ms per micro-batch;
  max-autotune-no-cudagraphs cold 498.8 s / warm 3.8 s, 163.4-163.8 ms (-1.3%; "Not enough SMs
  to use max_autotune_gemm mode": the 4070 laptop's 36 SMs, so no GEMM autotuning here);
  reduce-overhead FAILS on both graphs as the code stands: the loss ("accessing gradient
  tensor output of CUDAGraphs that has been overwritten": gradient accumulation over
  micro-batches), act ("accessing tensor output of CUDAGraphs that has been overwritten",
  rollout.py:616, in the first segment). Re-queued the loss with one micro-batch per
  minibatch (no accumulation) to get a number. BC collect warm first segment 14.9 s (cold 94.3).
- 05:53-06:05: collect max-autotune-no-cudagraphs 25.04 ms a tick (default 25.34, -1.2%), cold
  305.7 s (default 94.3). Memory curve (384 views): B 32 158.8 ms per 64 views, B 64 165.0
  (smaller is faster per view here: L2?), B 128 and 256 OOM on 8 GB; checkpoint per sublayer at
  B 64 198.7 ms (+20%), peak 4.61 -> 2.98 GiB. Added B 16 / 48 and budget 0.5 at B 32.
  Remaining: checkpoint 128/256, budget 64/128/256/32, fold check + time, reduce-overhead with
  one micro-batch per minibatch, math attention, max-autotune (ETA 07:15, worst 07:45).
