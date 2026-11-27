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
