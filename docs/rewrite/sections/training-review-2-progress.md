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

