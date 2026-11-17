---
covers: [src/bnb/runtime/train.py, bench/run.py]
---

# Compute

**The question:** what does one training run cost, and how does it scale from a laptop GPU
to a desktop one?

Everything runs on one GPU: the game is pure functions of tensors, stepped in compiled
stages and CUDA graphs, so a batch of 2,048 games steps together without leaving the GPU
([Architecture](../explanation/architecture.md)). The same code runs on both machines
below; only the batch's micro-batching changes to fit the memory.

## Throughput

{{figure: docs/results/throughput.png | Environment steps per second for behaviour cloning and reinforcement learning, on an RTX 4070 Laptop and an RTX 5090}}

| GPU | Behaviour cloning (steps/s) | Reinforcement learning (steps/s) |
|---|---|---|
| RTX 4070 Laptop (8 GB) | {{throughput_laptop_4070_bc_steps_per_second:,.0f}} | {{throughput_laptop_4070_rl_steps_per_second:,.0f}} |
| RTX 5090 (32 GB) | {{throughput_rtx_5090_bc_steps_per_second:,.0f}} | {{throughput_rtx_5090_rl_steps_per_second:,.0f}} |

A step is one environment advancing one tick. The rate is over whole updates, collection
and learning together, at steady state after the first compile, from the run's own record
(train/environment_steps_per_second). On the laptop the power profile is recorded with
each measurement, since it sets the GPU's power limit.

The RTX 5090's numbers are dry run 3's, `dry-v26-s1`{.plain}: commit `c23706cc`{.plain},
preset `real_scale_all_modes`{.plain}, 2,048 environments, `--micro-batch-rows 480`{.plain},
one chain, {{chain_steps_bc:,.0f}} steps of behaviour cloning then
{{chain_steps_rl:,.0f}} of reinforcement learning (a budget of 200M, extended by hand, then
stopped), on a rented machine. The compile time is the first update's time over the median
update's, in each stage.

{{pending: throughput_conditions_laptop | The RTX 4070 Laptop's measurement: its commit, configuration (micro-batch rows) and power profile.}}

## One run

| | RTX 5090 |
|---|---|
| Behaviour cloning, {{chain_steps_bc:,.0f}} steps | {{chain_hours_rtx_5090_bc:.1f}} h |
| Reinforcement learning, {{chain_steps_rl:,.0f}} steps | {{chain_hours_rtx_5090_rl:.1f}} h |
| First compile (not cached), both stages | {{compile_minutes_rtx_5090:.0f}} min |

{{pending: compute_reading | One or two sentences: the cost of the whole arm program in GPU hours, and what the laptop can and cannot do (for example a full run in days rather than hours).}}
