# Buffered environment/policy concurrency: implementation and benchmark plan

Status: implemented and measured on the target RTX 4070 Laptop GPU. Interactive
NN actions now use the trained one-tick delay. The realtime harness compares
sequential execution, two CUDA streams dispatched by one host thread, and two
persistent host threads under identical buffered semantics. Returned observation
channels own their final storage, so concurrency does not copy the observation.
Concurrency remains a benchmark option rather than the interactive default
because the measured change was within run-to-run variation.

## Objective

Measure and improve environment/policy overlap while preserving the one-decision
action delay used by PPO for neural-network policies. Cover single-environment
5v5 and 50v50 inference and actual PPO collection batches. Human and scripted
controllers remain immediate. Keep sequential execution available as a reference
and as a fallback when concurrency loses throughput.

The interactive target is a simulation decision median at or below 25 ms
(preferably 20 ms), p99 below 33.3 ms, with rendering excluded. Measure rendering
separately afterward. Training success means a repeatable improvement in rollout
throughput and total PPO iteration time with equivalent training semantics.

## Measured result (RTX 4070 Laptop GPU)

The representative compiled setting was `reduce-overhead`, two policy sides,
bullet perception enabled, no rendering, 30 warmup steps, and 100–200 measured
steps per arm. The buffered 5v5 medians were 83.4 ms sequential, 87.1 ms with
one-thread streams, and 84.7 ms with two host threads. For 50v50, a 200-step run
measured 90.0, 87.3, and 82.8 ms respectively, but reversed interleaved pairs
measured 89.7 ms sequential versus 88.7 ms threaded on average. That 1.1% paired
difference is within observed variation, and the threaded p99 was less stable.
Policy-first stream dispatch measured 89.1 ms for 50v50.

The policy and environment become slower when dispatched from competing host
threads, erasing most nominal overlap. One-thread streams cannot feed both
branches simultaneously while Python dispatch is the limiting resource. Keep
sequential interactive execution; retain all three harness modes for future
environment-fusion or hardware experiments. These results remain far above the
33.3 ms deadline, so concurrency does not make either scenario realtime.

## Existing implementation and evidence

- `train/rl/opponents.py::_step_environment_and_network` already runs primary
  PPO environment and network work on separate CUDA streams, with waits before
  dispatch and a join afterward. `ppo.py::_initialize_rollout_runtime` creates
  the streams.
- `ppo.py::_collect_aux_steps` uses delayed actions but executes sequentially.
- `modes/interactive.py::_run_interactive_loop` and
  `benchmarks/realtime_latency.py` currently infer NN actions and immediately
  apply them. That is a correctness bug: NN policies were trained with a one-tick
  action delay. Fix policy-controlled teams to apply their buffered action while
  preserving immediate input for human and scripted teams. Do not attribute the
  semantics fix itself to a concurrency speedup.
- `benchmarks/rl_pipeline_profile.py` already exposes `--no-overlap`, `--timing
  wall`, and `--detail`. Detailed timing disables overlap; it is diagnostic,
  not the headline concurrency comparison.
- [The previous throughput investigation](rl-throughput.md#overlapping-the-env-and-policy-cuda-streams)
  measured 46.16 s serialized versus 45.48 s with streams, within noise. Both
  workloads were dispatched by one Python thread. This is historical evidence,
  not a current baseline or proof that overlap cannot help smaller batches.
- The supplied 50v50 GPU measurements are 17.7 ms for policy/belief work and
  52.3 ms for the environment. Perfect overlap alone would still miss 30 Hz.

## 1. Establish the timing and ownership contract

For an NN-controlled team at decision t, snapshot observation o_t and pending
action b_t. Physics computes s_(t+1), reward r_t, and termination using b_t while
inference computes action a_t, its log probability/value, predictions, and next
recurrent state from o_t. At the join, set b_(t+1) = a_t for continuing episodes
and compose o_(t+1), including the pending action in the existing previous-action
feature. Human and scripted actions are merged into the action applied on the
current tick; they do not enter the NN delay buffer.

Preserve the existing PPO row convention (o_t, a_t, logprob_t, value_t, r_t).
The environment state is augmented by b_t; do not shift rollout rows or rewards
to match the later physical effect of a_t. Keep policy weights fixed throughout
collection; optimizer updates start only after collection completes.

Audit observation aliases, cached fields, visibility, team IDs, privileged
targets, recurrent state, belief state, and compiled output ownership. In
particular, inspect paths without belief composition: its cloning must not be
the accidental reason concurrent reads are safe. Ensure rollout-buffer copies
finish before any backing observation is reused.

Confirm the owned-output property with parity tests whenever observation assembly
changes. Old observations must remain immutable through inference, belief advance,
and rollout storage. Scripted agents that inspect live state must finish before
physics mutates that state; they remain on the immediate pre-launch path.

At episode reset, clear pending actions and reset hidden/belief state before the
next decision; ensure reset observations expose the cleared action. Audit ship
death/respawn separately so a queued action from a previous life cannot affect a
new life unintentionally. Preserve truncation, transition-continuity, bootstrap,
and existing recurrent-reset semantics; fix any discovered semantic bug in both
reference and concurrent paths before comparing performance.

## 2. Make scheduling selectable and verify safety

Expose an explicit sequential/streams execution setting for rollout collection
and the realtime benchmark. Reuse the current primary scheduler rather than
adding a second implementation. CPU uses the sequential path. Both modes execute
the same controller-specific action timing with identical numerical settings.

Use a coordinator stream, an environment stream, and a network stream. Both
workers wait for input preparation; the coordinator waits for both workers
before action merge, belief updates, resets, buffer writes, and the next tick.
Keep both team forwards on the network stream initially. Test network-first and
environment-first enqueue order as separate benchmark variants: one Python
thread still issues the work sequentially.

Establish tensor lifetime safety using persistent ownership and explicit joins,
or `record_stream` where needed. Keep recurrent outputs owned across compiled
calls. Do not replay graphs concurrently from a shared graph memory pool.
Warm compilation/capture before timing on the intended execution paths.

Do not compile the mutable wrapper wholesale. The existing throughput report
and `tests/env/test_compiled_tick.py` document dropped mutation/buffer writes.
Any compilation experiment must use an explicitly functional boundary and pass
state/observation parity tests.

## 3. Add focused correctness tests before performance work

- Compare sequential and streams execution over multiple ticks from identical
  world, observation, pending-action, hidden, and belief state. Check complete
  state, visibility, rewards, action application, observation features, model
  outputs, rollout rows, and final bootstrap/advantages.
- Use fixed actions or controlled sampling for deterministic comparisons;
  merely setting one global seed is insufficient when concurrent resets and
  policy sampling change random-number consumption order. Then exercise normal
  stochastic execution separately.
- Force episode termination, truncation, ship death/respawn, no-bullet and dense
  projectile states, and rapid buffer reuse. Verify no old NN action crosses
  reset. Verify human and scripted actions affect the current tick while policy
  actions affect the following tick, including mixed-controller matches.
- Cover belief enabled/disabled, both team perspectives, shared and distinct
  policies, scripted opponents, and eager/compiled entry points. Add a bounded
  PPO update parity check with fixed minibatch order, then a training smoke run.
- Require exact equality for discrete state and documented tolerances for
  floating outputs. GPU tests skip explicitly when CUDA is unavailable.

## 4. Benchmark current primary concurrency before extending it

Start from `rl_pipeline_profile.py` with wall timing, with and without
`--no-overlap`; keep detailed synchronized profiling in separate runs. Extend
`realtime_latency.py` to compare sequential-buffered versus streams-buffered NN
execution. Remove immediate NN action application from the supported benchmark;
if retained temporarily for historical comparison, label it as buggy legacy
semantics. Immediate human/scripted timing remains supported behavior.

| Workload | Initial matrix |
|---|---|
| Interactive simulation | B=1; 5v5 and 50v50; one and two policy sides |
| PPO collection | 5v5 and 50v50; small batch, intermediate batch, actual configured batch that fits VRAM |
| Production training | Current multiscale profile, opponent mix, evaluators, and PPO update configuration |
| Execution | Sequential delayed; existing streams delayed; enqueue-order variant |

Use real checkpoints and representative combat states for final results. Record
policy architecture, checkpoint, ship/field counts, bullet capacity and activity,
perception flags, precision, compile mode, threads, GPU/CPU, software versions,
power/clock conditions, seeds, and git revision. Random-policy timings may be a
smoke check but do not substitute for trained-policy combat workloads.

Warm compilation and device clocks; record startup cost separately. Use at least
five interleaved A/B run pairs, initially at least 2,000 measured interactive
ticks per run and one full rollout per collection run. Extend noisy runs before
claiming a gain. Separate reset-inclusive results and steady-state diagnostics.
Use identical reset/episode workloads where practical and report reset counts.

Report interactive median/p90/p99/max decision latency, fraction above 33.3 ms,
total elapsed ticks per second, and headroom. Report collection env transitions/s,
ship decisions/s, wall time per rollout, full PPO iteration time, peak allocated
and reserved memory. Do not label inverse median latency as measured sustained
throughput. Include the cost of snapshots, joins, belief handling, and storage.

For training throughput, synchronize only around timed windows. For interactive
latency, wait at the real tick completion boundary. Never insert synchronization
between environment and inference in the measured concurrent arm.

Capture short, separate profiler traces with ranges for snapshot, physics,
perception/observations, each policy, beliefs, buffer storage, and joins. Check
actual overlapping kernel intervals, host launch gaps, and synchronization.
Summed kernel durations are not GPU utilization when kernels overlap. Include
unprofiled results as the performance evidence.

## 5. Extend only after the reference comparison is trustworthy

Route auxiliary scales through the verified scheduling pattern and measure its
incremental effect. Initially overlap only environment versus network within
each scale; avoid concurrent calls to the same compiled policy across scales.
Then fix NN action buffering in play/watch and integrate concurrent scheduling,
covering immediate keyboard/scripted overrides, policy-versus-policy and mixed
matches, shared-agent shortcuts, terminal display, and reset handling.
Read renderer state only after the simulation join; benchmark rendering as its
own follow-up workload rather than mixing it into the physics baseline.

If traces show host dispatch starvation, test functional compilation/fusion or
safe graph replay for the dominant region, then repeat the same A/B comparison.
If kernels saturate the GPU, retain serial scheduling for those batch sizes.
If the environment remains above budget, prioritize measured environment
bottlenecks (including rewards/statistics and perception) in separately attributed
changes. Python threads, multiprocessing, and full engine rewrites are not the
initial implementation.

## Decision gates and deliverables

Correctness is mandatory. Treat a speedup as actionable when it exceeds measured
run-to-run variability; target at least 5% improvement and check paired-run
uncertainty before changing defaults. For training, require no meaningful total
iteration regression or loss of the intended batch size to extra memory. For
interactive use, report whether the latency target was actually achieved even
if concurrency improves performance. A neutral result is useful evidence and
does not justify further scheduling complexity on its own.

Deliver in order: (1) timing/ownership audit and selectable reference;
(2) parity tests and reproducible primary/realtime A/B benchmark;
(3) auxiliary and interactive integration;
(4) JSON results, representative traces, and a report choosing execution mode
per workload. Preserve current primary training defaults until evidence supports
a change. This plan does not authorize claiming benchmark success in advance.

Implementation references: [PyTorch CUDA stream and graph semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
and [tensor stream lifetime tracking](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html).
