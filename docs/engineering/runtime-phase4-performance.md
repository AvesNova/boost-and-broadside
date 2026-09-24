# Decision runtime Phase 4 performance validation

Date: 2026-09-24

This report compares the runtime before the semantic migration
(`48042f23e1e017598fb6755e7170eec925a8e656`) with the completed Phase 3 runtime
(`138f540beaa8e4abe504b7808c057f7dfdd0db76`). The result is a performance
regression check, not a claim that all real-time targets are met.

## Result

No substantial throughput or latency regression was found in paired CUDA tests. The
production-shaped PPO/BC update was 1.2% faster in the closest matched pair, batched
environment throughput was 4.4% higher by the median-of-runs throughput metric, 5v5
latency was 3.9% lower, and 50v50 latency was effectively unchanged at +0.24%. These
small differences are close enough to host and GPU variation that they should be read
as **no regression**, not as an optimization claim.

The separate CUDA streams remain useful. Serializing environment and network work made
the rollout phase 2.5% slower and the complete update 1.3% slower in the measured host
state.

The measured low-power configuration did not meet the 30/60 FPS targets. Absolute
single-game timing varied sharply during the session, while the paired comparisons
remained directionally stable. This variability calls for interleaved repetitions, not
for a different power mode: low-power operation is the project's established benchmark
condition and its results are valid. Further real-time optimization is still required.

## Method and machine state

- NVIDIA GeForce RTX 4070 Laptop GPU, 8,188 MiB; driver 595.84.
- PyTorch 2.13.0+cu130 and Python 3.13.11.
- Three alternating before/after runs for environment throughput and each real-time
  scenario. Each result below is the median across the three runs; the regression
  column is the median of the three matched-pair percentage changes.
- Before runs used a detached worktree and the current virtual environment. Depending on
  the harness, either its matching old entry point or the old package through `PYTHONPATH`
  was used. Some JSON metadata therefore describes the invoking harness checkout rather
  than the imported package. The explicit revisions above are authoritative.
- The laptop CPUs were in the `powersave` governor at roughly 0.55--0.80 GHz when sampled.
  During an observed run, the GPU was in P0 but drew only about 17 W with roughly 33%
  utilization. No power settings were changed.
- Every comparison used CUDA. CPU was used only for ordinary Python orchestration and
  was not selected as the tensor execution device.

## Large-batch results

### Environment throughput

The environment-only workload used 128 parallel 5v5 environments for 100 measured
steps. It isolates batched simulation and observation work from policy/update cost.

| Metric | Before | After | Interpretation |
| --- | ---: | ---: | --- |
| Window mean | 66.01 ms | 63.21 ms | No regression |
| Environment transitions/s | 1,939 | 2,025 | +4.4% |
| Ship decisions/s | 19,390 | 20,249 | +4.4% |
| Paired median sample-latency change | -- | -4.74% | Pair changes: -4.74%, +0.30%, -5.11% |
| Peak allocated | 19.148 MiB | 19.151 MiB | +0.003 MiB |

### Production-shaped PPO/BC pipeline

The end-to-end profile used the real trainer with 128 parallel 5v5 environments, eight
steps per rollout, 1,024 environment transitions and 10,240 ship transitions per
update, four PPO epochs/minibatches, the active behavior-cloning and policy-gradient
losses, CUDA bf16 rollout inference, and separate environment/network streams. Elo was
disabled to isolate the training pipeline. Compilation was disabled in both revisions
so compile-cache state could not dominate the short comparison.

The closest CPU-pinned matched pair was:

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Complete update | 3.825 s | 3.777 s | -1.24% |
| Environment transitions/s | 267.7 | 271.1 | +1.25% |
| Rollout/update | 1.499 s | 1.485 s | -0.89% |
| Optimizer phase/epoch | 0.565 s | 0.556 s | -1.60% |
| Peak allocated CUDA memory | 594.89 MiB | 597.04 MiB | +2.15 MiB |
| Peak reserved CUDA memory | 738 MiB | 742 MiB | +4 MiB |

An unpinned matched pair was also effectively flat (about 0.7% faster after). A later
old-revision run completed 3.7 times faster than adjacent old and new runs while every
phase scaled together; that operating-state outlier was rejected rather than attributed
to code.

The small memory increase is expected from the wider pending-action input and joint
42-way actor. It is 0.36% of allocated memory in this reduced workload.

### CUDA-stream overlap

On the after revision, the matched overlap-enabled and serialized runs measured:

| Metric | Streams | Serialized | Serialization cost |
| --- | ---: | ---: | ---: |
| Complete update | 3.777 s | 3.828 s | +1.34% |
| Rollout/update | 1.485 s | 1.522 s | +2.50% |
| Peak allocated | 597 MiB | 587 MiB | -10 MiB |
| Peak reserved | 742 MiB | 626 MiB | -116 MiB |

The concurrency path therefore remains active and beneficial, although its measured gain
was modest under low GPU utilization. Its additional reserved memory is the cost of
concurrent workspaces, not a semantic-runtime allocation leak.

## Low-N latency results

The real-time harness ran two randomly initialized neural policies sequentially, with
10 warmup decisions and 40 measured decisions for 5v5 and 30 measured decisions for
50v50. It includes belief updates, policy inference, environment advance, observation
construction, and the canonical pending-action state. It excludes rendering.

| Scenario | Before median | After median | Paired change | After measured rate |
| --- | ---: | ---: | ---: | ---: |
| 5v5 | 120.47 ms | 115.99 ms | -3.88% | 8.53 Hz |
| 50v50 | 119.65 ms | 120.40 ms | +0.24% | 8.16 Hz |

The 50v50 pairs changed by +0.24%, +6.17%, and +0.18%; the middle run is consistent with
the observed machine-state noise rather than a repeatable regression. A stream-scheduled
low-N run was slower than sequential execution in this CPU-dispatch-limited state, so
sequential remains the relevant low-latency result for now.

Absolute results varied too much to infer a precise hardware ceiling. Earlier in the
same Phase 4 session, the old revision measured 35.13 ms at 50v50 (28.5 Hz), versus
about 120 ms during the paired series. This fourfold movement without a code change is
why comparisons must remain repeated and interleaved. It does not invalidate the tested
condition: neither the paired 5v5 nor 50v50 workload met 30 FPS, so the current low-power
performance result is a failed real-time target, not an unresolved benchmark gate.

## Episode-stable neural league identity

The identity-drain profile used 128 environments, a 64-environment league block, four
16-environment neural slots, eight rollout steps, eager CUDA policies, and distinct
resident policy modules. It compared steady `K=4` generation forwarding with the
bounded `K+1=5` state while one generation drains.

| State | Median per eight steps | Allocated CUDA memory |
| --- | ---: | ---: |
| Stable four generations | 3.170 s | 347.03 MiB |
| One draining generation | 3.564 s | 355.55 MiB |
| Delta | +12.43% | +8.52 MiB |

This validates the user's concern: a small extra generation has a disproportionate
forward-pass cost because the current implementation retains fixed slot shapes and runs
one forward per generation. The cost is temporary and bounded to one extra generation,
which is preferable to changing identity mid-episode, but drain duration should be
reported in long training runs and optimized if it occupies a meaningful fraction of
training.

There are two separate bounds:

- active policy forwards are bounded to `K+1` generations;
- the checkpoint roster's LRU may keep up to `league_size` checkpoint weights resident
  (20 in the production profile), even when those policies are not currently forwarded.

The latter consumes memory and can make load/eviction pauses visible, but does not add a
per-step forward pass. The profile measured roughly 8.5 MiB for one additional eager
policy generation plus its state; compiled-policy cache residency may be larger. Changing
the cache cap needs a longer representative training measurement, not an assumption that
`K+1` is automatically the best memory/IO tradeoff.

## Benchmark maintenance completed

The real-time and deadline-subphase harnesses previously reimplemented pending-action
buffering and wrote unmasked actions into both team observations. They now use
`PendingActionState`, including actuator-continuity cancellation, reset handling, and
authoritative perspective masking. `rl_kernel_profile.py` was also updated for the
renamed rollout runtime field. The environment-only fixed-action throughput harness is
intentionally lower-level and does not implement a controller decision loop.
The CUDA smoke test also found and fixed an older CLI bug where the deadline profiler's
advertised `--compile none` option passed the literal string `none` to `torch.compile`.

These changes ensure future performance measurements exercise the semantics they claim
to measure instead of preserving a stale pre-refactor loop in benchmark code.

## Decision

Phase 4's relative performance gate passes: the shared semantic primitives did not cause
a meaningful regression in either tested regime, and CUDA-stream overlap remains intact.
The small model/runtime memory increase is acceptable.

The absolute real-time gate fails in the established low-power benchmark condition;
meeting it requires future optimization, not different hardware or a different power
mode. Continue to use repeated, interleaved measurements because the condition still
showed substantial clock/load variation. Separately, add long-run telemetry for league
generation count, drain duration, checkpoint loads/evictions, and resident policy memory
before changing the stable-identity design or roster cache.
