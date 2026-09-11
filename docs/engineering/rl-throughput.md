# Where RL training time goes, and what moved it

A profiling pass over the whole `bnb train --profile rl` pipeline: what it spends
wall clock on, what was changed, what each change was worth, and what was tried
and rejected. Companion to [memory optimization](memory-optimization.md), which
covers the same pipeline's memory rather than its speed.

The short version: the pipeline was **CPU-dispatch bound with a synchronization
stall**, not compute bound. It issued 8,498 CUDA kernels per rollout step against
118–148 ms of actual GPU work, and drained the launch queue 41 times per step on
small tensors built from host data. The GPU idled roughly 40% of the time.

## Result

Interleaved against the pre-optimization commit (`d7ebb1c`), same machine, same
launch, one warmup and two measured updates per run:

| | before | after | change |
|---|---:|---:|---:|
| **sustained throughput** | **2,413 env steps/s** | **3,083 env steps/s** | **+27.8%** |
| seconds per update | 203.69 | 159.44 | −21.7% |
| rollout collection | 131.07 s/update | 84.67 s/update | **−35.4%** |
| PPO update | 45.43 s/epoch | 47.05 s/epoch | +3.6% (noise) |
| peak allocated | 2527 MiB | 2526 MiB | — |

Run-to-run spread between the two "after" runs is 2.7%. The gain is concentrated
entirely in rollout collection; the update phase is unchanged within noise, and
none of the four changes targeted it directly.

The four changes, and what each was worth measured on its own:

| # | change | class | measured |
|---|---|---|---|
| 1 | read observation channels by slice, not list index | behaviour-preserving | +11.0% end to end; 41.3 → 16.3 host syncs per rollout step |
| 2 | keep per-step constants off the host-to-device path | effectively equivalent | +15.0% end to end; 16.3 → **0.3** host syncs per rollout step |
| 3 | scan the field axis field-major | effectively equivalent | 42x on the dominant shape; +1.5% end to end |
| 4 | compile the policy entry point callers actually use | effectively equivalent | 1.97x on the rollout forward; **+6.5%** end to end, interleaved |

Changes 1–3 compound to −16.3% per update against the old baseline; change 4
takes another −6.1% on top. `203.69 × 0.837 × 0.939 = 160.1` against the 159.44
measured, which is the two independent A/B suites agreeing.

Nothing here changes the objective, the batch construction, the opponent
curriculum, or the learning algorithm.

## Hardware, configuration, and method

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Laptop, 8188 MiB, driver 595.84, CUDA 13.2 |
| CPU | 13th Gen Intel Core i7-13620H, 10C/16T |
| RAM | 46 GiB |
| Torch | 2.10, bf16 autocast, `--compile default` |
| Branch | `frontline/07-map-memory` |

The measured configuration is what `bnb train --profile rl` resolves to on this
machine with no `.vram.json` present:

| | |
|---|---:|
| parallel environments | 1280 |
| rollout length | 128 steps |
| rollouts per update | 3 |
| **environment steps per update** | **491,520** |
| PPO minibatches | 32 |
| microbatch tokens | 25,000 (5 micro-batches per shard minibatch) |
| PPO epochs | 2, cut to 1 by `target_kl` on some updates |
| ships / fields / zones / boundary | 8 / 10 / 5 / 1 → 24 entity tokens per env |
| paradigm | `ego_pass` |
| Elo evaluator | 5 slots × 512 = **2560** environments, stepped every rollout step |
| checkpoint interval | every update (38 MB, written from a worker thread) |

### The harnesses

- [`benchmarks/rl_pipeline_profile.py`](../../benchmarks/rl_pipeline_profile.py)
  builds the trainer through the same `resolve_training_launch` path `bnb train`
  uses, attaches timers by monkeypatch, and runs N updates after a warmup.
  Nothing in `src/` is modified, so the pipeline being measured is the shipped
  one. Two modes: `wall` synchronizes only at update boundaries; `sync`
  synchronizes around every timed region (correct attribution, and `--detail`
  additionally serializes the env/policy streams so the two are separable).
- [`benchmarks/rl_kernel_profile.py`](../../benchmarks/rl_kernel_profile.py)
  runs the real trainer and profiles a bounded window — a fixed number of rollout
  steps, or a fixed number of optimizer minibatches — and reports the CPU/GPU
  split, the launch count, and the top operators. The window is bounded because
  profiling a whole PPO epoch buffers around ten million events and exhausts host
  RAM.

Every figure below is one warmup update followed by two measured updates unless
stated otherwise. Results are reported as **rollout seconds per update** and
**update seconds per epoch** separately, because `target_kl` early-stops the
second epoch on some updates and a per-update total mixes one-epoch and
two-epoch updates.

### A measurement confound worth knowing about

Partway through this work the full-pipeline numbers stopped agreeing with each
other: a change that touches only the PPO update made the *rollout* read 46%
slower than the same code path hours earlier. The machine had changed, not the
code.

```
platform_profile               = quiet   (choices: quiet balanced performance)
cpufreq governor               = powersave
energy_performance_preference  = power
RAPL PL1 / PL2                 = 35 W / 45 W
busiest core under 105% load   = ~1.0 GHz   (cpuinfo_max_freq 4.7 GHz)
GPU                            = 2445 MHz, 62 C, no throttle flags
```

The CPU is held near 1 GHz and shares a power budget with the GPU, so the part
of the pipeline that dominates wall clock drifts by tens of percent between
processes depending on how warm the machine is. **Comparisons separated in time
are not reliable on this machine.** Everything below is either an in-process A/B
(both arms in one process, the first arm repeated at the end to expose drift) or
an interleaved control/variant sequence.

This is deliberately left as it is: `quiet` is the machine's own configuration
and training really runs under it. It is a configuration tradeoff, noted at the
end, not a code change.

## The baseline breakdown

`--mode sync --detail`, two measured updates averaging 1.5 epochs, 154.92 s per
update. Attribution is correct here; the syncs cost about 3% over `wall` mode,
which by itself says the CPU was never running ahead of the GPU.

| phase | s/update | share |
|---|---:|---:|
| **rollout collection** | **99.33** | **64.1%** |
| ├ Elo evaluation | 53.05 | 34.2% |
| │  ├ policy forwards (live 2048 env + avg 512 env) | 21.28 | 13.7% |
| │  ├ evaluator `env.step` (2560 env) | 13.01 | 8.4% |
| │  ├ evaluator observation build | 8.47 | 5.5% |
| │  ├ anchor actions (scripted/random blend) | 2.91 | 1.9% |
| │  └ rating update, match counts, resampling | 0.97 | 0.6% |
| ├ primary rollout step | 46.16 | 29.8% |
| │  ├ training `wrapper.step` | 19.45 | 12.6% |
| │  │   └ of which observation build | 6.24 | 4.0% |
| │  ├ policy forward (2560-batch ego pass) | 17.14 | 11.1% |
| │  ├ belief compose + advance (4 each per step) | 7.20 | 4.6% |
| │  ├ scripted agent | 2.76 | 1.8% |
| │  └ privileged targets, action select, buffer writes | 3.63 | 2.3% |
| **PPO update** (at 1.5 epochs; 34.9 s per epoch) | **52.39** | **33.8%** |
| ├ backward | 26.92 | 17.4% |
| ├ forward + loss | 21.03 | 13.6% |
| ├ host minibatch gather | 1.37 | 0.9% |
| ├ host→device staging | 0.86 | 0.6% |
| └ denominators, optimizer, clipping | 0.39 | 0.3% |
| lambda aggregation | 0.59 | 0.4% |
| rollout shard → host copy | 0.80 | 0.5% |
| GAE | 0.13 | 0.1% |
| next-state labels | 0.12 | 0.1% |
| checkpoint dispatch | 0.04 | 0.0% |
| schedule, metrics, logging, ladder | <0.01 | 0.0% |

Everything periodic — GAE, auxiliary labels, lambda aggregation, the host copy,
checkpointing, logging, the ladder — totals **1.7 s per update, 1.1%**. There is
nothing to win there and it is not mentioned again.

An ablation bounds the evaluator's cost: deleting its per-step work entirely
takes the rollout from 99.33 to 45.48 s per update, so **Elo evaluation is
53.9 s per update**, about 35% of training wall clock. It will grow: two of its
five slots idle as random-vs-random until the first checkpoint milestone, and
become real policy forwards after it.

## Where the time actually was: the kernel profile

12 real rollout steps, CPU and CUDA activities:

| | |
|---|---:|
| summed self CUDA time | 1.775 s → **148 ms per rollout step** |
| summed self CPU time | 3.502 s → 292 ms per step (profiler-inflated) |
| `cudaLaunchKernel` | 101,976 calls → **8,498 launches per rollout step**, 673 ms CPU |
| `cudaStreamSynchronize` | 492 calls → **41 per rollout step**, 699 ms CPU (**20%**), 1.42 ms each |
| profiler events | 48,127 per rollout step |

Top GPU consumers over the window:

| operator | self CUDA | share | calls | average |
|---|---:|---:|---:|---:|
| **`aten::cumprod`** | 357.7 ms | **20.2%** | 288 (24/step) | **1.242 ms** |
| `aten::mul` | 235.1 ms | 13.2% | 11,988 | 19.6 µs |
| `aten::copy_` | 192.3 ms | 10.8% | 17,880 | 10.8 µs |
| `aten::mm` | 167.9 ms | 9.5% | 1,800 | 93.3 µs |
| `aten::add` | 122.6 ms | 6.9% | 5,412 | 22.7 µs |
| `aten::cat` | 115.6 ms | 6.5% | 5,568 | 20.8 µs |
| `aten::gelu` | 98.6 ms | 5.6% | 612 | 161.7 µs |
| `aten::_fused_rms_norm` | 95.3 ms | 5.4% | 864 | 119.0 µs |
| SDPA forward | 55.7 ms | 3.1% | 216 | 257.9 µs |

Two things stand out. The pipeline issues 8,498 kernels for 148 ms of GPU work —
an average kernel of 17 µs, and 41 full queue drains per step on top. And a
single operator, a cumulative product over a length-10 axis, costs more GPU time
than every matmul in the policy put together.

### Locating the synchronizations

`torch.cuda.set_sync_debug_mode("warn")` over three real rollout steps,
attributed by Python stack: **41.3 host synchronizations per step**.

| per step | site | cause |
|---:|---|---|
| 15.0 | `features.py` `Accessor.get`, via `get_input_vector` | `val[..., [0]]` — a list index builds the index tensor on the host |
| 10.0 | the same, via `get_target_vector` | " |
| 4.0 | `observation.py` entity and bullet paths | `torch.tensor(field_index_step, device=cuda)` to take its log |
| 4.0 | `frontline.py` `roles_from_front`, tick and reset | `torch.tensor([...five roles...], device=cuda)` |
| 3.0 | `scripted_utils.py` `select_targets` | `torch.tensor(inf, device=cuda)` |
| 2.0 | `field_physics.py` `index_from_level` | `torch.as_tensor(index_step, device=cuda)` |
| 2.0 | `field_generation.py` | `torch.tensor([-2,-1,1,2], device=cuda)` |
| 1.0 | privileged observation build | as `observation.py` above |

Every one is the same shape of mistake: **a small CUDA tensor built from host
data inside a per-step loop**. Each costs a full queue drain — 1.42 ms of CPU
waiting — which is also why the CPU never ran ahead and why `--mode sync` was
almost free.

Boolean-mask assignment (`tensor[done_mask] = 0`, used by the belief reset and
the action buffer) was checked and does **not** synchronize: PyTorch routes it
to `masked_fill_`. That pattern is innocent.

## What changed

> **On the per-change numbers below.** Changes 1–3 were measured before the
> CPU-clamp drift was understood, in a single 50-minute window as a
> monotonically improving sequence. Their *direction* is independently
> corroborated — the synchronization census for 1 and 2, the microbenchmark for
> 3 — and their total is confirmed by the interleaved cumulative comparison
> above. Their individual percentages, and in particular the update-phase
> columns, carry drift uncertainty and should be read as indicative. Change 4
> and the cumulative result were measured interleaved and do not.

### 1. Read observation channels by slice, not by list index

`Accessor.get` narrowed a declared channel list with `val[..., [0]]`. That is
advanced indexing: PyTorch builds the index tensor on the host and copies it to
the device, and the copy drains the queue. The feature pipeline runs on every
observation read — twice per rollout step for the training pass, twice more for
the belief trackers, twice for the evaluator's policies, once per micro-batch in
the PPO update — so it accounted for 25 of the 41 synchronizations per step.

Every channel list either shipped pipeline declares is a contiguous run, so it
is expressible as a slice: identical values, a view instead of a gather, and no
host round trip. Non-contiguous lists keep the old path.

| | rollout s/upd | update s/epoch | total s/upd | env steps/s | syncs/step |
|---|---:|---:|---:|---:|---:|
| before | 96.44 | 32.42 | 148.29 | 3,315 | 41.3 |
| after | **87.38** | **28.67** | **133.58** | **3,680** | **16.3** |
| | −9.4% | −11.6% | −9.9% | **+11.0%** | −60% |

Behaviour-preserving: same values, exactly. Five regression tests pin the slice
form, check it against advanced indexing, assert the result is a view, and
assert no shipped feature declares non-contiguous channels.

### 2. Keep per-step constants off the host-to-device path

The remaining five sites, all the same class: the frontline zone-role pattern
and the field index-level ladder are now cached per device (the idiom
`physics._lookup_tables` already used); `index_from_level` raises a Python base
with `torch.pow` instead of materializing a 0-d tensor; the refractive log scale
is a constant of the ship config and stays a Python float; `scripted_utils`
passes Python scalars to `torch.where` instead of building 0-d tensors for a
masked minimum and four turn constants.

| | rollout s/upd | update s/epoch | total s/upd | env steps/s | syncs/step |
|---|---:|---:|---:|---:|---:|
| before | 87.38 | 28.67 | 133.58 | 3,680 | 16.3 |
| after | **69.21** | 29.20 | **116.15** | **4,232** | **0.3** |
| | −20.8% | +1.8% (noise) | −13.0% | **+15.0%** | −98% |

The update phase does not move, correctly: its share of the syncs was the
feature-pipeline pair, already removed by the previous change.

One value changes: the log scale now comes from `math.log` in double rather than
a float32 device `log`, which is strictly more accurate and differs in the eighth
significant digit of one observation channel — a channel the rollout buffer
already stores in bfloat16, whose resolution is four orders of magnitude coarser.

### 3. Scan the field axis field-major

`compose_refractive_index` builds exclusive prefix/suffix products with
`cumprod` along the field axis, which sits innermost. CUDA runs an
innermost-dimension scan with one thread per row and no parallelism inside the
scan, so a length-10 axis is close to its worst case. Transposing the field axis
out of the innermost position hands the same work to the outer-dim scan kernel,
which parallelises across points instead.

Microbenchmark, 200 iterations after warmup, at the four shapes the pipeline
actually runs:

| shape | innermost cumprod | transposed + contiguous | max abs diff |
|---|---:|---:|---:|
| (1280, 8, 10) ships, training | 0.520 ms | 0.097 ms | 3.0e-8 |
| (1280, 80, 10) bullets, training | 4.089 ms | **0.097 ms** (42x) | 3.0e-8 |
| (2560, 8, 10) ships, evaluator | 0.828 ms | 0.137 ms | 6.0e-8 |
| (2560, 80, 10) bullets, evaluator | 8.277 ms | 0.269 ms | 3.0e-8 |

Two alternatives were also implemented and measured, both correct and both
slower: an unrolled Python loop over the field axis (0.15–0.51 ms) and a
Hillis-Steele doubling scan (0.15–0.58 ms).

| | rollout s/upd | update s/epoch | total s/upd | env steps/s |
|---|---:|---:|---:|---:|
| before | 69.21 | 29.20 | 116.15 | 4,232 |
| after | **67.35** | 29.26 | **114.41** | **4,296** |
| | −2.7% | — | −1.5% | +1.5% |

Small end-to-end, because with the synchronizations gone the pipeline is
CPU-dispatch bound again and freed GPU time does not convert one for one. Kept
because the microbenchmark is unambiguous, the rollout phase moves −2.7%, and
it removes ~30 ms per step of GPU work that anything later made GPU-bound would
have paid.

This is a **layout change, not an arithmetic one** — the same factors multiply
in the same order — but the scan kernel's internal association differs, which
moves results by two to four float32 ulps. A test pins the output against the
point-major formulation it replaces, including the fully-covered case where
`1 - alpha` is exactly zero, which is what the no-division form exists for.

### 4. Compile the policy entry point callers actually use

`torch.compile(module)` wraps `forward` and nothing else, and
`OptimizedModule.__getattr__` hands every other attribute straight back from the
original module. Nothing in this project calls a policy's `forward`: rollout
calls `get_action_and_value`, the PPO update calls `evaluate_actions`. Both
bypassed the wrapper and ran eager. Verified on a trainer built with the shipped
default:

```
policy type                             : OptimizedModule
get_action_and_value bound to _orig_mod : True
evaluate_actions     bound to _orig_mod : True
dynamo frames compiled                  : {}          <- zero
```

The warmup block that exists to "force torch.compile to trace both policies
under autocast" was warming the eager path. The 3.5% that `--compile none` used
to measure came from `collision_compile_mode`, which compiles the collision
solver and is wired up correctly.

In-process, at the real shapes:

| entry point | eager | compiled (`default`) | speedup | graph breaks |
|---|---:|---:|---:|---:|
| `get_action_and_value`, B=2560 | 49.69 ms | 25.20 ms | **1.97x** | 0 |
| whole primary rollout step | 159.3 ms | 112.9 ms | **1.41x** | — |
| `evaluate_actions`, per optimizer minibatch | 1442 ms | 807 ms | **1.79x** | 0 |

End to end, control and variant interleaved twice each:

| | rollout s/upd | update s/epoch | total s/upd | env steps/s |
|---|---:|---:|---:|---:|
| control, run A | 97.05 | 46.36 | 171.17 | 2,871 |
| compiled, run A | 83.96 | 46.95 | 158.52 | 3,101 |
| control, run B | 97.23 | 45.21 | 169.69 | 2,897 |
| compiled, run B | 85.93 | 47.64 | 161.68 | 3,040 |
| **control mean** | **97.14** | 45.79 | **170.43** | **2,884** |
| **compiled mean** | **84.95** | 47.30 | **160.10** | **3,071** |
| | **−12.6%** | +3.3% (noise) | **−6.1%** | **+6.5%** |

Control-to-control spread 0.9%, variant-to-variant 2.0%, so the rollout effect
is far outside the noise and the total is comfortably outside it. The update
column should not move and does not, beyond noise.

This is a **category-2 change**: inductor fuses and reassociates, so results are
not bit-identical. Measured against eager on the same inputs under bf16 autocast,
`evaluate_actions` agreed to 2.0e-4 (logprob) and 1.5e-4 (value, logits) — below
the ~4e-3 relative resolution of the bfloat16 the forward pass already runs in.

#### `evaluate_actions` is left eager on purpose

Compiling it is worth 1.79x on the update phase and is **not taken**. A compiled
backward is one fused AOT-autograd function whose saved tensors do not survive a
second traversal, and two paths traverse a micro-batch's graph more than once:
the gradient diagnostics, and the actor/critic split probe that runs on the
**histogram cadence in every ordinary run**, not only diagnostic ones.

```
RuntimeError: This backward function was compiled with non-empty donated buffers
which requires create_graph=False and retain_graph=False.
```

`torch._functorch.config.donated_buffer = False` clears that and exposes the
next one:

```
RuntimeError: one of the variables needed for gradient computation has been
modified by an inplace operation
```

Running only the *measured* micro-batches eager fixes both, and was implemented
and tested. It also makes the applied gradient depend on whether it was
measured, by ~1.9e-9 absolute, and
`test_measuring_does_not_disturb_the_gradient_that_gets_applied` exists to forbid
exactly that. It caught it, and the approach was reverted. Unlocking the 1.79x
means giving the probes a forward pass of their own — a change to the
diagnostics, not to the training path, and out of scope here.

#### Two things that fell out of making compilation real

`--compile reduce-overhead` was the shipped default and **does not run**:

```
RuntimeError: Error: accessing tensor output of CUDAGraphs that has been
overwritten by a subsequent run.
  elo_eval.py:693 in _compute_team_actions
```

CUDA-graph trees reuse static output buffers; the rollout and the evaluator both
hold policy outputs (hidden state, next-state predictions, actions) across calls.
Harmless while compilation was a no-op, fatal the moment it was not. The default
is now `default`.

And a compiled policy is **not freed by reference counting** — dynamo keeps the
traced instance alive from its own caches. `EloRoster._unload` now collects,
because `league_size` is supposed to bound device memory and would otherwise stop
doing so. A test pins the behaviour and says what to remove if a future torch
makes the collect unnecessary.

## Tried and rejected

### Larger micro-batches (tier 1, `--microbatch-tokens`)

Hypothesis: 480 forward/backward passes per epoch at 24,576 entity tokens each
is launch-bound, so fewer and larger passes should be much faster.

| microbatch_tokens | micro per shard minibatch | update s/epoch | total s/upd | env steps/s | reserved MiB |
|---|---:|---:|---:|---:|---:|
| 25,000 (shipped) | 5 | 32.42 | 148.29 | 3,315 | 3684 |
| 50,000 | 3 | **29.61** | 143.56 | 3,424 | 5006 |
| 92,160 (validator ceiling) | 2 | 30.53 | 143.93 | 3,415 | 6012 |
| none (one pass per shard minibatch) | 1 | **OOM at 7.55 GiB** | — | — | >7620 |

**Mostly wrong.** The update improves 6–9% and saturates at three micro-batches;
two is *worse* than three despite half the passes, because reserved memory
reaches 6 GB of 7.62 GB and the allocator starts costing more than the launches
save. This reproduces the A4 finding in [memory
optimization](memory-optimization.md#a4-_microbatch_tokens-divisor): finer
chunking is faster on this VRAM-constrained card.

Not taken. −3.2% end to end for 1.3 GB of allocator headroom on an 8 GB card is
a poor trade, and the measurement predates the drift controls, so the number
itself is soft. It remains available as `--microbatch-tokens 50000`.

### `torch.compile` mode, before the compile bug was found

| mode | rollout s/upd | update s/epoch | total s/upd | env steps/s | rollout-forward CPU s/upd |
|---|---:|---:|---:|---:|---:|
| `reduce-overhead` | 95.13 | — | 148.29 | 2,973 | 5.09 |
| `default` | 95.28 | 32.74 | 147.55 | 3,331 | 5.16 |
| `none` (eager) | 98.69 | 33.86 | 152.85 | 3,216 | 17.98 |

`reduce-overhead` and `default` are indistinguishable — which is the clue that
led to the compile bug: neither was doing anything to the policy. Note the
paradox this table contains, worth keeping as a diagnostic pattern: eager spends
17.98 s per update of CPU inside the rollout forward against 5.16 s "compiled",
yet total rollout time moves only 3.4 s. The saved CPU is reabsorbed waiting on
the GPU. That is the signature of a pipeline that alternates between CPU- and
GPU-bound rather than being cleanly one.

### Alternatives to the innermost-dimension scan

An unrolled Python loop over the field axis and a Hillis-Steele doubling scan
were both implemented and both correct. Both are slower than the transposed
cumprod at every shape measured (0.15–0.58 ms against 0.10–0.27 ms), and the
doubling scan also has a larger numerical spread. Rejected.

### Overlapping the env and policy CUDA streams

Already in the code, and measured to be worth nothing here. Serializing them
(`--mode sync --detail`, which runs both on the default stream) gives a primary
step of 46.16 s per update against 45.48 s with the streams on — inside the
noise. Both are issued by the same Python thread, so when the limit is CPU
dispatch a second stream buys nothing. Left alone; it costs nothing either.

### Micro-optimizing individual kernels

Counted and dropped without implementing: the 23 separate scalar reductions per
physics tick in the wrapper's source-statistics block are about 23 launches ×
384 ticks × ~10 µs ≈ 88 ms per update, under 0.1%. The belief tracker's
clone-every-channel could save ~40 launches per step, similar order. With 8,480
launches per step and no single dominant call site, there is no individual
kernel worth chasing — the wins are structural.

## Correctness

Every change was checked against this branch's own test baseline rather than
against zero: `frontline/07-map-memory` carries **40 pre-existing failures**
(the Frontline reward components and the launch-geometry token count, neither
related to this work). After all four changes the suite reports the **same 40
failures, none new and none fixed**.

Tests added:

| change | test |
|---|---|
| channel slicing | five tests in `tests/models/test_encoder.py::TestAccessorChannelSelection` — slice resolution, non-contiguous fallback, exact agreement with advanced indexing, that the result is a view, and that no shipped feature declares non-contiguous channels |
| field-major scan | `tests/env/test_field_physics.py::test_field_major_scan_matches_the_point_major_reference` — pins index and gradient against the point-major formulation, including fully-covered points where `1 - alpha` is exactly zero |
| compiled entry point | four tests in `tests/train/test_policy_io.py::TestCompilePolicy` — that the rollout entry point no longer resolves to the eager bound method, that the update entry point deliberately still does, that the policy is otherwise unchanged, and that an evicted compiled policy is reclaimed |

Numerical behaviour, stated plainly:

| change | class | what differs |
|---|---|---|
| channel slicing | behaviour-preserving | nothing; same values, a view instead of a copy |
| per-step constants | effectively equivalent | one observation channel's log scale now comes from a double-precision `math.log` rather than a float32 device `log` — more accurate, differing in the eighth significant digit of a channel stored in bfloat16 |
| field-major scan | effectively equivalent | float32 reassociation inside the scan kernel; measured maximum 6e-8 absolute on values in [0, 1], two to four ulps |
| compiled rollout entry point | effectively equivalent | inductor fusion and reassociation; measured against eager under bf16 autocast at 2.0e-4 (logprob) and 1.5e-4 (value, logits), below bfloat16's own ~4e-3 relative resolution |

None of them changes the objective, the batch construction, the opponent
curriculum, or the learning algorithm.

`bnb smoke` was **not** available as a check: it is broken on this branch for an
unrelated reason (see the defect below), identically before and after. In its
place, every measurement run in this document is itself a real training run —
the harness drives `PPOTrainer` through rollout collection, GAE, the auxiliary
labels, the full PPO update, Elo evaluation, logging and a 38 MB checkpoint
write on every update — so the pipeline was exercised end to end dozens of
times across these changes. `--compile reduce-overhead` is no longer
the default, which is a change to how a launch is *executed* and not to what it
computes — and the mode it replaces does not run at all.

## Remaining bottlenecks

Re-profiled after the changes, same 12-step rollout window:

| | before | after | change |
|---|---:|---:|---:|
| self CUDA time | 148 ms/step | **118 ms/step** | −20.2% |
| self CPU time | 292 ms/step | **203 ms/step** | −30.6% |
| `cudaStreamSynchronize` | 699 ms, 20% of CPU, #1 item | **absent from the top 20** | gone |
| `cudaLaunchKernel` | 673 ms, 19.2% of CPU | 585 ms, **24.1%, now #1** | — |
| launches per rollout step | 8,498 | 8,480 | — |
| `aten::cumprod` | 357.7 ms, 20.2% of GPU | **absent** | gone |

The GPU-time drop is exactly the cumprod share; the CPU-time drop is the
synchronization stall. **The pipeline is now cleanly kernel-launch bound.** The
top CPU items after `cudaLaunchKernel` are `mul` 118 ms, `empty_strided` 102,
`copy_` 96, `empty` 81, `where` 81, `cat` 66, `sub` 62, `add` 55 — dispatch
overhead spread thin over small tensors, with no single dominant call site.

The update phase is the same story: profiled over four optimizer minibatches,
**33,265 launches per optimizer minibatch** (2,218 per forward/backward pass) for
665 ms of GPU work against 1,509 ms of wall — **44% GPU occupancy**. Its top GPU
consumers are `mm` 20.5%, `mul` 20.1%, `copy_` 12.8%, `_fused_rms_norm_backward`
8.3%, SDPA backward 5.9%. The `mul` count is dominated by the Hillis-Steele
parallel scan in the RG-LRU, which is work-inefficient by construction: seven
rounds of full-size elementwise ops for a 128-step sequence.

Ranked, with the shares from the final profile:

1. **Elo evaluation, ~36% of wall clock.** A 2560-environment world — twice the
   training width — stepped every rollout step, with two to six policy forwards
   on top, producing no gradient. Nothing about it is *inefficient*; it is simply
   a lot of work, and it grows once the floating-checkpoint slots stop idling.
   No behaviour-preserving win was found. See the tradeoffs below.
2. **PPO update, ~38% of wall clock at two epochs.** 44% GPU-occupied. The
   1.79x from compiling `evaluate_actions` is measured and waiting on the
   gradient probes.
3. **Primary rollout step, ~23%.** Environment physics, observation
   construction, belief tracking, the scripted agent.

## Not implemented, worth doing

- **Compile `evaluate_actions`** — 1.79x on the update phase, measured. Needs the
  actor/critic split probe and the gradient diagnostics to run their own forward
  pass rather than re-traversing the training graph. Roughly a 17% end-to-end
  win at two epochs, and the single largest remaining item.
- **A work-efficient RG-LRU scan.** `_parallel_scan` is Hillis-Steele: O(T log T)
  work for an O(T) recurrence, seven full-size rounds at T=128, and it dominates
  the `mul` count in the update. A Blelloch scan or a chunked two-level scan
  would cut the elementwise traffic. Worth measuring before building.
- **Build both team observation views in one batched pass.**
  `perceived_observation_from_state` calls `observation_from_state` twice, once
  per team, each assembling ~27 channels; only the visibility mask and the
  pending-action masking differ. Folding the team axis into the batch would
  roughly halve the op count of a path that costs 14.1 s per update across the
  training and evaluator call sites.
- **Cut the belief tracker's clone-everything.** `compose` clones all 27
  observation channels and writes 17 of them. Small individually, but it runs
  four times per rollout step.
- **Re-probe `--vram` on this configuration.** The 8 GB preset row was measured
  in August 2026 on a pre-Frontline, field-free `rl` profile and the basis note
  already says the wider belief observation was never re-probed. The profile now
  peaks at 2.5 GB allocated and 3.5 GB reserved of 7.62 GB — a lot of unused
  headroom that a fresh probe could spend.

## A defect found along the way, not fixed

The launch geometry counts entity tokens as `num_ships + num_fields`
([`config/resolve.py`](../../src/boost_and_broadside/config/resolve.py)), which
is 18 for the `rl` profile. The rollout buffer's real token width is 24: eight
ships, ten fields, **five zones and one boundary**. Every derived quantity —
`logical_batch_tokens`, the valid shard widths, the `--vram` preset ceilings, the
`microbatch_tokens` validator bound — is therefore computed on two thirds of the
tokens that exist. It is why `--microbatch-tokens 125000` is rejected as
exceeding a "minibatch size" of 92,160 when the minibatch really holds 122,880
tokens, and it is the likely cause of the two pre-existing
`test_print_config_*` failures on this branch.

It also breaks `bnb smoke` outright — every case dies in
`smoke._smoke_resolved_profile` with

```
ValueError: logical_batch_tokens must be divisible by the fixed-environment rollout size
```

which is why the smoke matrix could not be used to validate this work. It fails
identically at `d7ebb1c`, before any of these changes, and the same failure
appears as the four `tests/artifacts/test_mode_artifacts.py` errors in the
branch's pre-existing baseline. Out of scope for a throughput pass, but it
should be reconciled before the VRAM presets are trusted again.

## Tradeoffs worth considering separately

These change training behaviour or the machine, so none of them is applied here.

### Raise the platform profile (machine configuration)

The CPU is held near 1 GHz by `quiet` + `powersave` + EPP `power`, against a
4.7 GHz ceiling, with RAPL PL1 at 35 W. This pipeline is CPU-dispatch bound, so
that clamp lands directly on the part that dominates wall clock. `balanced` or
`performance` is available. It is not free — fan noise, battery, and the notes on
this machine say the profile also limits the GPU — so it is a decision to make
deliberately rather than a setting to flip. The size of the win has not been
measured, because measuring it means changing the machine.

### Elo evaluation budget (changes the rating estimator)

Evaluation is ~36% of training wall clock and rising. Two knobs reduce it
proportionally, and both reduce rated games proportionally too:

| knob | current | effect of halving |
|---|---:|---|
| `elo_eval.envs_per_matchup` | 512 | halves the evaluator's environment count and its whole cost |
| `elo_eval.step_interval` | 1 | halves how often the evaluator advances |

Either one buys roughly 18% of training wall clock for half the games per
update. Whether that is a good trade depends on how much precision the live
rating needs, and the live rating steers opponent selection, milestone
placement, the behaviour-cloning gate and the trust region — so this is a
training-behaviour change, not an optimization. Worth a deliberate experiment:
the `elo_diag/*` series already instrument how well-identified the rating is,
and `elo_diag/movement_z` would show directly whether halved games make the
filter noisier than it can afford.

One free-ish observation within it: two of the five slots play unscored
random-vs-random until the first checkpoint milestone. That is 20% of the
evaluator's cost doing nothing, early in every run. It is transient, which is why
it was not chased, but a run that spends a long time below the first milestone
pays it the whole time.

### `--microbatch-tokens 50000` (tier 1)

Measured at −3.2% end to end for +1.3 GB of reserved memory, in the drift-prone
window before the controls were in place. Available, marginal, and it eats
headroom on an 8 GB card that the allocator was shown to want.

## Reproducing any of this

```bash
# Phase breakdown, correct attribution, env and policy streams separated
uv run --no-sync python benchmarks/rl_pipeline_profile.py \
    --updates 2 --warmup 1 --mode sync --detail --checkpoint-dir /tmp/ckpt

# End-to-end throughput, minimal perturbation
uv run --no-sync python benchmarks/rl_pipeline_profile.py \
    --updates 2 --warmup 1 --mode wall --checkpoint-dir /tmp/ckpt

# A/B control: the same launch with policy compilation disabled
uv run --no-sync python benchmarks/rl_pipeline_profile.py \
    --updates 2 --warmup 1 --mode wall --compile-entry none --checkpoint-dir /tmp/ckpt

# Kernel-level: launches, CPU/GPU split, top operators
uv run --no-sync python benchmarks/rl_kernel_profile.py --warmup-steps 24 --profile-steps 12
uv run --no-sync python benchmarks/rl_kernel_profile.py --phase update --update-minibatches 4
```

Interleave control and variant, at least twice each, and read the
control-to-control spread as the noise floor. On this machine a comparison
between two processes run an hour apart is not evidence.

To find host synchronizations, wrap a few real rollout steps in
`torch.cuda.set_sync_debug_mode("warn")` and attribute the warnings by Python
stack. Five lines of code, and it found 41 per step that nothing else had
surfaced in the two years this pipeline has existed.

## Raw numbers

Every run below is one warmup update plus two measured updates of the `rl`
profile at 491,520 environment steps per update, `--mode wall` unless noted.
Epochs per update were `[2, 1]` in every run, so the `update s/epoch` column is
the comparable one.

| run | what | rollout s/upd | update s/epoch | total s/upd | env steps/s | alloc MiB | reserved MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| M1 | baseline, 3 updates | 95.13 | — | 165.33 | 2,973 | 2527 | 3684 |
| M2 | baseline, `--mode sync --detail` | 99.33 | 34.90 | 154.92 | 3,173 | 2517 | 2620 |
| base2 | baseline | 96.44 | 32.42 | 148.29 | 3,315 | 2527 | 3684 |
| mb50k | `--microbatch-tokens 50000` | 96.22 | 29.61 | 143.56 | 3,424 | 3655 | 5006 |
| mb92k | `--microbatch-tokens 92160` | 95.22 | 30.53 | 143.93 | 3,415 | 4777 | 6012 |
| noelo | Elo evaluator ablation | 45.48 | 34.14 | 99.85 | 4,923 | 2504 | 3646 |
| compile_default | `--compile default` | 95.28 | 32.74 | 147.55 | 3,331 | 2527 | 3684 |
| compile_none | `--compile none` | 98.69 | 33.86 | 152.85 | 3,216 | 2526 | 3690 |
| o1a | + channel slicing | 87.38 | 28.67 | 133.58 | 3,680 | 2527 | 3684 |
| o1b | + per-step constants | 69.21 | 29.20 | 116.15 | 4,232 | 2527 | 3684 |
| o2 | + field-major scan | 67.35 | 29.26 | 114.41 | 4,296 | 2527 | 3684 |
| o5 control A | no policy compile | 97.05 | 46.36 | 171.17 | 2,871 | 2527 | 3684 |
| o5 compiled A | + compiled rollout | 83.96 | 46.95 | 158.52 | 3,101 | 2526 | 3530 |
| o5 control B | no policy compile | 97.23 | 45.21 | 169.69 | 2,897 | 2527 | 3684 |
| o5 compiled B | + compiled rollout | 85.93 | 47.64 | 161.68 | 3,040 | 2526 | 3530 |
| o6 old B | `d7ebb1c`, clean | 131.07 | 45.43 | 203.69 | 2,413 | 2527 | 3684 |
| o6 new A | HEAD | 83.41 | 46.63 | 157.23 | 3,126 | 2526 | 3530 |
| o6 new B | HEAD | 85.93 | 47.46 | 161.65 | 3,041 | 2526 | 3530 |

`o6 old A` is omitted: a leftover process from an earlier launch of the same
script was sharing the GPU with it, which is why it took 16 minutes against the
others' 11 and read 243.18 s/update. Caught by the run duration, not the number.

Runs M1 through o2 predate the CPU-clamp drift controls and were taken in one
50-minute window; runs o5 and o6 are interleaved. The machine is measurably
slower in the o5/o6 window than in the M1–o2 window — the same code path reads
97 s/update in o5 against 67 s in o2 — which is why the cumulative claim is
taken from o6 alone and not by chaining the earlier deltas.
