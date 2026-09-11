# Where RL training time goes, and what moved it

A profiling pass over the whole `bnb train --profile rl` pipeline: what it spends
wall clock on, what was changed, what each change was worth, and what was tried
and rejected. Companion to [memory optimization](memory-optimization.md), which
covers the same pipeline's memory rather than its speed.

The short version: the pipeline was **CPU-dispatch bound with a synchronization
stall**, not compute bound. It issued 8,498 CUDA kernels per rollout step against
118–148 ms of actual GPU work, drained the launch queue 41 times per step on
small tensors built from host data, and — although every launch passed
`--compile` — had never compiled a single frame of the policy.

## Result

Interleaved against the pre-optimization commit (`d7ebb1c`), same machine, same
launch, one warmup and two measured updates per run, twice each way:

| | before | after | change |
|---|---:|---:|---:|
| **sustained throughput** | **2,371 env steps/s** | **3,810 env steps/s** | **+60.6%** |
| seconds per update | 207.32 | 129.03 | −37.8% |
| rollout collection | 132.99 s/update | 86.34 s/update | −35.1% |
| PPO update | 46.62 s/epoch | 25.83 s/epoch | −44.6% |
| peak allocated | 2527 MiB | 2143 MiB | −15.2% |

Run-to-run spread is 3.2% on the "before" arm and 1.5% on the "after" arm.

The five changes, and what each was worth measured on its own:

| # | change | class | measured |
|---|---|---|---|
| 1 | read observation channels by slice, not list index | behaviour-preserving | +11.0% end to end; 41.3 → 16.3 host syncs per rollout step |
| 2 | keep per-step constants off the host-to-device path | effectively equivalent | +15.0% end to end; 16.3 → **0.3** host syncs per rollout step |
| 3 | scan the field axis field-major | effectively equivalent | 42x on the dominant shape; +1.5% end to end |
| 4a | compile the policy's rollout entry point | effectively equivalent | 1.97x on the forward; **+6.5%** end to end, interleaved |
| 4b | compile the PPO update's forward too | effectively equivalent | **1.81x on the update phase**; **+23.1%** end to end, interleaved |
| 5 | size a launch from every entity token | bug fix | no throughput change; unbreaks `bnb smoke` and 11 tests |

Nothing here changes the objective, the batch construction, the opponent
curriculum, or the learning algorithm.

Four things were tried and rejected on evidence: larger micro-batches, compiling
the environment (which is *wrong*, not merely unhelpful — see below), a
work-efficient RG-LRU scan, and reusable observation buffers for the evaluator.

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

`--timing sync --detail`, two measured updates averaging 1.5 epochs, 154.92 s per
update. Attribution is correct here; the syncs cost about 3% over `wall` timing,
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
waiting — which is also why the CPU never ran ahead and why `--timing sync` was
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

#### Then `evaluate_actions` too

The update's entry point was left eager in the first pass because two paths
re-traverse a micro-batch's backward graph and a compiled backward is one fused
AOT-autograd function whose saved tensors do not survive the second:

```
RuntimeError: This backward function was compiled with non-empty donated buffers
which requires create_graph=False and retain_graph=False.
```

and, once `torch._functorch.config.donated_buffer = False` clears that,

```
RuntimeError: one of the variables needed for gradient computation has been
modified by an inplace operation
```

The two callers need different answers, which is why the first attempt — route
*measured micro-batches* to the eager implementation — was wrong and was caught
by `test_measuring_does_not_disturb_the_gradient_that_gets_applied`. It made the
applied gradient depend on whether a micro-batch had been measured.

- The **actor/critic split probe** differentiates two loss terms, so two
  traversals, and it runs on the histogram cadence in every ordinary run. It now
  evaluates the micro-batch on a graph of its own and takes both gradients off
  that. One extra forward and two extra backwards, on one micro-batch per
  histogram interval, against a probe whose docstring already budgets two extra
  backwards. The forward and backward that actually move the policy are then
  identical to an unmeasured micro-batch's — the guarantee comes out stronger
  than it went in, not weaker.
- The **gradient diagnostics** differentiate once per decomposed term, up to
  seventeen traversals at `reward_full`. Giving each its own forward is not
  affordable, so a diagnosed run evaluates eagerly throughout.
  `_update_evaluate_actions` reads that off the diagnostic level at call time,
  which makes the choice a property of the *run* and never of the micro-batch.
  That is exactly what the bit-identity test asserts.

Interleaved twice against a control compiling only the rollout entry point:

| | rollout s/upd | update s/epoch | total s/upd | env steps/s |
|---|---:|---:|---:|---:|
| rollout only, A | 83.75 | 46.33 | 157.75 | 3,116 |
| both, A | 86.48 | 24.67 | 127.59 | 3,852 |
| rollout only, B | 86.74 | 47.11 | 161.69 | 3,040 |
| both, B | 87.15 | 26.83 | 131.84 | 3,728 |
| **control mean** | 85.24 | **46.72** | **159.72** | 3,078 |
| **variant mean** | 86.82 | **25.75** | **129.72** | **3,790** |
| | +1.9% (noise) | **−44.9%** | **−18.8%** | **+23.1%** |

Control spread 2.4%, variant 3.2%. The update phase is **1.81x** faster,
matching the 1.79x measured in isolation, and the rollout column correctly does
not move.

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

### 5. Size a launch from every entity token

How wide the observation's token axis will be was written out three times, and
two copies omitted Frontline's five zone tokens and its boundary token:
`launch_geometry` counted all six and was right; `validate_resolved_config`
counted ships plus fields, and so capped `microbatch_tokens` at 92,160 when the
minibatch it guards holds 122,880; and `smoke._smoke_resolved_profile` sized its
one-environment batch the same way, leaving it too small to hold a single
environment — which is why **`bnb smoke` has never run on this branch**.

`config.core.entity_token_count` is the one derivation now, with `EnvConfig`
exposing it as a property and all three sites reading it. Adding a token kind
means adding a term in one place.

`tests/config/test_entity_tokens.py` pins the prediction against the token axis
a real `TensorEnv` builds — across field counts, ship counts and both game modes
— so a kind added to the environment without a matching term fails there rather
than silently resizing the batch.

The resolved `rl` launch is unchanged (1280 environments, 3 rollouts per update,
25,000 microbatch tokens), so this is a bookkeeping correction and not a move of
the experiment. It clears 11 of the branch's 40 pre-existing test failures: the
four `test_mode_artifacts` errors and seven `test_fields_evaluation` errors, all
of which were the smoke profile failing to resolve.

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

### Compiling the environment — rejected on correctness

The environment is the largest remaining dispatch cost and none of it was
compiled, so this looked like the obvious next win. A candidate sweep, one
function at a time, in-process with a drift check (the two eager arms differed
by 6.3%, so the reference is the faster of them):

| compiled | ms / primary step | vs reference |
|---|---:|---:|
| eager baseline | 108.8 | — |
| `observation_from_state` | 101.5 | −6.6% |
| `perceived_observation_from_state` | 102.9 | −5.4% |
| `update_ships` | 105.2 | −3.3% |
| `advance_bullets` | 105.4 | −3.1% |
| `evaluate_fields` | 113.4 | +4.3% |
| `team_visibility_from_state` | 113.2 | +4.0% |
| `_line_of_sight_clear` | 117.7 | +8.2% |
| `apply_frontline_tick` | 117.9 | +8.4% |
| `resolve_collisions` | 120.3 | +10.6% |
| **all three winners together** | **86.7** | **−20.3%** |

**That −20.3% is measuring broken code.** `update_ships`, `advance_bullets` and
`perceived_observation_from_state` all communicate by writing their results back
onto objects the caller owns — the state, and the reusable observation buffers —
and dynamo does not replay those writes. Compiled, they run and produce nothing:

- a tick commanding all 64 ships to fire produced **64 shots eager and 0
  compiled**
- the buffered observation came back with positions off by **16,135 pixels** on a
  16,384-pixel torus

A world with no shots has no bullet physics to simulate, which is exactly why the
combined arm looked so good. Reverted in full.
`tests/env/test_compiled_tick.py` now pins that a compiled launch simulates the
same world and builds the same observation as an eager one — using a
shoot-everything probe, because the shot gate is discrete and a dropped write
shows up as *no shots at all* rather than as numeric drift.

This is a structural block rather than an incidental one: making the environment
compilable means making these stages functional, which is a refactor of
`physics.py` and `observation.py`, not a flag.

Note also the stages that got *slower* compiled. A graph boundary inside an
otherwise eager region buys guard checks without buying fusion, and
`resolve_collisions` already has a compiled kernel inside it.

Two measurement traps turned up on the way, both in the harness rather than the
code: the two environments draw from the same global CUDA RNG, so the arms have
to save and restore it or bullet spread decorrelates them for an unrelated
reason; and the rollout buffer holds `num_steps` transitions, so it has to be
recycled between arms or it overflows mid-sweep. The first run of this sweep hit
both and reported a spurious 2x.

### A work-efficient RG-LRU scan — no longer worth it

`_parallel_scan` is Hillis-Steele: O(T log T) work for an O(T) recurrence, seven
full-size elementwise rounds at T=128, and it dominated `aten::mul` at 20.1% of
update GPU time. A Blelloch or chunked scan would cut that memory traffic, and
it would be a pure reassociation — same recurrence, same weights, same objective.

Re-profiling the update with `evaluate_actions` compiled says not to bother:

| | eager update | compiled update |
|---|---:|---:|
| wall per optimizer minibatch | 1509.3 ms | **1036.6 ms** |
| self CUDA per minibatch | 665 ms | **445 ms** |
| profiler events per minibatch | 234,430 | **86,149** |

`aten::mul` is gone from the top of the compiled profile — inductor fused those
rounds. `aten::mm` now leads at 31.25%, real matmul work, then SDPA backward at
9.1% and two fused triton gelu kernels at 7.5% and 4.7%. A hand-written scan
would be attacking a target compilation has already largely absorbed.

### Reusable observation buffers for the evaluator — no measurable effect

The evaluator allocates a fresh observation for all 2560 of its environments on
every rollout step; the training wrapper reuses buffers. Giving the evaluator the
same treatment, interleaved:

| | total s/upd | env steps/s | Elo phase s/upd | peak allocated |
|---|---:|---:|---:|---:|
| fresh allocation each step | 129.72 | 3,790 | 44.43 | 2143 MiB |
| reusable buffers | 129.20 | 3,805 | 44.14 | 2145 MiB |
| | −0.4% | +0.4% | −0.7% | +2 MiB |

Variant spread is 2.8%, so all of it is inside the noise, and the buffers are
persistent so peak memory is marginally worse. It also adds an invariant: every
`reset_envs` has to be paired with a field-state refresh, or those environments
keep observing the map they used to be in — silently, because every other
channel stays correct. No measurable gain for a new way to be quietly wrong.
Reverted; the staleness guard test was kept, since the training wrapper relies
on the same pairing.

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
(`--timing sync --detail`, which runs both on the default stream) gives a primary
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

### Merge the evaluator's environment with the training one

The evaluator is not a different kind of work. It uses the same `ship_config`,
the same `env_config`, the same ship and token counts, and it steps once per
training rollout step. It is a second `TensorEnv` of 2560 environments running
beside the training one of 1280, doing structurally identical work — and it is
strictly serial with it, on the same stream and the same Python thread.

Nothing about it has to be separate at the tensor level. One environment of 3840
would issue one set of kernels where there are now two, and the live policy's
evaluation forward (4×512 environments) is the same weights as the training pass
(2×1280) and could join one 4608-environment call. What cannot merge is the
*other* policies — the running average, the anchor ladder, the floating
checkpoint all have different weights.

How much that is worth is bounded by how much of the evaluator is dispatch
rather than GPU work, which is measurable:

| half of a rollout step | wall | self CUDA | GPU busy | events |
|---|---:|---:|---:|---:|
| Elo evaluation | 113.9 ms | 51.1 ms | **45%** | 20,034 |
| primary rollout | 245.9 ms | 95.0 ms | **39%** | 39,912 |

So the evaluator is ~55% dispatch. Taking the phase split of its 53.05 s/update:
`env.step` (25%) and the observation build (16%) fuse completely, and the
live-policy forward (about two thirds of the 21.28 s policy line) fuses with the
training pass. At 55% dispatch that is roughly **20 s of 53, about 13% end to
end** — real, and less than it looks, because the GPU work does not go away and
a merged environment would compute rewards for 3840 environments instead of 1280
unless it masked.

Not attempted here because it is a genuine refactor across two subsystems that
both communicate by mutating shared state, and this session established twice
over how quietly that goes wrong.

### Make the environment compilable

The blocked −5 to −7% from compiling the observation builder, and whatever the
physics would give, are available to a version of `physics.py` and
`observation.py` whose stages return their results instead of writing them onto
the caller's objects. That is the structural fix behind the rejection above.

### Other items

- **A work-efficient RG-LRU scan.** Measured as no longer worth it once the
  update is compiled; see above.
- **Re-probe `--vram` on this configuration.** The 8 GB preset row was measured
  in August 2026 on a pre-Frontline, field-free `rl` profile, and its own basis
  note says the wider belief observation was never re-probed. The profile now
  peaks at 2.1 GB allocated and 3.1 GB reserved of 7.62 GB — a lot of unused
  headroom a fresh probe could spend. The rows are also stale in a way that
  already shows: `--vram 16` proposes a width of 864 environments, which does
  not divide the Frontline profile's logical batch, and that is what the two
  remaining `test_print_config_*` failures are.
- **Cut the belief tracker's clone-everything.** `compose` clones all 27
  observation channels and writes 17 of them, four times per rollout step.
  Small, but it is on the hottest path left.

## What `bnb smoke` says now

It still does not pass, and it is worth being precise about why, because it was
the one whole-system check unavailable for this work.

Before the token-count fix above, every case died resolving its profile —
`logical_batch_tokens must be divisible by the fixed-environment rollout size` —
so the matrix had not run on this branch at all. That is fixed, and eleven of the
branch's forty pre-existing test failures went with it. What remains is
unrelated to throughput: the `rl` profile is a Frontline profile on a
16,384-pixel world, and several fixtures build policies against a 1024-pixel
one, so they stop at the physics-drift check:

```
ConfigDriftError: checkpoint ... trained under different physics constants than
the current run (world_size: checkpoint=(16384.0, 16384.0) runtime=(1024.0, 1024.0))
```

In place of the smoke matrix, every measurement run in this document is itself a
real training run: the harness drives `PPOTrainer` through rollout collection,
GAE, the auxiliary labels, the full PPO update, Elo evaluation, logging and a
38 MB checkpoint write on every update. The pipeline was exercised end to end
several dozen times across these changes, and `bnb train --profile rl` was run
directly through the CLI as a final check.

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
    --updates 2 --warmup 1 --timing sync --detail --checkpoint-dir /tmp/ckpt

# End-to-end throughput, minimal perturbation
uv run --no-sync python benchmarks/rl_pipeline_profile.py \
    --updates 2 --warmup 1 --timing wall --checkpoint-dir /tmp/ckpt

# A/B control: the same launch with policy compilation disabled
uv run --no-sync python benchmarks/rl_pipeline_profile.py \
    --updates 2 --warmup 1 --timing wall --compile-entry none --checkpoint-dir /tmp/ckpt

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

Every run is one warmup update plus two measured updates of the `rl` profile at
491,520 environment steps per update, `--timing wall`, epochs `[2, 1]`. The
`update s/epoch` column is the comparable one.

### First pass

| run | what | rollout s/upd | update s/epoch | total s/upd | env steps/s |
|---|---|---:|---:|---:|---:|
| M1 | baseline, 3 updates | 95.13 | — | 165.33 | 2,973 |
| M2 | baseline, `--timing sync --detail` | 99.33 | 34.90 | 154.92 | 3,173 |
| base2 | baseline | 96.44 | 32.42 | 148.29 | 3,315 |
| mb50k | `--microbatch-tokens 50000` | 96.22 | 29.61 | 143.56 | 3,424 |
| mb92k | `--microbatch-tokens 92160` | 95.22 | 30.53 | 143.93 | 3,415 |
| noelo | Elo evaluator ablation | 45.48 | 34.14 | 99.85 | 4,923 |
| compile_default | `--compile default` | 95.28 | 32.74 | 147.55 | 3,331 |
| compile_none | `--compile none` | 98.69 | 33.86 | 152.85 | 3,216 |
| o1a | + channel slicing | 87.38 | 28.67 | 133.58 | 3,680 |
| o1b | + per-step constants | 69.21 | 29.20 | 116.15 | 4,232 |
| o2 | + field-major scan | 67.35 | 29.26 | 114.41 | 4,296 |
| o5 control A / B | no policy compile | 97.05 / 97.23 | 46.36 / 45.21 | 171.17 / 169.69 | 2,871 / 2,897 |
| o5 compiled A / B | + compiled rollout | 83.96 / 85.93 | 46.95 / 47.64 | 158.52 / 161.68 | 3,101 / 3,040 |

### Second pass

| run | what | rollout s/upd | update s/epoch | total s/upd | env steps/s |
|---|---|---:|---:|---:|---:|
| e2 rollout-only A / B | compile the rollout entry point only | 83.75 / 86.74 | 46.33 / 47.11 | 157.75 / 161.69 | 3,116 / 3,040 |
| e2 both A / B | + compiled update | 86.48 / 87.15 | 24.67 / 26.83 | 127.59 / 131.84 | 3,852 / 3,728 |
| e5 buffered A / B | + evaluator observation buffers (reverted) | 83.65 / 87.90 | 26.34 / 25.84 | 127.36 / 131.03 | 3,859 / 3,751 |
| **e6 old A / B** | **`d7ebb1c`** | **135.27 / 130.70** | **47.25 / 46.00** | **210.55 / 204.09** | **2,334 / 2,408** |
| **e6 new A / B** | **HEAD** | **86.65 / 86.03** | **25.14 / 26.52** | **128.03 / 130.04** | **3,839 / 3,780** |

### Component measurements

| what | eager | changed | note |
|---|---:|---:|---|
| `get_action_and_value`, B=2560 | 49.69 ms | 25.20 ms | in-process |
| whole primary rollout step | 159.3 ms | 112.9 ms | in-process |
| `evaluate_actions`, per optimizer minibatch | 1442 ms | 807 ms | in-process, eager arm repeated (1.1% drift) |
| `cumprod` (1280, 80, 10) | 4.089 ms | 0.097 ms | microbenchmark, 200 iterations |
| update phase, per optimizer minibatch | 1509.3 ms | 1036.6 ms | kernel profile, 4 minibatches |
| update self CUDA, per optimizer minibatch | 665 ms | 445 ms | " |
| update profiler events, per optimizer minibatch | 234,430 | 86,149 | " |

### Discarded runs

`o6 old A` from the first pass: a leftover process from an earlier launch of the
same script was sharing the GPU with it, which is why it took 16 minutes against
the others' 11 and read 243.18 s/update. Caught by the run duration, not by the
number.

The first environment-compile sweep: the rollout buffer overflowed part way
through, so the later arms measured a crashed pipeline and `advance_bullets`
reported a spurious 54.1 ms against a 111.6 ms baseline.

Runs M1 through o2 predate the CPU-clamp drift controls and were taken in one
50-minute window. Everything from o5 onward is interleaved. The machine is
measurably slower in the later windows — the same code path reads 97 s/update in
o5 against 67 s in o2 — which is why the cumulative claim comes from e6 alone
and not from chaining the earlier deltas.
