# Framework spike: PyTorch versus JAX on this project's workload

A small benchmark to help choose the framework for the lean rewrite. It implements the same
three workloads in PyTorch and in JAX, checks that the two compute the same numbers, and
measures the things that hurt in PyTorch: compile time, speed of many small operations, one
model at many team sizes, Python in the rollout loop, and batch-size-1 latency for play.

**Status.** The CPU part has been run (in a cloud container, not the owner's laptop); its records
are in `results/`. The GPU part has **not** been run: this machine has no GPU, and the `cuda`
code paths of both scripts are untested. Nothing in this directory states or estimates a GPU number.

## Files

| file | what |
|---|---|
| `spec.py` | Shared specification: every constant, the initial weights, initial game state, stored rollout (all numpy), plus the timing helpers. Both scripts import it, so they run the same maths on the same numbers. |
| `torch_bench.py` | PyTorch implementation and benchmark. |
| `jax_bench.py` | JAX implementation and benchmark. |
| `parity.py` | Checks that the two implementations agree (see "Parity"). |
| `summarize.py` | Turns JSON records into the markdown tables. |
| `results/` | JSON records measured here, and `results.md`, the tables made from them. |

## The three workloads

**A. Policy forward.** Tokens of width 128: one global token plus one per ship slot (both teams).
Per-ship input: 20 features -> Linear -> RMSNorm -> GELU -> Linear; the global token the same
from 8 features. Two blocks. Each block is one temporal sublayer, then two spatial sublayers.

- Temporal: `x + RG-LRU(RMSNorm(x))`, an independent recurrence per token (Griffin, arXiv
  2402.19427): `r = sigmoid(W_a x)`, `i = sigmoid(W_x x)`, `a = sigmoid(Lambda)`,
  `a_t = a^(8 r)`, `h_t = a_t h_{t-1} + sqrt(1 - a_t^2) (i * x)`; Lambda initialised so `a^8` is
  uniform in [0.9, 0.999]. Single-step form with carried state for the rollout; scan-over-time
  form for the learner (a Python loop over time in PyTorch, `lax.scan` in JAX).
- Spatial: pre-RMSNorm self-attention over the tokens of one environment (2 heads, a boolean key
  mask so dead or absent ships are not attended to, rotary rotation of queries and keys from the
  token's 2D position in metres, 6 frequency bands per axis, wavelengths 200 m to 16 km), then a
  gated MLP (RMSNorm, Linear to 256, `GELU(first half) * second half`, Linear back).
- Heads on the ship tokens: 42-way policy logits and 77-bin value logits.

PyTorch uses `F.scaled_dot_product_attention` with `attn_mask`; JAX writes attention out
(matmul, mask, softmax, matmul) with no library beyond `jax` and `optax`. 580,983 parameters.

**B. Toy game step.** `E` environments, `N` ship slots (two teams of `N/2`), 8 bullet slots per
ship. Heading, speed, turn and thrust actions, linear drag, position integration in a circular
arena of radius 4000 m with a soft boundary; firing with a cooldown puts a bullet (ship velocity
plus 900 m/s, lifetime 11 ticks) in the first free slot; a bullet within 60 m of a living enemy
ship hits it; damage, death, respawn at the spawn point after 30 ticks. Fully vectorised with
masks; the hit test is a dense `[E, 8N, N]` comparison. Action = turn (3) x thrust (7) x fire (2)
= 42. **This is a stand-in for the operation structure of the real game, not a port of it**: it
has the same kind of work (many small elementwise operations, masks, a bullet-ship distance
matrix) but none of the real game's rules, and far fewer operations (see "Operation counts").

A **rollout tick** (A+B) is: observe (game state -> features), two policy forwards (one per
team's view, each over all tokens, each with its own recurrent state), sample one action per
ship from its own team's forward, game step.

**C. PPO update.** One minibatch gradient step on a stored rollout `[E, L = 128, N]`: learner
forward, clipped surrogate (0.2), cross-entropy of the 77-bin value logits against a two-hot
target, entropy bonus, global-norm clip (0.5), Adam (optax in JAX). The stored rollout is
synthetic (right shapes and ranges); the benchmark measures the cost of an update, not learning.

Mixed precision (GPU only by default): PyTorch uses `torch.autocast(dtype=torch.bfloat16)`. JAX
has no autocast (I found none in the installed `jax` 0.12.1 or `optax` 0.2.9); the plain
equivalent, which `jax_bench.py` implements, is to keep float32 master weights and cast weights
and inputs to `bfloat16` inside the differentiated function, casting logits back to float32 for
the loss. In both, the RG-LRU recurrence is kept in float32 (`1 - a_t^2` with `a_t` near 1 has no
precision in bf16). The two bf16 variants are not the same computation operation for operation
(autocast chooses per operation, the JAX version casts everything but the recurrence), so
compare them for speed only. Both ran on the CPU here on a tiny batch as a smoke test; neither
has run on a GPU.

## What is measured

All suites run each measurement in a fresh child process. Records go to a JSON file; build the
tables with `summarize.py`.

1. `compile` - first-call time in a fresh process for A (single step), B, the rollout tick, and
   C. Cold = empty on-disk cache. Warm = a second fresh process reusing the cache the cold one
   wrote: for PyTorch the inductor cache directory (`TORCHINDUCTOR_CACHE_DIR`; FX-graph cache and
   AOTAutograd cache are on by default in 2.15), for JAX the persistent compilation cache
   (`jax_compilation_cache_dir`, with its minimum-compile-time and minimum-size thresholds set to
   keep everything; off by default in JAX). PyTorch is measured as `torch.compile(fn)` with all
   defaults (`backend="inductor"`, `mode="default"`, `dynamic=None`, `fullgraph=False`) and as
   plain eager; JAX as `jax.jit(fn)`.
2. `steady` - milliseconds per call after warm-up: median and quartiles over 7 repeats, each a
   loop of about one second (at least 3 calls). GPU timers stop after `torch.cuda.synchronize()` /
   `jax.block_until_ready`.
3. `sizes` - compile the rollout tick at N = 10, then meet N = 20, 40, 100 and N = 10 again:
   time to first result, steady time, and how many new graphs were compiled. PyTorch: default
   `torch.compile`; `dynamic=True`; `torch._dynamo.mark_dynamic` on the ship axis of every input;
   padding every size to 100 slots; and eager (exact and padded) for reference. JAX: re-jit per
   size; padding to 100.
4. `loop` - a rollout of T = 128 ticks. JAX: one jitted `lax.scan`, and a Python loop over the
   jitted tick. PyTorch: a Python loop over the compiled tick, and over the eager tick. All keep
   the sampled actions and log-probs.
5. `latency` - E = 1: one game tick plus two policy forwards, each tick timed, 3,000 ticks after
   300 warm-up ticks, at N = 10 and N = 40, with three thread settings (see below).
6. `ops` - PyTorch operations dispatched per uncompiled rollout tick (a `TorchDispatchMode`
   counter, the approach of `../simplification/regions.py`), and for context the number of
   primitives in JAX's traced program.

Only the `compile` and `sizes` suites control the on-disk caches. First-call times recorded by
the other suites depend on what an earlier run left in the cache; do not read them as compile times.

### Thread settings (latency)

- `default`: whatever the framework does (PyTorch reported 4 intra-op threads here).
- `1`: PyTorch `torch.set_num_threads(1)`. JAX has no documented thread setting that I could
  find; the script sets `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`,
  which is folklore from JAX issue threads, **not** something I could confirm in the docs. XLA
  accepted the flags without complaint, but the latency did not change, so I cannot say they did
  anything.
- `pin`: the whole process is pinned to one core with `os.sched_setaffinity` before the framework
  is imported (Linux only; on Windows this setting fails, so pass `--latency-threads default,1`).

## CPU results measured here

Machine: Intel Xeon @ 2.10 GHz (model 207), 4 logical CPUs, KVM guest, Linux; a shared cloud
container. PyTorch 2.15.0+cu132 (CUDA build, run on CPU) on Python 3.14.6; JAX 0.12.1 /
jaxlib 0.12.1 / optax 0.2.9 on Python 3.13.18; numpy 2.5.5 in both. Full tables:
[`results/results.md`](results/results.md). Sizes used on the CPU: E = 1 and E = 256 (not 2048)
for the steady and loop suites, E = 32 for the size test, and the full 16 x 128 x 10 micro-batch for C.

**Batch-size-1 latency (the main result), milliseconds, game tick + two forwards.** Two runs of
3,000 ticks each; both shown as run 1 / run 2.

| N | variant | threads | p50 | p99 | max |
|---|---|---|---|---|---|
| 10 | PyTorch eager | default | 4.97 / 5.10 | 10.13 / 10.81 | 21.33 / 149.22 |
| 10 | PyTorch compiled | default | 1.47 / 1.75 | 2.75 / 4.11 | 7.33 / 86.29 |
| 10 | JAX jit | default | 1.59 / 1.50 | 2.32 / 2.18 | 4.28 / 4.49 |
| 10 | PyTorch eager | pin | 4.92 / 4.89 | 8.98 / 8.48 | 14.01 / 11.94 |
| 10 | PyTorch compiled | pin | 1.60 / 1.55 | 3.09 / 2.52 | 5.99 / 4.83 |
| 10 | JAX jit | pin | 0.95 / 0.89 | 1.66 / 1.75 | 4.63 / 3.54 |
| 40 | PyTorch eager | default | 6.36 / 7.00 | 13.38 / 12.96 | 26.52 / 28.02 |
| 40 | PyTorch compiled | default | 2.52 / 2.34 | 4.46 / 4.51 | 26.83 / 12.81 |
| 40 | JAX jit | default | 3.17 / 3.09 | 4.71 / 4.90 | 7.73 / 15.87 |
| 40 | PyTorch eager | pin | 6.86 / 6.04 | 12.02 / 11.04 | 14.44 / 15.95 |
| 40 | PyTorch compiled | pin | 2.61 / 2.52 | 4.93 / 3.84 | 11.91 / 5.63 |
| 40 | JAX jit | pin | 1.92 / 1.88 | 3.09 / 3.05 | 5.42 / 5.40 |

(The `1`-thread rows are in `results/results.md`; they are close to the `default` rows.)

- Every variant's p99 was inside 16 ms in both runs. Compiled PyTorch and jitted JAX have p99 of
  2 to 5 ms; eager PyTorch has p99 of 8 to 13 ms, which leaves little of a 16 ms budget.
- The maximum is not stable on this machine: single ticks of 86 ms and 149 ms appeared in run 2
  for PyTorch with default threads and not in run 1. I did not find the cause (other tenants on
  the host, the OpenMP thread pool, or Python's garbage collector are all candidates). Treat the
  max column as a property of this container until it is repeated on the laptop.
- Pinning the process to one core made JAX about 40% faster at the median and left PyTorch
  unchanged. So at batch size 1 neither framework gains from more threads here, and JAX's default
  thread pool costs it.

**Compile time, seconds, fresh process (cold / warm cache).**

| unit | PyTorch `torch.compile` | JAX `jax.jit` |
|---|---|---|
| A policy step | 27.8 / 2.20 | 1.05 / 0.24 |
| B game step | 19.1 / 1.30 | 0.51 / 0.11 |
| A+B rollout tick | 32.6 / 3.08 | 3.13 / 0.62 |
| C PPO update (16 x 128 x 10) | 67.8 / 4.59 | 5.39 / 2.61 |

Importing the framework took a further 1.6 to 2.0 s (PyTorch) and 0.5 to 0.6 s (JAX) per process.

**Steady time, milliseconds per call, median (N = 10).**

| unit | size | PyTorch eager | PyTorch compiled | JAX jit |
|---|---|---|---|---|
| A policy step | E = 1 | 1.81 | 0.82 | 0.50 |
| A policy step | E = 256 | 27.8 | 16.6 | 33.3 |
| B game step | E = 1 | 0.39 | 0.057 | 0.21 |
| B game step | E = 256 | 1.89 | 1.81 | 0.54 |
| rollout tick | E = 1 | 4.84 | 1.53 | 1.10 |
| rollout tick | E = 256 | 54.4 | 37.4 | 60.5 |
| C PPO update | 16 x 128 x 10 | 874 | 717 | 1203 |

On this CPU, JAX is faster at batch size 1 and slower than PyTorch on the matmul-heavy batched
work (policy at E = 256, the PPO update). That is a statement about XLA's CPU backend against
PyTorch's MKL build on 4 cores; it says nothing about either on a GPU.

**Team sizes (rollout tick, E = 32).** Time to first result in seconds, then steady milliseconds.

| method | N = 10 (compile) | N = 20 | N = 40 | N = 100 | graphs compiled |
|---|---|---|---|---|---|
| PyTorch default | 29.8 s, 8.06 ms | 23.1 s, 10.3 ms | 0.02 s, 20.8 ms | 0.07 s, 70.5 ms | 2 |
| PyTorch `dynamic=True` | 49.0 s, 7.61 ms | 0.03 s, 11.9 ms | 0.02 s, 26.8 ms | 0.07 s, 62.4 ms | 1 |
| PyTorch `mark_dynamic` | 36.9 s, 7.82 ms | 0.01 s, 10.0 ms | 0.02 s, 19.2 ms | 0.06 s, 59.3 ms | 1 |
| PyTorch padded to 100 | 28.0 s, 50.6 ms | 0.05 s, 53.3 ms | 0.05 s, 51.0 ms | 0.05 s, 47.8 ms | 1 |
| JAX re-jit per size | 3.47 s, 15.3 ms | 3.60 s, 18.2 ms | 3.75 s, 30.2 ms | 4.03 s, 64.1 ms | 4 |
| JAX padded to 100 | 3.41 s, 65.8 ms | 0.08 s, 65.5 ms | 0.07 s, 66.0 ms | 0.06 s, 69.3 ms | 1 |

- PyTorch's dynamic shapes worked here without error: by default the first new size cost one
  recompile (23 s) into a size-generic graph and later sizes cost nothing; with `dynamic=True` or
  `mark_dynamic` there were no recompiles at all, and steady times at N = 10 matched the
  specialised graph within noise. Only the ship axis changed (E was fixed) and the model is this
  small toy, so this does not show the real code would behave the same.
- JAX recompiles for every new size (about 3.5 to 4 s each on this CPU) and keeps each size cached.
- Padding to 100 slots costs, at N = 10, 50.6 ms against 8.06 ms in compiled PyTorch (6.3 times)
  and 65.8 ms against 15.3 ms in JAX (4.3 times).

**Whole rollout, T = 128 ticks, ticks per second.**

| variant | E = 1 | E = 256 |
|---|---|---|
| PyTorch eager tick, Python loop | 145 | 18 |
| PyTorch compiled tick, Python loop | 461 | 28 |
| JAX jitted tick, Python loop | 872 | 17 |
| JAX `lax.scan`, one jit | 1006 | 17 |

At E = 1 the scan is about 15% faster than a Python loop over the jitted tick; at E = 256 on the
CPU the arithmetic dominates and the loop form makes no difference.

**Operation counts.** The uncompiled PyTorch rollout tick dispatches 713 compute operations plus
276 view operations (one policy forward 265, game step 102, observe 69). For comparison the
project's v1 was about 2,600 per tick and v2 about 14,900, so this toy is roughly a quarter of
v1's operation count and a twentieth of v2's. JAX's traced tick has 1,354 primitives, which is
not the same kind of number (they are fused and compiled, not dispatched one by one).

## What the CPU numbers do and do not show

- **Batch-size-1 latency is the meaningful result.** Real-time play is expected to run on a CPU
  at batch size 1, and that is what was measured, though on a container CPU and on a toy game with
  about a quarter of v1's operations per tick. It should be repeated on the laptop's CPU.
- **Compile times are indicative only.** On a CPU build `torch.compile` generates C++ and runs
  `g++`; on a GPU it generates Triton kernels instead. JAX compiles through XLA's CPU backend
  here and its GPU backend there. The ratio between the frameworks may be quite different on a GPU.
- **Steady GPU speed is not measured at all.** The batched CPU numbers (E = 256, the PPO update)
  compare two CPU backends and do not predict the GPU ranking.
- **The size test shows mechanism, not cost.** Whether dynamic shapes hold, and how many
  recompiles happen, should carry over; the seconds will not.

## Limitations

- Toy game, small model, synthetic stored rollout. The real game has several times more
  operations per tick and may contain operations that force `torch.compile` to specialise or
  break the graph; this spike cannot show that.
- CPU numbers only, from a 4-core cloud container with visible timing noise.
- JAX ran on Python 3.13 and PyTorch on 3.14 (the versions in the two scratch environments).
  `uv` resolves jax 0.12.1 for Python 3.14 as well (checked with a dry run), but I did not run it there.
- Sampled actions differ between the frameworks (different random generators), so rollouts
  diverge after the first tick; timings are of the same operations on different trajectories.
- The recompile counts come from private counters (`torch._dynamo.utils.counters`, the jitted
  function's `_cache_size()`); the first-result times are the independent evidence.
- PyTorch runs with its default float32 matmul precision (`highest`, no TF32). I could not
  confirm what XLA does for float32 matmul on a GPU; `torch_bench.py --matmul-precision high`
  is there if the GPU comparison needs checking both ways.
- Under bf16 autocast on the CPU, `nn.RMSNorm` printed "Mismatch dtype between input and weight
  ... Cannot dispatch to fused implementation"; I do not know whether the GPU path does the same.

## Parity

```
<torch python> parity.py dump torch /tmp/parity_torch.npz
<jax python>   parity.py dump jax   /tmp/parity_jax.npz
<any python>   parity.py compare /tmp/parity_torch.npz /tmp/parity_jax.npz
```

Run here on the CPU in float32, the same numpy weights and inputs in both: policy step, learner
forward, 600 game ticks under the same random actions (ships start one hit from death; 12 of 40
died and respawned), the observation, the two forwards of a tick, the PPO loss, its gradient
norm and the loss after each of 5 Adam steps all agree; the worst `max|diff| / (1 + max|value|)`
was 1.8e-06 (largest absolute difference 7e-03 m on bullet positions of magnitude 6,000 m), and
every integer and boolean array is identical (output in `results/parity_cpu.txt`). Each side also confirmed that the learner form
equals repeated single steps (1e-06) and that padding 10 ships to 100 slots leaves the real
ships' game state identical and their policy logits within 1e-06. Not checked: compiled PyTorch
against eager PyTorch, the bf16 variants, and anything on a GPU.

## Running the GPU part on the laptop (RTX 4070 Laptop, 8 GB)

The JAX GPU build needs Linux or WSL2. The JAX installation page
(<https://docs.jax.dev/en/latest/installation.html>, source `docs/installation.md` on the `main`
branch, read on 2026-11-22) lists, for "NVIDIA GPU": Linux x86_64 "yes", Windows x86_64 **"no"**,
Windows WSL2 x86_64 **"experimental"**; and says of the CUDA wheels "Note: wheels only available
on linux" and "These `pip` installations do not work with Windows, and may fail silently". It
also says "the driver version must be ... >= 580 for CUDA 13 on Linux", and "On CUDA 13, JAX
supports NVIDIA GPUs that have SM version 7.5 or newer". CPU JAX on native Windows is listed as
"yes" in the platform table and as "Windows, x86_64 (*experimental*)" in the wheel list. What I
do not know: how well the "experimental" WSL2 GPU path works in practice, which Windows driver
version WSL2 needs for CUDA 13, and whether an RTX 4070 Laptop has any quirk; none of that could
be tested here.

So run both frameworks in the same place, native Linux or WSL2 (Ubuntu), so the comparison is
like for like. For PyTorch on native Windows I could not confirm that `torch.compile` works on a
GPU (it needs Triton, which the Linux wheel installs as a dependency); if it fails, the script
records the failure and carries on with the eager rows.

Setup, from this directory (`docs/rewrite/sections/framework-spike`). A C++ compiler (`g++`) is
needed for PyTorch's compiled rows on the CPU.

```bash
# PyTorch: the repository's own environment pins torch 2.15.0 (cu132) on Python 3.14 (pyproject.toml).
# Not run here: I used a scratch environment holding that same torch build, plus numpy.
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# JAX: a separate environment outside the repository (CUDA libraries come from pip).
uv venv ~/spike-jax --python 3.13
uv pip install --python ~/spike-jax/bin/python "jax[cuda13]==0.12.1" optax==0.2.9
~/spike-jax/bin/python -c "import jax; print(jax.__version__, jax.devices())"   # expect a CudaDevice
```

I checked that `jax[cuda13]==0.12.1` resolves on Linux (dry run: it pulls `jax-cuda13-plugin`
and the NVIDIA CUDA 13 pip packages); I could not install or run it without a GPU.

The GPU part, one command per framework (E = 1 and 2048, N = 10, PPO 16 x 128 x 10 in float32
and bf16, size test at E = 256):

```bash
uv run python torch_bench.py --device cuda --suite all --out results/torch_gpu.json
~/spike-jax/bin/python jax_bench.py --device cuda --suite all --out results/jax_gpu.json
```

Then the tables, and the latency suite on the laptop's own CPU (the `all` suite on `cuda` also
runs latency on the GPU, which is not the number play needs):

```bash
uv run python torch_bench.py --device cpu --suite latency --out results/torch_laptop_cpu_latency.json
~/spike-jax/bin/python jax_bench.py --device cpu --suite latency --out results/jax_laptop_cpu_latency.json
python summarize.py results/torch_gpu.json results/jax_gpu.json results/torch_laptop_cpu_latency.json results/jax_laptop_cpu_latency.json > results/results_laptop.md
```

Notes for the GPU run:

- Run one framework at a time. The JAX GPU memory page says "JAX will preallocate 75% of the
  total GPU memory when the first JAX operation is run"; on an 8 GB card that leaves little for
  anything else running at the same moment.
- If the size test at N = 100 runs out of memory (the hit test is `[E, 800, 100]`), lower
  `--size-envs` (default 256 on `cuda`) and say so next to the result.
- `--precisions f32` skips the bf16 PPO rows; `--skip-compiled-c` skips `torch.compile` of the
  PPO update if it takes too long; `--envs`, `--micro`, `--sizes`, `--pad-to`, `--repeats`,
  `--budget` change the sizes. Each suite can be run alone with `--suite`.
- A failed child process is recorded in the JSON (`"suite": "error"`, with the end of its stderr)
  and listed under "Failures" by `summarize.py`; the run continues.

## How to read the results

- `compile`: `cold first call` is what a first launch pays; `warm first call` is what every
  later launch pays if the cache directory is kept. Subtract nothing: the first call includes
  one execution, which is small next to the compile.
- `steady`: compare medians only where the quartile ranges do not overlap.
- `sizes`: a `first result` of seconds at a new N means a recompile; `graphs` counts them. The
  padded rows at N = 10 against the exact rows at N = 10 give the price of one shape for all sizes.
- `loop`: `ticks/s` is rollout ticks; `env-ticks/s` multiplies by E.
- `latency`: p99 is the number to hold against the 16 ms budget; treat `max` with care (one
  sample in 3,000).
