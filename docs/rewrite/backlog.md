# Backlog: deferred until v2 trains (2026-11-03)

The author's priority is a minimum viable v2: a BC agent trained, then an RL agent trained
to about 1500 Elo (scripted = 1000), a play interface that makes sense and looks good, and
the analysis tooling up to spec. Once we know things are good, we decide what else is
worth pursuing. These items wait until then. Each says where it came from.

## Speed

- **A per-component profile of the training update:** forward and backward time for the
  encoder, each spatial and temporal layer, each head, the losses, the label codes and the
  optimizer, in bf16 with the kernels. This comes first, before any item below.
  (Update profiling phase 2; branch `perf/update` holds earlier, pre-kernel work.)
- **The shot-trajectory head in bf16:** it runs in fp32 even when training is in bf16,
  because it is called outside the model's autocast.
- **Fewer shot ages:** stratified sampling (an age at or near the end, one at or near the
  hit, two at random in the middle), weighted so the loss stays unbiased. Test it against
  all ages over three seeds. An earlier uniform 3-age test scored 0.62 against 0.97 on one
  seed.
- **A Triton kernel for the shot-trajectory head.**
- **Compile time** (author, 2026-11-07): the first BC update costs 550-900 s against about
  245 s steady, and a tournament's first batch 560-750 s against about 145 s, so compiling
  and graph capture take roughly 5-10 minutes each time, even with the on-disk inductor
  cache warm. In the physics race JAX compiled in under a minute. Find where the time goes
  (`TORCH_LOGS=recompiles,graph_breaks`, compile versus CUDA-graph capture versus Triton
  autotuning), then cut it without losing steady-state speed: regional compilation (compile
  the repeated Yemong block once instead of the whole unrolled graph), PyTorch's portable
  compile caches (`torch.compiler.save_cache_artifacts` / `load_cache_artifacts`), an
  ahead-of-time export (AOTInductor) of the rollout and learner graphs, no recompiles from
  shape or dtype changes, and fewer autotune candidates. Measure steady update time before
  and after; a faster compile that costs steady speed is not wanted.
- **`bnb ratings --device cpu` fails** in inductor (`KeyError: 'buf52'`), found 2026-11-07 by
  the run-control queue's end-to-end check; the GPU path works. Reproduce on v2 without the
  queue, then fix: CPU tournaments are useful for tests and small checks.

## Portability

- **Portable batches** (decisions in `sections/portable-batch.md`): the micro-batch size
  and host storage chosen automatically from the memory probe. Needed before the rented-GPU
  publication run, not for the MVP on the laptop.

## Later, by agreement

- The 128-versus-256 width comparison (the tooling comes with the cleanup pass).
- An autoregressive-predictor diagnostic.
- Capsule fields and a new mode.
- Native goldens; power-profile profiling; results and charts; client packaging.
