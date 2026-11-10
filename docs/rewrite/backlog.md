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

## Tournaments

- **Compile once per build, not per field (do before the next tournament sweep; the author,
  2026-11-09).** Today `bnb.league.matches` compiles one step graph for the whole field (every
  network group's forward, the lanes, the blocks), so a new mix of builds, block count or mode
  retraces and recompiles: ~15 min per mode on the 3090, and the 16-block layout took ~33 min with
  no new cache files (all Inductor cache hits: the cost is dynamo/AOT tracing, paid per process).
  Fix: (1) compile each build's network forward on its own and keep the engine loop uncompiled,
  so any field reuses it; (2) persist the traced graphs across processes (PyTorch's precompile /
  cache-artifact APIs) so a new tournament skips tracing; (3) pass build settings that are only
  constants of the forward (e.g. `extrapolation_decay`, `physics_hidden_decay`,
  `physics_hidden_fade`) as runtime inputs so builds differing only there share one group.
  Measured context: `docs/rewrite/sections/gpu-box-log.md` (2026-11-09, tournament 4). Related:
  more blocks per batch did not raise throughput (5.3 vs 5.1 games/s), so per-block work, not GPU
  fill, bounds a batch; batching the networks across blocks would be the throughput fix.
- **Game reuse is built** (`bnb ratings --reuse`, players matched by weights hash): use it for
  every sweep.

## Portability

- **Portable batches** (decisions in `sections/portable-batch.md`): the micro-batch size
  and host storage chosen automatically from the memory probe. Needed before the rented-GPU
  publication run, not for the MVP on the laptop.

## The scripted agent

- **A proper scripted retune for the real-scale deathmatch and Frontline** (2026-11-10). The
  light pass (`docs/explanation/game/real-scale.md`, "The scripted agent's real-scale
  sets") searched each preset for 12-13 generations from the standard set, on the P-51D
  lineup alone, 2,048 games a generation, and chose among eight finalists at T = 1. The new
  sets beat the standard ones 0.99 and 0.97 and kill faster, but the Frontline set captures
  less than half as often in self-play (0.20 front moves a minute against 0.44; 69% of its
  games drawn): a retune should score captures, not only wins.
  It did not cover: the other 26 types or mixed lineups (the sets are tuned on the P-51D,
  and the ship-type tournament rates every type with them); the timing parameters' bounds
  (the memory fade's centre is held to 0.5-2 s, the switch rates to 1-30 per second, both
  set at the standard tempo); the per-decision floors (`EPSILONS`, not searched); the
  chaser (still the standard distances in card lengths, a far coarser aim than the real
  cones); and the search's plateau rule, which fires early once every candidate beats the
  reference (ratings anchored on a saturated reference are held by the prior: a fresh
  anchor per generation, or the population's own median, would fix it). The shot logistic's
  centre moved past one gun range (shooting at targets still closing); whether that is
  wanted, or a sign the shot cost is too low, is open.
  Since done in part by the balancing (`docs/explanation/game/real-scale-balance.md`, round
  1): both sets searched against their champions on mixed lineups of every type, with a
  plateau rule that survives a strong anchor; the Frontline set now captures twice as often
  as the light retune's (0.69 front moves a minute in self-play). Still open: the timing
  bounds, the floors, the chaser, and the shooting from beyond a gun range (kept by the
  author's decision).

## Later, by agreement

- The 128-versus-256 width comparison (the tooling comes with the cleanup pass).
- An autoregressive-predictor diagnostic.
- Capsule fields and a new mode.
- Native goldens; power-profile profiling; results and charts; client packaging.
