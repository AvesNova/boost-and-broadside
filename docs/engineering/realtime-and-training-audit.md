# Real-time and training performance audit

2026-09-19. Scope: one Frontline match with **two distinct fleet policies**, 5v5
and 50v50, plus batched RL training. This is an audit and a bounded optimization
experiment, not a claim that the game now runs in real time.

**Battery constraint:** all running and queued intensive tests were stopped
when the user reported 40% battery with no charger. Power-saving settings were
never changed. The report uses completed measurements only, with the partial
training observation explicitly identified below. Further experiments are
deferred until the user authorizes them.

**Handoff update:** the user subsequently plugged in and authorized further
intensive experiments, retaining the prohibition on changing power settings.
This report remains the historical audit. See the
[benchmark archive](performance-audit/README.md) and
[fresh-agent prompt](performance-handoff-prompt.md) for the next run.

## Recommendation

**Keep PyTorch for the policy and trainer. First optimize the environment's
data flow, visibility, and presentation separately. Evaluate Warp as the next
simulation backend if functional PyTorch kernels cannot meet the measured
budgets. Do not start with a complete Rust or Godot rewrite.**

The current results do not establish that the game design or neural policy is
inherently too expensive. They establish that the current implementation misses
the deadline. Much of its cost comes from dispatching many small operations,
materializing dense geometric intermediates, and repeating CPU drawing work.
Those are tractable implementation problems, although meeting the deadline is
not yet demonstrated.

Two changes deserve priority across backend choices:

1. A narrow simulation interface with explicit state inputs/outputs, plus a
   lean inference path. This enables correct compilation, differential testing,
   Warp/native kernels, and a separate renderer without replacing PPO.
2. Visibility and rendering algorithms that avoid unnecessary work. A language
   port retaining the same dense tensors and per-observer compositing may move
   the bottleneck without eliminating it.

## Evidence and limits

The supplied screenshot reports these two-policy timings (milliseconds):

| Scenario | Device | No render | With render | Separately measured render cost |
|---|---|---:|---:|---:|
| 5v5 | CPU | 53.1 | 102.5 | 16.7 |
| 5v5 | CUDA | 65.9 | 81.9 | 13.5 |
| 50v50 | CPU | 1114 | 1215 | 77.7 |
| 50v50 | CUDA | 66.8 | 144.3 | 75.4 |

These columns are **not additive**: separate runs, trajectories, clock state,
and phase attribution can differ. In particular, 102.5−53.1 is not 16.7.
Use complete frame measurements for the deadline, not sums of unrelated medians.

Fresh measurements here used the available i7-13620H, Python 3.13.11,
PyTorch 2.13.0+cu130, one Torch CPU thread, the current RL model (width 128,
two Yemong blocks, full map attention), 10 projectile slots per ship, seed 99,
and 900×900 offscreen rendering. The machine reports `quiet`, `powersave`, and
energy preference `power`; these settings were not changed. The sandbox hid
CUDA initially. After the user identified that restriction, execution outside
the sandbox confirmed the RTX 4070 Laptop, 8,188 MiB, driver 595.84, CUDA 13.2,
and Torch CUDA availability. GPU measurements below use that access. Rust and
Godot were not benchmarked.

The existing realtime harness uses randomly initialized policies, not trained
combat checkpoints. The initial CPU reproduction used 10 warmup and 40 measured
frames per scenario. It is a diagnosis, not a p99 certification or an exact
reproduction of the screenshot's unknown run conditions. The newest inspected
trained checkpoint, `icy-energy-741/best_training.pt`, records
`recursive_belief_v8`; the runtime requires `frontline_shields_v9`. It was not
silently loaded under a different observation contract.

| Fresh CPU reproduction | Median frame | p99 estimate | Policy phase | Environment phase | Render phase |
|---|---:|---:|---:|---:|---:|
| 5v5, two policies | 95.04 | 109.26 | 50.77 | 29.56 | 16.00 |
| 50v50, two policies | 1638.63 | 1727.43 | 139.21 | 1413.83 | 113.11 |

Phases come from a later diagnostic window, not the same frames as the median.
They should not be subtracted to infer small speedups. The slower absolute
numbers reinforce the importance of interleaving alternatives under the same
power conditions and pinning the software version.

Existing repository evidence is useful but historical:

- [RL throughput investigation](rl-throughput.md): 2,436→5,099 environment
  transitions/s on the RTX 4070 Laptop after several changes, with interleaved
  validation. That was a different Torch version and an eight-ship configuration.
  It does not establish today's 5v5/50v50 performance.
- [Concurrency investigation](concurrent-rollout-plan.md): sequential, streams,
  and host threads all remained far above 33.3 ms. Its reversed 50v50 pairs
  found only about 1.1% difference, within variation. The document also retains
  some pre-implementation planning text; its measured-results section is the
  relevant evidence, not a second claim that buffering is still unimplemented.
- [Compile regression tests](../../tests/env/test_compiled_tick.py) and the
  throughput investigation document incorrect shots and stale observations from
  earlier mutable-state compilation attempts. Treat those apparent gains as
  invalid, not as optimization targets already achieved.

User changes to the realtime harness, interactive mode, tests, and concurrency
document were present before this audit and were preserved.

Additional harness caveats: the one-policy mode supplies zero actions for the
other team, so it does not measure a scripted controller's analysis cost.
Interactive inference here uses FP32 unless explicitly changed; the trainer
uses CUDA bf16 autocast. GPU warm-up deliberately sustains load, while a real
bursty client can run at lower clocks. Offscreen drawing excludes display flip,
vsync/compositor behavior, input polling, and frame pacing. These differences
must be recorded before comparing screenshot rows, training, and actual play.

## What is actually slow

### Visibility is the large CPU scaling failure

The fresh [50v50 component profile](performance-audit/results/performance-audit-cpu-profile-50.json) used
fixed firing/turning actions, eight unprofiled samples per component, then a
separate five-call `cProfile` window:

| Component | Unprofiled median, ms |
|---|---:|
| Physics tick alone | 46.53 |
| Observation including projectile visibility | 1323.70 |
| Entire wrapper step | 1387.06 |
| Offscreen drawing | 106.23 |
| Observation without projectile visibility | 70.71 |

These components were sampled sequentially, sometimes advancing the state;
they are diagnostic timings, not an additive decomposition or paired full-game
ablation. During the profiled wrapper window, LOS occupied roughly 6.52 of
7.07 seconds. The renderer profile spent roughly 0.98 of 1.17 seconds in fog
drawing. Profiling approximately doubled drawing time; use the unprofiled
106 ms for latency, not the profiled per-frame figure.

[`perception.py`](../../src/boost_and_broadside/env/perception.py) builds dense
LOS geometry over `(B, observers, targets, occluders)`. For projectiles this is
`B*N*(N*K)*M`. It computes geometry before masking inactive bullets and before
using the range mask to reject irrelevant targets.

- 5v5: `1*10*100*15 = 15,000` observer/projectile/core combinations.
- 50v50: `1*100*1000*77 = 7,700,000` combinations, **513× larger**.
- One float32 intermediate at the large shape is 30.8 MB (decimal); one
  complex64 intermediate is 61.6 MB. Several intermediates coexist. Training
  multiplies these sizes by batch width if projectile perception is enabled.

The 50v50 scenario increases fields from 10 to 72 and scales the map and sight
range, so this is not a controlled experiment changing only ship count.
Ship LOS has the same 513× shape growth with one tenth the target count.

**Do not simply disable bullet visibility in play.** It determines which
projectiles the renderer may display. The current default policy does not
consume bullet tokens, and the training wrapper already defaults to omitting
their perception. This large interactive opportunity therefore does not imply
an equally large training gain.

### Rendering has its own independent deadline failure

[`renderer.py`](../../src/boost_and_broadside/ui/renderer.py) fetches many
separate CPU tensors, extracts scalars in Python, enumerates toroidal images,
and builds/composites an observer mask for each living allied ship. Fog already
uses a quarter-resolution mask, so merely proposing low-resolution fog is not
a new optimization here. At 50v50,
the profile recorded about 116,000 Python/native calls per rendered frame.
The comment claiming fog geometry is negligible is not supported at this size.

Even perfect zero-cost simulation would not make a 75–113 ms renderer reach
30 FPS. Conversely, deleting rendering would leave the supplied GPU
simulation/inference timings above budget. Both paths must improve.

### There is still work after visibility

On this CPU, 50v50 physics alone exceeds 33.3 ms, and two eager policies alone
exceed it at both sizes. Field transport, dense swept collision tests, and
Frontline state updates appear in the physics profile. Removing one bottleneck
does not finish the job.

The interactive wrapper also computes rewards, source statistics, episode
accumulators, and perception diagnostics. Most are not needed to decide actions
or display a match. This is an opportunity for an explicit inference mode,
provided it retains the authoritative dynamics, observations, belief updates,
termination, and controller-specific action timing.

## Audit of the implementation options

| Option | Single-game latency | Parallel training | Cost/risk | Decision |
|---|---|---|---|---|
| Functional PyTorch fusion + hot-path cleanup | Directly attacks small-operation dispatch and intermediates | Reuses GPU tensors and PPO; likely broadest low-migration opportunity | Moderate refactor; mutation, graph breaks, specialization, compile startup | First implementation track |
| Exact geometric pruning + tiled kernels | Especially valuable at 50v50 | Reduces memory traffic and permits larger batches; small shapes may prefer dense | Numerical/discrete parity; broad-phase worst cases | First algorithm track, regardless of language |
| GPU drawing + compact render snapshot | Essential for large matches | No direct benefit to headless training | Moderate presentation work | Independent first track |
| Warp simulation kernels + PyTorch policies | Plausible low-dispatch GPU path; CPU must be measured separately | Strong fit for batched custom physics and GPU-resident observations | Substantial port and parity work; NVIDIA GPU dependency | Preferred backend experiment after targeted fusion |
| Rust native CPU sim + Python/PyTorch | Plausible strong B=1 path if loops actually become native | CPU workers may scale; GPU inference transfer and batching can dominate | Substantial port, bindings, two-backend maintenance | Conditional alternative for CPU-first play |
| Full Rust sim + renderer + NN runtime | Broad control but solves several unrelated problems at once | Rust alone does not supply a CUDA-batched trainer | Highest integration scope; model/runtime migration | Defer |
| Godot front end + shared custom sim | Useful rendering, UI, input, interpolation | Training remains external and headless | Moderate client integration | Valid presentation choice |
| Full Godot simulation with per-object nodes | Could support ordinary interactive play, unmeasured here | No demonstrated replacement for thousands of GPU tensor environments | Physics/rule drift and per-world engine overhead | Not the training optimization strategy |

### 1. More `torch.compile`, and optimizing Python

Compile larger **pure numerical regions**, not more tiny wrappers. Proposed
boundaries are ship integration/shooting, projectile transport, collision
reduction, Frontline transitions, observation/perception, and packed reward
computation. Pass explicit tensor bundles and static configuration; return all
changed tensors, events, and masks. Commit state changes outside the compiled
region. Keep a reference implementation for differential tests.

Use real x/y arrays or real tensor views where complex math inhibits useful
fusion. Avoid allocating dense observer/target/core intermediates if a fused
reduction can consume them directly. Begin with `fullgraph=True` in isolated
experiments to surface graph breaks, static shape buckets, and compile logs;
measure first-call cost separately from steady state.

PyTorch documents `reduce-overhead` specifically as a CUDA-graph mode useful
for small batches, subject to capture constraints. Thus the realtime harness's
introductory suggestion that batch one has little to gain from compilation is
not a reason to skip the test. This project now clones graph outputs for
ownership, and its earlier graph attempts were largely neutral. Retry graphs
after removing the remaining eager dispatch, not just by changing the policy
flag. See [compile modes](https://docs.pytorch.org/docs/stable/generated/torch.compile)
and [CUDA graph semantics](https://docs.pytorch.org/docs/main/notes/cuda.html).

The repository's historical mutation failures are specific failures that need
regression coverage, not proof that PyTorch can never compile any mutation.
Replacing mutable object plumbing with explicit outputs makes correctness and
ownership much easier to establish.

Useful Python work is removing whole categories of tensor calls: pack reward
components, specialize inference without training metrics, cache static map
data, and reuse prepared constants. Renaming helpers or replacing a short
Python loop with a comprehension will not address dense tensor traffic.
Do not blindly delete observation clones: they protect in-flight policy reads
and buffered action concurrency.

### 2. Warp

Warp compiles typed kernel functions to native CPU or CUDA code. Its value here
is expressing fused per-ship/per-projectile work and geometric queries without
dispatching every arithmetic operation from Python. CPU availability does not
by itself establish competitive multithreaded CPU throughput. See
[Warp's programming model](https://nvidia.github.io/warp/stable/user_guide/programming_model.html).

Retain the existing Torch policy, recurrent state, rollout buffer, and PPO.
Prototype only LOS and field transport first, then a complete tick if they win.
Use structure-of-arrays state, bounded projectile storage, explicit event
buffers, and reductions with defined tie handling. Map independent worlds and
objects to kernel indices; tile the field dimension when appropriate. A single
giant kernel may suffer register pressure/divergence, so benchmark several
coarse phases rather than assuming maximum fusion is optimal.

Warp supports zero-copy Torch array views, stream conversion, and mixed
Warp/Torch graph capture. Construct views once where storage is stable; manage
lifetimes and stream dependencies explicitly. Zero-copy is not zero-cost
synchronization, and graph replay must not overwrite retained observations.
See [Torch interop](https://nvidia.github.io/warp/stable/user_guide/interoperability/pytorch.html).

PPO does not require differentiating through this simulator. Do not spend the
port budget on a differentiable-physics tape unless the learning objective
changes. Floating atomic reductions and parallel event ordering still require
careful parity design; see [deterministic execution](https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/deterministic_execution.html).

Warp 1.17.0 was installed in `/tmp/bnb-audit-warp`, independently of the project
environment, to test an isolated LOS kernel. No advertised robotics
or particle benchmark is a speed estimate for this game's fields, visibility,
observations, resets, and policies. A full tick benchmark is the decision gate.

### 3. Rust simulation and rendering

A native Rust CPU loop can avoid Torch's per-operation dispatch, exploit
early exits and cache-local arrays, and parallelize independent worlds. Merely
moving calls to the same eager tensor operators into Rust does not provide
those benefits. Keep the Python boundary coarse: one batch step in, contiguous
observations/events out, with native computation detached from Python where
appropriate. [PyO3 documents this parallelism pattern](https://pyo3.rs/main/parallelism.html).

For B=1, compare CPU simulation plus CPU inference against CPU simulation plus
GPU inference **including both transfers and synchronization**. For training,
compare native CPU actor workers feeding batched GPU inference against a
GPU-resident simulation, including rollout storage and policy-version handling.
Per-ship RPC or tensor conversion is the wrong interface. More worker processes
also consume memory and can oversubscribe Torch threads.

For this existing batched trainer, prioritize GPU-resident simulation with
batched Torch inference/PPO. Treat CPU actor workers plus a GPU learner as a
conditional alternative whose transfer, scheduling, and policy-lag costs must
win on complete iteration time and time to score. Neither topology is selected
by environment-only transitions/s. Across languages, organize simulation data
as contiguous component arrays with explicit active masks and coarse batch
interfaces; adopting an ECS framework alone does not remove dense geometry.

Rust rendering can use a GPU API such as
[wgpu](https://docs.rs/wgpu/latest/wgpu/), but CUDA/PyTorch buffer sharing is a
separate integration problem; do not assume a portable graphics API exposes a
portable zero-copy CUDA path. A packed B=1 CPU snapshot is a simpler first
interface. Keep inference in PyTorch initially rather than porting the custom
recurrent model and checkpoint format at the same time.

Rust is most compelling if CPU-only distribution is a firm requirement or
profiling shows native CPU simulation wins after including inference. It is
not inherently a GPU-training solution. No Rust or Godot executable was found
on the current PATH, and neither was benchmarked in this audit.

The migration inventory is larger than `physics.py`: the nine inspected
environment/state/observation/wrapper files contain about 4,400 lines, plus
1,240 renderer lines, including comments. A port also needs the observation and
reward contracts, test fixtures, and Python/NN interface. These counts describe
scope, not an effort estimate or a reason to avoid a port that wins decisively.

### 4. Godot

Godot can replace presentation and input while a shared custom core remains
authoritative. Its low-level rendering/physics servers bypass scene-node
overhead; this is preferable to assigning a heavyweight node to every bullet
in a training world. See [server optimization](https://docs.godotengine.org/en/stable/tutorials/performance/using_servers.html).

Its stock rigid-body physics does not implement the project's optical field
transport, toroidal swept bullets, aggregated shield damage, or respawn rules.
Reimplementing these changes the scope from adopting a renderer to migrating
the environment. Running many headless Godot processes also introduces an
observation/inference transport problem; headless execution alone is not
equivalent to GPU vectorization.

Choose Godot for client/editor/UI needs, not on an unsupported claim that
it will accelerate PPO. A native extension/shared simulation can support both
the client and headless training, with the Torch reference as a test oracle.

## Additional options worth auditing

**Exact pruning before geometry.** Reject dead observers, inactive bullets,
out-of-range targets, and cores whose bounds cannot intersect the segment.
For team bullet visibility, stop after the first allied observer sees a target;
retain per-observer ship masks where diagnostics need them. Use a static
spatial index for map cores and a dynamic grid for ships/projectiles if density
justifies it. CPU code can compact aggressively; GPU code may prefer fixed-size
tiles with masks, device-side queues, or bounded candidate lists. Define an
overflow fallback; never silently discard collisions or sight lines. Benchmark
the overhead at 5v5 and dense worst cases. Preserve both-endpoints-inside-core,
strict tangent comparisons, shot reveal, and toroidal seam behavior.

**A lighter render interface.** Build one render snapshot per tick, reuse
static fields until map reset, convert arrays once, cull offscreen geometry,
cache camera transforms within a frame, and batch sprites/lines. On CUDA,
replace many blocking `.cpu()`/`.item()` operations with a packed transfer.
Double buffering can help only with explicit events and immutable ownership.
Render fog on the GPU or rasterize it into a bounded-resolution mask with
cached static occluders. Removing fog is an ablation, not an acceptable silent
gameplay/visibility change; image comparisons should check visible information.

**Separate simulation and display cadence.** Frontline is currently 30 Hz,
not 60 Hz physics. A 60 FPS client can interpolate snapshots while simulation
and policy remain at 30 Hz. Preserve the trained one-tick NN action delay;
interpolation adds presentation latency and needs input-to-display measurement.
It cannot rescue a simulation that itself runs at 10 Hz. Godot describes this
[interpolation pattern](https://docs.godotengine.org/en/stable/tutorials/physics/interpolation/using_physics_interpolation.html).
Changing `dt` or increasing action repeat changes control dynamics and needs
training/evaluation; it is not a free speed optimization.

**Inference specialization.** Interactive calls currently compute value heads
and log probabilities whose results are discarded. Add an inference entry
point returning actions, belief predictions, and next recurrent state. The
next-state head cannot simply be removed: belief advancement consumes it.
Compile feature preparation and belief arithmetic where safe. If both teams
share weights, batch their separate perspectives/states into one call; two
distinct policies need distinct weights and cannot be treated as one shared
policy. Test CUDA autocast against today's interactive precision separately
from the trainer's existing bf16 path.

**Policy architecture and deployment.** The model already supports map-memory
alternatives. Benchmark routing static map tokens through cached K/V or a
smaller map encoder, local attention at large fleets, a smaller distilled
student, and deployment quantization only after the environment floor falls.
These require accuracy/Elo and recurrent-state validation; they are not
drop-in arithmetic equivalences. Exporting to another inference runtime may
help deployment but must cover the custom feature pipeline and recurrence.
Only cache genuinely static field features: capture progress/roles and deep
map representations mixed with ship tokens are dynamic in the current model.

**Native kernels without a full backend port.** A targeted C++/CUDA extension,
Triton kernel, or Warp kernel can replace the same pure LOS/collision boundary.
Compare them against Inductor-generated kernels on identical inputs. Backend
choice is secondary to avoiding the `B*N*N*K*M` materialization. Keep the
candidate with the best complete-path benefit and maintenance cost.

**Power, versions, and thread configuration.** Preserve the user's power-saving
settings as a fixed constraint. No performance-profile test ran, and no power-mode
speedup is claimed. CPU dispatch and the shared laptop power budget remain
relevant to interpreting CUDA measurements. Pin Torch/compiler versions
for comparisons. Tune CPU thread count separately for small play and wide
training/large geometry; `play` already pins one thread, while the inspected
`watch` path does not apply that same override. A thread-count change or faster
CPU cannot by itself justify retaining a renderer that still misses its budget.

**Training-specific changes.** Audit the primary wrapper, auxiliary scales,
scripted opponents, privileged targets, evaluator, rollout storage, and PPO
update separately. The wrapper generates masked reset candidates each tick,
even with no resets: fuse this path or use device-side conditional work rather
than introducing a per-step GPU→CPU `any()` check. Preserve per-world RNG and
reset distributions. Reconsider evaluator cadence/width or a separate frozen
evaluator only as explicit sampling/policy-version changes. Increasing batches
or rollout lengths can change optimization, memory, and time-to-learning even
when steps/s improves.

The historical investigation attributed about 20.6% of wall time to environment
stepping. At that particular mix, even an infinitely fast replacement would
cap total speedup at `1/(1−0.206) = 1.26×`. That is an illustration of Amdahl's
law, not a current estimate. Re-profile before pricing a multi-week rewrite.
Keep total iteration time and time to a fixed evaluation score as outcomes.

## Acceptance gates and execution order

1. **Correct benchmark contract.** Record commit plus dirty changes, checkpoint
   and model config, device/software, power mode/clocks, precision, batch width,
   ships/fields/bullet capacity and active density, seed, rendering resolution,
   reset count, and compile/startup time. Use compatible checkpoint combat and forced
   stress states, including mass respawns, full projectile capacity, overlapping
   cores, sparse worlds, and seam crossings. This audit fixed the standalone
   kernel harness: `primary_only` previously called the evaluator, and
   `rollout_step` called it again. The combined profile double-stepped evaluation
   and the primary-only profile included evaluation. Each now measures its
   declared workload.
2. **Small functional kernels and lean inference.** Test the LOS prototype below,
   then functional physics/observations and inference-only bookkeeping removal.
   Use parity as a hard gate before promoting any timing. Re-run full-frame and
   full-iteration A/B measurements; do not select from microbenchmarks alone.
3. **Presentation in parallel as a workstream.** First pack snapshots and reduce
   Python drawing work, then prototype GPU fog/sprites. Pick Pygame plus a GPU
   renderer, Godot, or Rust based on client requirements, not training claims.
4. **Backend bake-off if budgets still fail.** Implement the same bounded
   visibility/field-transport slice in Warp and optionally native Rust. If it
   wins, extend to one complete 5v5/50v50 tick and batched rollout. Include
   conversions, observations, reset handling, memory, compile cost, and both
   policies. Require a material total-path gain before a broad migration.

Suggested allocation targets, **not predictions**: for a serial 30 Hz client,
aim for ≤10 ms environment/observations, ≤10 ms two-policy/belief work, ≤5 ms
snapshot/drawing, leaving about 8 ms for jitter and input/presentation. For
60 FPS display with 30 Hz simulation, keep drawing comfortably below 16.7 ms
and independently require every decision to complete on its 33.3 ms schedule.
True 60 Hz simulation would need a new <16.7 ms total budget and policy timing
validation.
Validate graphics and simulation together on the shared GPU, with the display
loop able to present without waiting for each policy call. Separate cadence
budgets do not imply independent hardware resources.

For promotion, use at least five interleaved A/B pairs, initially ≥2,000 timed
interactive ticks per run, longer when tails are unstable. Report sustained
ticks/s, median/p90/p99/max, misses at both deadlines, and reset-inclusive
spikes. Render offscreen for attribution, then validate actual window/display
presentation. Random-policy short runs cannot certify responsiveness.

For throughput, sweep B=1, 32, 128, and actual VRAM-fitting production widths
at both fleet sizes; report transitions/s, ship decisions/s, peak allocated
and reserved memory, rollout seconds, update seconds, and complete iteration
seconds. Synchronize only at window boundaries for headline GPU throughput;
separately profile ranges for attribution. Summed overlapping kernel time is
not GPU utilization. Include evaluator and storage costs and keep effective
PPO batch/epochs/opponents constant in implementation comparisons.

Correctness coverage must include complete state and both-team observations,
projectile spawn/ring reuse, collision target/tie rules, shield break versus
death on a later hit, friendly fire, boundary damage, zone capture, respawn,
termination/truncation, reward components, retained observation ownership,
belief/recurrent state, buffered actions, and rollout rows/GAE. Compare discrete
events exactly; set explicit floating tolerances and use fixed input action
traces before stochastic long-horizon comparisons. A different RNG stream is
not automatically a bug, but it prevents seed-only trajectory comparisons.

## Audit experiment and reproduction

The new [performance audit harness](../../benchmarks/performance_audit.py)
contains a pure real-valued LOS prototype, eager/compiled comparisons, parity
probes, component profiling, a two-policy end-to-end A/B mode, and a CPU
environment-only batch-throughput mode. It makes no production changes.

The LOS experiment reformulates complex magnitudes as squared real distances
and lets Inductor fuse the dense reduction. It does not yet implement spatial
pruning. Float32 reassociation can change near-tangent classifications; sampled
parity is evidence for a prototype, not proof of complete semantic equivalence.

### Fresh CPU experiments

The kernel tables report the median of three arm medians, reversing execution
order on alternate pairs. They include input-view preparation and output
allocation. These are static-input kernel workloads, not combat trajectories.

| LOS workload | Existing complex eager | Real eager | Real compiled |
|---|---:|---:|---:|
| 5v5 ship LOS | 1.283 ms | 0.634 ms | 0.445 ms |
| 5v5 projectile-capacity LOS | 1.641 ms | 1.071 ms | 0.948 ms |
| 50v50 ship LOS | 69.303 ms | 32.602 ms | 37.265 ms |
| 50v50 projectile-capacity LOS | 1244.732 ms | 626.559 ms | 344.380 ms |

Sources: [small LOS](performance-audit/results/performance-audit-los-5.json) and
[large LOS](performance-audit/results/performance-audit-los-50.json). The large experiment checked
240,012 boolean comparisons across random, edge, scenario, and changed-value
inputs; its random/edge cases checked the real eager formulation, while its
scenario/changed-value checks also covered compilation. The subsequent small
experiment also checked compiled random/edge probes (22,212 comparisons total).
The large compiled shape first calls took 21.87 s and 6.61 s; compilation/cache
state makes these startup observations, not a portable startup estimate.

Real arithmetic is already a substantial gain at the large shape. Compilation
helps projectile LOS further, but **loses to real eager for large ship LOS**.
This is evidence for choosing boundaries and representations carefully, not
wrapping every function in `compile`.

| Full two-policy CPU frame, 900×900 | Reference median | Compiled LOS median | Verdict |
|---|---:|---:|---|
| 5v5, both policies compiled | 84.16 ms | 80.54 ms | Small/variable gain, still misses 30 Hz |
| 50v50, both policies eager | 1605.14 ms | 680.30 ms | About 2.36× faster, still far from real time |

Sources: [small full-frame pairs](performance-audit/results/performance-audit-realtime-5.json), 60 frames
per arm, and [large pairs](performance-audit/results/performance-audit-realtime-50.json), 20 frames per
arm, three pairs each, ten warmup frames each. Values are medians of run medians.
Small-frame paired reductions ranged from about 1.6% to 8.8%; do not promote
that as a stable gain above a 5% threshold. Large-frame reductions were about
57–58%. Different policy compile settings mean the two rows are not a fleet-size
scaling comparison. Neither test establishes long-run tail latency.

Environment-only CPU batch scaling, with no projectile perception and with
the wrapper's normal automatic reset path, measured approximately 33, 544,
and 855 environment transitions/s at B=1, 32, and 128 respectively. Sources:
[B=1](performance-audit/results/performance-audit-throughput-1.json),
[B=32](performance-audit/results/performance-audit-throughput-32.json),
[B=128](performance-audit/results/performance-audit-throughput-128.json). These are short, one-thread,
fixed-firing-action runs: no neural policies, PPO, evaluator, or storage. They
illustrate throughput amortization, not achieved training throughput or a
comparison against native CPU actors.

### Fresh CUDA kernel comparison

On the RTX 4070 Laptop, the same 50v50 LOS shapes gave the following completion
latencies (median of three arm medians, 50 samples per arm, order reversed on
alternate pairs). GPU synchronization is included; these are callable latency
measurements, not isolated CUDA event kernel durations.

| LOS workload | Existing complex eager | Real eager | Real compiled | Warp prototype |
|---|---:|---:|---:|---:|
| Ships | 1.080 ms | 0.954 ms | 0.200 ms | 0.376 ms |
| Projectile capacity | 5.494 ms | 5.476 ms | 0.271 ms | 0.545 ms |

[Raw GPU kernel results](performance-audit/results/performance-audit-gpu-los-50.json). All three
alternatives matched the reference on 240,012 sampled boolean outputs each,
including random, edge, live-layout, and changed-value inputs. Inductor's
projectile call was about **20× faster** than the existing one. The Warp
prototype was about **10× faster**, but about **2× slower than Inductor** here.

Warp uses one thread per observer/target pair, loops over cores, and exits on
the first blocker. Its times include Torch↔Warp view construction, allocation,
stream conversion, and launch. A persistent Warp-owned state/graph could have
different overhead; this experiment neither establishes a fully optimized Warp
ceiling nor justifies a whole-simulator speed estimate. The important finding
is that PyTorch can express an effective kernel boundary for this workload.

The 5v5 full-frame CUDA A/B used two `default`-compiled policies, 900×900
rendering, and three reversed pairs of 100 frames. Run medians were
120.85/121.38, 117.22/120.63, and 117.40/115.61 ms (reference/compiled LOS).
That is **no repeatable full-frame win**, despite the kernel results.
Diagnostic windows remained around 37 ms policy/beliefs, 67 ms environment,
and 17 ms drawing. [Raw small-game GPU pairs](performance-audit/results/performance-audit-gpu-realtime-5v5-compiled.json).

The large-game full-frame comparisons likewise did not produce a useful win:

| 50v50 CUDA, two compiled policies + rendering | Reference | Candidate | Interpretation |
|---|---:|---:|---|
| Compiled real LOS | 205.54 ms | 211.40 ms | No gain; paired results changed direction |
| Warp LOS | 200.40 ms | 198.27 ms | About 1% difference, not actionable |

These are medians of three run medians, 100 measured frames per arm, with
reversed order. Sources: [Inductor full-frame pairs](performance-audit/results/performance-audit-gpu-realtime-50v50-compiled.json)
and [Warp full-frame pairs](performance-audit/results/performance-audit-gpu-realtime-50v50-warp.json).
Renderer phases remained around 92–109 ms, and the remaining environment
and policy/belief work are also material. Rendering is the largest individual
phase. No candidate met 30 Hz. These full-frame measurements include drawing,
projectile perception, and both policies; they should not be compared directly
with the screenshot's CUDA **no-render** column. Even its with-render column
is not an identical workload/software/power-condition reproduction.

The Warp comparison's compiler counters recorded one captured policy graph
with 567 operations and no graph-break counter. Thus this run does not repeat
the old bug where policy compilation was accidentally bypassed. Capturing a
graph is still not the same as executing the entire observation/belief/policy/
physics path as one efficient replay.

### CUDA environment batch scaling

| Parallel 5v5 environments | Reference transitions/s | With Warp LOS | Reference batch-step ms |
|---|---:|---:|---:|
| 1 | 15.0 | 14.6 | 66.77 |
| 32 | 462.4 | 468.6 | 69.21 |
| 128 | 1878.6 | 1898.1 | 68.13 |
| 960 | 13150.6 | 12269.8 | 73.00 |

Sources: [B=1 reference](performance-audit/results/performance-audit-gpu-throughput-1-compiled.json),
[B=1 Warp](performance-audit/results/performance-audit-gpu-throughput-1-warp.json),
[B=32 reference](performance-audit/results/performance-audit-gpu-throughput-32-compiled.json),
[B=32 Warp](performance-audit/results/performance-audit-gpu-throughput-32-warp.json),
[B=128 reference](performance-audit/results/performance-audit-gpu-throughput-128-compiled.json),
[B=128 Warp](performance-audit/results/performance-audit-gpu-throughput-128-warp.json),
[B=960 reference](performance-audit/results/performance-audit-gpu-throughput-960-compiled.json), and
[B=960 Warp](performance-audit/results/performance-audit-gpu-throughput-960-warp.json).

Each is one 100-step completion-timed window after warmup, fixed firing actions,
no projectile perception, normal wrapper resets. There are no policies,
rollout storage, or PPO updates. In this harness's **throughput mode**, the
`compiled` filename/selector denotes the default Torch reference; it does not
compile LOS or the tick. Metadata explicitly identifies `eager_reference`.
The collision kernel is also eager, whereas the real compiled trainer has its
own collision/perception compilation settings. Do not equate these rates with
production training throughput.

The nearly flat batch-step latency over a 960× batch-width increase strongly
supports a large fixed dispatch cost. GPU batching is valuable even though
B=1 is slow. Warp LOS produced no compelling improvement in this path; the
large-batch slowdown is a single-window observation, not a proven regression.
Ship decisions/s here are ten times environment transitions/s; neither should
be relabeled as rendered FPS.

### Full training: interrupted measurement

The actual `rl_pipeline_profile.py` run used 960 environments, 128 steps,
four rollouts per update, 32 minibatches, 62,500 microbatch tokens, ten ships,
and ten fields: 491,520 environment transitions per update. Unlike the small
standalone benchmarks, this trainer used its default ten Torch CPU threads.
One warmup update took 578.93 seconds; the first completed measured update took
215.94 seconds and completed two PPO epochs, approximately **2,276 environment
transitions/s**. This is a single completed update, not a repeated estimate or
a comparison with the historical eight-ship result. Warmup includes work beyond
compilation and is not a standalone compiler-startup measurement.

The user requested stopping intensive work during the next update. The process
was terminated, so the profiler did not emit its final aggregate JSON or phase
table. The [preserved partial log](performance-audit/logs/performance-audit-gpu-ppo-partial.txt) is the
evidence; no current end-to-end training speedup or phase share is established.
This also illustrates why environment-only throughput must not be presented as
complete training throughput.

Queued renderer ablations, CPU Warp comparisons, the additional perception
test-suite run, a corrected primary-kernel profile, and a current-version
whole-tick compile experiment were cancelled before execution. Their harness
code is provided for later work, but these are **unvalidated experiments**,
not measured improvements. Completed LOS parity checks reported above remain
the correctness evidence for the tested kernels.

Example commands for a later authorized session, from the repository root:

```bash
.venv/bin/python benchmarks/performance_audit.py --mode profile \
  --scenario 50v50 --steps 8 --out /tmp/profile-50.json
TORCHINDUCTOR_CACHE_DIR=/tmp/bnb-audit-inductor .venv/bin/python \
  benchmarks/performance_audit.py --mode los --scenario 50v50 \
  --steps 10 --pairs 3 --out /tmp/los-50.json
TORCHINDUCTOR_CACHE_DIR=/tmp/bnb-audit-inductor .venv/bin/python \
  benchmarks/performance_audit.py --mode realtime --scenario 5v5 \
  --steps 100 --pairs 5 --policy-compile default --out /tmp/realtime-5.json
.venv/bin/python benchmarks/performance_audit.py --mode throughput \
  --scenario 5v5 --batch 32 --steps 50 --out /tmp/env-b32.json
```

GPU follow-up uses the existing `realtime_latency.py` and
`rl_pipeline_profile.py` with identical workload configurations for each arm.
The prototype harness supports `--device cuda` outside the sandbox, synchronizes
completion for latency, and synchronizes only window boundaries for environment
throughput. `--mode profile` remains CPU-only. Optional `--warp-path` enables
the isolated dependency and `--backend warp` selects it for full-path experiments.
