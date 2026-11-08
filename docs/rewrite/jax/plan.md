# The JAX version: plan

Status: **agreed with the author 2026-11-08; not started.** Waiting on the round-2 winner
and the v2 freeze (see "Before the first task"). Progress is recorded in
[progress.md](progress.md), not here. This file changes only when a decision changes, and
every change is dated.

Any agent picking this up: read this file, then `progress.md`, then the brief for your
task. If something here contradicts an older document under `docs/rewrite/`, this file
wins for the JAX work.

## Why

- **Compile time.** v2's `torch.compile` stages now take 20-30 minutes, which kills
  iteration. In the physics race, JAX compiled about 100x faster.
- **Speed.** v2 is still largely launch-bound. JAX was clearly faster in the race's own
  tests.
- The author originally chose PyTorch because the change looked smaller. That cost has
  now come back as compile time. Token budget is not a constraint.

This reverses the 2026-10-31 framework decision ([../decisions.md](../decisions.md),
[../scaffold.md](../scaffold.md) §11) for the JAX work.

## Decisions (author, 2026-11-08)

### Scope

- **v2 is frozen first**, at a tag (proposed `v2-torch-final`, like `v1-torch-final`).
  The PyTorch v2 stays runnable as the yardstick and the live oracle.
- **Only the round-2 winner is ported**, not every experiment arm or switch. The arms'
  digests (`tests/sections/arm_goldens.py` and its goldens) are not ported: they exist
  only for the arms comparison, which finishes before the port starts.
- **Linux only for now.** Human play on Windows (or WebGPU, or similar) is a far-future
  want: keep it in mind, don't build for it.
- **No ONNX export, and no loading of PyTorch checkpoints.** ONNX comes after a strong
  agent and a polished game.
- **Hardware:** one GPU. Benchmarks on the author's RTX 4070 Laptop; training on a
  rented GPU picked by spot price. Micro-batching keeps batch semantics the same on both.

### Precision: no bitwise exactness anywhere

- **Nothing in the code, the tests or the agents' work aims for bitwise equality or
  float64 agreement.** Different summation order, atomic adds and the last few bits
  differing are all fine.
- **Float32 is the working precision.** Norms (RMSNorm especially), reductions, losses
  and anything that accumulates small values over a long run stay in float32, never
  bf16. bf16 matmuls stay an option where v2 has one.
- **Integers, counts, masks and discrete outcomes compare exactly.**
- **"Stochastically the same" is the bar for anything chaotic or random.** The system is
  chaotic, so one-bit differences grow; long trajectories are compared by statistics,
  not step by step. Seeds must give equivalent maps, models and so on, not identical
  bits.
- **The reference goldens pass at float32 tolerance**, run in float32. Where a golden set
  has only a float64 tolerance today, a float32 tolerance is set from a float32 error
  analysis. That is a one-time loosening of a fixed test file, and needs the author's
  sign-off, recorded in `progress.md`.

### What "done" means

1. **The reference goldens pass** at float32 (`tests/reference/`, through a JAX adapter;
   the `REFERENCE_ARM` mechanism already supports another implementation).
2. **Every v2 test becomes a JAX test**, unless it is PyTorch-specific or over-strict in
   the sense above. Over-strict tests are softened rather than dropped where they still
   test something real.
3. **Faster than v2 on both:**
   - steps per second;
   - compile time, both cold and with a warm on-disk compilation cache.

   Measured on the 4070 Laptop against the baseline recorded at the freeze.
4. **Training matches:** the JAX learner reaches v2's Elo against the fixed scripted
   ladder, on the same Frontline-plus-deathmatch mix, at about 50M steps, over a few
   seeds, within a margin set before the run. (v2 reaches close to 2000 Elo at about 50M
   steps with little RL.)

### Libraries

- **Equinox** for models (author, 2026-11-08; resolves item 6 in the decision log).
  - Why: a module is a frozen dataclass pytree, the same idea as v2's contracts
    (`contracts/tree.py`); v2's `nn.Module` classes translate one to one; stacking league
    slots for `vmap` and stacking layers for `lax.scan` are easy; random keys are
    explicit; the API is small and stable.
  - Flax NNX was the earlier lean. It was rejected because its advantage, mutating
    modules, is unused (Yemong never mutates its modules) while its API churn and
    split/merge machinery are paid for.
- **Our own small layer set (about 100 lines)**, not `eqx.nn`: `Linear` acting on the
  last axis with weights stored `(out, in)` (the Yemong goldens' layout), RMSNorm always
  in float32, attention, causal conv. This avoids `eqx.nn`'s one-vector convention and
  keeps the precision rules in one file.
- **optax** for optimizers.
- **Checkpoints:** safetensors plus JSON, keyed by pytree key paths. No Orbax.
- **jaxtyping** for shape annotations.
- **Pinned versions** in `uv.lock`, bumped deliberately with the skeleton benchmark as
  the check.
- **No hand-written kernels at first.** Pure JAX and XLA first. Pallas kernels (replacing
  v2's Triton kernels) are a finishing-touch experiment, kept only if they buy a clear,
  measured improvement.

## Design rules for JAX code

These go into the skeleton and every brief, and the reviewer checks them:

1. **Pure functions of pytrees**, as in v2's rule "pure inside, mutable only at the
   edge". State in, state out; nothing mutated.
2. **One `jit` per natural unit** (a rollout, an update), not many small staged regions.
   No `jit` created per call.
3. **Loops are `lax.scan` / `lax.fori_loop`** over ticks, decision steps and rollout
   steps. No Python loops inside traced code.
4. **Identical layers are stacked and scanned, not unrolled.** This is the main lever on
   compile time.
5. **Static config is static and traced config is traced**, so nothing recompiles when a
   tunable value changes. A test counts compiles after warmup and fails on any
   recompile.
6. **No host syncs inside a traced unit** (no `.item()` equivalents, no `device_get`).
7. **Buffers that are replaced are donated** (`donate_argnums`).
8. **Fixed shapes.** Death, respawn and padding are masks, not shape changes.
9. **No float64** in any training or play path.
10. **Random keys:** one root seed, one stream per name via `fold_in` (v2's rule, in its
    JAX form).
11. **The on-disk compilation cache is on** for every entry point.

## Process

### Roles

- **Coordinator.** Holds this plan, writes the briefs, integrates, runs the gates,
  updates `progress.md`. Reads summaries and reports, not diffs. Does not write section
  code.
- **Author.** Signs off at four gates: the brief, any loosened tolerance, the review
  verdict, the merge.
- **Oracle agent** (before each implementer). Builds the checks the implementer is held
  to, and commits them before the implementer starts. After that they are fixed:
  - differential tests against live v2 PyTorch on the same numpy inputs, at float32
    tolerance, for parts without goldens;
  - statistical tests for anything random or chaotic (inject the same random draws, or
    compare distributions, e.g. scripted-versus-scripted match outcomes);
  - the ported v2 tests, softened only with the author's approval;
  - a mutation check: each new test must fail on deliberate bugs, as v2's goldens do.

  **Whoever writes the code never writes its checks.**
- **Implementer.** Sees v2's code and specs, and works from the design rules above, so it
  translates intent rather than line by line. Own worktree, own scratch directory. Runs
  sanity benchmarks only. Fixes its own review findings. If a second review round still
  fails, or the problem is in the design, a fresh implementer takes over from the notes.
  At most two review rounds before it goes back to the author.
- **Reviewer, blind to v2's code.** Sees the spec, the oracle tests, the JAX diff and the
  benchmark numbers. It judges JAX-nativeness and fidelity to the spec, not resemblance
  to PyTorch. It reports size, test coverage, and a checklist pass against the design
  rules.
- **Integration and official benchmarks: one place only.** On the 4070 Laptop, under v2's
  benchmark lock (`bench.lock.bench_lock()`), so parallel agents don't spoil each other's
  numbers. Each section ships with a speed figure and a compile-time figure, and any
  slowdown blocks the merge.
- **Consistency audits** across the whole tree after the game section and after the
  learner, not only at the end.

### Races on the two hot paths

For the **physics tick (1a)** and the **Yemong network trunk (2a)**:

1. Two implementers build it independently and blind to each other, against the same
   fixed oracle.
2. A judge agent reads both implementations and both benchmark results. It picks the
   better one, or combines the best of both into one hot path.
3. The combined version goes through the normal review and integration.

### Task size

A task is **one directory with one owner, checked by its own oracle, and reviewable in
one pass**: roughly 1-3k lines of source plus tests. Larger sections split along their
internal seams and run one after another when they share files.

## Tasks, in order

| # | Task | Oracle | Notes |
|---|---|---|---|
| 0 | **Skeleton.** Contracts as Equinox/pytrees; random-key streams; static vs traced config; the rollout loop with a random controller; benchmark harness measuring compile time cold and with a warm cache; a test that counts recompiles; a numpy bridge to v2 for differential tests; the float32 golden-test setup; JAX version pins | Differential tests against v2's runtime with a random controller; author's design review | First, and with extra care: every speed and compile-time rule is set here |
| 1a | Physics | Race goldens (`tests/reference/golden`) | **Race** |
| 1b | Perception and observation | Differential tests | |
| 1c | Spawns, maps, modes, rewards | Rewards goldens + differential tests | |
| 1d | Scripted agent | Statistical match outcomes | |
| 2a | Yemong network: layers, spatial, temporal | Yemong goldens | **Race** |
| 2b | Yemong heads, codes, belief, controller | Yemong goldens + differential tests | |
| 3a | PPO, losses, labels | Learner goldens | |
| 3b | Rollout buffer, behaviour cloning | Differential tests; statistical BC test | |
| 4 | League and ratings; runs and checkpoints; figures; play shell | League goldens; ported tests | The play shell's GL/imgui code is mostly framework-neutral |
| — | Final gate | Training run, as in "What done means" item 4 | |

Order: the skeleton first, then the game, then the model, then the learner, then the
rest. Once the skeleton is merged, game and model tasks may run in parallel (the model
goldens don't need the game). The learner waits for both.

## Before the first task

1. The author picks the round-2 winner; the arms comparison finishes.
2. Freeze v2 at a tag; record it in `progress.md`.
3. Measure the baseline on the 4070 Laptop (steps per second; compile time cold and with a
   warm cache) and record it in `progress.md`.
4. Write the skeleton brief (task 0) and get the author's sign-off.

## Keeping the record

- **`progress.md` is the source of truth for where the work stands.** If an agent dies,
  the next one starts from it.
- Every agent updates it at each milestone (brief signed off, oracle committed,
  implementation pushed, review verdict, merge, benchmark), **commits and pushes
  immediately**, and does not batch updates up for later.
- Per-task working notes go in `docs/rewrite/jax/tasks/<task>-notes.md`, as v2's sections
  did. `progress.md` links to them.
- Decisions go in this file with a date; if a decision changes, the old text is struck
  through or marked superseded, not silently rewritten.
