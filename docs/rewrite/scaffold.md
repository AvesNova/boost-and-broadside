# PyTorch rewrite: scaffold plan

Status: agreed in part, 2026-10-31 (see §11, Decisions). Nothing here is built. Items marked **[decide]** were
open when drafted; §11 records how each was settled.

## 1. What the scaffold is

The scaffold is the part of the rewrite that is designed as a whole before any section is
written:

- the tree (one directory per seam);
- the contracts between sections (types, shapes, dtypes, who owns what);
- the runtime rules (compile and CUDA-graph boundaries, memory, seeding, devices);
- a walking skeleton: the full training loop running end to end, with every section a
  small stub, measured from day one;
- the test and benchmark harness that every section plugs into.

Sections are then replaced one at a time (spec, agent, blind review, integrate), and every
replacement is measured in the real loop, not in isolation.

## 2. Inputs carried over

**From the decision log** (`docs/rewrite/decisions.md`, branch `jax-rewrite/design`).
These seams are framework-neutral and carry over unchanged:

- env = rules only; emits events, not rewards;
- `observe(state, team)` is raw and team-canonical; `privileged(state)` is a separate type,
  reachable only from learner batches;
- controllers map to ships; every controller is an agent; each agent owns an opaque carry,
  its feature encoding and its auxiliary losses;
- PPO needs only log-prob, entropy and value;
- N teams from the start;
- fixed counts per run;
- metres;
- physics tick = decision tick;
- time-native rules are exact multiples of the decision period;
- one root seed;
- step-count schedules only;
- nats-per-token losses;
- single GPU;
- frozen-dataclass configs;
- naming canon (ship, team, controller, lineup, view, carry, match, decision step, tick).

**From the race** (`physics-race*`):

- The physics section exists. It is the PyTorch arm (06ec7e3), once its review list is
  fixed.
- Four lessons from the arm:
  - pure functions of tensors;
  - a team-contiguous layout, so the enemy is a view, not a mask;
  - staged `torch.compile` regions, which made 5× the difference that Inductor options did;
  - CUDA graphs over many ticks.
- Process lessons:
  - size the measurement first;
  - give each agent its own scratch directory;
  - use the lock pass-through;
  - set tolerances from a float32 analysis;
  - benchmark at steady state and at random sizes;
  - prefer general speed over a single headline number.

**Translated from JAX to PyTorch** (the log assumed JAX, so these change):

| JAX plan | PyTorch scaffold |
|---|---|
| `jit` one update | Compiled **stages** + CUDA graphs, chosen by the runtime (§5) |
| pytrees | frozen dataclasses of tensors, registered with `torch.utils._pytree` |
| `fold_in(root, name)` | one `torch.Generator` per named stream, seeded `hash(root, name)` |
| static vs traced config | Python numbers (graph-specialized) vs 0-d tensors (tunable, no recompile) |
| `memory_analysis()` | one measured probe per config at startup (allocator peak after one update) |
| `jax.export` | `torch.export` / ONNX for model exchange, later |
| JaxMARL adapter | dropped, or a PettingZoo adapter later |

**Superseded by the race:** the log planned "physics changes land in the current env
first". The new flight model now exists fresh in the PyTorch arm, so that step is dropped.

## 3. Design rules (speed and maintainability together)

1. **Pure inside, mutable only at the edge.**
   - Sections are pure functions: tensors in, tensors out.
   - Only `runtime/` writes into persistent buffers (`copy_` into the carry).
   - This removes the old blocker where dynamo dropped writes onto caller-owned objects,
     and keeps every section testable on CPU in eager mode.
2. **Sections don't compile themselves.**
   - No `torch.compile` and no `torch.cuda` calls in section code.
   - The runtime decides the stage cuts, and they are listed in one file. The race showed
     that the cut points matter more than any kernel trick.
   - **Rationale:** a section's speed is measured in the loop, where its neighbours are.
3. **No process-global settings in library code.**
   - Leave TF32, thread counts, recompile limits, global seeds and guard skipping alone.
   - The entry point sets them once, from config, and logs them.
4. **One code path for CPU and GPU, at any batch size.**
   - Realtime play is the same `step` at B = 1 on CPU, compiled with the C++ backend.
   - No preset-gated or size-gated algorithm branches unless the same benchmark run
     ablates them (lesson: general speed).
5. **Fixed shapes per run; preallocate everything.**
   - Allocate the rollout buffer, carry and observation once.
   - No allocation inside the captured region.
6. **Struct-of-arrays state with a team axis.** Ship tensors are `(B, T, S, ...)` with an
   `active` mask, so N teams and the enemy view are free.
7. **Dtypes:**
   - float32 state (positions included; the race showed float32 is enough with
     float32-aware tolerances);
   - int32 counters, bool masks;
   - bf16 only inside models;
   - float64 only in the reference oracle and offline tools;
   - the global step lives on the host as a Python int.
8. **Contracts are types, versioned.**
   - Each contract is a frozen dataclass with field docs: shape, dtype, unit.
   - Each carries a `SCHEMA_VERSION`.
   - A contract test checks shapes and dtypes for every implementation.
9. **Client-portable models (author, 2026-10-31: agents and rendering on the client's
   GPU).**
   - Every model's single-step forward must export with `torch.onnx` (dynamo exporter).
     The client runs agents in ONNX Runtime: Windows ML picks the vendor backend on
     Windows, CoreML on Mac.
   - The client sim stays on CPU torch.
   - Rendering reads snapshots on the GPU through OpenGL or wgpu, independent of torch.
   - A contract test checks that the export matches the eager step.
10. **Names in full words** (author, 2026-10-31). Variables, functions, classes and
    contract columns use clear, unambiguous words. Universal abbreviations (`cos`, `x`,
    `y`) are fine; spec symbols (`c`, `w`, `r`, `E`, `gx`) are not.
11. **No legacy at the end** (author, 2026-10-31). The finished rewrite has no dead code,
    shims or stale docs from v1 or the race unless kept by an explicit decision.
    Compatibility layers (such as the race-layout test adapter) are temporary: once
    something is proven, native tests replace them.
12. **Small files with one job.**
   - Names follow the spec.
   - Comments say *why*; the spec says *what*.
   - The old tree is 46k lines; the scaffold sets no target, but each section review
     reports its size.

## 4. Tree

Same repo, fresh branch (decision log, "Option 3"; see §9). Package `bnb` (**[decide]**
name). Top-level layout mirrors the seams:

```
src/bnb/
  contracts/      # types only, no logic: config, state, obs, action, events, trajectory,
                  #   privileged labels, carry protocol; SCHEMA_VERSION per type
  game/           # pure rules; imports contracts only
    physics/      #   medium (fields), flight, bullets, combat, spawn   <- race arm
    perception/   #   sight, observe (raw, team-canonical), privileged
    modes/        #   rule components: zones, capture, win, respawn; events -> rewards
    step.py       #   reset / step / observe composition
  controllers/    # Controller protocol + scripted, random, human; the model adapter
  models/         # model zoo; each model owns its feature encoding, carry, aux losses
    mlp/          #   skeleton baseline
    yemong/       #   later
  learn/          # learners: ppo, bc; labels (shot re-simulation, next-state targets)
  runtime/        # the only stateful code: buffers, carry, stage compilation, CUDA graphs,
                  #   rollout collector, match engine (who controls which ship), seeding
  league/         # pool, opponent sampling, live ratings, eval matches
  play/           # shell: window, input, render snapshots, replay, spectator (CPU)
  diagnostics/    # metric registry, wandb sink, dumps for crash replay
  config/         # presets (Python), CLI (tyro)
bench/            # in-tree harness: lock, sizes, steady-state cells, budgets (race_harness)
reference/        # float64 numpy oracle(s), outside the package; tests only
tests/
  contract/       # every implementation of every contract: shapes, dtypes, conformance
  reference/      # section vs oracle, goldens + mutants
  sections/       # unit tests per section
  loop/           # skeleton end-to-end, toy learning tests
```

Dependency rule, checked by a test that walks imports:

```
contracts <- game <- controllers/models <- learn <- runtime <- league <- play
```

Nothing imports `runtime` except entry points. `play` uses `runtime` but adds no rules.

## 5. Runtime: stages, graphs, memory

**Stages.** The unit of compilation is a named pure function from carry to carry. The
skeleton's tick is:

```
act        (controllers + models forward, sample)        compiled
advance    (fire, fly, bullets fly)                      compiled   <- race "move"
resolve    (hits, damage, clocks, death, respawn, reset) compiled
perceive   (sight, observe, privileged)                  compiled
record     (write the step into the rollout buffer)      compiled or eager copy_
```

- Each stage is `torch.compile(fullgraph=True, dynamic=False)`. The runtime calls it and
  `copy_`s the outputs into the carry.
- K ticks are captured as one CUDA graph. K is the largest divisor of the rollout length
  that fits; the race arm used 32.
- The learner update is its own stages: GAE, minibatch forward/backward, optimizer. It is
  graph-captured if measurement says it helps.
- The cut list lives in `runtime/stages.py`. Moving a cut is a one-line change plus a
  benchmark.

**Memory.**
- Every buffer is allocated at startup from the config.
- A probe at startup runs one update and records the allocator peak. It fails fast if the
  peak is over budget, rather than OOMing hours in.
- Microbatching is a loop inside the update.

**Seeding.**
- Named streams: `env.reset`, `env.spread`, `act.<controller>`, `league.sample`, and so on.
- Each stream is a `torch.Generator` seeded from `hash(root_seed, name)` and registered
  with the CUDA graph, so replays draw fresh numbers.
- The race checked that replays draw fresh noise with the default generator. The skeleton
  repeats that check for named generators.

**Crash replay.**
- Keep the last few updates' inputs in host RAM: params, optimizer state, carry,
  generator states.
- On error, dump them for an offline eager replay.

**Overlap.** The env step and the network forward are separate stages. Running them on
separate streams is a later experiment, not a scaffold rule.

## 6. Contracts (first cut; each becomes a section-0 spec)

Shapes: B envs, T teams, S ships per team, F fields, P bullets per ship.

- **Config:**
  - `GameConfig` (static: counts, dt, mode components);
  - `ShipTypes` (per-type parameters as tensors: types are data, from race v2);
  - `RunConfig`.
  - Static values are Python numbers; tunable coefficients are tensors.
- **State:**
  - `GameState`: ships `(B,T,S,·)`, bullets `(B,T,S,P,·)`, fields `(B,F,·)`, clocks.
  - Opaque to everything outside `game/`, except `runtime` (which copies it) and
    `play` (which reads render snapshots through a pure `snapshot(state)`).
- **Action:**
  - `Action(B,T,S,…)` integer components: turn, thrust/energy, fire.
  - **[decide]** factored ints (race: 3 ints) or the decision log's 42-way joint command.
  - Recommendation: factored in the contract; a model may use a joint head internally and
    convert.
- **Events:**
  - `Events` per decision step: damage, kills, deaths, fired, captures, outcome.
  - Summed over ticks if repeat > 1.
  - `modes/` maps Events to rewards.
- **Observation:**
  - `Obs` is raw, team-canonical (self team = 0), with masks, visible flags and lifecycle
    flags.
  - It carries no model features.
  - The schema follows the race's 16/17 columns plus the global token. Finalized with the
    perception section.
- **Privileged:** truth for labels. A separate type; `Controller.step` cannot accept it.
- **Controller:**
  - `init_carry(B) -> Carry`;
  - `step(params, carry, obs, gen) -> (action, carry, outputs)`;
  - `unroll(params, carry0, obs_seq) -> outputs` (conformance test: `unroll == loop(step)`).
  - `outputs` holds logp, entropy, value, and optional `inspect`.
- **Learner:**
  - `update(params, opt_state, batch) -> params, opt_state, metrics`;
  - `aux_loss` hook on the model.
- **Trajectory:**
  - a fixed-shape rollout buffer of obs, action, events, outputs and privileged;
  - plus delayed labels, filled by `learn/labels`.
- **Lineup:**
  - which controller owns which `(team, ship)` slots: data, not code;
  - the runtime gathers actions per slot.

## 7. Walking skeleton (milestone 0)

Everything runs end to end, nothing is good yet.

| Section | Skeleton stub | Replaced by |
|---|---|---|
| physics | race torch arm as is (it already meets the contracts' shape) | arm + review fixes |
| perception | race arm's sight + observe | perception section |
| modes | team deathmatch: reward = kills − deaths (zero-sum) | modes section |
| controllers | random + race's chaser | scripted agent section |
| model | 2-layer MLP over the raw obs, factored action heads | Yemong (and others) |
| learner | plain PPO, no aux heads | PPO + BC sections |
| league | self-play only, no ratings | league section |
| play | headless match runner printing events | play shell |

**Milestone 0 is done when:**
- `bnb train --preset skeleton` runs for a fixed number of updates at 3 sizes;
- the contract tests pass;
- the MLP measurably beats random on a toy check (win rate after N updates);
- the import-direction test passes;
- a small, pre-sized benchmark records SPS, update time and peak VRAM in under 5 minutes.

**The benchmark is part of the scaffold:**
- fixed cells: 3 sealed random sizes × 1 batch budget, plus CPU realtime p99 at one size;
- one run each, after a one-episode warm-up;
- results appended to `bench/results.jsonl` with the commit.

Every later section reports against this table.

### Milestone 0 status (2026-10-31): built

Code in `src/bnb`; how it differs from the contracts text is in the contracts doc's
skeleton notes.

Done criteria:

| Criterion | Result |
|---|---|
| Runs end to end at 3 sizes | yes (`bnb` CLI; benchmark cells) |
| Contract tests | pass (`tests/contract`, including the import direction and ONNX export) |
| Race reference tests on the ported physics and the new observation | 61 pass |
| Toy learning | **changed:** the MLP learns to stop dying against the chaser (−0.05 → −0.01 reward/s; random scores −0.05). It does not learn to hunt random ships within 500 updates, so "beats random" was not met; hunting is left to the learning sections (BC) |
| Benchmark | 127 s with a warm compile cache; a cold run is a few minutes |

Benchmark, commit `4cc7c13c`, sealed sizes (seed 20261031), 40,960 ships per batch,
RTX 4070 Laptop:

| Size | Rollout env-ticks/s | Train env-ticks/s | Learn per update | VRAM |
|---|---|---|---|---|
| 114 ships, 22 fields, R 9654 m (360 envs) | 71k | 17k | 2.0 s | 4.4 GB |
| 40 ships, 8 fields, R 6146 m (1024 envs) | 407k | 56k | 2.0 s | 4.4 GB |
| 26 ships, 54 fields, R 4950 m (1576 envs) | 323k | 76k | 2.0 s | 4.8 GB |
| CPU realtime, 10 ships | p50 0.15 ms, p99 0.17 ms | | | |

Notes:
- The learner dominates training throughput. The stub's PPO makes 4 epochs over 1.5
  views per env, in fp32 with TF32 off. The cost is the same at every size because the
  ship budget is fixed.
- At the largest map the teams spawn about 15 km apart and never meet within a 60 s
  episode, so that load has no hits. Dense pair tests cost the same either way, but a
  combat-load check needs scripted fleets.

Not yet in the skeleton:
- named, graph-registered random generators;
- the determinism cost measurement;
- crash replay;
- fog-off envs and fog share;
- mode state (the attribution ledger);
- PPO microbatching;
- the TF32 decision;
- the spawn seat leak, fixed in the physics section.

## 8. Section order and acceptance

| # | Section | Acceptance type |
|---|---|---|
| 0 | contracts + skeleton | contract tests, toy learning, benchmark baseline |
| 1 | game: physics, perception, spawns, maps (arm + review fixes + decided changes; `sections/game.md`) | race oracle + goldens + mutants (updated for the changes) |
| 1b | ship card: handles to physics parameters (offline solver) | round-trip and independence tests against the forward model |
| 3 | modes + events | rule tests per component; zero-sum property tests |
| 4 | scripted agent | behavioural tests; Elo vs today's scripted, as a sanity range |
| 5 | model (Yemong) | `unroll == loop(step)`; parity with today's forward where specced |
| 6 | PPO + labels | toy-learning tests; label equivalence (resim matched bit for bit before) |
| 7 | BC | toy test; pilot run against the BC calibration record |
| 8 | league + ratings | rating recovery on synthetic matches; sampling determinism |
| 9 | diagnostics | each metric against a hand-computed case |
| 10 | play shell | manual playtest checklist; packaged Windows match (early, per log) |
| — | final pass | whole-tree review for fit; training comparison vs today's best run |

Sections 2–4 can be specced in parallel; agents run one at a time on the GPU lock.

**Game design gate.** The open items (flight-model.md 7–10: battery, gravity, full
battery; bases, turrets, roles):
- block only the section that owns them;
- otherwise the spec makes them configurable components with the feature off.
- Recommendation: physics v1 = the race spec; bases and turrets come later as mode
  components (`modes/`), with shared hit code.

## 9. Repository mechanics

**[decide]** Recommendation: the decision log's option 3, in the same repo.
- Tag the current state `v1-torch-final` on `feat/resim-shot-labels`.
- Branch `v2` with the old `src/`, `tests/` and `scripts/` removed.
- Bring over:
  - `docs/rewrite/`, from `jax-rewrite/design`; that merge conflict is still open, and
    only the docs folder is needed;
  - the race harness, oracle and goldens, as `bench/` and `reference/`;
  - the torch arm's `sim/`, as `game/physics` + `game/perception`.
- Merge to `main` when v2 is playable.
- The old tree stays reachable through the tag and a worktree.

The alternative is a new repo:
- cleaner history and access control for agents;
- loses the shared issue and docs history.

Not recommended.

## 10. Process per section (from the race)

1. Spec (math and contracts only), written and owned by the author with me. Tolerances
   come from a float32 analysis.
2. A reference plus goldens, perturbation-screened, plus a mutation check. For learning
   sections: toy tests plus an equivalence check against today.
3. An agent implements it in its own worktree and its own scratch directory, benchmarking
   in the skeleton loop under the lock (pass-through env var).
4. Blind review (anonymized bundle). The review reports size, test coverage and stage
   placement.
5. Integrate, rerun the skeleton benchmark, append results.
6. Pre-size every benchmark. One run; cells that answer the question only.

## 11. Decisions (author, 2026-10-31)

- **Framework:** PyTorch, for now. The framework question was reopened and closed again;
  the proposed tests were not run.
- **Repo:** option 3, same repo, `v2` branch, tag `v1-torch-final`. Only `docs/rewrite/`
  comes over from the design branch.
- **Package name:** `bnb`.
- **Dependencies:** the latest that everything supports, pinned in `uv.lock`:
  - Python 3.14 (onnxruntime 1.31.1 has no 3.15 wheels);
  - torch 2.15.0+cu132 (driver 595.99 supports CUDA 13.2; cu134 needs a newer driver);
  - numpy 2.5, onnxruntime 1.31, wandb 0.31, tyro 1.0, moderngl 5.13, pygame-ce 2.5.

  Bump deliberately, with the skeleton benchmark as the check.
- **Physics:** the new physics (the race spec: rigid-body nose, energy store, refractive
  fields, ship types).
  - flight-model.md items 7-10 (battery, full battery, gravity, full-stick sink) are
    confirmed or changed when the physics section is specced.
  - Capsule fields, turrets and bases come later, as components.
- **Lineups:** fixed-ratio env pools: X% self-play, X% per league slot, X% scripted.
  League slot parameters are swapped by `copy_`.
- **Teams:** equal sizes. Layout (B, T, S) with T = 2 for now.
- **Mutation:** sections are pure. The learner owns parameters and optimizer state and
  uses plain PyTorch optimizers in place ("don't rewrite PyTorch").
- **Decision-log seams:** all carried over (§2, items 1-20 of the review), with one change:
  - **Team canonicalization relabels the team id only, never geometry.**
  - **Agents never see the real team id** (author, revised 2026-10-31):
    - the observation carries `team_rel = (team - viewer_team) mod T` only, so the
      viewing team is 0;
    - the real `team` lives in the game state and `Privileged` only.
  - **Seats are indistinguishable by construction:**
    - Map generation must leave no way to tell which side you are on: geometry symmetric
      under the map's symmetry, and every side-specific role randomized per episode.
      Example: which team gets Frontline's clockwise direction.
    - Tests:
      - the distribution of observations is the same from either seat;
      - eval checks a trained model scores the same from either seat.
- **Still open, for section 0:** the action contract (separate integers vs the 42-way
  joint command).
