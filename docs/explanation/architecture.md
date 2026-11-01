---
covers: [src/bnb/runtime/rollout.py, src/bnb/runtime/realtime.py, src/bnb/contracts/tree.py, src/bnb/game/ship_types.py, tests/contract/test_imports.py]
---

# Architecture

## Layers and the dependency rule

The package `bnb` is cut into layers, one directory per seam between sections. A module
may import from its own layer and the layers below it, never from a layer above:

| Layer | Package | Holds |
|---|---|---|
| 0 | `bnb.contracts` | the types shared between sections: shapes, dtypes and meanings, no logic |
| 1 | `bnb.game` | the rules: physics, perception, modes, the batch layout |
| 2 | `bnb.controllers`, `bnb.models` | scripted controllers and model adapters; models and their heads |
| 3 | `bnb.learn` | rewards and learners |
| 4 | `bnb.runtime` | buffers, carries, compiled stages, CUDA graphs, the rollout, the training loop |
| 5 | `league` | opponent pools and ratings (not built) |
| 6 | `play` | the interactive shell (not built) |
| 7 | `bnb.cli` | the entry point |

`tests/contract/test_imports.py` walks every import in `src/bnb` and fails on one that
points up. Nothing imports `bnb.runtime` except entry points.

The rule keeps each section replaceable: a section sees only the contracts and the layers
below it, so it can be specified, built and tested on its own, then measured in the real
loop.

## Pure inside, stateful at the edge

Every section is pure functions of tensors: tensors in, tensors out. The game's step
takes a state and returns a new one; a model's step takes a carry and returns a new one.
Only `bnb.runtime` owns persistent buffers, and it writes them in one way: it copies a
stage's outputs into the buffers with `copy_`. The one other mutation is the learner's: it
owns the model's parameters and its optimizer state, and updates them in place with a
plain PyTorch optimizer.

- Section code runs eagerly on the CPU, so every section is testable without a GPU and
  without compilation.
- The same code compiles: nothing in a section writes into an object the caller owns,
  which `torch.compile` could drop.
- Section code makes no `torch.compile` or `torch.cuda` calls and sets no process-wide
  options (TF32, thread counts, seeds, recompile limits). Entry points set those, once.

The contract types are frozen dataclasses registered as pytrees
(`bnb.contracts.tree.contract`), so the runtime maps `copy_`, `clone` and `to` over a whole
state, observation or carry without knowing its fields. No contract boundary passes a dict.

Configuration splits by how it reaches compiled code:

- **Static:** sizes and shared constants are Python numbers (`bnb.contracts.GameConfig`,
  `bnb.game.rules.Rules`, `bnb.game.modes.rules.ModeRules`). A compiled graph specializes
  on them; a different value is a different graph.
- **Tunable:** the ship card (`bnb.game.ship_types.ShipTypes`) is tensors, so new values
  change no graph. A sweep can give every environment its own card.

## Compiled stages

The unit of compilation is a **stage**: a pure function of the rollout's carry whose
outputs the runtime copies back in. `bnb.runtime.rollout.Rollout` cuts one tick into these stages:

| Stage | Does |
|---|---|
| act | every controller chooses its actions; the tick's random draws; the learner's views are recorded |
| advance | physics phases 1 to 3: fire, ships fly, bullets fly (`bnb.game.api.move`) |
| resolve | phases 4 to 8: hits to the episode end (`bnb.game.api.resolve`), the rewards and their ledger, the seat redraw at resets |
| perceive | visibility and the observation of the new state (`bnb.game.api.perceive`) |

Each stage is `torch.compile(fullgraph=True, dynamic=False)`. The cut list is the tuple of
stages in `Rollout`; moving a cut is a one-line change and a benchmark. The physics race
found that where the cuts fall changed throughput more than any compiler option did.

The PPO loss is compiled the same way, separately (`bnb.learn.ppo.PPO`).

## CUDA graphs

On a GPU the rollout captures a block of ticks as one CUDA graph and replays it:

- The block is the largest divisor of the rollout length up to the rollout's
  `graph_ticks`.
- Capture runs warm-up ticks on a side stream, which compiles every stage, records the
  block, then restores the carry, so capture changes nothing.
- A rollout of `length` ticks is then `length / block` graph replays.

Two rules keep the captured graph correct:

- **No host copies inside a tick.** The game reads constant tables as chains of selects
  (`bnb.game.ship_types.per_ship`), not by copying from the host.
- **Weights change in place.** The learner updates the model's parameters in place, so the
  graph always reads the current weights. A league slot's weights are copied in.

Random draws come from torch's default generator, which a captured graph draws afresh on
every replay (`tests/sections/test_game_section.py::test_compiled_rollout_graph_replays_draw_fresh_randomness`).

On the CPU the rollout runs the stages tick by tick. The realtime path
(`bnb.runtime.realtime.Realtime`) runs one match at batch size 1 with the whole tick
compiled as one function by Inductor's C++ backend.

## Shapes and layout

- Ships are stored team-major: ship `n = team * ships_per_team + slot`. A `(B, N, ...)`
  tensor viewed as `(B, 2, S, ...)` is indexed by team without a copy.
- Every size is fixed for a run. The rollout allocates its buffers once, when it is built.
- State is float32; counters are int32 and flags bool. Models may use bfloat16 inside.
  float64 appears only in the reference tests.
- There are two teams, a constant (`bnb.game.rules.TEAMS`).

## One code path

The same functions run at any batch size, on the CPU or the GPU, eagerly or compiled. No
algorithm branches on a preset or a batch size unless the same benchmark run ablates the
branch, so a reported speed-up holds in general and not at one size.
