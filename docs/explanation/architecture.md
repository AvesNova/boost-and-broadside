---
covers: [src/bnb/runtime/rollout.py, src/bnb/runtime/realtime.py, src/bnb/contracts/tree.py, src/bnb/game/ship_types.py, tests/contract/test_imports.py]
reviewed: 2026-11-02
---

# Architecture

## Layers and the dependency rule

The package `bnb` is cut into layers, one directory per seam between sections. A module
may import from its own layer and from lower-numbered layers, never from a higher one.
The layers come from `LAYER` in `tests/contract/test_imports.py`, which walks every import
in `src/bnb` and fails on one that points up:

<!-- generated: packages -->

Nothing imports `bnb.runtime` except entry points.

The rule keeps each section replaceable: a section sees only the contracts and the layers
below it, so it can be specified, built and tested on its own, then measured in the real
loop.

## Pure inside, stateful at the edge

Every section is pure functions of tensors: tensors in, tensors out. The game's step
takes a state and returns a new one; a model's step takes a [carry](glossary.md#carry) and
returns a new one.

- **Only `bnb.runtime` writes persistent buffers,** in place, between stages. The one other
  mutation is the learner's: it owns the model's parameters and its optimizer state, and
  updates them in place with a plain PyTorch optimizer.
- **Section code runs eagerly on the CPU,** so every section is testable without a GPU and
  without compilation.
- **The same code compiles:** nothing in a section writes into an object the caller owns,
  which `torch.compile` could drop.
- **Sections do not compile themselves,** call `torch.cuda` or set process-wide options
  (TF32, thread counts, seeds, recompile limits); the runtime chooses what to compile, and
  entry points set process-wide options once.

The data types that cross sections inside a tick are frozen dataclasses of tensors,
registered as [pytrees](glossary.md#pytree) (`bnb.contracts.tree.contract`): the
observation, the privileged truth, the action, the events, a controller's outputs, the
trajectory, the game state and the ship card. The runtime maps copies, clones and device
moves over a whole state, observation or carry without knowing its fields, and no
contract boundary passes a dict. The configuration types (`bnb.contracts.GameConfig`,
`bnb.contracts.Pool`, `bnb.game.rules.Rules`) are plain frozen dataclasses of Python values.

Configuration splits by how it reaches compiled code:

- **Static:** sizes and shared constants are Python numbers (`bnb.contracts.GameConfig`,
  `bnb.game.rules.Rules`, `bnb.game.modes.rules.ModeRules`). A compiled graph specializes
  on them; a different value is a different graph.
- **Tunable:** the ship card (`bnb.game.ship_types.ShipTypes`) is tensors, so new values
  change no graph. A sweep can give every environment its own card.

## Compiled stages

The unit of compilation is a **stage**: a pure function of the rollout's carry whose
outputs the runtime copies back in. `bnb.runtime.rollout.Rollout` cuts one tick into these
stages:

| Stage | Does |
|---|---|
| act | every controller chooses its actions; the tick's random draws; the learner's views are recorded |
| advance | physics phases 1 to 3: fire, ships fly, bullets fly (`bnb.game.api.move`); the learner's shot capture is recorded |
| resolve | phases 4 to 8, hits to the episode end, with the unnumbered mode phase after damage (`bnb.game.api.resolve`); the rewards and their ledger; the seat redraw at resets |
| perceive | visibility, the observation and the privileged truth of the new state (`bnb.game.api.perceive`) |

Each stage is `torch.compile(fullgraph=True, dynamic=False)`. The cut list is the tuple of
stages in `Rollout`; moving a cut is a one-line change and a benchmark.

The learner's stages are cut in `bnb.runtime.train.compiler`: the micro-batch loss and the
shot re-simulation's tick, each compiled whole (see [The learner](learner.md)).

## CUDA graphs

On a GPU the rollout captures a block of ticks as one CUDA graph and replays it:

- The block is the largest divisor of the rollout length up to the rollout's
  `graph_ticks`.
- Capture runs warm-up ticks on a side stream, which compiles every stage, records the
  block, then restores the carry, so capture changes nothing: every draw is keyed by the
  carry (below), and a run resumed in a new process, which captures again, draws what the
  first process drew ([Runs and checkpoints](runs.md)).
- A rollout of `length` ticks is then `length / block` graph replays.

Two rules keep the captured graph correct:

- **No host copies inside a tick.** The game reads constant tables as chains of selects
  (`bnb.game.ship_types.per_ship`), not by copying from the host.
- **Weights change in place.** The learner updates the model's parameters in place, so the
  graph always reads the current weights. The league copies an opponent's weights into its
  slots' stacked parameters in place, between segments.

## Random draws

Every random draw of training, evaluation, behaviour cloning and the rating matches is
keyed (`bnb.streams`): a pure function of the run's seed and of what the draw is for, never
a position in a shared sequence. Arms of an experiment that differ in one setting therefore
draw alike wherever they are alike (common random numbers), and the difference between two
arms within one seed is not buried under unrelated re-rolls.

| Draw | Keyed by |
| --- | --- |
| a parameter's initial values (`bnb.models.initialise`) | the seed, the path of the module that draws them, the parameter's name (a generator on the CPU) |
| an episode's map, arena scale, formations and first spawns; its seat | the seed, the environment, the environment's episode count (the batch number in a rating match) |
| a running episode's tick uniforms (respawns) and the bullet spread | the seed, the environment, the episode, the tick within it |
| a decision's draws (the action, Yemong's position offset at a reset, the scripted agent's plan switches and flags, the uniform floor) | the tick's key and the row's role: the decision key each controller's `step` receives |
| a row's starting clock (staggered starts) | the seed, the environment |
| the league's swaps | the seed, the swap count |
| the learner's minibatch order and shot spread | the seed and the stream's name (a seeded generator) |

The hash is lowbias32, one round per coordinate, on int64 tensors, so it runs inside a
compiled stage and a captured graph with no state and no host read; a replay draws afresh
because the carry's clock has moved
(`tests/sections/test_game_section.py::test_compiled_rollout_graph_replays_draw_fresh_randomness`).
Actions are sampled by the inverse CDF of each factor in turn (the power, the stick given
the power, the shoot given both), one uniform each, so two policies that are close pick the
same action from the same uniform wherever their cumulative probabilities agree. The arena
scale's draw is made whether or not a pool samples map sizes, so turning sampling on or off
moves no other draw. Consequences:

- a model option that adds a module or changes another module's shape leaves every other
  parameter's initial values bit-identical, and the maps and draws of every episode;
- environment e's k-th episode plays the same map in every arm of one seed, even after the
  arms' episode lengths diverge;
- the same configuration on the CPU reproduces exactly; on the GPU within rounding.

The game itself draws nothing: `bnb.game.api.resolve` takes the tick's uniforms and the
next episode's (`next_episode_uniforms`). The play shell keys its sides' decisions from
generators of its own (`bnb.play.sides`). Runs from before the keyed streams (rl-v20 and
earlier) do not reproduce under them.

On the CPU the rollout runs the stages tick by tick. The realtime path
(`bnb.runtime.realtime.Realtime`) runs one match at batch size 1 with the whole tick
compiled as one function by Inductor's C++ backend.

## Shapes and layout

- Ships are stored team-major: ship `n = team * ships_per_team + slot`. A `(B, N, ...)`
  tensor viewed as `(B, 2, S, ...)` is indexed by team without a copy.
- Every size is fixed for a run. The rollout allocates its buffers once, when it is built.
- State is float32; counters are int32 and flags bool. Models may use bfloat16 inside.
  float64 appears only in the reference tests.
- There are two teams, a constant (`bnb.contracts.config.TEAMS`).

## One code path

The same functions run at any batch size, on the CPU or the GPU, eagerly or compiled. No
algorithm branches on a preset or a batch size unless the same benchmark run ablates the
branch, so a reported speed-up holds in general and not at one size.
