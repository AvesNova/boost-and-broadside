---
covers: [pyproject.toml, src/bnb/contracts/tree.py, tests/conftest.py, tests/contract/test_imports.py]
reviewed: 2026-11-02
---

# Code style

The rules the code in `src/bnb`, `bench`{.plain} and `tests` follows. They come from the design
rules in `docs/rewrite/scaffold.md` (sections 3 and 4) and the decisions recorded in
`docs/rewrite/`. Where a rule is checked by a tool or a test, the check is named.
[Writing docs](writing-docs.md) covers the documentation.

## Tooling

- Python 3.14 (`requires-python` in `pyproject.toml`), managed with uv. Run everything
  through `uv run --no-sync`, so a command never changes the environment.
- `uv run --no-sync ruff check .` must pass. The rule sets and the line length (100) are
  in `[tool.ruff]` in `pyproject.toml`.
- The code is formatted with `uv run --no-sync ruff format .`. The suite runs
  `ruff format --check` and fails on unformatted code.
- A library that a task needs may be added. Tooling-only libraries go in an optional
  extra or a dependency group, so the runtime install stays small.

## Layers

- The packages form one dependency order:

  <!-- not-runnable: the dependency order, not a command -->
  ```text
  contracts <- game <- controllers, models <- learn <- runtime <- runs <- league <- play
  ```

  An import points down this order, never up. `tests/contract/test_imports.py` checks it.
- **Sections are pure functions:** tensors in, tensors out. Only `bnb.runtime` writes
  into persistent buffers (`copy_` into the carry).
- **The game state is opaque outside `bnb.game`.** Other packages use the functions in
  `bnb.game.api` and never read or write a field of the state.
- **Process-wide settings belong to the entry points** (`bnb.cli`, `bench`{.plain}, a package's
  `__main__`). Library code never seeds the global generators, sets TF32, thread counts
  or compiler limits, or calls `torch.cuda` configuration functions.
- **Section code does not compile itself.** The runtime chooses the compiled stages and
  lists them in one place.
- **One code path** for the CPU and the GPU and for every batch size. A preset-gated or
  size-gated branch needs a benchmark that measures both sides.

## Names

- **Full words:** `environment`, `probability`{.plain}, `observation`, `length`. Universal
  abbreviations (`cos`, `sin`, `x`, `y`, `id`) are fine; spec symbols (`c`, `w`, `gx`)
  and truncations (`obs` as a variable, `buf`{.plain}, `ctrl`{.plain}, `idx`) are not.
  - The exceptions are names that are themselves contracts or established terms: the
    `Obs` contract, the `ppo` module, `bnb`.
- **Units in full words** as suffixes when a name carries one: `_seconds`{.plain}, `_metres`{.plain},
  `_ticks`{.plain}. A quantity in the base units of the conventions (m, s, kg, J, W, rad) needs
  no suffix unless another unit of the same quantity is in scope.
- **Booleans** are named for the state they describe: `alive`, `fogged`, `visible`, not
  `is_alive`{.plain} or `alive_flag`{.plain}.
- **Metric names are quantity first:** `{group}/{quantity}/{slice}`, in full words
  (diagnostics spec, section 1).
- **One meaning per name** across the project. `reset` is the environment resetting
  before an observation; `done` is an episode ending after a step.
- **Shape letters** appear in shape comments and docstrings only. Each has one meaning:

  | Letter | Meaning |
  |---|---|
  | `B` | environments in the batch |
  | `T` | teams |
  | `S` | ships per team |
  | `N` | ships, `T·S`, in team-major order |
  | `M` | fields |
  | `K` | bullet slots per ship |
  | `Z` | zone slots |
  | `C` | spawn candidates |
  | `L` | steps of a rollout segment |
  | `R` | views: the rows of the batch one controller plays |

  The reward levels have no letter: `L` was taken by the segment's steps, so a shape
  writes them as `levels`, as in `(L, R, S, levels)`.

  In code, a dimension is a named variable (`ships_per_team`), or unpacked from a shape
  with these letters at the top of a function (`B, N, K = bullets.moves_left.shape`).

## Contracts and types

- A contract is a frozen dataclass of tensors, declared with `contract` (from
  `bnb.contracts.tree`), so the runtime can map `copy_`, `clone` and `to` over it. No
  dicts cross a package boundary.
- **Field documentation:** the class docstring gives each field's shape, dtype and unit.
- **Versions:** a contract that a checkpoint or an exported model depends on carries
  `SCHEMA_VERSION`{.plain}. Any change to its fields, their meaning or their order raises the
  version, and `docs/rewrite/contracts.md` records the new schema.
- **Model versions:** every registered model (`bnb.runs.registry.MODELS`) carries
  `VERSION`{.plain}, an int, and builds from its build record alone (its config and the numbers it
  derives from the game). A saved model loads only at its own version.
  - **A new option's default must reproduce the old behaviour.** If it cannot, the change
    alters what saved weights compute, and `VERSION`{.plain} goes up. Such changes include a
    different normalisation, a reordering of input columns and a new part that is always
    on. A refactor that computes the same numbers needs no bump.
  - The compatibility set (`tests/runs/checkpoints/`) holds one tiny policy file per model
    at its current version, with its outputs on a fixed input;
    `tests/runs/test_compatibility.py` fails on a change of computation without a bump. A
    bump replaces the model's entry (`tests/runs/compatibility.py`), and the commit says
    why.
- **Type hints** on every function signature, in the modern syntax: `X | None`,
  `list[int]`, `tuple[torch.Tensor, ...]`.
- **Dtypes:**
  - float32 for game and runtime state;
  - int32 for counters and categories;
  - bool for flags;
  - int64 only where torch requires it (indexing), converted where it is used;
  - bfloat16 only inside models;
  - float64 only in reference oracles and offline tools.

## Tensors and speed

- **Shapes:** a tensor whose shape is not obvious from its line gets a shape comment,
  `# (B, N, K)`. A shape comment is code: it changes in the same commit as the shape.
- **No host synchronisation in the hot path:** no `.item()`, `.cpu()`, `.tolist()`,
  `nonzero`, or boolean-mask indexing with a data-dependent shape inside a step, a stage
  or a minibatch. Read results back once per update, in one transfer.
- **Fixed shapes per run.** Buffers are allocated once, at startup, from the config.
  Nothing allocates inside a captured graph.
- **No Python loops over the batch.** A loop over a small, fixed axis (teams, modes,
  heads) is fine when it is unrolled at compile time.
- **Devices** are passed explicitly. Nothing relies on a default device.
- **In-place writes** only in `bnb.runtime`'s buffers. Elsewhere, build new tensors and
  return them.
- **Speed claims are measurements:** made with the benchmark lock held
  (`bench.lock.bench_lock()`), at steady state, with the cell, the commit and the machine
  stated. See [Benchmarking](../guides/benchmarking.md).

## Randomness

- **The game takes its randomness as input:** the uniform draws for a tick
  (`bnb.game.api.uniforms`) and for a spread (`bnb.game.api.spread`) are arguments, so
  the game is a deterministic function of its inputs.
- **Draws come from named streams:** one `torch.Generator` per stream, seeded from the
  run's root seed and the stream's name. A new consumer of randomness gets its own
  stream, so adding it does not shift any other draw.
- **No global seeding** outside the entry points. Tests use a local generator for every
  draw.

## Configuration

- Configuration is frozen dataclasses. A default is the documented baseline value. Every
  run writes its full resolved configuration, so a result never depends on a default
  that has since changed.
- **Validate at construction** (`__post_init__`), and fail with a message that names the
  field and the value.
- **Time-native rules** are written in seconds and converted to ticks once, with
  `ticks_of`, which fails unless the time is a whole number of ticks.
- **A number that is a rule** lives in `Rules`, a config field or a named module
  constant, defined once. A number that two places share by intent has one definition;
  two equal numbers that mean different things get two names.

## Docstrings and comments

- **Every public module, class and function has a docstring.** The API reference is
  generated from them. `uv run --no-sync python -m tests.docs.docstrings` lists the
  missing ones.
- **A module docstring** says the module's one job, and cites the spec section it
  implements (`game spec 7.2`).
- **A function docstring** says what it returns, with shapes and units, and anything a
  caller could not guess from the signature. Argument lists in the Google style are used
  only where the arguments need explanation.
- **Do not restate values the code defines:** name the config field instead of its
  default, and the constant instead of its count. They drift.
- **Comments say why.** What the code does should be clear from the code and its names;
  a comment that only restates it is removed.

## Errors

- **Fail fast:** invalid configuration, a contract mismatch or a schema version mismatch
  raises at once, with a message that says what was expected.
- **No silent fallbacks:** a missing input is an error, not a default. `assert` is for
  internal invariants, never for checking user input.

## Files

- **Small files with one job.** There is no line limit; each section review reports its
  size.
- **Shallow packages:** a subpackage exists when its files form a unit (`game/physics`,
  `models/yemong`).
- **No legacy:** when code is replaced, the old code, its tests and its docs are deleted
  in the same change. No deprecation shims and no aliases for old names. A temporary
  compatibility layer (such as a test adapter) is named as temporary, and has a planned
  native replacement.

## Tests

- **Layout:**

  | Directory | Holds |
  |---|---|
  | `tests/contract/` | the contracts: shapes, dtypes, versions, exports, the dependency order |
  | `tests/reference/` | equivalence with float64 fixtures generated from v1's code; see [Reference fixtures](../guides/reference-fixtures.md) |
  | `tests/sections/` | each section's own tests |
  | `tests/loop/` | end-to-end runs and small learning tests |
  | `tests/docs/` | the documentation checks |

- **Fixed reference sets:** the goldens and their tests are never edited to make a change
  pass. An adapter changes only when the code's interface changes.
- **Markers:** `gpu` on every test that can touch CUDA; `slow` on every test that takes
  more than 10 seconds on a quiet machine. `-m "not gpu"` never initialises CUDA.
- **No shared state:** every random draw uses a local generator, and no test changes a
  process-wide setting that another test can see.
- **One behaviour per test,** named as a statement of it:
  `test_a_dead_ship_fires_nothing`{.plain}.
- **Invariants first:** conservation, symmetry under the team swap, bounds and limits
  find more than hand-picked values do.
- **No mocking the game.** Tests run the real functions on real tensors (the CPU is
  fine).
- **Tolerances are stated and float32-aware:** a tolerance says why it has the value it
  has. A statistical test states how many standard deviations its bound is.
- **A test must be able to fail:** for a new behaviour, break the code by hand and watch
  the test fail before you commit it.

## Commits

- [Conventional Commits](https://www.conventionalcommits.org/): `type: summary` or
  `type(scope): summary`, lowercase, imperative, at most 72 characters, no full stop.
- **Types:** `feat`{.plain}, `fix`{.plain}, `perf`{.plain}, `refactor`{.plain}, `test`{.plain}, `docs`{.plain}, `bench`{.plain}, `build`{.plain},
  `chore`{.plain}.
- **The body** says why, and states any measured result with where it was measured. The
  diff already shows what changed.
- **One idea per commit.** Fold fixes to unpushed work into the commit they belong to.
- **Commit and push as you go:** each finished step is pushed, so no work lives only on
  one machine.
