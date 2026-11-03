# Hygiene round: notes

Branch `fix/hygiene`, from v2 at `c29daffb`. Brief: `hygiene-brief.md`.

## Progress

- [x] 5 the retention docstring (`6c8f2bd6`)
- [x] 4 the game state's integer dtypes (`1cf9fc6d`)
- [x] 3 the CUDA guard's false errors (`c22822bf`)
- [x] 1 model settings and the ship card in `TrainConfig` (`76b27fa0`)
- [x] 2 GPU resume within rounding (`c3337300`)
- [x] 3 follow-up: a non-gpu test after a gpu one sees CUDA as it is (`6866aa94`)
- [x] seeded module fixtures (`f3da255c`, found while measuring item 6)
- [x] 6 the quick tier's time (`2b0c053b`)
- [x] the full suite under the lock (on `2b0c053b`: one failure, the play shell's, below)

## Items

### 1. Model settings and the ship card in `TrainConfig`

- `TrainConfig.yemong` (`YemongConfig`) and `TrainConfig.relational` (`RelationalConfig`)
  replace `relational_hidden`. `model_build(config)` builds from them; the old
  `architecture` argument is gone, and so is `Trainer`'s `build` argument (the trainer
  always builds `model_build(config)`).
- `bfloat16_matmuls` in either config must stay False (`TrainConfig.__post_init__`
  refuses it): `precision` sets it, so the stored config has one source for it.
- `TrainConfig.ship_card` is the card as JSON data: field name to one float per type,
  every field given (default `bnb.game.api.default_card`, new). `TrainConfig.ship_types(device)`
  makes the tensors. A dict, not a dataclass: `ShipTypes` already lists the fields once and
  `DEFAULTS` once more, and a third hand-written list would drift; the dict is checked
  against `DEFAULTS` at construction (missing, unknown, wrong type count).
- `Rollout(config, types, pools, ...)` and `first_results(game, types, pools, ...)` take
  the card; so does `cloning.teacher_distributions`. The `Trainer` passes its config's
  card (kept as `trainer.types`). `bnb.league.matches.play` still plays the default card
  (`api.ship_types(device)`), as before.
- CLI: `--set yemong.width=256` and `--set ship_card.hit_damage=[14,18]` go through the
  existing `with_settings`; nothing new was needed in the parser.
- Records: `config.json` (beside `run.json`) and every training checkpoint's
  `train_config` hold the settings and the card, since both are `to_json(TrainConfig)`.
  (`run.json` itself holds no config; the brief's "run.json" is read as the run
  directory's record.)
- Resume: none of the new fields is in `OVERRIDABLE`, so `load_training` refuses a change
  ("a resume cannot change relational.hidden") and the CLI refuses `--set` of one.
- Fork and warm start: `check_build` (was `run_build`) compares the file's build record
  with `model_build(config)`, precision aside. Deviation from "as now": before, the
  architecture came from the file and only `relational_hidden` from the run. To keep that
  ("a changed code default never blocks loading an old file"), the CLI takes the file's
  model settings into the config (`with_model_of`) when the command sets none of them
  (no `--set yemong.*`/`relational.*`); a command that sets one must match the file.
- Tests: `test_set_yemong_width_reaches_the_built_model` (CLI to weights),
  `test_the_model_settings_round_trip_through_a_checkpoint`,
  `test_a_resume_with_another_width_is_refused_naming_the_setting`,
  `test_a_resume_that_sets_a_model_setting_is_refused_with_its_reason`,
  `test_a_fork_takes_the_files_model_settings_unless_given`,
  `test_a_fork_that_sets_no_model_setting_takes_the_parents`.
- Run directories written before this branch cannot be resumed (`relational_hidden` is an
  unknown field).

### 2. GPU resume within rounding

- `tests/runs/test_resume.py`: the CPU cases stay bitwise; the GPU cases compare every
  float to `GPU_TOLERANCE` = 1e-4 (relative, the same absolute floor), and ints, strings,
  flags and the float counts (`results`, `games/`) exactly. No other GPU test compares
  bits between processes (`test_scripted_acceptance` is statistical).
- Measured (RTX 4060 Laptop, lock held, `/tmp/bnb-scratch-hygiene/resume_diff.py`): the
  undamaged GPU resume was bitwise on both models in this session.
- By hand, both models on the GPU, the test's comparison applied to each run:

  | Damage | Model | Weights past the bound | Largest weight difference | Floats past the bound |
  |---|---|---:|---:|---:|
  | `minibatch_order` generator not restored | relational | 10 of 10 | 1.2e-2 | 17 |
  | optimizer state not restored | relational | 10 of 10 | 3.5e-2 | 18 |
  | `minibatch_order` generator not restored | yemong | 111 of 116 | 3.0e-2 | 36 |
  | optimizer state not restored | yemong | 111 of 116 | 4.0e-2 | 33 |

  Reverted after each (`git checkout`).
- Finding: a one-ulp change of every weight right after the restore (a stand-in for a
  kernel configuration that rounds differently) also fails: relational 1 weight and 3
  floats past the bound (explained variance moves 4%), yemong 109 of 116 weights (largest
  2.3e-2) and 35 floats. Adam divides each step by the gradient's running scale, so a
  parameter whose gradient is near zero turns rounding noise into a full step of the
  learning rate (3e-3 in the worker). At this test's learning rate no tolerance separates
  a last-bit drift from a lost state after one update. The tolerance therefore guards what
  the brief asks (state left out fails) and passes the bitwise-or-near resumes this
  machine produces; a resume whose autotuner picks another configuration can still fail.
  If that shows up, the options are pinning the kernels' configurations in the test's
  processes or comparing only the first segment after the resume (before any step).

### 3. The CUDA guard's false errors

Three causes, each reproduced by running the test alone with a plugin that prints the stack
of the first CUDA initialisation:

- **Adam's step** (`test_reinforcement_learning_starts_from_the_clone`): torch 2.15's
  `Optimizer._accelerator_graph_capture_health_check` reads
  `torch.accelerator.current_stream()` on every `step`, which initialises CUDA whenever the
  machine has it. `bnb.learn.ppo.Adam` overrides it to run only when a parameter is off the
  CPU. A library change, not a test one: a CPU training process on a GPU machine no longer
  creates a CUDA context either. Same arithmetic.
- **`torch.compile`** (`test_a_compiled_mixed_tick_equals_each_game_compiled_alone`,
  `test_realtime_steps_one_match[True]`): dynamo saves and restores
  `torch.cuda.get_rng_state()` around every compile when `torch.cuda.is_available()`, and
  inductor's cache key (`CacheBase.get_system`, cached per process) reads
  `torch.cuda.current_device()`'s name. Neither can be avoided from library code. The guard
  fixture now runs a non-gpu test as on a machine without CUDA: `torch.cuda.is_available`
  returns False and `torch.cuda.current_device` raises `RuntimeError` (which
  `get_system` catches, leaving the device out of the key).
- **`torch.random.fork_rng()`** without `devices` forks every CUDA device's generator. The
  CPU tests that used it (`test_cli`, `test_training_checkpoints`, `test_upload`,
  `test_bc`, `test_league_slots`, `test_league_metrics`) pass `devices=[]`.

No test was marked `gpu`. Each listed test passes alone with the GPU visible. A plugin that
recorded every non-gpu test reaching `torch.cuda._lazy_init` over the whole GPU quick tier
then found one: `test_records`' provenance reads the GPU's name when CUDA is already
initialised, which the first version of the guard broke (it hid CUDA from every non-gpu
test). The guard now hides CUDA only while it is uninitialised (`6866aa94`).

**Seeded module fixtures** (`f3da255c`): a module-scoped fixture is built before any test's
own fixtures, so neither the global-generator guard nor `global_generator` covered it, and
its data depended on the draws of the modules before it. One quick-tier run failed
`test_learner.py::test_each_loss_term_matches_an_independent_computation` this way (its
stored rollout had no hit). `module_global_generator` (conftest) seeds the default
generators at 0 for the module and restores them; the six module fixtures that draw from
them take it (`test_learner` two, `test_bc`, `test_yemong_model`, `test_learner_precision`,
`test_yemong_kernels`).

### 4. Integer dtypes

`mode.score`'s per-team sum promoted it to int64 (`torch.sum` of int32). Fixed with
`sum(-1, dtype=torch.int32)`. A probe of every state leaf over 90 ticks of each preset and
the sandbox, with and without one-second episodes, found no other field that changes
dtype. `tests/contract/test_game_dtypes.py::test_a_tick_keeps_every_state_fields_dtype`
ticks each mode through deaths, respawns and an episode end; it fails without the fix.

### 5. Retention docstring

`bnb.runs.records.retained` and the runs page now say the league's snapshots have their
own rule (`bnb.league.pool.retained`).

### 6. The quick tier

From `--durations` on the quick tier (GPU, lock held), the tests over 10 s, now `slow`:

| Test | Time | Quick variant |
|---|---:|---|
| `test_probe.py` (both; the first compiles the GPU rollout and loss) | 22-28 s | the CPU probe in `test_cli` |
| `test_cli.py::test_ctrl_c_finishes_the_update_writes_a_checkpoint_and_stops` (a fresh process) | 15 s | `test_an_interrupt_finishes_the_update_writes_a_checkpoint_and_stops` (SIGINT in-process) |
| `test_mixed.py::test_one_slice_of_several_arenas_and_clocks_equals_each_game_alone` | 9-28 s | the same over 40 ticks |
| `test_mixed.py::test_an_unaligned_mixed_batch_equals_each_game_alone_to_float32_rounding` | 9-15 s | the same over 60 ticks |

The range is the load: the other agent kept the machine at load 3 to 9 through this round.
After the marks, no quick test takes 10 s; the largest are the league slots' drain fixture
(6.6 s, 600 ticks shared by its module) and `test_unroll_equals_steps_float64` (4.7 s).
The tier is not at 2.5 min on this loaded machine (below). What is left is a long tail:
1161 tests, most of them 0.5 to 3 s; the next cuts would be per-module (the drain
fixture's 600 ticks, the float64 Yemong comparisons), each a judgement on coverage.

## Measurements

All on the RTX 4060 Laptop (8 GB, 16 threads), provisional: another agent's work held the
load average between 3 and 9 throughout.

| Run | Commit | Result | Wall time | Load (start, end) |
|---|---|---|---:|---|
| quick, GPU, lock held, before | `6c8f2bd6` | stopped at the play failure (`-x`), 905 tests | 3 min 50 s | 0.5 |
| quick, GPU, lock held, before the marks | `c3337300` + guard fix | 1163 passed | 4 min 0 s | - |
| quick, GPU, lock held | `2b0c053b` | 1161 passed, 88 deselected | 3 min 18 s | 3.4, 7.2 |
| quick, CUDA hidden | `2b0c053b` | 1123 passed, 38 skipped | 3 min 28 s | 7.2, 9.1 |
| full, GPU, whole run under the lock | `2b0c053b` | 1247 passed, 1 failed, 1 skipped | 56 min 23 s | 5.5, 4.5 |

The full run's failure is
`tests/sections/test_play_interface.py::test_a_match_is_set_up_watched_flown_and_replayed_from_the_interface`
(`StopIteration`: the saved replay's markers hold no kill). It is the play shell's
(`src/bnb/play/`, off limits here) and intermittent: it failed in the first quick run, on
`6c8f2bd6` (before any change of this round), and passed in the three quick runs after.

## Requests

1. **The play interface test** above, for the play fix round.
2. **The GPU resume's tolerance** (item 2): a last-bit difference grows past it within an
   update. If the GPU resume test starts failing after an autotune picks another
   configuration, pin the kernels' configurations in its worker processes, or compare
   only the first segment after the resume.
3. **Old run directories** cannot be resumed after item 1 (`relational_hidden`).
