# Section 11 (runs and checkpoints): notes

Branch `section/runs`. The spec is `runs-spec.md` (with the author's 2026-11-02 update:
two-word run names, the league's retention, W&B as the only upload target); the decisions
are in `runs.md`.

## Progress

- **For the coordinator: `save_policy` and `load_policy` are pushed, tested and
  merge-ready** (since commit 2a4c36e7). The BC hand-off can call:
  - `bnb.runs.checkpoint.save_policy(directory, model, *, return_scale=None, steps=None,
    game=None, types=None)`: `return_scale` is the learner's `ReturnScale`, `steps` a
    dict such as `{"environment_steps": n}`;
  - `bnb.runs.checkpoint.load_policy(directory, device) -> model`;
  - `bnb.runs.checkpoint.read_policy(directory) -> PolicyRecord` (`return_scale`: the
    state tensors `{"low", "high"}`, `steps`), and `bnb.runs.checkpoint.warm_start(policy, config, device)
    -> Trainer`, which `bnb train --init-from POLICY` calls.
  - Every model is built from its build record: `Yemong(yemong_build(config, rules,
    types))`, `Relational(RelationalBuild(config, ships_per_team, num_ships))`.
- **Done:** every spec item, each with tests under `tests/runs/`; the docs; origin/v2
  merged (df04ab99).
- **Fix round (after the 7.5/10 review): done.** All ten fixes landed with their tests
  (see "Fix round" at the end); full suite passed under the bench lock; origin/v2 merged;
  roadmap status `merged`.
- **Next:** the coordinator's quiet-machine measurements (acceptance 12, provisional
  below); the requests at the end.

## Design

| Module | Job | Lines |
|---|---|---:|
| `runs/serialise.py` | `to_json`/`from_json` driven by type hints (exact floats, unknown field fails with its path, missing fields reported); `named_tensors`/`tree_from_named` (key paths, names, shapes and dtypes checked) | 168 |
| `runs/registry.py` | `MODELS` (name to class) and `BUILDS` (class to record type) | 32 |
| `runs/checkpoint.py` | atomic `write_checkpoint`; `save_policy`, `read_policy`, `load_policy`; `save_training`, `load_training`, `fork_training`, `warm_start`, `restore_default_generators`; `OVERRIDABLE` | 420 |
| `runs/records.py` | `Run` (run.json, segments, status, stale detection, checkpoints, retention, rewind), `run_name`, `retained` | 238 |
| `runs/session.py` | one process's segment: the probe on the first update, the cadence, Ctrl-C, failure | 84 |
| `runs/probe.py` | peaks, budget, the refusal | 49 |
| `runs/provenance.py` | commit, `provenance()`, `write_result`, file hashes | 86 |
| `runs/upload.py` | `bnb runs upload` to W&B | 40 |

- **Models.** Each model class has `NAME`, `VERSION` and a constructor taking its build
  record alone (`YemongBuild`, `RelationalBuild`, `MLPBuild`); an instance keeps it as
  `build`. `yemong_build(config, rules, types)` computes the derived numbers: tick length,
  number of ships (the density head's starting rate only), ship types, bullet moves, the
  fastest muzzle speed, the smallest hit radius, and the state code's ranges
  (`StateCodeRanges`: angle-of-attack limit, longest shield delay, cooldown bins, per-type
  max shield and energy), i.e. everything Yemong reads from section 1b's card.
- **Policy file:** `checkpoint.json` (format 1, kind, model name/version/build, schema
  versions, game, ship card as per-type lists, whether it holds a return scale, steps,
  commit, dirty, time, file hashes), `weights.safetensors` (the state dict) and
  `return_scale.safetensors` (the return scale's state tensors, when saved with one).
- **Training checkpoint:** adds `train_config`, the optimizer's param groups and `extra`
  to the record, and `training.safetensors`: `optimizer.<index>.<key>`, `league.<name>`
  (the league slot's weights), `carry.<path>` (`Rollout.saved_carry()`: state, types,
  observation, truth, seats, actions, spread, uniforms, sweep, ledger, every controller's
  carry, `t_idx`, counters; not the staging ring or segment buffers, which each segment
  writes before reading), `generator.shot_spread`, `generator.minibatch_order`,
  `default_generator.cpu` and `default_generator.cuda`.
- **Exact resume.** Three things beyond saving every state:
  1. `Rollout._capture` restores the device's default generator after its warm-up, as
     it already restored the carry. Without it, a resumed process (which captures again)
     starts its replays two ticks of draws later (the GPU resume test fails with it
     removed).
  2. `Trainer.pool_metrics` uses masked sums instead of `index_add_` (CUDA atomics), so a
     metric of the same data is the same number (see measurements).
  3. The entry point restores the default generators after building the trainer
     (`restore_default_generators`).
- **CLI.** `bnb train` (new run, `--resume RUN [--at STEP]`, `--from RUN [--at STEP]`,
  `--init-from POLICY`, `--print-config`, `--probe`, `--name`, `--runs-directory`,
  `--checkpoint-every`, and `--set path=value` for any `TrainConfig` field),
  `bnb runs [show RUN | upload RUN [--at STEP] [--league] [--to wandb]]`,
  `bnb ship-card`.

## Decisions and deviations

1. **The registry is an explicit dict in `runs/registry.py`, not a decorator.** A
   decorator needs the registry below `models` in the layer order and import side
   effects to fill it; one line per model is smaller and cannot miss a model.
2. **`build_record()` is the attribute `build`; `from_build_record` is the constructor.**
   The constructor takes exactly the record, so a separate classmethod would be the same
   call.
3. **The return scale's state tensors** (`ReturnScale.low` and `high`, `SCALE_STATE`) are
   a file of the policy part, `return_scale.safetensors`, hashed like the weights (the
   coordinator's request from the BC hand-back), not in `training.safetensors`: a policy
   file needs them for the warm start, and a training checkpoint is a policy file.
   `read_policy(...).return_scale` gives them.
4. **The default generators are restored by the entry point**
   (`restore_default_generators`, called by `bnb train --resume` and the resume worker),
   not inside `load_training`: scaffold rule 3 keeps process-wide state out of library
   code. It is the one process-wide effect of a resume.
5. **`--set path=value`** is the CLI's generic setting of `TrainConfig` fields (JSON
   values; fields inside tuples, such as `pools`, cannot be set). On `--resume` only
   `OVERRIDABLE` paths (learning rate, warm-up, loss coefficients, entropy, reward
   weights, precision, micro-batch rows), `--updates`, `--precision`,
   `--micro-batch-rows` and `--checkpoint-every` are accepted; every other option fails
   and points to `--from`. The spec's "logging and diagnostic cadence" does not exist yet
   (section 9). Shards, old log-probabilities, league cadence and seed stay fixed.
6. **`updates` is the run's total budget,** so a plain resume continues to it; a fork's
   update count starts at zero, so for a fork `--updates` is what it trains.
7. **A fork keeps the environment-step count and the league slot** besides the model,
   optimizer and return scale (runs.md lists the league; a loaded optimizer with the
   warm-up restarted would be inconsistent). It does not load the carry or generators
   (the sizes may differ), and its update count starts at zero.
8. **The default cadence is measured in the run:** about 30 minutes at the throughput of
   the segment's second update (the first compiles), in whole updates, recorded in
   run.json as `checkpoint_every`. Until it is measured no periodic checkpoint is written.
   A per-preset table of measured throughputs would go stale per machine.
9. **Retention (the coordinator's rule, fix round):** by ordinal i (order written), with
   age = n − i: keep age < `RECENT` (8), or i divisible by 2^floor(log2(age)), and the
   final one. Stable, one checkpoint per doubling band. The ordinals live in run.json
   (`checkpoint_ordinals`); `retained` takes ordinals only, for the league's snapshots too.
10. **A rewind** (`--resume RUN --at STEP`) moves the newer checkpoints into
    `checkpoints/rewound_<segment>/` (kept, outside the retention), so the run can write
    those steps again; the segment records `rewound_from`.
11. **A checkpoint is never overwritten** (`FileExistsError`); a stop right after a
    periodic checkpoint reuses it.
12. **`bnb` needs a command now** (`bnb train`, per spec 6); `bnb --preset ...` is gone.
    The docs and the docs check (now per command) follow.
13. **The run name's words:** 32 adjectives and 32 nouns in `runs/records.py`.
14. **The W&B run id** is recorded as `null`: section 9's sink sets it. A fork's parent is
    in run.json (`parent`); naming it in the W&B config is the sink's job.
15. **The compatibility set** stores the inputs (observation, carry) beside the outputs,
    so a game change does not fail it. Yemong's smallest entry is 160 kB on disk (width 8;
    the 490-wide state code dominates), not "a few kB". Tolerance: rtol 1e-4, atol 1e-5,
    for another CPU or BLAS; on this machine it reproduces bit for bit.
16. **The MLP is registered** ("while it exists") with `MLPBuild`.
17. **The probe runs on the first update of every segment** (every process), so a resume
    on another machine is checked too.
18. **Policy files take `game` and `types` as optional keywords;** a training checkpoint
    always records both.

## Measurements

**Provisional:** RTX 4070 Laptop (8 GB), commit 797199f3, bench lock held, but two other
agents had processes on the GPU (1 to 4 GB) and loaded the CPU, so the 5v5 cells did not
fit at their bench size (4096 environments) and the update times are several times a quiet
machine's. Script: `/tmp/bnb-scratch-runs/measure.py` (one cell per process; Yemong,
float32, one update first so the optimizer has state; min of 3 saves and loads).

### Checkpoint size and times (rollout carry included)

| cell | environments | weights | training file | of which carry | of which optimizer | save | load (hashes + both files to the GPU) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `deathmatch` | 1024 | 10.4 MB | 128.5 MB | 97.3 MB | 20.7 MB | 0.21 s | 0.10 s |
| `deathmatch_wide` | 2048 | 10.4 MB | 226.2 MB | 195.0 MB | 20.7 MB | 0.34 s | 0.17 s |
| `frontline` | 2048 | 10.4 MB | 224.5 MB | 193.4 MB | 20.7 MB | 0.47 s | 0.25 s |
| `mixed` | 2048 | 10.4 MB | 226.2 MB | 195.0 MB | 20.7 MB | 0.41 s | 0.21 s |
| 26 ships, 54 fields | 784 | 10.4 MB | 218.0 MB | 186.8 MB | 20.7 MB | 0.34 s | 0.16 s |

- The carry is 95 kB per environment at 5v5 and 238 kB at 26 ships, almost all Yemong's
  carries (about 49 kB per learner view, learner notes), so at the bench sizes a
  checkpoint is about 420 MB (5v5, 4096 environments) and 410 MB (26 ships, 1576),
  extrapolated linearly; the save then takes about 1 s at this machine's rate.
- The league slot's weights (10.4 MB) are inside the training file.

### Throughput with and without checkpoints

- `deathmatch`, 1024 environments, 3 updates per arm, two arms each, best of each: 2042
  environment steps per second without checkpoints, 1913 with a checkpoint after **every**
  update (64 s per update on the loaded machine). One save is 0.21 s, 0.3% of an update;
  the 6% gap is the shared machine's noise.
- **At the default cadence** (about 30 minutes), one save of under 1 s costs under 0.05%
  of the run's time. The coordinator's quiet-machine run should confirm.

### The exact resume

**Bitwise.** `tests/runs/test_resume.py`: 2 updates, a training checkpoint, a resume in a
fresh process and 2 more equal 4 updates straight through: every parameter
(`torch.equal`) and every logged metric (compared as JSON floats, so exactly), on the CPU
(eager) and on the GPU (compiled stages, the CUDA-graph rollout), for the relational
model and Yemong. Two things were needed beyond saving every state (see Design): the
capture restoring the device generator (removing it fails the GPU test), and per-pool
metrics without atomics (with `index_add_`, the weights matched bit for bit but one
metric, self-play damage per second, came out 1.4e-11 against -4.1e-11, both zero up to
summation order). Two straight GPU runs were bit-identical before the fix too, so the
atomic order varied only across the processes' different histories. Request 3 names the
remaining atomics a long GPU run could meet. Each GPU case took 4 to 5 minutes on the
loaded machine (Yemong's compile dominates).

### The full suite

939 passed, 7 skipped (none under `tests/runs/`; the skips are the suite's optional extras and hardware checks), under the bench lock (34.6 minutes on the loaded machine). An
earlier run failed only `tests/sections/test_learner_precision.py` by out-of-memory while
another agent's process held 4.2 GB; it passed in the locked rerun.

## Changes outside `bnb.runs`

- **`bnb.models`:** `YemongBuild`, `yemong_build`, `StateCodeRanges`, `state_code_ranges`;
  `Yemong`, `ShipStateCode`, `Relational` (`RelationalBuild`) and `MLP` (`MLPBuild`) take
  their records; `NAME` and `VERSION` on each. Every Yemong fixture passes unchanged.
- **`bnb.runtime.rollout`:** `saved_carry`, `load_carry`; `_capture` restores the device's
  default generator.
- **`bnb.runtime.train`:** `build_model` uses the builds; `pool_metrics` without atomic
  adds.
- **`bnb.cli`:** rewritten for `bnb train` / `bnb runs` / `bnb ship-card`.
- **`pyproject.toml`, `uv.lock`:** `safetensors`, `wandb`. **`.gitignore`:** `/runs/`;
  `checkpoints/` anchored to the root (it ignored `tests/runs/checkpoints/`).
- **Tests:** call sites of the constructors (`tests/contract/test_export.py`,
  `test_yemong_contract.py`, `test_yemong_export.py`, `tests/sections/yemong_cases.py`,
  `test_yemong_codes.py`, `test_learner_precision.py`, `test_ship_card_game.py`);
  `tests/contract/test_imports.py` (runs in the layer order); `tests/docs/resolve.py`
  (commands parsed per subcommand, `tyro.conf.arg` names, bare `bnb` allowed); new
  `tests/runs/`.
- **Docs:** `docs/explanation/runs.md` (replaces the in-progress page),
  `docs/guides/resuming-forking-sharing.md`, `docs/reference/runs.md`, `packages.yml`,
  `roadmap.yml` (status merged, page), `mkdocs.yml`; `bnb train` in the README, the
  training guide and the first run (which gains "Where it goes"); the code style (the
  model-version rule, runs in the layer order); the writing guide (commands);
  architecture (the capture restores the generator); Yemong (the build record).

## Requests and open questions

1. **A game version.** The spec asks policy files to record "the game version"; there is
   no such constant. Files record the `GameConfig` and the ship card, and the build record
   catches what Yemong reads. Proposal: `bnb.game.VERSION` (or the `Events` schema
   version) for ratings keys.
2. **`TrainConfig` has no model config or ship card.** A new run builds Yemong with
   `YemongConfig` defaults (apart from bf16) and the default card. Resume, fork and warm
   start build from the checkpoint's record (fix round), so a changed default no longer
   blocks them; a new run with another width or card still needs `TrainConfig` fields for
   them (section 8 or the cleanup pass).
3. **GPU determinism beyond the tests.** The game's damage uses `scatter_add` (float
   atomics, `bnb.game.physics.combat`) and Yemong's density counts use `scatter_add_`.
   When two bullets hit one ship in one tick, the sum's order can vary between processes.
   The resume tests are bitwise (the tiny cases' sums happen to be exact or ordered), but
   a long run could differ by rounding after such a tick, and then diverge. If exact
   resume matters on long GPU runs, the game's damage sum needs a deterministic reduction.
4. **`contracts.md` current schema:** the build records and `VERSION` (code style now
   states the rule); runs between runtime and league in the layer order.
5. **A GPU rollout with `compile=False`** fails in capture (`Random`'s `_uniform` builds
   a host tensor inside the graph); it predates this section.
6. **The skeleton MLP** is registered; when the cleanup pass deletes it, its
   compatibility entry goes too.
7. **Overridable on resume:** `micro_batch_rows` and `precision` are allowed (the
   coordinator's decision, fix round).

## Size

| part | lines |
|---|---:|
| `src/bnb/runs/` | 1117 |
| `src/bnb/cli.py` | 378 (was 126) |
| `src/bnb` in total, against the branch point 3395b218 | +1552 / -57 |
| of which outside `bnb.runs` and the CLI (models, runtime) | about +110 / -40 |
| tests (`tests/runs/` 1007, call sites, docs check) | +1193 / -25 |

## Fix round

The coordinator's decisions on the review (7.5/10), each with the test that guards it.
All under `tests/runs/` unless named.

1. **Retention by ordinal, stable.** `records.retained(ordinals, final)`: age < 8, or i
   divisible by 2^floor(log2(age)), plus the final. Run keeps `checkpoint_ordinals` in
   run.json (a checkpoint whose run.json write a crash lost gets the next ordinal, in step
   order; a rewind drops the set-aside ones). Guards:
   `test_records.py::test_the_retention_applied_after_every_write_is_stable_with_one_checkpoint_per_band`
   (2,000 writes: each band from 8 holds exactly one, count ≤ 8 + log2 n, newest 8 and
   final kept), `..._keeps_the_newest_eight_one_per_doubling_band_and_the_final`,
   `..._the_run_keeps_its_checkpoints_by_ordinal_and_deletes_by_renaming`. Broken by hand
   (exponent off by one): both rule tests fail. Explanation page states the exact rule.
2. **A fork's budget counts from the fork.** `fork_training` sets `updates_done = 0`;
   environment steps kept. Guards: `test_cli.py::test_a_fork_is_a_new_run_naming_its_parent_that_trains_its_own_updates`
   (the guide's example, `--set learner.entropy=-0.005`, from a 3-update parent with
   `--updates 2`: trains 2, finished), and the fork assertion in
   `test_training_checkpoints.py::test_a_fork_takes_the_model_optimizer_and_return_scale_but_not_the_carry`.
   Broken by hand: both fail. The guide's example now passes `--updates`.
3. **A resume uses only what the checkpoint holds.**
   - `train.model_build(config, architecture)` and `Trainer(config, device, build)`;
     `checkpoint.run_build` gives resume, fork and warm start the record's architecture
     with the run's precision, and refuses when the numbers the game derives differ (or
     another model). Guard: `test_a_resume_builds_the_model_from_the_checkpoint_not_the_code_defaults`
     (a `value_bins=41` model resumes with its own build; broken by hand: fails).
   - `RewardConfig` is `TrainConfig.reward`: saved, restored, validated. The five level
     weights are in `OVERRIDABLE` (loss coefficients, recorded); `damage_unit` is fixed.
     Guard: `test_the_reward_weights_are_saved_and_a_resume_may_change_them`.
   - A default-filled build field warns `FilledDefaults` on resume and fork: guard
     `test_a_build_field_the_checkpoint_lacks_is_reported_on_resume_and_fork`. A model
     version mismatch still refuses (`read_policy`, unchanged tests).
4. **No resume of a live run:** `Run.live()`; the CLI refuses naming the process. Guard:
   `test_cli.py::test_a_resume_of_a_live_run_is_refused`.
5. **Atomic deletion and durability.** Retention renames to `step_N.deleting/` before
   `rmtree`; `checkpoint_steps` accepts only `step_<digits>`; leftovers (`.deleting`,
   `.tmp`) are removed after each retention; `write_checkpoint` flushes the `.tmp`
   directory before the rename. The explanation now claims what this guarantees
   (`<name>/` complete or absent). Guard: `test_the_run_keeps_its_checkpoints_by_ordinal_and_deletes_by_renaming`.
6. **The exact-resume test sees more.** `resume_worker.py` copies the league every 3
   updates with K = 2 (the slot lags at the checkpoint), at a learning rate of 3e-3 and a
   return-scale floor of 1e-7 (under the tiny games' return spread, about 1e-5, so the
   scale is live). `test_resume.py::test_the_exact_resume_test_sees_a_damaged_league_slot_or_return_scale`
   runs the resume with noise added to the league weights after the restore, and with a
   fresh return scale: both differ from straight through. At the review's settings
   neither damage showed (floor 1.0 hid the scale; a league perturbed by 1e-3, or set to
   the model's weights, flipped no sampled action), which is why the settings changed.
   The fork test compares optimizer values (`_assert_same_optimizer`).
7. **CLI and loading.**
   - `SerialiseError` and `MemoryBudgetError` (probe and run) exit with the one-line
     message: `test_a_checkpoint_with_an_unknown_config_field_is_a_one_line_refusal`,
     `test_a_probe_over_the_budget_is_a_one_line_refusal`.
   - One seed: `LearnerConfig.seed` removed; `Learner(..., seed=)` takes
     `TrainConfig.seed`. Guard: `test_seed_and_set_seed_are_one_setting`.
   - `--set` help says tuple fields cannot be set.
   - Optimizer state reassembles through `tree_from_named` (`checkpoint.restore_optimizer`,
     Adam's layout per parameter): `test_an_optimizer_state_of_another_shape_is_refused`.
   - `--runs-directory` defaults to `runs/` at `git rev-parse --show-toplevel`, else the
     current directory: `test_the_runs_directory_is_runs_at_the_repository_root`.
   - `--precision`, `--micro-batch-rows` may change on resume, recorded:
     `test_cli.py::test_the_micro_batch_rows_and_the_precision_may_change_on_resume`,
     `test_training_checkpoints.py::test_the_micro_batch_rows_and_the_precision_may_change_on_resume`
     (bfloat16: the build's `bfloat16_matmuls` follows the precision).
8. **Not in scope here.**
   - W&B run ids, and a resume continuing the same W&B run, come with section 9
     (diagnostics). The field stays (`wandb_id`, null until section 9's sink sets it);
     guard: `test_a_resume_carries_the_runs_wandb_id`.
   - There is no game version constant; policy files record the game config and the ship
     card instead (request 1).
9. **Quality and deviations.**
   - Deviations from the spec's wording, kept: the registry is a dict in
     `runs/registry.py`, not a decorator (decision 1); constructors take the build record,
     and `build` is an attribute, not `build_record()`/`from_build_record` (decision 2).
   - `stage: Callable[[str], None] | None`; `runtime.train` imported at the top of
     `runs/checkpoint.py` and `runs/session.py`; `cfg` became `config` (in `bnb.runs`,
     `bnb.cli`, and `Trainer.config` in `bnb.runtime.train`); `Trainer.steps_per_update`
     is the one definition.
   - The docs check no longer accepts positional fields as options (`bnb runs --action
     show`): `tests/docs/test_references.py::test_the_command_check_takes_positional_arguments_but_not_as_options`.
   - `wandb` stays a core dependency.
10. origin/v2 merged (already up to date at f117874f); section 11 is `merged` in
    `docs/roadmap.yml` (it already was).

**Full suite (fix round):** 956 passed, 7 skipped (optional extras and hardware checks;
none under `tests/runs/`), under the bench lock, 24 minutes, GPU resume tests included.
The remaining `PossiblyStalePage` warnings are pages this round did not touch, or pages
already reviewed today whose code changed in this round's commits and still read
correctly (training loop, first run, writing docs).

**Changes outside `bnb.runs` in this round:** `bnb.runtime.train` (`model_build`,
`build_model(build)`, `Trainer(config, device, build)`, `Trainer.config`,
`steps_per_update`, `TrainConfig.reward`); `bnb.learn.ppo` (`LearnerConfig.seed` removed,
`Learner(seed=)`); `bnb.cli`; `tests/sections/test_learner.py` (the stream seed);
`tests/docs/resolve.py`; docs (`explanation/runs.md`, `explanation/learner.md`,
`guides/resuming-forking-sharing.md`, `guides/training.md`, `guides/designing-a-ship.md`
reviewed).
