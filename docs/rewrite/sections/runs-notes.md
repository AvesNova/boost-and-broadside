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
  - `bnb.runs.checkpoint.read_policy(directory) -> PolicyRecord` (`return_scale` as
    `(low, high)`, `steps`), and `bnb.runs.checkpoint.warm_start(policy, config, device)
    -> Trainer`, which `bnb train --init-from POLICY` calls.
  - Every model is built from its build record: `Yemong(yemong_build(config, rules,
    types))`, `Relational(RelationalBuild(config, ships_per_team, num_ships))`.
- **Done:** every spec item, each with tests under `tests/runs/`; the docs; origin/v2
  merged (df04ab99).
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
  versions, game, ship card as per-type lists, return scale, steps, commit, dirty, time,
  file hashes) and `weights.safetensors` (the state dict).
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
3. **The return scale lives in `checkpoint.json`** (the policy part), not in
   `training.safetensors`: a policy file needs it for the warm start, and a training
   checkpoint is a policy file. Two float32 values round-trip exactly through JSON.
4. **The default generators are restored by the entry point**
   (`restore_default_generators`, called by `bnb train --resume` and the resume worker),
   not inside `load_training`: scaffold rule 3 keeps process-wide state out of library
   code. It is the one process-wide effect of a resume.
5. **`--set path=value`** is the CLI's generic setting of `TrainConfig` fields (JSON
   values). On `--resume` only `OVERRIDABLE` paths (learning rate, warm-up, loss
   coefficients, entropy), `--updates` and `--checkpoint-every` are accepted; every other
   option fails and points to `--from`. The spec's "logging and diagnostic cadence" does
   not exist yet (section 9). Fields neither structural nor listed (precision,
   micro-batch rows, shards, old log-probabilities, league cadence, seed) are fixed on
   resume (most literal reading).
6. **`updates` is the run's total budget,** so a plain resume continues to it.
7. **A fork keeps the step counts and the league slot** besides the model, optimizer and
   return scale (runs.md lists the league; a loaded optimizer with the warm-up restarted
   would be inconsistent). It does not load the carry or generators (the sizes may
   differ).
8. **The default cadence is measured in the run:** about 30 minutes at the throughput of
   the segment's second update (the first compiles), in whole updates, recorded in
   run.json as `checkpoint_every`. Until it is measured no periodic checkpoint is written.
   A per-preset table of measured throughputs would go stale per machine.
9. **Retention (author's update):** newest `RECENT` (8), the one nearest each age 8, 16,
   32, … checkpoints back (age = step gap over the cadence), and the final one. Read
   literally, every age beyond the oldest is nearest the oldest, so the oldest is always
   kept. Applied after each checkpoint, a deleted one cannot come back, so the history has
   gaps: at 300 checkpoints the older ages kept are 8, 19, 75 and 299. If the league wants
   an evenly geometric history, a stable rule (by index bits) would need a spec change.
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

MEASUREMENTS

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
2. **`TrainConfig` has no model config or ship card.** The trainer builds Yemong with
   `YemongConfig` defaults (apart from bf16) and the default card. A resume refuses when
   the code's build or card differs from the checkpoint's; a run with another width or
   card needs `TrainConfig` fields for them (section 8 or the cleanup pass).
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
7. **Overridable on resume:** confirm whether `micro_batch_rows` (memory only, same
   result) and `precision` should be allowed.

## Size

| part | lines |
|---|---:|
| `src/bnb/runs/` | 1117 |
| `src/bnb/cli.py` | 378 (was 126) |
| `src/bnb` in total, against the branch point 3395b218 | +1552 / -57 |
| of which outside `bnb.runs` and the CLI (models, runtime) | about +110 / -40 |
| tests (`tests/runs/` 1007, call sites, docs check) | +1193 / -25 |
