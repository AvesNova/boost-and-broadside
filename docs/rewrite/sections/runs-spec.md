# Runs spec (section 11): checkpoints, resume, run records

Status: spec, 2026-11-02. Decisions: `runs.md` (settled with the author 2026-11-02).
**(proposal)** marks a choice the spec makes, implemented as written and listed for the
author.

Order: after section 6 merges. The scripted agent and BC (sections 4 and 7) may run at
the same time: BC's hand-off calls `save_policy` and `load_policy` (section 3 below), so
define and push those first. The league (section 8) comes after this section.

## 1. Packages

- **`bnb/runs/`** (new). It goes into the layer order between `runtime` and `league`
  (`... <- learn <- runtime <- runs <- league <- play`); update the dependency test and
  the architecture page:
  - `records.py`: the run directory, `run.json`, the run name;
  - `checkpoint.py`: policy files and training checkpoints;
  - `serialise.py`: frozen dataclasses to JSON and back; pytrees of tensors to named
    tensors;
  - `registry.py`: model names to classes and versions;
  - `probe.py`: the memory probe;
  - `provenance.py`: the record every offline result carries;
  - `upload.py`: explicit uploads.
- **Library:** `safetensors`, added to the dependencies.
- **Layering:** `runs` may import `contracts`, `game.api`, `models`, `learn` and
  `runtime`. The models never import `runs`.

## 2. Serialisation

- **`to_json(value)` and `from_json(cls, data)`** for frozen dataclasses: nested
  dataclasses, tuples, lists, numbers, strings, `None`, enums by name. The round trip is
  exact (floats written with `repr`).
  - An unknown field fails, naming it.
  - A missing field takes the class's default, and the load returns the list of fields
    it filled (the model-version rule, section 4).
- **Named tensors:** a pytree (the carry, the league state, the optimizer state) is
  flattened with its key paths (`torch.utils._pytree.tree_flatten_with_path`) into
  `{"path.to.leaf": tensor}`. Loading checks every name and shape against the target
  pytree.
- **Generator states:** every named stream (`random_stream`), plus the default CPU
  generator and the default CUDA generator, saved as `uint8` tensors. The CUDA default
  matters: the policy's sampling inside the captured graph draws from it (Yemong
  `policy.py`), so an exact resume restores it.

## 3. Policy files and training checkpoints

- **A policy file** is a directory:
  - `checkpoint.json`:
    - the format version and the kind (`"policy"`);
    - the model's registered name and version;
    - its build record;
    - the `Obs` and `Action` schema versions;
    - the game version and the ship card it trained on (for ratings keys);
    - the step counters;
    - the commit and whether the tree was dirty;
    - the time;
    - each file's SHA-256;
  - `weights.safetensors`.
- **A training checkpoint** adds:
  - to `checkpoint.json`: the kind (`"training"`) and the full resolved `TrainConfig`;
  - `training.safetensors`: the optimizer state, the return scale, the league state (once
    section 8 exists), the generator states, the rollout carry (game state, recurrent
    state and controller state of every environment), and the diagnostics rules' state
    (once section 9 exists);
  - the non-tensor optimizer state (parameter groups, step counts) goes in
    `checkpoint.json`.
- **The functions:**
  - `save_policy(directory, model, ...)` and `load_policy(directory, device) -> model`.
    Loading builds the model from the file's own build record, never from the current
    configuration;
  - `save_training(directory, trainer)` and `load_training(directory, device) -> trainer`;
  - a training checkpoint is also a policy file: `load_policy` reads the policy part of
    either kind.
- **Atomic writes:** write into `<name>.tmp/`, flush every file, rename to `<name>/`, then
  flush the parent directory. A crash at any point leaves the previous checkpoint intact
  and loadable.
- **Loading checks**, failing with a message that names what differs:
  - the hashes;
  - the format version;
  - the `Obs` and `Action` schema versions against the code's;
  - the model's version against the code's.

## 4. Models: registry, version, build record

- **The registry** maps a name to a class: `yemong`, `relational` (and `mlp` while it
  exists). A model registers itself with a decorator. Loading goes by name, never by a
  Python class path.
- **Each model class has `VERSION`,** an int, starting at 1.
- **Each model provides `build_record()`:** a JSON-serialisable record of everything its
  constructor takes:
  - its config dataclass;
  - the numbers it derives from the rules and the ship card. Yemong takes the tick
    length, the muzzle speed, the bullet moves, the hit radius and the number of ship
    types.
- **Each model provides `from_build_record(record)`,** the inverse. Today Yemong's
  constructor takes `(config, rules, types)`. Split it so that the derived numbers are
  computed once, outside, into the record, and the constructor takes only the record's
  contents. The game's defaults at construction then cannot leak into a loaded model.
- **The version rule** (`runs.md`): a new option's default reproduces the old behaviour,
  or the version goes up. Document it in `docs/contributing/code-style.md` (Contracts and
  types).
- **The compatibility set:** `tests/runs/checkpoints/<model>-v<VERSION>/`.
  - It holds one tiny policy file per model at its current version: the smallest
    configuration, a few kB.
  - Beside it go the outputs of its single-step forward on a fixed observation and
    carry, with their own seed.
  - A test loads each one, runs the step, and compares (float32, with the tolerance
    stated).
  - A behaviour change without a version bump fails it. A version bump replaces the
    model's entry, and the commit says why.

## 5. The run directory

- **`runs/<run>/`** at the repository root (configurable, `--runs-directory`), ignored by
  git.
- **The run name:** `<YYYY-MM-DD>-<HHMM>-<preset or mode>`, plus a suffix `-2`, `-3`, …
  if it exists **(proposal)**. `--name` overrides it.
- **`run.json`**, rewritten atomically on every change:
  - the name and the status (`running`, `stopped`, `finished`, `failed`), with the
    process id and the host;
  - the command line;
  - the W&B run id;
  - the newest checkpoint and the memory probe's peak;
  - **segments:** one entry per process that ran the run, with the commit, whether the
    tree was dirty, the start step and time, the end step and time, and the run-time
    overrides it applied (section 6).
- **`config.json`:** the resolved `TrainConfig` at the start. Overrides are recorded in
  the segments, not by rewriting this file.
- **`checkpoints/step_<environment steps>/`:** training checkpoints, every N environment
  steps.
  - The default N gives about 30 minutes at the preset's measured throughput, and the
    run records N.
  - Kept: the newest 2 plus the final one **(proposal)**.
- **`league/`:** the league's policy snapshots (section 8 writes them; retention is the
  league's).
- **`metrics.jsonl`:** section 9's local sink (this section creates the path).
- **`artifacts/<kind>/<id>/`:** offline results. Each has `result.json` with the
  provenance (section 8 of this spec).
- **Stopping:**
  - Ctrl-C (SIGINT) finishes the current update, writes a checkpoint, sets `stopped` and
    exits;
  - an exception writes `failed`, with the message, before it propagates;
  - a process killed outright leaves `running`. `bnb runs` shows it as `running (stale)`
    when the process id is not alive on this host.

## 6. The command line

| Command | What it does |
|---|---|
| `bnb train [options]` | A new run. |
| `bnb train --resume RUN` | Continue from the newest checkpoint, with every setting from the checkpoint. |
| `bnb train --resume RUN --at STEP` | Continue from an older kept checkpoint. Newer ones are kept, and the segment records the rewind. |
| `bnb train --from RUN [--at STEP]` | Fork: a new run with the code's current settings plus the command line. It loads the checkpoint's model, optimizer and return scale, and refuses unless the model's name, version and build record match. |
| `bnb train --init-from POLICY` | Warm start: a new run whose model starts from a policy file's weights and return scale, with a fresh optimizer. The BC → RL hand-off. Refuses unless the build record matches the run's model. |
| `bnb train --print-config` | Resolve and print the configuration; allocate nothing. |
| `bnb train --probe` | Run the memory probe and exit. |
| `bnb runs` | List runs: name, status, newest step, start time, last segment's commit. |
| `bnb runs show RUN` | `run.json`, readable: the segments, overrides and checkpoints. |
| `bnb runs upload RUN [--at STEP] [--league]` | Upload a chosen checkpoint (or the league's snapshots) as a W&B artifact, named for the run and step. |

- **Run-time overrides on resume:**
  - the update budget;
  - the learning rate and the warm-up;
  - the loss coefficients and the entropy;
  - the logging, diagnostic and checkpoint cadence.

  Each is recorded in the new segment with its step. A structural option (the model, the
  game and ship card, the fleet sizes and the training mix, the environment count, the
  rollout length) given with `--resume` fails, and the message points to `--from`.
- **A different commit on resume:** a warning, and the new segment records it. A
  model-version mismatch still refuses (section 3).
- **The W&B run:** a resume continues the same W&B run (the recorded id). A fork starts a
  new one, and its config names the parent run and step.
- **Other upload targets** (the Hugging Face Hub, Git LFS) are not built now. `upload`
  takes `--to wandb` as its only value for now **(proposal)**.

## 7. The memory probe

- **The run's first update is the probe.** After it, read the allocator's peak
  (`torch.cuda.max_memory_allocated` and the reserved peak).
  - Record both in `run.json`.
  - If the reserved peak is over the budget, stop with a message naming the peak, the
    budget and the sizes that set them (environments, micro-batch rows). The budget is
    the device's total memory less a margin, 0.5 GB by default **(proposal)**.
- **`--probe`** runs the setup and one update, prints the peaks, and exits without
  writing a run.
- **On the CPU,** the probe records nothing and never stops a run.

## 8. Provenance for offline results

- **`provenance()`** returns:
  - the commit and whether the tree was dirty;
  - the host and the GPU name;
  - the Python, torch and CUDA versions;
  - the command line;
  - the time.
- **`write_result(run, kind, payload, inputs)`** writes `artifacts/<kind>/<id>/result.json`
  with the provenance, the payload and its inputs, each policy file by path and hash.
  `bnb ratings` and head-to-head (section 8), and clips (section 10), use it.

## 9. Acceptance

1. **Round trips:**
   - `save_policy` then `load_policy` gives a model whose step equals the original's
     bitwise, on CPU and GPU;
   - the same for a training checkpoint's every part (optimizer, return scale,
     generators, carry).
2. **Two widths:** a width-128 and a width-256 Yemong (tiny otherwise) saved, loaded in
   one process, and each run one step against the other in a match.
3. **The exact resume:** on the GPU, with the compiled, graph-captured rollout:
   - train K updates;
   - stop and save, resume in a fresh process, and train K more;
   - the parameters and the logged metrics equal an uninterrupted 2K-update run,
     bitwise where the stack is deterministic, otherwise within float32 tolerance with
     the reason written down.

   Also on the CPU.
4. **Refusals, each with its message:**
   - a hash mismatch;
   - a schema-version mismatch;
   - a model-version mismatch;
   - an unknown config field;
   - a structural override on `--resume`;
   - a fork whose build differs.
5. **Default-fill:** a checkpoint whose saved config lacks a field loads with that
   field's default, and the load reports it.
6. **The compatibility set** exists for every registered model, and its test passes.
   Changing a model's computation (in a scratch copy) fails it.
7. **Atomicity:** a write killed at each stage (simulated by raising between files)
   leaves the previous checkpoint loadable and no `step_N/` directory half-written.
8. **The run record:** `run.json` across new, stop by SIGINT, resume with an override,
   resume on a different commit (simulated), fork and failure; `bnb runs` and
   `bnb runs show` print them; a stale `running` is detected.
9. **The warm start:** `--init-from` with a BC policy file starts RL with BC's weights and
   return scale and a fresh optimizer (the scripted-BC spec's hand-off).
10. **The probe:** it records the peaks, and a configuration over a budget set low for the
    test stops with the message.
11. **Uploads** are tested against W&B's offline mode only; no test touches the network.
12. **Cost:**
    - a checkpoint's size and its save and load times at the 5v5 training cells and the
      26-ship cell, rollout carry included;
    - training throughput with checkpoints against without, at the default cadence.
      Report all of these.
13. **Docs:**
    - replace the section's in-progress page with an explanation (runs, checkpoints,
      versions and the compatibility set, resume and fork);
    - add a guide, "Resuming, forking and sharing runs";
    - set `docs/roadmap.yml`;
    - follow `docs/contributing/writing-docs.md`.
