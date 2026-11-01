# Section 11: runs (checkpoints, resume, run records)

Status: decisions settled 2026-11-02 (drafted 2026-11-01); the spec follows. Background:
the gap analysis (`../audits/2026-11-01-v1-gaps.md`, gap 1). **(proposal)** marks a
choice made here that the author has not overruled.

## Already decided

- **A section owns checkpoints and runs** (author, 2026-11-01). The league (resume), BC
  (the hand-off to RL), diagnostics (A9, resume) and the play shell (loading agents) all
  depend on it.
- **Every checkpoint carries its own settings** (author, 2026-11-01). It holds
  everything needed to rebuild what it saved, without the current code's defaults or the
  current run's configuration. For example, a checkpoint of a width-128 model and one of a
  width-256 model load side by side in one process.
- From `decisions.md`:
  - the frozen configuration is serialised with each checkpoint and logged to W&B;
  - no fingerprints or drift checks;
  - checkpoints live outside git;
  - safetensors plus JSON was the lean;
  - one root seed, with named streams.
- From the scaffold (section 5): one memory probe at startup, which fails fast over
  budget.
- From the league spec: snapshots every N environment steps; retention (the newest 8,
  plus ages 8, 16, 32, …); a resumed run continues identically.

## Proposals

### 1. Two kinds of saved thing

- **A policy file:** what an agent needs to play:
  - the model's name;
  - its build record (section 2);
  - its weights;
  - the `Obs` and `Action` schema versions it was trained on;
  - for information, the game version and ship card it trained on (ratings are keyed by
    game version).

  League snapshots, `bnb ratings`, head-to-head, the play shell and ONNX export all read
  policy files.
- **A training checkpoint:** a policy file plus everything a resume needs:
  - the optimizer state and the return scale;
  - the league (entries, ratings, records, sampling state);
  - every named generator stream's state;
  - the step counters;
  - the diagnostics rules' state;
  - the rollout carry (open question 4).

### 2. The build record

- **Each model declares a build record:** a JSON-serialisable record of everything its
  constructor takes:
  - its config dataclass (`YemongConfig`: width, blocks, heads, …);
  - the numbers it derives from the rules and the ship card. Yemong takes the tick
    length, the muzzle speed, the bullet moves, the hit radius and the number of ship
    types.
- **Loading rebuilds the model from the record,** never from the current `Rules` or the
  current defaults. So:
  - a width-128 and a width-256 checkpoint load together;
  - a model saved before a ship-card change keeps the code scales it learned with.
- **A registry** maps the model's name (`yemong`, `relational`, `mlp`) to its class.
  Loading never depends on a Python class path, so renaming a class or a module (the
  cleanup pass renames things) does not break old checkpoints.
- **The league's stacked slots** need identical shapes (one vmapped forward). Only
  policies with the live model's build record can share them. Different builds play each
  other in head-to-head, ratings and the play shell. A league that mixes architectures,
  with one slot group per build, comes later if wanted.

### 3. The file format

- One directory per checkpoint:
  - `checkpoint.json`:
    - the format version and the kind (policy or training);
    - the build record;
    - the full resolved configurations;
    - the schema versions;
    - the step counters;
    - the commit and whether the tree was dirty;
    - the time;
    - each file's hash;
  - `weights.safetensors`: the model;
  - `training.safetensors`: the rest of a training checkpoint (optimizer, return scale,
    generator states as bytes, league tensors, carry).
- **Not `torch.save`:**
  - pickle runs code when it loads;
  - it ties a file to Python class paths;
  - other tools cannot read it.

  safetensors loads no code, and can be read lazily (memory-mapped) and from any
  language.
- **Atomic:** write into `step_N.tmp/`, flush, then rename. A crash mid-write leaves the
  previous checkpoint intact.
- **Loading checks the hashes,** and checks the schema versions against the code's. A
  mismatch fails with a message naming both versions.
- **A frozen dataclass round-trips through JSON** exactly (nested dataclasses, tuples,
  numbers, strings), with one small helper in `bnb.contracts`. An unknown field fails;
  for a missing field, see open question 1.

### 4. The run directory

- `runs/<run>/`, ignored by git:
  - `run.json`:
    - the name;
    - the status (running, finished, failed);
    - the command line;
    - the commit and whether the tree was dirty;
    - the machine;
    - the W&B run id;
    - the newest checkpoint;
    - the memory probe's peak;
  - `config.json`: the resolved configuration;
  - `checkpoints/step_<environment steps>/`: training checkpoints;
  - `league/`: the league's policy snapshots;
  - `metrics.jsonl`: section 9's local sink;
  - `artifacts/<kind>/<id>/`: offline results (ratings, head-to-head, clips). Each has a
    `result.json` with the commit, the machine, the command and its inputs (the policy
    files' hashes).
- **The run name:** `<date>-<time>-<preset>`, for example `2026-11-02-0930-mixed`.
- **`bnb runs`** lists runs from their `run.json` files: name, status, newest step,
  start time.
- **Retention:**
  - a training checkpoint every N environment steps, by default about every 30 minutes
    at the preset's throughput;
  - keep the newest 2 plus the final one;
  - league snapshots follow the league's retention.

### 5. The command line

| Command | What it does |
|---|---|
| `bnb train ... ` | A new run. |
| `bnb train --resume RUN` | Continue a run from its newest checkpoint. |
| `bnb train --from RUN [--at STEP]` | Fork: a new run that starts from another run's checkpoint (weights, optimizer, return scale, league). |
| `bnb train --init-from POLICY` | A warm start: weights and return scale only, a fresh optimizer. This is the BC → RL hand-off. |
| `bnb train --print-config` | Resolve and print the configuration; allocate nothing. |
| `bnb runs` | List runs. |

### 6. The memory probe

- **Before the first update,** run one full update and read the allocator's peak.
- **Record it** in `run.json`. If it is over budget (device memory less a margin), fail
  before training starts, with the peak and the budget in the message.
- **One measurement of the configured run.** No search over sizes, and no cache per
  machine: v1's tiered probe is dropped.

### 7. Order

- **After section 6 merges.** It is small: about the size of a section's tests.
- **Alongside the scripted agent and BC:** the BC hand-off calls this section's
  `save_policy` and `load_policy`, which the spec defines first. That agent writes its
  hand-off against the interface and finishes it once section 11 merges.
- **Before the league (section 8),** whose resume test needs it.

## Decided (author, 2026-11-02)

- **Storage:** checkpoints are local by default. Only chosen checkpoints are uploaded,
  by an explicit command: to W&B artifacts, the Hugging Face Hub, or the repository
  through Git LFS. Nothing uploads automatically.
- **A full resume:** a training checkpoint holds the rollout carry (game, recurrent and
  controller state of every environment), so a resumed run continues exactly where it
  stopped.
- **League opponents** share the live policy's architecture (league spec, section 6).

## Decided (author, 2026-11-02): loading and resuming

### Model versions and old checkpoints

- **Each model carries a version** in its code (for example `Yemong.VERSION`), and every
  checkpoint records it.
- **Loading:**
  - same version: the checkpoint loads. Options it lacks (added later) take their
    defaults;
  - different version: it refuses, with both versions in the message. The way back is a
    worktree at an older commit;
  - no migration code, and no editing of checkpoint files.
- **The rule for a new option:** its default must reproduce the old behaviour. If it
  cannot, the change alters what old weights compute, and the version goes up. A change
  of that kind includes:
  - a different normalisation;
  - a reordering of input columns;
  - a new part that is always on.
- **Refactors do not matter:** loading goes by the model's registered name, never by a
  Python class path.
- **The test that keeps it honest:** the repository keeps a few tiny checkpoints, one per
  model version, with their outputs on a fixed input.
  - The suite loads each one and checks the outputs.
  - A behaviour change without a version bump fails it.
  - A version bump retires the old version's checkpoint and adds a new one.

### Resume and fork

- **`--resume RUN`** uses only what the run's checkpoint holds: every setting, not the
  code's current defaults.
  - **Run-time overrides on the CLI** are allowed: the update budget, the learning rate,
    loss coefficients, entropy, logging and checkpoint cadence. Each is recorded in
    `run.json` with the step it took effect at.
  - **Structural settings** cannot change on a resume: the model architecture, the game
    rules and ship card, the fleet sizes and the training mix, the environment count, and
    the rollout length.
- **`--from RUN [--at STEP]` (a fork)** starts a new run with the code's current settings
  (plus the command line), loading the checkpoint's state where it still fits.
  - The model's weights load only if its version and build match. If they do not, the
    fork refuses: a new architecture is a new run, or a warm start from a policy file of
    the same build.
- **Code changes since the checkpoint:** a resume on a different commit goes ahead, with
  a warning. `run.json` records each commit the run ran on and the step it started at,
  for example "steps 0 to 412M at abc123, from 412M at def456". Model-version mismatches
  still refuse, as above.
