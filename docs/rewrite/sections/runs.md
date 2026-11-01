# Section 11: runs (checkpoints, resume, run records)

Status: decisions draft, 2026-11-01. The spec follows once the open questions are
answered. Background: the gap analysis (`../audits/2026-11-01-v1-gaps.md`, gap 1).
**(proposal)** marks a choice made here for the author to confirm.

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

## Open questions

1. **A newer code version adds a model option.** Older checkpoints do not have the
   field. When they load:
   - **(a)** take the option's default and log that it was missing **(proposal)**.
     The style guide then requires that a new option's default reproduces the old
     behaviour;
   - **(b)** fail, so nothing is ever loaded with a value it was not trained with.
2. **Resuming with a changed configuration:**
   - **(a)** refuse every change except the budget (more updates) **(proposal)**. A
     change of anything else is a fork (`--from`), so a run's record stays one
     configuration;
   - **(b)** allow any override, and record it in `run.json`.
3. **Off-machine storage:**
   - **(a)** local only, with a `bnb runs upload RUN` command that sends chosen
     checkpoints to W&B artifacts **(proposal)**;
   - **(b)** automatic upload of every final policy and league snapshot;
   - **(c)** the Hugging Face Hub instead of W&B (more natural for a public release).
4. **Saving the rollout carry** (game state, recurrent state and controller state of
   every environment) for an exact resume:
   - **(a)** save it **(proposal)**. A resume then continues mid-episode, bit for bit
     where the stack is deterministic. It costs tens of MB per checkpoint at training
     sizes (to be measured);
   - **(b)** do not; a resume starts every environment on a new episode. It is smaller,
     but not identical to an uninterrupted run, which the league spec's resume test
     requires.
