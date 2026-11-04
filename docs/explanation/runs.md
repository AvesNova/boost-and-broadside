---
covers: [src/bnb/runs/, src/bnb/runtime/rollout.py, src/bnb/models/yemong/model.py, src/bnb/models/relational.py, src/bnb/models/mlp.py, tests/runs/]
reviewed: 2026-11-03 16:00
---

# Runs and checkpoints

<!-- generated: section-status -->

A training run lives in a directory of its own, and everything it saves carries the
settings it was made with. `bnb.runs` holds the run directory and its record, policy files
and training checkpoints, the model registry, the memory probe, the provenance of offline
results and explicit uploads. It sits between `bnb.runtime` and the league in the layer
order. How to use it is in the guide
[Resuming, forking and sharing runs](../guides/resuming-forking-sharing.md).

## The run directory

`bnb train` writes a run under `runs/` at the repository root (`--runs-directory`
moves it), which git ignores:

<!-- not-runnable: a directory layout -->
```text
runs/2026-11-02-mixed-purple-pumpkin/
  run.json              the record
  config.json           the resolved TrainConfig at the start
  checkpoints/step_<environment steps>/   training checkpoints
  league/               the league's policy snapshots
  policy/               a BC run's clone, a policy file (bnb train --objective bc)
  metrics.jsonl         the local metrics sink
  artifacts/<kind>/<id>/result.json        offline results
```

- **The name** is the date, the preset (or the mode), and an adjective and a noun from the
  small lists in `bnb.runs.records` (`ADJECTIVES`, `NOUNS`), drawn again while a run of
  that name exists. `--name` sets it.
- **`run.json`{.plain}** (`bnb.runs.records.Run`) holds the status (`running`, `stopped`,
  `finished` or `failed`, with the message of a failure), the process id and host, the
  command line, the W&B run id, the checkpoint cadence, the newest and final checkpoints,
  the memory probe's peaks, the parent of a fork or a warm start, and one **segment** per
  process that ran the run. A segment records the commit and whether the tree was dirty,
  the start and end steps and times, the run-time overrides it applied, and a rewind. The
  file is rewritten atomically (a flushed temporary file renamed over it) on every change.
- **A stale run:** a process killed outright leaves `running`. `bnb runs` shows such a run
  as `running (stale)` when its process id is not alive on this host. A resume of a run
  whose process is alive on this host is refused, so two processes never write one run.
- **Stopping:** Ctrl-C finishes the current update, writes a checkpoint and marks the run
  `stopped`; an exception marks it `failed` before it propagates
  (`bnb.runs.session.train`).

## Policy files and training checkpoints

Both are directories of safetensors files and one `checkpoint.json`{.plain}, never
pickle: safetensors runs no code when it loads, ties a file to no Python class path, and
other languages read it.

- **A policy file** (`bnb.runs.checkpoint.save_policy`) is what an agent needs to play: the
  model's registered name, version and build record, the `Obs` and `Action` schema
  versions, its weights, the return scale's state tensors and the step counts (for a warm
  start), the game and ship card it trained on and the game's version (`bnb.game.VERSION`,
  which rises whenever the game computes differently), the commit and the time, and each
  file's SHA-256.
- **A training checkpoint** (`bnb.runs.checkpoint.save_training`) is a policy file plus
  the full `TrainConfig`, the optimizer, the opponent slots, the league's state (its
  entries, gauges, sampling state, and the averages' and snapshots' weights), every random
  generator, the rollout carry (the game state, every controller's carry, the seats, the
  reward ledger) and the step counters. `bnb.runs.checkpoint.load_policy` reads the policy part of
  either kind.
- **Writes are atomic** (`bnb.runs.checkpoint.write_checkpoint`): the files go into
  `<name>.tmp/`, each file and then the directory are flushed, the directory is renamed to
  `<name>/`, and the parent directory is flushed. After a crash, `<name>/` is either
  complete or absent; a leftover `<name>.tmp/` is never listed as a checkpoint and is
  removed by the next write. A checkpoint is never overwritten.
- **Loading checks** the format version, every file's hash, the schema versions and the
  model's version against the code's, and fails with a message naming both sides.

## Build records, versions and the compatibility set

A model is built from its **build record** alone: a frozen dataclass of everything its
constructor takes. `bnb.models.yemong.model.YemongBuild` holds the config and the numbers
Yemong derives from the rules and the ship card (the tick length, the bullet's lifetime,
the fastest muzzle speed, the smallest hit radius, the number of ship types and the state
code's ranges); `bnb.models.yemong.model.yemong_build` computes them from a game. Loading
rebuilds the model from the file's record, never from the current game or the current
defaults, so a model saved before a ship-card change keeps the code ranges it learned with,
and models of two widths load side by side in one process.

A run's model settings (`TrainConfig.yemong` or `TrainConfig.relational`) and ship card
(`TrainConfig.ship_card`) are part of its `TrainConfig`, so `config.json`{.plain} and every
training checkpoint record them; `bnb.runtime.train.model_build` turns them into the build
record. A resume, a fork and a warm start check that build against the file's
(`bnb.runs.checkpoint.check_build`): apart from the precision, they must be equal, or the
start is refused. A resume takes the settings from its checkpoint and may not change them.
A fork or a warm start whose command sets no model setting takes the file's
(`bnb.runs.checkpoint.with_model_of`), so a changed code default does not block it.

- **The registry** (`bnb.runs.registry.MODELS`) maps a model's name (`yemong`,
  `relational`, `mlp`) to its class. Loading goes by that name, so renaming a class or a
  module leaves old files loadable.
- **Versions:** each model class has `VERSION`{.plain}. A file loads only at its own version; the
  way back to an older one is a worktree at an older commit. A field the record lacks
  takes its default, and the load says so (`bnb.runs.checkpoint.FilledDefaults`). That is
  sound because of the rule in [Code style](../contributing/code-style.md#contracts-and-types):
  a new option's default reproduces the old behaviour, or the version rises.
- **The compatibility set** (`tests/runs/checkpoints/`) keeps one tiny policy file per model
  at its current version, with a fixed input and the outputs of its single step.
  `tests/runs/test_compatibility.py` fails when a model computes something else without a
  version bump.
- The serialisation (`bnb.runs.serialise`) reads JSON by the dataclass's type hints:
  floats round-trip exactly, an unknown field fails naming its path, and a missing field
  is reported.

## Resume, fork and warm start

| Start | Loads | Settings |
|---|---|---|
| `--resume RUN` | everything: model, opponent slots and the league, optimizer, return scale, generators, rollout carry, counters | the checkpoint's own, plus the run-time overrides |
| `--from RUN` (a fork) | model, optimizer, return scale, environment steps; the league starts afresh | the code's current settings plus the command line; a new run |
| `--init-from POLICY` (a warm start) | weights and return scale; a fresh optimizer | the command line, with the warm start's entropy coefficient unless it sets one; a new run |

- **A resume** (`bnb.runs.checkpoint.load_training`) may change only the fields in
  `bnb.runs.checkpoint.OVERRIDABLE` (the update budget, the learning rate and its
  warm-up, the loss coefficients and the entropy, the reward weights, and the precision
  and micro-batch rows, which change speed but not shapes) and the checkpoint cadence.
  Anything else given with `--resume` fails and points to `--from`. Each override is
  recorded in the new segment. The update budget is the run's total. A resume on another
  commit goes ahead with a warning, and the segment records the commit.
- **A fork's** update count starts at zero, so `--updates` is the number of updates the
  fork trains; its environment-step count goes on from the parent's, so the learning
  rate's warm-up does not restart.
- **A fork** (`bnb.runs.checkpoint.fork_training`) and **a warm start**
  (`bnb.runs.checkpoint.warm_start`) refuse unless the model's name, version and build
  record match what the run builds. A new architecture is a new run. The warm start is
  behaviour cloning's hand-off to RL.
- **The resume keeps the run's integrity:** a run stopped, saved and resumed in a fresh
  process continues the same run. The restore itself is exact: right after loading, before
  any compute, the parameters, optimizer state, every generator, the rollout's carry, the
  league's state, the counters and the schedule's position equal what the first process
  held when it saved, bit for bit. The test enumerates them from the checkpoint's own list
  (`bnb.runs.checkpoint.training_state`), so a new field is covered at once. On the CPU the
  continuation is bitwise too. On the GPU it is the same run within rounding, not the same
  bits (`tests/runs/test_resume.py`). Three things make the restore and the continuation
  hold:
  - the checkpoint holds every generator: the learner's named streams, and the default
    CPU and CUDA generators, which the rollout's draws and the policy's sampling inside the
    captured graph use. The entry point restores the default ones
    (`bnb.runs.checkpoint.restore_default_generators`), the one process-wide effect of a
    resume;
  - the rollout's graph capture restores the device's generator after its warm-up, as it
    restores the carry, so a process that captures again draws what the first one did;
  - the per-pool metrics are masked sums, not atomic adds, so the same data gives the same
    numbers.

  Bit equality after the restore is not promised. Yemong's kernels autotune their tiles for
  each GPU and keep the choice in that machine's cache, not in the checkpoint, and a
  different tile can round the last bits differently. Adam turns rounding in near-zero
  gradients into full steps within one update, so a one-ulp change to every weight already
  moves elementwise weights past 1e-4, and with Yemong it changes the sampled actions and so
  the rollouts. The GPU test therefore compares the entropy, the approximate KL and the
  value losses of the updates after the resume to the straight run's within a relative
  tolerance of 0.1, which a one-ulp change passes and another minibatch order fails, and
  the integers and categoricals (counters, games, results, opponents drawn) exactly over
  the first update after it. This is accepted.

  On the same GPU a long run can drift too. The game's damage sum and Yemong's density
  counts add with float atomics, whose order is not fixed: after a tick in which two hits
  land on one ship, a resumed run can differ from the original in the last bit, and the
  difference grows from there. This is accepted; the sums stay as they are.

## Retention

A training checkpoint is written every N environment steps. Without
`--checkpoint-every`, N is about 30 minutes at the run's throughput, measured on its
second update (the first compiles) and recorded in `run.json`{.plain}. The kept checkpoints
follow `bnb.runs.records.retained` (the league's snapshots have their own rule,
`bnb.league.pool.retained`). Number the checkpoints 0, 1, 2, …
in the order written (a stop's off-cadence checkpoint counts too); with n the newest and
age = n − i, checkpoint i is kept if its age is under `RECENT`, or if i is divisible by
2^⌊log₂ age⌋, and the final one is always kept. A multiple of 2^(j+1) is a multiple of
2^j, so a checkpoint kept now was kept at every earlier step, and each band of ages
[2^j, 2^(j+1)) from `RECENT` on holds exactly one checkpoint: about `RECENT` + log₂ n in
all. A deleted checkpoint is first renamed `step_<step>.deleting`, which the listing
ignores, then removed. A rewind (`--resume RUN --at STEP`) moves the newer
checkpoints into `checkpoints/rewound_<segment>/`, where they are kept.

## The memory probe

The run's first update is the probe (`bnb.runs.probe`): after it the allocator's peaks,
allocated and reserved, go into `run.json`{.plain}. If the reserved peak exceeds the budget (the
device's memory less `bnb.runs.probe.MARGIN`), the run stops with the peak, the budget and
the sizes that set them. It is one measurement of the configured run, with no search over
sizes. On the CPU it records nothing. `bnb train --probe` runs the setup and one update,
prints the peaks and exits without writing a run.

## Offline results and uploads

- **Provenance** (`bnb.runs.provenance.provenance`): the commit and whether the tree was
  dirty, the host and GPU, the Python, torch and CUDA versions, the command line and the
  time. `bnb.runs.provenance.write_result` writes an offline result (ratings, head-to-head,
  clips) into the run's `artifacts/` with its provenance, its payload and its input policy
  files by path and hash.
- **Uploads are explicit:** nothing uploads by itself. `bnb runs upload` sends a chosen
  checkpoint, or the league's snapshots, to W&B as an artifact named for the run and the
  step (`bnb.runs.upload.upload`).
