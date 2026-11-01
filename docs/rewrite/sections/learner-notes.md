# Section 6 (the learner): notes

Branch `section/learner`. The spec (`learner-spec.md`) is implemented in `bnb.learn`
(`ppo`, `returns`, `losses`, `labels`, `shots`), `bnb.runtime` (`store`, `rollout`, `train`)
and `bnb.models.relational`. MEASUREMENTS_STATUS

## Design

### Layout

| Module | Job | Lines |
|---|---|---:|
| `learn/ppo.py` | `LearnerConfig`; `Learner`: the per-shard returns, the pass over the views (token counts, recomputed old log-probabilities), the labels of a micro-batch, the micro-batch loss, `gradient` (one minibatch, micro-batch by micro-batch), `update`, the drift and explained variance | LINES_PPO |
| `learn/returns.py` | per-level GAE, team mixing with the level sum | LINES_RETURNS |
| `learn/losses.py` | the clipped surrogate, cross-entropies, the critic's per-level losses, the shot cross-entropies; masked sums over a given denominator | LINES_LOSSES |
| `learn/labels.py` | next-state targets and mask, enemy actions and mask, the density counts, the privileged rows | LINES_LABELS |
| `learn/shots.py` | `ShotCapture`, `Shots`; `tick` (one tick of the scan) and `resimulate` | LINES_SHOTS |
| `runtime/store.py` | `SegmentStore`: pinned host memory or the device; asynchronous writes from the ring, staged micro-batches with prefetch | LINES_STORE |
| `runtime/rollout.py` | the staging ring, the label records (privileged rows, every row's action, the shot capture), `bootstrap` | LINES_ROLLOUT |
| `runtime/train.py` | `TrainConfig`, `Trainer` (shards, pool metrics), `win_rate`, `set_precision`, the learner's stage cut | LINES_TRAIN |
| `models/relational.py`, `controllers/relational.py` | the toy model and its controller | LINES_RELATIONAL |

### The data path

- **Records.** Each tick the stages write the learner's records into a **staging ring of two
  graph blocks** on the device (`Rollout.ring`, 2 x 32 ticks), indexed by `t mod ring`:
  - act: the observation parts, every row's action (view order), logp, value, values,
    outcome, the replay, `alive (R, N)` and `valued (R, S)`, and the enemy rows of the
    fog-off privileged view (`truth`);
  - advance: the shot capture, the game's launch state of every row at tick start and the
    tick's ship displacement;
  - resolve: reward and done.
- **Drain.** After each graph replay (one block) the block's slots are copied into the
  `SegmentStore` on a copy stream, waiting on an event; before a replay writes those slots
  again the compute stream waits on that copy. The segment's initial carry and the truth of
  the observation after its last tick (`final_truth`) are written the same way. On a CPU
  run every copy is a plain copy, so the same code runs everywhere.
- **The store.** Per-tick leaves `(L, V, ...)` and per-segment leaves `(V, ...)`, `V = shards
  x R`, in pinned host memory (`host_store`, the training default) or on the device (the
  all-GPU path, used by the tests and the budget comparison).
- **Staging back.** `SegmentStore.batches` gathers a micro-batch's views on the host
  (`index_select` into a pinned buffer), copies it to a device buffer on the copy stream,
  and gathers and copies the next one while the caller computes. Buffers alternate in pairs
  per micro-batch size, each with the event of the copy that filled it and of the compute
  that last read it.
- **Bootstrap.** `Rollout.bootstrap` runs the learner's controller (compiled like the act
  stage) on the current views and carry without acting; that observation is the next
  segment's first (tested: its values equal the next segment's first stored values).

### Compaction (spec 2)

- **The replay keeps the enemy rows only** (Yemong's `Replay.valid/age/means/spreads`
  are `(R, N - S, ...)`). An own ship is always visible, so its row is the observation's
  shifted truth, valid, age 0, no spread. `inputs.believed` rebuilds the full rows and both
  `step` and `unroll` compose from it, so step reads exactly what the replay holds.
- **The privileged rows keep the enemy rows only.** An own row of `Privileged.ships` equals
  the observation's row (tested); `labels.privileged_rows` rebuilds them.

### Returns (spec 3)

Per shard, from the small records staged whole: values in reward units (the outcome column
times the outcome weight), per-level GAE with `next_done = done[L - 1]` (the fixtures pinned
that both the last step's end and `next_done` cut the bootstrap), team mixing over the
view's own ships, the outcome targets (`critic.outcome_targets` with the outcome's discount
and trace decay; the view's result is the sign of its outcome reward). The return scale is
updated once per batch over every shard's summed returns of living own ships.

### Losses (spec 4)

Every term is a masked sum over its tokens divided by its minibatch total, so the
micro-batches of a minibatch add up to the unsplit minibatch. The token counts per view are
computed before the epochs: the policy, valued, enemy and step counts from the small records,
the next-state and shot counts in one pass over the views (the shot validity needs the
re-simulation). A minibatch's denominators are its views' sums.

- policy: own ships alive at t; entropy: the same tokens, coefficient -0.0081 on the mean
  entropy;
- value: the per-level critic losses summed (level weights 1);
- next state: every row valid at t (own rows always) at a step that did not end an episode,
  against the sharp code of the truth at t + 1 shifted by the view's offset;
- enemy action: enemies alive at t;
- density: per view-step, the Poisson terms minus their saturated value summed over cells,
  over the view-step's valued-ship count, averaged over view-steps;
- shots: valid shots of own ships and of enemies visible at t; the trajectory term averages
  each shot's parts over its valid ages.

### Shot re-simulation (spec 6)

- **Capture.** `bnb.game.physics.bullets.launch_state` exposes what `fire` reads; the
  rollout stores it with the tick's ship displacement and alive flag (`ShotCapture`), in the
  view's row order.
- **Scan.** One bullet per (row, age), newest first, as the game keeps one per slot. Each
  tick every row fires a fresh bullet with the game's `launch`, every bullet moves with the
  game's `move_bullets`, and the game's `contacts` and `first_contact` test it against the
  tick's captured ship motion. Every bullet flies all 12 moves; a shot's first contact is its
  hit (where the game spends the bullet), its later ages invalid. The closest pass is the
  smallest miss distance over the flight, over living ships but the shooter.
- **Validity.** The shooter was alive, and the flight (to its hit, or all its moves) ends
  inside the segment with no episode end before its last move.
- **Spread.** One standard-normal draw per shard from the learner's own generator, so each
  epoch sees the same labels.
- **Equality with the game** is bit for bit (tested on 32 environments of a crowded 3v3 with
  fields and episode ends: 10k+ snapshots, every hit, both sides).
- The per-tick step is a stage the runtime compiles (357 ms to 41 ms per 16-view
  micro-batch at 5v5).

### Compilation and settings

- No `torch.compile` in `bnb.learn`: the `Learner` takes a `stage` function from the
  runtime (`runtime.train.compiler`), which compiles the micro-batch loss and the
  re-simulation tick (`fullgraph=True, dynamic=False`).
- No process-wide settings in library code: the entry point (`bnb.cli`) seeds the process and
  calls `set_precision`. `train()` no longer seeds.
- Precision: float32 by default; `TrainConfig.precision` is "tf32" (the entry point's
  matmul precision) or "bfloat16" (the model's `bfloat16_matmuls`). Recurrent state, norms,
  scans and every loss reduction stay float32 (the losses read float32 logits).

### The toy learner (spec 9)

`bnb.models.relational.Relational`: per own ship, its own state (15 columns, including its
position relative to the arena centre in its heading frame) and, for each other row (allies
then enemies, row order), position and velocity relative to it in its heading frame,
distance, relative heading (cos, sin) where visible, and team, alive, visible (10 each). Two
GELU layers of 256, the joint 42-way policy, the per-ship categorical critic, and the
outcome from the mean of the own ships' hidden states. Stateless; `unroll` is the forward
over every step. Heads start at zero (a uniform policy).

## Decisions and deviations

1. **Labels are built per micro-batch, not stored per shard.** Spec 2 says everything
   derived is computed on the GPU one shard at a time. The returns, the return scale and the
   token counts are (per shard, once per update). The labels are rebuilt per micro-batch
   from the stored rows, because expanded they do not fit: at 5v5 and 1536 views the
   next-state codes alone are 3.9 GB per shard, the shot trajectory codes 1.5 GB. The shot
   spread is drawn once per shard, so every epoch sees the same labels. Cost: the
   re-simulation runs once in the counting pass and once per epoch (41 ms per 16-view
   micro-batch, compiled).
2. **The token counts need a pass over the views before the epochs.** A minibatch's
   denominators must be known before its first micro-batch; the shot validity needs the
   re-simulation, so the counting pass re-simulates every view once. The recomputed old
   log-probabilities ride on the same pass.
3. **The replay and the privileged rows keep the enemy rows only** (spec 2's compaction,
   read with static shapes). "Hidden rows only" would need a data-dependent gather; the
   enemy rows are the only rows that can be hidden, so the fixed-shape version stores them
   and rebuilds the own rows. Visible enemy rows are still stored (their entries equal the
   observation's truth).
4. **Every row's action replaces the own action in `Trajectory.action`** (now `(L, R, N)`,
   view order), rather than a second field: the learner's are `[..., :S]`.
5. **`alive (L, R, N)` and `valued (L, R, S)` are recorded beside the observation,** so the
   returns and the token counts read 15 bytes per view-tick instead of staging the whole
   observation per shard.
6. **The view's result** for the outcome targets is the sign of its outcome reward (the
   outcome weight is positive); the spec stores no result.
7. **`next_done` for the GAE is `done[L - 1]`**: the fixtures pinned that the last step's cut
   is `~terminated[L - 1] & ~next_done`, and in the rollout the bootstrap observation's
   reset is exactly `done[L - 1]`.
8. **Density:** the per-cell term minus its saturated value (zero at a perfect forecast; the
   same gradient), summed over cells, over the view-step's valued-ship count, averaged over
   view-steps.
9. **The KL early stop** checks each minibatch's approximate KL, `mean((r - 1) - ln r)` over
   living own ships, after its step, and stops the update there (one host sync per
   minibatch). The reported clip fraction and KL are the first epoch's means.
10. **Uneven minibatches.** Where the views do not divide into 32 minibatches (the 26-ship
    bench cell has 2364 views), `tensor_split` makes them as equal as possible, and each
    minibatch's last micro-batch is smaller. The compiled loss then has at most three
    shapes. The pinned staging keeps a pair of buffers per size.
11. **Adam with torch's defaults** (eps 1e-8; the skeleton used 1e-5): the spec says only
    "Adam at 3e-4".
12. **The toy learner plays team deathmatch's own clock (300 s).** With 60 s games most ended
    drawn (40%); at the mode's clock the same run wins 96-98%.
13. **Progress metrics:** reward per level and episode results per pool; the losses, KL, clip
    fraction and explained variance per level are over the whole batch (spec 7 can be read
    either way).
14. **Shots of a dead ship** are invalid (the game's `fire` needs a living ship); a ship dead
    at tick start is not a target (the game's reach is -1 for it).
15. **The learner's compiled stages** (the micro-batch loss, the re-simulation tick) are cut
    in `runtime.train.compiler`, beside the rollout's tuple of stages; there is no
    `runtime/stages.py` yet.
16. **Names:** new code uses `discount` and `trace_decay` for gamma and lambda,
    `log_probability`, `approximate_kl`; the fixtures' names stay in the adapter.

## Measurements

MEASUREMENTS

## Changes outside `bnb.learn`, `bnb.runtime` and `bnb.models.relational`

- **`bnb.game.physics.bullets`:** `Launch`, `launch_state` and `launch` (what `fire` reads,
  and the bullet it fires); `fire` composes them. Behaviour unchanged: every game test and
  fixture passes untouched.
- **`bnb.game.physics.combat`:** `Contacts`, `contacts` and `first_contact` (the swept test
  and the first-contact rule); `hits` composes them.
- **`bnb.contracts.trajectory`:** `Trajectory.action` holds every row; new `alive`,
  `valued`, `truth`, `final_truth`, `capture` (all default None).
- **`bnb.models.yemong.inputs` and `model`:** the replay's enemy rows; `believed` rebuilds
  the own rows for `compose`.
- **`bnb.controllers.relational`** (new): the toy model's controller.
- **`bnb.cli`:** `--shards`, `--model`, `--precision`, `--old-log-probabilities`,
  `--micro-batch`; it seeds the process and sets the matmul precision (moved out of
  `runtime.train.train`).
- **Tests:** `tests/sections/test_yemong_model.py` (replay rows), `yemong_cases.py`
  (`joint_actions` takes the own rows), `test_mixed.py` (the trainer test on the new
  learner), `tests/loop/test_learning.py` (the toy learner); new `learner_cases.py`,
  `test_learner.py`, `test_learner_shots.py`, `test_learner_streaming.py`;
  `tests/reference/learner/learner_arm.py`.
- **`tests/docs/test_references.py`** (outside my list, forced): it rendered the status block
  for `explanation/in-progress/learner.md`, which the coordinator asked me to replace; it
  now uses the league's in-progress page.
- **Docs** (coordinator's request): `docs/explanation/learner.md` (new; the in-progress
  page removed), the training loop, the training guide (with a toy-learner section), the
  first run, architecture, contracts, Yemong, the writing guide, the reference intros,
  `packages.yml`, `roadmap.yml` (status built), `mkdocs.yml`'s nav, and the hook
  (`learner-config` replaces `ppo-config`; a "built" status).

## Requests and open questions

1. **`contracts.md` current schema:** `Trajectory.action` (every row), `alive`, `valued`,
   `truth`, `final_truth`, `capture`; `Replay`'s enemy rows; `bnb.game.physics.bullets.Launch`.
2. **The skeleton MLP and `ModelController`** are now used only by
   `tests/contract/test_export.py` and `test_contracts.py`, which I may not change. Proposal:
   point those tests at the relational model and Yemong, then delete the MLP,
   `models/heads.py` and `controllers/model.py`.
3. **`done` and `reset`:** the learner reads `reset` off the observation; the contracts'
   `done` fields are unchanged for the cleanup pass.
4. **A runtime stage list.** The scaffold puts the cut list in `runtime/stages.py`; the
   rollout's tuple and `runtime.train.compiler` are the two lists today.
5. **Spec 7's metrics per pool:** confirm whether the losses should also be split by pool.
6. **Packed hidden rows** (decision 3): if streaming bytes matter later, a packed layout of
   the hidden enemy rows would save the visible enemy rows' means and spreads.

## Size

SIZE

## Progress (working state, 2026-11-01, evening)

- **Done and tested:** everything in the design above; `origin/v2` merged; the docs
  updated (`tests/docs` passes); the full suite passed (315) before the last performance
  changes (host view indices, compiled label codes, the staging worker thread), which
  passed their own tests.
- **In flight:** `/tmp/bnb-scratch-learner/chain3.sh` (streaming cost, then the precision
  matrix's Yemong rows) into `chain3.log`, then `chain2.sh` (toy curves for the matrix, the
  four 5v5 training cells and the 26-ship cell through `bench.cell`) into `chain2.log`.
- **Next:** fill the measurement tables below from those logs; rerun the full suite under
  the lock; hand back.
