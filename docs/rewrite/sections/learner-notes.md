# Section 6 (the learner): notes

Branch `section/learner`. The spec (`learner-spec.md`) is implemented in `bnb.learn`
(`ppo`, `returns`, `losses`, `labels`, `shots`), `bnb.runtime` (`store`, `rollout`, `train`)
and `bnb.models.relational`. STATUS_LINE

## Design

### Layout

| Module | Job | Lines |
|---|---|---:|
| `learn/ppo.py` | `LearnerConfig`; `Learner`: the per-shard returns, the pass over the views (token counts, recomputed old log-probabilities), the labels of a micro-batch, the micro-batch loss, `gradient` (one minibatch, micro-batch by micro-batch), `update`, the drift and explained variance | 462 |
| `learn/returns.py` | per-level GAE, team mixing with the level sum | 56 |
| `learn/losses.py` | the clipped surrogate, cross-entropies, the critic's per-level losses, the shot cross-entropies; masked sums over a given denominator | 65 |
| `learn/labels.py` | next-state targets and mask, enemy actions and mask, the density counts, the privileged rows | 61 |
| `learn/shots.py` | `ShotCapture`, `Shots`; `tick` (one tick of the scan) and `resimulate` | 187 |
| `runtime/store.py` | `SegmentStore`: pinned host memory or the device; asynchronous writes from the ring, staged micro-batches with prefetch | 183 |
| `runtime/rollout.py` | the staging ring, the label records (privileged rows, every row's action, the shot capture), `bootstrap` | 350 (was 241) |
| `runtime/train.py` | `TrainConfig`, `Trainer` (shards, pool metrics), `win_rate`, `set_precision`, the learner's stage cut | 223 |
| `models/relational.py`, `controllers/relational.py` | the toy model and its controller | 159 + 25 |

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
- **Validity.** The shooter was alive, the shot was launched at least the longest flight
  before the segment's end (fix round, fix 4), and its flight (to its hit, or all its moves)
  has no episode end before its last move.
- **Spread and storage.** One standard-normal draw per update from the learner's own stream;
  the re-simulation runs once, in `prepare`, and its shots and loss mask go into the store
  (fix round, fix 6).
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

1. **(Superseded by the fix round, fix 6: the shot labels are stored; the codes stay per
   micro-batch.)** **Labels are built per micro-batch, not stored per shard.** Spec 2 says everything
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
6. **The view's result** for the outcome targets is the match result the rollout records
   (`Trajectory.result`, fix round, fix 11).
7. **`next_done` for the GAE is `done[L - 1]`**: the fixtures pinned that the last step's cut
   is `~terminated[L - 1] & ~next_done`, and in the rollout the bootstrap observation's
   reset is exactly `done[L - 1]`.
8. **Density:** the per-cell term minus its saturated value (zero at a perfect forecast; the
   same gradient), summed over cells, over the view-step's valued-ship count, averaged over
   view-steps.
9. **The KL early stop** reads the epoch's mean approximate KL, `mean((r - 1) - ln r)` over
   living own ships, once per epoch, and stops the update after that epoch (fix round,
   fix 2). The reported clip fraction and KL are the first epoch's means.
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

RTX 4070 Laptop (8 GB), bench lock held, a fresh process per cell. The ship-card agent
shared the machine (mostly CPU load), so update times vary by about 15% between runs.
Commit d4631a83 unless noted: the last performance change
(`perf: the pinned staging gathers the next micro-batch on a worker thread`).

### Bytes per view-tick in the store (spec 2, compaction)

From the store's leaves (`/tmp/bnb-scratch-learner/bytes.py`):

| size | per view-tick | replay (before compaction) | privileged rows (all rows) | without both compactions |
|---|---:|---:|---:|---:|
| 5v5, 10 fields | 3799 B | 1013 B (1578 B) | 480 B (960 B) | 4844 B |
| 26 ships, 54 fields | 10039 B | 2621 B (4090 B) | 1248 B (2496 B) | 12756 B |
| 40 ships, 8 fields | 13809 B | 4028 B (6288 B) | 1920 B (3840 B) | 17989 B |
| 114 ships, 22 fields | 38879 B | 11465 B (17906 B) | 5472 B (10944 B) | 50792 B |

The rest at 5v5: the observation 1384 B, the shot capture 530 B, every row's action 120 B,
rewards and values 200 B, the small flags 16 B. Per view-segment the initial carry is
49 kB. A 5v5 segment of 1536 views is 747 MB of pinned memory.

### Streaming cost (spec 2's budget)

`deathmatch`, 512 environments (768 learner views), float32; rollout: min of 5, update:
min of 2, after a compile update and three rollouts (`stream_cost.py`):

| store | rollout | update | peak VRAM |
|---|---:|---:|---:|
| pinned host memory | 1691 ms | 30.68 s | 3.28 GB |
| device | 1689 ms | 30.25 s | 3.65 GB |

The rollout costs 0.1% more and the update 1.4%: 1.3% of rollout and update together,
inside the 5% budget. The device holds two blocks of records instead of a segment.

How the budget was reached (each change committed):
- the staged micro-batch's view indices were moved to the host with `.cpu()`, a device
  sync per micro-batch, so the host gather waited for the compute: +30% on the update;
- with indices kept on the host and copied from pinned memory, still +29% under the other
  agent's CPU load: the host gather (25 ms per 16-view micro-batch in the trainer, 1-4 ms
  alone) ran on the main thread between kernel launches;
- with the gather on a worker thread, a 48-view minibatch's gradient takes 341 ms from
  pinned memory and 340 ms from the device.

### Precision matrix (spec 8)

Yemong, `deathmatch`, 256 environments (384 views), warmup off so the first epoch is a real
update at 3e-4 (`precision.py`). Two updates after a compile update and three rollouts;
drift is before the update's first step, against the rollout's log-probabilities; clip
fraction and KL are the first epoch's means. Values: the first update, then the second.

| precision | old log-probabilities | rollout | update | ratio drift max | ratio drift mean | clip fraction | KL |
|---|---|---:|---:|---:|---:|---:|---:|
| float32 | rollout | 868 ms | 16.3 / 18.9 s | 7.2e-7 / 6.2e-6 | 6.3e-8 / 2.3e-7 | 0.031 / 0.032 | 0.0046 / 0.0041 |
| float32 | recomputed | 871 ms | 18.2 / 17.9 s | 7.2e-7 / 6.4e-6 | 6.4e-8 / 2.3e-7 | 0.031 / 0.040 | 0.0046 / 0.0044 |
| TF32 | rollout | 631 ms | 19.4 / 16.6 s | 3.7e-5 / 1.7e-4 | 5.2e-6 / 2.4e-5 | 0.033 / 0.037 | 0.0046 / 0.0042 |
| TF32 | recomputed | 635 ms | 17.8 / 18.8 s | 3.4e-5 / 1.6e-4 | 5.3e-6 / 2.5e-5 | 0.031 / 0.043 | 0.0046 / 0.0044 |
| bf16 | rollout | 479 ms | 13.8 / 15.5 s | 1.1e-3 / 3.8e-3 | 1.0e-4 / 3.7e-4 | 0.027 / 0.037 | 0.0043 / 0.0041 |
| bf16 | recomputed | 479 ms | 16.1 / 14.9 s | 1.1e-3 / 3.9e-3 | 1.0e-4 / 3.5e-4 | 0.028 / 0.039 | 0.0049 / 0.0042 |

- The rollout gains 1.4x with TF32 and 1.8x with bf16 matmuls.
- The update's time is within run-to-run noise across the three precisions here (the
  update at this size is dominated by the re-simulation scan's launches and the
  micro-batch count, not the matmuls).
- Drift: float32's ratios agree to float32 rounding; TF32's to 2e-4; bf16's to 4e-3. With
  the clip at 0.2 none of them moves the clip fraction or KL measurably: a real update's
  ratios move by far more.
- "Recomputed" ratios start at exactly 1, but the first-epoch clip fraction and KL are the
  same within noise, because the drift is small next to the update's own step.

TOY_MATRIX

The default stays float32 (spec 8); the author chooses from the matrix.

TOY_BUDGET

THROUGHPUT


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

| part | lines |
|---|---:|
| `bnb.learn` new code (ppo, returns, losses, labels, shots) | 831 (the skeleton PPO's 92 removed) |
| `bnb.runtime` (store new; rollout and train rewritten) | 756 |
| `bnb.models.relational` and its controller | 184 |
| `src/bnb` in total, against the branch point 8a1ad198 | +1655 / -232 |
| of which outside the three packages (game exposure, Yemong replay, contracts, CLI) | +186 / -51 |
| tests (sections, loop, the adapter) | +716 / -29 |

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

## Fix round (2026-11-01)

The review's fixes, as the coordinator decided them. Every test below was seen to fail
with the fix reverted or the bug put back by hand, unless it says otherwise.

1. **The rollout's shot capture.** New tests on a real Yemong rollout (self-play and the
   chaser, so team-1 views exist):
   `test_learner_shots.py::test_the_rollouts_capture_is_the_tick_start_state_in_view_order`
   (the capture's x, y, heading, speed and `alive` at t are the stored privileged rows at
   t) and `::test_the_rollouts_displacement_is_the_next_row_minus_this_one` (to 1e-3 m).
   A capture taken after the move, and one in sim order, each fail them.
2. **No host sync per minibatch.** The update accumulates on the device and reads once at
   its end (one `tolist`), plus the epoch's mean KL once per epoch but the last; the drift
   and the explained variance are computed without boolean indexing. The store waits on
   its last write's event (`SegmentStore._written`): the compute stream on the GPU for a
   device store, the host only before it reads pinned memory itself. Indexing the levels
   with a Python list copied an index to the device per micro-batch (the sync-debug test
   found it); the index is built once. Guard:
   `test_learner_streaming.py::test_an_update_synchronises_only_to_read_its_kl_and_its_results`
   (`set_sync_debug_mode("warn")`, exactly `epochs` syncs; a copy-stream synchronise in
   `batches` fails it). **Change:** the early stop is per epoch now, not per minibatch.
3. **The update's draws.** `random_stream(seed, name)` gives the learner a generator per
   stream (`shot_spread` on the device, `minibatch_order` on the host), seeded from
   `LearnerConfig.seed` and the name. Guard:
   `test_learner.py::test_the_update_is_the_same_whatever_the_global_seed` (fails with
   `randperm` on the global generator).
4. **Tail shots** (coordinator's decision). A shot launched within the longest flight
   (`Rules.bullet_moves` ticks) of the segment's end is invalid whatever it hit, as v1's
   last rows carried no shot labels. Guard:
   `test_learner_shots.py::test_a_shot_launched_near_the_segments_end_is_invalid_whatever_it_hit`
   (no valid tail shot; every body shot of a 60-tick segment has the label it has in the
   100-tick recording, so the body's hit rate is unchanged). **Deviation from the spec's
   literal reading** ("invalid if its flight crosses ... the segment's end"): a hit inside
   the segment no longer makes a tail shot valid. Episode ends keep the flight-end rule (the
   game clears the bullets there); the coordinator may want the same rule there.
5. **Closest pass and side.** `test_learner_shots.py::test_a_miss_records_its_closest_pass_and_that_ships_side`:
   straight shots past standing ships, the nearest 30 m off the line, from a shooter in
   either block, ally or enemy nearest. Swapping `SIDE_ALLY` and `SIDE_ENEMY`, or dropping
   the `sqrt`, fails all four cases.
6. **Labels once per update.** `prepare`'s pass re-simulates every view once and writes
   the shots and their loss mask into the store (`SegmentStore.write_views`, per-view
   leaves `(V, L, ...)`, so a micro-batch's contiguous views are one copy);
   `Learner.staged` reads them back with each micro-batch. The codes (next state, enemy
   action, density, the shot codes) stay per micro-batch, because memory forbids storing
   them: expanded they are about 97 kB per 5v5 view-tick (next state 19.6 kB, shot
   trajectory 73.9 kB, density 2.6 kB), 25 times the 3.8 kB segment, while the stored shot
   labels are about 2.7 kB per 5v5 view-tick (snapshots 2.4 kB). Guard:
   `test_learner.py::test_the_stored_shot_labels_are_the_updates_resimulation`. The
   re-simulation was about a quarter of the update (5 passes of 41 ms per 16-view
   micro-batch, provisional, from the earlier notes); the coordinator measures the new time.
7. **Streaming.** `test_the_update_through_pinned_memory_equals_the_device_path` now runs
   two rollouts from the same seed, one into pinned memory and one into a device store
   (`host_store=False`), with one and two shards: the stored records, the counts, the
   terms, the value losses and the gradients are equal bit for bit.
   `docs/explanation/learner.md` says exactly that.
8. **Untested paths:** `test_recomputed_old_log_probabilities_are_the_unrolls`,
   `test_the_drift_is_the_step_unroll_ratio_error` (against boolean indexing, and a
   perturbed model drifts), `test_the_explained_variance_matches_an_independent_computation`,
   `test_the_kl_early_stop_ends_the_update_after_an_epoch` (all in `test_learner.py`), and
   `test_learner_precision.py::test_a_precision_setting_agrees_with_float32` (GPU): TF32 and
   bf16 give finite terms and gradients, within 1e-2 and 5e-2 of float32's (relative; the
   bounds come from the mantissas, and measured on the case the gradient's error is 8e-4
   and 1e-2 of its norm).
9. **Compaction: not done; a request.** Storing beliefs for hidden rows only cannot be
   exact with static shapes: the count of hidden enemy rows varies from 0 to N - S per
   view-tick, so a fixed layout that holds every case is the enemy block, which is what
   the store keeps; packing them needs a data-dependent count (a host sync per block, or a
   capacity that can overflow). The 16 columns the observation zeroes for a hidden row could
   hold its 12 means, but not its 15 spreads, and multiplexing the observation would make
   every reader of it unmask it. Static options for the coordinator:
   - store the belief only for fogged views (a fog-off view never hides a row), with an
     index map in the store; it saves the pools' fog-off share of the replay;
   - a capacity per view-tick with an overflow count, exact only below it.
   The existing exactness test (`test_the_compacted_replay_rebuilds_the_own_rows_sight_gave`)
   is unchanged.
10. **Dead code.** `returns.outcome_values` is gone (no caller; `critic.outcome_values`
    remains). The skeleton models were left to the cleanup pass.
11. **The outcome class.** The rollout records each view's match result
    (`Trajectory.result`, `(L, R)` int32: `Events.result`, negated for a team-1 view); the
    outcome targets, `Trainer.pool_metrics` and `win_rate` read it. Guards:
    `test_learner.py::test_the_view_records_its_teams_match_result` and
    `::test_the_outcome_targets_read_the_match_result_at_any_outcome_weight`, on a rollout
    whose score is set to 1-0 (short random games end drawn) with the outcome weighted
    zero.
12. **Small items:** the training guide's option table (`--model`, `--precision`,
    `--shards`, `--old-log-probabilities`, `--micro-batch-rows`); the relational model's
    scales are named constants with what each normalises; `buf`, `e`, `R` and `dev` in
    `rollout.py` became `records`, `environments`, `views` and `device` (the style guide's
    `buf` example is now marked plain, since the name left the code).
13. **History.** The WIP commit became
    `refactor: size learner micro-batches in ship rows, not views` (the CLI option is
    `--micro-batch-rows`); each fix above is its own commit.

Outside the section's packages this round: `bnb.contracts.trajectory` (`result`),
`docs/guides/training.md`, `docs/explanation/learner.md`, `docs/contributing/code-style.md`
(one `{.plain}`), and the new `tests/sections/test_learner_precision.py`. Request:
`contracts.md`'s current schema gains `Trajectory.result`.

