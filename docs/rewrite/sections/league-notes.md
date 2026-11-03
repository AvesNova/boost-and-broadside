# League notes (section 8)

Branch `section/league`. Spec: `league-spec.md`. Brief: `league-brief.md`.

## Progress

- [x] Part 1: the ratings (`bnb.league.ratings`, `bnb.league.gauge`); the six equivalence
  tests pass through `league_arm.py`.
- [x] Part 2: the pool's entries, sampling weights, retention, averages (`bnb.league.pool`).
- [x] Part 3: the league's seat (`bnb.runtime.opponents`), the League hooks
  (`bnb.league.league`), the tally in the rollout, the new default lineup.
- [x] Part 4: resume (opponent slots + league state in the training checkpoint).
- [x] Part 5: metrics (`League.metrics`), with a test.
- [x] Part 6: `bnb ratings`, `bnb head-to-head` (`bnb.league.matches`,
  `bnb.league.tournament`, `bnb.cli`).
- [x] Docs: `docs/explanation/league.md` (replaces the in-progress page), reference page,
  pages the lineup changed.
- See "Status at hand-back" at the end for the full suite.

## Spec items and what was done

### 1. The pool
- Entries: snapshots (`snapshot_<steps>`), `uniform_average`, `exponential_average`
  (decay 0.95 per update, `LeagueConfig.exponential_decay`), the ladder (`random`,
  `rung_0.1` … `rung_0.9`, `rung_0.95`, `scripted`; `LeagueConfig.ladder`).
- Self-play is a fixed share: `bnb.runtime.train.LINEUP` is self-play 0.5 and the league's
  seat (`opponents`) 0.5; `mixed()` crosses it with the training mix, so every mode has
  both blocks. The old `league_0` lagged copy, `league_every` and the chaser/random pools
  of the default lineup are gone (no legacy). Chaser and Random remain usable controllers.
- The uniform average starts at the attach-time weights (count 1) and folds in the weights
  after every update.

### 2. Slots and the compiled graph
- `OpponentSlots`: parameters stacked `(slots + 1, ...)`, slot k plays rows
  k, k + slots, … (every slot plays every mode); the extra set plays the draining slot's
  rows; per-row selection by `where`; each row keeps its own network carry.
- **Deviation: not `torch.vmap`.** vmap over `functional_call` works with Yemong's Triton
  kernels on, but on the plain path (CPU, or kernels off) scaled-dot-product attention under
  vmap skips autocast's casts and fails on bf16 (query float32, value bf16). The sets run
  one after another inside the same compiled, captured graph instead. Probe on the dev GPU,
  kernels on, 5 sets × 128 rows of 5v5 Yemong: vmap 3.21 ms/step, the loop 3.35 ms
  (one 640-row forward 2.49 ms); load ~3, provisional. **Request:** a cast in
  `SpatialBlock._attend` (models, outside this section's boundary) would let vmap run on
  both paths, if wanted.
- The ladder: one `Scripted` pass over every opponent row with a per-row probability tensor
  written in place (`ladder_probability`), as scripted spec A6 intends.
- Every network forward and the scripted pass run on all opponent rows every tick (static
  shapes): a slot holding the ladder still pays its network forward, and the extra set's
  forward runs even when nothing drains (5/4 of the network cost).
- In-graph draws: `counter_uniform` (lowbias32 hash of the stream key, the tick counter and
  the row; int64 arithmetic kept below 2^63), inverse-CDF `draw`. The counter lives in the
  seat's carry, so a resume continues it. The host's slot draws use the same functions.
- The drain: after each shard, if the draining slot's rows have all switched
  (`OpponentSlots.drained`, one host read), the extra set is copied into the slot; then, if
  none drains, the next slot (round robin) gets a drawn entry. **Reading:** "one slot per
  rollout shard" and "at most one slot drains at a time" conflict when episodes outlast a
  shard (they do: a drain lasts until every row's episode ends); the implementation starts
  a drain only when none is running.
- A ladder row draws its rung at each of its resets, from its own mode's ladder weights.
  When a slot's drain brings in a ladder entry, its rows draw a rung at their switch; the
  drawn ladder entry only decides the kind. The slot's initial fill draws rungs for rows
  that play the ladder before their first reset (`draw_ladder`).
- **Reading (request):** a network slot plays every mode, so its entry is drawn from the
  modes' weights mixed by the modes' shares of the opponent rows (`League.mixed`); ladder
  rows use their own mode's weights.
- An average loaded into a slot is a copy at load time; it is not refreshed while loaded
  (so it never changes mid-episode).

### 3. Sampling
- `sampling_weights`: `p(1 − p)` normalised to `1 − floor`, plus `floor / K` each (all
  floor if every `p(1 − p)` is 0). Per mode from that mode's gauge.

### 4. The live gauge
- `bnb.ratings`: `fit_ratings`, `rate_against`, `rate_live`, `fit_ladder`, `allocate`;
  float64 Newton with backtracking to a decrement < 1e-13.
  - `allocate`'s value is the derivative of the summed variance, `p(1 − p) v'C²v` (no
    Sherman–Morrison denominator), and the remainder goes to the largest shares: only this
    reproduces the golden (both variants compared).
- `Gauge.update`: ladder fit → live (`rate_live`, this update only) → averages
  (`rate_against`, window 200, per-game live rating; no games: the live rating).
- **Reading:** training plays only live-vs-entry games, so the record among stationary
  players comes from snapshots: a snapshot is the weights that played its update (taken in
  the `collected` hook, before learning), and that update's games against stationary
  entries are credited to it. It starts at that update's live rating.
- Sweeps: `reported` is ±inf, `rating` holds the last finite value (used by sampling,
  fits, and the `live_elo` metric). The initial live rating is 0.
- A snapshot with no stationary games stays out of the ladder fit.
- Games against an entry retention has dropped (still loaded in a slot) are not counted.

### 5. Snapshots and retention
- Every `LeagueConfig.snapshot_every` environment steps; `None` → measured once at the
  second update (30 min), recorded in `run.json` and the league state.
- Retention (`bnb.league.pool.retained`): newest 8 + the snapshot nearest each age 2^k,
  k ≥ 3 (equal distance keeps the older); the first snapshot always survives as the
  deepest history. `bnb.runs.records.retained` (checkpoints) is a different rule; its
  docstring's remark that league snapshots follow it is now inaccurate (**request**: drop
  the remark, a runs-section file).
- Disk: `runs/<run>/league/step_<steps>/{checkpoint,uniform_average,exponential_average}`
  policy files, so the offline tournament can rate frozen averages. Retention deletes the
  step directory. A resume deletes step directories beyond its step (a rewound timeline).
- **Deviation:** the training checkpoint embeds the snapshots' and averages' weights
  (about 10 MB per snapshot for Yemong, ~16 snapshots), rather than referring to the
  `league/` files: retention after a checkpoint could delete a file the checkpoint needs.
  **Request:** de-duplicate by reference if checkpoint size matters.

### 6. Published ratings
- `bnb.league.matches.play`: one rollout per batch and preset, a pool per pair, a
  `Recorded` controller per player counting its rows' first results (`tally`), seats
  alternating per environment within each pair (exactly half each). Eager (not compiled):
  shapes change per batch.
- `bnb.league.tournament.rate`: first batch even (`first_games` per pair), later batches by
  `allocate` (≥ 1 per pair), each pair rounded up to an even count (so a batch can exceed
  its size by one game per pair); `fit_ratings` with prior 1, anchor scripted, +1000; stop
  on max stderr < target (25) or the budget. Rows keyed by game version (`bnb.game.VERSION`),
  mode, map (preset name), fleet size; written via `provenance.write_result`.
- `bnb head-to-head A B [C D ...] --games N --presets ...`: per pair W/L/D, score ± SE
  (from the per-game score variance), per-preset split; artifact under the first named
  run, else the runs directory.
- `bnb ratings RUN [--with ...]`: field = the run's league policies + the ladder + `--with`.
- `bnb train` attaches the league to every PPO run whose lineup seats it, and restores it
  on a resume (`bnb.runs.checkpoint.league_state`). A fork and a warm start reset the slots
  to the model's weights and start a fresh league.

### 7. Metrics
- `League.updated` returns `{"league": {...}}`: `live_elo/<mode>` (held value),
  `live_elo_stderr/<mode>`, `rating/<mode>/<entry>`, `sampling_share/<mode>/<entry>`,
  `score/<mode>/<entry>` (the entry's score against the live policy this update, when it
  played), `games/<kind>`, `pool/entries`, `pool/snapshots`, `pool/slots`,
  `pool/draining_slot`, `pool/incoming`.

## Acceptance tests and hand breaks

| Spec 8 item | Test |
|---|---|
| 1 equivalence | `tests/reference/league` (6) via `league_arm.py` |
| 2 recovery | `tests/sections/test_league_ratings.py::test_the_live_rating_recovers_a_known_rating`, `::test_the_offline_fit_recovers_known_ratings` (200 trials each, ties with an unbiased expected score) |
| 3 sampling | `tests/sections/test_league_slots.py`: counter draws vs `p(1−p)` + floor; slot draws vs the mixed weights; ladder rows by their own mode's weights; same seed same opponents (and another seed differs) |
| 4 drain | `test_league_slots.py`: no change mid-episode; every row of a drained slot switched at its next reset; at most one slot drains; a drain ends only when all switched (600 ticks, CPU) |
| 5 graph | `tests/sections/test_league_graph.py` (gpu): no recapture over 40 segments with ≥ 3 drains; the league adds the same syncs per doubled segment as self-play |
| 6 resume | `tests/runs/test_resume.py` with the league attached (CPU and GPU, relational and Yemong), plus damage cases `league` (slot weights), `draws` (swap counter), `gauges` (record and windows) that must fail |
| 7 speed | see Measurements |

Also: `test_league_pool.py` (retention, weights, averages), `test_league_metrics.py`,
`test_league_tournament.py` (engine, tournament anchor and budget, head-to-head, CLI parse),
the tally orientation test.

Hand breaks (each reverted):
- Gauge: the live rating assigned even when infinite → the sweep test fails; the live
  rating from an accumulated record → the sweep and window tests fail.
- Sampling: `draw` without normalising by the total → the ladder-rows test fails; ladder
  rows drawing from mode 0's weights → the same test fails. (The counter-draw test uses
  weights that already sum to 1, so it does not see the first break.)
- Drain: rows switching without waiting for their reset → `test_no_opponent_changes_mid_episode`
  fails; `drained` always true → that test and `test_a_drain_ends_once_every_row_has_switched`
  fail.

## Measurements

Spec 8.7, `/tmp/bnb-scratch-league/speed.py`: the 5v5 training mix (`mixed()`), 1024
environments, 128-tick segments, Yemong, bf16, kernels on, compiled and captured; RTX 4070
Laptop GPU, commit after `test(league): compare a segment's syncs...`; best of 4 after one
update and 6 warm-up segments; lock held, alone on the GPU.

| Lineup | Rollout (s / segment) | Update (s) | Peak (GB) | Load average (1, 5, 15 min) |
|---|---|---|---|---|
| self-play only | 1.613 | 33.98 | 2.56 | 2.15, 2.55, 2.56 |
| self-play 0.5 + league 0.5 | 1.859 | 25.45 | 2.48 | 1.39, 1.93, 2.27 |

- The league adds 0.25 s per segment (+15%). It replaces 512 learner views with 512
  opponent rows that run five network forwards (four slots and the extra set, 640 rows of
  network work) and one scripted pass; that is about the extra cost.
- The update is shorter with the league because only the learner's views train: 1024
  views per environment-segment instead of 2048 in self-play.
- Load average ~2 (another agent's idle session), so the numbers are provisional.
- One-off compile of the league rollout at this cell: ~5.5 minutes (dev GPU), most of it
  autotuning the Triton ops for the per-set shapes.

## Deviations, requests and open questions

- No vmap (above); request a model-side cast if vmap is wanted.
- Drain start rule (above); slot entries from mixed mode weights (above).
- Checkpoints embed the pool's weights (above).
- `bnb.runs.records.retained` docstring remark (above).
- **Pre-existing, not this section:** on a machine with CUDA, a non-gpu test that runs an
  Adam step (`trainer.update()`) or compiles initialises CUDA (torch 2.15's Adam
  `_accelerator_graph_capture_health_check`; inductor), and the suite's guard then errors
  at teardown when no earlier test initialised CUDA. Seen on the base commit too
  (`test_bc.py::test_reinforcement_learning_starts_from_the_clone`,
  `test_mixed.py::test_a_compiled_mixed_tick_equals_each_game_compiled_alone` run alone).
  The league's own non-gpu tests avoid `ppo.update` for this reason.
- First compile of the league rollout at the 5v5 mix with Yemong: ~330 s rollout + 180 s
  learner on the dev GPU (one-off; the Triton ops' autotuning per set shape).
- **GPU Yemong resume, first run after new shapes:** once, the straight and first runs of
  `test_a_resumed_run_equals_one_straight_through_on_the_gpu[yemong]` differed at the
  second update; rerun by hand (straight, first, second) they were bitwise equal in metrics
  and weights. The likely cause is the kernels' autotune choice for the opponent sets' new
  shapes being made in the first process and read from the disk cache in the next
  (`bnb.models.yemong.kernels`, author 2026-11-03: the choice is not in checkpoints).
- Open: whether a slot holding the ladder should skip its network forward (needs a
  data-dependent branch or a different row layout).
