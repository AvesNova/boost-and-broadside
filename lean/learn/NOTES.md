# lean/learn and lean/train.py

The learner: rollouts, a BC stage, RL against a league, weight averages, checkpoints. Decisions:
`docs/rewrite/core-decisions.md`; history: `docs/rewrite/core-worklog.md` (and this file's git
log: the long notes before step L1). Evaluation is its own package (`lean/eval/NOTES.md`).

## Entry points

    python -m lean.train                                   # BC (96 updates), then RL (1000)
    python -m lean.train --stage bc --out DIR              # BC only: DIR/bc.pt
    python -m lean.train --stage rl --start DIR/bc.pt --out DIR   # RL from the hand-off: DIR/rl.pt
    python -m lean.train --resume DIR [--updates N]        # DIR's latest checkpoint (or a file)
    python -m lean.train --smoke --device cpu [--stage bc|rl]     # tiny, eager

## Modules

- `config.py`: `LearnConfig` (every number, with its reason), `learning_rate`.
- `rollout.py`: `start`, `collect`, `abandon`, `Opponent`, `CollectCarry`, `Rollout`; seats, the
  league side (`slot_actions`, capacity routing), hit labels, the teacher's record. A tick is
  `act` (returns a `Decision`: views, the live policy's sample through
  lean.model.agent.agent_step, value, the scripted agent's per-ship parameters; the live
  policy's seats in `policy_step_chunks` chunks, `chunked_step`, each chunk a nested compile
  region traced once), `scripted_moves` (the scripted agent where it is used, below),
  `league_actions` (a `LeagueMove`) and `advance` (the game half is lean.game.tick). `advance` returns
  each ship's bullet outcomes by age; `collect` assembles the segment's hit labels once at its
  end (`assemble_hit_labels`), nothing is written in place in the tick.
- `store.py`: `RolloutStore`, an update's segments, compact, streamed to pinned host memory on a GPU.
- `returns.py`: GAE per level, chaining across segments, advantage scaling and shares, the value
  supports' range monitoring.
- `training.py`: `update` (both stages), `TeacherTerm`, `make_optimiser`, the losses.
- `geometry.py`: the rows the auxiliary losses train (`observed_rows`), the geometry heads'
  pairs (`pair_query`: K sampled per view and tick from `draw_keys`' uniform keys), targets
  (lean.game's functions) and losses (`geometry_losses`).
- `weights.py`: `WeightAverages` (EMA and uniform), checkpoints.
- `pool.py`: `League` (the pool of members and tallies, level 1 `allocate`, level 2 `draw` /
  `slot_draws`, draining slots, the persona parameter table), `SnapshotSchedule`, `retained`.
- `rewards.py`: v2's five reward levels and the damage ledger.
- `compiled.py`: `graphs(shapes)`: act, advance, scripted_actions (one graph per environment
  count: one at self_play_share 0.5), slot_actions, merge_league, minibatch_values,
  minibatch_loss. `LearnConfig.compile_shapes`: "static" (the default) compiles
  each for the run's sizes (environments, slots, fields, micro-batch, the league table: all
  fixed in a run) and any second compile raises; "dynamic" compiles once for every size (duck
  sizing off). The critic's support (`returns.value_support`) is built on the device once per
  process (cached per ranges, bins and device) and passed into the graphs; the next-state support is a policy buffer (a table made
  with torch.tensor inside a graph is a pageable copy at every call).
- `lean/train.py`: the stages, checkpoints, `--resume`.

## The recipe

- **Update:** 2,048 environments x 4 segments x 128 ticks (1,048,576 environment ticks), returns
  chained over the 512 ticks of each seat and bootstrapped only at the end, 4 epochs, 32
  minibatches, micro-batches of 8 views with gradient accumulation (speed only, per GPU), targets once per update.
- **Dead ships** (author 2026-11-28, option C): a fallen ship's value in the returns (its
  post-death ticks and the update's bootstrap) is its living team-mates' mean value at that tick
  and level (`returns.fallen_team_values`, the store's `fallen` record), so the outcome reaches
  its earlier ticks at gamma^k; its death penalty stays. Team-shared payments (boundary deaths,
  friendly-fire compensation, unattributed boundary healing) go to the living ships only.
- **Seeds:** the explicit generator takes `--seed` + 1,000,003, torch's default generators
  (initial weights, the compiled rollout's draws) `--seed`.
- **BC stage** (`update(..., cloning=True)`): the student plays its own games (half self-play,
  half against the scripted agent). Loss: cross-entropy of the 42-way policy against the scripted
  teacher's full distribution (its joint over the 42 actions, flight x fire, `lean.scripted.Choice`) at 1.0, plus value, next-state and hit; no
  surrogate, no entropy. 96 updates (100.7M ticks, v2's 100M-tick stage). The hand-off is the
  stage's EMA, with a fresh optimiser, warm-up, league and averages.
- **Teacher term in RL:** the same cross-entropy at 0.5 x 0.95^k at RL update k; off for good once
  the league's win rate against the scripted agent reaches 0.45 or the weight falls below 0.005.
- **The scripted agent where it is used** (2026-11-28, item 6; audit findings 5 and 6):
  `rollout.scripted_moves` calls it on the league environments every tick (the scripted slots'
  moves; rows playing networks are computed and unused) and, while the teacher is used (BC; RL
  until the term is off for good: `collect(teacher=...)` from lean.train), on the self-play
  environments too. The two calls are one graph (`scripted_actions`, the same sizes at
  self_play_share 0.5), both from the first tick, so the term switching off drops a call and
  compiles nothing. The teacher's record (`Rollout.teacher`, (T, S, M, 22) float16) holds the
  seat's own rows only (`own_slots`: its living ships in slot order, M = team_size_maximum,
  zero past them) as the 21 flight probabilities and the fire probability; the loss gathers
  the policy's own rows and forms the joint in float32 (`teacher_joint`). With the term's
  weight 0 the record is neither written nor copied (a zero record of the same shape goes into
  the same loss graph; scoring it costs nothing measurable) and once the term is off the store
  drops it (`RolloutStore.drop`).
- **Averages:** an EMA (0.95 per update) and a uniform mean from the end of the warm-up; side
  copies never read by training, saved with the live weights.
- **Optimiser:** fused Adam, no weight decay, epsilon 1e-5, global clip 1.0, a non-finite step
  skipped and counted; learning rate 3e-4 after a 2,000-step linear warm-up per stage, then
  constant.
- **PPO:** clip 0.2; no KL stop (`approximate_kl_epoch_<k>` logged, a warning above 0.02);
  advantages summed over levels and divided by their standard deviation, mean kept.
- **Loss coefficients** (fixed, on scale-free terms): value 0.05, next-state 0.1, hit 0.05, entropy
  0.003 on the joint 42-way entropy in RL; the geometry heads (PROVISIONAL, 2026-11-28, for the
  author) ship pairs 0.1, ship-field pairs 0.05, map edge 0.05.
- **Auxiliary rows** (author 2026-11-28): next-state, hit and the geometry heads train on the
  rows the seat sees (`observed`: its living ships and the enemies visible this tick; hidden
  enemies and dead ships masked). A next-state row also needs the ship alive and visible to
  the seat after the step (its target is what the team sees on both ticks) and the episode
  not ended; a hit row a label (a plain game label, an enemy's bullet too).
- **Geometry heads** (lean/model/NOTES.md "Geometry heads"): per view and tick K = 8 valid ship
  pairs of the 28 (4v4) and K = 8 valid ship-field pairs of the 80 (`pair_samples`,
  `field_pair_samples`), drawn without replacement before the pair layers; the uniform keys
  are drawn eagerly per micro-batch (`draw_keys`, the update's generator when it lives on the
  device) and passed into the compiled loss, so its shapes are fixed. A sampled pair counts c /
  min(K, c) (c valid pairs on its tick) and the sum is divided by the minibatch's valid pairs:
  an unbiased estimate of the mean over every valid pair, exact when K >= c. Losses in nats per
  column: the ship pairs' symmetric columns on the upper triangle, the train angle and gun
  error averaged over both orders. Off for an ablation: `ModelConfig(geometry_heads=False)`.
- **Rewards:** v2's five levels; value supports from measured returns (2026-11-25): damage
  +-5.4, kill_death +-7.5, outcome +-10, objective levels +-1 (placeholders), each 57 bins
  across the range plus 3 buffer columns of the same width beyond both ends (63 bins;
  lean/model/NOTES.md). Damage and kill_death are symlog (`value_knee`, the event rule: one
  event x e^-3: a hit of 8 points 0.0109 (0.0286 until 2026-11-28, from 21-point hits) and a
  kill 0.141; innermost bins 0.0027 and 0.022, outermost 2.09 and 1.56, outermost centres
  +-10.5 and +-11.6); outcome (bimodal) and the objectives linear (bins
  0.357 and 0.0357). The range monitoring counts bins in bin space (`usage_counts` takes the
  knees); its thresholds (`range_narrow_bins` 8, `range_bin_share`) are unchanged.
- **Store:** positions, the geometry record and PPO numbers float32, other floats float16,
  actions int8. The geometry record (`lean.game.geometry_record`: world velocity, launch
  velocity, store, map edge, bullet drag and life; 8 floats per ship and tick) adds 32 bytes
  per ship-tick; `observed` is resident (counted on the device).

## The league

Two levels (`pool.py`; a proposal in docs/rewrite/core-decisions.md "The two-level league"
until the author decides; numbers in docs/rewrite/sections/league-pool/progress.md).

- Environments 0 to E/2 - 1 are self-play (two live seats); the rest are league games (one live
  seat); the live policy's games continue across updates.
- The pool (host memory), fixed member indices: the yardstick scripted agent (0), the EMA (1),
  the uniform average (2), up to 12 snapshots, then the scripted personas
  (`LearnConfig.league_personas`, four placeholders from the persona study's round 1). All are
  sampled alike. The BC stage's league holds the yardstick only; RL opens it
  (`League.open`): the personas and the EMA join before the first deal, the uniform average
  once it holds an update, snapshots by the schedule. Tallies: AlphaStar's decaying games,
  wins, draws (0.99 a game, unplayed 0.5); targets PFSP (1 - p)^2 with the 1/8 uniform floor.
- Level 1 (`allocate`, after every update, host side): K = 12 loaded slots, R = 4 spare. Each
  member's target is q K slots; the rounding is carried as a debt from update to update (so the
  long-run share is exact); one swap per free slot at most (unload the most served member's
  slot with the fewest games, load the most short member), only while some member is short by
  more than one slot against another. Duplicates allowed. A load copies weights into the
  slot's own device buffers (graph inputs: no compile) or sets a scripted slot's parameter row.
  An unloaded slot drains (takes no game, plays its running ones to the end, frees when empty);
  a member dropped by retention unloads every slot (its games RETIRED, untallied). No game is
  ever re-dealt.
- Level 2 (`draw`, at a game's end, in the tick): uniform over the loaded slots with a free row
  (C = ceil(1.25 L / K) = 107 at L 1,024): accepted within the free rows, one
  redraw, then in slot order; past every free row the overflow slot (a capacity-free yardstick
  slot; `overflow_games` counts, never seen in the simulations).
- Compute: each network slot holding weights (loaded or draining) runs once on W = C rows
  (`slot_actions`, one graph; empty slots skipped); scripted slots (yardstick, personas) take the
  scripted call over the league environments with per-row parameters (`rollout.scripted_rows`:
  the league side of a persona's game reads its row of `persona_table`, every other ship the
  yardstick's), in both stages, so BC and RL share one scripted graph; `merge_league` takes the
  network slots' rows from K + R entries. The EMA plays from RL's first tick, so the routing
  compiles there and never again (`test_the_league_tick_is_static_through_loads_unloads_and_duplicates`).
- Snapshots: v2's schedule (5, 10, ... 160M RL steps, then every 30M) and retention (newest 8,
  plus the nearest to each age 2^k back, at most 12).
- Checkpoints: every 10M environment steps, `DIR/checkpoints/<stage>_<steps>.pt`, none deleted:
  live weights, averages, optimiser, league tallies, teacher term, counters, the games in progress
  and the generators' states. Each snapshot's weights are written once to `DIR/snapshots/`.
  `--resume` continues with the checkpoint's settings and plan, giving the same updates as a run
  that never stopped (to rounding; `test_train.py`).

## Numbers (laptop RTX 4070, performance profile, 2,048 x 4 x 128, compiled, bf16, core game)

- Teacher 6 (2026-11-28, `docs/rewrite/sections/teacher-6-progress.md`, the real trainer, 3
  updates): the store 3,345.5 -> 2,601.5 MiB with the teacher (its record 1,008.1 -> 264.0
  MiB), 2,337.4 MiB once the term is off; RL collect segment with the teacher off 3.07 ->
  2.71 s (-2.8 ms a tick, -1.4 s an update); BC speed unchanged; the BC losses equal to the
  shown digits; the teacher's BC cross-entropy from the compact record equals the joint
  record's to 1.2e-7 relative.

- Speed 5 (2026-11-28, `docs/rewrite/sections/speed-5-progress.md`; core 2b7bc58e against
  the branch, quiet machine): micro-batches 64 -> 8 views, a whole BC update on the real
  trainer 124.8 -> 107.4 s (16: 111.2), peak 5.1 -> 1.3 GiB, the BC loop 7,605 -> 8,711
  steps/s; the live policy in 6 seat chunks, BC tick 25.34 -> 20.87 ms, RL tick (EMA, 2 slot
  calls) 27.70 -> 23.22 ms, cold first segment +2 s. The real trainer, 3 updates: BC 7,605 -> 8,933 / 8,747
  steps/s, RL (fresh weights) 7,545 -> 8,678 / 8,786, peak 5.2 -> 1.2 GiB. Both are per-GPU settings
  (`micro_batch_views`, `policy_step_chunks`): re-measure on the 5090 (larger L2, the host's
  ~6 ms per micro-batch binding sooner).
- Loop speed (2026-11-27, before the two-level league): BC 8,255 steps/s, 127.0 s per update
  (collect plus update); the update alone 114.5 s (149.1 ms per micro-batch), peak allocated
  4.74 GiB.
- Geometry heads (2026-11-28 ~04:56, quiet machine, `docs/rewrite/sections/heads-8c/cost.py`):
  the compiled update per micro-batch of 64 views x 128 ticks, 4v4: off 150.2 ms, K = 4 162.1
  (+8%), K = 8 165.4 (+10%), every pair (28 + 80) 193.1 (+29%); peak allocated 4.24 / 4.57 /
  4.59 / 5.38 GiB; most of K = 8's cost is fixed (per-ship projections, targets, edge column).
- League side, two levels (2026-11-27, back to back with the one-level league): the RL tick
  27.9 ms with no snapshot (EMA, 2 x 107 rows; one-level 24.1, no network), 28.6 ms with 1
  (26.4), 33.1 ms with 12 (32.5); the BC tick 25.5 ms (24.2: the per-row scripted parameters,
  1.3 ms). Share tracking and the smoke run: docs/rewrite/sections/league-pool/.
- Compile (baseline reset 2026-11-28 ~09:48 on `teacher-6`: 6 graphs, act 555 traced
  operations and scripted_actions 5,258 (the scripted call out of act, traced once for both
  of its calls), advance 3,306, minibatch_loss 1,645, slot_actions 720, merge_league 27; cold
  164.7 s, still over the 150 s budget, budget kept). The bisect (teacher-6-progress.md):
  the scripted-delay merge (8b) adds about 9.5 s of act's backend, the geometry heads (8c)
  about 13.6 s of the loss's; either alone measured 139-152 s, the spread between runs about
  10 s. Before: (baseline reset 2026-11-28 ~07:50 on `speed-5`, `compile_baseline.json`): 5 graphs
  (act 5,788 traced operations, advance 3,307, minibatch_loss 1,627, slot_actions 720,
  merge_league 28), cold 164.3 s, **over the 150 s budget** (budget kept; for the author). Core
  before speed 5 (2b7bc58e) already failed the old baseline (05:05, act 4,613, cold 141.6 s):
  act 6,325 operations, cold 160.3 s (measured 07:34; the scripted-delay merge reached lean/
  after that baseline, the likely cause, not bisected); the seat chunks then cut act's
  operations (the step traced once) and add about 3-4 s of cold compile. The
  compile tripwire (`compile_check.py`) fails on any graph break, new graph, recompile, or a
  function's traced operations over 10% above its baseline (`docs/rewrite/scoreboard.md`
  "Compile time"). Run it at the end of every step:

      PYTHONPATH=. python docs/rewrite/sections/scoreboard/compile_check.py

- The inductor and triton caches default to ~/.cache/bnb-inductor (lean.contract, when
  TORCHINDUCTOR_CACHE_DIR is unset): /tmp is wiped at reboot, and a warm cache skips most of the
  cold compile.

## Tests

All run the real game, scripted agent and model at tiny sizes (`tests/setting.py`).
`test_recipe.py` (the teacher's compact record, its BC loss equal to the whole joint record's
within float16 rounding, the scripted calls without the teacher and the weight-0 update, soft cross-entropy, BC ignores the advantages, the teacher
term's decay and gate, averages, checkpoints, optimiser, warm-up, KL monitor, advantage shares),
`test_train.py` (BC then RL, checkpoints, an exact resume with the league's slots and debts),
`test_league.py` (level 2's slot draws, PFSP, level 1's loads and duplicates, drain, share
tracking over a simulated run, personas per row, routing, state round trip, schedule and
retention), `test_learn.py` (also the chunked live-policy step equals one call, the critic's
support built once), `test_hit_head.py`, `test_geometry.py` (sampling unbiased and exact at K >=
c, distinct valid pairs, the upper triangle, the trained rows are the visible ships in a fogged
rollout, the pair loss with every pair equals a direct sum of the game's targets),
`test_rewards.py`, `test_store.py`; slow, `test_compiled.py` (compiled collect and update, loss
equal to eager, the league tick fullgraph, and static for the whole run: one compile at RL's
start through loads, unloads, duplicates and a retention drop).

## Open

- The geometry heads' coefficients (0.1 / 0.05 / 0.05) and K = 8 are provisional (for the
  author): the coefficients by analogy (ship pairs like the next state, the single-column
  field and edge targets like hit), K from the cost table in
  docs/rewrite/sections/heads-8c-progress.md. Whether these heads help is a training question
  (an arm with `geometry_heads=False`).
- The league's personas are placeholders (round 1's strongest four) until the persona study
  picks them; the cheese agents (step C) would join the same way, as scripted members with
  their own parameter rows.
- Early in RL the PFSP targets themselves are mostly scripted (yardstick + 4 personas against
  the EMA and few snapshots, the learner losing to the scripted ones): whether the personas
  should share one member's share is the author's call.
- The teacher term's numbers (0.5, 0.95, cut-off 0.005) and the BC length are defaults with
  reasons, not measured; v2's BC ran 8x the Adam steps per update.
- The KL warning fired on every update after the first in the first BC check (3 updates, GPU,
  2026-11-23); if it fires routinely in a longer run, make 0.02 a stop checked after each epoch
  (for the author).
- Steady loop speed with league networks playing (the two-level league) is not measured.
- Entropy stays on during the teacher phase (VPT dropped it there). The uniform average in BC
  includes the stage's early weights; the hand-off uses the EMA.
- Checkpoints repeat the games in progress and Adam's moments (43 MB at smoke size).
