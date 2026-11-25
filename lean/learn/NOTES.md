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
  league side (`slot_actions`, capacity routing, `league_rows`), hit labels, the teacher's
  distribution. A tick is `act` (returns a `Decision`: views, the live policy's sample through
  lean.model.agent.agent_step, value, the scripted `ScriptedMove`), `league_actions` (a
  `LeagueMove`) and `advance` (takes both and the `HitLabels`; the game half is lean.game.tick).
- `store.py`: `RolloutStore`, an update's segments, compact, streamed to pinned host memory on a GPU.
- `returns.py`: GAE per level, chaining across segments, advantage scaling and shares, the value
  supports' range monitoring.
- `training.py`: `update` (both stages), `TeacherTerm`, `make_optimiser`, the losses.
- `weights.py`: `WeightAverages` (EMA and uniform), checkpoints.
- `pool.py`: `League` (members, network slots, capacity draws, tallies, retiring slots),
  `SnapshotSchedule`, `retained`.
- `rewards.py`: v2's five reward levels and the damage ledger.
- `compiled.py`: `graphs()`: act, advance, slot_actions, minibatch_values, minibatch_loss, each
  compiled once with dynamic sizes (no recompile at a new team size, league size or batch size).
- `lean/train.py`: the stages, checkpoints, `--resume`.

## The recipe

- **Update:** 2,048 environments x 4 segments x 128 ticks (1,048,576 environment ticks), returns
  chained over the 512 ticks of each seat and bootstrapped only at the end, 4 epochs, 32
  minibatches, micro-batches of 64 views with gradient accumulation, targets once per update.
- **BC stage** (`update(..., cloning=True)`): the student plays its own games (half self-play,
  half against the scripted agent). Loss: cross-entropy of the 42-way policy against the scripted
  teacher's full distribution (`Marginals` product) at 1.0, plus value, next-state and hit; no
  surrogate, no entropy. 96 updates (100.7M ticks, v2's 100M-tick stage). The hand-off is the
  stage's EMA, with a fresh optimiser, warm-up, league and averages.
- **Teacher term in RL:** the same cross-entropy at 0.5 x 0.95^k at RL update k; off for good once
  the league's win rate against the scripted agent reaches 0.45 or the weight falls below 0.005.
- **Averages:** an EMA (0.95 per update) and a uniform mean from the end of the warm-up; side
  copies never read by training, saved with the live weights.
- **Optimiser:** fused Adam, no weight decay, epsilon 1e-5, global clip 1.0, a non-finite step
  skipped and counted; learning rate 3e-4 after a 2,000-step linear warm-up per stage, then
  constant.
- **PPO:** clip 0.2; no KL stop (`approximate_kl_epoch_<k>` logged, a warning above 0.02);
  advantages summed over levels and divided by their standard deviation, mean kept.
- **Loss coefficients** (fixed, on scale-free terms): value 0.05, next-state 0.1, hit 0.05, entropy
  0.003 on the joint 42-way entropy in RL.
- **Rewards:** v2's five levels; value supports from measured returns (2026-11-25): damage
  +-5.4, kill_death +-7.5, outcome +-10, objective levels +-1 (placeholders), each 57 bins
  across the range plus 3 buffer columns of the same width beyond both ends (63 bins;
  lean/model/NOTES.md). Damage and kill_death are symlog (`value_knee`, the event rule: one
  event x e^-3, 0.0286 and 0.141; innermost bins 0.0056 and 0.021, outermost 1.79 and 1.67,
  outermost centres +-9.5 and +-11.6); outcome (bimodal) and the objectives linear (bins
  0.357 and 0.0357). The range monitoring counts bins in bin space (`usage_counts` takes the
  knees); its thresholds (`range_narrow_bins` 8, `range_bin_share`) are unchanged.
- **Store:** positions and PPO numbers float32, other floats float16, actions int8.

## The league

- Environments 0 to E/2 - 1 are self-play (two live seats); the rest are league games (one live
  seat); the live policy's games continue across updates.
- Members (fixed indices): the scripted agent (0), the EMA (1), the uniform average (2), then up
  to 12 snapshots. Before the first snapshot every league game is against the scripted agent.
- Draws at a game's end: PFSP (1 - p)^2 with a 1/8 uniform floor (decaying tallies, 0.99 a game,
  unplayed 0.5), within row capacities ceil(1.25 L / present).
- Routing: each tick, for each network slot, `slot_actions` takes that slot's environments into a
  batch, runs the slot's network once and reads the rows back; one compiled graph serves every
  slot, K and W. A dropped snapshot plays its running games to their end in a retiring slot (2
  + 12 + 4 slots; with none free, the games of the slot with the fewest are re-dealt).
- Snapshots: v2's schedule (5, 10, ... 160M RL steps, then every 30M) and retention (newest 8,
  plus the nearest to each age 2^k back, at most 12).
- Checkpoints: every 10M environment steps, `DIR/checkpoints/<stage>_<steps>.pt`, none deleted:
  live weights, averages, optimiser, league tallies, teacher term, counters, the games in progress
  and the generators' states. Each snapshot's weights are written once to `DIR/snapshots/`.
  `--resume` continues with the checkpoint's settings and plan, giving the same updates as a run
  that never stopped (to rounding; `test_train.py`).

## Numbers (laptop RTX 4070, performance profile, 2,048 x 4 x 128, compiled, bf16, core game)

- Loop speed (collect plus update): 6,670 steps/s in RL before the first snapshot (step L2,
  2026-11-23); BC 6,668. Peak allocated 5.7 GiB. Steady speed with league networks playing is
  not measured under the per-slot routing (the one short run included its compile: 4,852).
- League side per tick at parity, 8 slots, per-slot form: 12.75 ms (vmap form 10.5 ms, which
  recompiled at every new K and W, so it was dropped); at 256 environments the per-slot form is
  launch-bound (about 1.35 ms a slot). One CUDA graph over every slot's call gave 10,700 to
  18,325 steps/s at 256 environments and nothing at parity; reverted (commit 992bb7ef).
- Compile: cold start about 240 s; the compile tripwire (`compile_check.py`) fails on any graph
  break, new graph, recompile, or a function's traced operations over 10% above its baseline
  (`docs/rewrite/scoreboard.md` "Compile time"). Run it at the end of every step:

      PYTHONPATH=. python docs/rewrite/sections/scoreboard/compile_check.py

- First BC check (3 updates, GPU): teacher cross-entropy 3.65, 2.91, 2.23 nats; the KL monitor
  fired in the warm-up (fourth-epoch means 0.14 to 0.34).

## Tests

All run the real game, scripted agent and model at tiny sizes (`tests/setting.py`).
`test_recipe.py` (teacher joint, soft cross-entropy, BC ignores the advantages, the teacher
term's decay and gate, averages, checkpoints, optimiser, warm-up, KL monitor, advantage shares),
`test_train.py` (BC then RL, checkpoints, an exact resume), `test_league.py` (capacities, PFSP,
routing, retiring slots, schedule and retention), `test_learn.py`, `test_hit_head.py`,
`test_rewards.py`, `test_store.py`; slow, `test_compiled.py` (compiled collect and update, loss
equal to eager, the routing graph fullgraph and reused at a new K and W).

- The inductor and triton caches default to ~/.cache/bnb-inductor (lean.contract.compile_dynamic, when TORCHINDUCTOR_CACHE_DIR is unset): /tmp is wiped at reboot and the warm cache saves about 140 s.

## Open

- Step C: the cheese agents as league members of kind "scripted" after the snapshots (their rows
  need their own scripted call; `pool.py`'s docstring says where they go).
- The teacher term's numbers (0.5, 0.95, cut-off 0.005) and the BC length are defaults with
  reasons, not measured; v2's BC ran 8x the Adam steps per update.
- The KL warning fired on every update after the first in the GPU check; if it fires routinely in
  a longer run, make 0.02 a stop checked after each epoch (for the author).
- Entropy stays on during the teacher phase (VPT dropped it there). The uniform average in BC
  includes the stage's early weights; the hand-off uses the EMA.
- Checkpoints repeat the games in progress and Adam's moments (43 MB at smoke size).
