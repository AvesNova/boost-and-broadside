# lean/learn and lean/eval: notes

What is, and what is open. History: `docs/rewrite/core-worklog.md` and this file's git log
(the long notes before 2026-11-23's step L1 are there). Decisions: `docs/rewrite/core-decisions.md`.

## Modules

- `config.py`: `LearnConfig` (every number, with its reason), `learning_rate(config, step)`.
- `rollout.py`: `start`, `collect`, `abandon`, `Opponent`, `CollectCarry`, `Rollout`; seats,
  the league side (`network_actions`, capacity routing), hit labels, the teacher's
  distribution (`teacher_marginals`, `teacher_joint`).
- `store.py`: `RolloutStore`, an update's segments, compact, streamed to pinned host memory on a GPU.
- `returns.py`: GAE per level, chaining across segments, advantage scaling and shares, the
  value supports' range monitoring.
- `training.py`: `update` (both stages), `TeacherTerm`, `make_optimiser`, the losses.
- `weights.py`: `WeightAverages` (EMA and uniform), checkpoints.
- `pool.py`: `League` (members, network slots, capacity draws, tallies, retiring slots),
  `SnapshotSchedule`, `retained`.
- `rewards.py`: v2's five reward levels and the damage ledger.
- `compiled.py`: `graphs()`, the compiled tick halves, the league routing, target pass and loss.
- `lean/train.py`: the stages, checkpoints, `--resume`. `lean/eval`: the runner, the result store,
  Elo, `crossover` (lean/eval/NOTES.md); `python -m lean.eval`.

## Commands

    python -m lean.train                                   # BC (96 updates), then RL (--updates, 1000)
    python -m lean.train --stage bc --out DIR              # BC only: DIR/bc.pt
    python -m lean.train --stage rl --start DIR/bc.pt --out DIR   # RL from the BC hand-off: DIR/rl.pt
    python -m lean.train --resume DIR [--updates N]        # from DIR's latest checkpoint (or a file)
    python -m lean.train --smoke --device cpu [--stage bc|rl]     # tiny, eager
    python -m lean.eval crossover STORE DIR/rl.pt [--weights ema|uniform|live] [--sizes 4 8] [--games 64]

## The recipe (decided 2026-11-23)

- **BC stage** (`update(..., cloning=True)`): the student plays its own games (half self-play,
  half against the scripted agent; no snapshots join the league). Loss: the cross-entropy of
  the 42-way policy against the scripted teacher's full distribution (the product of
  `lean.scripted.Marginals`' stick, power and fire, recorded per ship each tick) at
  `bc_coefficient` 1.0, plus value, next-state and hit as in RL; no surrogate, no entropy
  bonus. The critics learn the stage's own returns. 96 updates by default: 100.7M ticks at
  parity, v2's 100M-tick stage. `bc_kl` is the cross-entropy less the teacher's entropy.
- **Hand-off**: RL starts from the BC stage's EMA (`handoff`), with a fresh optimiser, warm-up,
  league and averages.
- **Teacher term in RL** (`TeacherTerm`): the same cross-entropy at 0.5 x 0.95^k at RL update k,
  off for good once the league's win rate against the scripted agent reaches 0.45 (it reads 0
  until a game against it has ended) or the weight falls below 0.005 (from RL update 90).
- **Weight averages** (`weights.py`): an EMA (0.95 per update) and a uniform mean of the weights
  after every update from the end of the warm-up; side copies, never read by training; saved
  with the live weights in each stage's checkpoint (`DIR/bc.pt`, `DIR/rl.pt` at the stage's end).
- **Optimiser**: Adam, no weight decay, epsilon 1e-5, fused (on the CPU too), global clip 1.0;
  a step whose gradient norm is not finite is skipped through the fused optimiser's
  `found_inf` (no host read) and counted (`skipped_steps`, a warning line). Learning rate 3e-4
  after a linear warm-up of 2,000 Adam steps (16 updates of 128 steps), counted per stage.
- **KL**: no early stop; `approximate_kl_epoch_<k>` per epoch, `kl_warning` above 0.02.
- **Advantages**: summed over levels, divided by their standard deviation over the update,
  mean kept. `advantage_share_<level>` = Cov(A_level, A) / Var(A) (the shares add up to 1),
  `advantage_std` the scale before the division.
- **Losses**: fixed coefficients on scale-free terms: surrogate 1, value 0.05 (nats per level),
  next-state 0.1 (nats per column), hit 0.05 (nats), entropy 0.003 on the joint 42-way
  entropy in RL; `entropy_fraction` = H / ln 42. No gradient-share tuning (lean never had it).
- **Crossover**: evaluation only (`python -m lean.eval`), not in the training loop.

## The league (decided 2026-11-23, built in step L2)

- Split: environments 0 to E/2 - 1 self-play, the rest (L) league games; the live policy's
  games continue across updates.
- Members (`pool.League.kinds`, fixed indices): the scripted agent (0), the EMA (1) and the
  uniform average (2), then up to 12 snapshots. Before the first snapshot every league game
  is against the scripted agent; the EMA joins with the first snapshot, the uniform average
  with it once it holds an update. The averages' slots take the new averages after every
  update; their running games continue (one moving player, like the live policy).
- Draws at a game's end: PFSP (1 - p)^2 with a 1/8 uniform floor (AlphaStar's decaying
  tallies, 0.99 a game, unplayed 0.5) within row capacities ceil(1.25 L / present). Rule:
  independent PFSP draws over the members with a free row; draws past a member's free rows
  (later environments) drawn once more over the members still free; what is left takes the
  free rows in member order (scripted first). Exact PFSP when nobody is full.
- Routing: `carry.slot` names the network slot playing each league environment. Each tick,
  for each of the K slots holding weights, `rollout.slot_actions` takes that slot's
  environments in order into a batch of W rows on the device (rank by a cumsum; W = the
  capacity, or more where a slot already holds more rows), runs the slot's network once on
  them and reads each environment's row back; padding rows are never read. One compiled
  graph serves every slot, K and W (the slot is a 0-d tensor, W a symbolic size): no
  recompile as the league changes. `League.prepare` builds the table (the K slots' weights,
  W) before each segment, with one host read of the rows per slot. A vmap over the K stacked
  weights was faster (below) but recompiled at every new K and every new W.
- Retiring slots: 2 + 12 + 4 network slots. A dropped snapshot's member index goes at once
  (its games RETIRED, not tallied), its slot keeps its weights and plays its running games
  to their end, then is freed. Sizing in `LearnConfig.retiring_slots`: retention drops at most
  2 at one addition, drops start at the 11th snapshot (75M) where additions are 15M+ steps
  (14+ updates) apart, and a game lasts at most 5,528 ticks (11 updates): 2 suffice at
  parity, 4 are kept. With none free the games of the retiring slot with the fewest are
  re-dealt (`League.abandoned`, a warning line).
- Schedule: v2's LEAGUE_SNAPSHOTS (5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110, 130,
  160M RL steps, then every 30M), the live weights after the update that crosses each.
  Retention: v2's (the newest 8, plus the snapshot nearest each age 2^k back, k >= 3), at
  most 12 (the oldest go past it): 11 kept to 500M steps, 12 to about 1.5B.
- Checkpoints (`lean.train`): every 10M environment steps of a stage,
  DIR/checkpoints/<stage>_<steps>.pt, none deleted: live weights, averages, optimiser state,
  league (tallies, which snapshot each slot holds; no slot weights), teacher term, step counters, the games in
  progress (game state, carry) and the generators' states. Each league snapshot's weights are
  written once, when taken, to DIR/snapshots/<steps>.pt (training dtype); a checkpoint names
  them (`slot_snapshot`) and a resume reads them back (a retiring snapshot is still a file),
  copying them into the new run's directory when that differs. The averages' two slots are
  rebuilt from the averages on load. A checkpoint is then live + both averages + Adam's two
  moments (about 5 snapshots' worth, 43 MB at smoke size, constant as the league fills; it was
  60 to 113 MB there with 3 to 6 snapshots) plus the games. `--resume` continues with the
  checkpoint's settings and plan; a checkpoint copied elsewhere needs the snapshots directory beside its
  checkpoints directory.

## Rollouts and the update (as built before L1, unchanged)

- Seats: the first half of the environments are self-play (two live seats each), the rest
  league games (one live seat, see "The league"). Deals are drawn once per segment.
- An update is 2,048 environments x 4 segments x 128 ticks; returns chained over the 512 ticks
  of each seat, bootstrapped only at the end; hit labels chained across segments; 32
  minibatches of views, micro-batches of 64 views with gradient accumulation; targets once per
  update from the stored values (`targets_every_epoch=False`).
- Rewards: v2's five levels with the ledger (`rewards.py`); supports damage +-7, kill_death +-8,
  outcome +-12, objective levels +-1 (placeholders); range warnings per level.
- The store keeps positions, PPO numbers float32, other floats (the teacher's 11 probabilities
  included) float16, actions int8; the teacher adds 22 bytes per ship row.
- Compiled: `dynamic=True`, duck sizing off from the first compiled call. Four graphs (act,
  slot_actions, advance, minibatch_loss), each compiled once, 0 graph breaks (step K; counted
  by docs/rewrite/sections/scoreboard/compile_check.py, the compile tripwire).

## Ran (step L2, 2026-11-23)

CPU (OMP_NUM_THREADS=4): `pytest lean -m "not slow"`: 197 passed; slow: 4 passed (the routing and
`advance` compile fullgraph; the routing graph is reused at a new K and W). `python -m
lean.train --smoke --device cpu` (both stages) and `--stage rl --updates 4` then `--resume`
from its first checkpoint: the resumed updates print the same metrics. `--stage rl --start`
from the smoke BC checkpoint, and `python -m lean.eval` on its rl.pt.

GPU (laptop RTX 4070, profile "performance", nothing else on the GPU; units bnb-l2-gpu,
bnb-l2-gpu2). The league side per tick (`league_actions`, compiled, bf16, 200 ticks after 20
warm-up, members drawn by PFSP from unplayed tallies):

| environments (league) | core: vmap every slot on every row, 8 slots | L2 vmap, 8 slots | L2 per slot, 8 slots | L2 per slot, 11 slots (12 members) |
|---|---|---|---|---|
| 256 (128) | 8.00 ms | 1.56 ms | 11.13 ms | 14.77 ms |
| 2,048 (1,024) | 81.5 ms | 10.5 ms | 12.75 ms | 15.29 ms |

The per-slot form is launch-bound at small sizes (about 1.35 ms a slot at 256 environments);
at parity it costs 22% more than the vmap form and 84% less than core's. One short RL run each
at parity (2,048 x 4 x 128, compiled, bf16; RL from fresh weights): 6,670 steps/s before the
first snapshot (updates 1 to 3); with the vmap form and the EMA plus one snapshot playing,
6,490 (updates 6 to 9); with the per-slot form only update 5 had networks playing, 4,852,
its first compile of the routing graph included (and one recompile each of the routing
graph and `advance` at that update, guard "'NoneType' object has no attribute 'stride'",
cause not found; none after it within the update). A checkpoint at 10.49M steps (2
snapshots and the EMA in the league) is 249 MB.

## Compile (step K, 2026-11-24)

The recompile of the routing graph (at its second slot) and of `advance` (when the league's
first network joins, BC to RL) was a view: the scripted rows of the league side were a slice
of the scripted actions, and Dynamo guards a view's base; the next input (a graph's output)
has none ("'NoneType' object has no attribute 'stride'"). `rollout.league_rows` copies them;
test_compiled.py checks 2 graphs where there were 4. Compile table, causes and the tripwire:
docs/rewrite/scoreboard.md "Compile time". Run the tripwire at the end of every step:

    PYTHONPATH=. python docs/rewrite/sections/scoreboard/compile_check.py

Checks (2026-11-24, on 245e537f plus K): `pytest lean -m "not slow"` 224 passed; slow
test_compiled.py 4 passed; `python -m lean.train --smoke --device cpu` completes.

## Tests

All learner, evaluation and caller tests run the real game, scripted agent and model at tiny
sizes (`tests/setting.py`; the stand-ins are gone). `StepRecorder` records the game steps
(hit-label and self-play checks against a search over them). `test_recipe.py`: the teacher's
joint, the soft-target cross-entropy, the BC stage training every head and ignoring the
advantages, the teacher term's decay, gate and latch, the averages' formulas and that they are
not fed back, checkpoints, the optimiser settings and the non-finite skip, the warm-up, the KL
monitor, the advantage shares. `test_train.py`: `lean.train` BC then RL from the checkpoint;
checkpoints every period and a resume giving the same updates as a run that never stopped.
`test_league.py`: rows per member never above the capacity, draws matching PFSP when no
member is full, a full member's draws going to the others, the draws compiling fullgraph
(no host read), routing giving each environment its own member's step, a dropped
snapshot's game finishing with its old weights in a retiring slot (and the re-deal when no
slot is free), the averages joining and following their weights, v2's schedule and
retention. Slow (`test_compiled.py`): the routing and the tick's second half (tallies,
capacity draws at re-deal) compile fullgraph.

## Ran (step L1, 2026-11-23, CPU, OMP_NUM_THREADS=4)

`pytest lean -m "not slow"`: 185 passed. Slow: 3 passed (compiled collect and update, compiled
loss equals eager). `python -m lean.train --smoke --device cpu` with `--stage bc`, `--stage rl
--start`, and the default (both). `python -m lean.eval` on the smoke checkpoint.

GPU check (laptop RTX 4070, profile "performance", nothing else on the GPU; unit
bnb-l1-gpu, 20:27 to 20:46): `python -m lean.train --stage all --bc-updates 3 --updates 3
--device cuda` at parity (2,048 x 4 x 128, compiled, bf16, core game). Both stages ran and saved
their checkpoints; no step skipped. Loop steps per second (collect plus update): BC 3,368
(update 0, with the cold compile), 6,669, 6,668; RL (same process, graphs reused) 6,667, 6,669,
6,671. Peak allocated 5.68 GiB (BC), 5.70 GiB (RL). BC's teacher cross-entropy 3.65, 2.91,
2.23 nats (bc_kl 2.69, 1.87, 1.14); RL from the 3-update EMA started at 3.50 (the EMA is
still near the initial weights after 3 updates). The KL monitor fired from update 1 in both
stages: epoch means up to 0.34 (BC) and 0.18 (RL) in the fourth epoch, while the warm-up's rate
was below a fifth of 3e-4. One run each.

## Open

- Step C: the cheese agents as league members of kind "scripted" after the snapshots (no
  slot; their rows need their own scripted call: today every scripted row takes the one
  scripted agent's action). `pool.py`'s docstring says where they go.
- The routing's cost at small sizes (launch-bound, one compiled call a slot). Step K tried
  one CUDA graph of every slot's call per network table: collect at 256 environments 10,700
  -> 18,325 steps/s, at parity nothing (loop 6,320 against 6,301; +0.28 GiB reserved), so it
  was reverted (commit 992bb7ef and its revert hold it).
- Steady steps per second with league networks at parity under the per-slot form: not
  measured (the short run had one such update).
- Checkpoints repeat the games in progress and Adam's moments (backlog P done: snapshots are
  written once, see Checkpoints above; the 249 MB figure in "Ran (step L2)" is the old form).
- The teacher term's numbers (0.5, 0.95 per update, cut-off 0.005) and the BC length are
  defaults with reasons, not measured here; v2's BC ran 256 minibatches (8x the Adam steps per
  update); lean's BC uses the RL minibatches (32).
- The KL warning fired on every update after the first in the GPU check (fourth-epoch means
  0.14 to 0.34, early in the warm-up). The literature's rule: if it fires routinely, make 0.02
  a stop checked after each epoch. For the author once a longer run shows whether it persists.
- Entropy stays on during the teacher phase (VPT dropped it there); the decision is a constant 0.003.
- The uniform average in BC starts after the warm-up and includes the stage's early weights;
  the hand-off uses the EMA.
- The events bridge (`rewards.tick_events`) and the supports: as before (objective supports are
  placeholders).
