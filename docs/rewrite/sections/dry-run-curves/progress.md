# Dry-run curves on every condition: progress

The brief: the first baseline (dry) run rates its curves on every evaluation condition (the author, 2026-11-16). The reference is [../../dry-run.md](../../dry-run.md).

## 2026-11-16

- **Tried:** extra rating lanes during training against rating the kept snapshots afterwards.
- **Chosen:** afterwards, in a rating pool per run with the suite's conditions as the pool's games. Lanes share the training rollout's fixed 5-a-side tensors: 10v10 and break-even can't be lanes without widening every training environment, and 1v1/2v2 would need lane benching (the A2 area).
- **Built (behind options, defaults unchanged):**
  - `bnb pool --presets` takes the suite's head-to-head condition names (`bnb.cli.named_game`, the suite's `game_of`);
  - `bnb pool create --lineup-types`: each game's lineup drawn as the suite draws it, keyed by the session's seed, the condition and the batch;
  - `bnb pool add --runs RUN ... --kinds ...`: a run's snapshots at the schedule, so a run with every update's files (R0) gives the same snapshots;
  - `bnb pool curves --figure PNG [--break-even DIR]`: the small multiples (`bnb.figures.ladder`).
- **Simulated (CPU, a Bradley–Terry truth, the real planner):** a 53-player session meets ±45 Elo in about 10,000 games per condition (190 per snapshot); ±60 still needs 8,400.
- **Bug fixed:** the fit and the plan ran their small matrix operations on every CPU thread. On the loaded laptop (load average about 50), that cost 30 s per batch for 53 players; on one thread it takes 0.3 s (`bnb.league.design.one_thread`, applied to `fit_posterior`, `pair_weights` and `plan_batch`). The same math, so ratings are unchanged.
- **Not measured:** the games' cost on the 5090. The estimate in dry-run.md is scaled from the pilot's laptop seconds per game. A GPU check needs Yemong v9 policies, and none exist on the laptop (v8 files no longer load).
- **Next:** the dry run itself; read the seconds per batch from the pool's records and replace the estimate.
