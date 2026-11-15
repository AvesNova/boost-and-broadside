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
- **Not measured:** the cost on the 5090. The estimate in dry-run.md is scaled from laptop ladder and pilot numbers. A GPU check needs Yemong v9 policies, and none exist on the laptop (v8 files no longer load). Planning a 53-player session was not timed on a real pool either.
- **Next:** the dry run itself; read the seconds per batch from the pool's records and replace the estimate.
