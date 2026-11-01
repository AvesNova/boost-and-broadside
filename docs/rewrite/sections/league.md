# Section 8: league and ratings

Status: decisions settled, 2026-11-01. Spec: `league-spec.md`. Fixtures: `tests/reference/league/`.

## Already decided (training review 4b, 4e and the 2026-10-30 findings)

- **No dedicated evaluator.** Live ratings come from the games the training rollouts
  already play.
- **One pool for every opponent kind:** checkpoints, the averaged policies, the scripted
  agent, the rungs and random. Self-play is a fixed share, not a pool weight.
- **Sampling weight `p(1 − p)`** on the predicted win probability, plus a uniform floor.
- **Snapshots every N updates** (time-based); Elo milestones only after training.
- **Slots:** fused NN opponents with stacked parameters (A1), drained swaps with no
  mid-episode change (B1); revisited with profiles after implementation.
- **The leaderboard** is keyed by game version × mode × fleet size, with the scripted
  agent at 1000.

## Decided (author, 2026-11-01)

- **Both averaged policies,** the uniform average and an EMA, are pool entries. The EMA
  starts from v1's decay: 0.95 per update (a time constant of 20 updates). Testing other
  values is later.
- **Two different ratings:**
  - **The live gauge (during training):** random at 0, rung `p` at `1000 p`, the scripted
    agent at 1000, all **fixed**. During training no rung plays another rung or random;
    each plays only the live policy, so fitting their ratings would be circular. Below
    1000 the gauge says where the live policy sits on the ladder (700–800: between the
    0.7 and 0.8 rungs), not a true Elo. Above 1000 it is largely a real Elo against the
    scripted anchor and the checkpoints. Its job is a consistent live dashboard (W&B).
  - **Published ratings (after training):** a Bradley–Terry fit over a full field, with
    only the scripted agent pinned at 1000. Random is fitted too.
- **The ladder:** random and the rungs are sampled like any other entry. The rungs are
  v1's spacing in tenths, p = 0.1 … 0.9, plus v1's 0.95.
- **One pass for the whole stationary ladder (speed):** random, the rungs and the
  scripted agent are one controller with a per-view `p`. One scripted pass and one
  uniform draw over all its views, then per decision the choice. (Section 4's spec, A6.)
- **Ratings per mode:** separate ratings per mode and map: Frontline, team deathmatch on
  the squad map, and team deathmatch on the battle map.
- **Opponents in every mode:** each mode's pool is split into a self-play block and the
  opponent slots, so every opponent plays every mode. Self-play stays at a fixed share
  (0.5, as in v1).
- **One compiled graph:** self-play, the NN opponents, the stationary ladder and the
  sampling all run inside the captured rollout graph:
  - opponent parameters stacked, one vmapped forward;
  - per-view opponent index and `p` as tensors;
  - selection with `where`, sampling in-graph from `fold_in` streams;
  - no host synchronisation per step;
  - swaps copy weights into the stacked buffer in place between shards, so the captured
    graph stays valid.
- **The pool:**
  - 4 opponent slots loaded at once;
  - a snapshot every N updates, N set by time (about 30 minutes of training);
  - the newest 8 snapshots kept, plus a geometrically spaced history;
  - a uniform floor of 5% of the pool's weight.
