# League spec (section 8): opponents and ratings

Status: spec, 2026-11-01. Decisions: `league.md`. Fixtures: `tests/reference/league/`
(README there). **(v2)** marks a decided change from v1. **(proposal)** marks a choice the
spec makes, implemented as written and listed for the author.

Order: after sections 4 and 7. The stationary ladder is section 4's controller (spec A6).

## 1. The pool

- **`bnb/league/`** holds the pool, its sampling, the ratings and the records.
- **Entries:**
  - **checkpoints:** frozen snapshots of the live policy;
  - **the uniform average and the EMA** of the live policy's weights, updated every
    update, the EMA with decay 0.95 per update;
  - **the stationary ladder:** random (`p = 0`), the rungs `p = 0.1 … 0.9` and 0.95, and
    the scripted agent (`p = 1`).
- **Self-play is not an entry:** it is a fixed share, 0.5 of each mode's pool.
- **Every mode:** each mode's pool (section 3c) is split into its self-play block and its
  opponent block, so every entry plays every mode.

## 2. Slots and the compiled graph

- **Opponent slots:** 4 network slots loaded at once.
  - Their parameters are stacked, `(slots + 1, ...)`. The extra set is for the drain.
  - One vmapped forward serves every network opponent environment (review 4b A1).
  - Each slot keeps its own carry, per environment.
- **The stationary ladder** runs as one controller over its environments: one scripted
  pass and one uniform draw, then each environment's `p` decides.
- **The per-environment opponent:** an index tensor (which slot, or the ladder) and the
  ladder's `p`, both `(B,)`. Actions are selected with `where`.
- **Inside the captured graph:** self-play, network opponents, the ladder and the
  sampling all run in the rollout's compiled, CUDA-graph-captured stages.
  - No host synchronisation per step.
  - Sampling draws from `fold_in` streams, so it is deterministic.
- **Changing an opponent (the drain, review 4b B1):**
  - one slot per rollout shard, round robin, receives a newly sampled entry;
  - its weights are copied **in place** into the spare parameter set, so the captured
    graph stays valid;
  - each environment of the slot switches to the new generation only when its own episode
    ends. No opponent changes mid-episode;
  - at most one slot drains at a time.
- **A ladder environment** draws a new `p` at each of its resets, from the sampling
  weights restricted to ladder entries.

## 3. Sampling

- **The weight of an entry** is `p(1 − p)`, where `p` is the predicted probability that
  the live policy beats it, from the live gauge (section 4).
- **The uniform floor:** 5% of the pool's weight, spread evenly over the entries.
- **Per mode:** each mode's pool samples by that mode's ratings.
- **The draw happens in-graph** from the rating vector, at each slot swap and each ladder
  reset.

## 4. Ratings during training: the live gauge

- **The fixed gauge:** random 0, rung `p` at `1000 p`, scripted 1000. These are not
  estimates and are never refitted during training **(decided)**. Below 1000 the gauge
  says where the live policy sits on the ladder; above 1000 it is a rating against the
  scripted anchor and the checkpoints. It is named `live_elo` everywhere and is never
  published.
- **Two stages, once per update, per mode** (v1's `live_rating`, fixtures
  `league_ladder` and `league_live`):
  1. **The ladder:** the checkpoints' ratings, fitted to the accumulated record among
     stationary players with the gauge held (`fit_ladder`, prior 1 game at the strongest
     fixed rating).
  2. **The live policy:** rated from **this update's** record alone against the entries'
     current ratings (`rate_live`, no prior; ties as half wins). One update, not a window:
     v1 measured every window as worse.
  - A sweep is `±inf`. The display holds the last finite value; nothing averages it.
- **A new snapshot** starts at the live rating of the update it was taken at, and the
  ladder fit refines it.
- **The averages move, so they are not in the ladder fit.**
  - Their sampling rating is the live rating plus the offset their recent record against
    the live policy implies (`rate_against`, over their last 200 games) **(proposal)**.
  - With no record yet, they take the live rating.
- **Ratings per mode and map:** Frontline, team deathmatch squad, and team deathmatch
  battle map are separate gauges, records and ladders.
- **The record:** a per-mode win/loss/tie matrix among entries. Only games between
  stationary players feed the ladder; the live player's games feed stage 2.

## 5. Snapshots and retention

- **A snapshot every N environment steps (deterministic).**
  - The default N gives about 30 minutes at the preset's measured throughput; the run
    records N.
  - Wall-clock triggers are not used, because they would break reproducibility.
- **Retention:**
  - the newest 8 snapshots;
  - plus a geometric history: of the older ones, the snapshot nearest each age `2^k`
    snapshots back, for `k ≥ 4` **(proposal)**.
- **On disk:** every retained snapshot. Only the 4 slots are on the GPU.
- **Resuming:** the run's checkpoint holds the pool (entries and their weights on disk),
  the ratings, the records and the sampling state. A resumed run continues identically.

## 6. Published ratings (after training)

- **`bnb ratings`:** an offline tournament over a field (the run's snapshots, the
  averages, the ladder, other runs' entries), in batches.
  - The first batch is spread evenly. Later batches are split by A-optimal value
    (`allocate`, fixture `league_allocation`), with at least one game per pair, and
    every pair's games split evenly between the seats.
  - Draws are scored as half a win.
- **The fit:** Bradley–Terry with standard errors (`fit_ratings`, fixtures
  `league_ratings*`), prior 1 game, then shifted so the scripted agent is 1000. **Only
  the scripted agent is pinned**; random is fitted too.
- **Stopping:** when every rating's standard error is below a target (default 25 points)
  or a game budget runs out.
- **The leaderboard:** keyed by game version × mode × map × fleet size; each row has the
  rating, its standard error and the games behind it. Named `calibrated_elo`.

## 7. Metrics

- Per mode: the live gauge and its standard error, and each entry's rating, sampling share
  and win rate against the live policy.
- The pool: entries, slots, the draining slot, snapshot count.
- Games per update per opponent kind.

## 8. Acceptance

1. **Equivalence:** `tests/reference/league` passes through `league_arm.py`.
2. **Rating recovery:** on synthetic matches from known ratings (both the per-update live
   rating and the offline fit), the recovered ratings fall within two standard errors of
   the truth in at least 90% of 200 seeded trials.
3. **Sampling:**
   - the empirical shares over many draws match `p(1 − p)` plus the floor;
   - the same seed gives the same opponents, run to run.
4. **The drain:** no environment's opponent changes mid-episode; after a swap every
   environment reaches the new generation by its next episode end; at most one slot
   drains.
5. **The graph:** a rollout with the league is CUDA-graph captured, with no host
   synchronisation per step (the runtime's capture test), and a swap does not recapture.
6. **Resuming:** a run stopped and resumed draws the same opponents and reports the same
   ratings as an uninterrupted run.
7. **Speed:** the rollout's cost with the league against self-play only, at the 5v5
   training cells with Yemong. The extra cost should be the opponents' forwards and
   little more; report it.
