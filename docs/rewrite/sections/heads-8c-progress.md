# Queue item 8c: heads (progress)

Branch `heads-8c` (from core 75e9c8ea). Spec: docs/rewrite/queue.md "8c. Heads" (author
2026-11-28 ~04:05).

## Tried / worked

- Game side (one source of truth): `lean/game/core.py` "relative geometry": `boundary_distance`,
  `field_edge_distance`, `world_velocity`, `launch_velocity`, `bullet_reach` /
  `bullet_flight_time`, `visible_rows`, `geometry_record`, `pair_geometry`. The step, `medium`,
  `gun_range` and `observe` now call them (game quick tier unchanged: 120 passed).
- Next-state and hit on the rows the seat sees (`observed` = key_mask & visible): hidden enemies
  and dead ships masked; next-state also needs the ship visible after the step.
- Pairwise head (`lean/model/geometry.py`): per-ship gated projections, GELU(u_i + v_j), a Linear
  per target; ship pairs (range, range rate, line-of-sight rate, energy difference on the upper
  triangle; train angle and gun error in both orders), ship-field edge distance; the map edge on
  the auxiliary head group.
- Pair sampling (`lean/learn/geometry.py`): K valid pairs per view and tick without replacement
  (top K of uniform keys drawn eagerly per micro-batch, an input to the compiled loss), weight c /
  min(K, c): unbiased, exact at K >= c (tested).
- Target ranges measured on the CPU (scripted 4v4 training game, 64 x 1,800 ticks): table in
  lean/model/NOTES.md "Geometry heads".
- The first measurement found the lead's fixed point diverging beyond the gun range (NaN on a
  far pair: the flight time grows with the lead distance); the flight time is now capped at the
  bullet's life. A bullet the game flies along the gun solution passes its target within 3 cm.

## Didn't / open

- Coefficients (pair 0.1, field 0.05, edge 0.05) and K = 8 are provisional: chosen by analogy
  (the pair columns are shaping targets like the next state's 0.1; the single-column field and
  edge targets like hit's 0.05), not measured. No training run yet.
- A pre-8c checkpoint does not load with the heads on (load it with geometry_heads=False).

## Cost (K = 4 / 8 / all)

`docs/rewrite/sections/heads-8c/cost.py` (results in `heads-8c/cost.jsonl`), unit bnb-8c-cost,
04:56-05:00 on a quiet machine (no other bnb unit, load about 2, performance profile, RTX 4070
laptop). Compiled `update`, 1 epoch, one minibatch of three micro-batches of 64 views x 128
ticks, 4v4 (8 slots, 10 fields), bf16; median of 6 steady updates after the compiling one; each
variant in its own process with an empty inductor cache. K is both families (ship pairs and
ship-field pairs). Logits: the geometry heads' per view and tick (ship pairs K x (4 x 32
one-order + 2 x 2 x 32 both-order) + field pairs K x 32 + 8 ships x 16 edge).

| Variant | Geometry logits / view-tick | Per micro-batch (ms) | Update (ms) | Peak allocated (GiB) | First update, cold compile included (s) |
|---|---|---|---|---|---|
| off | 0 | 150.2 | 450.6 | 4.24 | 32.5 |
| K = 4 | 1,280 | 162.1 (+7.9%) | 486.4 | 4.57 | 47.9 |
| K = 8 | 2,432 | 165.4 (+10.1%) | 496.3 | 4.59 | 47.9 |
| all (28 + 80) | 9,856 | 193.1 (+28.6%) | 579.2 | 5.38 | 49.2 |

Spread of the six steady updates within 1 ms. K = 4 to 8 adds 3.3 ms; most of the K = 8 cost
is fixed (the per-ship projections of every ship, the targets, the edge column, the top-K).
The default stays K = 8 (provisional).

Compile check (`compile_check.py`, 05:00 and 05:03, quiet): 5 graphs, no graph break, no
recompile; minibatch_loss 861 -> 1,627 traced operations (+89%: the geometry losses' per-column
Python loop unrolls), act 4,535 -> 4,613, advance 3,194 -> 3,307; cold 142.2 s and 141.6 s
(120.9 before; budget 150 s). Baseline reset on this branch (05:05).

## Next

For item 4 (training review): the fixed cost (fold the columns' supports into one padded
tensor, one cross-entropy call); the store's extra 32 bytes per ship-tick of the geometry
record at the real update size.
