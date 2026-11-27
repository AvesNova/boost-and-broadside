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

- Coefficients (pair 0.1, field 0.05, edge 0.05) and K = 8 are provisional.
- A pre-8c checkpoint does not load with the heads on (load it with geometry_heads=False).

## Cost (K = 4 / 8 / all)

Pending: `docs/rewrite/sections/heads-8c/cost.py`, unit bnb-8c-cost, launched 04:55 on a quiet
machine (no other bnb unit, load 2.0, performance profile).

## Next

Cost table, compile check against the 150 s budget, the slow learn tests, the quick tier.
