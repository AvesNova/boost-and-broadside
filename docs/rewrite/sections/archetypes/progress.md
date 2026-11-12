# Archetype pairs: progress

Branch `tune/archetypes` (from `origin/tune/tank-armour` 010a0931). The task: pairs of
different ship types, each with its own way to fly, where all three matchups (A v A, B v B,
A v B) are fair and fun at every fleet size (1v1 to 10v10, 20v20 a spot check). Fun: more
decisions and more control; not a tank that wins by wearing the opponent down. Hard stop
08:00 KST. GPU: gpubox7 (RTX 5060 Ti 16 GB, shared with other agents' benches).

## Code

- **Arming distance** (`arming_distance`, a card field, the shooter's; 0 = off, every ship
  file): a hit's damage times a smoothstep of the distance the bullet has travelled over
  `[0, X]`, the distance being its moves so far (this one included) at its present speed
  (`physics/combat.py`, beside the closeness and the aspect). A card tensor, so sweeps never
  recompile; a handle too (`Handles.arming_distance`, through `weapons.combat_card` /
  `combat_handles`). Test: `test_arming_distance_ramps_the_shooters_damage_over_the_first_metres`.
  The researched types' notes now carry the head-on sharpness and the arming distance (the
  notes test had failed since the sharpness landed).

## Tools

- `pairs.py`: a pair as `sizes.py` candidates. Ship A sits in the card table's first slot
  (the Yongma row: lineup "energy"), ship B in the second (the Jujak row: "turn"). A hull sets
  a slot's flight fields: `Y`, `J` (the frozen files), `M` (their geometric mean in flight,
  shield, damage and size: one neutral ship in both slots), then factors.
- `sizes.py` (tank-armour's, unchanged): the candidates in self-play per size, one compiled
  loop a size.
- `score.py`: every goal in one row per candidate, and a `badness` (the summed misses against
  the goals, 0 when all are met) to sort by.
- `make_map.py`: a candidate as an opt-in play map.

Columns: A v B is ship A's score (win = more kills, draws half). AA / BB are the 1v1 mirrors:
mutual head-on kill share, life (s), merges a kill mean / median / p90, H the post-merge choice
entropy in bits (same sense, opposite sense, extend, turn v extender; r9 never extends, so
the practical ceiling is 1 bit: same v opposite sense evenly).
