---
covers: [src/bnb/game/layout.py, src/bnb/game/rules.py, src/bnb/contracts/pools.py, src/bnb/runtime/train.py, src/bnb/models/yemong/network.py]
---

# Mixed batches

One batch can hold several games: team deathmatch on two maps and Frontline, fogged and
clear, in the same tensors and one compiled graph. The training mix does this. The spec is
`docs/rewrite/sections/mixed-spec.md`.

## What may differ, and what may not

- **Shared by the whole batch:** the sizes, ships per team and fields, from the
  `GameConfig`. Every environment has the same tensor shapes.
- **Per pool:** the mode, the arena radius, the spawn groups and the two clocks
  (`bnb.contracts.Pool`). Every environment of a pool shares them.

`bnb.contracts.pools.pool_game` gives a pool's game: the config with the pool's values
where it sets them. `bnb.game.api.rules_of(config, pools, num_envs)` then builds
`Rules.pools`, a tuple of `bnb.game.rules.PoolRules`, one per game, as static slices of the
batch in pool order. Adjacent pools of the same game merge; if one game remains, it covers
the whole batch, so a single-game run compiles as it would without pools.

## Two kinds of step

**Shared steps** run once over the whole batch: flight, bullets, hits, damage, the
boundary, the map's fields, the formation, the clocks, sight and most of the observation.
They read the per-environment rules from `bnb.game.layout.Environment`: the arena radius,
the map scale, the episode length and the respawn wait, as `(B,)` tensors built from the
static pools.

**Mode-owned steps** run on their mode's slice only: spawn protection, the mode phase and
its clock result, a new match's map and mode state, spawn placement, and the mode's
observation. `bnb.game.layout.per_mode` cuts every tensor at the static slice bounds, calls
the mode's function on each slice and concatenates the results along the batch. One slice
is the whole batch, uncut. No environment pays for another mode's rules.

`bnb.game.layout.slices` cuts the batch where the mode changes, so two pools of one mode
with different arenas or clocks run in one slice.

## Fixed shapes across modes

- **Zones.** Every environment carries the same zone slots. In a mode without zones they
  are absent: `present` 0, every other column 0, no effect on the game. The mode's own code
  passes them through without copying.
- **Tick uniforms.** One width for the batch, the widest any mode needs. Each environment
  reads its own mode's prefix.
- **In Yemong,** an absent zone's map-memory token is one learned vector
  (`Encoder.absent_zone` in `src/bnb/models/yemong/network.py`), chosen with `where` rather
  than masked out, so the fused attention kernel stays usable. It takes the global token's
  rotation, so it reveals nothing about the view's position offset.

## The training mix

`bnb.runtime.train.mixed` crosses every game of `TRAINING_MIX` with every pool of the
lineup, game by game, so each game is one contiguous slice. The tables are in the
[training guide](../../guides/training.md).

## Equivalence

A mixed batch equals each game run alone, environment by environment, given the same
per-environment uniforms: state, events, observation and rewards
(`tests/sections/test_mixed.py::test_a_mixed_batch_equals_each_game_run_alone`). On the
CPU this is bit for bit when each pool is a whole number of vector widths. Otherwise eager
kernels compute a tensor's tail with scalar code, whose `exp` or `sin` can differ in the
last bit, and the test holds the batch to 1e-6 per tick instead
(`tests/sections/test_mixed.py::test_an_unaligned_mixed_batch_equals_each_game_alone_to_float32_rounding`).

## Cost

Measured by the mixed section after its fix round (commit `81f54b32`), on an RTX 4070
Laptop GPU under the benchmark lock, 4096 environments, every game 5v5 with 10 fields and
every seat random: the mixed game tick took
0.807 to 0.812 ms against a share-weighted 0.716 ms for the games alone, 13% over. A
mixed Yemong rollout cost 0.3% more than the weighted single games. The full table and its
method are in `docs/rewrite/sections/mixed-notes.md`.
