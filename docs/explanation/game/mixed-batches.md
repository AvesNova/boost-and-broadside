---
covers: [src/bnb/game/layout.py, src/bnb/game/rules.py, src/bnb/contracts/pools.py, src/bnb/runtime/train.py, src/bnb/models/yemong/network.py]
reviewed: 2026-11-02
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
- **Per episode:** the arena scale, where the pool samples map sizes (below).

`bnb.contracts.pools.pool_game` gives a pool's game: the config with the pool's values
where it sets them. `bnb.game.api.rules_of(config, pools, environments)` then builds
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

## Map-size sampling

A pool with `arena_scale` (`bnb.contracts.ArenaScale`, game v19) plays a different arena
size each episode. At every reset each of its environments draws a scale: ln scale ~
Normal(0, sigma), truncated to [low, high]. The draw is one tick uniform through the
truncated inverse CDF (`bnb.game.layout.draw_arena_scale`), so there is no rejection loop
and no branch. The scale is kept in `Match.arena_scale` (B,). `bnb.game.layout.sized` makes
the episode's arena radius from it: the pool's radius times the scale, at least the mode's
minimum (`bnb.game.api.arena_radius_for`; team deathmatch 1000 m at map scale 1, 3000 m at
real scale). The map scale follows the radius, so every map length that scales with the map
moves with it: fields, zones (down to Frontline's zone floors), spawns and the boundary's
ramp. The observation's lengths stay normalised by the episode's own radius, which is in
the globals.

Only the radius tensors change, never a shape or a static value, so the compiled tick does
not recompile when the radii vary. This was checked by compiling a sampled tick with
dynamo's `error_on_recompile`{.plain} through 128 resets and 145 distinct radii: one frame, one
graph. The scale takes one more tick uniform, but only when some pool samples, so a batch
without sampling draws exactly as before. Pools without `arena_scale` play their own radius
every episode.

Training does not sample by default (`TrainConfig.arena_scale=None`): every map is
fixed-size until an agent has trained successfully on variable-size maps (the author's
rule, 2026-11-15). Sampling (sigma 0.35, bounds 0.5 to 2) is an opt-in setting for the
map-size arm. The rating lanes (`bnb.runtime.lanes.lane_pools`), evaluation and
play keep their games' own sizes. To rate at another size, give the game that radius
(`api.arena_radius_for(game, ships_per_team, scale)`).

## Field-count sampling

A game or pool with `active_fields` (`bnb.contracts.ActiveFields`: low, high) plays a drawn
number of its `num_fields` slots each episode; the rest are absent. The fields come in
point-symmetric pairs (with one centre field when the slot count is odd), so the count is
drawn uniformly among the counts in [low, high] that keep the map symmetric: even counts
only for an even slot count. The active fields are the first pairs, plus the centre field
for an odd count (`bnb.game.physics.spawning.draw_active_fields`). A map's fields are
independent draws, so the active ones lie as a map of that many fields would.

The state keeps the result as `Fields.present` (B, M). An absent field keeps its drawn
values and has no effect: the medium gives it a profile of exactly 0 (so flight, bullets
and spawns see the map without it), the sight test gives it no core and no outer disc,
and its observation row is zero but for `present` 0. The draw takes one tick uniform,
placed after every other, and only when some pool samples, so a batch without sampling
draws exactly as before and turning sampling on moves no other draw. In the rollout that
uniform is keyed by (seed, environment, episode) like the rest of the episode's map.

Training samples when `TrainConfig.active_fields` is set (default None: every field, in
every preset); the rating lanes keep their games' own fields. `bnb ratings --active-fields
LOW HIGH` rates on games that sample. A Yemong playing such games masks the absent fields
(`Yemong.mask_absent_fields`, [Yemong](../yemong.md)).

## Team-size sampling

A pool with `team_sizes` (`bnb.contracts.TeamSizes`) draws each episode's two team sizes,
equal or not, and its active field count together, as one row of a table (team 0's ships,
team 1's ships, fields, probability; `TeamSizes.table`). The ship distribution is a rule
over the pairs of sizes, larger a against smaller b, at least the mode's smallest team
(`bnb.game.api.MINIMUM_SHIPS_PER_TEAM`: Frontline 3, deathmatch and elimination 1), each uneven pair
split evenly between its two orientations:

| `rule` | the pairs' probabilities (S slots a side) |
|---|---|
| `uniform_pairs` (default) | S against S `full_share` (0.5), the rest evenly over the other pairs |
| `independent_teams` | each team on its own: S with chance sqrt(`full_share`), the other sizes alike |
| `near_full` | S against S `full_share`, the rest in proportion to `decay`^(2S - a - b) |
| `table` | `pairs`, (a, b, weight) rows, renormalised |

The field count follows the ships by default (`fields: per_ship`): the game's M field slots
times (a + b) / 2S, to the nearest count the map's pair symmetry plays, ties up (10 fields
at 5v5, 8 at 5v3, 2 at 1v1 on a 10-slot map), or drawn evenly from the `field_spread`
playable counts either side; `fields: game` leaves the count to the game (every field, or
the pool's `active_fields`). Each team's slots past its size are benched, the game's
mechanism for uneven teams (`bnb.game.physics.step`), the one the break-even lineups use:
kept as dead ships waiting to respawn, out of play, shown with the `benched` flag. The
fields past the count are absent. The batch keeps its one static size, so nothing
recompiles per size.

The draw is the runtime's, not the game's: the rollout draws each environment's row for
its k-th episode from one uniform keyed by (seed, environment, episode) under its own
purpose (`bnb.runtime.rollout`), so turning it on moves no other draw and the game's state
gains no leaf. The game takes the next episode's counts in `resolve` (`next_active`,
`next_field_count`): an environment whose episode ends spawns, benches and lays its fields
by them at its reset, and the rollout passes the ship counts from the next tick. The rules
that depend on the size read the active counts: team deathmatch's winning score is the
tickets for the enemy's ships (4 N^(4/3) to the nearest 5 at real scale: 5, 10, 15, 25
and 35 for 1 to 5 enemy ships; in 3v5 the 3-ship team needs 35 kills, the 5-ship team
15), elimination's wipe of the enemy's active ships. The clocks and elimination's shrink
are the same at every size (the freeze), and so is the arena. A benched slot is neither
valued nor rewarded, and the enemy's unforced events are split over the present slots
only (`bnb.learn.rewards.enemy_team_share`).

Training samples when `TrainConfig.team_sizes` is set (default None: every slot). The arm
A2 setting is provisional, `--set 'team_sizes={}'`: 5v5 in half the games, the other half
spread evenly over the other pairs (1 to 5 a side in deathmatch and elimination, 3 to 5 in
Frontline), one field per ship; the equations are a proposal to the author
(`docs/rewrite/a2-sampling-proposal.md`). It covers BC's data and
RL's games alike; the rating lanes, evaluation and play keep their games' own sizes. A
Yemong playing such games masks the benched ships and the absent fields
(`Yemong.mask_benched`, `mask_absent_fields`, [Yemong](../yemong.md)); the other models are
refused. The training log gets the drawn sizes per training mode, over the episodes ended
in each update (`TeamSizes.series`, on by default): `team_size/larger/{mode}`,
`team_size/smaller/{mode}`, `team_size/fields/{mode}` and `team_size/share/{a}v{b}/{mode}`.

## Lineup draw

Training draws each episode's lineup (`TrainConfig.lineup_draw`, on by default; decided
2026-11-14). Every slot of a team is one of the ship set's drawn types with equal chance
(`bnb.game.api.DRAWN_LINEUPS`: Yongma or Jujak in the real-scale set, Fighter or Heavy in
the standard one), and the other team flies the same lineup slot for slot, so neither side
gets the better ships. The draw is keyed by (seed, environment, episode), apart from the
map's tick uniforms, so turning it off moves no other draw (`bnb.runtime.rollout`,
`Rollout._lineups`). An episode that ends hands the next one's lineup to the game
(`resolve`'s `next_ship_type`), and its ships spawn as their new types. The rating lanes
draw too; the evaluation suite draws its games' lineups the same way
(`bnb.league.crossover.lineup_draw`). A fixed `TrainConfig.lineup` replaces the draw.

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
per-environment uniforms: state, events, observation and rewards agree to float32 rounding
on every tick
(`tests/sections/test_mixed.py::test_a_mixed_batch_equals_each_game_run_alone`).

## Cost

Measured at commit `81f54b32` on an RTX 4070 Laptop GPU under the benchmark lock, with
4096 environments, every game 5v5 with 10 fields and every seat random: the mixed game tick
took 0.807 to 0.812 ms, against a share-weighted 0.716 ms for the games alone (13% over).
A mixed Yemong rollout cost 0.3% more than the weighted single games. The full table and its
method are in `docs/rewrite/sections/mixed-notes.md`.
