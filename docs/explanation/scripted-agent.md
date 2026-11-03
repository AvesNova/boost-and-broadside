---
covers: [src/bnb/controllers/scripted.py, src/bnb/controllers/scripted_parameters.py, src/bnb/controllers/scripted_combat.py, src/bnb/controllers/scripted_frontline.py, src/bnb/game/handling/characteristics.py, src/bnb/runtime/evaluate.py]
reviewed: 2026-11-03 09:45
---

# The scripted agent

<!-- generated: section-status -->

The scripted agent is one memoryless [controller](glossary.md#controller),
`bnb.controllers.scripted.Scripted`, with three uses: a league opponent, the rungs of the
stationary ladder between random and scripted play, and the teacher for
[behaviour cloning](behaviour-cloning.md). Its carry is `None`.

## What it reads and gives

It reads only the observation, the parts `Scripted.reads` names: the ship rows, the zone
rows and the globals. It never reads `bnb.contracts.Privileged` or the game state, so it
sees exactly what a model sees, fog included, and a student can match it.

`bnb.controllers.scripted.distribution` is the pure function behind it: a team's views in,
a `(R, S, 42)` distribution over the joint action out, in `bnb.contracts.action.to_joint`
order. The joint is the outer product of three marginals, power (3), stick (7) and shoot
(2), which `bnb.controllers.scripted.marginals` returns. `Scripted.step` samples each
marginal by its inverse CDF from the default generator, so a captured graph draws fresh
numbers on every replay. A dead ship, or one waiting to respawn, takes the no-op action
with probability 1.

The global mode one-hot selects the layers: team deathmatch runs the combat layer;
Frontline runs the Frontline layer on top of it. Both are computed for every view and the
mode selects per view, so a mixed batch is one graph.

## Characteristics

Every threshold is a dimensionless parameter times a characteristic quantity of the
observing ship's type (or the target's, for its hit radius). The quantities are
`bnb.game.handling.characteristics.Characteristics`, read per ship type from the
[ship card](game/ship-card.md)'s own measurements, flown in the game's code at the
standard field index, so they follow the card
(`bnb.game.handling.characteristics.measure_characteristics`):

- `cruise_speed` and `gun_range`: the card's handles (`bnb.game.handling.measure`);
- `top_speed`: the peak of a straight boost from cruise with a full store;
- `corner_speed`: the game's own `bnb.game.physics.flight.Airframe` value,
  `sqrt(max_acceleration / max_lift_coefficient)`;
- `turn_rate`: the coast datum's settled flight-path rate at each of the three stick
  levels;
- `bullet_speed`: cruise plus muzzle speed, the launch speed the gun range assumes;
- the card's values (`max_energy`, `bullet_energy_cost`, `max_shield`, `hit_radius`,
  `max_acceleration`) and the per-type scaled rules (`stall_angle`, `bullet_drag`,
  `vision_range`).

`bnb.game.handling.characteristics.time_of_flight` gives a bullet's time to a distance from
the drag law, `t = (exp(k s) − 1) / (k w₀)`. `bnb.game.handling.characteristics.characteristics`
caches the measurement per card and rules; it takes about 12 s on the CPU.

`tests/sections/test_handling.py` checks the quantities against independent runs of the
game (a bullet fired with `bnb.game.api.move`, the reference flight maneuvers, a settled
turn flown through the game's move) and that a changed card moves them the right way.

## The combat layer

Each own ship fights its target, the nearest visible living enemy (ties: lowest row).

- **The lead:** the aim point is the target's displacement when a bullet fired now
  arrives, `(p_target + v_target t) − (p_own + v_own t)` with `t` the time of flight to the
  target's distance (`bnb.controllers.scripted_combat.lead`). The distance is capped at
  `lead_cap_in_gun_range` gun ranges, so a far target does not get a lead of seconds.
- **The stick:** the angle error from the nose to the wanted direction, positive to the
  left, sets P(turn) by one ramp and the level by two more (level two over one, three over
  two), all in stall angles. The side follows the sign.
- **Power:** boost when slow (in corner speeds) or when the store fraction times the
  distance in gun ranges is large; reverse when close; coast otherwise.
- **Shoot:** P(aligned), by the nose's angle to the aim point over the target's angular
  size `2 atan(hit_radius / distance)` (clamped to 1–45°), times P(in range), by the
  distance in gun ranges. It is 0 while the gun cools or the store cannot pay for a shot:
  such a shot does nothing, and a 0 keeps the teacher's labels honest about it.
- **The boundary:** a ship within a margin of g-limited turn radii (`speed² /
  max_acceleration`) of the arena's edge and flying outward blends a steer toward the
  centre into its wanted direction, by a ramp on its depth into the margin.
- **No visible enemy:** it heads for the centre and holds its course once within
  `centre_hold_in_arena_radius` of it, and does not fire.

## The Frontline layer

Strength is a sum of Gaussian kernels over visible living ships, each weighted by its
health `0.2 + 0.8 · shield / max_shield`.

- **Who defends:** the own defense zone wants the larger of a floor and the enemy's
  strength near it plus a margin, and more while its meter moves against the viewer: the
  urgency times one plus the meter's progress, so the nearer the enemy's capture, the more
  ships it calls back.
  Ships nearer the defense zone than the attack zone are assigned to it first (ties by
  row), until
  their strength covers the want; the rest attack the enemy's defense zone, unless their
  strength times `exp(aggression)` is below the enemy's there, when everyone defends.
  Every ship of a view computes the same assignment, so the team splits without memory.
- **Separation** from allies within a radius, stronger the nearer.
- **Recovery:** a ship below a shield fraction with enemy strength above a threshold
  at its position falls back toward its own spawn zone. Recovery overrides power and
  stick.
- **The blend with combat:** the two layers' power and stick marginals are blended in log
  probability by the nearest enemy's distance, all combat at the inner shooting range and
  all strategy at the combat radius. The shot is the combat layer's. The boundary term
  applies to every wanted direction.

## The ladder's rungs

`Scripted(scripted_probability=p)` takes a float or a tensor `(R,)` with one probability
per view: each ship plays the scripted action with probability `p` and a uniform one
otherwise, from one scripted pass and one uniform draw over all views. Its distribution,
`Scripted.action_distribution`, is `p q + (1 − p) / 42`; 0 is random play and 1 the
scripted agent. The league changes a tensor in place, and the captured graph reads it.

## Parameters

`bnb.controllers.scripted_parameters.ScriptedParameters` holds the dimensionless numbers,
each named for the quantity it scales, with the ramps as
`bnb.controllers.scripted_parameters.Ramp`. Its defaults are the values the parameter
search chose. A value may also be a tensor `(R, 1)` with one value per view
(`bnb.controllers.scripted_parameters.unpack` of a parameter table), so one compiled
controller plays many parameter sets side by side; the search used that.

The search is `bench/scripted_search.py` (`uv run --no-sync python -m bench.scripted_search`).
Each round plays 32 candidates, the best so far and log-normal perturbations of it,
against the chaser and against its starting values
(`bench.scripted_search.STARTING_PARAMETERS`) in the team deathmatch and Frontline
presets, and keeps the best mean score. It seeds round `r`'s games with
`bench.scripted_search.game_seed`, below `bench.scripted_search.SEED_LIMIT`; the section
notes (`docs/rewrite/sections/scripted-bc-notes.md`) record the run that chose the
defaults.

## Measurements

`bnb.runtime.evaluate.first_results` plays fresh games to their end in the training
rollout and returns each view's result. Measured on 2026-11-02 with the default parameters
on the development laptop (RTX 4070 Laptop GPU), 5v5 at the training presets, fog on in
every game, the seat drawn per game, on the acceptance tests' seeds (the starting
parameters' column on seeds of its own, also outside the search's). A score is wins plus
half the draws.

| Preset | Against random (2048 games) | Against the chaser (1024 games) | Against the starting parameters (1024 games) |
|---|---|---|---|
| `deathmatch` | 2048 wins | 0.999 (one loss) | 0.893 |
| `frontline` | 2048 wins | 0.991 (1.9% draws) | 0.934 |
| `deathmatch_wide` | 2048 wins | 1.000 (one draw) | 0.867 |

The starting parameters are `bench.scripted_search.STARTING_PARAMETERS`.
`tests/sections/test_scripted_acceptance.py` plays the first two columns on these seeds, at or
above the search's `SEED_LIMIT`: at most 2 non-wins in 2048 games against random, and above
0.5 against the chaser, at every preset.

Its cost inside the compiled rollout, at the 5v5 training mix and 4096 environments
(bench's ship budget), one team's views scripted against random play: 0.059 ms per tick,
5.6% of the 1.05 ms tick with random play on both sides (same laptop, a GPU shared with
another process, so provisional).

## Tests

`tests/sections/test_scripted.py` covers the behaviour on constructed observations for
both ship types (the stick, the gun, power, the lead with a still and a moving shooter, the
boundary heading out and heading in, no target; in Frontline the defense assignment with
its urgency, progress and tie-break, the hand-over to combat by distance, and recovery),
the invariances (the seat, a mirrored world, permuted rows, the
values of hidden rows' zeroed columns), the rungs and per-view parameters.
