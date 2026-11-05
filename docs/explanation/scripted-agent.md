---
covers: [src/bnb/controllers/scripted.py, src/bnb/controllers/scripted_parameters.py, src/bnb/controllers/scripted_combat.py, src/bnb/controllers/scripted_frontline.py, src/bnb/game/handling/characteristics.py, src/bnb/runtime/evaluate.py]
reviewed: 2026-11-03 09:45
---

# The scripted agent

<!-- generated: section-status -->

The scripted agent is one memoryless [controller](glossary.md#controller),
`bnb.controllers.scripted.Scripted`, with three uses: a league opponent, the rungs of the
stationary ladder (the agent at higher temperatures), and the teacher for
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
with probability 1; a living ship never has an action at probability 0 or 1 (see
[Softness](#softness-logistics-the-floor-and-the-temperature)).

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
  left, sets P(turn) by one logistic and the level by two more (level two over one, three
  over two), all in stall angles. The side follows the sign, with probability 1 − ε (an
  error of exactly 0 picks either side evenly).
- **Power:** P(boost) is high when slow (in corner speeds) or when the store fraction
  times the distance in gun ranges is large (either, as independent events); P(reverse)
  is high when close; coast otherwise.
- **Shoot:** P(aligned), by the nose's angle to the aim point over the target's angular
  size `2 atan(hit_radius / distance)` (clamped to 1–45°), times P(in range), by the
  distance in gun ranges. Without a target, while the gun cools or while the store cannot
  pay for a shot it is 0: such a shot does nothing, or only spends energy.
- **The boundary:** a ship within a margin of g-limited turn radii (`speed² /
  max_acceleration`) of the arena's edge and flying outward blends a steer toward the
  centre into its wanted direction, by a linear ramp (a weight, not a probability) on its
  depth into the margin.
- **No visible enemy:** it heads for the centre and holds its course once within
  `centre_hold_in_arena_radius` of it (an angle error of 0), and does not fire.

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
  probability by the nearest enemy's distance, by a linear ramp (a weight): all combat at
  the shoot logistic's centre and all strategy at the combat radius. The shot is the combat layer's. The boundary term
  applies to every wanted direction.

## Softness: logistics, the floor and the temperature

Every probability the agent reads from a quantity is a logistic in the quantity's
logarithm (`bnb.controllers.scripted_parameters.Logistic`):
`σ(±(ln x − ln centre) / (scale · T))`, the Hill function `x^k / (x^k + centre^k)` with
`k = 1 / (scale · T)`. The centre is in the quantity's unit, the scale in natural-log units
(at T = 1 the 10–90% transition spans a factor `exp(4.39 · scale)` in the quantity), and the
direction (rising or falling) is fixed by the field. The search moves the centres and
scales.

A logistic's argument is held within ±9.2
(`bnb.controllers.scripted_parameters.LOGIT_LIMIT`), so it never rounds to exactly 0 or 1.
Every binary decision is then floored (`bnb.controllers.scripted_parameters.floored`):
`p ↦ ε + (1 − 2ε) p`, with each decision's ε fixed in
`bnb.controllers.scripted_parameters.EPSILONS`: 0.02 for P(turn), P(level ≥ 2 | turn),
P(level 3 | level ≥ 2) and the side; 0.04 for P(reverse) and P(boost | not reverse); 0 for
P(shoot). A per-tick floor on the shot fires an energy-costly shot at 30 decisions a
second, and even 0.001 cost measurable strength (the section notes give the ablation); the
shot's softness is its two logistics'. A decision built from two logistics (boost when slow
or far; shoot when aligned and in range) is floored once, after they combine. So at a
finite temperature no power or stick value of a living ship has probability 0 or 1, and
the shot is certain only at its gates (no target, or a gun that cannot fire).

Two weights are not probabilities and stay linear `Ramp`s: the boundary's steer toward the
centre and the Frontline layer's hand-over to combat. The goal choices (the target, the
zone assignment, recovery) are hard selections of where to fly; the stick and power that
follow them are soft.

**The temperature** T multiplies every logistic's scale: T = 1 is the agent, a larger T
flattens every logistic toward an even choice, and the floors stay. `Scripted(temperature=T)`
takes a float or a tensor `(R,)` with one temperature per view; `+inf` plays uniformly over
the 42 joint actions (random play). The league changes a tensor in place, and the captured
graph reads it. The ladder's rungs are the agent at temperatures, and BC's teacher is the
agent at its own temperature (`bnb.learn.bc.TeacherConfig`, T = 1 by default).

**Smoothness:** each logistic's scale has a floor
(`bnb.controllers.scripted_parameters.SCALE_FLOORS`), 0.1 times the standard deviation of
the logarithm of its input over the earlier agent's games, the smallest over the three
training presets; the search cannot go below it, so no logistic can act as a step.

## Parameters

`bnb.controllers.scripted_parameters.ScriptedParameters` holds the dimensionless numbers,
each named for the quantity it scales, with the logistics as `Logistic` and the weights as
`Ramp`. Its defaults are the values the parameter search chose. A value may also be a
tensor `(R, 1)` with one value per view (`bnb.controllers.scripted_parameters.unpack` of a
parameter table), so one compiled controller plays many parameter sets side by side; the
search uses that.

The search is `bench/scripted_search.py` (`uv run --no-sync python -m bench.scripted_search`),
a population round robin. Each generation plays 32 candidates, one parameter-table row per
view, in one compiled rollout per preset (team deathmatch, the wide deathmatch and
Frontline): each candidate's games are mostly against other candidates drawn at random, and
a quarter against the reference players (the scripted agent at its defaults, and the
chaser). Each candidate seat's temperature is drawn per game from {1, 1.5, 2, 3}, so the
ranking rewards strength across temperatures. A Bradley–Terry fit over the generation's
games (`bnb.league.ratings.fit_ratings`, the defaults anchored) ranks them; the top 8 carry
over and replay, and children of them (crossover, then a log-normal step that shrinks per
generation) fill the rest. It stops when the top's rating plateaus, and it resumes from its
JSON-lines log. It seeds generation `g`'s games with `bench.scripted_search.game_seed`, below
`bench.scripted_search.SEED_LIMIT`; the section notes
(`docs/rewrite/sections/scripted-redesign-notes.md`) record the run that chose the
defaults.

## Measurements

Measured on 2026-11-05 on the development laptop (RTX 4070 Laptop GPU), 5v5 at the three
training presets, fog on, the seat drawn per game, 128 games per preset; the section notes
(`docs/rewrite/sections/scripted-redesign-notes.md`) have the per-preset split, the search
and the softness audit.

| The agent at | Against | Score |
|---|---|---|
| T = 1 | the earlier (linear-ramp) agent | 0.483 ± 0.025 |
| T = 1 | the earlier BC teacher (that agent, 1% uniform) | 0.630 ± 0.024 |
| T = 1 | the chaser | 1.000 |
| T = 1.5, 2, 3, 4 | itself at T = 1 | 0.477, 0.408, 0.172, 0.034 |
| T = 8, 16 | itself at T = 1 | 0.007 |
| T = 32 | random | 0.999 |
| T = 1, 1% uniform | itself at T = 1 | 0.406 ± 0.024 |

Its joint entropy at T = 1 is 0.84 to 0.92 nats per living ship step (1.06 to 1.18 at
T = 2, 1.40 to 1.57 at T = 4). `tests/sections/test_scripted_acceptance.py` plays it
against random and the chaser on seeds at or above the search's
`bench.scripted_search.SEED_LIMIT`: at most 2 non-wins in 2048 games against random, and
above 0.5 against the chaser, at every preset.

## Tests

`tests/sections/test_scripted.py` covers the behaviour on constructed observations for
both ship types (the stick, the gun, power, the lead with a still and a moving shooter, the
boundary heading out and heading in, no target; in Frontline the defense assignment with
its urgency, progress and tie-break, the hand-over to combat by distance, and recovery),
the invariances (the seat, a mirrored world, permuted rows, the
values of hidden rows' zeroed columns), the softness (no certain action of a living ship,
the temperature's flattening, the scale floors), the rungs and per-view parameters.
`tests/sections/test_scripted_search.py` covers the search's parts on the CPU.
