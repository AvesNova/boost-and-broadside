---
covers: [src/bnb/controllers/scripted.py, src/bnb/controllers/scripted_parameters.py, src/bnb/controllers/scripted_defaults.py, src/bnb/controllers/scripted_memory.py, src/bnb/controllers/scripted_combat.py, src/bnb/controllers/scripted_doctrine.py, src/bnb/controllers/scripted_frontline.py, src/bnb/game/handling/characteristics.py, src/bnb/runtime/evaluate.py]
reviewed: 2026-11-03 09:45
---

# The scripted agent

<!-- generated: section-status -->

The scripted agent is one [controller](glossary.md#controller),
`bnb.controllers.scripted.Scripted`, with three uses: a league opponent, the rungs of the
stationary ladder (the agent at higher temperatures, and mixed with random play), and the
teacher for [behaviour cloning](behaviour-cloning.md). Its carry is its
[memory](#memory-soft-hysteresis-and-dead-reckoning): the plan choices each ship has
committed to and what it last saw of each enemy. It plays with one parameter set per game
mode ([Parameters](#parameters)).

## What it reads and gives

It reads only the observation, the parts `Scripted.reads` names: the ship rows, the zone
rows and the globals. It never reads `bnb.contracts.Privileged` or the game state, so it
sees exactly what a model sees, fog included, and a student can match it.

`bnb.controllers.scripted.decide` is the function behind it: a team's views and the carry
in, the marginals and the carry after the decision out. `bnb.controllers.scripted.distribution`
gives the `(R, S, 42)` distribution over the joint action given a carry (`None`: a fresh
one, the memoryless agent), in `bnb.contracts.action.to_joint` order. The joint is the outer product of three marginals, power (3), stick (7) and shoot
(2), which `bnb.controllers.scripted.marginals` returns. `Scripted.step` samples each
marginal by its inverse CDF from the default generator, so a captured graph draws fresh
numbers on every replay. A dead ship, or one waiting to respawn, takes the no-op action
with probability 1; a living ship never has an action at probability 0 or 1 (see
[Softness](#softness-logistics-the-floor-and-the-temperature)).

The global mode one-hot selects the layers: team deathmatch and elimination run the combat
layer; Frontline runs the Frontline layer on top of it. Both are computed for every view and the
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

Each own ship fights its committed target ([Memory](#memory-soft-hysteresis-and-dead-reckoning)):
a fresh ship takes the nearest known enemy by its distance over its weight (ties: lowest
row), a visible one weighing 1. A dead-reckoned target is aimed at where it is predicted,
with its last seen velocity.

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
  distance in gun ranges, times the target's weight. Without a target, while the gun cools
  or while the store cannot pay for a shot it is 0: such a shot does nothing, or only
  spends energy. A dead-reckoned target may be shot at, less the staler it is: the weight
  is the same fade that discounts it as a target, so a target just lost behind a core is
  still worth a shot and one predicted from seconds ago is not; and beyond vision range
  (1024 m against a gun range of about 210 m) the in-range logistic already gives a hidden
  target almost no shots.
- **The boundary:** a ship within a margin of g-limited turn radii (`speed² /
  max_acceleration`) of the arena's edge and flying outward blends a steer toward the
  centre into its wanted direction, by a linear ramp (a weight, not a probability) on its
  depth into the margin.
- **No known enemy:** it heads for the centre and holds its course once within
  `centre_hold_in_arena_radius` of it (an angle error of 0), and does not fire.

## Doctrines

How a ship fights can depend on its type: `bnb.controllers.scripted_doctrine.Doctrine` is
one type's doctrine, and the agent reads a table of them (`doctrines`, one row per type, or
per view and type) and flies each own ship by its own type's row, so a mixed team flies
each type its own way. Without a table (the default) no ship has a doctrine. The tuned
doctrines of the real-scale set's Yongma (the energy fighter) and Jujak (the turn fighter)
are in `src/bnb/controllers/scripted_doctrines.json`
(`bnb.controllers.scripted_doctrine.set_table` builds a set's table from it); the tuning is
in [the archetypes page](game/ship-design/archetypes.md#5-the-scripted-agent).

A doctrine changes the aim point and the power, never the shot (a shot still needs the
nose on the lead point). Its weight against the committed target is `strength` times
`in_advantage`, a logistic in the product of the own-over-target ratios of the sustained
turn rate, the climb power, the cruise speed and the gun range, each raised to its
exponent, so it can act against one opponent type and not another. Its elements, each with
a weight of its own:

- **Extend**: within the fight's range (`fight_in_gun_range`), rather than turn after a
  target its nose is far from, or engage without an energy advantage (own over target
  specific energy, speed²/2 plus the store, of a seen target; and a store floor), fly
  away from the target, boosting; come back from beyond the fight's range.
- **Slash**: boost into the attack within `slash_in_gun_range`, and never reverse.
- **Bleed**: within the fight's range, the nose far off the target and faster than the
  corner speed, reverse to slow toward it, where the turn is fastest.
- **Lag**: aim behind the target along its track, the more the farther the ship is from
  its tail (none dead astern), to come round onto it.
- **Deny**: when the target's nose is on the ship within `deny_in_gun_range`, turn
  `deny_angle` off the line to it, to the side the ship is already off its nose, so the
  pass is an offset one, not a mutual head-on.

The maneuvers act on the stick and the power after the combat layer has chosen them. Each
has its own weight and gates, not `strength` and `in_advantage`, so they also fly in the
mirrors (where the advantage is 1 and the tuned doctrines' gates are shut). Each is off
by default:

- **Lead turn**: on a head-on approach (the flight paths within `lead_turn_in_crossing` of
  opposed) with the merge `lead_turn_in_seconds` away (range over closure), turn early at
  `lead_turn_stick` of full stick toward the side the target will pass on (the two-circle
  sense), or away from it (`lead_turn_one_circle`). The falling `lead_turn_in_crowd` on the
  sense's crowd count (below) can keep it to a lone duel, and the falling
  `lead_turn_in_others` on the count of the other known enemies at any range to the last one.
- **Yo-yo**: after `yoyo_after_seconds` of a close fight with no angle gained (a clock in
  the carry), with the target within `yoyo_phase` of the nose and its nose off the ship,
  ease the stick to `yoyo_ease` and climb (reverse) for `yoyo_climb_seconds`, then pull
  and dive (boost) for `yoyo_dive_seconds`. The low yo-yo (`yoyo_low`) dives first, up to
  `yoyo_to_corner_speed` corner speeds.
- **Brake**: with a closing target in the rear hemisphere, its nose on the ship, inside
  `brake_in_gun_range`, reverse (the store as a speed brake) to force an overshoot.
- **Sense** (the post-merge choice): with the target passed behind the 3-9 line within
  `sense_in_gun_range` and turning, turn at full stick against its rotation (the
  one-circle) with probability `sense_one_circle`, drawn once per merge, else with it (the
  two-circle). It is gated by the lift-coefficient ratio (`sense_in_lift_ratio`; the slow
  turn's radius is one over the maximum lift coefficient) and, optionally, to the own type
  (`sense_mirror_only`). With another known enemy within `sense_crowd_in_gun_range` (the
  falling `sense_in_crowd` on their soft count), it hands over to the **crowd sense**: the
  same side at `crowd_sense_stick` of full stick (0 flies straight, an extension),
  boosting with weight `crowd_sense_boost`.
  The crowd can count only threats (`sense_crowd_in_nose_angle`: an other enemy whose track
  is within that angle of the line to the ship) and less the own ships near
  (`sense_crowd_ally_weight`), and the crowd extension can give way to the sense's turn
  above an energy or shield ratio (`crowd_sense_in_energy_ratio`,
  `crowd_sense_in_shield_ratio`).
- **Duel extension** (`duel_extend`): alone with the target after the pass, extend (the
  crowd sense's stick and boost) instead of the sense's turn when the own shield, specific
  energy or maximum lift coefficient over the target's is below its gate
  (`duel_extend_in_shield_ratio`, `_in_energy_ratio`, `_in_lift_ratio`): a hurt ship
  leaves the turning fight. `duel_extend_mirror_only` keeps it to the own type;
  `duel_extend_cross_in_others` lets it act against another type only with another known
  enemy anywhere (in a team game, not a lone duel).
- **Hold fire** (`hold_fire`): on a closing approach with the target's nose on the ship
  inside `hold_fire_in_gun_range`, withhold the shot (the only element on the shot).

What each one did against r9, and why the yo-yo does not work in this flight model, is in
`docs/rewrite/sections/maneuvers/progress.md`; the situational post-merge choice, the
duel extension and hold fire against r9m are in
`docs/rewrite/sections/maneuvers-2/progress.md`.

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
  Each rule gives a margin in ship strengths (the want less the strength ahead; the
  enemy's strength there times `exp(−aggression)` less the attackers'), and the ship's
  committed goal switches on the larger at `goal_switch_in_strength`'s rate.
- **Separation** from allies within a radius, stronger the nearer.
- **Recovery:** a ship below a shield fraction with enemy strength above a threshold
  at its position falls back toward its own spawn zone. Recovery overrides power and
  stick. Its margin is the smaller of the two log ratios, and the committed choice
  switches at `recovery_switch_in_log_ratio`'s rate.
- **The blend with combat:** the two layers' power and stick marginals are blended in log
  probability by the target's distance, by a linear ramp (a weight) times the target's
  weight: all combat at the shoot logistic's centre and all strategy at the combat radius.
  The blend is already smooth, so it takes no hysteresis. The shot is the combat layer's.
  The boundary term applies to every wanted direction.

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
centre and the Frontline layer's hand-over to combat. The plan choices (the target, the
turn side, the zone assignment, recovery) are committed options that switch at a smooth
rate ([Memory](#memory-soft-hysteresis-and-dead-reckoning)); they take no floor beyond it.

**The temperature** T multiplies every logistic's and switch's scale: T = 1 is the agent, a larger T
flattens every logistic toward an even choice, and the floors stay. `Scripted(temperature=T)`
takes a float or a tensor `(R,)` with one temperature per view; `+inf` plays uniformly over
the 42 joint actions (random play). The league changes a tensor in place, and the captured
graph reads it. The ladder's rungs are the agent at temperatures, and BC's teacher is the
agent at its own temperature (`bnb.learn.bc.TeacherConfig`, T = 1 by default).

**The share** (`Scripted(share=p)`, a float or a tensor `(R,)`) makes a mixture rung: at
each episode reset each ship draws whether it plays the agent (probability p) or uniformly
for the whole game. The temperature alone never reaches random play (the agent at T = 32
still beats it in 383.5 of 384 games), so the ladder's bottom rungs are mixtures.

**Smoothness:** each logistic's scale has a floor
(`bnb.controllers.scripted_parameters.SCALE_FLOORS`), 0.1 times the standard deviation of
the logarithm of its input over the earlier agent's games, the smallest over the three
training presets; the search cannot go below it, so no logistic can act as a step.

## Memory: soft hysteresis and dead reckoning

The carry (`bnb.controllers.scripted_memory.ScriptedCarry`) holds, per view, each own
ship's committed target, turn side (one per layer), Frontline goal and recovery, and a
`fresh` flag; each enemy's last seen position and velocity and whether it is known; and
each ship's agent flag (the share's draw).

**A switching rate, not thresholds.** Each decision, a plan choice keeps its committed
option or switches to the best alternative with probability `1 − exp(−rate · dt)`, `rate =
r_max · σ((advantage − centre) / (scale · T))`
(`bnb.controllers.scripted_parameters.Switch`, `switch_probability`), the advantage being
how much better the alternative is in the choice's own unit:

| Choice | Advantage | Parameter |
|---|---|---|
| target | log ratio of the committed target's weighted distance to the best's | `target_switch_in_log_distance` |
| turn side | how much shorter the other way round is, in radians | `side_switch_in_radians` |
| Frontline goal | the other goal's margin, in ship strengths | `goal_switch_in_strength` |
| recovery | the other choice's log margin | `recovery_switch_in_log_ratio` |

Near-equal options switch at `r_max · σ(−centre / scale)`, almost never; a clearly better
one within a fraction of a second. A fresh ship (after an episode reset, its death or its
spawn) re-chooses each outright, the memoryless choice; so does a ship whose target died or
was forgotten. The turn side is committed only for large errors: the committed side
outweighs the error's sign by `side_commit_in_stall_angle`, a logistic in the error, so a
near-zero error follows its sign and an error near ±180° holds its side.

**Dead reckoning.** A hidden enemy is predicted at its last seen position plus its last seen
velocity times the time since seen (the observation's `time_since_seen`), held inside the
arena. Its weight as a target fades by `memory_fade_in_seconds`, a falling logistic in that
time (its centre searched within 0.5–2 s); the target's distance is divided by it. It is
forgotten when it dies, when its weight falls below
`bnb.controllers.scripted_memory.FORGET_WEIGHT` (1%), or when its prediction falls inside
an own living ship's vision range (it would be seen there, so the prediction was wrong).

**Memory off.** `Scripted(memory=False)` makes every ship fresh and forgets every hidden
enemy at every decision: the memoryless agent, through the same code. Its distribution is
the round-one agent's exactly.

**Labels.** BC's teacher runs its carry over the stored segments before each update, in
time order and on from the last update's (`bnb.learn.bc.teacher_carries`), keeping the
carry after every tick; a label is the teacher's distribution given its carry there
(`decide` without switching re-reads a carry and gives the same distribution).

## Parameters

`bnb.controllers.scripted_parameters.ScriptedParameters` holds the dimensionless numbers,
each named for the quantity it scales, with the logistics as `Logistic`, the weights as
`Ramp` and the switches as `Switch`. Its defaults are the starting point of the searches
(round one's choice, with the memory's starting values).

**One set per game mode.** The agent's defaults are
`bnb.controllers.scripted_defaults.MODE_DEFAULTS`, a
`bnb.controllers.scripted_parameters.ModeParameters` with a set for each of the training
presets and the real-scale ones (`bnb.controllers.scripted_parameters.MODES`:
deathmatch, deathmatch_wide, Frontline, real_scale_deathmatch, real_scale_frontline,
real_scale_elimination), each from a search of its own preset but elimination's, a copy of
real_scale_deathmatch's set without a search (2026-11-12). Each view plays its mode's set, chosen from its
observation and the card (`bnb.controllers.scripted.mode_index`): on a real-scale card (its
gravity below `REAL_SCALE_GRAVITY`, 16.3 m/s^2, the geometric mean of 27 and 9.81) the
real-scale Frontline or elimination set by the mode's flag, else the real-scale deathmatch
set; otherwise Frontline by the mode's flag (a standard card plays elimination with its
deathmatch sets), else the wide set when the arena radius is more than
`WIDE_ARENA_IN_GUN_RANGES` of the card's longest gun range (20.9: the geometric mean of the
two deathmatch presets' 2600 and 8200 m on the standard card, 11.8 and 37.1 of its 220.8 m
gun range), else deathmatch's. The real-scale sets are a light retune
([The real-scale variant](game/real-scale.md#the-scripted-agents-real-scale-sets)). So one
batch may mix the modes, and every use
plays the mode's set: BC's teacher, the lanes' scripted anchor, the ladder's rungs and the
head-to-head's `scripted`. Each search's final population (its last generation's 32 sets,
best first) is kept in `src/bnb/controllers/scripted_populations.json`
(`bnb.controllers.scripted_defaults.populations`) for later use; nothing samples from it. A
value may also be a tensor `(R, 1)` with one value per view
(`bnb.controllers.scripted_parameters.unpack` of a parameter table), so one compiled
controller plays many parameter sets side by side; the search uses that.

The search is `bench/scripted_search.py` (`uv run --no-sync python -m bench.scripted_search`),
a population round robin. Each generation plays 32 candidates, one parameter-table row per
view, in one compiled rollout per preset (`--preset` searches one preset alone, as each of
the defaults' searches did): each candidate's games are mostly against other candidates
drawn at random, and a quarter against the reference players (by default the scripted
agent at its defaults and the chaser; the section notes name each run's). Each candidate seat's temperature is drawn per game from {1, 1.5, 2, 3}, so the
ranking rewards strength across temperatures. A Bradley–Terry fit over the generation's
games (`bnb.league.ratings.fit_ratings`, the defaults anchored) ranks them; the top 8 carry
over and replay, and children of them (crossover, then a log-normal step that shrinks per
generation) fill the rest. It stops when the top's rating plateaus, and it resumes from its
JSON-lines log. It seeds generation `g`'s games with `bench.scripted_search.game_seed`, below
`bench.scripted_search.SEED_LIMIT`; the section notes
(`docs/rewrite/sections/scripted-memory-notes.md`) record the runs that chose the
defaults.

## Measurements

Measured on 2026-11-05 on the development laptop (RTX 4070 Laptop GPU), 5v5 at the three
training presets, fog on, the seat drawn per game, 128 games per preset, the agent with its
per-mode defaults; the section notes (`docs/rewrite/sections/scripted-memory-notes.md`)
have the per-preset split, the searches and the softness audit.

| The agent at | Against | Score | frontline | deathmatch | deathmatch_wide |
|---|---|---|---|---|---|
| T = 1 | round one's (memoryless) agent | 0.454 ± 0.024 | 0.539 | 0.363 | 0.461 |
| T = 1 | the earlier (linear-ramp) agent | 0.464 ± 0.024 | 0.594 | 0.312 | 0.484 |
| T = 1 | itself with memory off, T = 1 | 0.374 ± 0.023 | 0.387 | 0.375 | 0.359 |
| T = 1.5, 2, 3, 4 | itself at T = 1 | 0.522, 0.385, 0.207, 0.074 | | | |

The memory costs strength: the agent loses to its own parameters played without it in
every mode, and no single part of the memory carries the cost (the notes). The ladder's
mixtures (T = 2, shares 0.1, 0.3, 0.5) score 0.648, 0.897 and 0.980 against random, each
0.71 to 0.78 against the one below, and T = 8 scores 0.693 against the share 0.5.

Its joint entropy at T = 1 is 0.85 to 0.88 nats per living ship step (1.11 to 1.12 at
T = 2, 1.50 to 1.54 at T = 4). `tests/sections/test_scripted_acceptance.py` plays it
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
`tests/sections/test_scripted_memory.py` covers the memory: the switching rate, the
target's hysteresis and a forced re-choice, dead reckoning, its fade and its drop inside
the own view, the resets, the turn side's commitment, memory off as the memoryless agent,
the mixture's agent flags, and the per-mode parameters.
`tests/sections/test_scripted_search.py` covers the search's parts on the CPU.
