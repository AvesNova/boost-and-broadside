---
covers: [src/bnb/controllers/imugi.py, src/bnb/controllers/scripted.py, src/bnb/controllers/imugi_parameters.py, src/bnb/controllers/imugi_defaults.py, src/bnb/controllers/imugi_memory.py, src/bnb/controllers/imugi_combat.py, src/bnb/controllers/imugi_doctrine.py, src/bnb/controllers/imugi_frontline.py, src/bnb/controllers/cheese.py, src/bnb/game/handling/characteristics.py, src/bnb/runtime/evaluate.py]
reviewed: 2026-11-03 09:45
---

# Imugi

<!-- generated: section-status -->

Imugi (이무기), the scripted agent, is one [controller](glossary.md#controller),
`bnb.controllers.imugi.Imugi`, with three uses: a league opponent, the rungs of the
stationary ladder (the agent at higher temperatures, and mixed with random play), and the
teacher for [behaviour cloning](behaviour-cloning.md). Its carry is its
[memory](#memory-soft-hysteresis-and-dead-reckoning): the plan choices each ship has
committed to and what it last saw of each enemy. It plays with one parameter set per game
mode ([Parameters](#parameters)).

## What it reads and gives

It reads only the observation, the parts `Imugi.reads` names: the ship rows, the zone
rows and the globals. It never reads `bnb.contracts.Privileged` or the game state, so it
sees exactly what a model sees, fog included, and a student can match it.

`bnb.controllers.imugi.decide` is the function behind it: a team's views and the carry
in, the marginals and the carry after the decision out. `bnb.controllers.imugi.distribution`
gives the `(R, S, 42)` distribution over the joint action given a carry (`None`: a fresh
one, the memoryless agent), in `bnb.contracts.action.to_joint` order. The joint is the outer product of three marginals, power (3), stick (7) and shoot
(2), which `bnb.controllers.imugi.marginals` returns. `Imugi.step` samples each
marginal by its inverse CDF from the decision key's uniforms (`bnb.streams.action_uniforms`,
the same every controller reads), and its plan switches, its sense choice and a mixture
rung's agent flags from the same key under their own names
([Random draws](architecture.md#random-draws)). A dead ship, or one waiting to respawn, takes the no-op action
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
  target's distance (`bnb.controllers.imugi_combat.lead`). The distance is capped at
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
  depth into the margin. Where the boundary closes (elimination), the steer works in the
  edge's own frame (2026-11-15): the edge's speed v_b comes from the last two
  observations, "outward" is outward relative to the edge, and the margin grows by
  f(β) = √(1 − β²) + β·arccos(−β), β = v_b / speed, the deepest a ship turning back gets
  relative to a straight edge closing steadily (1 for a still edge, π when the edge is as
  fast as the ship). Where the boundary is only 2–4 turn radii across, the steer fades out
  and the ship fights.
- **No known enemy (the explore search, 2026-11-15):** the team keeps a hex grid of how
  recently it has seen each cell (8 rings over the arena, about 1 km cells at real scale;
  1 in view, fading over 20 s). It heads for the nearest unseen cell inside the boundary,
  preferring cells ahead of its motion, each ship pulled toward the team's centre of mass
  beyond 2.5–4.5 gun ranges and pushed off allies within 1.6. It does not fire. The old
  rule (head for the centre, hold near it) stays as `imugi_combat.SEARCH = "centre"`; it
  made both teams meet in the middle by agreement.

## Doctrines

How a ship fights can depend on its type: `bnb.controllers.imugi_doctrine.Doctrine` is
one type's doctrine, and the agent reads a table of them (`doctrines`, one row per type, or
per view and type) and flies each own ship by its own type's row, so a mixed team flies
each type its own way. Without a table (the default) no ship has a doctrine. The tuned
doctrines of the real-scale set's Yongma (the energy fighter) and Jujak (the turn fighter)
are in `src/bnb/controllers/imugi_doctrines.json`
(`bnb.controllers.imugi_doctrine.set_table` builds a set's table from it); the tuning is
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
  sense), or away from it (`lead_turn_one_circle`).
- **Yo-yo**: after `yoyo_after_seconds` of a close fight with no angle gained (a clock in
  the carry), with the target within `yoyo_phase` of the nose and its nose off the ship,
  ease the stick to `yoyo_ease` and climb (reverse) for `yoyo_climb_seconds`, then pull
  and dive (boost) for `yoyo_dive_seconds`. The low yo-yo (`yoyo_low`) dives first, up to
  `yoyo_to_corner_speed` corner speeds.
- **Brake**: with a closing target in the rear hemisphere, its nose on the ship, inside
  `brake_in_gun_range`, reverse (the store as a speed brake) to force an overshoot.
- **Sense** (the post-merge choice): with the target passed behind the 3-9 line within
  `sense_in_gun_range` and turning, turn at full stick, choosing one circle or two from
  the card and the state now: with the target's rotation (the two-circle) where the own
  turn rate at the own speed and store is the higher, else against it (the one-circle)
  where the own turn radius is the smaller, else with it. The choice is the same in every
  matchup and is not drawn. It is **committed** once per turning fight
  (`imugi_combat.turn_commitment`, `BNB_TURN_COMMIT`, on): the ship that out-turns or
  out-energies its target takes the **initiative**, turning on a head-on approach at the
  last tick that has its turn at full rate at the crossing (its turn onset: 3.3 nose
  periods, about 0.3 s at the corner speed, the nose still inside the head-on armour's
  angle), toward the side of the target's tail; the other answers the observed sense once
  at the merge; an even pair turns by the geometry (a dead-on merge toward the map centre,
  then left). The sense then holds until the next merge (after 2.5 s at least), the
  target's loss or change, an extension or the edge steer. **Stick by energy**
  (`BNB_STICK_BY_ENERGY`, on): a two-circle turn below the corner speed eases to the
  sustained stick level, the turn that holds speed. With another known enemy within `sense_crowd_in_gun_range` (the
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
  (`duel_extend_in_shield_ratio`, `duel_extend_in_energy_ratio`, `duel_extend_in_lift_ratio`): a hurt ship
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
- **The blend with combat:** the two layers' power and stick marginals are blended in log
  probability by the target's distance, by a linear ramp (a weight) times the target's
  weight: all combat at the shoot logistic's centre and all strategy at the combat radius.
  The blend is already smooth, so it takes no hysteresis. The shot is the combat layer's.
  The boundary term applies to every wanted direction.
- **Frontline's timers** (game version 8): the context both layers read (`Context` in
  `bnb.controllers.imugi_combat`) carries the own team's next respawn wave
  (`next_wave`) and each zone's capture lock on the own team and on the enemy's
  (`zone_lock_own`, `zone_lock_enemy`), in seconds, from the observation. No rule reads
  them yet, so the agent plays as before. A doctrine that uses them (do not attack a zone
  the own team is locked out of; delay a push until the wave lands) is a proposal for the
  author.

## Softness: logistics, the floor and the temperature

Every probability the agent reads from a quantity is a logistic in the quantity's
logarithm (`bnb.controllers.imugi_parameters.Logistic`):
`σ(±(ln x − ln centre) / (scale · T))`, the Hill function `x^k / (x^k + centre^k)` with
`k = 1 / (scale · T)`. The centre is in the quantity's unit, the scale in natural-log units
(at T = 1 the 10–90% transition spans a factor `exp(4.39 · scale)` in the quantity), and the
direction (rising or falling) is fixed by the field. The search moves the centres and
scales.

A logistic's argument is held within ±9.2
(`bnb.controllers.imugi_parameters.LOGIT_LIMIT`), so it never rounds to exactly 0 or 1.
Every binary decision is then floored (`bnb.controllers.imugi_parameters.floored`):
`p ↦ ε + (1 − 2ε) p`, with each decision's ε fixed in
`bnb.controllers.imugi_parameters.EPSILONS`: 0.02 for P(turn), P(level ≥ 2 | turn),
P(level 3 | level ≥ 2) and the side; 0.04 for P(reverse) and P(boost | not reverse); 0 for
P(shoot). A per-tick floor on the shot fires an energy-costly shot at 30 decisions a
second, and even 0.001 cost measurable strength (the section notes give the ablation); the
shot's softness is its two logistics'. A decision built from two logistics (boost when slow
or far; shoot when aligned and in range) is floored once, after they combine. So at a
finite temperature no power or stick value of a living ship has probability 0 or 1, and
the shot is certain only at its gates (no target, or a gun that cannot fire).

Two weights are not probabilities and stay linear `Ramp`s: the boundary's steer toward the
centre and the Frontline layer's hand-over to combat. The plan choices (the turn side
and the zone assignment) are committed options that switch at a smooth rate
([Memory](#memory-soft-hysteresis-and-dead-reckoning)); they take no floor beyond it. The
target is held by a deterministic rule instead.

**The temperature** T multiplies every logistic's and switch's scale: T = 1 is the agent, a larger T
flattens every logistic toward an even choice, and the floors stay. `Imugi(temperature=T)`
takes a float or a tensor `(R,)` with one temperature per view; `+inf` plays uniformly over
the 42 joint actions (random play). The league changes a tensor in place, and the captured
graph reads it. The ladder's rungs are the agent at temperatures, and BC's teacher is the
agent at its own temperature (`bnb.learn.bc.TeacherConfig`, T = 1 by default).

**The share** (`Imugi(share=p)`, a float or a tensor `(R,)`) makes a mixture rung: at
each episode reset each ship draws whether it plays the agent (probability p) or uniformly
for the whole game. The temperature alone never reaches random play (the agent at T = 32
still beats it in 383.5 of 384 games), so the ladder's bottom rungs are mixtures.

**Smoothness:** each logistic's scale has a floor
(`bnb.controllers.imugi_parameters.SCALE_FLOORS`), 0.1 times the standard deviation of
the logarithm of its input over the earlier agent's games, the smallest over the three
training presets; the search cannot go below it, so no logistic can act as a step.

## Memory: soft hysteresis and dead reckoning

The carry (`bnb.controllers.imugi_memory.ImugiCarry`) holds, per view, each own
ship's committed target, turn side (one per layer), Frontline goal and a
`fresh` flag; each enemy's last seen position and velocity and whether it is known; and
each ship's agent flag (the share's draw).

**A switching rate, not thresholds.** Each decision, a plan choice keeps its committed
option or switches to the best alternative with probability `1 − exp(−rate · dt)`, `rate =
r_max · σ((advantage − centre) / (scale · T))`
(`bnb.controllers.imugi_parameters.Switch`, `switch_probability`), the advantage being
how much better the alternative is in the choice's own unit:

| Choice | Advantage | Parameter |
|---|---|---|
| turn side | how much shorter the other way round is, in radians | `side_switch_in_radians` |
| Frontline goal | the other goal's margin, in ship strengths | `goal_switch_in_strength` |

Near-equal options switch at `r_max · σ(−centre / scale)`, almost never; a clearly better
one within a fraction of a second.

**The target holds** (since 2026-11-16, `imugi_combat.choose_target`; deterministic, not
a switching rate). A ship keeps its committed target until the target dies, has been
unseen for `TARGET_LOST_SECONDS` (3 s; a hidden committed target stays known, dead
reckoned, that long), or for `TARGET_DOUBT_SECONDS` (2 s) without a break has been either
beyond `TARGET_FAR_IN_GUN_RANGES` (3) own gun ranges or `TARGET_SWITCH_RATIO` (2) times
farther than the best known enemy (distance over weight); then it takes the best. So a
fight is pressed through the merge and the turn back, and two ships do not both re-score
to another fight on the same tick.

**One extension** (`imugi_combat.extend_and_return`, the switch `EXTEND_AND_RETURN`).
After a pass (the target behind the beam inside the own gun range, the two opening), a
ship whose sustained turn rate w is below its seen target's and whose energy (kinetic plus
store) is above it flies straight away, boosting, until one own turn diameter (2 v_cruise
/ w) or 2 / w + 1 s, then comes straight back at the same target; no other extension for
π / w plus that bound. Otherwise it stays in the turn. The doctrines' own extensions
(`extend`, `duel_extend`, `crowd_sense`) were already 0 in the real-scale doctrines.
A fresh ship (after an episode reset, its death or its
spawn) re-chooses each outright, the memoryless choice; so does a ship whose target died or
was forgotten. The turn side is committed only for large errors: the committed side
outweighs the error's sign by `side_commit_in_stall_angle`, a logistic in the error, so a
near-zero error follows its sign and an error near ±180° holds its side.

**Dead reckoning.** A hidden enemy is predicted at its last seen position plus its last seen
velocity times the time since seen (the observation's `time_since_seen`), held inside the
arena. Its weight as a target fades by `memory_fade_in_seconds`, a falling logistic in that
time (its centre searched within 0.5–2 s); the target's distance is divided by it. It is
forgotten when it dies, when its weight falls below
`bnb.controllers.imugi_memory.FORGET_WEIGHT` (1%), or when its prediction falls inside
an own living ship's vision range (it would be seen there, so the prediction was wrong).

**Memory off.** `Imugi(memory=False)` makes every ship fresh and forgets every hidden
enemy at every decision: the memoryless agent, through the same code. Its distribution is
the round-one agent's exactly.

**Labels.** BC's teacher runs its carry over the stored segments before each update, in
time order and on from the last update's (`bnb.learn.bc.teacher_carries`), keeping the
carry after every tick; a label is the teacher's distribution given its carry there
(`decide` without switching re-reads a carry and gives the same distribution).

## Parameters

`bnb.controllers.imugi_parameters.ImugiParameters` holds the dimensionless numbers,
each named for the quantity it scales, with the logistics as `Logistic`, the weights as
`Ramp` and the switches as `Switch`. Its defaults are the starting point of the searches
(round one's choice, with the memory's starting values).

**One set per game mode.** The agent's defaults are
`bnb.controllers.imugi_defaults.MODE_DEFAULTS`, a
`bnb.controllers.imugi_parameters.ModeParameters` with a set for each of the training
presets and the real-scale ones (`bnb.controllers.imugi_parameters.MODES`:
deathmatch, deathmatch_wide, Frontline, real_scale_deathmatch, real_scale_frontline,
real_scale_elimination), each from a search of its own preset but elimination's, a copy of
real_scale_deathmatch's set without a search (2026-11-12). Each view plays its mode's set, chosen from its
observation and the card (`bnb.controllers.imugi.mode_index`): on a real-scale card (its
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
best first) is kept in `src/bnb/controllers/imugi_populations.json`
(`bnb.controllers.imugi_defaults.populations`) for later use; nothing samples from it. A
value may also be a tensor `(R, 1)` with one value per view
(`bnb.controllers.imugi_parameters.unpack` of a parameter table), so one compiled
controller plays many parameter sets side by side; the search uses that.

The search is `bench/imugi_search.py` (`uv run --no-sync python -m bench.imugi_search`),
a population round robin. Each generation plays 32 candidates, one parameter-table row per
view, in one compiled rollout per preset (`--preset` searches one preset alone, as each of
the defaults' searches did): each candidate's games are mostly against other candidates
drawn at random, and a quarter against the reference players (by default the scripted
agent at its defaults and the chaser; the section notes name each run's). Each candidate seat's temperature is drawn per game from {1, 1.5, 2, 3}, so the
ranking rewards strength across temperatures. A Bradley–Terry fit over the generation's
games (`bnb.league.ratings.fit_ratings`, the defaults anchored) ranks them; the top 8 carry
over and replay, and children of them (crossover, then a log-normal step that shrinks per
generation) fill the rest. It stops when the top's rating plateaus, and it resumes from its
JSON-lines log. It seeds generation `g`'s games with `bench.imugi_search.game_seed`, below
`bench.imugi_search.SEED_LIMIT`; the section notes
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
T = 2, 1.50 to 1.54 at T = 4). `tests/sections/test_imugi_acceptance.py` plays it
against random and the chaser on seeds at or above the search's
`bench.imugi_search.SEED_LIMIT`: at most 2 non-wins in 2048 games against random, and
above 0.5 against the chaser, at every preset.

## Cheese agents

Six crude players, each one plain strategy, give a check that a trained policy cannot be
beaten by a single trick, and are league entries by default. They are one controller,
`bnb.controllers.cheese.Cheese`, a function of the observation with no carry. They
read the ship rows, the zone rows and the globals, use an enemy row only where it is visible
and alive, and draw nothing: the decision key goes unused, and the same observation gives the
same action. The flight is the chaser's bearing-to-stick rule
(`bnb.controllers.scripted.stick_toward`), aiming straight at the target with no lead, and
a shot is the chaser's (inside 8 degrees of the target and 250 m of the standard card).
Every agent turns toward the map centre as it nears the boundary's radius (the last quarter
of it), and when it has no visible enemy to play against (the zone agents: always, outside
Frontline) it flies a circle about the map centre or its zone.

| Player | Behaviour |
|---|---|
| `cheese:zone_rusher` | Ignores combat. Full boost to the enemy's defense zone (the one the team has yet to take: the roles rotate at each capture, so it is always the next), then circles inside it, coasting. Never fires. |
| `cheese:turtle` | Circles the zones its team holds, coasting: even slots its defense zone, odd slots its spawn zone. Engages the nearest visible enemy within 1.5 zone radii of its zone: turns to it, and fires when aligned. |
| `cheese:evader` | Never engages or fires. Turns directly away from its nearest visible enemy, boosting while it is within two gun ranges, and drifts to the map centre as the boundary shrinks. |
| `cheese:deathball` | Holds a formation of half a gun range about the team's centre and focuses fire: every ship turns for the visible enemy nearest the centre at full boost and fires when aligned, but a ship outside the formation returns to the centre first. |
| `cheese:jouster` | Every ship goes at its nearest visible enemy at full boost and fires whenever aligned. No defensive manoeuvre. |
| `cheese:kiter` | Holds 0.9 gun ranges (a band of 0.1 either side) from its nearest visible enemy: closes at full boost beyond the band, turns away at full boost inside it, and aims, coasts and fires within it. |

The zone agents suit Frontline; outside it they circle the map centre in place of a zone
(the turtle then engages enemies near the centre). In Frontline a spawn zone shields a ship
from damage, so the turtle's odd slots are the hardest to dislodge; it is a trick, not an
oversight. The names are player names: they work wherever Imugi does in `bnb
ratings --with` and `bnb head-to-head`.

**The exploit check.** A policy's score against each, per preset (Frontline, elimination
and deathmatch by default), 128 games each split between the seats:

<!-- not-runnable: needs CUDA and a run -->
```bash
uv run --no-sync bnb head-to-head RUN cheese:zone_rusher RUN cheese:turtle \
    RUN cheese:evader RUN cheese:deathball RUN cheese:jouster RUN cheese:kiter \
    --games 128
```

The command prints each pair's wins, losses and draws, its score with its standard error and
the split by preset. A policy that cannot beat a crude player in some mode shows it as a
score near or below one half in that mode's row. The evader and the zone rusher never fire,
so against them a game is decided by the other side's play alone; the results are a floor to
read, not a rating.

**League entries.** All six are league entries by default: `LeagueConfig.cheese` lists them
(`()` turns them off, for an ablation). Each is an entry named `cheese:<name>`, sampled
with the others as any entry is, by its predicted win probability against the live policy
(the existing per-entry rule, with its floor); that is provisional for them, as the weights
across league members are to be decided separately. The slot holding one plays it on its
rows and tallies to the slot's own column. The lanes do not play them, so they have no
rating of their own at first: the gauge starts them at the live policy's rating, and the
league's games against them (`bnb.league.lanes.RunRatings`) rate them from the first update
that has some. The list is fixed at a run's start. `bnb ratings` also plays all six by
default (`--no-cheese` leaves them out), so a published rating table carries the policy's
results against each.

## Tests

`tests/sections/test_cheese.py` covers the cheese agents: valid actions per mode,
that a hidden or dead enemy changes nothing, determinism, each agent's behaviour on
constructed observations, a short game of each in each mode, and their play in the match
engine; `tests/sections/test_league_cheese.py` covers them as league entries.
`tests/sections/test_imugi.py` covers the behaviour on constructed observations for
both ship types (the stick, the gun, power, the lead with a still and a moving shooter, the
boundary heading out and heading in, no target; in Frontline the defense assignment with
its urgency, progress and tie-break, and the hand-over to combat by distance),
the invariances (the seat, a mirrored world, permuted rows, the
values of hidden rows' zeroed columns), the softness (no certain action of a living ship,
the temperature's flattening, the scale floors), the rungs and per-view parameters.
`tests/sections/test_imugi_memory.py` covers the memory: the switching rate, the
target's hysteresis and a forced re-choice, dead reckoning, its fade and its drop inside
the own view, the resets, the turn side's commitment, memory off as the memoryless agent,
the mixture's agent flags, and the per-mode parameters.
`tests/sections/test_imugi_search.py` covers the search's parts on the CPU.
