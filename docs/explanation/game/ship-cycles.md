---
covers: [src/bnb/controllers/scripted_combat.py, src/bnb/controllers/scripted_memory.py, src/bnb/controllers/scripted_parameters.py, docs/explanation/game/real-scale-tournament/]
---

# Ship cycles

The goal is a set of ship types whose head-to-head results are not a ladder: first three
types of equal rating in a rock-paper-scissors cycle, each edge won about 60% of the time
(0.57 to 0.65), then a second and a third cycle built on other mechanisms, and in the end
seven types whose results form the Paley tournament on seven vertices (type i beats
i + 1, i + 2 and i + 4 mod 7). Skill should still decide most games; the type should
matter, not settle them. Every type stays within the envelope of the
[real-scale set](real-scale.md), and only the ship handles and the scripted agent's
parameters may move.

This page covers what the [tournament](real-scale-tournament.md) already says about
cycles, the mechanisms the physics offers, the scripted agent's relative tactics, and the
plan for the first cycle.

## The tournament is cyclic in size, not in direction

By majority only one of the tournament's 2,925 triples is a cycle. That is a statement
about direction. Split each pair's log-odds into a part a rating explains (the
Bradley-Terry fit) and a cyclic remainder (the Hodge decomposition of the pair matrix: a
gradient plus a curl), and the remainder is far larger than its noise: weighted against
the fit, χ²/dof is 6.4, and 324 triangles have a curl more than three standard errors from
zero. The cycles are there; a rating spread of about 2,000 Elo hides them. A triangle's
curl, the sum of its three log-odds, is what is left when its three ratings are made equal,
and each edge then has a third of it. A 0.60 edge needs a curl of 1.22.

The best-measured triangles (every edge between 0.1 and 0.9, 128 games each), with the
edge each would have once balanced:

| Cycle | Curl | Balanced edge | Scores now | Elo now |
|---|---|---|---|---|
| Albatross > Zero > Anvil > Albatross | 2.78 ± 0.36 | 0.72 | 0.78, 0.50, 0.82 | 19, 52, 46 |
| Fw 190D > Yak-3 > Me 163 > Fw 190D | 2.26 ± 0.35 | 0.68 | 0.84, 0.62, 0.53 | 833, 740, 730 |
| Me 262 > Shrike > Spitfire IX > Me 262 | 2.46 ± 0.38 | 0.69 | 0.61, 0.90, 0.47 | 1375, 1348, 1102 |
| Corsair > Spitfire IX > Me 262 > Corsair | 2.19 ± 0.35 | 0.68 | 0.83, 0.47, 0.67 | 1231, 1102, 1375 |
| Mosquito > Zerstörer > P-38L > Mosquito | 1.94 ± 0.34 | 0.66 | 0.67, 0.79, 0.48 | 1459, 1414, 1364 |

The first is the textbook triangle: an energy fighter (the Albatross: a long, low-drag
wing and a large store) beats a turn fighter (the Zero), which draws with a gunship (the
Anvil: one 133-damage cannon and a 160 shield), which beats the energy fighter. Its three
ratings are already within 33 points of each other, and its curl is more than twice what a
0.60 cycle needs: it is a tuning problem, not a search for a mechanism.

Which handles carry the remainder: the pair log-odds were fitted with the ratings plus
antisymmetric bilinear terms in pairs of standardised log handles,
`a(i) b(j) - a(j) b(i)`{.plain} (the form multidimensional Elo takes), scored by 10-fold
cross-validation over the 325 pairs (the Firecracker left out). The ratings alone leave a
weighted mean square of 0.86 in log-odds, against about 0.12 of sampling noise. The best
single terms explain 15 to 24% of it, six terms chosen greedily 37%:

- the guns against the shield: spread x shield (24%), weight of fire x shield (20%),
  damage a hit x shield (19%);
- turn against energy: turn x spread (20%), stall speed x shield (15%), turn x cruise,
  turn x store, stall speed x climb (11 to 13%);
- turn against gun range (14%).

A fit on 26 types with features chosen from 91 pairs is a pointer, not a model: the
families are robust, the coefficients are not.

## Mechanisms

A cycle needs an interaction: if a type's time to kill another were a product of a
property of the shooter and a property of the target, the log-odds would be a difference
of ratings and no cycle could form. Each mechanism below is a way the game makes that
time depend on the pair. The edges are calibrated on the tournament: the pair log-odds
against the log ratio of the two kill times (hits to kill over the fire rate times the
shooter's measured hit share) has a slope of 4.0 (weighted R² 0.34). A 5v5 deathmatch
amplifies small kill-time differences.

### 1. Energy, turn and toughness

**Handles.** The energy fighter: a large store (`ceiling`, toward the Heavy's and the
Hammerhead's 174 kJ/kg), a low zero-lift drag (toward 3.0e-5 /m), a short climb time,
a high cruise; a poor sustained turn (12 to 14 deg/s) and a high stall speed. The turn
fighter: a sustained turn of 21 to 23 deg/s, a stall speed of 32 to 38 m/s, a g limit of 8
to 10, a shield of 80, many light shots (15 Hz). The gunship: a shield of 140 to 160, a hit
that breaks the turn fighter's shield in one (`hit_damage` at or above its `max_shield`), a
low fire rate (2 to 5 Hz), a poor turn.

**Why it cycles.** The turn fighter out-turns the gunship and stays out of its arc, so the
gunship's few shots rarely land, and the light stream it takes kills in time. The gunship
needs two hits on anything (the break and the kill), so the energy fighter's fleeting
passes cost it a ship whenever two land, while the gunship's 160 shield takes the energy
fighter's light guns 10 hits to kill. The energy fighter beats the turn fighter by not
turning with it: it fights in the vertical of the store (dive with boost, zoom with
reverse) and slashes, which the scripted agent does not do yet. In the tournament the
Albatross beats the Zero 0.78 even so; with relative tactics the energy leg should be the
part a searched parameter controls.

**Recipe** (from the Albatross, the Zero and the Anvil, Elo 19, 52, 46): pull the two
strong edges in and push the even one out. Lower the Anvil's damage a hit toward 80 to 95
(three hits on the Albatross's 100 shield instead of two), raise the Zero's turn toward 23
deg/s or its fire rate to 15 Hz, and turn the Albatross's energy fight on against the Zero
only (turn exponent -1). Needs card solves (the flight handles) and the agent change.

### 2. Burst against the shield

**Handles.** The damage a hit, the fire rate (whole Hz dividing 30), the shield, and, as a
second lever, the shield recharge rate and delay. Combat handles only: an instant card
(`bnb.game.handling.weapons.combat_card`), no flight solve.

**Why it cycles.** A kill takes `ceil(S / d) + 1`{.plain} hits (the hit that breaks the
shield, then any hit), so at an equal weight of fire the kill time is
`d (ceil(S / d) + 1)`{.plain} over the damage per second: the overkill on the breaking hit
is wasted, and how much depends on the pair. With a hit's damage averaged over aspect
factors of 0.6 to 1 (hits off the tail, so that no cycle rests on one exact threshold), fire
rates that divide 30 within the set's 2 to 15 Hz, and an equal damage per second, the
largest curl within the envelope is 1.2 to 1.5 in log-odds by the calibration (at 180 to
360 damage per second): a balanced edge of 0.60 to 0.62, at the bottom of the band.

**Recipe** (240 damage per second): a cannon (120 at 2 Hz, shield 80), an armoured gun (80
at 3 Hz, shield 160) and a sprayer (16 at 15 Hz, shield 120). The cyclic part is small
against the gradient: by the same closed form the sprayer kills both others far faster
(log-odds of 2 to 4), and that has to be given back by handles that move all of a type's
edges alike (its spread, its hit radius, or its flight). Alone, this mechanism is marginal;
its natural use is as the guns of mechanism 1's legs (the sprayer is the turn fighter's
gun, the cannon the gunship's), where the flight pays for the gradient. The recharge (12 /s
after 6 s on every type) is a second interaction (a long delay favours a continuous stream,
a short one the burst), but it is a placeholder, the same on every type, so the realism
bound gives it no range; see [Open questions](#open-questions).

### 3. Gun geometry against the turn

**Handles.** The gun range (muzzle speed 520 to 880 m/s and the bullet life), the spread,
the hit radius, the turn.

**Why it cycles.** The fit's turn x range and turn x spread terms: a long, tight gun hits a
turning target from outside its circle, and a type that turns well brings a short gun to
bear. A third leg (the wide, fast ship that closes the range) would make it a triangle.
At real scale the spread is small against the target (1.5 m/s at 856 m/s is 1.8 mrad, a
metre at 600 m against a 5.6 m half-span), so most misses are lead and tracking, not
dispersion: the target-size-against-spread interaction alone is weak, and the gun range,
which a cruise and a muzzle speed set, carries this family.

### 4. Speed and acceleration against reach

The Me 262 and Me 163 triangles: a type far faster than the rest (cruise 228 to 236 m/s,
climb 60 to 81 m/s) chooses when to fight, wins against the types it can catch and loses to
the turners that make its passes short. It shares the speed handles with mechanism 1, so it
is unlikely to be independent of it; it is listed so that a search does not rediscover it
under another name.

### Ranking

| Mechanism | Expected strength (balanced edge) | Independence | Solves | Agent change |
|---|---|---|---|---|
| 1. Energy, turn, toughness | 0.72 measured (Albatross, Zero, Anvil) | shares the guns with 2 | flight: card solves | the energy fight helps its energy leg |
| 2. Burst against the shield | 0.60-0.62 predicted (closed form), behind a large gradient | shares the shield with 1 | combat only: instant | none |
| 3. Gun geometry against the turn | 0.6-0.65 suggested by the fit, unmeasured | shares the turn with 1 | combat (muzzle) plus a cruise | possibly (the turn exponent's mirror: close in when out-ranged) |
| 4. Speed against reach | 0.68-0.69 measured (Me 262, Me 163) | not independent of 1 | flight | the energy fight |

Mechanism 1 is the strongest and is already measured; 2 is the best understood and the
cheapest to search, but marginal on its own. They share the guns and the shield, so
independence has to be built, not found: build each cycle on a common base that holds the
other mechanisms' handles fixed (cycle 1 at one gun and one shield, cycle 2 on one
airframe), as below.

## The scripted agent's relative tactics

Every distance, angle and speed the agent uses is already in its own ship's units. Two
things were missing: its times were in seconds, and it flew one doctrine against every
enemy, so a type could not fight to its strength against a particular one. Both are now
parameters of `bnb.controllers.scripted_parameters.ScriptedParameters`, off by default.

**What the agent can know.** The enemy's type is the `ship_type` column, which the
observation always shows, even for a ship the team cannot see; the agent already reads it
for the target's hit radius. The card is common knowledge (the agent's characteristics
are measured from it). So comparing the target's type with its own is honest: it is
recognising an aircraft. The enemy's store (`energy`) is shown only while it is seen; the
doctrine below uses the types alone.

**Own time** (`time_unit_exponent`, 0 by default). The memory fade's time and every
switch's rate are in the ship's own time unit
(`bnb.controllers.scripted_memory.time_unit`): its corner turn time, the corner speed over
the g limit's acceleration, over the card's first type's (the type each set was tuned on),
raised to the exponent. At 0 the unit is exactly one second; at 1 a type that turns half as
fast remembers twice as long and reconsiders half as often. The memory is the team's, so
it fades in the team's mean unit (each ship's own when a team flies one type). The
remaining set-wide quantities (the mode classifier's longest gun range, the chaser's
maxima, the Frontline strengths in ships) do not enter a one-type-a-side deathmatch.

**The energy fight** (`energy_fight_strength`, 0 by default). Against its committed target,
a ship's doctrine weight is the strength times `energy_fight_in_advantage`, a logistic in
the product of four own-over-target ratios, each raised to its exponent: the settled
full-stick turn rate (exponent -1 by default, so the weight rises when out-turned), the
climb power (`climb_power`, the cruise times the coast exchange: g times the climb rate),
the cruise speed and the gun range (exponents 0). In the doctrine a ship whose target is
close (`extend_in_gun_range`, falling) and far off its nose (`extend_in_stall_angle`,
rising, in stall angles) extends: it steers away from the target and boosts instead of
turning after it, then comes back for another pass; it never reverses to tighten a turn.
The steer and the power are mixed in linearly by the weight
(`bnb.controllers.scripted_combat.energy_fight`); the shot is untouched.

**Off is the agent exactly.** At a strength of 0 the mix adds an exact zero, and at an
exponent of 0 the time unit is exactly 1, and no new random number is drawn. Seeded CPU
rollouts of the agent against itself in all five modes (six games each, three 64-tick
segments, random lineups over every type of each set) give bitwise the same actions and
marginals before and after the change, and
`tests/sections/test_scripted_relative.py::test_off_the_relative_tactics_and_own_time_change_nothing_bitwise`
checks it with every other new number moved.

**Searching them.** `bench/scripted_search.py` keeps these fields fixed unless
`--relative-tactics` is given, so a search without it draws exactly as before. With it,
the additive numbers (`ADDITIVE_BOUNDS`) step by sigma times their range and stay within
it; the three logistics step like the others, above their scale floors:

| Parameter | Default | Search range |
|---|---|---|
| `time_unit_exponent` | 0 | 0 to 1 |
| `energy_fight_strength` | 0 (off) | 0 to 1 |
| `energy_fight_turn_exponent` | -1 | -2 to 2 |
| `energy_fight_climb_exponent`, `energy_fight_speed_exponent`, `energy_fight_range_exponent` | 0 | -2 to 2 |
| `energy_fight_in_advantage` | centre 1, scale 0.1 | centre 0.5 to 2 (a factor 2 either side of an even match), scale 0.05 to 0.5 |
| `extend_in_gun_range` | centre 1, scale 0.2 | centre 0.3 to 3 gun ranges, scale 0.1 to 1 |
| `extend_in_stall_angle` | centre 4, scale 0.2 | centre 1 to 10 stall angles, scale 0.158 to 1.58 |

In a one-type-a-side match each team's views can play their own set (parameters are per
view), so per-type doctrine needs no new code in a search; the league's ship-type players
all play the mode's set.

## The first cycle: the plan

**Start.** Re-measure the Albatross, Zero and Anvil triangle at 1,024 games a pair, with the
energy fight off and on for the Albatross (about 35 s of GPU). It is the closest to the
target and its edges are larger than needed, so the search mostly pulls them in.

**The realism bounds.** Each handle stays within the 27 types' range, with a margin only
where the data is soft:

| Handle | Bound | Margin and why |
|---|---|---|
| stall speed | 32-56 m/s | none: the set spans the Wren to the Hammerhead |
| g limit | 6-10 | none |
| cruise | 130-236 m/s | none |
| store (`ceiling`) | 31-174 kJ/kg | none |
| climb time | 260-6,760 s | none |
| sustained turn | 11-25 deg/s | ±10% past the set's 11.8-22.8: no turn rate is measured, the literature values are compilations and the rest a model times 0.88 |
| low-speed acceleration | 1.6-8.2 m/s² | none (all model estimates) |
| zero-lift drag | 3.0e-5 - 1.06e-4 /m | none |
| nose reversal | 0.48-1.31 s | none (placeholders scaled by roll rate) |
| damage a hit | 9-133 | none |
| fire rate | 2, 3, 5, 6, 10, 15 Hz | whole Hz dividing 30 (the tick) |
| muzzle speed | 520-880 m/s | none |
| spread | 0.27-1.54 m/s | none |
| hit radius | 4.6-8.3 m | none: the half-span |
| shield | 80-160 | none (a proposal's 1-5 survivability scale) |

The envelope includes the eight invented designs; the 19 real aircraft alone span less
(turn 14.1-22.5 deg/s, cruise 130-231 m/s). Placeholders that are the same on every type
(the recharge, the head-on fraction, the half-lift drag, the dive and zoom accelerations,
the vision range) stay fixed unless the author gives them a range.

**The objective.** Orient the cycle (A > B > C > A) and take each edge's log-odds; the loss
is the sum of their squared distances from logit(0.61) = 0.45. Three equal log-odds are
three equal ratings within the triple, so the band and the equal Elo are one condition. A
candidate whose edge points the wrong way scores far worse than one at 0.50, which is what
the square already does.

**The variables.** Six to eight handles a type (the four of its mechanism's leg and the two
to four that balance it) and four agent numbers a type (`energy_fight_strength`, the turn
exponent and the two extend centres), about 30 numbers, each in the log of its handle
scaled to its bound's width.

**The optimiser.** CMA-ES (population 14 for 30 numbers) on that loss. A flight handle
moves the card through `bnb.game.handling.estimate.Estimate`, anchored on the type the
candidate started from: instant, within 1.5% for a 10% move and about 7% for a third. A
combat handle moves it exactly (`combat_card`). Each candidate's card is a three-type card
with the same shapes, so its values change without a recompile; its characteristics are
measured on the CPU (about 30 s for three types), alongside the GPU's play. Only the
winner is solved exactly (the climb iteration takes minutes per type at real scale: about
15 minutes for three types on 14 cores), its handles re-measured against the bounds and its
three edges played again at 2,048 games a pair. If the exact cards move an edge out of the
band, CMA-ES restarts from them as new anchors for a few generations.

**Noise.** At 256 games a pair, an edge near 0.6 has a standard error of 0.031, 0.13 in
log-odds, so the loss of a good candidate has a standard deviation of about 0.15 (three
edges): enough to rank a population whose spread is wider than that. The mean is
re-evaluated every generation, and the games a pair go to 1,024 once the step size has
shrunk by half.

**Cost.** A generation is 14 x 3 x 256 = 10,752 games plus 768 for the mean, about 11,500;
at the laptop GPU's measured rate (1,152 real-scale 5v5 games in 13 s, about 88 a second)
that is about 130 s. Sixty generations: about 2.2 h; the confirmation and the exact solves
add about 25 minutes. Started at 16:00, it would end about 18:40 (worst 20:00, if the
noise needs 1,024 games a pair for the last 20 generations). A CPU check is not a
substitute: a real-scale 5v5 game is about 1.6 s of CPU at 256 games a batch, so one
generation would take over five hours.

## The second and third cycles, and seven types

**Building independence.** Build cycle 2 on one airframe (the three types share every
flight handle; only the guns, the shield and the recharge differ: mechanism 2), and cycle
3 on one gun and one shield with different flight (mechanism 1, or 3 if 1 is taken). A
cycle whose types differ only in one mechanism's handles cannot borrow another's.

**Testing it.** Cross the cycles: give cycle 2's three gun sets to cycle 1's three
airframes (nine types) and play the 9 x 9 round robin. If the two are independent, the
pair log-odds are the sum of the two cycles' (an additive model with two cyclic planes
fits within noise), and each cycle's edges, read with the other's handles held, do not
move. A fit of multidimensional Elo with two planes against one with an interaction term
tells which.

**Seven from three.** The Paley tournament's antisymmetric matrix has rank 6: three
planes, and on the cyclic group it is exactly a sum of three rotations,
`S(i, j) = (2 / sqrt 7) sum_k chi(k) sin(2 pi k (j - i) / 7)`{.plain} over k = 1, 2, 3, with
chi the quadratic character (chi(1) = chi(2) = 1, chi(3) = -1). In plane k type i sits at
angle `2 pi k i / 7`{.plain}, all seven on one circle. For every edge to be 0.60, each
plane needs an amplitude of 0.31 in log-odds, against 0.47 for a lone three-cycle at 0.60:
a mechanism strong enough for a 0.60 triangle is strong enough for its plane of the seven.

What it needs that a triangle does not: each mechanism has to place seven types around a
circle, not three at its corners. A triangle's three legs (energy, turn, toughness; or
sprayer, cannon, armour) are the corners of a simplex, and a type in between is a mixture
of them; each of the seven is a point in each of the three simplices, at its phase. So
each mechanism must be continuous between its corners, the planes must add in log-odds,
and seven mixtures of three mechanisms must still fit the envelope. The first is a
measurement on each cycle (play mixtures against the corners), the second the
independence test, the third a constraint for the seven-type search, which is the same
CMA-ES with 21 types' handles and a loss over the 21 pairs.

## Open questions

- **The placeholders.** The recharge rate and delay, the head-on fraction and the
  half-lift drag are the same on every type. The burst mechanism's second lever is the
  recharge; the realism rule gives it no range. A range (say ±50%) is a game decision.
- **The balance pass.** Equalising the damage per second removes the strongest transitive
  axis and part of the guns-against-shield interaction. Stage 1 should start from the set
  as the balance pass leaves it, and the Hodge split above should be re-run on its
  tournament.
- **Which agent rates a type.** The cycle is measured with the scripted agent, per-type
  doctrine included. Whether a trained policy sees the same cycle is the real test; the
  energy fight is only the agent's version of fighting to a type's strength.
- **Per-type doctrine in the league.** Ship-type players play the mode's set; giving a
  player its own set is a small change to `bnb.league.matches.Player` when the league needs
  it.
- **This page's place.** The plan sections are a proposal; when stage 1 runs they move to
  the design records and this page keeps what was measured.
