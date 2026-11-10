---
covers: [bench/case_study.py]
---

# Case studies of the trade-offs

Before a search moves many values at once, each trade-off is measured on its own: one
airframe (or a few contrasting ones), a family of variants that changes one thing or trades
two against each other, every variant rated against the others by the league's tournament.
They say how steep each direction is (Elo per doubling), which combinations leave the rating
where it was, and whether a family's variants go round among themselves: what a tuner can
move without large swings, and what a cycle can be built from.

## How a study runs

`bench/case_study.py play --spec SPEC.json --out DIR`, then `report` (the module docstring
has the spec's fields). Each (airframe, variant) is a type slot of the 36-slot arena whose
card is written once (`bnb.league.matches.Arena.set_card`; nothing recompiles between
studies); every ship of a team flies the slot's type and the scripted agent flies both
teams (its real-scale deathmatch set, T = 1). The ratings are the league's A-optimal
tournament (`bnb.league.tournament.fit_field`) to a standard error of 20 Elo or the budget.
A variant is knobs (`bnb.game.ship_design.KNOBS`: the fire rate keeps the damage per second,
the dispersion is the spread over the muzzle speed), knob multipliers, handle multipliers (the
shot cost, the recharge) or card-field multipliers (the live engine knee); the instant card
estimates any flight change from the airframe's solved card. The report's model is a
least-squares line of each variant's Elo against its airframe's baseline in the log changes
of what the variants vary, weighted by the ratings' errors.

The specs and reports are in [case-studies/](case-studies/guns-la7.json). The set is the
paced one ([Pacing](real-scale-pacing.md): four times the researched climb).

## The cycle the balancing found: Mosquito, Zerstörer and the light fighters

In the balancing's round 2 the Mosquito beats the Zerstörer (0.65), and the Zerstörer beats
six light single-engine types (La-7, P-51D, Spitfire IX and XIV, Wren, Yak-3: 0.55 to 0.69),
each of which beats the Mosquito (0.62 to 0.73). Their round-2 values (the ship files that
tournament played):

| Type | Shield | Hit damage | Fire rate | Turn deg/s | Hit radius m |
|---|---|---|---|---|---|
| Mosquito | 115 | 36.9 | 10 | 15.0 | 8.3 |
| Zerstörer | 130 | 37.1 | 10 | 11.8 | 7.8 |
| La-7 | 115 | 34.8 | 10 | 18.9 | 4.9 |
| Spitfire IX, XIV | 100 | 36.1 | 10 | 18.6-19.5 | 5.6 |
| P-51D, Yak-3, Wren | 100 | 22.8-24.0 | 15 | 15.7-22.8 | 4.6-5.6 |

and the hits a kill takes (the hits that break the shield, then one more; head-on hits
aside):

| Shooter on target | Mosquito (115) | Zerstörer (130) | a light type (100) |
|---|---|---|---|
| Mosquito (36.9) | | 5 | 4 |
| Zerstörer (37.1) | 5 | | 4 |
| light 10 Hz (34.8-36.1) | 5 | 5 | |
| light 15 Hz (22.8-24.0) | 6-7 | 7 | |

The reading: the Zerstörer's 130 shield is just over three of every light type's hits and
over five of the 15 Hz ones', so the light types need five to seven hits on it where it needs
four on them, and its poor turn matters little against types that must come to it. The light
types out-turn the Mosquito (15.7 to 22.8 against 15.0 deg/s; the P-51D only just) and present about two thirds
of its hit radius, which outweighs the hit or two more they need on it. The Mosquito and the
Zerstörer need five hits on each other, and the Mosquito turns better (15.0 against 11.8).
Two mechanisms make the cycle: the shield's threshold against the opponent's damage per hit
(the discrete ceil(shield / damage), the burst mechanism of the cycles analysis), and turn with
target size. The armour study below measures the first on its own.

## Guns on one airframe

**Question.** With the airframe fixed, how steep is each of a gun's values, and which
combinations leave the rating where it is ("the same gun performance, but it feels
different")?

**Design.** The La-7 on both sides (10 Hz, 348 damage a second, 770 m/s, its shield 110);
fifteen guns: the fire rate at 5, 6 and 15 Hz (the damage per second held, so the damage per
hit follows), the damage per second against the muzzle speed (1.15 x and 0.85 x, 1.3 x and
0.72 x, 0.9 x and 1.13 x), each of the two alone, the dispersion halved and doubled, the shot's
cost four times and a quarter. Every pair, A-optimal, 6,912 games
([case-studies/guns-la7/](case-studies/guns-la7/report.json)).

| Gun | Elo against the La-7's own |
|---|---|
| 5 Hz | -175 ± 15 |
| 6 Hz | -128 ± 15 |
| 15 Hz | +68 ± 15 |
| damage x 1.15, muzzle x 0.85 | -164 ± 15 |
| damage x 1.3, muzzle x 0.72 | -279 ± 16 |
| damage x 0.9, muzzle x 1.13 | +129 ± 15 |
| damage x 0.85 | -293 ± 17 |
| damage x 1.15 | +144 ± 15 |
| muzzle x 0.8 | -432 ± 18 |
| muzzle x 1.13 | +272 ± 17 |
| dispersion x 0.5 | +36 ± 15 |
| dispersion x 2 | +13 ± 14 |
| shot cost x 4 | -231 ± 16 |
| shot cost x 0.25 | +30 ± 15 |

The line through them (RMS residual 47 Elo), in Elo per doubling: muzzle speed +1,413, damage
per second +969, fire rate +164 (at an equal damage per second), shot cost (as shots per
store) +61, dispersion -11.

**What it says.**

- **The muzzle speed is the steepest value of the game**: 13% more is worth 272 Elo, 20% less
  costs 432. It sets both the reach and the lead a shot needs, and at real scale most misses
  are lead (the cone is a metre at 600 m). The guns' Spearman correlation with the balancing's
  ratings is this, more than the weight of fire.
- **The iso-rating trade of damage for muzzle speed** is about 1.46 doublings of damage per
  doubling of muzzle speed (1,413 / 969): 15% less muzzle speed needs 27% more damage. The
  "heavy slow cannon" is a fair gun only at that price, which pushes the damage per second
  beyond the envelope quickly; slower guns are better paid for with something other than
  damage.
- **Damage is not linear**: 15% less costs twice what 15% more gains (-293 against +144), at
  the La-7's 34.8 a hit on 100-115 shields, 15% less damage (29.6) needs a fourth breaking hit
  on a 100 shield where it needed three. The hits-to-kill threshold is a cliff the tuning has
  to see.
- **The fire rate is a mild lever** (+68 for 15 Hz, -175 for 5 Hz at an equal damage per
  second): faster is better for this agent (more chances a pass), not a matter of taste.
- **The dispersion is free**: halving or doubling the cone moves the rating by 13 to 36 Elo,
  within two errors of nothing. It is the knob that makes a gun feel different (a tight
  sniper's line, a shotgun's spray) without moving the balance.
- **The shot's cost only matters when it bites**: a quarter of today's is worth +30, four
  times costs 231 (the store runs dry in a fight).
- **No cycle among the guns**: every pair's result follows the ratings (no majority cycle). On
  one airframe, guns alone make a ladder.

## Shield, recharge and damage per hit

**Question.** How steep is the shield, does the recharge matter, and do the shield and the
damage per hit interact (the threshold of the Mosquito's cycle)?

**Design.** The La-7 (shield 110, recharge 12 a second after 6 s), every pair of fourteen
variants: the shield at 90, 130 and 160; the recharge delay halved and doubled, its rate three
times and a third; 5 and 15 Hz (the damage per second held); and the four crossings of 5 or
15 Hz with a shield of 90 or 130. 16,128 games ([case-studies/armour/](case-studies/armour/report.json)).

| Variant | Elo against the La-7 |
|---|---|
| shield 90 | -280 ± 11 |
| shield 130 | +319 ± 12 |
| shield 160 | +867 ± 19 |
| recharge delay x 0.5 / x 2 | +33 ± 10 / -34 ± 10 |
| recharge rate x 3 / x 0.33 | +31 ± 10 / -29 ± 10 |
| 5 Hz / 15 Hz | -170 ± 10 / +75 ± 10 |
| 5 Hz, shield 90 | -468 ± 14 |
| 5 Hz, shield 130 | +85 ± 10 |
| 15 Hz, shield 90 | -211 ± 11 |
| 15 Hz, shield 130 | +385 ± 12 |

The line (RMS residual 65 Elo): shield +1,195 Elo a doubling, fire rate +173, recharge rate
+19, recharge delay -33.

**What it says.**

- **The shield is as steep as the muzzle speed**: 20 points (18%) is 300 Elo; the cap raised
  to 160 is worth almost 900. The balancing's step per survivability rating (now 0.1, ten
  points a rating) is the right order: a rating step of 10 points is about 150 Elo.
- **The recharge is nearly free**: three times faster or a third, half or twice the delay, all
  within ±35 Elo. Ships die within a pass or survive it with time to spare; the recharge
  between passes rarely decides. Like the dispersion, it is a knob for feel, not balance (and
  too weak a lever for the burst mechanism's second axis).
- **Shield and damage per hit interact a little**: 5 Hz (70 a hit) with a 130 shield is 64 Elo
  below the sum of its parts; the 15 Hz crossings add up. Against an opponent's damage per
  hit the shield's value steps at its thresholds; on one airframe and one opponent gun family
  that is a 60-Elo effect, enough for a 0.58 edge and no more. No majority cycle.

## Climb and store against turn

**Question.** Is energy (the climb, the store, the cruise) a lever against the turn, as the
energy-turn-toughness mechanism of the cycles analysis needs?

**Design.** The La-7, every pair of twelve variants: the climb (from the paced level) halved
and doubled, the store at 0.75 and 1.33 times, the sustained turn at 0.9 and 1.1 times, three
energy-against-turn trades, the stall speed at 0.9 times and the cruise at 1.05 times (flight
changes on the instant card). 4,608 games: every variant is near the baseline, so the target
error came quickly ([case-studies/energy-turn/](case-studies/energy-turn/report.json)).

| Variant | Elo against the La-7 |
|---|---|
| climb x 0.5 / x 2 | +20 ± 15 / -33 ± 15 |
| store x 0.75 / x 1.33 | +1 ± 15 / +7 ± 14 |
| turn x 0.9 / x 1.1 | -70 ± 15 / +75 ± 15 |
| climb x 2, turn x 0.9 | -114 ± 16 |
| store x 1.33, turn x 0.9 | -76 ± 15 |
| climb x 0.5, turn x 1.1 | +86 ± 15 |
| stall speed x 0.9 | -60 ± 15 |
| cruise x 1.05 | +5 ± 15 |

The line fits within 6 Elo: turn +507 Elo a doubling, stall speed +398, cruise +77, store +6,
climb -27.

**What it says.**

- **The scripted agent fights on the turn alone.** A tenth more sustained turn is worth 75
  Elo; a third more store, twice the climb or 5% more cruise are worth nothing (the climb's
  slight cost is the stronger engine's drag, [Pacing](real-scale-pacing.md)). So energy cannot
  pay for turn in this agent's hands, and a flight cycle (energy fighter, turner, climber) will
  not form from the flight values with today's doctrine: the energy leg needs the agent's
  energy fight ([relative tactics](ship-design.md#the-knobs), off by default), which extends and
  boosts against a better turner instead of turning with it.
- **A lower stall speed costs** (-60 for 10% lower). With the sustained turn held the card
  meets it with more induced drag; which of the changes costs is not measured here.
- The four majority cycles have weakest edges of 0.51 to 0.52: noise among near-equal
  variants, not cycles.

## The engine knee

**Question.** Does the engine knee (where the engine's power starts to fall as the store
fills) change a fight?

**Design.** The La-7 with the knee at 0.3, 0.5 (the rules'), 0.7 and 0.9 of the store (the live
card scale), and at half the climb with the knee at 0.5 and 0.9; every pair, 2,304 games
([case-studies/knee/](case-studies/knee/report.json)).

| Variant | Elo against the La-7 |
|---|---|
| knee 0.3 | -13 ± 15 |
| knee 0.7 | +12 ± 15 |
| knee 0.9 | -23 ± 15 |
| climb x 0.5, knee 0.5 | -45 ± 15 |
| climb x 0.5, knee 0.9 | -39 ± 15 |

**What it says.** Nothing a fight notices: every knee is within two errors of the rules' (-1
Elo a doubling of the knee). The ships seldom hold more than half their store
([Pacing](real-scale-pacing.md)), where the knee acts. Half the climb costs 39 to 45 Elo here
and gained 20 in the climb-and-turn study, within three errors of each other: the climb is a
lever of at most a few tens of Elo either way, and its sign depends on the field.

## Guns on four airframes

**Question.** Does a gun's value depend on the airframe that carries it? A gun that is worth
more on one airframe than another is a lever for cycles; one that is worth the same
everywhere is a pure rating.

**Design.** The La-7's gun (10 Hz, 348 a second, 770 m/s, its dispersion) on the La-7, the
Zero (the turner), the P-38L (the energy twin) and the P-47D (the tough one), each airframe
keeping its own shield, and five variants of that gun on each; a variant plays the other
variants of its airframe, and the four baselines play each other. 20,736 games
([case-studies/guns-airframes/](case-studies/guns-airframes/report.json)).

| Gun | La-7 | Zero | P-38L | P-47D |
|---|---|---|---|---|
| the La-7's (the airframe's Elo against the La-7) | 0 | -350 ± 10 | -56 ± 8 | +13 ± 8 |
| 5 Hz | -209 ± 14 | -171 ± 17 | -69 ± 15 | -231 ± 16 |
| 15 Hz | +60 ± 13 | +34 ± 16 | +2 ± 15 | +45 ± 15 |
| damage x 1.3, muzzle x 0.72 | -328 ± 15 | -360 ± 19 | -154 ± 16 | -215 ± 16 |
| muzzle x 1.13 | +231 ± 14 | +219 ± 17 | +214 ± 16 | +256 ± 17 |
| dispersion x 2 | -1 ± 13 | +3 ± 16 | -20 ± 15 | -15 ± 15 |

Per airframe, in Elo a doubling: muzzle speed +1,213 to +1,452; damage per second +601 (Zero)
to +1,249 (P-47D); fire rate +52 (P-38L) to +188 (P-47D); dispersion -20 to +3.

**What it says.**

- **The muzzle speed is worth the same on every airframe** (about 220 Elo for 13% on all
  four): a pure rating, no lever for a cycle.
- **The slow heavy gun and the slow fire rate depend on the airframe.** The heavy cannon
  (30% more damage at 28% less muzzle speed) costs the light La-7 and Zero 330 to 360 Elo and
  the P-38L 154; 5 Hz costs the P-47D 231 and the P-38L 69. The P-38L, the steady twin with a
  large store, loses least from guns that need a steady approach: about 170 Elo of airframe
  dependence, the size of a 0.70 edge. That is a lever: a slow, hard-hitting gun on a steady
  airframe against a fast, light gun on an agile one.
- **The damage per second matters twice as much on the tough P-47D as on the fragile Zero**
  (1,249 against 601 Elo a doubling): a ship that survives longer converts more of its damage.
- **The dispersion is free on every airframe.**
- The one majority cycle (weakest edge 0.50) is noise.

## What the studies imply

- **To move a rating without a swing**, use the steep values in small steps: 1% of muzzle
  speed is about 20 Elo, 1% of shield about 17, 1% of damage per second about 14 (more on a
  tough airframe, and with a cliff where a hit's damage crosses a shield's threshold).
- **To make types feel different at the same rating**, use the free values: the dispersion
  (a tight line against a spray) and the shield's recharge (fast and short against slow and
  long) move the rating by less than 40 Elo across a factor of four.
- **To build a cycle**, use what interacts: the shield against the opponent's damage per hit
  (the thresholds), the gun's character against the airframe (a slow heavy gun on a steady
  airframe, a fast light one on an agile one: about 170 Elo of dependence), and the turn with
  the target's size. Energy (the store, the climb, the cruise) does not enter the scripted
  agent's fight until it flies the energy fight, per type.
- **What not to use**: the engine knee (it does nothing below half the store), and the muzzle
  speed for anything but the rating (it is worth the same on every airframe).
