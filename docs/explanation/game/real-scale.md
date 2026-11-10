---
covers: [src/bnb/game/real_scale.py, src/bnb/game/real_scale_aircraft.py, src/bnb/game/ships/real-scale/, src/bnb/game/handling/similarity.py]
---

# The real-scale variant

The standard game is built for play: its ships turn fast, on a small map, under a gravity
of 27 m/s^2. The real-scale variant keeps the game and changes its units so that one
metre is a metre and one second a second, with late-war fighters (the first a P-51D) and a
few invented designs that fly as the research has them. It is
the ship set `real-scale` (`src/bnb/game/ships/real-scale/`) and the presets
real_scale_deathmatch and real_scale_frontline (`bnb.game.modes.rules.PRESETS`). Everything stays per unit mass:
energies are J/kg and powers W/kg.

`bnb.game.real_scale` holds the arithmetic and writes the set
(`uv run --no-sync python -m bnb.game.real_scale`).

## From the standard fighter to a P-51D

Today's fighter maps onto a P-51D with lengths x 5.82 and times x 4.00: its stall speed,
31.6 m/s, becomes 46 m/s (x 1.455), and its gravity, 27 m/s^2, becomes 9.81 m/s^2
(x 0.3633, length over time squared). The time factor is the one that turns one gravity
into the other, sqrt(5.82 x 27 / 9.81) = 4.0023.

Every flight constant scales by its units (`bnb.game.handling.similarity`): the
exchanges, as accelerations, by 0.3633; the nose frequency floor, a rate, by 1/4.00; the
induced drag factor, a length at the default bleed exponent, by 5.82; the dimensionless
ones (stall angle, engine knee, fade band, damping) not at all. This gives the solver a
starting point in the right proportions; each type's card then fits its own handles. The
bullet drag per metre scales by 1/5.82.

## The map and the clock

The map is three times the standard one in metres: the presets' arena radius is 7800 m
with the map scale's reference radius kept at 2600 m, so the map scale is 3 and the
fields, the zones and the spawn disc scale by themselves. The distances that do not follow
the map scale are set three times in the set's rules: the field band width (120 m) and
team deathmatch's spawn distances (group radius 900 m, respawn distances 750 to 1800 m,
spawn margin 150 m).

The rules' times follow the map, not the ship. A P-51 cruises about 1.57 times as fast as
today's fighter in metres per second, over a map three times as wide, so crossing between
fields takes about twice as long. The episode doubles to 600 s, and the capture time (20 s)
and the respawn waits (10 s in team deathmatch, 16 s in Frontline) double with it.

## The rules

| Rule | Value | Why |
|---|---|---|
| gravity | 9.81 m/s^2 | real |
| tick | 30 Hz | unchanged |
| bullet life | 0.6 s (18 ticks) | the P-51's gun range of about 600 m (587 m at its researched speeds) |
| bullet drag | 1.375e-4 /m | today's 8e-4 /m over 5.82 |
| vision range | 3072 m | today's 1024 m times the map scale |
| spawn speed | 72.6 m/s | 1.3 times the set's fastest stall speed (the Hammerhead's 55.9 m/s) |
| field band width | 120 m | three times today's |
| episode | 600 s | twice today's |
| capture time | 20 s | twice today's |

Bullet life is one rule for every type, so another type's gun range follows its own
muzzle and cruise speed. The spawn speed is set above every stall speed so that no ship
starts in a stall; the build takes 1.3 times the fastest in the set, so it follows the set
as types are added.

## The ships

The set holds 27 types: the P-51D first, the heavy placeholder second (the game's default
lineup alternates the first two), then 17 more late-war fighters and 8 invented designs.
Their researched quantities, each with a confidence and a source, are in
`bnb.game.real_scale_aircraft`; the research itself (sources, notes and the gun data) is
in [real-scale-research/](real-scale-research/aircraft.json). `bnb.game.real_scale`
converts them to handles:

| Handle | From | How |
|---|---|---|
| stall speed | clean, power-off stall speed | as is |
| g limit | the limit load factor | as is |
| cruise speed | sea-level maximum speed, military power | as is |
| ceiling (the store) | service ceiling h | g h, J/kg |
| climb time | sea-level climb rate, military power | solved so that the initial store charge, coasting at cruise, is g x the climb rate |
| low-speed acceleration | the research's point-mass model at 1.1 x stall | x 0.88 for propeller types (the research's own correction: its model reads 8-17% high where it can be checked) |
| sustained turn | a literature value where the research has one, else its model | the model x 0.88 for propeller types |
| nose reversal | peak roll rate | the P-51D's (a placeholder) x its 95 deg/s over the type's |
| stall angle | estimated stall angle of attack | as is |
| zero-lift drag | C_D0 and wing loading W/S | rho C_D0 / (2 W/S), rho 1.225 kg/m^3 |
| hit radius | wing span | half of it |
| muzzle speed | the gun carrying most of the weight of fire | as is |
| bullet spread | the volley's 75% cone diameter D75 (mil) | sigma = D75 / (2 x 1.665) rad, times the muzzle speed; gun spacing ignored (nose guns, no convergence) |
| max shield | survivability rating (1-5) | proposal: 100 x (1 + 0.2 per step from the P-51D's 2): 80 to 160 |
| fire rate, damage, shots per store | weight of fire, main gun, climb rate | the combat economy below (a proposal) |

The invented designs are given as multipliers of the P-51D; each value is the P-51D's
converted value times the design's multiplier (marked "invented" in the notes). Every
handle a type does not give is a placeholder, its `like` type's standard handle scaled as
the standard fighter is to the P-51D: the dive and zoom accelerations and the half-lift
drag for all types, the nose reversal for the P-51D, and the stall angle and hit radius
for the invented designs. The gun range follows from the bullet life, cruise and muzzle
speed; the spawn energy keeps the `like` type's 40% of the store.

The P-51D's researched values moved its handles from the first build's: stall speed
46 to 44.7 m/s (-2.8%), cruise 163 to 162.7 m/s, store 125,300 to 124,391 J/kg (the
NAA-calculated 12,680 m ceiling), climb rate 16.3 to 15.2 m/s (-6.7%; the research
estimates military power from the 17.3 m/s WEP figure), sustained turn 18.0 to 15.7 deg/s
(-12.8%, a simulator compilation), stall angle 16 to 15.5 deg, muzzle speed 880 to 856 m/s
(the 36-inch barrel figure), and hit radius 5.64 to 5.645 m (the 37 ft 0.5 in span).
The climb time follows: 3,918 to 4,163 s. Two placeholders became the research's values:
the low-speed acceleration 3.48 to 2.73 m/s^2 (-22%, its model's estimate) and the
zero-lift drag 1.20e-4 to 4.97e-5 /m (-58%, Loftin's C_D0 of 0.0163). The bullet spread
went from the standard dispersion angle (21.1 m/s, 24 mrad) to the P-51's 6 mil wing
battery (1.54 m/s, 1.8 mrad), 14 times tighter. The g limit stays 8. The scaling
constants (LENGTH 5.82, TIME 4.00) keep the first build's 46 m/s: they set only the flight
constants and the placeholders.

The heavy stays a placeholder (today's heavy scaled as today's fighter is to the P-51D):
no researched type replaces it cleanly, and the default lineup and the tests name it.

Each ship file's `[notes]` says, per handle, where its value comes from and how sure it is:
"researched (good/medium/low)", "estimated" (and how), "derived", "invented", "placeholder"
or "proposal". A handle the solver could not meet is marked MISSED with its error: only
the Hammerhead's sustained turn (2.1%) and low-speed acceleration (1.0%), which its card
cannot meet together (the solver finds the two nearly redundant for it, condition number
1e9); the card kept is the closest it found.

| Type | Role | Stall / cruise m/s | Climb m/s (climb time s) | Turn deg/s | g | Store kJ/kg | Guns: Hz x damage, spread m/s | Shield | Data (stall, cruise, climb, turn, roll, C_D0, dispersion) | Worst flown error |
|---|---|---|---|---|---|---|---|---|---|---|
| P-51D | long-range escort fighter | 44.7 / 162.7 | 15.2 (4163) | 15.7 | 8 | 124.4 | 15 x 24.0, 1.54 | 100 | m, g, e, l, m, g, l | 2.4e-06 |
| Heavy | placeholder | 49.5 / 166.3 | 15.5 (5702) | 16.3 | 5.93 | 174.1 | 10 x 32.0, 1.54 | 160 | p, p, p, p, p, p, p | 4.5e-07 |
| P-47D | fighter / fighter-bomber | 49.2 / 152.0 | 14.1 (4424) | 14.4 | 8 | 122.6 | 15 x 32.0, 1.54 | 160 | e, m, e, l, m, m, l | 1.6e-06 |
| P-38L | twin-engine long-range fighter | 52.8 / 148.0 | 18.1 (3697) | 16.7 | 7.5 | 131.6 | 15 x 25.2, 0.93 | 140 | e, g, m, l, l, m, l | 6.4e-09 |
| Spitfire IX | air-superiority fighter | 41.1 / 150.2 | 23.5 (2645) | 19.5 | 8 | 122.3 | 10 x 37.5, 1.22 | 100 | e, g, g, l, l, e, e | 2.7e-06 |
| Spitfire XIV | air-superiority fighter / interceptor | 43.4 / 161.8 | 25.6 (2553) | 18.6 | 8 | 128.6 | 10 x 37.5, 1.22 | 100 | e, m, g, e, e, e, e | 1.2e-06 |
| Tempest V | low/medium-altitude fighter | 45.2 / 168.1 | 22.3 (2314) | 15.7 | 8 | 101.6 | 15 x 42.4, 1.01 | 140 | e, g, g, l, e, e, e | 4.4e-07 |
| Mosquito | twin-engine fighter-bomber | 49.2 / 148.4 | 10.2 (4444) | 15.0 | 6 | 89.1 | 10 x 59.4, 0.82 | 120 | e, g, e, e, e, e, e | 2.0e-06 |
| Fw 190A | fighter / bomber-destroyer | 52.8 / 154.7 | 14.4 (3569) | 15.7 | 7 | 101.0 | 10 x 48.4, 0.97 | 140 | e, g, g, l, m, e, l | 7.3e-06 |
| Fw 190D | air-superiority fighter | 52.8 / 151.4 | 17.0 (3089) | 16.0 | 7 | 103.0 | 10 x 26.8, 0.97 | 120 | e, g, g, l, l, e, l | 4.4e-06 |
| Bf 109K | interceptor / air-superiority fighter | 47.2 / 165.3 | 22.5 (2770) | 17.6 | 7 | 122.6 | 10 x 42.2, 0.37 | 100 | e, m, m, l, e, e, l | 9.7e-07 |
| Zero | carrier fighter | 37.5 / 130.6 | 15.7 (3731) | 22.5 | 7 | 115.2 | 10 x 23.2, 0.61 | 80 | e, e, l, l, l, e, e | 1.7e-06 |
| Me 262 | jet interceptor | 55.6 / 227.8 | 20.0 (2866) | 14.1 | 7 | 112.3 | 10 x 133.4, 0.28 | 120 | e, m, m, e, e, e, e | 1.6e-08 |
| Ki-84 | fighter | 44.4 / 155.6 | 18.5 (2832) | 18.7 | 7.5 | 103.0 | 15 x 19.8, 0.99 | 120 | e, e, e, e, e, e, e | 6.4e-07 |
| Yak-3 | low-altitude air-superiority fighter | 45.8 / 157.5 | 21.0 (2471) | 19.5 | 8 | 102.0 | 15 x 16.2, 0.71 | 100 | e, m, e, m, e, e, e | 7.3e-08 |
| Corsair | carrier fighter / fighter-bomber | 45.4 / 155.6 | 16.5 (3569) | 14.7 | 7.5 | 115.8 | 15 x 24.0, 1.54 | 140 | g, m, g, e, m, e, l | 5.6e-06 |
| La-7 | low/medium-altitude fighter | 45.8 / 160.8 | 18.0 (2980) | 18.9 | 8 | 105.5 | 10 x 18.8, 0.81 | 120 | e, m, e, m, e, e, e | 6.5e-08 |
| Me 163 | rocket point-defence interceptor | 44.4 / 230.6 | 81.3 (743) | 19.3 | 7 | 118.7 | 10 x 66.7, 0.39 | 80 | e, e, m, e, e, e, e | 2.3e-07 |
| Hellcat | carrier fighter | 43.4 / 142.2 | 14.8 (3834) | 15.4 | 7.5 | 111.5 | 15 x 24.0, 1.54 | 160 | e, l, g, e, m, g, l | 9.2e-06 |
| Hammerhead | Energy interceptor: a huge engine on a small wing. | 55.9 / 179.0 | 28.9 (3067) | 11.8 | 7 | 174.1 | 5 x 64.8, 0.27 | 120 | i, i, i, i, i, i, i | 2.1e-02 MISSED low_speed_acceleration, sustained_turn_rate |
| Wren | Nimble brawler: very low wing loading, light guns. | 32.2 / 139.9 | 17.5 (3077) | 22.8 | 10 | 105.8 | 15 x 8.9, 1.03 | 80 | i, i, i, i, i, i, i | 8.6e-06 |
| Anvil | Heavy gunship: one big slow cannon, armoured bathtub. | 51.4 / 146.4 | 11.4 (4438) | 12.6 | 6 | 99.5 | 2 x 133.2, 0.44 | 160 | i, i, i, i, i, i, i | 2.8e-06 |
| Albatross | Glider-like energy fighter: very low drag, long wing, weak engine. | 35.8 / 149.7 | 12.2 (6763) | 15.7 | 7 | 161.7 | 10 x 18.0, 1.16 | 100 | i, i, i, i, i, i, i | 1.3e-05 |
| Firecracker | Rocket sprinter: enormous thrust, a tiny store (propellant). | 49.2 / 235.9 | 60.8 (260) | 15.7 | 8 | 31.1 | 5 x 64.8, 0.39 | 80 | i, i, i, i, i, i, i | 4.8e-06 |
| Zerstörer | Twin-engine destroyer with a concentrated nose battery. | 53.6 / 157.8 | 14.4 (4164) | 11.8 | 6.5 | 118.2 | 10 x 64.4, 0.82 | 140 | i, i, i, i, i, i, i | 3.1e-05 |
| Shrike | Roll-and-dive slasher (Fw 190 vibe). | 50.1 / 167.6 | 15.2 (3746) | 13.8 | 9 | 111.9 | 10 x 43.2, 0.90 | 140 | i, i, i, i, i, i, i | 9.3e-08 |
| Lantern | Fast, fragile skirmisher. | 44.7 / 187.1 | 16.7 (4163) | 14.4 | 8 | 136.8 | 15 x 8.9, 1.03 | 80 | i, i, i, i, i, i, i | 1.2e-05 |

Data letters: g good, m medium, l low, e estimated, i invented, p placeholder, for the
stall speed, cruise speed, climb rate, sustained turn, roll rate, C_D0 and dispersion.
The worst flown error is the solver's worst relative error over the flown handles (its
acceptance is 0.5%); every type but the Hammerhead is within 3.1e-5.

Where the data is weakest: no turn rate is measured (literature values are simulator or
NII VVS compilations; the rest the research's model x 0.88); every low-speed acceleration
is the model's; most C_D0 values are back-solved from the measured top speed (the Tempest's
0.0151 and the F4U-1D's 0.0165 look low for that reason); the non-US g limits, the Japanese
and Soviet dispersions, most stall speeds (from wing loading) and every survivability
rating are estimates; the Me 163's sea-level speed is an estimate and its store is its
ceiling's, not its 7.5 minutes of propellant.

To add an aircraft, add an `Aircraft` with its researched quantities to
`bnb.game.real_scale_aircraft.AIRCRAFT` and run the module: it rewrites the set's rules
(the spawn speed follows the fastest stall speed) and solves every type, each in its own
process, writing each ship file as it finishes (`--resume` keeps the written cards whose
targets are unchanged; `--only KEY,...` solves just those). A real-scale solve takes
minutes per round of the climb iteration, so a full build takes about an hour and a half
on 14 cores.

## The combat economy (a proposal)

With the game's combat values the real-scale fighting was rare: a shot cost 3,759 J/kg
(the store over the standard 33 shots), about 23 s of the P-51's initial charge where a
standard shot costs about 1 s of its; and a 5.64 m half-span at 600 m is a quarter of the
angle a standard fighter presents at its gun range. The proposal, all in
`bnb.game.real_scale` (`CombatEconomy`, `combat_economy`) and the set's ship files, so the
standard game is unchanged:

- **Shot cost.** The standard fighter's continuous fire (15 shots/s at 300 J) costs 14.46
  times its initial climb charge (311 W/kg): 0.96 s of charge a shot. Every real-scale
  type's continuous fire costs the same 14.46 times its own initial charge (g x its climb
  rate). A P-51 shot costs 144 J/kg, 0.96 s of its 149 W/kg, and its store pays for 865
  shots. A type that fires slower volleys pays more a volley, the same per second.
- **Fire rate.** Real fire rates (the P-51's six .50s: 80 rounds/s) are beyond the 30 Hz
  tick, so each type fires whole-Hz volleys dividing 30: its main gun's volley rate from
  the research (15 Hz for .50s, 10 Hz for most cannon, 5 or 2 Hz for the invented heavy
  cannon).
- **Damage.** Damage per second follows the weight of fire: a type with weight of fire w
  deals D x w / 3.68 kg/s per second (3.68 is the P-51's), its damage a shot that over its
  fire rate. D, the P-51's damage per second, is provisional: 360, twice the standard
  fighter's 180 (12 x 15). At 180 the scripted agent's real-scale deathmatch had 16 kills
  (below); at 360, 28; at 540, 32. A P-51 then needs five hits on a P-51 (24 a shot against
  a shield of 100), the standard game's nine. The Me 262's 13.6 kg/s gives 1,334 per second,
  133 a shot at 10 Hz: one hit takes any shield below 140, as two or three MK 108 hits took
  a fighter. The Yak-3's 2.5 kg/s gives 243 per second.
- **Shields** follow the survivability rating (above): 80 for the Zero, the Me 163 and the
  fragile invented designs, 160 for the P-47D, F6F-5 and Anvil.
- **Dispersion** is researched, not proposed, but it moves the hit rate most: the real
  cones are 13 to 44 times tighter than the standard angle.

An alternative kept for comparison: the store pays for the type's real firing time
(ammunition over rate of fire). The P-51's 23.5 s at 15 Hz would be 353 shots, 2.2 s of
charge a shot; the Me 262's MK 108s, 9.7 s.

Headless matches of the scripted agent and the scripted chaser against themselves (seed
1, a whole episode each, `bnb.play.loop.MatchLoop`):

Kills / shots in one episode (600 s real-scale, 300 s standard); the default lineup
alternates the P-51D and the heavy, the mixed one is Me 262, Spitfire IX, Fw 190A,
Hammerhead and P-47D on both teams:

| Match | Game's combat values (first build) | Economy, D = 180 | Economy, D = 360 (chosen) | Standard game |
|---|---|---|---|---|
| deathmatch, scripted agent, default lineup | 3 / 195 | 16 / 1,052 | 28 / 1,440 | 45 / 1,280 |
| deathmatch, scripted agent, mixed lineup | | 17 / 835 | 35 / 533 | |
| Frontline, scripted agent, default lineup | 1 / 170 | 9 / 678 | 19 / 858 | 46 / 1,408 |
| deathmatch, scripted chaser, default lineup | 0 / 148 | 0 / 686 | 13 / 4,010 | 35 / 1,257 |
| deathmatch, scripted chaser, mixed lineup | | 2 / 2,506 | 12 / 2,538 | |

The real-scale fights now happen: 28 kills an episode against the standard 45, though at
about a third of the standard rate a minute (2.8 against 9), as the map is three times as
wide and the ships only 1.6 times as fast, so contact is rarer (shots a minute 144 against
256). The kills a shot (1.9% against 3.5%) are lower too: the target is a quarter of the
angle and the real cones put a burst all on or all off. The scripted chaser, whose aim
(within 8 degrees) is far coarser than a 1.8 mrad cone, hits rarely; the higher damage
is what lets its rare hits kill.
These are single seed-1 episodes, a provisional economy for the author to judge in play.

## The scripted agent's real-scale sets

The scripted agent plays the real-scale presets with sets of its own,
real_scale_deathmatch and real_scale_frontline in
`bnb.controllers.scripted_defaults.MODE_DEFAULTS`, chosen whenever the card's gravity is the
real-scale one ([The scripted agent](../scripted-agent.md)); the standard sets are
unchanged. They are a light retune: each preset's parameter search
(`bench/scripted_search.py --preset <preset> --ship-type 0 --from-defaults`) started from
the standard mode's set and played the P-51D lineup (every ship a P-51D), 32 candidates of
64 games a generation, for 12 generations in deathmatch and 13 in Frontline, about 17
minutes of GPU time with compiling. Eight finalists of each (the top four of the last
generation, two of each of the two before) then played 512 games each at T = 1, among
themselves and against the standard set; all eight were within noise of each other and
each beat the standard set in 95-100% of its games. Deathmatch keeps the best rated;
Frontline the one within noise of the best whose self-play drew least and captured second
most.

The check, `bench/scripted_compare.py`, 512 fresh games per row on the P-51D lineup, fog
on, seats drawn per game, seeds above the search's (kills are combat deaths, a front move
is one capture's step of the front):

| Preset | Match | Score of the first | Kills a game | Kills a minute | Shots a game | Hit share | Front moves a minute | Ended by the clock |
|---|---|---|---|---|---|---|---|---|
| deathmatch | real-scale set against the standard set | 0.994 +- 0.003 | 32.6 | 4.8 | 2,271 | 0.22 | | 11% |
| deathmatch | standard set, self-play | | 43.2 | 5.3 | 922 | 0.76 | | 23% |
| deathmatch | real-scale set, self-play | | 46.4 | 7.0 | 3,284 | 0.21 | | 11% |
| Frontline | real-scale set against the standard set | 0.969 +- 0.006 | 24.0 | 4.2 | 1,378 | 0.27 | 0.43 | 12% |
| Frontline | standard set, self-play | | 38.0 | 3.8 | 824 | 0.74 | 0.44 | 97% |
| Frontline | real-scale set, self-play | | 68.1 | 6.8 | 3,424 | 0.30 | 0.20 | 99% |

(A hit is a shot that damaged an enemy. The real-scale self-play drew 9% of its deathmatch
games and 69% of its Frontline games, the standard set 3% and 38%.)

What moved most: both sets fire at targets well beyond a gun range (the shot logistic's
centre from 0.53 to 1.71 gun ranges in deathmatch, 0.73 to 1.90 in Frontline) and accept
a coarser aim (the aim tolerance about doubled in deathmatch, 0.20 to 0.35 target sizes in Frontline), and fight from further
(the combat radius 3.0 to 4.7 gun ranges, 4.0 to 4.8). At 0.96 s of charge a shot, spraying
is cheap: the real-scale set fires three to four times as often for a quarter of the hit
share, and kills faster. In Frontline it captures less: in self-play the front moves
0.20 times a minute against the standard set's 0.44, and two thirds of its games end
drawn on the clock. Every Frontline finalist did the same (0.14 to 0.21 a minute in
self-play), so it is the search's direction, not one candidate's.

The pass did not cover the other types, the timing parameters' bounds, the decision floors
or the chaser; the backlog holds a proper retune (`docs/rewrite/backlog.md`).

With these sets the scripted agent has flown every type against every other: [The ship-type
tournament](real-scale-tournament.md).

## What is not scaled

The model and the learner (Yemong's position and velocity codes, the delta and correction
bounds, the discounts) are tuned to the standard game; the real-scale variant is for play,
by people and the scripted agent. The scripted agent states its distances in gun ranges,
corner radii and the like, and plays the real-scale presets with its own lightly retuned
sets (above); the chaser keeps the standard distances in the card's lengths. The solver's
windows and error floors follow each card's own scales ([The ship
card](ship-card.md#measuring)); a 125 kJ store climbs for about an hour of game time, so
a real-scale card takes minutes to measure or solve on a CPU, where a standard one takes
seconds. The tuning panel shows its sliders at once from the ship files and measures only
the shown type and the types in the match, 7 to 20 s a real-scale type.

The playback speed and the tuning panel's [time scale](../in-progress/play-shell.md#live-tuning)
are the way to try the variant faster than real time.
