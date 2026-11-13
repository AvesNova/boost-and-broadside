---
covers: [src/bnb/reference/real_planes.py, src/bnb/reference/real-plane-types/]
---

# The real planes (reference)

The real-scale set was first built from research: 18 late-war fighters (the P-51D first)
and 8 invented designs given as multipliers of the P-51D, each quantity with its confidence
and source, converted to handles and solved into cards. On 2026-11-13 the author dropped
those types from the game: the set now holds only the ships the game uses, Yongma and Jujak
and vizia, the P-51D-derived hull both come from ([The real-scale variant](real-scale.md)).
The research stays, as reference data that no game preset loads:

- `bnb.reference.real_planes.AIRCRAFT`: every plane's researched quantities (SI;
  `bnb.reference.real_planes.QUANTITIES` gives the units), each a `Datum` with its value, confidence and source;
- `bnb.reference.real_planes.solved_types()`: every type as the set last held it, one TOML
  file each in `src/bnb/reference/real-plane-types/` (the ship-file format, at game v16 in
  the set's x2 native units): its handles as it flew, its solved card and a note per handle;
- [real-scale-research/](real-scale-research/aircraft.json): the raw research of 2026-11-10
  (aircraft, guns and the invented designs), with its conventions;
- [real-scale-balance/moves.json](real-scale-balance/moves.json): the balance rounds' moves
  ([Balancing the real-scale set](real-scale-balance.md)), and the
  [ship-type tournament](real-scale-tournament.md)'s ratings of the types.

The design tools use it: the realism envelope of a ship design is the 16 real propeller
fighters' knobs (`bnb.game.ship_design.ENVELOPE`, [Designing ship types](ship-design.md)),
and the similarity ratios of the ship card are stated against the P-51D
([The ship card](ship-card.md)).

The P-51D's values were not re-solved after game v12 (its hit radius is still the half-span,
5.645 m, and no type has the core radius v12 added); the table below is the research's
conversion, before the later game changes.

## The researched types

Their researched quantities, each with a confidence and a source, are in
`bnb.reference.real_planes.AIRCRAFT`; the research itself (sources, notes and the gun data)
is in [real-scale-research/](real-scale-research/aircraft.json). The set's builder (the
module `bnb.game.real_scale` until 2026-11-13) converted them to handles:

| Handle | From | How |
|---|---|---|
| stall speed | clean, power-off stall speed | as is |
| g limit | the limit load factor | as is |
| cruise speed | sea-level maximum speed, military power | as is |
| ceiling (the store) | service ceiling h | g h, J/kg |
| climb time | sea-level climb rate, military power, times the set's pacing (4: [Pacing](real-scale-pacing.md)) | solved so that the initial store charge, coasting at cruise, is g x the climb rate |
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
for the invented designs. The gun range is the bullet's reach in the gun flight time at cruise and
muzzle speed; the spawn energy keeps the `like` type's 40% of the store.

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

The heavy was a placeholder (today's heavy scaled as today's fighter is to the P-51D):
no researched type replaced it cleanly. It is not kept in the reference data.

Each type's file (`src/bnb/reference/real-plane-types/`) has `[notes]` saying, per handle, where its value comes from and how sure it is:
"researched (good/medium/low)", "estimated" (and how), "derived", "invented", "placeholder"
or "proposal". A handle the solver could not meet is marked MISSED with its error. At the
researched climbs only the Hammerhead missed (its sustained turn by 2.1% and low-speed
acceleration by 1.0%, nearly redundant for it, condition number 1e9). At four times the
climb ([Pacing](real-scale-pacing.md)) three invented designs miss: the Albatross its
sustained turn by 4.3% (the stronger engine against its long, low-drag wing), the Hammerhead
its sustained turn by 0.7%, and the Firecracker, whose climb is 243 m/s, its low-speed
acceleration by 27% and its cruise, turn and zoom by 2 to 4%; the cards kept are the closest
found.

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

The table gives the researched climbs. The set flies four times each
([Pacing](real-scale-pacing.md)): every climb time is a quarter of the table's, the shots
keep their cost in J/kg (so a shot is a quarter of the seconds of charge given below), and
the two rocket types' dive handles rise so that a boost still trades store for speed.

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

The builder rewrote the set's rules (the spawn speed followed the fastest stall speed) and
solved every type in its own process; a full build took about an hour and a half on 14
cores. It left the code with the types (git history keeps it, last at the commit before
`chore/drop-unused-planes`).

## The combat economy (the proposal of the time)

With the game's combat values the real-scale fighting was rare: a shot cost 3,759 J/kg
(the store over the standard 33 shots), about 23 s of the P-51's initial charge where a
standard shot costs about 1 s of its; and a 5.64 m half-span at 600 m is a quarter of the
angle a standard fighter presents at its gun range. The proposal, all in
the builder (CombatEconomy, combat_economy) and the set's ship files, so the
standard game was unchanged:

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
  The [balancing](real-scale-balance.md) has since evened it: every type now deals within
  5% of 360 per second at its own fire rate (the Me 262 378, 37.8 a shot), so the damages
  in this page's table are the economy's before the balancing.
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

