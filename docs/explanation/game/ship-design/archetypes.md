---
covers: [bench/gun_lethality.py, src/bnb/controllers/scripted_combat.py]
---

# Two archetypes: the turn fighter and the energy fighter (draft)

A working spec for the author to edit. It proposes two variants of `vizia`, Jujak (주작,
the vermilion bird; the turn fighter) and Yongma (용마, the dragon-horse; the energy
fighter), and the conditions under which their matchup is decided by doctrine rather than
by who fires first. Nothing here is a ship file yet; every value is a proposal.

## 1. The two fighters of the Second World War, in numbers

**The turn fighter** (Spitfire, A6M Zero, Yak-3, Ki-43) slows the fight down and stays in
it: it draws the opponent into a turning contest at low speed, where its turn rate and
radius bring its guns to bear first. **The energy fighter** (P-47, Fw 190, F4U Corsair,
Tempest) keeps the fight fast: slash (a high-speed firing pass), extend (fly away faster
than the opponent can follow), climb or turn wide, and re-engage. Its advantage is
initiative: it chooses when a fight happens and when it ends. Each wins on its own terms
and loses on the other's; doctrine ("never turn with a Zero", boom-and-zoom) is the
discipline of staying on one's own terms.

The numbers below are the set's research (`real-scale-research/aircraft.json`, every value
with its source and confidence; sea level, military power, combat mass) and figures derived
from it by `docs/rewrite/sections/gun-study/archetypes_ww2.py` (`archetypes_ww2.json`):
the instantaneous turn at the corner speed (the stall speed times the root of the g limit)
and at 100 m/s (the lift limit or the g limit, whichever is lower), the sustained turn's
radius at the research model's best-rate speed, and the ballistic coefficient W/S over
C_D0, which sets the drag per unit weight at a given speed (how fast a type loses speed
when it turns gently, and how fast it accelerates and how far it goes in a dive).

| Type | W/S (kg/m²) | Power / mass (kW/kg) | Top speed, sea level / altitude (m/s) | Climb (m/s) | Stall (m/s) | Sustained turn (°/s; radius m) | Instant turn at 100 m/s (°/s) | Corner radius (m) | Roll (°/s) | Dive limit (m/s) | W/S / C_D0 (t/m²) | Weight of fire (kg/s) | Survivability (1-5) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Spitfire IX | 147 | 0.36 | 150 / 182 | 23.5 | 41.1 | 19.5; 183 | 32.8 | 174 | 90 | 201 | 7.6 | 3.83 | 2 |
| Spitfire XIV (leaning mixed) | 170 | 0.37 | 162 / 200 | 25.6 | 43.4 | 18.6; 205 | 29.3 | 194 | 90 | 210 | 9.1 | 3.83 | 2 |
| A6M5 Zero | 127 | 0.31 | 131 / 157 | 15.7 | 37.5 | 22.5; 152 | 38.9 | 145 | 80 | 183 | 5.8 | 2.37 | 1 |
| Yak-3 (light, fast low down: mixed) | 181 | 0.35 | 158 / 179 | 21.0 | 45.8 | 19.5; 228 | 26.2 | 216 | 100 | 194 | 8.9 | 2.48 | 2 |
| Ki-84 (mixed) | 171 | 0.37 | 156 / 173 | 18.5 | 44.4 | 18.7; 212 | 28.0 | 203 | 90 | 222 | 8.2 | 3.04 | 3 |
| P-47D | 220 | 0.24 | 152 / 198 | 14.1 | 49.2 | 14.4; 285 | 22.5 | 249 | 85 | 224 | 10.4 | 4.91 | 5 |
| Fw 190A-8 | 230 | 0.30 | 155 / 181 | 14.4 | 52.8 | 15.7; 310 | 19.4 | 287 | 150 | 236 | 10.2 | 4.95 | 4 |
| Fw 190D-9 | 233 | 0.30 | 151 / 182 | 17.0 | 52.8 | 16.0; 312 | 19.4 | 287 | 130 | 236 | 9.5 | 2.74 | 3 |
| F4U-1D Corsair | 187 | 0.27 | 156 / 187 | 16.5 | 45.4 | 14.7; 235 | 26.7 | 212 | 85 | 210 | 11.3 | 3.68 | 4 |
| Tempest V (low altitude) | 184 | 0.31 | 168 / 193 | 22.3 | 45.2 | 15.7; 226 | 26.9 | 210 | 100 | 241 | 12.2 | 6.50 | 4 |
| Me 262 (extreme) | 295 | 0.48 (thrust power) | 228 / 241 | 20.0 | 55.6 | 14.1; 350 | 17.3 | 318 | 60 | 278 | 15.4 | 13.64 | 3 |
| **P-51D (mixed reference)** | 201 | 0.26 | 163 / 198 | 15.2 | 44.7 | 15.7; 229 | 27.6 | 205 | 95 | 226 | 12.3 | 3.68 | 2 |

The archetypes as ratios (geometric means of each group: turn the first five, energy the
next five; the fleet is all 18 researched types), with the handle each maps to:

| Quantity | Turn / fleet | Energy / fleet | Turn / P-51D | Energy / P-51D | **Turn / energy** | Handle (`handles.py`) | How strongly |
|---|---|---|---|---|---|---|---|
| survivability | 0.71 | 1.48 | 0.94 | 1.97 | **0.48** | `max_shield` (the set's rule, 100 x (1 + 0.2 a step): 0.71) | directly |
| mass | 0.72 | 1.13 | 0.73 | 1.15 | **0.64** | none (per unit mass) | through the rows below |
| weight of fire | 0.74 | 1.07 | 0.83 | 1.19 | **0.70** | `hit_damage` (the set balances it to equal damage per second) | directly |
| wing loading | 0.81 | 1.08 | 0.79 | 1.04 | **0.75** | `stall_speed` (as its square root) | directly |
| W/S / C_D0 | 0.81 | 1.11 | 0.63 | 0.87 | **0.73** | `zero_lift_drag` (rho C_D0 / 2 W/S: its inverse) | directly |
| sustained turn | 1.16 | 0.90 | 1.26 | 0.97 | **1.29** | `sustained_turn_rate` | directly |
| instant turn at 100 m/s | 1.21 | 0.90 | 1.11 | 0.82 | **1.35** | follows `stall_speed` and `g_limit` | through them |
| instant turn at corner | 1.12 | 0.95 | 1.03 | 0.88 | **1.17** | follows `stall_speed` and `g_limit` | through them |
| low-speed acceleration | 1.34 | 0.87 | 1.60 | 1.04 | **1.54** | `low_speed_acceleration` | directly |
| power / mass | 1.04 | 0.84 | 1.37 | 1.12 | **1.23** | the engine (solved from cruise and climb) | through them |
| climb | 1.07 | 0.87 | 1.35 | 1.09 | **1.24** | `climb_time` (the store's charge) | directly |
| dive limit | 0.93 | 1.05 | 0.89 | 1.01 | **0.88** | `dive_acceleration` (store into speed) | loosely: a rate, not a limit |
| roll | 0.97 | 1.15 | 0.95 | 1.13 | 0.84 | `nose_reversal_time` (the set's stand-in; 2D has no roll) | weak: the Fw 190s alone |
| stall speed | 0.91 | 1.05 | 0.95 | 1.10 | 0.86 | `stall_speed` | directly |
| top speed, altitude / sea level | 0.94 / 0.94 | 0.99 / 0.97 | 0.90 / 0.93 | 0.95 / 0.96 | 0.94 / 0.96 | `cruise_speed` | directly |
| service ceiling | 1.01 | 0.97 | 0.91 | 0.87 | 1.05 | `ceiling` (the store) | shared |
| unpowered zoom, 200 to 60 m/s | 0.97 | 1.01 | 0.95 | 0.99 | 0.96 | `zoom_acceleration` | shared |
| g limit | 1.04 | 1.01 | 0.96 | 0.94 | 1.03 | `g_limit` | shared |
| half span | 0.93 | 0.99 | 0.97 | 1.03 | 0.94 | `hit_radius` | directly |

**What separates them** is weight and what it buys. The energy fighter is half again as
heavy (1.56 times) on a wing a third more loaded, so it turns worse at every speed
(sustained 1.29 times slower, at 100 m/s 1.35 times), stalls faster and needs 1.4 times the
room to turn; in exchange its drag is a smaller share of its weight (W/S over C_D0 1.37
times the turn fighter's), so it keeps speed in gentle manoeuvres, accelerates harder in a
dive and can be taken faster before it is limited (dive limit 1.14 times). It carries 1.4
times the weight of fire and is twice as tough. **What they share** is surprising: level top
speed (within 4-6%), the ceiling, the g limit and a zoom from equal speed. The turn fighters
climb and accelerate better at low speed and have more power per kilogram (1.23 times): the
energy fighter is neither the faster nor the better climber in level flight. Its speed
advantage was vertical, from height kept and a dive the other could not follow (the RAE's
and AFDU's 1942 trials of a captured Fw 190A-3 against the Spitfire IX: the Fw 190 faster in
the initial dive and the roll, the Spitfire better in the turn and, at most heights, the
climb; recalled, not re-checked). The famous pairs spread wider than the groups: the P-47D
over the Zero is 1.73 times the wing loading, 0.64 times the sustained turn and 1.27 times
the top speed at altitude; the Fw 190A-8 over the Spitfire IX 1.56, 0.81 and 1.00 (and roll
1.67, climb 0.61).

**What maps.** Wing loading, the turn, drag over weight, low-speed acceleration, the
charge (climb), armour and guns map directly onto handles. The dive maps loosely: our
dive is a boost (the store into speed), whose rate is a handle and whose top speed follows
from it and the drag. Roll has no 2D analogue (the set's nose reversal time stands in for
it, and only the Fw 190s separate on it). The ceiling and the zoom are shared. Altitude,
the energy fighter's real sanctuary, has none: the store stands in for it, equal in both.

## 2. The paradigm in our 2D game

### What maps to what

| Requirement | Handle (`bnb.game.handling.Handles`) | Card field it sets |
|---|---|---|
| wing loading | `stall_speed` (lift reach 1 g); the corner speed follows with `g_limit` | `max_lift_coefficient`, `max_acceleration` |
| turn rate | `sustained_turn_rate` (settled full-stick coasting turn) | `induced_drag_scale` |
| energy kept in turns | `half_lift_drag` (low: turns hard but bleeds in hard turns, keeps energy in gentle ones) | `bleed_exponent` |
| speed | `cruise_speed` | `engine_power_full_store` |
| drag | `zero_lift_drag` | itself (held by the solver) |
| altitude (stored energy) | `ceiling` (the store, J/kg) | `max_energy` |
| power / climb | `climb_time` (empty to full, coasting from cruise: store over climb power) | `coast_exchange_scale` (and the engine power below the knee) |
| dive | `dive_acceleration` (boost from cruise: store into speed) | `dive_exchange_scale` |
| zoom | `zoom_acceleration` (reverse from cruise: speed into store) | `climb_exchange_scale` |
| armament | `hit_damage`, `fire_rate`, `muzzle_speed`, `bullet_spread`, `gun_range` | `hit_damage`, `fire_rate`, `bullet_life`, `bullet_drag_scale`, `bullet_spread_scale` |
| armour | `max_shield`, `head_on_fraction`, `head_on_angle`, recharge | the shield and the head-on aspect |
| size | `hit_radius` | itself |

**What carries over.** The store is the altitude stand-in: boost is a dive (store into
speed), reverse is a zoom (speed into store), and the engine charges it like a climb. An
energy fighter's larger store and stronger exchange are its height advantage, and its
higher cruise is its level speed. Wing loading, drag and power map directly.

**What does not.** There is no altitude sanctuary: in 2D nobody is above anybody, and every
ship is reachable by every other at the same "height". The energy fighter's initiative can
come only from speed, its kinetic energy and its store, so the speed gap has to do the work
that height did. Nor is there a vertical manoeuvre (the high yo-yo, the zoom away): the
store's exchange stands in for it, but a zooming ship slows in the plane rather than
climbing out of reach.

### The conditions for the paradigm

(a) **A speed gap the turn fighter can neither close nor escape.** The energy fighter
extending at cruise must open the range on a turn fighter chasing at its best, and must
catch a turn fighter running. Both ships can boost from the store (the P-51D's straight
boost from cruise peaks at 2.4 times its cruise), so the gap is an energy gap: cruise and
store together. The turn fighter must not be able to chase down an extending energy fighter
by emptying its store; that needs the energy fighter's cruise above the turn fighter's and
its store at least as large.

(b) **No one-pass kills.** If a single head-on pass kills, the first to fire wins and
nothing else matters: neither doctrine has time to act. Part 1's numbers (below) give the
current set's lethality and a proposed target.

(c) **Room to extend.** The energy fighter must open beyond the turn fighter's gun range,
turn round (its turn circle at cruise is 2 v / omega across, about 1.2 km for the P-51D at
its sustained turn) and come back, without meeting the boundary. The arena's size and the
boundary damage (5 per s plus 0.05 per s per metre of depth) are the balance knobs: a
small arena or a harsh boundary favours the turn fighter, a large one the energy fighter.

**If one fails.** Without (a) the energy fighter cannot disengage: it is a worse turn
fighter and loses. Without (b) the fight is decided in the first merge, by spawn geometry
and aim, and the types differ only in their guns. Without (c) the extension ends at the
wall and becomes a turning fight the energy fighter loses.

## 3. Gun lethality today

`bench/gun_lethality.py` (records and tables in `docs/rewrite/sections/gun-study/`): 1v1
duels of the scripted agent (its real-scale deathmatch set) on both sides, real-scale
deathmatch at one ship a side, no fields, a 3,488 m arena (the 1v1 size by the fleet
presets' density rule), 512 duels a matchup. Each duel counts its first kill only, from a
neutral spawn at full shields (a later kill starts from unequal shields).

- **The merge**: the first tick both ships are within the pair's longer gun range with
  each nose within 30 degrees of the line to the other (closing head-on). It ends when
  either nose is more than 90 degrees off that line (they have passed) or the range opens
  past 1.2 gun ranges; a kill up to 2 s after that is a first-merge kill. A trade is both
  ships dying on one tick (it counts as a first-merge kill when it is one).
- **Time to kill (card)**: from full shield at 100% hits, (hits needed - 1) / fire rate,
  the hits needed being the shield over the damage times the aspect (the target's head-on
  fraction head-on, 1 from the tail). **Measured**: from the victim's first hit taken to
  its death (median).
- **Turn**: 180 degrees at the sustained turn rate, and at the instantaneous rate
  (a_max / corner speed). **Pass**: the time a head-on pass spends in gun range, gun range
  over twice the cruise.
- **P(win | first hit)**: of the duels one ship won (trades out), the share the first ship
  to land a hit won.

Standard errors are binomial; at 512 duels a share near 0.97 has ±0.01, near 0.5 ±0.02-0.03.

| Type | Head-on TTK (s) | Tail TTK (s) | 180° sustained (s) | 180° instant (s) | Head-on TTK / 180° instant | Pass (s) | First-merge kills | Trades | P(win \| first hit) | Measured TTK (s) | Hit fraction |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P-51D | 0.87 | 0.27 | 11.5 | 5.1 | 0.17 | 1.8 | 0.97 ± 0.01 | 0.32 ± 0.02 | 0.94 ± 0.02 | 0.70 | 0.39 |
| Heavy | 1.30 | 0.30 | 11.1 | 6.5 | 0.20 | 1.6 | 0.98 ± 0.01 | 0.36 ± 0.02 | 0.97 ± 0.01 | 1.00 | 0.51 |
| P-47D | 1.13 | 0.33 | 12.5 | 5.6 | 0.20 | 1.9 | 0.98 ± 0.01 | 0.41 ± 0.02 | 0.93 ± 0.02 | 0.90 | 0.48 |
| P-38L | 1.07 | 0.27 | 10.8 | 6.2 | 0.17 | 2.0 | 0.97 ± 0.01 | 0.38 ± 0.02 | 0.94 ± 0.02 | 0.83 | 0.47 |
| Spitfire IX | 0.90 | 0.20 | 9.2 | 4.7 | 0.19 | 2.0 | 0.97 ± 0.01 | 0.29 ± 0.02 | 0.95 ± 0.01 | 0.77 | 0.38 |
| Spitfire XIV | 0.90 | 0.20 | 9.7 | 4.9 | 0.18 | 1.9 | 0.97 ± 0.01 | 0.34 ± 0.02 | 0.92 ± 0.02 | 0.77 | 0.39 |
| Tempest V | 1.07 | 0.27 | 11.5 | 5.1 | 0.21 | 1.7 | 0.97 ± 0.01 | 0.36 ± 0.02 | 0.92 ± 0.02 | 0.83 | 0.45 |
| Mosquito | 0.90 | 0.20 | 12.0 | 6.4 | 0.14 | 2.0 | 0.96 ± 0.01 | 0.31 ± 0.02 | 0.97 ± 0.01 | 0.77 | 0.37 |
| Fw 190A | 1.00 | 0.30 | 11.5 | 6.4 | 0.16 | 1.8 | 0.96 ± 0.01 | 0.28 ± 0.02 | 0.93 ± 0.01 | 0.80 | 0.44 |
| Fw 190D | 1.00 | 0.30 | 11.3 | 6.4 | 0.16 | 1.8 | 0.96 ± 0.01 | 0.29 ± 0.02 | 0.95 ± 0.01 | 0.80 | 0.45 |
| Bf 109K | 0.90 | 0.20 | 10.2 | 5.7 | 0.16 | 1.2 | 0.98 ± 0.01 | 0.37 ± 0.02 | 0.94 ± 0.01 | 0.63 | 0.51 |
| Zero | 0.80 | 0.20 | 8.0 | 4.5 | 0.18 | 2.0 | 0.98 ± 0.01 | 0.30 ± 0.02 | 0.94 ± 0.01 | 0.67 | 0.37 |
| Me 262 | 0.90 | 0.20 | 12.8 | 6.7 | 0.13 | 1.0 | 0.98 ± 0.01 | 0.43 ± 0.02 | 0.96 ± 0.01 | 0.60 | 0.51 |
| Ki-84 | 1.00 | 0.27 | 9.6 | 5.2 | 0.19 | 1.7 | 0.99 ± 0.01 | 0.39 ± 0.02 | 0.95 ± 0.01 | 0.77 | 0.46 |
| Yak-3 | 0.93 | 0.27 | 9.2 | 5.2 | 0.18 | 1.8 | 0.97 ± 0.01 | 0.30 ± 0.02 | 0.88 ± 0.02 | 0.73 | 0.42 |
| Corsair | 1.07 | 0.27 | 12.2 | 5.3 | 0.20 | 1.9 | 0.97 ± 0.01 | 0.38 ± 0.02 | 0.97 ± 0.01 | 0.87 | 0.44 |
| La-7 | 1.00 | 0.30 | 9.5 | 5.2 | 0.19 | 1.7 | 0.97 ± 0.01 | 0.33 ± 0.02 | 0.92 ± 0.02 | 0.80 | 0.45 |
| Me 163 | 0.80 | 0.20 | 9.3 | 5.4 | 0.15 | 0.9 | 0.99 ± 0.00 | 0.27 ± 0.02 | 0.88 ± 0.02 | 0.50 | 0.56 |
| Hellcat | 1.20 | 0.33 | 11.7 | 5.1 | 0.24 | 2.0 | 0.96 ± 0.01 | 0.37 ± 0.02 | 0.93 ± 0.02 | 0.97 | 0.47 |
| Hammerhead | 1.40 | 0.40 | 15.2 | 6.8 | 0.21 | 1.1 | 0.25 ± 0.02 | 0.05 ± 0.01 | 0.69 ± 0.03 | 14.3 | 0.52 |
| Wren | 0.93 | 0.27 | 7.9 | 3.3 | 0.29 | 2.1 | 0.99 ± 0.00 | 0.22 ± 0.02 | 0.73 ± 0.03 | 0.80 | 0.40 |
| Anvil | 1.50 | 0.00 | 14.3 | 6.7 | 0.22 | 1.5 | 0.11 ± 0.01 | 0.04 ± 0.01 | 0.62 ± 0.02 | 12.0 | 0.48 |
| Albatross | 1.10 | 0.30 | 11.0 | 4.3 | 0.25 | 1.9 | 0.97 ± 0.01 | 0.31 ± 0.02 | 0.96 ± 0.01 | 0.93 | 0.38 |
| Firecracker | 1.00 | 0.20 | 11.1 | 5.6 | 0.18 | 0.9 | 0.96 ± 0.01 | 0.20 ± 0.02 | 0.88 ± 0.02 | 0.70 | 0.53 |
| Zerstörer | 1.00 | 0.30 | 15.3 | 6.7 | 0.15 | 1.9 | 0.97 ± 0.01 | 0.30 ± 0.02 | 0.97 ± 0.01 | 0.83 | 0.42 |
| Shrike | 1.10 | 0.30 | 13.0 | 5.3 | 0.21 | 1.7 | 0.96 ± 0.01 | 0.31 ± 0.02 | 0.92 ± 0.02 | 0.87 | 0.47 |
| Lantern | 0.93 | 0.27 | 12.5 | 5.1 | 0.18 | 1.6 | 0.97 ± 0.01 | 0.37 ± 0.02 | 0.93 ± 0.02 | 0.73 | 0.40 |
| **vizia** | **4.13** | **0.80** | **11.5** | **4.8** | **0.87** | **1.8** | **0.08 ± 0.01** | **0.02 ± 0.01** | **0.87 ± 0.02** | **20.6** | **0.79** |

In every mirror 96-99% of the damage lands in the front hemisphere (inside the head-on
angle). The Anvil's tail TTK is 0: one hit from behind breaks its shield. vizia's
P(win | first hit) stays at 0.87 although its first merge rarely kills: its 10 m hit
radius (the P-51D's is 5.6 m) makes it an easy target (hit fraction 0.79), so the ship
that lands first tends to win the following head-on passes too (no tail chase: see
"Damage and kills by aspect" below); the P-51D with vizia's guns alone gives 0.59 (below).

Cross pairs (the first type's score; the head-on race is decided by the cards):

| Pair | First type wins | First-merge kills | P(win \| first hit) |
|---|---|---|---|
| vizia v P-51D | 0.02 ± 0.01 | 0.94 ± 0.01 | 0.89 ± 0.02 |
| vizia v Spitfire XIV | 0.01 ± 0.00 | 0.94 ± 0.01 | 0.97 ± 0.01 |
| Me 262 v Zero (fast v nimble) | 0.01 ± 0.00 | 0.94 ± 0.01 | 0.99 ± 0.01 |
| P-47D v Spitfire IX | 0.92 ± 0.01 | 0.95 ± 0.01 | 0.28 ± 0.02 |
| Fw 190A v Yak-3 | 0.52 ± 0.03 | 0.95 ± 0.01 | 0.53 ± 0.03 |
| P-51D v Spitfire XIV | 0.63 ± 0.03 | 0.95 ± 0.01 | 0.74 ± 0.03 |

**The hypothesis holds.** In 25 of the 28 mirrors the first kill falls in the first
head-on merge 96-99% of the time, a fifth to two fifths of the duels are trades, and the
first ship to land a hit wins 88-97% of the rest. The first kill comes about 22 s after
the spawn (the time to the merge), and from the first hit it takes 0.5-1.0 s, less than a
pass. The fast and nimble types are no exception (Me 262, Zero, Wren): speed and turn never
come into it. In a cross pair the pass is a race the cards decide: the Zero, with 75 m
more gun range, beats the Me 262 in 99% of duels; the P-47D, landing second but with 30
more shield, beats the Spitfire IX in 92% (P(win | first hit) 0.28: the first hit there
is the Spitfire's longer range, and it loses the race anyway).

**What drives it.** The head-on budget: the hits a head-on kill needs, the shield over the
damage times the head-on fraction, over the fire rate, against the time a pass spends in
gun range (the gun range over the closing speed). The balancing set every type within 5%
of 360 damage per second, so every type kills head-on in 0.8-1.5 s at full hits, inside a
1-2 s pass. The variants of the P-51D mirror (one handle at a time, 512 duels each) show a
threshold:

| P-51D mirror variant | Head-on TTK (s) | Head-on TTK / pass | First-merge kills | P(win \| first hit) | Measured TTK (s) | First kill (s, median) |
|---|---|---|---|---|---|---|
| as is | 0.87 | 0.48 | 0.97 ± 0.01 | 0.94 ± 0.02 | 0.70 | 23 |
| head-on angle x 2 | 0.87 | 0.48 | 0.97 ± 0.01 | 0.94 ± 0.02 | 0.70 | 23 |
| damage x 1/2 (or shield x 2, or head-on fraction 0.15) | 1.80 | 0.99 | 0.97 ± 0.01 | 0.93 ± 0.02 | 1.40 | 23 |
| damage x 1/3 (or head-on fraction 0.1) | 2.73 | 1.50 | 0.04 ± 0.01 | 0.62 ± 0.03 | 10.6 | 35 |
| vizia's guns (damage x 1/3, head-on fraction 0.2) | 4.13 | 2.27 | 0.02 ± 0.01 | 0.59 ± 0.03 | 30.3 | 54 |
| damage x 1/6 | 5.53 | 3.04 | 0.00 ± 0.00 | 0.56 ± 0.03 | 50.1 | 72 |
| damage x 1/3, head-on fraction 0.1 | 8.27 | 4.54 | 0.04 ± 0.02 | 0.71 ± 0.06 | 62.4 | 80 (83% time out at 90 s) |

The damage, the shield and the head-on fraction act only through the head-on hits a kill
needs (the halved damage, the doubled shield and the halved head-on fraction gave the same
duels, seed for seed). The head-on angle does nothing here: the hits that matter come
within a few degrees of the nose. The switch lies between a head-on TTK of 1.0 and 1.5
passes: below it one pass kills; above it the first merge kills 2-4% and P(win | first
hit) falls toward 0.6. Past about 2 passes the fight stretches (the recharge restores a
shield that is left alone for 6 s), and at 4.5 passes most duels time out.

**A proposed target.**

- **Head-on TTK (card, 100% hits) between 1.5 and 2.5 head-on passes** (gun range over the
  closing speed), which at the P-51D's numbers is 2.7-4.5 s and 0.5-0.9 of its
  instantaneous 180-degree turn: one pass cannot kill, a turn back does not give the other
  ship a second free pass.
- **Tail TTK at most a fifth of the instantaneous 180-degree turn** (about 1 s), so a
  ship that gets behind kills before its target can reverse: position, not the merge,
  decides.
- Together they set the head-on fraction near 0.2 to 0.3 (head-on over tail TTK is about
  one over it) and the damage per second near 100 to 130 against a shield of 100. vizia
  already sits inside (4.1 s, 2.3 passes; tail 0.8 s); damage x 1/3 at the head-on
  fraction 0.3 sits at the target's short end (2.7 s, 1.5 passes) with fights about a
  third shorter than vizia's (first kill 35 s against 54 s).

These are the scripted agent's duels on one type's card; the threshold should be checked
on the other speeds and gun ranges (the pass time scales with them) and with learned
agents before it becomes a rule.

### Damage and kills by aspect

The aspect statistics (`bnb.contracts.ASPECTS`: head-on under 45 degrees off the target's
nose, tail over 135, the flank between; a first kill's class is its killing hit's) for
the same duels, 512 each (`docs/rewrite/sections/gun-study/aspect/`; the Yongma and Jujak
cards of section 4, s = 1, as temporary cards in the bench):

| Duels | Damage head-on / flank / tail | First kills head-on / flank / tail (n) | First type wins |
|---|---|---|---|
| P-51D mirror | 0.99 / 0.01 / 0.00 | 0.99 / 0.01 / 0.00 (657) | |
| P-51D with vizia's guns | 0.99 / 0.01 / 0.00 | 0.99 / 0.01 / 0.00 (477) | |
| vizia mirror | 0.97 / 0.03 / 0.00 | 0.98 / 0.02 / 0.00 (501) | |
| Yongma mirror | 0.95 / 0.05 / 0.00 | 0.94 / 0.06 / 0.00 (498) | |
| Jujak mirror | 0.98 / 0.02 / 0.00 | 0.99 / 0.01 / 0.00 (514) | |
| Yongma v Jujak | 0.83 / 0.17 / 0.00 (Yongma's 0.98 / 0.02, Jujak's 0.76 / 0.24) | 0.60 / 0.40 / 0.00 (505) | 0.08 ± 0.01 |

**No duel has a tail shot.** Not one hit of these 4,608 duels comes from behind; an
independent check of 6,793 hits in 64 vizia duels finds none more than about 72 degrees
off the nose. The long fights of the lethal-target cards (vizia's median 21 s from first
hit to kill) are repeated head-on passes, not tail chases: the scripted agent turns to
face its target, and a target that turns to face back meets it nose to nose. The flank
appears only where the turn rates differ: Jujak, turning inside Yongma, takes a quarter of
its damage and two fifths of the kills from the flank, and wins 92% of the duels. With
the speed group at s = 2 nothing changes (0.08 ± 0.01): without a doctrine Yongma is the
worse turn fighter whatever its speed, as condition (a)'s failure case says. Both valid
strategies the author wants, the flank snap and the tail track, need doctrine (or learned
play) to appear; these statistics are now in every training run (`combat/damage_share/`,
`combat/kill_share/`), the match statistics and this bench to track them.

## 4. Proposed starting points: two variants of vizia

Handle deltas from `vizia` (src/bnb/game/ships/real-scale/vizia.toml), scaled from the
real split of section 1 (`docs/rewrite/sections/gun-study/archetypes_solve.py`): a mapped
handle is vizia's times r^(s/2) for Jujak and r^(-s/2) for Yongma, r the turn-over-energy
ratio of its real quantity, so vizia sits at the geometric midpoint and the pair's split
is the real split to the power s.

- **s = 1 for flight**: the real group split. It is about the split of the famous pairs'
  milder half (the Fw 190A over the Spitfire IX turns 0.81 times as fast; the groups 0.78);
  the pairs' wider split (the P-47D over the Zero, 0.64) is the room the tuning loop has.
- **s = 1/2 for armour and guns**: the full split (shield and damage each about 1.19 times
  either way) would leave Jujak's head-on kill of Yongma at 3.3 passes, outside the
  lethality target; at a half it is 2.7. The shield and the damage scale together, so each
  mirror keeps vizia's time to kill exactly.
- **Not used**: the ceiling (the store), the zoom and the g limit, which the archetypes
  share; the nose reversal (roll: only the Fw 190s separate); the half-lift drag (no real
  basis: the energy fighter keeps its energy in gentle turns through its lower zero-lift
  drag, here). The fire rate stays vizia's 15 Hz.

| Handle (from the real quantity) | vizia | Yongma (energy) | Jujak (turn) |
|---|---|---|---|
| stall speed (stall speed, r 0.86) | 44.7 m/s | 48.1 (x 1.075) | 41.6 (x 0.930) |
| sustained turn rate (sustained turn, 1.29) | 15.7 °/s | 13.8 (x 0.881) | 17.8 (x 1.136) |
| low-speed acceleration (1.54) | 2.73 m/s² | 2.20 (x 0.806) | 3.38 (x 1.241) |
| climb time (climb rate, 1.24) | 600 s | 667 (x 1.112: charges slower) | 540 (x 0.899) |
| cruise speed (top speed at altitude, 0.94) | 162.7 m/s | 167.5 (x 1.029) | 158.1 (x 0.972) |
| zero-lift drag (W/S over C_D0, 0.73) | 4.97e-5 /m | 4.25e-5 (x 0.855) | 5.81e-5 (x 1.169) |
| dive acceleration (dive limit, 0.88) | 10.9 m/s² | 11.6 (x 1.066) | 10.2 (x 0.938) |
| hit radius (half span, 0.94) | 10 m | 10.33 (x 1.033) | 9.68 (x 0.968) |
| max shield (survivability by the set's rule, 0.71; s = 1/2) | 100 | 109.2 (x 1.092) | 91.6 (x 0.916) |
| hit damage (weight of fire, 0.70; s = 1/2) | 8.0 | 8.76 (x 1.094) | 7.31 (x 0.914) |

Both cards solve (worst flown error 0; `archetypes_solved.json`). What they fly, and the
time-to-kill budget (card, 100% hits; a pass is the longer gun range over the two cruises):

| | vizia | Yongma | Jujak |
|---|---|---|---|
| cruise (m/s) | 162.7 | 167.4 | 158.1 |
| top speed, a straight boost from cruise (m/s) | 389 | 422 | 350 |
| corner speed (m/s) | 134 | 144 | 125 |
| sustained turn, sticks 1-3 (°/s) | 10.1, 12.9, 15.7 | 8.7, 11.2, 13.8 | 11.5, 14.9, 17.8 |
| instantaneous turn (°/s) | 37.7 | 35.1 | 40.6 |
| 180° sustained / instantaneous (s) | 11.5 / 4.8 | 13.0 / 5.1 | 10.1 / 4.4 |
| turn circle at cruise, sustained (m) | 1,187 | 1,388 | 1,016 |
| store (J/kg) | 124,391 | the same | the same |

| Shooter on target | Head-on TTK (s) | Head-on passes | Tail TTK (s) | Tail TTK / target's 180° instantaneous |
|---|---|---|---|---|
| Yongma on Yongma | 4.13 | 2.36 | 0.80 | 0.16 |
| Jujak on Jujak | 4.13 | 2.23 | 0.80 | 0.18 |
| Yongma on Jujak | 3.47 | 1.92 | 0.67 | 0.15 |
| Jujak on Yongma | 4.93 | 2.74 | 0.93 | 0.18 |

Every mirror keeps vizia's lethality inside the target, and every tail kill is under a
fifth of the target's turn. The cross pair is lopsided head-on by design: Yongma wins a
head-on exchange (1.9 against 2.7 passes), as a P-47 did against a Zero, so Jujak's
doctrine is to deny the head-on and take the tail.

The intended matchup: **Yongma wins if disciplined, Jujak wins if it drags the fight in.**
Jujak turns 29% faster (17.8 against 13.8 °/s sustained, 16% instantaneous) in a 27%
smaller circle, and charges its store 24% faster; Yongma cruises only 6% faster, as the
real pairs did in level flight, and its escape is the dive: its boost from cruise peaks at
422 m/s against 350 (1.21 times), from the lower drag and the faster exchange, with equal
stores (condition (a) is an energy-retention gap, not a cruise gap). Yongma's turn circle,
1.4 km at cruise, sets the room it needs to extend and come back (condition (c)).

**If the doctrine is not distinct enough** (the loop's step 3 finds Yongma unable to
disengage), the first knob is the speed group at s = 2 (cruise, zero-lift drag and dive
acceleration), standing in for the vertical that 2D lacks: cruise 172.3 against 153.6 m/s
(12%) and boost tops 448 against 317 m/s (1.41 times), the turn and the guns unchanged
(also solved, `archetypes_solved.json`, "s1-speed2"). The earlier draft's deltas (a 21%
cruise gap, a 37% turn gap, a 20% larger store) were about s = 3 on the cruise and 1.2 on
the turn, with a store the real types do not support.

## 5. The scripted agent

**What exists.** One agent with a doctrine per ship type
(`bnb.controllers.scripted_doctrine.Doctrine`, [Doctrines](../../scripted-agent.md#doctrines)):
a table the agent reads per own ship, so a mixed team flies each type by its own. A
doctrine's weight against its target is a strength times a logistic of an advantage (the
own-over-target ratios of the turn, the climb, the cruise and the gun range, each to an
exponent), and its elements are extend (with an energy and store rule for re-engaging),
slash, bleed, lag and deny. The tuned doctrines are in
`src/bnb/controllers/scripted_doctrines.json`; the tuning loop is below and in
`docs/rewrite/sections/doctrines/`.

**What each doctrine needs, beyond it.** One agent, with per-type doctrine parameters:

- *Energy (Yongma)*: (1) extend far enough: the extension ends at about one gun range
  today, inside the turn fighter's reach once it has turned; it should end at a distance
  in the opponent's turn circles or in seconds of its 180-degree turn, so the re-attack
  starts from outside its guns; (2) a store floor: do not engage below a share of the
  store (the altitude it fights from), coast or zoom to refill first; (3) the slash: boost
  into the pass, fire within the pass, never reverse in a fight; (4) refuse the head-on:
  turn off the merge line when the pass is not favourable (the advantage includes the
  guns).
- *Turn (Jujak)*: (1) drag the fight in: slow toward corner speed (reverse) when the
  enemy is close and turning; (2) do not chase: past a few gun ranges an extending energy
  fighter cannot be caught, so turn to face it and keep the store for the next pass;
  (3) the break: turn into an attacker at the moment its pass begins (its range over the
  closure), to deny the tail shot.

Each of these is a parameter (a logistic or a distance) the search can tune; the defaults
keep today's agent (strength 0).

**A capped tune loop.** Four rounds, each the same four steps, stopping early if the
targets below are met:

1. *Flight models*: solve the two cards; check the speed gap (cruise and the store), the
   turn gap (sustained and instantaneous) and the extension (condition (c)) against the
   arena.
2. *Doctrine*: a per-type parameter search (`bench/scripted_search.py` with a player per
   type) of each type's doctrine against the other type flown by its current doctrine,
   a fixed budget a round.
3. *Balance check*: duels and 2v2s, adaptive (a target standard error), in four
   combinations: each type with its doctrine and with the doctrine off. The intended
   matchup is met when both doctrines on is near even (0.45 to 0.55), Yongma without its
   doctrine loses to Jujak (at most 0.40), and Jujak chasing (its doctrine off) loses to a
   disciplined Yongma (at most 0.40); and `bench/gun_lethality.py` holds the lethality
   target in both the mirror and the cross duel.
4. *Adjust*: one or two handles (the speed gap, the turn gap, the store, the arena) by the
   round's results, from the round's best (a losing change never carries on).

After the fourth round the author judges what is left.

## 6. Names

The author's choice (2026-11-12), from Korean mythology: **Jujak** (주작), the vermilion bird
of the south, for the turn fighter, and **Yongma** (용마), the dragon-horse, for the energy
fighter.
