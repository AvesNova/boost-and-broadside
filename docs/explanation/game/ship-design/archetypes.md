---
covers: [bench/gun_lethality.py, src/bnb/controllers/scripted_combat.py]
---

# Two archetypes: the turn fighter and the energy fighter (draft)

A working spec for the author to edit. It proposes two variants of `vizia`, Jujak (주작,
the vermilion bird; the turn fighter) and Yongma (용마, the dragon-horse; the energy
fighter), and the conditions under which their matchup is decided by doctrine rather than
by who fires first. Nothing here is a ship file yet; every value is a proposal.

## 1. The two fighters of the Second World War

**The turn fighter** (Spitfire, A6M Zero, Yak-3, Ki-43). Light wing loading, so a low stall
speed and a small turn radius; a light airframe, often lightly armoured; modest top speed.
Its doctrine is to slow the fight down and stay in it: draw the opponent into a turning
contest at low speed, where its turn rate and radius bring its guns to bear first.

**The energy fighter** (P-47, Fw 190, F4U Corsair, P-51 at altitude). High wing loading, a
heavy airframe, a strong engine and low drag, so a high top speed, a fast dive and a long
zoom; heavy armament and armour. It turns worse at any speed it shares with the turn
fighter. Its doctrine keeps the fight fast: slash (a high-speed firing pass), extend (fly
away faster than the opponent can follow), climb or turn wide, and re-engage. Its real
advantage is initiative: it chooses when a fight happens and when it ends. Altitude gave it
both safety (a turn fighter below could not reach it) and stored energy (a dive converts
height into speed for the next pass).

**The trade-offs that separate them.**

| Quantity | Turn fighter | Energy fighter | What it buys |
|---|---|---|---|
| wing loading | low | high | low: turn radius and rate, low stall speed; high: less drag at speed, a faster dive |
| mass | low | high | high: dives faster, keeps speed in a zoom, carries armour and guns |
| power | moderate | high (power-to-weight) | climb and sustained speed |
| drag | higher | lower | lower: top speed, energy kept in gentle turns |
| dive and zoom | short | long | the energy fighter's escape and its re-attack |
| armament | lighter, often cannon on centreline | heavy (8 x .50, 4 x 20 mm) | a brief firing pass must be enough |

Each wins on its own terms and loses on the other's: a turn fighter that chases an energy
fighter in a dive is left behind; an energy fighter that turns with a turn fighter is shot
from behind. Doctrine (Thach weave, "never turn with a Zero", boom-and-zoom) is the
discipline of staying on one's own terms.

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
radius (the P-51D's is 5.6 m) makes it an easy target (hit fraction 0.79), so the first
hit is usually the start of a won tail chase; the P-51D with vizia's guns alone gives 0.59
(below).

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

## 4. Proposed starting points: two variants of vizia

Handle deltas from `vizia` (src/bnb/game/ships/real-scale/vizia.toml); every handle not
named is vizia's, including its guns (which meet the lethality target). Both cards solve
(`docs/rewrite/sections/gun-study/variants_solved.json`, worst flown error below 1e-4):

| Handle | vizia | Yongma (energy) | Jujak (turn) |
|---|---|---|---|
| stall speed (m/s) | 44.7 | 50.0 (wing loading +25%) | 40.0 (-20%) |
| cruise speed (m/s) | 162.7 | 182.0 (+12%) | 150.0 (-8%) |
| sustained turn rate (deg/s) | 15.7 | 13.5 | 18.5 |
| half-lift drag | 0.25 | 0.20 (keeps energy in gentle turns) | 0.30 (spreads the cost) |
| zero-lift drag (1/m) | 4.97e-5 | x 0.9 | x 1.1 |
| store (`ceiling`, J/kg) | 124,391 | x 1.2 | x 0.9 |
| spawn energy | 0.4 of the store | x 1.2 (the same share) | x 0.9 |
| climb time (s) | 600 | 600 (the charge rate x 1.2) | 600 (x 0.9) |
| dive acceleration (m/s^2) | 10.9 | x 1.2 | x 0.85 |
| low-speed acceleration (m/s^2) | 2.73 | | x 1.1 |
| nose reversal time (s) | 0.79 | | 0.70 |
| shield | 100 | 120 | 90 |
| hit radius (m) | 10 | 11 | 9 |

What the solved cards fly:

| | vizia | Yongma | Jujak |
|---|---|---|---|
| cruise (m/s) | 162.7 | 182.0 | 150.0 |
| top speed, a straight boost from cruise (m/s) | 389 | 444 | 344 |
| corner speed (m/s) | 134 | 150 | 120 |
| sustained turn, sticks 1-3 (deg/s) | 10.1, 12.9, 15.7 | 9.1, 11.0, 13.5 | 11.2, 15.3, 18.5 |
| instantaneous turn (deg/s) | 37.7 | 33.7 | 42.2 |
| 180° sustained / instantaneous (s) | 11.5 / 4.8 | 13.3 / 5.3 | 9.7 / 4.3 |
| turn circle at cruise, sustained (m) | 1,187 | 1,545 | 929 |
| head-on TTK in the mirror (s) | 4.1 | 4.9 | 3.7 |

The intended matchup: **Yongma wins if disciplined, Jujak wins if it drags the fight in.**
The cruise gap is 21% (182 against 150), about the P-47D's over the Zero at altitude, and
the turn gap 37% (18.5 against 13.5 deg/s). Jujak's boost peaks at 344 m/s, above Yongma's
cruise, so it can chase while its store lasts; the larger store and the faster dive are
what let Yongma out-run that chase (condition (a) is an energy gap, not a speed gap).
Yongma's turn circle, 1.5 km at cruise, sets the room it needs to extend and come back
(condition (c)): within the 1v1 arena's 7 km across, not within the 3.9 km of the 2v2
arena preset. The deltas are a first guess for the loop below to adjust, not a balance.

## 5. The scripted agent

**What exists.** The combat layer (`bnb.controllers.scripted_combat.combat`) already has an
energy doctrine, `energy_fight`, off in every set today (`energy_fight_strength` 0 in
`scripted_real_scale.json`). Its weight against the committed target is
`energy_fight_strength` times a logistic (`energy_fight_in_advantage`) of an advantage, the
product of the own-over-target ratios of the sustained turn rate, the climb power, the
cruise speed and the gun range, each raised to an exponent (by default only the turn, at
-1: the worse turner has the advantage). With that weight the ship extends rather than
turns after a target its nose is far from at close range: the aim point is mirrored away
from the target with weight `extend_in_gun_range` (falling through 1 gun range) times
`extend_in_stall_angle` (rising through 4 stall angles of aim error), the power is mixed
toward boost while extending, and it never reverses (it keeps its speed). The league arena
already plays each scripted player with its own parameter set (`Player.parameters`, in an
arena built with a `doctrine`), so per-type doctrines need no new machinery.

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
keep today's agent exactly (strength 0), as `energy_fight` does.

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
