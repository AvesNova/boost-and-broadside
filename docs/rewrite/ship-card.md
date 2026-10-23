# Ship card

What defines a ship type, what is derived and shown, and how a design compares with a
real fighter. Status: **draft under discussion**; see `decisions.md` for what is agreed.

*Note 2026-10-23: the flight handles below describe the old model. The new flight model
(`flight-model.md`) changes them; the card is rewritten in the physics-first plan
(step 5).*

Units: lengths in **m** (the world unit, formerly "px"; rendering applies a pixels-per-m
scale), time in **s**, angles in **deg**.

## Flight: per-type handles (10)

Verified full rank on the 10 remaining per-type flight constants, after the
reverse = -boost tie and the global shape rules below.

| Handle | Meaning | Pilot analog |
|---|---|---|
| `top_speed` | energy-limited boost terminal, straight | top speed |
| `cruise_speed` | coast terminal, straight | cruise |
| `boost_accel` | boost acceleration from rest | power-to-weight |
| `recharge_time` | empty to full energy store by regeneration | climb rate |
| `tank_speed` | speed a full store buys from rest | altitude bank / dive potential |
| `min_turn_radius` (`sharp_radius`) | lift-limited radius below corner speed | min turn radius |
| `corner_speed` | slowest speed at which the g-limit caps the sharp turn | corner speed |
| `sustained_circle_time` (`360 / sharp_sustained_rate`) | 360 deg at zero net energy loss | sustained turn time |
| `onset_time` | straight to 90% of a commanded slip | roll rate |
| `stall_speed` | below this, no lift | stall speed |

`stall_speed` on the card means **the speed below which a ship can no longer be
controlled** (today that role is played by `fade_speed`, 35 m/s). The current
`min_speed = 1` is only a numerical floor against divide-by-small-number problems; it
becomes a global internal constant, not a design handle.

## Flight: global shape rules (shared by every ship)

- **Turn levels are stick thirds:** 1/3, 2/3 and 3/3 of the stall angle, constant
  stick, the same for every throttle. 1/3 is today's normal turn, 3/3 today's sharp
  turn, 2/3 is new.
- **Design rule: max sustained rate (boosting) sits at 2/3 stick.** This fixes the
  drag curve's shape parameter, so the normal-slip drag point is derived rather than
  set. The shipped ship already satisfies it (optimum at slip fraction ~0.67).
- **One g-limit for every turn level** (`max_lateral_accel`), like a real plane. The
  hard-coded half-g cap on today's normal turn is removed.
- Slip fade behavior tied to the stall (control-loss) speed by a fixed rule.
- Stall angle (gun offset in a sharp turn); first candidate to promote to a per-type
  handle.
- Nose slew rate derived from onset time by a fixed ratio.

All of these are defaults on the card; the engine treats every flight constant as a
per-ship array, so any rule can be promoted to a per-type handle later.

## Flight: derived and shown

- `max_turn_rate` = corner_speed / min_turn_radius.
- `instant_circle_time` = 2 pi min_turn_radius / corner_speed.
- `unlimited_top_speed`, coast speeds in turns, normal-turn radius and rates.

## Combat: per-type handles

| Handle | Pilot analog |
|---|---|
| damage per shot, fire interval (shown as DPS) | weight of fire |
| bullet speed | muzzle velocity |
| range (bullet lifetime derived from it, given drag) | effective range / convergence |
| spread | dispersion |
| energy per shot (shown as firing endurance) | ammunition / firing time |
| shield capacity | durability |
| shield recharge rate and delay | damage control |
| head-on damage fraction | frontal armor |
| collision radius | size / target area |

## Similarity ratios vs a P-51D

Dimensionless ratios, so the comparison does not depend on absolute scale. "Ours" is
the current shipped ship (`config/defaults.py: STARTING_HANDLING`).

| Ratio | Ours | P-51D (est.) | Ours / P-51 |
|---|---:|---:|---:|
| top speed / low-speed limit (fade for ours, stall for P-51) | 3.9 | 4.4 | 0.9 |
| corner speed / top speed | 0.74 | 0.69 | 1.1 |
| sustained circle / instantaneous circle | 1.36 | 1.85 | 0.7 |
| ship radius / min turn radius | 0.27 | 0.024 | **11** |
| bullet speed / top speed | 3.7 | 4.5 | 0.8 |
| gun range / min turn radius | 11.4 | 1.1 | **10.5** |
| gun range / ship radius | 42 | 44 | 0.9 |
| min turn radius / top speed (s) | 0.27 | 1.18 | **0.2** |
| bullet flight time to range / sustained circle | 0.32 | 0.014 | **22** |
| degrees a target turns during bullet flight | 114 | 5 | **22** |

### Reading

- Speeds, gun reach measured in ship sizes, and bullet speed are all WW2-like.
- **The outlier is turn radius:** about 10x tighter, relative to ship size and gun
  range, than a WW2 fighter. That is why the guns *feel* long: they are not long for
  the ship's size; the turns are tight.
- **Consequence:** while a bullet flies to max range, the target can turn about 114
  deg (P-51: about 5 deg). Dodging and lead prediction dominate far more than in WW2
  gunnery. This may well be the fun of the game, but it is now an explicit design
  number rather than an accident.

### P-51D inputs and confidence

| Quantity | Value used | Source and confidence |
|---|---|---|
| Top speed | 195 m/s (437 mph at 25,000 ft) | Spec sheets; reliable |
| Stall speed | 44.7 m/s (100 mph) | Spec sheet; reliable |
| Half span (ship radius analog) | 5.64 m (37 ft span) | Spec sheet; reliable |
| Corner speed | 134 m/s (~260 kt at 8 g) | Enthusiast forum citing the V-g diagram; medium |
| Min turn radius | 231 m | Derived: r = v_c^2 / sqrt((8g)^2 - g^2) at the corner above; cross-checks with wing loading ~190 kg/m^2 and C_L,max ~1.5 (~207 m); medium |
| Sustained circle | ~20 s | Simulator tests (DCS: 17.5 deg/s at 500 ft; Aces High: 18.6 deg/s); medium |
| Muzzle velocity | ~880 m/s (.50 M2; ~870-930 cited) | Secondhand; medium |
| Gun range | 250 m (convergence 250-300 yd typical) | DCS manual, pilot accounts; medium |
| Armament | 6 x .50 M2, 1,880 rounds | Spec sheets; reliable |

Gaps to fill if the comparison matters more: roll rate, climb rate, firing endurance
and weight of fire as ratios. Primary sources would be NACA flight test reports and
the USAAF Pilot's Flight Operating Instructions.

Sources:
- https://www.warbirdsresourcegroup.org/URG/p51mustang_specifications.html
- https://forum.dcs.world/topic/101588-turn-performance-and-lift-coefficient/
- https://www.hitechcreations.com/wiki/index.php/P-51D
- https://www.digitalcombatsimulator.com/upload/iblock/aa2/DCS-P-51D_Flight_Manual_EN.pdf
- https://military-history.fandom.com/wiki/Gun_harmonisation
- https://www.gt40s.com/threads/p-51d.35582/latest

## What-if: ship radius and gun range divided by 3 and by 10

Everything else unchanged (bullet speed 500 m/s with the same drag, top speed 136 m/s,
min turn radius 37 m, sustained circle 3.16 s, fire rate 10/s).

| Scale | Ship radius | Gun range | radius / min turn radius | range / min turn radius | Bullet flight time | Target turns during flight | Lead (ship radii) | Shots per head-on pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| today | 10 m | 421 m | 0.27 | 11.4 | 1.00 s | 114 deg | 13.6 | 15.5 |
| / 3 | 3.3 m | 140 m | 0.09 | 3.8 | 0.30 s | 34 deg | 12.1 | 5.2 |
| / 10 | 1.0 m | 42 m | 0.027 | 1.1 | 0.086 s | 10 deg | 11.6 | 1.6 |
| P-51D | 5.6 m | 250 m | 0.024 | 1.1 | 0.28 s | 5 deg | 9.8 | ~48 rounds |

Only ratios matter: shrinking ships and guns by 10 is the same game as growing speeds,
turn radii, zones, fields and the map by 10.

## Proposal: halve ship size and gun range, double the other lengths

Speeds unchanged (top 136 m/s, corner 100 m/s, bullet 500 m/s with the same drag).
Doubling lengths at the same speeds doubles every turn time: corner speed is
unchanged while radius and lateral acceleration scale together.

| | Today | Proposal | P-51D |
|---|---:|---:|---:|
| Ship radius / min turn radius | 0.27 | 0.068 | 0.024 |
| Gun range / min turn radius | 11.4 | 2.8 | 1.1 |
| Gun range / ship radius | 42 | 42 | 44 |
| Instantaneous circle | 2.3 s | 4.6 s | ~11 s |
| Sustained circle | 3.2 s | 6.3 s | ~20 s |
| Bullet flight time to range | 1.00 s | 0.46 s | 0.28 s |
| Target turns during bullet flight | 114 deg | 26 deg | 5 deg |
| Lead (ship radii) | 13.6 | 12.5 | 9.8 |
| Shots per head-on pass (10/s) | 15.5 | 7.7 | ~48 rounds |
| Min turn radius / top speed | 0.27 s | 0.54 s | 1.18 s |

If speeds were doubled too ("double everything else"), the geometry would match the
proposal but every turn time would stay as fast as today.
