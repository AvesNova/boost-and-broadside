# Ship card

What defines a ship type, what is derived and shown, and how a design compares with a
real fighter. Status: **draft under discussion**; see `decisions.md` for what is agreed.

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

Open: the current ship has `stall_speed = 1` (effectively off) and `fade_speed = 35`,
so the low-speed limit players actually feel is the fade speed. Consider making the
per-type handle a "min control speed" (fade) and making stall speed global or derived.

## Flight: global shape rules (shared by every ship)

- Normal-turn lift as a fixed fraction of max lift.
- Normal-turn lateral cap as a fraction of `max_lateral_accel` (hard-coded 0.5 in
  `slip_command` today; make it explicit).
- Normal-slip drag tied to stall drag by one induced-drag relation (curve shape
  unchanged).
- Slip fade speed as a fixed multiple of stall speed (see the open note above).
- Stall angle (gun offset in a sharp turn).
- Nose slew rate derived from onset time by a fixed ratio.

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
