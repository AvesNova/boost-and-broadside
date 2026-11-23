# Core flight model: measured manoeuvres

Standard manoeuvres flown in both flight models (`GameConfig().flight_model` "v2" and
`GameConfig.core()`), one ship, no fields, coast unless stated, store at half unless stated.
Measured 2026-11-23 on the laptop CPU, one thread, by
`docs/rewrite/sections/turns/measure_core.py` (argument: the card index, 2 Vizia, 3 P-51D in the
card of that day; since G1 the card holds the P-51D alone, index 0, and the script prints the
core column only).
Game units (2x real time).

## Vizia

Unchanged by the three flight changes of 2026-11-23 below: the table is identical, digit for
digit, before and after them.

| vizia, no fields | v2 | core |
|---|---|---|
| top level speed, coast (m/s) | 325 | 325 |
| store after 120 s straight coast (%) | 95.3 | 50.0 |
| corner speed (m/s, g limit meets lift) | 268 | 268 |
| full stick from corner, coast: first second (deg) | 57 | 57 |
| full stick, settled after 30 s: speed (m/s), rate (deg/s), store (%) | 139, 38.6, 51.6 | 111, 25.0, 50.0 |
| sustained stick from corner, coast: first second (deg) | 43 | 43 |
| sustained stick, settled after 30 s: speed (m/s), rate (deg/s), store (%) | 161, 32.2, 77.2 | 161, 32.2, 50.0 |
| fine stick from corner, coast: first second (deg) | 13 | 13 |
| fine stick, settled after 30 s: speed (m/s), rate (deg/s), store (%) | 285, 16.9, 82.0 | 285, 16.9, 50.0 |
| full stick and dive from corner: first second (deg) | 59 | 59 |
| climb from top speed, half store: time to 99% (s), speed then (m/s) | not reached in 60 s (91%) | 1.6, 274 |
| dive from top speed, half store: time to 1% (s), peak speed (m/s) | 24.0, 442 | 1.4, 368 |
| speed lost to 1 s of firing at top speed (m/s, against not firing) | 0.0 | 6.0 |
| speed lost to 1 s of firing at corner speed (m/s, against not firing) | 0.0 | 7.3 |

## P-51D

With its own flight terms (the game terms off: its point mass, the October fit;
docs/rewrite/p51d-model.md). The "v2" column is the P-51D's card under v2's flight model.

| p51d, no fields | v2 | core |
|---|---|---|
| top level speed, coast (m/s) | 335 | 335 |
| store after 120 s straight coast (%) | 50.0 | 50.0 |
| corner speed (m/s, g limit meets lift) | 263 | 263 |
| full stick from corner, coast: first second (deg) | 60 | 58 |
| full stick, settled after 30 s: speed (m/s), rate (deg/s), store (%) | 245, 63.8, 14.6 | 151, 37.3, 50.0 |
| sustained stick from corner, coast: first second (deg) | 40 | 40 |
| sustained stick, settled after 30 s: speed (m/s), rate (deg/s), store (%) | 181, 33.3, 50.0 | 181, 33.3, 50.0 |
| fine stick from corner, coast: first second (deg) | 12 | 12 |
| fine stick, settled after 30 s: speed (m/s), rate (deg/s), store (%) | 290, 15.8, 50.0 | 290, 15.8, 50.0 |
| full stick and dive from corner: first second (deg) | 58 | 58 |
| climb from top speed, half store: time to 99% (s), speed then (m/s) | not reached in 60 s (74%) | 1.7, 280 |
| dive from top speed, half store: time to 1% (s), peak speed (m/s) | 11.8, 684 | 1.4, 381 |
| speed lost to 1 s of firing at top speed (m/s, against not firing) | 0.0 | 5.3 |
| speed lost to 1 s of firing at corner speed (m/s, against not firing) | 0.0 | 6.7 |

In real units the core P-51D flies 375 mph, climbs 3,599 ft/min, and holds 18.6 deg/s at
169 mph (r 233 m) as its best sustained turn: the October fit (`lean/game/tuning.py`
`performance`, a 0.05 m/s grid).

## The turn has precedence over the climb (author, 2026-11-23)

The lift budget (the lift limit, softly capped by the g limit) goes, in order: 1 g held (coast
holds altitude), then the turn (what the stick asks for), then a climb's extra demand gets
what is left. Dive keeps its order (its lower demand is held first and frees lift for the
turn). When the budget cannot hold 1 g plus the turn while coasting, altitude still wins.
Before, the climb's whole demand (about 2 g) was held first, so at low speed full stick and
climb flew nearly straight.

Full stick and climb against full stick and coast, 2 s from the given speed, half store
(`flight_tick`; turn of the flight path, the store's change as altitude, the speed after):

| ship, start speed (m/s, game) | coast | climb, before | climb, after |
|---|---|---|---|
| Vizia, corner 268 | 113.1 deg, +0 m, 189 m/s | 86.9 deg, +367 m, 110 m/s | 98.6 deg, +253 m, 152 m/s |
| Vizia, its settled full-stick speed 111 | 44.9 deg, 0 m, 116 m/s | 0.7 deg, +58 m, 97 m/s | 44.8 deg, +0.9 m, 116 m/s |
| Vizia, 1.15 x stall 103 | 38.6 deg, 0 m, 113 m/s | 0.3 deg, +42 m, 97 m/s | 38.6 deg, +0.2 m, 113 m/s |
| P-51D, corner 263 | 117.7 deg, 0 m, 222 m/s | 81.8 deg, +389 m, 125 m/s * | 117.6 deg, +1.1 m, 222 m/s |
| P-51D, its settled full-stick speed (152; before: 116) | 67.1 deg, 0 m, 154 m/s | 0.6 deg, +59 m, 100 m/s * | 67.1 deg, +0.0 m, 154 m/s |
| P-51D, 1.15 x stall 107 | 39.4 deg, 0 m, 126 m/s | 0.3 deg, +41 m, 100 m/s * | 39.4 deg, +0.0 m, 126 m/s |

\* "Before" is the P-51D with the game terms it had then (Vizia's: drag rise, separation drag,
soft minima): its coasting turn was 106.8 deg from 263 m/s, 42.4 deg from 116 m/s (its settled
speed then) and 35.8 deg from 107 m/s. The "after" column has the point-mass P-51D.

Reading:
- Slow, full stick and climb now turns exactly as full stick and coast; the climb gains only
  what lift is left (almost none). The author's case (turn plus climb flying straight) is gone.
- At corner speed Vizia still climbs (+253 m in 2 s) and turns 99 deg against 113 coasting.
  At the corner Vizia's soft minimum of lift against the g limit (exponent 16) holds the turn
  below the full budget, and the climb takes that slack (about 1 g); speed then bleeds into
  the store and the slower ship turns less over the 2 s. The P-51D's minimum is near hard
  (exponent 400): its turn takes the whole budget and the climb gets nothing.
- A climb's unmet demand is now climb not made; before, it counted as sinking and was paid from
  the store only above the bottom edge.
- Unchanged: coast and dive (bit for bit), and straight climbs at speed (the Vizia table above).

## Flight terms per ship type

The terms v2 shared across types are card columns (`lean.game.card.FLIGHT_TERMS`, v2's names):
`critical_speed` (drag rise), `drag_rise_exponent`, `separation_drag`, `separation_exponent`,
`lift_softmin_exponent`, `thrust_softmin_exponent`, `nose_damping`. Yongma, Jujak, Vizia and
vizia_plain keep v2's values (394 m/s, 12, 1, 8, 16, 2, 0.7). The P-51D has the game terms
off (`tuning.POINT_MASS_TERMS`: no drag rise, no separation drag, soft minima at exponent 400).
Shared still: gravity, the stall angle and induced-drag base constants (each with a per-type
scale), and the numerical floors (speed floor, nose-frequency floor).

Vizia's flight is bit-identical in its own lineup (630 manoeuvres of 90 ticks: every stick,
power, six speeds, five stores) and within 1.2e-6 relative when the lineup also holds the
P-51D (the terms become per-slot tensors there).

## Time scale (sandbox `--time-scale K`)

The game is the real aircraft at 2x real time. `lean.game.tuning.time_scale(K)` rebuilds the
card and the time-dimensioned rule constants at K times real time from their real-time values
(speeds and turn rates x K / 2 against the game's, accelerations and gravity x (K / 2)^2,
power per kg x (K / 2)^3, durations x 2 / K), for the sandbox's process only. Checked
(`test_time_scale_is_the_same_flight_faster_or_slower`): at K = 1, 1.5 and 3, gravity is
K^2 x 9.81 and every type's cruise and corner speed are the game's times K / 2, to 1e-5.

Guns scale too, in whole ticks: the bullet life is capped at what the 11 bullet slots hold
(at 1.5x: 11 ticks, not 15), and the fire interval rounds (at 3x: one shot a tick, 30 a
second, not 45).

## Charts

- P-51D against Vizia, core model, both doghouses on shared axes and Ps colours and both
  envelopes overlaid: `sections/turns/doghouse-p51d-vizia-core.png` (game units) and
  `sections/turns/doghouse-p51d-vizia-core-real.png` (1x real time, with an mph axis, to read
  against published P-51D energy-manoeuvrability diagrams); `-dark` variants.
- Each type's core doghouse (`doghouse-<type>-core[-real][-dark].png` in docs/rewrite/),
  redrawn for the turn-over-climb order: the "full stick, climbing" line now lies on the full
  stick line except near the corner.

  (Drawn by doghouse_compare.py, removed with the Vizia row at G1; `doghouse.py` draws the
  P-51D's.)

## Open for the author's flying

- **The store is crossed in about 1.5 s per half at full exchange rate** (Vizia, half to full
  in 1.6 s from top speed; half to empty in 1.4 s). The exchange rate is 38.6 m/s^2 (about the
  game's 1 g, what a vertical climb costs), and the whole band is worth about +-45 m/s around
  cruise (dive peak 368 m/s from 325). Larger store, slower exchange, or as is.
- **Firing costs 6.0 m/s per second at top speed and 7.3 m/s per second at corner speed**
  (Vizia) with the engine fully cut (`fire_thrust_fraction` 0.0): about 2 to 3% of speed per
  second of firing.
- **Vizia's soft minimum at the corner** lets a full-stick climb at corner speed still trade
  about 1 g into store (above). If the turn should take the whole budget there, Vizia's
  `lift_softmin_exponent` can be raised (it also raises its corner rate).
