# Core flight model: measured manoeuvres

Standard manoeuvres flown in both flight models (`GameConfig().flight_model` "v2" and
`GameConfig.core()`), one ship (Vizia), no fields, coast unless stated, store at half unless
stated. Measured 2026-11-23 on the laptop CPU, one thread, by
`docs/rewrite/sections/turns/measure_core.py`.

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
| climb from top speed, half store: time to 99% (s), speed then (m/s) | not reached in 60 s (91%) | 1.6, 273 |
| dive from top speed, half store: time to 1% (s), peak speed (m/s) | 24.0, 442 | 1.4, 368 |
| speed lost to 1 s of firing at top speed (m/s, against not firing) | 0.0 | 6.0 |
| speed lost to 1 s of firing at corner speed (m/s, against not firing) | 0.0 | 7.3 |

## Reading

- **Coasting holds altitude in core.** In v2 the store climbs while coasting (to 95% after
  120 s straight) because the engine's surplus charges it; in core it stays at 50.0%.
- **Full stick, coasting, settles slower and turns less in core (111 m/s, 25 deg/s against
  139 m/s, 38.6 deg/s).** Weight on the wing: coast holds 1 g first and the turn gets what is
  left. To turn harder at low speed the ship dives (the first second of full stick and dive
  turns 59 deg against 57 coasting). Sustained and fine stick are unchanged.
- **The two-loop store is crossed in about 1.5 s per half at full exchange rate** (half to
  full in 1.6 s from top speed; half to empty in 1.4 s). The exchange rate is 38.6 m/s^2
  (about the game's 1 g, which is what a vertical climb costs), and the whole band is worth
  about +-45 m/s around cruise (dive peak 368 m/s from 325). That follows from "two loops"
  and may feel thin: a question for the author (larger store, slower exchange, or as is).
- **Firing costs 6.0 m/s per second at top speed and 7.3 m/s per second at corner speed**
  with the engine fully cut (`fire_thrust_fraction` 0.0): about 2 to 3% of speed per second
  of firing. Whether that is enough of a cost is for the author's flying.
