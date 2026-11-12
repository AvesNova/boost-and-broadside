---
covers: [bench/case_study.py]
---

# sensitivity: flight and store: 9216 games

| Slot | Elo against its airframe's baseline |
|---|---|
| la7: baseline | +0 ± 0 |
| la7: stall speed x0.8 | -107 ± 17 |
| la7: stall speed x0.9 | -76 ± 16 |
| la7: stall speed x1.1 | +65 ± 16 |
| la7: stall speed x1.25 | +111 ± 17 |
| la7: cruise speed x0.8 | -34 ± 17 |
| la7: cruise speed x0.9 | -1 ± 17 |
| la7: cruise speed x1.1 | +4 ± 16 |
| la7: cruise speed x1.25 | +16 ± 17 |
| la7: turn rate x0.9 | -98 ± 17 |
| la7: turn rate x0.95 | -48 ± 16 |
| la7: turn rate x1.05 | +38 ± 17 |
| la7: turn rate x1.1 | +110 ± 17 |
| la7: climb rate x0.5 | -1 ± 17 |
| la7: climb rate x0.7 | +8 ± 17 |
| la7: climb rate x1.4 | -11 ± 16 |
| la7: climb rate x2 | -31 ± 17 |
| la7: store x0.5 | -76 ± 17 |
| la7: store x0.7 | -4 ± 16 |
| la7: store x1.4 | +10 ± 17 |
| la7: store x2 | +27 ± 16 |
| la7: half-lift drag x0.5 | +55 ± 17 |
| la7: half-lift drag x0.7 | +37 ± 17 |
| la7: half-lift drag x1.4 | -30 ± 16 |
| la7: half-lift drag x2 | -74 ± 17 |
| la7: engine knee x0.6 | -9 ± 17 |
| la7: engine knee x1.4 | +11 ± 16 |
| la7: engine knee x1.8 | +15 ± 16 |
| la7: coast exchange x0.5 | +21 ± 16 |
| la7: coast exchange x2 | -47 ± 16 |

**la7** (RMS residual 12 Elo): stall_speed +366 Elo a doubling, cruise_speed +67 Elo a doubling, sustained_turn_rate +689 Elo a doubling, climb_rate -16 Elo a doubling, ceiling +43 Elo a doubling, handle half_lift_drag -65 Elo a doubling, card engine_knee_scale +16 Elo a doubling, card coast_exchange_scale -34 Elo a doubling

**pooled** (RMS residual 12 Elo): stall_speed +366 Elo a doubling, cruise_speed +67 Elo a doubling, sustained_turn_rate +689 Elo a doubling, climb_rate -16 Elo a doubling, ceiling +43 Elo a doubling, handle half_lift_drag -65 Elo a doubling, card engine_knee_scale +16 Elo a doubling, card coast_exchange_scale -34 Elo a doubling

Majority cycles: 10
- la7: baseline > la7: cruise speed x0.8 > la7: store x0.7 > (weakest 0.51)
- la7: baseline > la7: coast exchange x0.5 > la7: cruise speed x1.25 > (weakest 0.51)
- la7: baseline > la7: climb rate x0.5 > la7: climb rate x0.7 > (weakest 0.52)
- la7: baseline > la7: climb rate x0.5 > la7: engine knee x1.8 > (weakest 0.52)
- la7: baseline > la7: store x2 > la7: climb rate x0.7 > (weakest 0.52)
- la7: baseline > la7: climb rate x1.4 > la7: half-lift drag x0.7 > (weakest 0.53)
- la7: baseline > la7: coast exchange x0.5 > la7: store x0.7 > (weakest 0.55)
- la7: baseline > la7: store x2 > la7: store x1.4 > (weakest 0.52)
- la7: baseline > la7: coast exchange x0.5 > la7: engine knee x1.4 > (weakest 0.53)
- la7: baseline > la7: coast exchange x0.5 > la7: engine knee x1.8 > (weakest 0.53)
