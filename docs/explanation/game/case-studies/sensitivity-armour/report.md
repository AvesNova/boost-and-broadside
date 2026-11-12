---
covers: [bench/case_study.py]
---

# sensitivity: shield and body: 41472 games

| Slot | Elo against its airframe's baseline |
|---|---|
| la7: baseline | +0 ± 0 |
| la7: shield x0.9 | -133 ± 8 |
| la7: shield x0.95 | -129 ± 8 |
| la7: shield x1.05 | +132 ± 8 |
| la7: shield x1.1 | +138 ± 8 |
| la7: shield recharge rate x0.5 | -15 ± 8 |
| la7: shield recharge rate x0.7 | +4 ± 8 |
| la7: shield recharge rate x1.4 | +17 ± 8 |
| la7: shield recharge rate x2 | +24 ± 8 |
| la7: shield recharge delay x0.5 | +15 ± 8 |
| la7: shield recharge delay x0.7 | +15 ± 8 |
| la7: shield recharge delay x1.4 | -26 ± 8 |
| la7: shield recharge delay x2 | -29 ± 8 |
| la7: head-on fraction x0.5 | +1282 ± 20 |
| la7: head-on fraction x0.75 | +621 ± 12 |
| la7: head-on fraction x1.5 | -387 ± 10 |
| la7: head-on fraction x2 | -587 ± 13 |
| la7: hit radius x0.8 | +136 ± 8 |
| la7: hit radius x0.9 | +50 ± 8 |
| la7: hit radius x1.1 | -47 ± 8 |
| la7: hit radius x1.25 | -89 ± 8 |
| la7: spawn energy x0.5 | -24 ± 8 |
| la7: spawn energy x2 | -12 ± 8 |

**la7** (RMS residual 126 Elo): max_shield +1105 Elo a doubling, handle shield_recharge_rate +18 Elo a doubling, handle shield_recharge_delay -26 Elo a doubling, handle head_on_fraction -818 Elo a doubling, handle hit_radius -348 Elo a doubling, handle spawn_energy +6 Elo a doubling

**pooled** (RMS residual 126 Elo): max_shield +1105 Elo a doubling, handle shield_recharge_rate +18 Elo a doubling, handle shield_recharge_delay -26 Elo a doubling, handle head_on_fraction -818 Elo a doubling, handle hit_radius -348 Elo a doubling, handle spawn_energy +6 Elo a doubling

Majority cycles: 20
- la7: baseline > la7: shield recharge rate x0.7 > la7: shield recharge delay x0.7 > (weakest 0.51)
- la7: baseline > la7: spawn energy x2 > la7: shield recharge rate x1.4 > (weakest 0.52)
- la7: baseline > la7: shield recharge delay x2 > la7: shield recharge delay x0.5 > (weakest 0.52)
- la7: shield x0.9 > la7: shield x0.95 > la7: hit radius x1.25 > (weakest 0.52)
- la7: shield recharge rate x0.5 > la7: spawn energy x2 > la7: shield recharge rate x0.7 > (weakest 0.54)
- la7: shield recharge rate x0.5 > la7: spawn energy x2 > la7: shield recharge rate x1.4 > (weakest 0.52)
- la7: shield recharge rate x0.5 > la7: shield recharge delay x2 > la7: shield recharge delay x0.5 > (weakest 0.52)
- la7: shield recharge rate x0.7 > la7: shield recharge delay x0.7 > la7: shield recharge rate x1.4 > (weakest 0.52)
- la7: shield recharge rate x0.7 > la7: shield recharge delay x0.7 > la7: shield recharge delay x0.5 > (weakest 0.54)
- la7: shield recharge rate x0.7 > la7: shield recharge delay x2 > la7: shield recharge delay x0.5 > (weakest 0.55)
- la7: shield recharge rate x0.7 > la7: shield recharge delay x0.7 > la7: spawn energy x2 > (weakest 0.54)
- la7: shield recharge rate x0.7 > la7: shield recharge delay x2 > la7: spawn energy x2 > (weakest 0.52)
- la7: shield recharge rate x1.4 > la7: shield recharge rate x2 > la7: shield recharge delay x0.7 > (weakest 0.52)
- la7: shield recharge rate x1.4 > la7: shield recharge rate x2 > la7: spawn energy x2 > (weakest 0.52)
- la7: shield recharge rate x1.4 > la7: shield recharge delay x0.5 > la7: spawn energy x2 > (weakest 0.51)
- la7: shield recharge rate x1.4 > la7: shield recharge delay x2 > la7: spawn energy x2 > (weakest 0.52)
- la7: shield recharge delay x0.5 > la7: shield recharge delay x1.4 > la7: shield recharge delay x2 > (weakest 0.53)
- la7: shield recharge delay x0.5 > la7: spawn energy x0.5 > la7: shield recharge delay x2 > (weakest 0.55)
- la7: shield recharge delay x1.4 > la7: shield recharge delay x2 > la7: spawn energy x2 > (weakest 0.51)
- la7: shield recharge delay x2 > la7: spawn energy x2 > la7: spawn energy x0.5 > (weakest 0.52)
