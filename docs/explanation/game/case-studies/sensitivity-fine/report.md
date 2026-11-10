# sensitivity: the steep knobs in small steps: 9216 games

| Slot | Elo against its airframe's baseline |
|---|---|
| la7: baseline | +0 ± 0 |
| la7: shield x0.97 | -5 ± 14 |
| la7: shield x0.98 | +5 ± 14 |
| la7: shield x1.02 | +29 ± 13 |
| la7: shield x1.03 | +9 ± 14 |
| la7: damage per second x0.975 | -12 ± 13 |
| la7: damage per second x1.025 | +28 ± 13 |
| la7: damage per second x1.075 | +162 ± 15 |
| la7: head-on fraction x0.9 | +119 ± 14 |
| la7: head-on fraction x0.95 | +142 ± 14 |
| la7: head-on fraction x1.05 | -13 ± 14 |
| la7: head-on fraction x1.1 | -132 ± 14 |
| la7: hit radius x0.95 | +52 ± 14 |
| la7: hit radius x1.05 | +0 ± 14 |
| la7: muzzle speed x0.98 | -31 ± 14 |
| la7: muzzle speed x1.02 | +50 ± 13 |
| la7: turn rate x0.975 | -23 ± 13 |
| la7: turn rate x1.025 | +39 ± 14 |
| la7: stall speed x0.95 | -31 ± 14 |
| la7: stall speed x1.05 | +27 ± 14 |

**la7** (RMS residual 26 Elo): max_shield +251 Elo a doubling, damage_per_second +1330 Elo a doubling, handle head_on_fraction -903 Elo a doubling, handle hit_radius -365 Elo a doubling, muzzle_speed +1414 Elo a doubling, sustained_turn_rate +837 Elo a doubling, stall_speed +397 Elo a doubling

**pooled** (RMS residual 26 Elo): max_shield +251 Elo a doubling, damage_per_second +1330 Elo a doubling, handle head_on_fraction -903 Elo a doubling, handle hit_radius -365 Elo a doubling, muzzle_speed +1414 Elo a doubling, sustained_turn_rate +837 Elo a doubling, stall_speed +397 Elo a doubling

Majority cycles: 15
- la7: baseline > la7: shield x0.97 > la7: shield x1.03 > (weakest 0.53)
- la7: baseline > la7: hit radius x1.05 > la7: shield x1.02 > (weakest 0.51)
- la7: baseline > la7: head-on fraction x1.05 > la7: shield x1.03 > (weakest 0.50)
- la7: baseline > la7: damage per second x0.975 > la7: damage per second x1.025 > (weakest 0.52)
- la7: baseline > la7: turn rate x0.975 > la7: damage per second x1.025 > (weakest 0.53)
- la7: shield x0.97 > la7: shield x1.03 > la7: shield x0.98 > (weakest 0.52)
- la7: shield x0.97 > la7: shield x1.03 > la7: stall speed x1.05 > (weakest 0.53)
- la7: shield x0.98 > la7: hit radius x1.05 > la7: shield x1.02 > (weakest 0.52)
- la7: shield x1.02 > la7: damage per second x0.975 > la7: hit radius x1.05 > (weakest 0.57)
- la7: shield x1.02 > la7: head-on fraction x1.05 > la7: hit radius x1.05 > (weakest 0.53)
- la7: shield x1.02 > la7: stall speed x1.05 > la7: turn rate x1.025 > (weakest 0.51)
- la7: damage per second x0.975 > la7: damage per second x1.025 > la7: muzzle speed x0.98 > (weakest 0.52)
- la7: damage per second x0.975 > la7: hit radius x1.05 > la7: muzzle speed x0.98 > (weakest 0.57)
- la7: damage per second x0.975 > la7: stall speed x0.95 > la7: muzzle speed x0.98 > (weakest 0.57)
- la7: damage per second x1.025 > la7: head-on fraction x1.05 > la7: turn rate x0.975 > (weakest 0.55)
