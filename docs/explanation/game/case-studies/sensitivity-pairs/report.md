# trade-off pairs: 18432 games

| Slot | Elo against its airframe's baseline |
|---|---|
| la7: baseline | +0 ± 0 |
| la7: muzzle x1.02 + turn x0.95 | -3 ± 11 |
| la7: muzzle x1.02 + stall x0.92 | +5 ± 11 |
| la7: muzzle x1.02 + hit radius x1.1 | -11 ± 11 |
| la7: turn x1.03 + muzzle x0.97 | -33 ± 11 |
| la7: turn x1.03 + hit radius x1.1 | -24 ± 11 |
| la7: turn x1.03 + half-lift drag x1.7 | -37 ± 11 |
| la7: stall x1.08 + half-lift drag x1.7 | +3 ± 11 |
| la7: stall x1.08 + turn x0.95 | -24 ± 11 |
| la7: hit radius x0.95 + muzzle x0.97 | -43 ± 11 |
| la7: hit radius x0.95 + stall x0.92 | -29 ± 11 |
| la7: 15 Hz + turn x0.95 | +23 ± 11 |
| la7: 15 Hz + hit radius x1.1 | +0 ± 11 |
| la7: muzzle x1.05 + turn x0.9 | +31 ± 11 |
| la7: turn x1.1 + muzzle x0.95 | -75 ± 11 |
| la7: hit radius x0.8 + turn x0.9 | +9 ± 11 |
| la7: shield x1.05 + head-on x1.1 | -9 ± 11 |
| la7: damage/s x1.1 + shield x0.95 | +13 ± 11 |
| la7: stall x1.1 + hit radius x1.25 | -76 ± 11 |
| la7: muzzle x1.02 | +30 ± 11 |
| la7: muzzle x0.97 | -82 ± 11 |
| la7: turn x1.03 | +22 ± 11 |
| la7: turn x0.95 | -43 ± 11 |
| la7: hit radius x0.95 | +24 ± 11 |
| la7: hit radius x1.1 | -55 ± 11 |
| la7: stall x1.08 | +52 ± 11 |
| la7: stall x0.92 | -65 ± 11 |
| la7: half-lift drag x0.5 | +28 ± 11 |
| la7: half-lift drag x1.7 | -56 ± 11 |

**la7** (RMS residual 13 Elo): muzzle_speed +1775 Elo a doubling, sustained_turn_rate +617 Elo a doubling, stall_speed +397 Elo a doubling, handle hit_radius -376 Elo a doubling, handle half_lift_drag -55 Elo a doubling, fire_rate +104 Elo a doubling, max_shield -54 Elo a doubling, handle head_on_fraction -37 Elo a doubling, damage_per_second +66 Elo a doubling

**pooled** (RMS residual 13 Elo): muzzle_speed +1775 Elo a doubling, sustained_turn_rate +617 Elo a doubling, stall_speed +397 Elo a doubling, handle hit_radius -376 Elo a doubling, handle half_lift_drag -55 Elo a doubling, fire_rate +104 Elo a doubling, max_shield -54 Elo a doubling, handle head_on_fraction -37 Elo a doubling, damage_per_second +66 Elo a doubling

Majority cycles: 71
- la7: baseline > la7: muzzle x1.02 + hit radius x1.1 > la7: muzzle x1.02 + turn x0.95 > (weakest 0.51)
- la7: baseline > la7: turn x1.03 + hit radius x1.1 > la7: muzzle x1.02 + turn x0.95 > (weakest 0.51)
- la7: baseline > la7: muzzle x1.02 + stall x0.92 > la7: muzzle x1.05 + turn x0.9 > (weakest 0.51)
- la7: baseline > la7: muzzle x1.02 + stall x0.92 > la7: hit radius x0.95 > (weakest 0.51)
- la7: baseline > la7: muzzle x1.02 + hit radius x1.1 > la7: 15 Hz + turn x0.95 > (weakest 0.52)
- la7: baseline > la7: muzzle x1.02 + hit radius x1.1 > la7: stall x1.08 > (weakest 0.53)
- la7: baseline > la7: turn x1.03 + muzzle x0.97 > la7: damage/s x1.1 + shield x0.95 > (weakest 0.51)
- la7: baseline > la7: turn x1.03 + half-lift drag x1.7 > la7: 15 Hz + turn x0.95 > (weakest 0.52)
- la7: baseline > la7: stall x1.08 + turn x0.95 > la7: turn x1.03 > (weakest 0.52)
- la7: baseline > la7: hit radius x0.95 + muzzle x0.97 > la7: damage/s x1.1 + shield x0.95 > (weakest 0.52)
- la7: baseline > la7: 15 Hz + hit radius x1.1 > la7: hit radius x0.8 + turn x0.9 > (weakest 0.52)
- la7: baseline > la7: 15 Hz + hit radius x1.1 > la7: muzzle x1.02 > (weakest 0.51)
- la7: baseline > la7: 15 Hz + hit radius x1.1 > la7: turn x1.03 > (weakest 0.53)
- la7: baseline > la7: shield x1.05 + head-on x1.1 > la7: hit radius x0.8 + turn x0.9 > (weakest 0.51)
- la7: muzzle x1.02 + turn x0.95 > la7: muzzle x1.02 + stall x0.92 > la7: muzzle x1.02 + hit radius x1.1 > (weakest 0.51)
- la7: muzzle x1.02 + turn x0.95 > la7: muzzle x1.02 + stall x0.92 > la7: hit radius x0.95 > (weakest 0.51)
- la7: muzzle x1.02 + turn x0.95 > la7: stall x1.08 + turn x0.95 > la7: muzzle x1.02 + hit radius x1.1 > (weakest 0.51)
- la7: muzzle x1.02 + turn x0.95 > la7: hit radius x0.95 + muzzle x0.97 > la7: muzzle x1.02 + hit radius x1.1 > (weakest 0.55)
- la7: muzzle x1.02 + turn x0.95 > la7: muzzle x1.02 > la7: muzzle x1.02 + hit radius x1.1 > (weakest 0.52)
- la7: muzzle x1.02 + turn x0.95 > la7: 15 Hz + hit radius x1.1 > la7: turn x1.03 + hit radius x1.1 > (weakest 0.53)
