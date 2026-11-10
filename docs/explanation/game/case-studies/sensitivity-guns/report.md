# sensitivity: the gun's values: 34560 games

| Slot | Elo against its airframe's baseline |
|---|---|
| la7: baseline | +0 ± 0 |
| la7: muzzle speed x0.9 | -181 ± 8 |
| la7: muzzle speed x0.95 | -98 ± 8 |
| la7: muzzle speed x1.05 | +114 ± 8 |
| la7: muzzle speed x1.1 | +210 ± 9 |
| la7: damage per second x0.9 | -148 ± 8 |
| la7: damage per second x0.95 | -115 ± 8 |
| la7: damage per second x1.05 | +4 ± 8 |
| la7: damage per second x1.1 | +154 ± 8 |
| la7: 5 Hz, damage/s held | -196 ± 8 |
| la7: 6 Hz, damage/s held | -128 ± 8 |
| la7: 15 Hz, damage/s held | +64 ± 8 |
| la7: 15 Hz, damage/bullet held | +491 ± 12 |
| la7: 6 Hz, damage/bullet held | -941 ± 20 |
| la7: dispersion x0.5 | -3 ± 8 |
| la7: dispersion x0.7 | +5 ± 8 |
| la7: dispersion x1.4 | -2 ± 8 |
| la7: dispersion x2 | -8 ± 8 |
| la7: shot cost x2 | +18 ± 8 |
| la7: shot cost x1.4 | +11 ± 8 |
| la7: shot cost x0.7 | -20 ± 8 |
| la7: shot cost x0.5 | -62 ± 8 |

**la7** (RMS residual 51 Elo): muzzle_speed +1364 Elo a doubling, damage_per_second +858 Elo a doubling, fire_rate +170 Elo a doubling, dispersion -3 Elo a doubling, handle shots_per_store +38 Elo a doubling

**pooled** (RMS residual 51 Elo): muzzle_speed +1364 Elo a doubling, damage_per_second +858 Elo a doubling, fire_rate +170 Elo a doubling, dispersion -3 Elo a doubling, handle shots_per_store +38 Elo a doubling

Majority cycles: 14
- la7: baseline > la7: dispersion x0.5 > la7: damage per second x1.05 > (weakest 0.50)
- la7: baseline > la7: dispersion x2 > la7: damage per second x1.05 > (weakest 0.50)
- la7: baseline > la7: shot cost x0.5 > la7: damage per second x1.05 > (weakest 0.50)
- la7: damage per second x1.05 > la7: shot cost x1.4 > la7: dispersion x0.5 > (weakest 0.53)
- la7: damage per second x1.05 > la7: shot cost x0.7 > la7: dispersion x0.5 > (weakest 0.53)
- la7: damage per second x1.05 > la7: shot cost x1.4 > la7: dispersion x0.7 > (weakest 0.55)
- la7: damage per second x1.05 > la7: dispersion x1.4 > la7: dispersion x2 > (weakest 0.54)
- la7: damage per second x1.05 > la7: dispersion x1.4 > la7: shot cost x0.5 > (weakest 0.54)
- la7: damage per second x1.05 > la7: shot cost x1.4 > la7: dispersion x2 > (weakest 0.54)
- la7: damage per second x1.05 > la7: shot cost x1.4 > la7: shot cost x2 > (weakest 0.51)
- la7: damage per second x1.05 > la7: shot cost x1.4 > la7: shot cost x0.5 > (weakest 0.56)
- la7: damage per second x1.05 > la7: shot cost x0.7 > la7: shot cost x0.5 > (weakest 0.56)
- la7: dispersion x0.5 > la7: dispersion x1.4 > la7: dispersion x2 > (weakest 0.52)
- la7: dispersion x0.5 > la7: dispersion x1.4 > la7: shot cost x0.7 > (weakest 0.55)
