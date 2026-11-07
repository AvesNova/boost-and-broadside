# Belief and next-state baselines: BC arm C (policy-uniform), CPU self-play

Measured 2026-11-07 on CPU (bf16 matmuls as the policy was built), the policy in both seats,
fog on, from fresh games. Per preset: 32 games (64 views), 6 consecutive segments of 256
ticks (51 s of play from the start of each game), 30 ticks per second. Smaller than "a few
hundred games": a first 200-game run used 28 GB and was killed; these are capped at 8 GB.

- **As run**: the pre-fix tree (v1 spread fit fed back), which is how C trained and plays.
- **Fixed decoder in the loop**: the same policy with `spread_fit="coarsest"` fed back. C never
  trained on wide input spreads, so this is out of distribution; it shows what the loop does,
  not what a retrained C would do.
- Belief: the replay record at t (what the policy believed after sight) against the truth at t,
  over hidden living enemies holding a belief, by time since last seen. Persistence: the last
  sighting. Dead reckoning: last position + velocity x age, heading held. "Clamped to arena":
  the same, clamped into the arena disc (the boundary only damages, so this is a bound). "Unpowered
  coast": x0 + v_hat ln(1 + c_d0 |v| t) / c_d0 with the type's zero-lift drag (ships thrust,
  so this is not the game's motion; the full drag needs the controls).
- Next state: KL to the sharp target code of t + 1 (cross-entropy minus the target's entropy),
  nats per ship, per run of the code. Persistence is today's baseline, log(code_t + 1e-3); dead
  reckoned is the same floor around the sharp code of (x + v dt, heading + nose_rate dt).
- Kinematic residual: own ships (always visible; each ship once per view), one tick, ticks
  where the ship lived at both ends and did not respawn. The sim's flight step (game spec 3.6,
  `bnb.game.physics.flight.flight_tick`) is kick-drift-kick with dt = 1/30 s: a half-tick kick
  (speed, angle of attack, nose rate, energy), a drift along the exact nose arc at the
  mid-step velocity (corrected to the index at the midpoint), the medium re-sampled, a second
  half kick. So the drift's velocity lies between v_t and v_(t+1): the mean of the two is
  closest, then v_t, then v_(t+1). "In a field": |log refractive index| > 1e-3 at t. "In a zone":
  inside a present zone's radius (frontline only). Thrust: the action's power (0 coast,
  1 boost, 2 reverse).

# As run (v1 spread fit, C as trained)

## deathmatch: 1 x 32 games, 6 segments of 256 ticks in all (the same games going on)

### Hidden enemies: position error (m), mean / median

| hidden for | count | belief | persistence | dead reckoning | DR clamped to arena | DR unpowered coast (zero-lift drag) |
|---|---|---|---|---|---|---|
| 0.1-0.5 s | 8555 | 7 / 4 | 14 / 12 | 2 / 1 | 3 / 1 | 2 / 1 |
| 0.5-1 s | 9899 | 28 / 22 | 37 / 34 | 11 / 9 | 12 / 9 | 11 / 9 |
| 1-2 s | 18349 | 101 / 70 | 76 / 71 | 38 / 32 | 39 / 32 | 38 / 32 |
| 2-5 s | 48741 | 634 / 390 | 170 / 152 | 152 / 135 | 150 / 133 | 149 / 134 |
| 5-10 s | 63005 | 1808 / 1460 | 427 / 415 | 469 / 440 | 453 / 427 | 458 / 435 |
| 10-30 s | 91761 | 3911 / 2691 | 1133 / 1058 | 1251 / 1151 | 1166 / 1090 | 1218 / 1131 |

### Hidden enemies: heading error (rad), mean / median (dead reckoning holds the heading: = persistence)

| hidden for | belief | persistence / dead reckoning |
|---|---|---|
| 0.1-0.5 s | 0.18 / 0.09 | 0.15 / 0.11 |
| 0.5-1 s | 0.64 / 0.43 | 0.47 / 0.44 |
| 1-2 s | 0.92 / 0.67 | 0.97 / 0.95 |
| 2-5 s | 0.85 / 0.63 | 1.52 / 1.56 |
| 5-10 s | 0.82 / 0.61 | 1.67 / 1.79 |
| 10-30 s | 0.77 / 0.60 | 1.69 / 1.83 |

### Hidden enemies: the belief's sigma, as run (v1 decoder in the loop) and the same forecasts read by the fixed decoder

Within 1 sigma: expected 63% (position, 2-D) and 68% (heading). The fixed-decoder columns re-read C's own one-step forecasts (the loop still fed back v1's sigma).

| hidden for | pos sigma median (v1) | within 1 sigma (v1) | pos sigma median (fixed) | within 1 sigma (fixed) | heading sigma median (v1) | within (v1) | heading sigma median (fixed) | within (fixed) |
|---|---|---|---|---|---|---|---|---|
| 0.1-0.5 s | 1.0 m | 12.9% | 3 m | 46.7% | 0.028 | 19.1% | 0.14 | 63.7% |
| 0.5-1 s | 1.3 m | 0.8% | 10 m | 33.9% | 0.030 | 4.1% | 0.71 | 65.2% |
| 1-2 s | 1.3 m | 0.0% | 40 m | 41.0% | 0.031 | 2.9% | 0.98 | 63.2% |
| 2-5 s | 1.3 m | 0.0% | 200 m | 33.1% | 0.032 | 3.2% | 0.91 | 63.9% |
| 5-10 s | 1.3 m | 0.0% | 422 m | 15.4% | 0.033 | 3.3% | 0.82 | 61.0% |
| 10-30 s | 1.2 m | 0.0% | 548 m | 6.1% | 0.033 | 3.2% | 0.74 | 58.4% |

### Next state one tick ahead: KL to the sharp target (nats per ship), by run

Head: the next-state head. Persistence: the baseline code log(code_t + 1e-3) (today's). Dead reckoned: the same floor around the sharp code of x + v dt, heading + nose rate dt. Rows alive and supervised at t.


**ally** (459792 rows)

| run | head | persistence | dead reckoned | dead reckoned, heading held |
|---|---|---|---|---|
| position | 1.007 | 6.246 | 0.049 | 0.049 |
| velocity | 0.901 | 1.650 | 1.650 | 1.650 |
| heading | 0.839 | 3.770 | 1.766 | 3.770 |
| nose_rate | 1.419 | 3.591 | 3.591 | 3.591 |
| angle_of_attack | 0.287 | 0.634 | 0.634 | 0.634 |
| shield_recharge_wait | 0.039 | 0.059 | 0.059 | 0.059 |
| shield | 0.048 | 0.063 | 0.063 | 0.063 |
| energy | 0.036 | 0.040 | 0.040 | 0.040 |
| log_index | 0.022 | 0.040 | 0.040 | 0.040 |
| cooldown | 0.030 | 0.239 | 0.239 | 0.239 |
| **total** | 4.63 | 16.33 | 8.13 | 10.14 |

**enemy visible** (215708 rows)

| run | head | persistence | dead reckoned | dead reckoned, heading held |
|---|---|---|---|---|
| position | 0.959 | 6.198 | 0.050 | 0.050 |
| velocity | 1.067 | 2.057 | 2.057 | 2.057 |
| heading | 0.813 | 4.499 | 1.658 | 4.499 |
| nose_rate | 1.305 | 3.358 | 3.358 | 3.358 |
| angle_of_attack | 0.284 | 0.616 | 0.616 | 0.616 |
| shield_recharge_wait | 0.076 | 0.097 | 0.097 | 0.097 |
| shield | 0.099 | 0.110 | 0.110 | 0.110 |
| energy | 0.054 | 0.058 | 0.058 | 0.058 |
| log_index | 0.015 | 0.034 | 0.034 | 0.034 |
| cooldown | 0.064 | 0.501 | 0.501 | 0.501 |
| **total** | 4.73 | 17.53 | 8.54 | 11.38 |

### Kinematic residual, position: x_(t+1) - (x_t + v_t dt) (m), own ships, one tick (dt = 0.03333 s)

| situation | rows | quantity | mean | 50% | 90% | 99% | 99.9% |
|---|---|---|---|---|---|---|---|
| all | 459571 | |r| | 0.0256 | 0.0203 | 0.0501 | 0.1075 | 0.2513 |
| all | 459571 | |r_x| | 0.0162 | 0.0105 | 0.0370 | 0.0800 | 0.2001 |
| all | 459571 | |r_y| | 0.0164 | 0.0107 | 0.0371 | 0.0787 | 0.1981 |
| in a field | 69727 | |r| | 0.0309 | 0.0118 | 0.0824 | 0.2295 | 0.3674 |
| in a field | 69727 | |r_x| | 0.0200 | 0.0066 | 0.0555 | 0.1799 | 0.3036 |
| in a field | 69727 | |r_y| | 0.0194 | 0.0065 | 0.0537 | 0.1781 | 0.2888 |
| not in a field | 389844 | |r| | 0.0246 | 0.0214 | 0.0476 | 0.0774 | 0.0929 |
| not in a field | 389844 | |r_x| | 0.0155 | 0.0112 | 0.0357 | 0.0633 | 0.0823 |
| not in a field | 389844 | |r_y| | 0.0158 | 0.0115 | 0.0360 | 0.0640 | 0.0817 |
| not in a zone | 459571 | |r| | 0.0256 | 0.0203 | 0.0501 | 0.1075 | 0.2513 |
| not in a zone | 459571 | |r_x| | 0.0162 | 0.0105 | 0.0370 | 0.0800 | 0.2001 |
| not in a zone | 459571 | |r_y| | 0.0164 | 0.0107 | 0.0371 | 0.0787 | 0.1981 |
| coast (power 0) | 363908 | |r| | 0.0242 | 0.0178 | 0.0490 | 0.1101 | 0.2561 |
| coast (power 0) | 363908 | |r_x| | 0.0153 | 0.0092 | 0.0362 | 0.0807 | 0.2036 |
| coast (power 0) | 363908 | |r_y| | 0.0155 | 0.0094 | 0.0364 | 0.0789 | 0.2021 |
| boost (power 1) | 38822 | |r| | 0.0281 | 0.0241 | 0.0421 | 0.0954 | 0.2219 |
| boost (power 1) | 38822 | |r_x| | 0.0178 | 0.0156 | 0.0334 | 0.0721 | 0.1813 |
| boost (power 1) | 38822 | |r_y| | 0.0181 | 0.0159 | 0.0338 | 0.0733 | 0.1734 |
| reverse (power 2) | 56841 | |r| | 0.0327 | 0.0291 | 0.0604 | 0.1019 | 0.2426 |
| reverse (power 2) | 56841 | |r_x| | 0.0210 | 0.0162 | 0.0442 | 0.0801 | 0.1933 |
| reverse (power 2) | 56841 | |r_y| | 0.0207 | 0.0157 | 0.0436 | 0.0806 | 0.1944 |

### Kinematic residual, position: x_(t+1) - (x_t + v_{t+1} dt) (m), own ships, one tick (dt = 0.03333 s)

| situation | rows | quantity | mean | 50% | 90% | 99% | 99.9% |
|---|---|---|---|---|---|---|---|
| all | 459571 | |r| | 0.0305 | 0.0251 | 0.0587 | 0.1371 | 0.2907 |
| all | 459571 | |r_x| | 0.0193 | 0.0131 | 0.0425 | 0.0952 | 0.2361 |
| all | 459571 | |r_y| | 0.0196 | 0.0133 | 0.0434 | 0.0930 | 0.2265 |

### Kinematic residual, position: x_(t+1) - (x_t + (v_t + v_{t+1})/2 dt) (m), own ships, one tick (dt = 0.03333 s)

| situation | rows | quantity | mean | 50% | 90% | 99% | 99.9% |
|---|---|---|---|---|---|---|---|
| all | 459571 | |r| | 0.0069 | 0.0035 | 0.0171 | 0.0380 | 0.0999 |
| all | 459571 | |r_x| | 0.0043 | 0.0017 | 0.0116 | 0.0295 | 0.0814 |
| all | 459571 | |r_y| | 0.0044 | 0.0017 | 0.0122 | 0.0299 | 0.0722 |

### Kinematic residual, heading: |heading_(t+1) - (heading_t + nose_rate_t dt)| (rad)

| situation | rows | mean | 50% | 90% | 99% | 99.9% |
|---|---|---|---|---|---|---|
| all | 459571 | 0.0137 | 0.0086 | 0.0329 | 0.0680 | 0.1506 |
| in a field | 69727 | 0.0147 | 0.0053 | 0.0384 | 0.1344 | 0.2180 |
| not in a field | 389844 | 0.0135 | 0.0094 | 0.0325 | 0.0596 | 0.0729 |
| not in a zone | 459571 | 0.0137 | 0.0086 | 0.0329 | 0.0680 | 0.1506 |
| coast (power 0) | 363908 | 0.0147 | 0.0098 | 0.0343 | 0.0695 | 0.1556 |
| boost (power 1) | 38822 | 0.0094 | 0.0039 | 0.0255 | 0.0603 | 0.1335 |
| reverse (power 2) | 56841 | 0.0101 | 0.0047 | 0.0269 | 0.0611 | 0.1272 |

### Kinematic residual, heading: |heading_(t+1) - (heading_t + nose_rate_(t+1) dt)| (rad)

| situation | rows | mean | 50% | 90% | 99% | 99.9% |
|---|---|---|---|---|---|---|
| all | 459571 | 0.0087 | 0.0059 | 0.0210 | 0.0396 | 0.0641 |

### Kinematic residual, heading: |heading_(t+1) - (heading_t + mean nose rate dt)| (rad)

| situation | rows | mean | 50% | 90% | 99% | 99.9% |
|---|---|---|---|---|---|---|
| all | 459571 | 0.0025 | 0.0013 | 0.0061 | 0.0146 | 0.0445 |

## deathmatch_wide: 1 x 32 games, 6 segments of 256 ticks in all (the same games going on)

### Hidden enemies: position error (m), mean / median

| hidden for | count | belief | persistence | dead reckoning | DR clamped to arena | DR unpowered coast (zero-lift drag) |
|---|---|---|---|---|---|---|
| 0.1-0.5 s | 4726 | 9 / 5 | 17 / 15 | 2 / 1 | 4 / 1 | 2 / 1 |
| 0.5-1 s | 4943 | 37 / 25 | 43 / 41 | 11 / 9 | 13 / 9 | 11 / 9 |
| 1-2 s | 9141 | 107 / 73 | 85 / 81 | 41 / 33 | 42 / 33 | 40 / 33 |
| 2-5 s | 23657 | 563 / 260 | 177 / 162 | 167 / 146 | 167 / 147 | 161 / 143 |
| 5-10 s | 21679 | 2087 / 1569 | 395 / 397 | 450 / 415 | 450 / 415 | 427 / 395 |
| 10-30 s | 32056 | 4948 / 4181 | 1082 / 1016 | 1217 / 1053 | 1217 / 1053 | 1115 / 985 |

### Hidden enemies: heading error (rad), mean / median (dead reckoning holds the heading: = persistence)

| hidden for | belief | persistence / dead reckoning |
|---|---|---|
| 0.1-0.5 s | 0.24 / 0.13 | 0.20 / 0.16 |
| 0.5-1 s | 0.81 / 0.59 | 0.51 / 0.48 |
| 1-2 s | 1.21 / 1.07 | 0.99 / 0.98 |
| 2-5 s | 1.39 / 1.32 | 1.54 / 1.52 |
| 5-10 s | 1.53 / 1.52 | 1.50 / 1.47 |
| 10-30 s | 1.57 / 1.58 | 1.62 / 1.63 |

### Hidden enemies: the belief's sigma, as run (v1 decoder in the loop) and the same forecasts read by the fixed decoder

Within 1 sigma: expected 63% (position, 2-D) and 68% (heading). The fixed-decoder columns re-read C's own one-step forecasts (the loop still fed back v1's sigma).

| hidden for | pos sigma median (v1) | within 1 sigma (v1) | pos sigma median (fixed) | within 1 sigma (fixed) | heading sigma median (v1) | within (v1) | heading sigma median (fixed) | within (fixed) |
|---|---|---|---|---|---|---|---|---|
| 0.1-0.5 s | 0.9 m | 9.8% | 3 m | 41.5% | 0.027 | 12.8% | 0.18 | 60.4% |
| 0.5-1 s | 1.3 m | 0.5% | 9 m | 22.6% | 0.033 | 3.3% | 0.79 | 57.1% |
| 1-2 s | 1.4 m | 0.0% | 28 m | 24.0% | 0.033 | 2.1% | 1.18 | 53.5% |
| 2-5 s | 1.4 m | 0.0% | 152 m | 32.6% | 0.034 | 1.7% | 1.25 | 48.1% |
| 5-10 s | 1.3 m | 0.0% | 485 m | 21.4% | 0.034 | 1.2% | 1.16 | 38.3% |
| 10-30 s | 1.2 m | 0.0% | 590 m | 2.4% | 0.035 | 1.0% | 1.00 | 31.2% |

### Next state one tick ahead: KL to the sharp target (nats per ship), by run

Head: the next-state head. Persistence: the baseline code log(code_t + 1e-3) (today's). Dead reckoned: the same floor around the sharp code of x + v dt, heading + nose rate dt. Rows alive and supervised at t.


**ally** (455953 rows)

| run | head | persistence | dead reckoned | dead reckoned, heading held |
|---|---|---|---|---|
| position | 0.879 | 5.814 | 0.049 | 0.049 |
| velocity | 0.859 | 1.825 | 1.825 | 1.825 |
| heading | 0.699 | 4.701 | 1.216 | 4.701 |
| nose_rate | 1.023 | 2.757 | 2.757 | 2.757 |
| angle_of_attack | 0.230 | 0.468 | 0.468 | 0.468 |
| shield_recharge_wait | 0.074 | 0.094 | 0.094 | 0.094 |
| shield | 0.068 | 0.078 | 0.078 | 0.078 |
| energy | 0.043 | 0.045 | 0.045 | 0.045 |
| log_index | 0.004 | 0.022 | 0.022 | 0.022 |
| cooldown | 0.050 | 0.342 | 0.342 | 0.342 |
| **total** | 3.93 | 16.14 | 6.89 | 10.38 |

**enemy visible** (355514 rows)

| run | head | persistence | dead reckoned | dead reckoned, heading held |
|---|---|---|---|---|
| position | 0.836 | 5.660 | 0.049 | 0.049 |
| velocity | 0.849 | 1.886 | 1.886 | 1.886 |
| heading | 0.572 | 4.924 | 0.869 | 4.924 |
| nose_rate | 0.888 | 2.289 | 2.289 | 2.289 |
| angle_of_attack | 0.180 | 0.367 | 0.367 | 0.367 |
| shield_recharge_wait | 0.091 | 0.111 | 0.111 | 0.111 |
| shield | 0.086 | 0.093 | 0.093 | 0.093 |
| energy | 0.044 | 0.051 | 0.051 | 0.051 |
| log_index | 0.003 | 0.021 | 0.021 | 0.021 |
| cooldown | 0.061 | 0.432 | 0.432 | 0.432 |
| **total** | 3.61 | 15.84 | 6.17 | 10.22 |

### Kinematic residual, position: x_(t+1) - (x_t + v_t dt) (m), own ships, one tick (dt = 0.03333 s)

| situation | rows | quantity | mean | 50% | 90% | 99% | 99.9% |
|---|---|---|---|---|---|---|---|
| all | 455708 | |r| | 0.0278 | 0.0273 | 0.0463 | 0.0765 | 0.1177 |
| all | 455708 | |r_x| | 0.0179 | 0.0148 | 0.0369 | 0.0613 | 0.0905 |
| all | 455708 | |r_y| | 0.0176 | 0.0144 | 0.0366 | 0.0619 | 0.0922 |
| in a field | 49047 | |r| | 0.0210 | 0.0136 | 0.0544 | 0.1142 | 0.2382 |
| in a field | 49047 | |r_x| | 0.0135 | 0.0077 | 0.0300 | 0.0884 | 0.1973 |
| in a field | 49047 | |r_y| | 0.0132 | 0.0077 | 0.0279 | 0.0901 | 0.1721 |
| not in a field | 406661 | |r| | 0.0287 | 0.0292 | 0.0462 | 0.0699 | 0.0831 |
| not in a field | 406661 | |r_x| | 0.0184 | 0.0161 | 0.0371 | 0.0572 | 0.0757 |
| not in a field | 406661 | |r_y| | 0.0181 | 0.0156 | 0.0368 | 0.0579 | 0.0760 |
| not in a zone | 455708 | |r| | 0.0278 | 0.0273 | 0.0463 | 0.0765 | 0.1177 |
| not in a zone | 455708 | |r_x| | 0.0179 | 0.0148 | 0.0369 | 0.0613 | 0.0905 |
| not in a zone | 455708 | |r_y| | 0.0176 | 0.0144 | 0.0366 | 0.0619 | 0.0922 |
| coast (power 0) | 281095 | |r| | 0.0282 | 0.0280 | 0.0483 | 0.0791 | 0.1331 |
| coast (power 0) | 281095 | |r_x| | 0.0182 | 0.0144 | 0.0385 | 0.0636 | 0.0957 |
| coast (power 0) | 281095 | |r_y| | 0.0179 | 0.0138 | 0.0383 | 0.0643 | 0.1006 |
| boost (power 1) | 55701 | |r| | 0.0285 | 0.0286 | 0.0393 | 0.0664 | 0.1032 |
| boost (power 1) | 55701 | |r_x| | 0.0182 | 0.0171 | 0.0332 | 0.0509 | 0.0823 |
| boost (power 1) | 55701 | |r_y| | 0.0181 | 0.0171 | 0.0332 | 0.0516 | 0.0817 |
| reverse (power 2) | 118912 | |r| | 0.0266 | 0.0258 | 0.0439 | 0.0719 | 0.0949 |
| reverse (power 2) | 118912 | |r_x| | 0.0171 | 0.0144 | 0.0349 | 0.0588 | 0.0812 |
| reverse (power 2) | 118912 | |r_y| | 0.0168 | 0.0140 | 0.0343 | 0.0586 | 0.0812 |

### Kinematic residual, position: x_(t+1) - (x_t + v_{t+1} dt) (m), own ships, one tick (dt = 0.03333 s)

| situation | rows | quantity | mean | 50% | 90% | 99% | 99.9% |
|---|---|---|---|---|---|---|---|
| all | 455708 | |r| | 0.0300 | 0.0292 | 0.0502 | 0.0829 | 0.1343 |
| all | 455708 | |r_x| | 0.0193 | 0.0160 | 0.0393 | 0.0685 | 0.1018 |
| all | 455708 | |r_y| | 0.0189 | 0.0156 | 0.0388 | 0.0684 | 0.1008 |

### Kinematic residual, position: x_(t+1) - (x_t + (v_t + v_{t+1})/2 dt) (m), own ships, one tick (dt = 0.03333 s)

| situation | rows | quantity | mean | 50% | 90% | 99% | 99.9% |
|---|---|---|---|---|---|---|---|
| all | 455708 | |r| | 0.0043 | 0.0015 | 0.0128 | 0.0272 | 0.0405 |
| all | 455708 | |r_x| | 0.0028 | 0.0008 | 0.0083 | 0.0213 | 0.0339 |
| all | 455708 | |r_y| | 0.0027 | 0.0007 | 0.0080 | 0.0210 | 0.0327 |

### Kinematic residual, heading: |heading_(t+1) - (heading_t + nose_rate_t dt)| (rad)

| situation | rows | mean | 50% | 90% | 99% | 99.9% |
|---|---|---|---|---|---|---|
| all | 455708 | 0.0095 | 0.0045 | 0.0264 | 0.0546 | 0.0735 |
| in a field | 49047 | 0.0060 | 0.0019 | 0.0137 | 0.0651 | 0.1663 |
| not in a field | 406661 | 0.0099 | 0.0050 | 0.0272 | 0.0540 | 0.0689 |
| not in a zone | 455708 | 0.0095 | 0.0045 | 0.0264 | 0.0546 | 0.0735 |
| coast (power 0) | 281095 | 0.0115 | 0.0066 | 0.0302 | 0.0581 | 0.0773 |
| boost (power 1) | 55701 | 0.0062 | 0.0021 | 0.0186 | 0.0457 | 0.0679 |
| reverse (power 2) | 118912 | 0.0062 | 0.0024 | 0.0179 | 0.0422 | 0.0634 |

### Kinematic residual, heading: |heading_(t+1) - (heading_t + nose_rate_(t+1) dt)| (rad)

| situation | rows | mean | 50% | 90% | 99% | 99.9% |
|---|---|---|---|---|---|---|
| all | 455708 | 0.0065 | 0.0033 | 0.0182 | 0.0344 | 0.0437 |

### Kinematic residual, heading: |heading_(t+1) - (heading_t + mean nose rate dt)| (rad)

| situation | rows | mean | 50% | 90% | 99% | 99.9% |
|---|---|---|---|---|---|---|
| all | 455708 | 0.0015 | 0.0006 | 0.0042 | 0.0101 | 0.0153 |

## frontline: 1 x 32 games, 6 segments of 256 ticks in all (the same games going on)

### Hidden enemies: position error (m), mean / median

| hidden for | count | belief | persistence | dead reckoning | DR clamped to arena | DR unpowered coast (zero-lift drag) |
|---|---|---|---|---|---|---|
| 0.1-0.5 s | 7819 | 6 / 3 | 12 / 10 | 1 / 1 | 1 / 1 | 1 / 1 |
| 0.5-1 s | 9216 | 24 / 18 | 33 / 29 | 7 / 5 | 7 / 5 | 7 / 5 |
| 1-2 s | 17167 | 101 / 68 | 71 / 64 | 24 / 20 | 24 / 20 | 25 / 21 |
| 2-5 s | 46143 | 658 / 429 | 192 / 176 | 100 / 86 | 100 / 86 | 102 / 88 |
| 5-10 s | 66012 | 1682 / 1458 | 482 / 466 | 284 / 268 | 284 / 268 | 297 / 280 |
| 10-30 s | 81381 | 2940 / 2504 | 1016 / 969 | 634 / 589 | 634 / 589 | 681 / 633 |

### Hidden enemies: heading error (rad), mean / median (dead reckoning holds the heading: = persistence)

| hidden for | belief | persistence / dead reckoning |
|---|---|---|
| 0.1-0.5 s | 0.14 / 0.08 | 0.10 / 0.07 |
| 0.5-1 s | 0.49 / 0.35 | 0.27 / 0.24 |
| 1-2 s | 0.65 / 0.52 | 0.42 / 0.36 |
| 2-5 s | 0.58 / 0.47 | 0.41 / 0.30 |
| 5-10 s | 0.49 / 0.39 | 0.30 / 0.20 |
| 10-30 s | 0.54 / 0.43 | 0.34 / 0.22 |

### Hidden enemies: the belief's sigma, as run (v1 decoder in the loop) and the same forecasts read by the fixed decoder

Within 1 sigma: expected 63% (position, 2-D) and 68% (heading). The fixed-decoder columns re-read C's own one-step forecasts (the loop still fed back v1's sigma).

| hidden for | pos sigma median (v1) | within 1 sigma (v1) | pos sigma median (fixed) | within 1 sigma (fixed) | heading sigma median (v1) | within (v1) | heading sigma median (fixed) | within (fixed) |
|---|---|---|---|---|---|---|---|---|
| 0.1-0.5 s | 0.9 m | 16.4% | 2 m | 50.7% | 0.028 | 20.5% | 0.12 | 60.1% |
| 0.5-1 s | 1.2 m | 1.2% | 10 m | 43.1% | 0.032 | 5.1% | 0.50 | 60.5% |
| 1-2 s | 1.3 m | 0.1% | 44 m | 46.6% | 0.035 | 4.5% | 0.64 | 60.3% |
| 2-5 s | 1.3 m | 0.0% | 197 m | 29.7% | 0.034 | 4.3% | 0.58 | 60.6% |
| 5-10 s | 1.4 m | 0.0% | 370 m | 11.1% | 0.034 | 4.9% | 0.52 | 62.1% |
| 10-30 s | 1.4 m | 0.0% | 523 m | 7.7% | 0.034 | 4.8% | 0.52 | 59.7% |

### Next state one tick ahead: KL to the sharp target (nats per ship), by run

Head: the next-state head. Persistence: the baseline code log(code_t + 1e-3) (today's). Dead reckoned: the same floor around the sharp code of x + v dt, heading + nose rate dt. Rows alive and supervised at t.


**ally** (455101 rows)

| run | head | persistence | dead reckoned | dead reckoned, heading held |
|---|---|---|---|---|
| position | 0.938 | 6.056 | 0.049 | 0.049 |
| velocity | 0.915 | 1.712 | 1.712 | 1.712 |
| heading | 0.957 | 4.092 | 2.308 | 4.092 |
| nose_rate | 1.339 | 4.011 | 4.011 | 4.011 |
| angle_of_attack | 0.400 | 0.906 | 0.906 | 0.906 |
| shield_recharge_wait | 0.037 | 0.057 | 0.057 | 0.057 |
| shield | 0.042 | 0.058 | 0.058 | 0.058 |
| energy | 0.028 | 0.036 | 0.036 | 0.036 |
| log_index | 0.020 | 0.039 | 0.039 | 0.039 |
| cooldown | 0.030 | 0.213 | 0.213 | 0.213 |
| **total** | 4.71 | 17.18 | 9.39 | 11.17 |

**enemy visible** (223337 rows)

| run | head | persistence | dead reckoned | dead reckoned, heading held |
|---|---|---|---|---|
| position | 0.898 | 6.030 | 0.049 | 0.049 |
| velocity | 0.983 | 1.977 | 1.977 | 1.977 |
| heading | 0.840 | 4.682 | 1.884 | 4.682 |
| nose_rate | 1.151 | 3.362 | 3.362 | 3.362 |
| angle_of_attack | 0.327 | 0.736 | 0.736 | 0.736 |
| shield_recharge_wait | 0.074 | 0.095 | 0.095 | 0.095 |
| shield | 0.085 | 0.097 | 0.097 | 0.097 |
| energy | 0.045 | 0.052 | 0.052 | 0.052 |
| log_index | 0.009 | 0.028 | 0.028 | 0.028 |
| cooldown | 0.060 | 0.429 | 0.429 | 0.429 |
| **total** | 4.47 | 17.49 | 8.71 | 11.51 |

### Kinematic residual, position: x_(t+1) - (x_t + v_t dt) (m), own ships, one tick (dt = 0.03333 s)

| situation | rows | quantity | mean | 50% | 90% | 99% | 99.9% |
|---|---|---|---|---|---|---|---|
| all | 454932 | |r| | 0.0263 | 0.0220 | 0.0523 | 0.0872 | 0.2161 |
| all | 454932 | |r_x| | 0.0168 | 0.0115 | 0.0382 | 0.0725 | 0.1601 |
| all | 454932 | |r_y| | 0.0167 | 0.0115 | 0.0381 | 0.0723 | 0.1629 |
| in a field | 81251 | |r| | 0.0220 | 0.0097 | 0.0567 | 0.1775 | 0.2943 |
| in a field | 81251 | |r_x| | 0.0141 | 0.0058 | 0.0347 | 0.1279 | 0.2501 |
| in a field | 81251 | |r_y| | 0.0138 | 0.0053 | 0.0351 | 0.1262 | 0.2540 |
| not in a field | 373681 | |r| | 0.0273 | 0.0246 | 0.0519 | 0.0777 | 0.0869 |
| not in a field | 373681 | |r_x| | 0.0173 | 0.0131 | 0.0385 | 0.0659 | 0.0808 |
| not in a field | 373681 | |r_y| | 0.0174 | 0.0132 | 0.0384 | 0.0657 | 0.0804 |
| in a zone | 83723 | |r| | 0.0188 | 0.0133 | 0.0393 | 0.0855 | 0.1966 |
| in a zone | 83723 | |r_x| | 0.0120 | 0.0078 | 0.0272 | 0.0643 | 0.1552 |
| in a zone | 83723 | |r_y| | 0.0119 | 0.0077 | 0.0272 | 0.0632 | 0.1384 |
| not in a zone | 371209 | |r| | 0.0281 | 0.0246 | 0.0542 | 0.0875 | 0.2221 |
| not in a zone | 371209 | |r_x| | 0.0178 | 0.0128 | 0.0399 | 0.0734 | 0.1606 |
| not in a zone | 371209 | |r_y| | 0.0178 | 0.0128 | 0.0398 | 0.0732 | 0.1670 |
| coast (power 0) | 374069 | |r| | 0.0259 | 0.0210 | 0.0521 | 0.0897 | 0.2229 |
| coast (power 0) | 374069 | |r_x| | 0.0164 | 0.0109 | 0.0381 | 0.0732 | 0.1655 |
| coast (power 0) | 374069 | |r_y| | 0.0165 | 0.0109 | 0.0381 | 0.0728 | 0.1701 |
| boost (power 1) | 16934 | |r| | 0.0282 | 0.0235 | 0.0519 | 0.0974 | 0.2129 |
| boost (power 1) | 16934 | |r_x| | 0.0179 | 0.0139 | 0.0378 | 0.0731 | 0.1430 |
| boost (power 1) | 16934 | |r_y| | 0.0180 | 0.0136 | 0.0387 | 0.0783 | 0.1563 |
| reverse (power 2) | 63929 | |r| | 0.0286 | 0.0257 | 0.0534 | 0.0807 | 0.1430 |
| reverse (power 2) | 63929 | |r_x| | 0.0183 | 0.0145 | 0.0392 | 0.0693 | 0.1051 |
| reverse (power 2) | 63929 | |r_y| | 0.0181 | 0.0144 | 0.0380 | 0.0688 | 0.1034 |

### Kinematic residual, position: x_(t+1) - (x_t + v_{t+1} dt) (m), own ships, one tick (dt = 0.03333 s)

| situation | rows | quantity | mean | 50% | 90% | 99% | 99.9% |
|---|---|---|---|---|---|---|---|
| all | 454932 | |r| | 0.0320 | 0.0266 | 0.0643 | 0.1133 | 0.2726 |
| all | 454932 | |r_x| | 0.0203 | 0.0139 | 0.0463 | 0.0877 | 0.2130 |
| all | 454932 | |r_y| | 0.0203 | 0.0139 | 0.0464 | 0.0872 | 0.2139 |

### Kinematic residual, position: x_(t+1) - (x_t + (v_t + v_{t+1})/2 dt) (m), own ships, one tick (dt = 0.03333 s)

| situation | rows | quantity | mean | 50% | 90% | 99% | 99.9% |
|---|---|---|---|---|---|---|---|
| all | 454932 | |r| | 0.0082 | 0.0036 | 0.0217 | 0.0388 | 0.1073 |
| all | 454932 | |r_x| | 0.0052 | 0.0019 | 0.0154 | 0.0324 | 0.0865 |
| all | 454932 | |r_y| | 0.0052 | 0.0019 | 0.0154 | 0.0323 | 0.0785 |

### Kinematic residual, heading: |heading_(t+1) - (heading_t + nose_rate_t dt)| (rad)

| situation | rows | mean | 50% | 90% | 99% | 99.9% |
|---|---|---|---|---|---|---|
| all | 454932 | 0.0172 | 0.0113 | 0.0431 | 0.0715 | 0.1622 |
| in a field | 81251 | 0.0146 | 0.0065 | 0.0339 | 0.1332 | 0.1979 |
| not in a field | 373681 | 0.0178 | 0.0128 | 0.0440 | 0.0677 | 0.0745 |
| in a zone | 83723 | 0.0151 | 0.0099 | 0.0352 | 0.0687 | 0.1470 |
| not in a zone | 371209 | 0.0177 | 0.0116 | 0.0452 | 0.0717 | 0.1661 |
| coast (power 0) | 374069 | 0.0186 | 0.0128 | 0.0453 | 0.0726 | 0.1664 |
| boost (power 1) | 16934 | 0.0179 | 0.0120 | 0.0449 | 0.0717 | 0.1522 |
| reverse (power 2) | 63929 | 0.0092 | 0.0032 | 0.0265 | 0.0616 | 0.1058 |

### Kinematic residual, heading: |heading_(t+1) - (heading_t + nose_rate_(t+1) dt)| (rad)

| situation | rows | mean | 50% | 90% | 99% | 99.9% |
|---|---|---|---|---|---|---|
| all | 454932 | 0.0112 | 0.0079 | 0.0278 | 0.0432 | 0.0704 |

### Kinematic residual, heading: |heading_(t+1) - (heading_t + mean nose rate dt)| (rad)

| situation | rows | mean | 50% | 90% | 99% | 99.9% |
|---|---|---|---|---|---|---|
| all | 454932 | 0.0030 | 0.0017 | 0.0078 | 0.0145 | 0.0456 |


# Fixed decoder in the loop

## deathmatch, fixed decoder in the loop

### Hidden enemies: position error (m), mean / median

| hidden for | count | belief | persistence | dead reckoning | DR clamped to arena | DR unpowered coast (zero-lift drag) |
|---|---|---|---|---|---|---|
| 0.1-0.5 s | 8697 | 30 / 13 | 14 / 12 | 2 / 1 | 2 / 1 | 2 / 1 |
| 0.5-1 s | 10082 | 487 / 208 | 38 / 34 | 12 / 9 | 12 / 9 | 12 / 9 |
| 1-2 s | 18388 | 1711 / 1434 | 76 / 71 | 39 / 33 | 39 / 33 | 39 / 33 |
| 2-5 s | 47112 | 3259 / 2108 | 171 / 153 | 154 / 136 | 151 / 134 | 152 / 136 |
| 5-10 s | 61504 | 4734 / 2719 | 438 / 428 | 476 / 444 | 461 / 432 | 466 / 440 |
| 10-30 s | 93412 | 5761 / 3686 | 1149 / 1067 | 1237 / 1144 | 1163 / 1091 | 1209 / 1130 |

### Hidden enemies: heading error (rad), mean / median (dead reckoning holds the heading: = persistence)

| hidden for | belief | persistence / dead reckoning |
|---|---|---|
| 0.1-0.5 s | 0.86 / 0.54 | 0.16 / 0.12 |
| 0.5-1 s | 1.39 / 1.29 | 0.49 / 0.45 |
| 1-2 s | 1.39 / 1.32 | 0.98 / 0.97 |
| 2-5 s | 1.34 / 1.27 | 1.55 / 1.57 |
| 5-10 s | 1.31 / 1.25 | 1.66 / 1.76 |
| 10-30 s | 1.29 / 1.21 | 1.66 / 1.74 |

### Hidden enemies: the belief's sigma with the fixed decoder in the loop (both column pairs read the fed-back belief / the same forecasts; they agree, as they should)

Within 1 sigma: expected 63% (position, 2-D) and 68% (heading). Here the loop fed back the fixed decoder's sigma.

| hidden for | pos sigma median (v1) | within 1 sigma (v1) | pos sigma median (fixed) | within 1 sigma (fixed) | heading sigma median (v1) | within (v1) | heading sigma median (fixed) | within (fixed) |
|---|---|---|---|---|---|---|---|---|
| 0.1-0.5 s | 15.8 m | 80.8% | 16 m | 81.0% | 1.059 | 65.5% | 1.05 | 65.5% |
| 0.5-1 s | 320.1 m | 83.2% | 320 m | 83.2% | 0.778 | 32.5% | 0.78 | 32.3% |
| 1-2 s | 1886.4 m | 75.4% | 1886 m | 75.4% | 0.519 | 18.9% | 0.52 | 19.0% |
| 2-5 s | 2130.6 m | 72.6% | 2131 m | 72.6% | 0.463 | 15.9% | 0.46 | 16.0% |
| 5-10 s | 2028.7 m | 53.1% | 2029 m | 53.1% | 0.425 | 14.9% | 0.43 | 14.9% |
| 10-30 s | 1802.7 m | 39.7% | 1803 m | 39.7% | 0.390 | 14.2% | 0.39 | 14.2% |


## deathmatch_wide, fixed decoder in the loop

### Hidden enemies: position error (m), mean / median

| hidden for | count | belief | persistence | dead reckoning | DR clamped to arena | DR unpowered coast (zero-lift drag) |
|---|---|---|---|---|---|---|
| 0.1-0.5 s | 5102 | 50 / 19 | 17 / 16 | 2 / 1 | 5 / 1 | 2 / 1 |
| 0.5-1 s | 5452 | 580 / 220 | 45 / 43 | 12 / 9 | 16 / 9 | 12 / 9 |
| 1-2 s | 10057 | 1774 / 1478 | 87 / 84 | 41 / 33 | 43 / 33 | 41 / 33 |
| 2-5 s | 23791 | 3485 / 2703 | 179 / 165 | 167 / 144 | 167 / 144 | 161 / 140 |
| 5-10 s | 22331 | 5207 / 4306 | 414 / 405 | 486 / 445 | 486 / 444 | 454 / 419 |
| 10-30 s | 39653 | 7231 / 6569 | 1104 / 1049 | 1372 / 1214 | 1371 / 1212 | 1224 / 1131 |

### Hidden enemies: heading error (rad), mean / median (dead reckoning holds the heading: = persistence)

| hidden for | belief | persistence / dead reckoning |
|---|---|---|
| 0.1-0.5 s | 1.07 / 0.86 | 0.18 / 0.13 |
| 0.5-1 s | 1.45 / 1.39 | 0.49 / 0.45 |
| 1-2 s | 1.49 / 1.45 | 0.98 / 0.93 |
| 2-5 s | 1.53 / 1.49 | 1.48 / 1.45 |
| 5-10 s | 1.53 / 1.47 | 1.50 / 1.46 |
| 10-30 s | 1.54 / 1.53 | 1.57 / 1.58 |

### Hidden enemies: the belief's sigma with the fixed decoder in the loop (both column pairs read the fed-back belief / the same forecasts; they agree, as they should)

Within 1 sigma: expected 63% (position, 2-D) and 68% (heading). Here the loop fed back the fixed decoder's sigma.

| hidden for | pos sigma median (v1) | within 1 sigma (v1) | pos sigma median (fixed) | within 1 sigma (fixed) | heading sigma median (v1) | within (v1) | heading sigma median (fixed) | within (fixed) |
|---|---|---|---|---|---|---|---|---|
| 0.1-0.5 s | 21.9 m | 73.4% | 22 m | 73.6% | 1.154 | 56.5% | 1.16 | 56.5% |
| 0.5-1 s | 302.9 m | 75.7% | 304 m | 75.5% | 0.723 | 27.7% | 0.72 | 27.7% |
| 1-2 s | 1821.4 m | 65.4% | 1824 m | 65.3% | 0.514 | 19.1% | 0.51 | 19.1% |
| 2-5 s | 2208.0 m | 65.5% | 2208 m | 65.5% | 0.463 | 16.7% | 0.46 | 16.7% |
| 5-10 s | 2234.5 m | 45.0% | 2235 m | 45.0% | 0.440 | 17.3% | 0.44 | 17.2% |
| 10-30 s | 2063.3 m | 27.9% | 2062 m | 27.9% | 0.402 | 15.1% | 0.40 | 15.2% |


## frontline, fixed decoder in the loop

### Hidden enemies: position error (m), mean / median

| hidden for | count | belief | persistence | dead reckoning | DR clamped to arena | DR unpowered coast (zero-lift drag) |
|---|---|---|---|---|---|---|
| 0.1-0.5 s | 7626 | 32 / 9 | 12 / 10 | 1 / 1 | 1 / 1 | 1 / 1 |
| 0.5-1 s | 9003 | 494 / 218 | 32 / 29 | 7 / 5 | 7 / 5 | 7 / 5 |
| 1-2 s | 17105 | 1670 / 1367 | 70 / 63 | 25 / 20 | 25 / 20 | 25 / 21 |
| 2-5 s | 46534 | 3072 / 1878 | 192 / 178 | 101 / 86 | 101 / 86 | 103 / 88 |
| 5-10 s | 66503 | 4830 / 2891 | 486 / 469 | 288 / 270 | 288 / 270 | 301 / 284 |
| 10-30 s | 82817 | 5643 / 4707 | 1028 / 976 | 647 / 596 | 647 / 596 | 695 / 641 |

### Hidden enemies: heading error (rad), mean / median (dead reckoning holds the heading: = persistence)

| hidden for | belief | persistence / dead reckoning |
|---|---|---|
| 0.1-0.5 s | 0.57 / 0.24 | 0.10 / 0.07 |
| 0.5-1 s | 1.10 / 0.92 | 0.27 / 0.24 |
| 1-2 s | 1.18 / 1.08 | 0.42 / 0.37 |
| 2-5 s | 1.18 / 1.13 | 0.40 / 0.30 |
| 5-10 s | 1.19 / 1.15 | 0.31 / 0.21 |
| 10-30 s | 1.18 / 1.13 | 0.35 / 0.22 |

### Hidden enemies: the belief's sigma with the fixed decoder in the loop (both column pairs read the fed-back belief / the same forecasts; they agree, as they should)

Within 1 sigma: expected 63% (position, 2-D) and 68% (heading). Here the loop fed back the fixed decoder's sigma.

| hidden for | pos sigma median (v1) | within 1 sigma (v1) | pos sigma median (fixed) | within 1 sigma (fixed) | heading sigma median (v1) | within (v1) | heading sigma median (fixed) | within (fixed) |
|---|---|---|---|---|---|---|---|---|
| 0.1-0.5 s | 12.5 m | 85.3% | 12 m | 85.5% | 0.805 | 72.3% | 0.80 | 72.1% |
| 0.5-1 s | 347.7 m | 86.9% | 348 m | 86.9% | 0.818 | 45.0% | 0.82 | 44.8% |
| 1-2 s | 1678.8 m | 74.0% | 1680 m | 73.9% | 0.531 | 26.9% | 0.53 | 26.9% |
| 2-5 s | 2091.0 m | 71.9% | 2091 m | 71.9% | 0.465 | 20.8% | 0.46 | 20.7% |
| 5-10 s | 1971.0 m | 49.0% | 1971 m | 49.0% | 0.389 | 17.2% | 0.39 | 17.2% |
| 10-30 s | 1746.9 m | 39.3% | 1746 m | 39.2% | 0.332 | 17.2% | 0.33 | 17.2% |

