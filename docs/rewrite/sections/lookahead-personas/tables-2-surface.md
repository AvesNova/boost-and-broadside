## The surface: fit

| lambda | per size | shared | per size explained | shared explained |
|---|---|---|---|---|
| 30.0 | 0.66164 | 0.66181 | 80.4% | 80.0% |
| 100.0 | 0.66074 | 0.66118 | 82.7% | 81.6% |
| 300.0 | 0.66080 | 0.66137 | 82.6% | 81.1% |

Coin 0.69315, saturated 0.65398 per game; agents 517, games {'1': 233728, '4': 233728, '16': 160000}.

## Peak, LCB optimum, robust centre (relative Elo per size: offsets arbitrary)

| point | mean over sizes | 1v1 / 4v4 / 16v16 | SD |
|---|---|---|---|
| peak | 673 | 329 / 537 / 1154 | 15 |
| lcb | 673 | 329 / 537 / 1154 | 15 |
| centre | 632 | 286 / 489 / 1122 | 10 |

## Rows: old yardstick, robust centre, peak, the 50-Elo region (5-95%)

| row | old yardstick | robust centre | peak | 50-Elo region |
|---|---|---|---|---|
| threat_weight | 1 | 0.573 (u -0.43) | 0.548 (u -0.50) | 0.546 - 0.625 (u -0.51 - -0.28) |
| angles_weight | 0.3 | 0.509 (u 1.16) | 0.644 (u 1.45) | 0.325 - 0.671 (u 0.60 - 1.51) |
| energy_weight | 0.25 | 0.305 (u 0.75) | 0.317 (u 0.80) | 0.235 - 0.412 (u 0.42 - 1.12) |
| aggression | 0.5 | 0.924 (u 1.11) | 0.616 (u 0.71) | 0.509 - 1.4 (u 0.52 - 1.53) |
| speed_weight | 0.3 | 0.623 (u 1.41) | 0.665 (u 1.49) | 0.544 - 0.673 (u 1.24 - 1.51) |
| collision_weight | 3 | 1.63 (u -0.52) | 1.6 (u -0.55) | 1.59 - 1.71 (u -0.56 - -0.43) |
| boundary_weight | 2 | 1.98 (u 0.49) | 1.52 (u 0.05) | 1.52 - 2.53 (u 0.05 - 0.89) |
| field_weight | 0.3 | 0.402 (u 0.87) | 0.493 (u 1.12) | 0.3 - 0.491 (u 0.50 - 1.12) |
| horizon | 1.5 | 0.806 (u -0.54) | 0.789 (u -0.57) | 0.785 - 0.848 (u -0.58 - -0.45) |
| preferred_speed | 0.9 | 0.87 (u 0.39) | 0.915 (u 0.56) | 0.786 - 0.967 (u 0.05 - 0.74) |
| reply_temperature | 0.2 | 0.0857 (u -0.35) | 0.0736 (u -0.50) | 0.0728 - 0.12 (u -0.51 - -0.01) |
| caution | 0 | 1.66 (u 0.91) | 3.95 (u 1.49) | -1.12 - 3.89 (u 0.22 - 1.47) |
| temperature | 0.05 | 0.0223 (u -0.51) | 0.0207 (u -0.60) | 0.0206 - 0.0258 (u -0.61 - -0.33) |
| stickiness | 0.05 | 0.0437 (u 0.37) | 0.0514 (u 0.53) | 0.0349 - 0.0559 (u 0.14 - 0.61) |
| store_weight | 0.5 | 0.283 (u -0.21) | 0.312 (u -0.09) | 0.224 - 0.383 (u -0.50 - 0.17) |
| power_horizon | 3 | 5.13 (u 1.39) | 4.79 (u 1.28) | 3.99 - 5.74 (u 0.98 - 1.58) |
| power_temperature | 0.05 | 0.105 (u 1.43) | 0.118 (u 1.57) | 0.0826 - 0.119 (u 1.13 - 1.58) |
| power_stickiness | 0.3 | 0.2 (u 0.09) | 0.225 (u 0.21) | 0.119 - 0.371 (u -0.42 - 0.71) |
| disengaged_energy_weight | 0.5 | 0.358 (u 0.17) | 0.287 (u -0.06) | 0.236 - 0.644 (u -0.25 - 0.75) |
| seek_weight | 0.1 | 0.0776 (u 0.32) | 0.054 (u 0.06) | 0.0538 - 0.154 (u 0.06 - 0.81) |
| fire_tolerance | 1 | 1.85 (u 1.53) | 1.89 (u 1.57) | 1.75 - 1.91 (u 1.43 - 1.57) |
| angles_weight_crowd | - | v -0.81 | v -0.98 | v -1.57 - +0.05 |
| energy_weight_crowd | - | v -0.54 | v -0.78 | v -1.03 - +0.04 |
| aggression_crowd | - | v -1.83 | v -1.58 | v -2.35 - -0.92 |
| speed_weight_crowd | - | v +0.86 | v +0.87 | v +0.32 - +1.27 |
| preferred_speed_crowd | - | v +1.22 | v +1.21 | v +0.44 - +1.75 |
| reply_temperature_crowd | - | v +1.82 | v +2.38 | v +1.19 - +2.32 |
| store_crowd | - | v -0.48 | v +0.39 | v -2.97 - +1.18 |

Crowd rows (v): the crowd value is the duel value x e^(2 width v); store_crowd k = e^(2 v).

## Directions at the centre: curvature (Elo per box^2) and the largest loadings

| curvature | loadings |
|---|---|
| +0.6 | caution -0.44, power_horizon -0.39, energy_weight_crowd -0.37, angles_weight_crowd -0.31 |
| -0.7 | store_crowd -0.81, store_weight -0.33, reply_temperature_crowd +0.22, preferred_speed_crowd +0.20 |
| -2.2 | field_weight -0.47, aggression_crowd -0.42, disengaged_energy_weight +0.36, aggression +0.29 |
| +3.1 | angles_weight_crowd +0.46, power_horizon -0.43, power_temperature -0.30, threat_weight -0.29 |
| -3.2 | aggression_crowd -0.43, field_weight +0.41, store_crowd -0.34, preferred_speed_crowd -0.32 |
| -300.7 | energy_weight -0.58, angles_weight +0.39, energy_weight_crowd -0.39, speed_weight -0.28 |
| -178.9 | stickiness -0.59, preferred_speed +0.49, temperature +0.33, speed_weight +0.29 |
| -137.0 | preferred_speed +0.64, stickiness +0.45, preferred_speed_crowd +0.42, temperature -0.30 |

## 1v1: largest interactions (Elo per box^2), Sobol on the surface

| product | coefficient |
|---|---|
| angles_weight x aggression | -58 +- 13 |
| speed_weight x reply_temperature | -57 +- 12 |
| angles_weight x speed_weight | +49 +- 13 |
| speed_weight x temperature | -44 +- 12 |
| angles_weight x reply_temperature | +39 +- 12 |
| energy_weight x stickiness | +38 +- 12 |

| row | S1 | ST | ST - S1 |
|---|---|---|---|
| angles_weight | 0.328 | 0.346 | +0.018 |
| temperature | 0.162 | 0.171 | +0.009 |
| aggression | 0.146 | 0.166 | +0.020 |
| energy_weight | 0.073 | 0.083 | +0.010 |
| reply_temperature | 0.069 | 0.082 | +0.013 |
| collision_weight | 0.029 | 0.033 | +0.004 |
| stickiness | 0.019 | 0.031 | +0.012 |
| horizon | 0.028 | 0.029 | +0.002 |
| disengaged_energy_weight | 0.023 | 0.028 | +0.005 |
| speed_weight | 0.008 | 0.028 | +0.019 |

Model Elo sd over the box: 66.

## 4v4: largest interactions (Elo per box^2), Sobol on the surface

| product | coefficient |
|---|---|
| angles_weight x aggression | -43 +- 14 |
| energy_weight x stickiness | -42 +- 12 |
| angles_weight x energy_weight | +41 +- 13 |
| energy_weight x speed_weight | -41 +- 12 |
| boundary_weight x fire_tolerance | -36 +- 14 |
| preferred_speed x seek_weight | -31 +- 15 |

| row | S1 | ST | ST - S1 |
|---|---|---|---|
| speed_weight | 0.221 | 0.229 | +0.008 |
| fire_tolerance | 0.146 | 0.149 | +0.003 |
| horizon | 0.095 | 0.098 | +0.002 |
| energy_weight | 0.075 | 0.089 | +0.014 |
| collision_weight | 0.077 | 0.080 | +0.003 |
| aggression | 0.069 | 0.075 | +0.006 |
| temperature | 0.065 | 0.070 | +0.005 |
| preferred_speed | 0.036 | 0.040 | +0.004 |
| store_weight | 0.035 | 0.040 | +0.005 |
| angles_weight | 0.026 | 0.034 | +0.008 |

Model Elo sd over the box: 67.

## 16v16: largest interactions (Elo per box^2), Sobol on the surface

| product | coefficient |
|---|---|
| angles_weight x energy_weight | +103 +- 13 |
| energy_weight x speed_weight | -67 +- 12 |
| energy_weight x horizon | +55 +- 14 |
| caution x stickiness | +54 +- 13 |
| angles_weight x horizon | -54 +- 14 |
| energy_weight x stickiness | -53 +- 13 |

| row | S1 | ST | ST - S1 |
|---|---|---|---|
| horizon | 0.157 | 0.157 | -0.000 |
| fire_tolerance | 0.152 | 0.153 | +0.001 |
| angles_weight | 0.130 | 0.135 | +0.005 |
| speed_weight | 0.120 | 0.121 | +0.001 |
| energy_weight | 0.106 | 0.110 | +0.005 |
| aggression | 0.103 | 0.108 | +0.005 |
| preferred_speed | 0.045 | 0.046 | +0.001 |
| store_weight | 0.041 | 0.042 | +0.001 |
| temperature | 0.039 | 0.039 | -0.000 |
| collision_weight | 0.028 | 0.028 | -0.000 |

Model Elo sd over the box: 172.

## Working ranges: u where f is within 50 / 100 Elo of the row's best (slice through the centre)

| row | 1v1 | 4v4 | 16v16 |
|---|---|---|---|
| threat_weight | best 0.91; -0.50-0.91 / -0.50-0.91 | best -0.50; -0.50-0.45 / -0.50-0.91 | best -0.50; -0.50-0.12 / -0.50-0.61 |
| angles_weight | best 1.50; 0.26-1.50 / 0.14-1.50 | best 1.00; 0.05-1.00 / 0.05-1.00 | best 0.66; -0.44-0.84 / -0.44-0.84 |
| energy_weight | best 0.68; 0.12-1.47 / 0.12-1.47 | best 0.58; 0.12-1.47 / 0.12-1.47 | best 0.12; 0.12-0.73 / 0.12-1.01 |
| aggression | best 1.21; 0.21-1.59 / 0.21-1.59 | best 0.02; -0.04-0.95 / -0.04-0.95 | best -0.67; -0.67-0.46 / -0.67-0.83 |
| speed_weight | best 1.49; 0.79-1.49 / 0.32-1.49 | best 1.99; 1.12-1.99 / 0.65-1.99 | best 2.32; 1.24-2.46 / 0.82-2.46 |
| collision_weight | best -0.55; -0.55-0.30 / -0.55-0.87 | best -0.55; -0.55-0.09 / -0.55-0.64 | best -0.55; -0.55-0.16 / -0.55-0.82 |
| boundary_weight | best 0.05; 0.05-0.92 / 0.05-0.92 | best 0.92; 0.05-0.92 / 0.05-0.92 | best 0.92; 0.05-0.92 / 0.05-0.92 |
| field_weight | best 1.12; 0.12-1.12 / 0.12-1.12 | best 0.52; 0.12-1.12 / 0.12-1.12 | best 1.12; 0.12-1.12 / 0.12-1.12 |
| horizon | best 0.82; -0.05-0.82 / -0.57-0.82 | best -0.57; -0.57--0.03 / -0.57-0.42 | best -0.57; -0.57--0.31 / -0.57--0.08 |
| preferred_speed | best 0.85; -0.47-0.85 / -0.47-0.85 | best 0.88; 0.06-1.13 / 0.06-1.13 | best 1.73; 0.84-1.73 / 0.46-1.73 |
| reply_temperature | best -0.50; -0.50-0.14 / -0.50-0.74 | best 1.02; 0.04-1.02 / 0.04-1.02 | best 1.58; 0.14-1.58 / 0.14-1.58 |
| caution | best 0.45; 0.13-1.50 / 0.13-1.50 | best 1.50; 0.13-1.50 / 0.13-1.50 | best 1.50; 0.13-1.50 / 0.13-1.50 |
| temperature | best -0.60; -0.60--0.09 / -0.60-0.29 | best -0.51; -0.60-0.62 / -0.60-0.85 | best -0.37; -0.60-0.60 / -0.60-0.85 |
| stickiness | best 0.46; 0.14-1.57 / 0.14-1.57 | best 0.52; 0.14-1.29 / 0.14-1.57 | best 0.19; 0.14-0.86 / 0.14-1.12 |
| store_weight | best -0.50; -0.50-0.89 / -0.50-0.89 | best -0.16; -0.56-0.89 / -0.56-0.89 | best -0.27; -0.61-0.89 / -0.61-0.89 |
| power_horizon | best 1.57; 0.13-1.57 / 0.13-1.57 | best 1.57; 0.13-1.57 / 0.13-1.57 | best 1.57; 0.13-1.57 / 0.13-1.57 |
| power_temperature | best 1.57; 0.17-1.57 / 0.17-1.57 | best 1.57; 0.17-1.57 / 0.17-1.57 | best 1.57; 0.17-1.57 / 0.17-1.57 |
| power_stickiness | best -0.11; -0.48-0.86 / -0.48-0.86 | best 0.86; -0.48-0.86 / -0.48-0.86 | best -0.48; -0.48-0.86 / -0.48-0.86 |
| disengaged_energy_weight | best 0.86; -0.27-0.86 / -0.27-0.86 | best -0.09; -0.27-0.86 / -0.27-0.86 | best -0.22; -0.27-0.86 / -0.27-0.86 |
| seek_weight | best 0.06; 0.06-0.93 / 0.06-0.93 | best 0.06; 0.06-0.93 / 0.06-0.93 | best 0.06; 0.06-0.93 / 0.06-0.93 |
| fire_tolerance | best 1.56; 0.18-1.56 / 0.18-1.56 | best 1.56; 0.84-1.56 / 0.20-1.56 | best 1.56; 1.12-1.56 / 0.74-1.56 |

## Proposed persona box (around the robust centre; inert rows held)

| row | law | round-1 width | proposed width | half-width (box u) |
|---|---|---|---|---|
| threat_weight | log | 0.3 | 0.186 | 0.31 |
| angles_weight | log | 0.4 | 0.381 | 0.48 |
| energy_weight | log | 0.4 | 0.242 | 0.30 |
| aggression | log | 0.5 | 0.492 | 0.49 |
| speed_weight | log | 0.4 | 0.283 | 0.35 |
| collision_weight | log | 0.3 | 0.192 | 0.32 |
| horizon | log | 0.3 | 0.0782 | 0.13 |
| preferred_speed | log | 0.15 | 0.133 | 0.44 |
| reply_temperature | log | 0.5 | 0.322 | 0.32 |
| caution | linear | 2 | 2 | 0.50 |
| temperature | log | 0.4 | 0.204 | 0.25 |
| stickiness | log | 0.5 | 0.358 | 0.36 |
| store_weight | log | 0.4 | 0.4 | 0.50 |
| power_horizon | log | 0.3 | 0.3 | 0.50 |
| power_temperature | log | 0.4 | 0.4 | 0.50 |
| power_stickiness | log | 0.5 | 0.5 | 0.50 |
| disengaged_energy_weight | log | 0.5 | 0.5 | 0.50 |
| fire_tolerance | log | 0.3 | 0.135 | 0.22 |

League personas (surface predictions, relative Elo):

| persona | mean | 1v1 / 4v4 / 16v16 | SD |
|---|---|---|---|
| league1 | 626 | 308 / 482 / 1088 | 15 |
| league2 | 629 | 272 / 502 / 1113 | 11 |
| league3 | 624 | 256 / 492 / 1124 | 18 |
| league4 | 631 | 264 / 473 / 1155 | 15 |
| league5 | 626 | 270 / 480 / 1130 | 17 |
| league6 | 624 | 290 / 515 / 1067 | 13 |

