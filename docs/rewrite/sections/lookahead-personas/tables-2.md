# Look-ahead personas round 2: tables

## Step 1: game durations (fields + fog, yardstick mirror + 6 personas against the yardstick)

| size | games | cap | p50 / p80 / p90 / p95 | at the cap | draw wipes | wiped by 60 s | by 90 s | by 120 s | by 300 s | mirror: by 60 s / p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1v1 | 7168 | 600 s | 75 / 204 / 395 / 600 s | 6.5% | 5.1% | 39.7% | 57.3% | 66.7% | 86.5% | 38.5% / 600 s |
| 2v2 | 3584 | 300 s | 60 / 94 / 131 / 191 s | 2.5% | 0.5% | 50.1% | 77.8% | 87.9% | 97.5% | 53.3% / 129 s |
| 4v4 | 3584 | 300 s | 49 / 66 / 81 / 105 s | 1.5% | 0.1% | 72.3% | 92.9% | 95.9% | 98.5% | 72.1% / 80 s |
| 8v8 | 2688 | 300 s | 48 / 61 / 71 / 82 s | 0.7% | 0.1% | 79.0% | 96.7% | 98.4% | 99.3% | 76.8% / 71 s |
| 16v16 | 2688 | 300 s | 53 / 66 / 76 / 86 s | 0.3% | 0.0% | 67.7% | 96.3% | 98.8% | 99.7% | 63.0% / 80 s |

## Step 3: crowd measures (yardstick mirror, every 15 ticks, living ships; mean / share with m <= 1 / mean weight at crowd_half 1)

| size | near (2 gun ranges) | known enemies | own living ships |
|---|---|---|---|
| 1v1 | 0.28 / 1.00 / 0.00 | 0.29 / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 |
| 2v2 | 0.67 / 0.84 / 0.08 | 0.92 / 0.75 / 0.12 | 1.87 / 0.13 / 0.43 |
| 4v4 | 1.48 / 0.55 / 0.27 | 2.33 / 0.26 / 0.46 | 3.49 / 0.04 / 0.68 |
| 8v8 | 2.29 / 0.34 / 0.43 | 5.33 / 0.05 / 0.76 | 6.52 / 0.01 / 0.82 |
| 16v16 | 2.75 / 0.22 / 0.53 | 11.12 / 0.02 / 0.87 | 12.26 / 0.00 / 0.90 |

## Step 2: skill per size (fields + fog, study clocks; 106 agents against the yardstick)

## Box edges (one row at u = 0 or 1, the rest at the default)

| row | 1v1 u=0 | 1v1 u=1 | 4v4 u=0 | 4v4 u=1 | 16v16 u=0 | 16v16 u=1 |
|---|---|---|---|---|---|---|
| threat_weight | 1020 +- 10 | 1014 +- 10 | 1030 +- 10 | 987 +- 10 | 1056 +- 14 | 947 +- 14 |
| angles_weight | 909 +- 9 | 1064 +- 11 | 1010 +- 10 | 963 +- 10 | 1076 +- 14 | 871 +- 15 |
| energy_weight | 1045 +- 11 | 970 +- 10 | 955 +- 10 | 1027 +- 10 | 907 +- 14 | 1129 +- 15 |
| aggression | 941 +- 9 | 1036 +- 11 | 1007 +- 10 | 946 +- 10 | 1083 +- 14 | 887 +- 15 |
| speed_weight | 971 +- 10 | 998 +- 10 | 941 +- 10 | 1060 +- 10 | 925 +- 14 | 1088 +- 14 |
| collision_weight | 1036 +- 11 | 981 +- 10 | 1030 +- 10 | 968 +- 10 | 1029 +- 13 | 961 +- 13 |
| boundary_weight | 988 +- 9 | 999 +- 9 | 1006 +- 10 | 999 +- 10 | 1013 +- 13 | 1030 +- 13 |
| field_weight | 995 +- 10 | 993 +- 10 | 979 +- 10 | 1001 +- 10 | 1010 +- 13 | 988 +- 13 |
| horizon | 963 +- 11 | 1014 +- 9 | 1026 +- 10 | 987 +- 10 | 1124 +- 15 | 912 +- 14 |
| preferred_speed | 1013 +- 9 | 984 +- 10 | 976 +- 10 | 1013 +- 10 | 945 +- 14 | 1039 +- 13 |
| reply_temperature | 1024 +- 10 | 948 +- 9 | 992 +- 10 | 1020 +- 10 | 979 +- 13 | 1052 +- 14 |
| caution | 983 +- 10 | 992 +- 10 | 972 +- 10 | 1000 +- 10 | 999 +- 13 | 1023 +- 13 |
| temperature | 1029 +- 10 | 954 +- 11 | 1022 +- 10 | 946 +- 10 | 1043 +- 13 | 930 +- 14 |
| stickiness | 977 +- 10 | 1030 +- 9 | 992 +- 10 | 1003 +- 10 | 1010 +- 13 | 1012 +- 13 |
| store_weight | 1019 +- 9 | 996 +- 10 | 1030 +- 10 | 970 +- 10 | 1074 +- 14 | 961 +- 13 |
| power_horizon | 1001 +- 10 | 1003 +- 10 | 993 +- 10 | 1026 +- 10 | 1011 +- 13 | 1024 +- 13 |
| power_temperature | 974 +- 9 | 993 +- 10 | 985 +- 10 | 1016 +- 10 | 1035 +- 13 | 983 +- 13 |
| power_stickiness | 1008 +- 10 | 993 +- 10 | 999 +- 10 | 969 +- 10 | 994 +- 13 | 983 +- 13 |
| disengaged_energy_weight | 1029 +- 10 | 982 +- 10 | 989 +- 10 | 1019 +- 10 | 990 +- 13 | 984 +- 13 |
| seek_weight | 1003 +- 10 | 994 +- 10 | 999 +- 10 | 991 +- 10 | 1008 +- 13 | 1009 +- 13 |
| fire_tolerance | 993 +- 10 | 993 +- 10 | 963 +- 10 | 1025 +- 10 | 897 +- 14 | 1090 +- 14 |

## The population (64 personas, spread 1)

| size | personas | games (all agents) | mean Elo | sd observed | sd without error | mean error | min / 10% / median / 90% / max | above 1000 | best | draws mean / max | games at the clock |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1v1 | 64 | 131072 | 984 | 71 | 71 | 10 | 801 / 875 / 989 / 1072 / 1090 | 30 | persona24 | 8.7% / 11.9% | 11.9% |
| 4v4 | 64 | 131072 | 978 | 58 | 57 | 10 | 812 / 902 / 981 / 1039 / 1103 | 27 | persona38 | 0.1% / 0.5% | 6.2% |
| 16v16 | 64 | 65536 | 973 | 156 | 155 | 17 | 497 / 770 / 978 / 1150 / 1350 | 29 | persona15 | 0.0% / 0.2% | 6.8% |

| personas' Elo | r |
|---|---|
| 1v1 against 4v4 | +0.08 |
| 1v1 against 16v16 | -0.27 |
| 4v4 against 16v16 | +0.90 |

## Linear fit: Elo change from u = 0 to u = 1 per row (weighted least squares)

| row | 1v1 | 4v4 | 16v16 |
|---|---|---|---|
| threat_weight | +9 +- 11 | -29 +- 9 | -74 +- 16 |
| angles_weight | +138 +- 11 | -39 +- 9 | -222 +- 15 |
| energy_weight | -37 +- 12 | +62 +- 10 | +198 +- 18 |
| aggression | +86 +- 12 | -55 +- 9 | -186 +- 16 |
| speed_weight | +17 +- 11 | +101 +- 9 | +176 +- 15 |
| collision_weight | -47 +- 11 | -41 +- 9 | -78 +- 15 |
| boundary_weight | -6 +- 12 | -9 +- 9 | -2 +- 16 |
| field_weight | +12 +- 11 | +9 +- 9 | +14 +- 16 |
| horizon | +12 +- 12 | -69 +- 10 | -216 +- 16 |
| preferred_speed | -39 +- 13 | +17 +- 10 | +81 +- 17 |
| reply_temperature | -67 +- 12 | -2 +- 9 | +73 +- 15 |
| caution | -1 +- 12 | +12 +- 10 | +33 +- 16 |
| temperature | -104 +- 11 | -50 +- 9 | -83 +- 15 |
| stickiness | +25 +- 12 | +2 +- 9 | +29 +- 15 |
| store_weight | -27 +- 11 | -38 +- 9 | -102 +- 16 |
| power_horizon | +20 +- 13 | +18 +- 10 | -4 +- 17 |
| power_temperature | +15 +- 11 | -1 +- 9 | -16 +- 15 |
| power_stickiness | -22 +- 12 | -20 +- 10 | +6 +- 17 |
| disengaged_energy_weight | -43 +- 11 | +1 +- 9 | +4 +- 15 |
| seek_weight | +4 +- 12 | +8 +- 10 | -3 +- 17 |
| fire_tolerance | -0 +- 12 | +71 +- 9 | +184 +- 16 |
| R^2 / leave-one-out R^2 / points | 0.85 / 0.72 / 106 | 0.85 / 0.75 / 106 | 0.91 / 0.83 / 106 |

## Quadratic fit: slope at the centre (Elo per box) / curvature (Elo per box^2) / best u on [0, 1]

| row | 1v1 | 4v4 | 16v16 |
|---|---|---|---|
| threat_weight | +7 / +45 +- 42 / 1.00 | -27 / +23 +- 27 / 0.00 | -73 / -37 +- 39 / 0.00 |
| angles_weight | +139 / -39 +- 44 / 1.00 | -43 / -49 +- 29 / 0.06 | -234 / -132 +- 44 / 0.00 |
| energy_weight | -37 / +17 +- 44 / 0.00 | +59 / -14 +- 28 / 1.00 | +193 / -5 +- 42 / 1.00 |
| aggression | +102 / -15 +- 42 / 1.00 | -45 / -71 +- 27 / 0.18 | -181 / -139 +- 41 / 0.00 |
| speed_weight | +16 / -75 +- 45 / 0.61 | +100 / -34 +- 29 / 1.00 | +183 / -29 +- 42 / 1.00 |
| collision_weight | -38 / -34 +- 35 / 0.00 | -43 / -19 +- 22 / 0.00 | -89 / -65 +- 33 / 0.00 |
| boundary_weight | -2 / -8 +- 38 / 0.35 | -11 / +18 +- 26 / 0.00 | -2 / +2 +- 38 / 0.00 |
| field_weight | +13 / -44 +- 39 / 0.65 | +8 / -1 +- 25 / 1.00 | -3 / -19 +- 38 / 0.42 |
| horizon | +15 / -3 +- 42 / 1.00 | -67 / +25 +- 28 / 0.00 | -220 / +32 +- 42 / 0.00 |
| preferred_speed | -36 / +5 +- 40 / 0.00 | +22 / -14 +- 27 / 1.00 | +88 / -55 +- 40 / 1.00 |
| reply_temperature | -66 / -27 +- 41 / 0.00 | -2 / +22 +- 27 / 0.00 | +74 / +25 +- 39 / 1.00 |
| caution | -13 / -55 +- 43 / 0.38 | +9 / -40 +- 28 / 0.61 | +24 / -5 +- 40 / 1.00 |
| temperature | -113 / -34 +- 40 / 0.00 | -48 / -52 +- 25 / 0.04 | -90 / -39 +- 38 / 0.00 |
| stickiness | +23 / -5 +- 40 / 1.00 | +7 / -3 +- 26 / 1.00 | +39 / +11 +- 38 / 1.00 |
| store_weight | -15 / -58 +- 40 / 0.37 | -34 / -38 +- 27 / 0.05 | -104 / +4 +- 38 / 0.00 |
| power_horizon | +21 / +26 +- 46 / 1.00 | +22 / +16 +- 29 / 1.00 | -3 / +24 +- 42 / 0.00 |
| power_temperature | +16 / -12 +- 36 / 1.00 | +7 / +9 +- 24 / 1.00 | -4 / +16 +- 35 / 0.00 |
| power_stickiness | -18 / +23 +- 43 / 0.00 | -21 / -54 +- 27 / 0.31 | +2 / -14 +- 40 / 0.56 |
| disengaged_energy_weight | -47 / +23 +- 40 / 0.00 | +4 / +6 +- 25 / 1.00 | +5 / -72 +- 36 / 0.53 |
| seek_weight | +1 / +43 +- 43 / 1.00 | +2 / +5 +- 28 / 1.00 | -3 / +31 +- 41 / 0.00 |
| fire_tolerance | +9 / -9 +- 38 / 0.95 | +77 / -31 +- 25 / 1.00 | +190 / -43 +- 38 / 1.00 |
| R^2 / leave-one-out R^2 | 0.90 / 0.68 | 0.94 / 0.81 | 0.97 / 0.89 |

## Size dependence: linear slope differences (Elo per box), z = difference / SE

| row | 4v4 minus 1v1 | 16v16 minus 1v1 | 16v16 minus 4v4 | sign flips (2.5 SE) |
|---|---|---|---|---|
| threat_weight | -37 (z -2.6) | -83 (z -4.3) | -45 (z -2.5) |  |
| angles_weight | -176 (z -12.1) | -359 (z -18.7) | -183 (z -10.2) | yes |
| energy_weight | +98 (z +6.3) | +235 (z +10.9) | +137 (z +6.7) | yes |
| aggression | -140 (z -9.4) | -272 (z -13.7) | -132 (z -7.1) | yes |
| speed_weight | +84 (z +5.8) | +159 (z +8.4) | +75 (z +4.3) |  |
| collision_weight | +6 (z +0.4) | -31 (z -1.7) | -37 (z -2.2) |  |
| boundary_weight | -3 (z -0.2) | +4 (z +0.2) | +7 (z +0.4) |  |
| field_weight | -3 (z -0.2) | +1 (z +0.1) | +4 (z +0.2) |  |
| horizon | -81 (z -5.3) | -228 (z -11.3) | -147 (z -7.8) |  |
| preferred_speed | +56 (z +3.4) | +120 (z +5.7) | +65 (z +3.3) | yes |
| reply_temperature | +65 (z +4.4) | +140 (z +7.3) | +74 (z +4.2) | yes |
| caution | +13 (z +0.8) | +34 (z +1.7) | +21 (z +1.1) |  |
| temperature | +55 (z +3.7) | +21 (z +1.1) | -34 (z -1.9) |  |
| stickiness | -22 (z -1.5) | +4 (z +0.2) | +26 (z +1.5) |  |
| store_weight | -11 (z -0.8) | -75 (z -3.9) | -64 (z -3.5) |  |
| power_horizon | -1 (z -0.1) | -24 (z -1.1) | -22 (z -1.1) |  |
| power_temperature | -16 (z -1.1) | -31 (z -1.7) | -15 (z -0.9) |  |
| power_stickiness | +2 (z +0.1) | +28 (z +1.3) | +26 (z +1.3) |  |
| disengaged_energy_weight | +44 (z +3.0) | +46 (z +2.5) | +2 (z +0.1) |  |
| seek_weight | +3 (z +0.2) | -7 (z -0.4) | -11 (z -0.6) |  |
| fire_tolerance | +71 (z +4.8) | +184 (z +9.5) | +113 (z +6.3) |  |

## Style, 1v1 (64 personas against the yardstick)

| 1v1 | yardstick | min / median / max | sd | r with Elo |
|---|---|---|---|---|
| turning | 0.493 | 0.398 / 0.495 / 0.622 | 0.049 | +0.15 |
| two_circle | 0.655 | 0.533 / 0.648 / 0.764 | 0.056 | +0.62 |
| store | 0.912 | 0.724 / 0.888 / 0.955 | 0.068 | -0.27 |
| speed | 0.997 | 0.801 / 0.999 / 1.135 | 0.058 | -0.35 |
| shots | 13.910 | 6.820 / 12.335 / 23.278 | 3.379 | +0.58 |
| hit_range | 0.205 | 0.164 / 0.194 / 0.241 | 0.015 | -0.18 |
| climb | 0.464 | 0.296 / 0.450 / 0.564 | 0.054 | -0.13 |
| dive | 0.023 | 0.006 / 0.038 / 0.142 | 0.037 | +0.12 |
| hits_per_minute | 3.702 | 1.890 / 3.578 / 5.282 | 0.725 | +0.85 |

## Style, 4v4 (64 personas against the yardstick)

| 4v4 | yardstick | min / median / max | sd | r with Elo |
|---|---|---|---|---|
| turning | 0.557 | 0.526 / 0.556 / 0.580 | 0.011 | +0.19 |
| two_circle | 0.546 | 0.530 / 0.544 / 0.559 | 0.006 | +0.39 |
| store | 0.615 | 0.303 / 0.591 / 0.846 | 0.126 | -0.30 |
| speed | 0.685 | 0.595 / 0.685 / 0.807 | 0.047 | +0.57 |
| shots | 74.470 | 54.817 / 73.613 / 101.933 | 10.520 | +0.60 |
| hit_range | 0.231 | 0.217 / 0.230 / 0.258 | 0.010 | +0.35 |
| climb | 0.376 | 0.148 / 0.356 / 0.613 | 0.105 | -0.43 |
| dive | 0.176 | 0.023 / 0.190 / 0.400 | 0.094 | +0.27 |
| hits_per_minute | 18.361 | 14.735 / 17.878 / 20.703 | 1.131 | +0.78 |

## Style, 16v16 (64 personas against the yardstick)

| 16v16 | yardstick | min / median / max | sd | r with Elo |
|---|---|---|---|---|
| turning | 0.528 | 0.503 / 0.528 / 0.551 | 0.011 | +0.05 |
| two_circle | 0.521 | 0.509 / 0.520 / 0.531 | 0.005 | +0.60 |
| store | 0.415 | 0.157 / 0.387 / 0.745 | 0.128 | -0.32 |
| speed | 0.680 | 0.560 / 0.676 / 0.811 | 0.052 | +0.71 |
| shots | 104.996 | 83.104 / 106.551 / 144.057 | 15.037 | +0.48 |
| hit_range | 0.236 | 0.224 / 0.237 / 0.266 | 0.010 | +0.61 |
| climb | 0.247 | 0.081 / 0.235 / 0.457 | 0.094 | -0.30 |
| dive | 0.291 | 0.052 / 0.295 / 0.513 | 0.109 | +0.31 |
| hits_per_minute | 24.610 | 21.450 / 24.512 / 27.897 | 1.392 | +0.54 |


## Redirected study: skill couplings around the old yardstick (step 2 data, interact.py)

## 1v1: skill surface (points 106)

Additive (linear + square): R^2 0.90, leave-one-out 0.68; with 6 interactions: R^2 0.97, leave-one-out 0.90; model Elo sd over the box 78.

| interaction (Elo per box^2) | coefficient |
|---|---|
| angles_weight x aggression | -194 +- 27 |
| speed_weight x stickiness | +171 +- 26 |
| power_temperature x seek_weight | -184 +- 31 |
| field_weight x stickiness | +125 +- 26 |
| horizon x temperature | +111 +- 28 |
| stickiness x seek_weight | -92 +- 37 |

| row | S1 | ST | ST - S1 |
|---|---|---|---|
| angles_weight | 0.315 | 0.347 | +0.033 |
| aggression | 0.155 | 0.194 | +0.039 |
| temperature | 0.190 | 0.169 | -0.021 |
| stickiness | 0.029 | 0.082 | +0.053 |
| reply_temperature | 0.073 | 0.065 | -0.008 |
| seek_weight | 0.001 | 0.049 | +0.048 |
| speed_weight | 0.014 | 0.045 | +0.032 |
| power_temperature | 0.007 | 0.040 | +0.032 |
| energy_weight | 0.033 | 0.033 | +0.001 |
| collision_weight | 0.021 | 0.029 | +0.008 |
| disengaged_energy_weight | 0.033 | 0.023 | -0.010 |
| horizon | 0.017 | 0.022 | +0.005 |
| field_weight | 0.009 | 0.022 | +0.013 |
| preferred_speed | 0.019 | 0.017 | -0.002 |
| store_weight | 0.009 | 0.008 | -0.002 |
| power_stickiness | 0.007 | 0.005 | -0.002 |
| threat_weight | 0.005 | 0.002 | -0.003 |
| fire_tolerance | 0.001 | 0.001 | +0.000 |
| caution | -0.000 | 0.001 | +0.001 |
| boundary_weight | 0.001 | 0.001 | -0.000 |
| power_horizon | 0.001 | 0.000 | -0.000 |

| row a | row b | equal Elo: du_b per du_a (linear, at the centre) |
|---|---|---|
| angles_weight | temperature | +1.41 |
| aggression | temperature | +0.99 |
| angles_weight | reply_temperature | +2.15 |
| angles_weight | energy_weight | +3.02 |
| aggression | reply_temperature | +1.52 |
| angles_weight | collision_weight | +3.26 |
| angles_weight | disengaged_energy_weight | +3.68 |
| energy_weight | aggression | +0.47 |

## 4v4: skill surface (points 106)

Additive (linear + square): R^2 0.94, leave-one-out 0.81; with 6 interactions: R^2 0.97, leave-one-out 0.91; model Elo sd over the box 61.

| interaction (Elo per box^2) | coefficient |
|---|---|
| angles_weight x aggression | -119 +- 21 |
| collision_weight x preferred_speed | -82 +- 24 |
| speed_weight x reply_temperature | +89 +- 20 |
| angles_weight x temperature | +67 +- 20 |
| caution x temperature | +69 +- 25 |
| field_weight x power_stickiness | -55 +- 23 |

| row | S1 | ST | ST - S1 |
|---|---|---|---|
| speed_weight | 0.218 | 0.272 | +0.054 |
| fire_tolerance | 0.137 | 0.136 | -0.001 |
| energy_weight | 0.113 | 0.115 | +0.001 |
| aggression | 0.094 | 0.095 | +0.001 |
| horizon | 0.067 | 0.078 | +0.012 |
| angles_weight | 0.044 | 0.067 | +0.022 |
| collision_weight | 0.026 | 0.059 | +0.033 |
| temperature | 0.065 | 0.059 | -0.006 |
| store_weight | 0.039 | 0.033 | -0.005 |
| preferred_speed | 0.016 | 0.033 | +0.017 |
| power_stickiness | 0.035 | 0.028 | -0.007 |
| threat_weight | 0.033 | 0.024 | -0.010 |
| field_weight | 0.026 | 0.020 | -0.006 |
| caution | 0.010 | 0.017 | +0.007 |
| reply_temperature | 0.008 | 0.016 | +0.007 |
| power_temperature | 0.006 | 0.007 | +0.002 |
| disengaged_energy_weight | -0.001 | 0.006 | +0.007 |
| power_horizon | 0.011 | 0.006 | -0.005 |
| boundary_weight | 0.002 | 0.003 | +0.001 |
| stickiness | 0.002 | 0.001 | -0.001 |
| seek_weight | -0.001 | 0.001 | +0.001 |

| row a | row b | equal Elo: du_b per du_a (linear, at the centre) |
|---|---|---|
| speed_weight | horizon | +1.81 |
| aggression | speed_weight | +0.48 |
| speed_weight | collision_weight | +2.36 |
| horizon | fire_tolerance | +0.76 |
| energy_weight | horizon | +1.20 |
| speed_weight | store_weight | +2.77 |
| speed_weight | temperature | +2.81 |
| aggression | fire_tolerance | +0.67 |

## 16v16: skill surface (points 106)

Additive (linear + square): R^2 0.97, leave-one-out 0.89; with 6 interactions: R^2 0.99, leave-one-out 0.95; model Elo sd over the box 160.

| interaction (Elo per box^2) | coefficient |
|---|---|
| preferred_speed x stickiness | -174 +- 38 |
| collision_weight x boundary_weight | -130 +- 37 |
| aggression x boundary_weight | +107 +- 32 |
| aggression x caution | -104 +- 38 |
| speed_weight x collision_weight | -93 +- 34 |
| collision_weight x power_temperature | +66 +- 30 |

| row | S1 | ST | ST - S1 |
|---|---|---|---|
| horizon | 0.171 | 0.178 | +0.006 |
| angles_weight | 0.174 | 0.177 | +0.004 |
| fire_tolerance | 0.128 | 0.127 | -0.001 |
| energy_weight | 0.123 | 0.124 | +0.001 |
| aggression | 0.121 | 0.122 | +0.001 |
| speed_weight | 0.093 | 0.107 | +0.013 |
| store_weight | 0.040 | 0.039 | -0.002 |
| preferred_speed | 0.023 | 0.031 | +0.008 |
| collision_weight | 0.012 | 0.028 | +0.016 |
| temperature | 0.032 | 0.027 | -0.005 |
| threat_weight | 0.023 | 0.021 | -0.002 |
| reply_temperature | 0.012 | 0.014 | +0.002 |
| stickiness | 0.003 | 0.013 | +0.010 |
| boundary_weight | -0.001 | 0.008 | +0.009 |
| caution | 0.004 | 0.007 | +0.003 |
| power_temperature | 0.003 | 0.003 | +0.000 |
| power_horizon | 0.002 | 0.001 | -0.001 |
| disengaged_energy_weight | 0.001 | 0.001 | -0.000 |
| power_stickiness | 0.000 | 0.000 | -0.000 |
| field_weight | 0.001 | 0.000 | -0.000 |
| seek_weight | -0.000 | 0.000 | +0.000 |

| row a | row b | equal Elo: du_b per du_a (linear, at the centre) |
|---|---|---|
| horizon | fire_tolerance | +1.19 |
| energy_weight | horizon | +0.83 |
| angles_weight | fire_tolerance | +1.17 |
| angles_weight | energy_weight | +1.18 |
| speed_weight | horizon | +0.76 |
| angles_weight | speed_weight | +1.29 |
| aggression | fire_tolerance | +0.95 |
| energy_weight | aggression | +1.04 |

