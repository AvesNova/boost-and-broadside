# Look-ahead personas: behaviour, skill and style against the parameters

Author's request 2026-11-27 about 19:00. The question: should BC (and the league) learn from a
persona spread? This doc gives which parameters change behaviour and by how much, each persona's
skill, and how much mixing personas would blur the labels. Measured 2026-11-27 19:02 to 19:56,
laptop RTX 4070, tree `defaults-1127` (base 215e3aec: hit damage 8, spread 5 mrad,
store_crowd on). `lean/` was not changed. Scripts are in this folder (`divergence.py`,
`skill.py`, `analyse.py`; full tables in `tables.md`, numbers in `analysis.json` and
`skill.json`). Raw data: `~/avesnova/boost-and-broadside/data/lookahead-personas/` (state
set, npz files, the SQLite game store).

**Conditions.** "1v1": fog off, no fields. "4v4": 10 fields, fog on. Otherwise the game's
defaults: a 1,800-tick clock (60 s) and the standard spawn. Every agent is the look-ahead agent
(`lean.scripted`) with the defaults except where a row is changed. **Spread 1** means each of
the 21 sampled rows sits on its `parameters.py` box: for a log row, default x e^(+-width), so
the box edges are u = 0 and u = 1. The yardstick is u = 0.5 on every row, and its Elo is 1000.

## Measurements

### 1. Behavioural divergence

**State set.** 40,030 1v1 ship-states and 80,211 4v4 ship-states, taken as sparse snapshots
(the first at 30-230 ticks, then one every 150-250 ticks) from games with sampled actions. Each
team in each game was either the yardstick (probability 1/3) or spread-1 personas drawn per
ship (2/3), redrawn at each deal. Bullets are not read by the agent: dropping them changes the
joint by 0. A configuration's behaviour is `explain`'s joint 42-way distribution on every
ship-state. The divergence is the Jensen-Shannon divergence (JS, natural log) per ship-state,
averaged over ship-states.

For scale: the yardstick's own mean entropy is **1.059 nats in 1v1 and 0.851 in 4v4**. JS is
bounded by ln 2 = 0.693.

**One row at a time** (the other rows at the default). JS from the yardstick at u = 0, 0.25,
0.75 and 1 (spread 1 box edges at u = 0 and 1; u = 0.25 and 0.75 are the spread-0.5 edges).
The last two columns per setup give each head's JS as a share of the joint JS at the larger
edge.

| row | default [spread-1 box] | 1v1 JS u=0 / .25 / .75 / 1 | 1v1 stick/power/fire % of joint | 4v4 JS u=0 / .25 / .75 / 1 | 4v4 stick/power/fire % of joint |
|---|---|---|---|---|---|
| threat_weight | 1 [0.741, 1.35] | 0.0010 / 0.0003 / 0.0003 / 0.0014 | 100/0/0 | 0.0019 / 0.0005 / 0.0006 / 0.0026 | 100/0/0 |
| angles_weight | 0.3 [0.201, 0.448] | 0.0023 / 0.0007 / 0.0009 / 0.0042 | 100/0/0 | 0.0035 / 0.0010 / 0.0014 / 0.0060 | 100/0/0 |
| energy_weight | 0.25 [0.168, 0.373] | 0.0006 / 0.0002 / 0.0003 / 0.0012 | 91/9/0 | 0.0029 / 0.0009 / 0.0011 / 0.0049 | 95/5/0 |
| aggression | 0.5 [0.303, 0.824] | 0.0011 / 0.0003 / 0.0005 / 0.0024 | 100/0/0 | 0.0018 / 0.0006 / 0.0008 / 0.0039 | 100/0/0 |
| speed_weight | 0.3 [0.201, 0.448] | 0.0044 / 0.0012 / 0.0020 / 0.0127 | 35/66/0 | 0.0033 / 0.0010 / 0.0018 / 0.0094 | 35/67/0 |
| collision_weight | 3 [2.22, 4.05] | 0.0011 / 0.0003 / 0.0003 / 0.0013 | 100/0/0 | 0.0017 / 0.0005 / 0.0006 / 0.0023 | 100/0/0 |
| boundary_weight | 2 [1.48, 2.7] | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 100/0/0 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 100/0/0 |
| field_weight | 0.3 [0.201, 0.448] | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0/0/0 | 0.0005 / 0.0002 / 0.0002 / 0.0009 | 100/0/0 |
| horizon | 1.5 [1.11, 2.02] | 0.0057 / 0.0013 / 0.0011 / 0.0039 | 100/0/0 | 0.0072 / 0.0017 / 0.0014 / 0.0051 | 100/0/0 |
| preferred_speed | 0.9 [0.775, 1.05] | 0.0054 / 0.0010 / 0.0004 / 0.0007 | 29/73/0 | 0.0063 / 0.0014 / 0.0007 / 0.0017 | 25/76/0 |
| reply_temperature | 0.2 [0.121, 0.33] | 0.0024 / 0.0007 / 0.0007 / 0.0029 | 100/0/0 | 0.0040 / 0.0012 / 0.0013 / 0.0048 | 100/0/0 |
| caution | 0 [-2, 2] | 0.0023 / 0.0006 / 0.0005 / 0.0018 | 100/0/0 | 0.0035 / 0.0009 / 0.0008 / 0.0029 | 100/0/0 |
| temperature | 0.05 [0.0335, 0.0746] | 0.0125 / 0.0033 / 0.0033 / 0.0127 | 100/0/0 | 0.0104 / 0.0028 / 0.0029 / 0.0115 | 100/0/0 |
| stickiness | 0.05 [0.0303, 0.0824] | 0.0036 / 0.0011 / 0.0018 / 0.0092 | 100/0/0 | 0.0030 / 0.0009 / 0.0015 / 0.0074 | 100/0/0 |
| store_weight | 0.5 [0.335, 0.746] | 0.0021 / 0.0007 / 0.0016 / 0.0093 | 0/100/0 | 0.0022 / 0.0008 / 0.0015 / 0.0082 | 0/100/0 |
| power_horizon | 3 [2.22, 4.05] | 0.0008 / 0.0002 / 0.0004 / 0.0014 | 0/100/0 | 0.0013 / 0.0004 / 0.0006 / 0.0024 | 0/100/0 |
| power_temperature | 0.05 [0.0335, 0.0746] | 0.0012 / 0.0005 / 0.0009 / 0.0044 | 0/100/0 | 0.0013 / 0.0005 / 0.0009 / 0.0046 | 0/100/0 |
| power_stickiness | 0.3 [0.182, 0.495] | 0.0123 / 0.0026 / 0.0014 / 0.0033 | 0/100/0 | 0.0137 / 0.0031 / 0.0018 / 0.0045 | 0/100/0 |
| disengaged_energy_weight | 0.5 [0.303, 0.824] | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0/0/0 | 0.0002 / 0.0001 / 0.0001 / 0.0004 | 85/15/0 |
| seek_weight | 0.1 [0.0497, 0.201] | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0/0/0 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 100/0/0 |
| fire_tolerance | 1 [0.741, 1.35] | 0.0008 / 0.0002 / 0.0002 / 0.0010 | 0/0/100 | 0.0013 / 0.0004 / 0.0004 / 0.0016 | 0/0/100 |

**Saltelli sample** (SALib, base 256, 21 rows: 5,888 configurations, spread 1). "Redraw JS" is
the mean JS(A, AB_i): the behaviour change from redrawing row i alone, with the other rows at
a random persona's values. It is the direct answer to "how much does this row change
behaviour". The share is against redrawing every row (A against B). ST and S1 are the Sobol
total and first-order indices of JS from the yardstick, with 95% intervals. JS from the
yardstick is not monotone in a row, so S1 is noisy and ST is the index to read.

| row | 1v1 redraw JS (share of all) | 1v1 ST | 1v1 S1 | 4v4 redraw JS (share of all) | 4v4 ST | 4v4 S1 |
|---|---|---|---|---|---|---|
| temperature | 0.0090 +- 0.0006 (22%) | 0.226 +- 0.085 | 0.096 +- 0.070 | 0.0078 +- 0.0006 (18%) | 0.204 +- 0.070 | 0.097 +- 0.069 |
| power_stickiness | 0.0074 +- 0.0006 (18%) | 0.447 +- 0.124 | 0.287 +- 0.113 | 0.0075 +- 0.0006 (17%) | 0.458 +- 0.120 | 0.293 +- 0.117 |
| speed_weight | 0.0072 +- 0.0006 (18%) | 0.289 +- 0.087 | 0.065 +- 0.089 | 0.0049 +- 0.0004 (11%) | 0.149 +- 0.046 | 0.018 +- 0.068 |
| horizon | 0.0031 +- 0.0002 (8%) | 0.066 +- 0.025 | -0.012 +- 0.041 | 0.0039 +- 0.0003 (9%) | 0.090 +- 0.034 | -0.002 +- 0.043 |
| store_weight | 0.0045 +- 0.0004 (11%) | 0.215 +- 0.082 | 0.024 +- 0.083 | 0.0037 +- 0.0003 (8%) | 0.172 +- 0.070 | 0.010 +- 0.084 |
| stickiness | 0.0042 +- 0.0004 (10%) | 0.109 +- 0.040 | 0.054 +- 0.067 | 0.0034 +- 0.0003 (8%) | 0.081 +- 0.026 | 0.040 +- 0.053 |
| angles_weight | 0.0022 +- 0.0002 (5%) | 0.032 +- 0.012 | -0.012 +- 0.035 | 0.0032 +- 0.0002 (7%) | 0.065 +- 0.021 | -0.009 +- 0.050 |
| reply_temperature | 0.0019 +- 0.0001 (5%) | 0.011 +- 0.004 | -0.001 +- 0.022 | 0.0031 +- 0.0002 (7%) | 0.035 +- 0.011 | 0.012 +- 0.038 |
| preferred_speed | 0.0017 +- 0.0002 (4%) | 0.041 +- 0.016 | 0.012 +- 0.031 | 0.0027 +- 0.0002 (6%) | 0.078 +- 0.025 | 0.014 +- 0.038 |
| energy_weight | 0.0007 +- 0.0001 (2%) | 0.010 +- 0.003 | -0.003 +- 0.013 | 0.0026 +- 0.0002 (6%) | 0.039 +- 0.012 | -0.000 +- 0.028 |
| caution | 0.0014 +- 0.0001 (3%) | 0.004 +- 0.001 | 0.004 +- 0.013 | 0.0022 +- 0.0002 (5%) | 0.012 +- 0.004 | 0.013 +- 0.022 |
| power_temperature | 0.0019 +- 0.0002 (5%) | 0.051 +- 0.021 | -0.006 +- 0.055 | 0.0020 +- 0.0002 (5%) | 0.072 +- 0.029 | -0.006 +- 0.053 |
| power_horizon | 0.0021 +- 0.0002 (5%) | 0.163 +- 0.050 | 0.038 +- 0.063 | 0.0019 +- 0.0002 (4%) | 0.119 +- 0.039 | 0.021 +- 0.052 |
| aggression | 0.0010 +- 0.0001 (3%) | 0.019 +- 0.007 | 0.010 +- 0.025 | 0.0017 +- 0.0002 (4%) | 0.040 +- 0.015 | 0.014 +- 0.034 |
| threat_weight | 0.0008 +- 0.0001 (2%) | 0.005 +- 0.002 | 0.002 +- 0.012 | 0.0015 +- 0.0001 (3%) | 0.018 +- 0.006 | 0.005 +- 0.023 |
| collision_weight | 0.0009 +- 0.0001 (2%) | 0.002 +- 0.000 | -0.001 +- 0.007 | 0.0013 +- 0.0001 (3%) | 0.004 +- 0.001 | -0.002 +- 0.010 |
| fire_tolerance | 0.0006 +- 0.0000 (1%) | 0.000 +- 0.000 | -0.001 +- 0.004 | 0.0010 +- 0.0001 (2%) | 0.001 +- 0.000 | -0.001 +- 0.007 |
| field_weight | 0.0000 +- 0.0000 (0%) | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.0005 +- 0.0000 (1%) | 0.001 +- 0.000 | 0.002 +- 0.004 |
| disengaged_energy_weight | 0.0000 +- 0.0000 (0%) | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.0002 +- 0.0000 (1%) | 0.000 +- 0.000 | 0.002 +- 0.003 |
| boundary_weight | 0.0000 +- 0.0000 (0%) | 0.000 +- 0.000 | -0.000 +- 0.000 | 0.0000 +- 0.0000 (0%) | 0.000 +- 0.000 | -0.000 +- 0.000 |
| seek_weight | 0.0000 +- 0.0000 (0%) | 0.000 +- 0.000 | 0.000 +- 0.000 | 0.0000 +- 0.0000 (0%) | 0.000 +- 0.000 | -0.000 +- 0.000 |
| every row redrawn (A against B) | 0.0404 +- 0.0014 |  |  | 0.0436 +- 0.0013 |  |  |

- **Matter for behaviour:** the flight temperature, power stickiness and speed weight (each
  11-22% of the all-rows change), then the store weight, stickiness and horizon (8-11%). The
  angles weight, reply temperature, preferred speed, power temperature and power horizon each
  give 4-9%. The energy weight is 2% in 1v1 and 6% in 4v4.
- **Small:** caution, aggression, threat weight, collision weight and fire tolerance (1-5%).
  Fire tolerance is small only because firing is rare: it is the whole of the fire head.
- **Inert on these states:** boundary weight, seek weight and disengaged energy weight (0-1%;
  ships are rarely at the edge or without a known enemy). The field weight is inert by
  construction in 1v1 (no fields) and 1% in 4v4.
- JS is roughly quadratic in the step: the spread-0.5 edges give about a quarter of the
  spread-1 edges' JS.

### 5. Label consistency (the population on the same states)

64 personas, uniform on the unit cube. The same points are used at spread 0.25, 0.5 and 1,
and they are also the population of the skill section. The **blur** is the mixture's entropy
minus the members' mean entropy, which is the generalized JS divergence. It is the extra
cross-entropy per label that one student pays for fitting the mixed labels instead of each
persona's own, if it cannot tell the personas apart (the persona is not observable).

| spread | setup | yardstick H | members' mean H | mixture H | blur: mixture H - mean H | JS member-yardstick mean | max | JS pairwise mean | JS mixture-yardstick |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 1v1 | 1.059 | 1.055 | 1.061 | 0.0058 (0.5%) | 0.0015 | 0.0049 | 0.0029 | 0.00006 |
| 0.25 | 4v4 | 0.851 | 0.849 | 0.856 | 0.0070 (0.8%) | 0.0018 | 0.0053 | 0.0033 | 0.00007 |
| 0.5 | 1v1 | 1.059 | 1.056 | 1.080 | 0.0238 (2.3%) | 0.0060 | 0.0226 | 0.0116 | 0.00070 |
| 0.5 | 4v4 | 0.851 | 0.851 | 0.879 | 0.0277 (3.3%) | 0.0070 | 0.0230 | 0.0130 | 0.00077 |
| 1.0 | 1v1 | 1.059 | 1.063 | 1.163 | 0.0994 (9.3%) | 0.0254 | 0.0921 | 0.0468 | 0.00656 |
| 1.0 | 4v4 | 0.851 | 0.866 | 0.972 | 0.1064 (12.3%) | 0.0274 | 0.0908 | 0.0504 | 0.00673 |

- At spread 1 the blur is 0.10-0.11 nats per label: 9% of the members' entropy in 1v1, 12% in
  4v4.
- At spread 0.5 it is 0.024-0.028 nats (2-3%). At spread 0.25 it is 0.006-0.007 nats (under
  1%).
- The mixture's mean is close to the yardstick: JS 0.007 at spread 1, 0.0007 at spread 0.5.
  A student fitting the mixture therefore learns, on average, nearly the yardstick's
  distribution, slightly flatter.

### 2. Skill (Elo against the yardstick)

Every agent plays only the yardstick: a team of one persona (all its ships alike) against a
yardstick team, with sides balanced. Games are added adaptively (`Evaluator.adaptive`) until
every pairing's score has a standard error of at most 0.015 (about 10-12 Elo). Elo is the
Bradley-Terry fit from `lean.eval.rating`, with the yardstick fixed at 1000. There are 106
agents: the 42 box edges and the 64 population personas. That took 118,784 games per setup
(1v1 about 1,150 games/s, 4v4 about 165 games/s, after a compile of about 1-1.5 minutes per
job). Draws: 1v1 mean 1.1% (largest 5.8%), 4v4 0.2%.

**Box edges** (one row at u = 0 or 1, the rest at the default):

| row | 1v1 Elo at u=0 | 1v1 Elo at u=1 | 4v4 Elo at u=0 | 4v4 Elo at u=1 |
|---|---|---|---|---|
| threat_weight | 1003 +- 10 | 976 +- 10 | 1007 +- 10 | 969 +- 10 |
| angles_weight | 835 +- 12 | 1162 +- 12 | 1017 +- 10 | 964 +- 10 |
| energy_weight | 1045 +- 10 | 920 +- 10 | 971 +- 10 | 1025 +- 10 |
| aggression | 906 +- 11 | 1119 +- 11 | 1030 +- 10 | 948 +- 11 |
| speed_weight | 1083 +- 10 | 805 +- 13 | 961 +- 10 | 1036 +- 10 |
| collision_weight | 1114 +- 11 | 925 +- 10 | 1030 +- 10 | 975 +- 10 |
| boundary_weight | 1011 +- 10 | 1006 +- 10 | 1004 +- 10 | 1007 +- 10 |
| field_weight | 991 +- 10 | 1001 +- 10 | 1005 +- 10 | 1011 +- 10 |
| horizon | 989 +- 10 | 999 +- 10 | 1030 +- 10 | 969 +- 10 |
| preferred_speed | 1058 +- 10 | 987 +- 10 | 975 +- 10 | 1015 +- 10 |
| reply_temperature | 1034 +- 10 | 952 +- 10 | 1003 +- 10 | 1002 +- 10 |
| caution | 953 +- 10 | 1035 +- 10 | 972 +- 10 | 1006 +- 10 |
| temperature | 1041 +- 10 | 935 +- 10 | 1032 +- 10 | 971 +- 10 |
| stickiness | 998 +- 10 | 1006 +- 10 | 986 +- 10 | 999 +- 10 |
| store_weight | 955 +- 10 | 1068 +- 10 | 1012 +- 10 | 977 +- 10 |
| power_horizon | 1007 +- 10 | 1008 +- 10 | 1005 +- 10 | 985 +- 10 |
| power_temperature | 942 +- 10 | 1019 +- 10 | 984 +- 10 | 991 +- 10 |
| power_stickiness | 992 +- 10 | 931 +- 10 | 987 +- 10 | 976 +- 10 |
| disengaged_energy_weight | 996 +- 10 | 1007 +- 10 | 996 +- 10 | 993 +- 10 |
| seek_weight | 999 +- 10 | 1012 +- 10 | 1002 +- 10 | 1002 +- 10 |
| fire_tolerance | 992 +- 10 | 1005 +- 10 | 947 +- 11 | 1037 +- 10 |

**The population** (64 personas, spread 1):

| setup | personas | games (all agents) | mean Elo | sd observed | sd without error | mean error | min / 10% / median / 90% / max | above 1000 | best |
|---|---|---|---|---|---|---|---|---|---|
| 1v1 | 64 | 118784 | 987 | 154 | 154 | 12 | 652 / 813 / 970 / 1178 / 1319 | 29 | persona14 |
| 4v4 | 64 | 118784 | 978 | 57 | 56 | 11 | 840 / 905 / 983 / 1046 / 1107 | 27 | persona21 |

"sd without error" is sqrt(observed variance - mean squared error). The skill spread is real
and not measurement noise: **sd 154 Elo in 1v1 (652 to 1319), 56 in 4v4 (840 to 1107)**. 29 of
64 personas beat the yardstick in 1v1 and 27 of 64 in 4v4. A persona's 1v1 Elo and its 4v4
Elo are slightly anti-correlated (r = -0.21). The best in both is persona27 (1248 / 1046);
persona09 is 1284 / 1010.

**What drives skill.** A weighted least-squares fit of Elo on the unit coordinates (the 106
agents, weights 1 / error^2) explains R^2 0.88 (1v1) and 0.87 (4v4): skill is close to
additive in the rows. Each coefficient is the Elo change from u = 0 to u = 1 (the whole
spread-1 box) with its standard error:

| row (Elo change from u=0 to u=1, linear fit) | 1v1 | 4v4 |
|---|---|---|
| threat_weight | -26 +- 19 | -26 +- 9 |
| angles_weight | +310 +- 19 | -41 +- 8 |
| energy_weight | -51 +- 20 | +60 +- 9 |
| aggression | +176 +- 20 | -60 +- 9 |
| speed_weight | -174 +- 19 | +86 +- 9 |
| collision_weight | -155 +- 18 | -44 +- 8 |
| boundary_weight | -6 +- 19 | -4 +- 9 |
| field_weight | +17 +- 18 | +15 +- 9 |
| horizon | -11 +- 19 | -82 +- 9 |
| preferred_speed | -19 +- 21 | +13 +- 10 |
| reply_temperature | -69 +- 19 | +3 +- 9 |
| caution | +57 +- 19 | +20 +- 9 |
| temperature | -107 +- 18 | -61 +- 9 |
| stickiness | -10 +- 19 | -4 +- 9 |
| store_weight | +69 +- 18 | -36 +- 9 |
| power_horizon | +46 +- 21 | +7 +- 9 |
| power_temperature | +23 +- 18 | -11 +- 9 |
| power_stickiness | -94 +- 20 | -9 +- 10 |
| disengaged_energy_weight | +29 +- 19 | -3 +- 9 |
| seek_weight | +19 +- 20 | +9 +- 9 |
| fire_tolerance | +23 +- 19 | +75 +- 9 |
| R^2 / points | 0.88 / 106 | 0.87 / 106 |

- **1v1:** angles weight (+310 across the box), aggression (+176), speed weight (-174),
  collision weight (-155), temperature (-107), power stickiness (-94), store weight (+69),
  reply temperature (-69), caution (+57). On every one of these, the yardstick does not sit at
  the best end of its box.
- **4v4:** horizon (-82), speed weight (+86), fire tolerance (+75), temperature (-61),
  aggression (-60), energy weight (+60), threat weight (-26), angles weight (-41), collision
  weight (-44), store weight (-36).
- The signs disagree between the setups for the angles weight, aggression, the speed weight,
  the energy weight and the store weight. The direction that wins 1v1 loses 4v4, which is why
  the two Elos are anti-correlated.
- Behaviour and skill are weakly linked. JS from the yardstick against |Elo - 1000| over the
  64 personas: r = 0.11 (1v1), 0.28 (4v4). Small changes in behaviour (JS 0.002-0.006 at one
  edge) move Elo by 100-300 in 1v1. The inert rows (boundary, seek, disengaged energy, field)
  are within +-30 Elo, as expected.

### 3. Style (games against the yardstick, `lean.eval.style`)

The style is per persona, over its games against the yardstick. These are the counters the
runner has: no tail-chase share of hits exists. Over the 64 personas: the yardstick's value,
then min / median / max, the standard deviation, and the correlation with the persona's Elo.

| 1v1 | yardstick | min / median / max | sd | r with Elo |
|---|---|---|---|---|
| turning | 0.668 | 0.580 / 0.667 / 0.699 | 0.023 | +0.41 |
| two_circle | 0.645 | 0.549 / 0.646 / 0.742 | 0.043 | +0.29 |
| store | 0.786 | 0.392 / 0.739 / 0.871 | 0.121 | +0.17 |
| speed | 0.634 | 0.542 / 0.640 / 0.786 | 0.056 | -0.74 |
| shots | 44.731 | 11.985 / 35.167 / 72.716 | 16.071 | +0.88 |
| hit_range | 0.239 | 0.163 / 0.215 / 0.255 | 0.020 | -0.17 |
| climb | 0.477 | 0.182 / 0.438 / 0.763 | 0.117 | +0.19 |
| dive | 0.028 | 0.001 / 0.036 / 0.252 | 0.072 | -0.10 |
| hits per ship-minute | 12.35 | 3.65 / 10.89 / 22.56 | 4.72 | +0.94 |

| 4v4 | yardstick | min / median / max | sd | r with Elo |
|---|---|---|---|---|
| turning | 0.552 | 0.522 / 0.552 / 0.578 | 0.012 | +0.17 |
| two_circle | 0.539 | 0.524 / 0.539 / 0.552 | 0.006 | +0.46 |
| store | 0.621 | 0.303 / 0.598 / 0.843 | 0.123 | -0.31 |
| speed | 0.688 | 0.598 / 0.688 / 0.813 | 0.046 | +0.54 |
| shots | 74.623 | 54.914 / 73.429 / 103.285 | 10.910 | +0.60 |
| hit_range | 0.231 | 0.214 / 0.230 / 0.260 | 0.010 | +0.37 |
| climb | 0.377 | 0.142 / 0.356 / 0.619 | 0.104 | -0.41 |
| dive | 0.174 | 0.022 / 0.185 / 0.393 | 0.092 | +0.28 |
| hits per ship-minute | 18.46 | 15.04 / 17.92 / 21.04 | 1.25 | +0.77 |

- **Personas fly differently mainly in energy management:** store 0.39-0.87 (1v1) and
  0.30-0.84 (4v4), climb share 0.18-0.76 and 0.14-0.62, dive share 0.00-0.25 and 0.02-0.39, and
  speed 0.54-0.79 and 0.60-0.81 corner speeds.
- **The geometry hardly moves:** the turning-fight share is 0.58-0.70 (1v1) and 0.52-0.58 (4v4),
  the two-circle share 0.55-0.74 and 0.52-0.55, and the range at hits 0.16-0.26 gun ranges.
  The personas fight the same kind of turning fight at the same range, with different power
  and speed.
- Shots and hits per minute follow the result (r +0.88 / +0.94 in 1v1): a losing persona gets
  fewer shooting chances. They are not an independent style axis here. 1v1 speed is
  anti-correlated with Elo (r -0.74): the slower, store-holding personas win the duel. In 4v4
  it is the reverse (r +0.54).

### 4. Degenerate personas

No agent stalls, spirals, sits in draws or stops firing. Every agent's draw share is at most
5.8% (1v1) and 0.7% (4v4). The slowest mean speed is 0.54 corner speeds, against the
yardstick's 0.63. Every persona fires at least 12 real shots per ship-minute. No agent's store
falls under 0.30 of full. Flags from `analyse.py` (shots under half the yardstick's, Elo under
800, slow under 0.8 x the yardstick's speed, the store under 0.15, hit range over 2 x the
yardstick's, draws over 20%):

| setup | agent | flag |
|---|---|---|
| 1v1 | speed_weight@high | fires under half the yardstick's shots |
| 1v1 | persona00 | fires under half the yardstick's shots; Elo 652 |
| 1v1 | persona02 | fires under half the yardstick's shots |
| 1v1 | persona06 | fires under half the yardstick's shots |
| 1v1 | persona20 | fires under half the yardstick's shots; Elo 702 |
| 1v1 | persona28 | fires under half the yardstick's shots |
| 1v1 | persona53 | fires under half the yardstick's shots; Elo 665 |
| 1v1 | persona58 | Elo 793 |
| 1v1 | persona59 | fires under half the yardstick's shots |
| 1v1 | persona60 | fires under half the yardstick's shots; Elo 761 |
| 1v1 | persona63 | fires under half the yardstick's shots |

All flags are in 1v1 and are weak duellists: low shots go with losing. The weakest are
persona00 (652), persona53 (665) and persona20 (702), and the high speed-weight edge is 805.
Ramming cannot be told apart: hits count bullets and collisions together, and the range at
hits does not move (the low collision-weight edge, +114 Elo in 1v1, has the same range at hits,
0.218 gun ranges).

## Interpretation and options for BC persona spread (not measured; for the author's decision)

What the measurements say for the choice:

- Spread changes **skill** far more than it changes **labels**. At spread 1 a typical persona
  is only JS 0.025-0.027 from the yardstick, but the population's Elo has sd 154 (1v1) / 56
  (4v4), and many personas beat the yardstick. The spread-1 box is not "the yardstick, slightly
  perturbed" in strength. The yardstick is not at a skill optimum of its own box in either
  setup, and the optima differ between 1v1 and 4v4.
- The **label blur** a student would pay without seeing the persona is about quadratic in the
  spread: 0.10-0.11 nats at 1, 0.024-0.028 at 0.5, 0.006-0.007 at 0.25 (against a yardstick
  entropy of 0.85-1.06). The mixture's mean stays near the yardstick (JS 0.007 at spread 1),
  so the main effect on the learned policy is a somewhat flatter yardstick, not a different
  one.
- **State diversity** comes mostly from energy-management differences (store, climb or dive,
  speed), not from different fight geometry. The state-distribution shift itself was not
  measured: only the per-agent style means.

Options:

1. **None (spread 0, now).** The labels are fully consistent, and BC's loss floor is the
   yardstick's own entropy. States come only from the yardstick's play (plus the student's
   own, if the rollout is on-policy). Cost: no coverage of the store, speed and power regimes
   the personas reach. The student imitates one agent whose box neighbours are up to about
   300 Elo stronger in 1v1.
2. **Small (0.25-0.5).** The blur is 0.006-0.028 nats (under 1% to 3% of the label entropy).
   The style differences scale down with the spread (not measured below 1; the
   one-row-at-a-time JS suggests about a quarter of spread 1's effect at 0.5). Most of the
   label consistency is kept, with a little state diversity. This is the trade-off that costs
   the least against the BC target, if diversity is wanted.
3. **Full (1).** About 0.1 nats of irreducible extra loss per label (9-12% of the entropy)
   unless the student is told the persona, plus the widest regime coverage. The teacher
   population then has a skill sd of 154 / 56 Elo, so "BC to the yardstick" no longer
   describes the target. A league of spread-1 personas holds opponents from about 650 to 1320
   Elo in 1v1.

Related points the data raise:

- **Narrowing the box.** The inert rows (boundary, seek, disengaged energy, and field in 1v1)
  add nothing as persona rows. Dropping them from `SAMPLED` (or keeping them) changes neither
  behaviour nor skill on these states.
- **The yardstick itself.** The fit points to stronger 1v1 settings (more angles weight and
  aggression, less speed and collision weight, a colder temperature) and different 4v4 ones
  (a shorter horizon, more speed weight and fire tolerance, a colder temperature). The flight
  temperature is the one row that gains Elo in both setups when lowered (-107 / -61 across
  the box). This is a separate question: re-centring the yardstick would change the 1000-Elo
  anchor.
- **Conditioning.** If spread is used and the label blur matters, the student could be given
  the persona's row values (an input). That would remove the blur, but the policy would then
  need persona inputs at play time. Not measured.
