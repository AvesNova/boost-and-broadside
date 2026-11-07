# What each one-tick baseline leaves, and how the axes of the residual correlate

Brief: `baselines-bench-brief.md`. Offline, CPU only, from the recorded segments of the
state-delta analysis (`state-deltas.md`, section 0: scripted and calm-fern self-play,
Frontline, Deathmatch and wide Deathmatch, 3 x 1024 ticks each). No new games, no model
change. Scripts:

| Script | What it does |
|---|---|
| `bench/baselines_core.py` | P, R and E for one tick; the CV, R and E forecasts over k ticks |
| `bench/baselines_measure.py` | one segment at a time (the previous segment's last tick prepended: the segments are one continuous play) into signed log histograms, win counts, regression moments, streaming 2-D moments and a 20k subsample per segment and horizon |
| `bench/baselines_report.py` | merges them; the tables here (`--part main`) and in `baselines-bench-tables.md` (`--part full`) |

Tests: `tests/sections/test_baselines.py`. E is exact on constant acceleration along a
line, on a steady turn and on a constant nose-rate change; the event rule; R agrees with
`DeltaCode.baseline` on random states; the forecasts are exact on their own models.

## 0. Definitions and choices

- **P**: nothing changes. **R**: `delta.py`'s `DeltaCode.baseline`, which is persistence for
  the nose rate, angle of attack, energy and ln n. (The earlier analysis's ln n dead
  reckoning along the gradient is not in `delta.py`, so it is not R here.) **E**: the
  brief's table, with these readings:
  - Position, literal: x + v dt + a dt^2 / 2, with a the transported acceleration that the
    velocity uses. On a circle this is second order only (about 1e-4 m per tick at 70 m/s
    and 1.7 rad/s), so the test asks for exactness of **E-arc**: x + Arc(phi) (v + du/2 u) dt,
    with phi and du last tick's path turn and speed change. Both are reported. They differ
    by less than the sampling noise (E-arc wins 48% of pairs, loses 52%).
  - Energy, shield and their clocks are clamped to [0, max] under E. "Event on t - 1"
    means: for energy and the cooldown, the ship fired on t - 1 (the cooldown jumped). For
    the shield and the wait, damage on t - 1 or a wait reset.
  - E falls back to R where t - 1 is missing (the first tick of a life or a game): 0.1-0.2%
    of pairs (table 1.1). The velocity frame never fell back: no living ship was below 1 m/s.
- **Entropy.** H is the entropy of the mean HL-Gauss code (`codes.symlog_encode`, sigma
  0.75 bins). Codes are 51 real bins for position and velocity components and 31 for the
  rest, with one code per series from the pooled distribution. **Own code**: the knee at
  that baseline's median and the range at its max, as the brief asks. Where the median is
  0, the knee is the median above 1e-6, because float rounding of the recorded state
  leaves 1e-8 residues. **Today's code**: `delta.COMPONENTS`, the same knee, bound and bins
  for every baseline, with two-bin events for the wait and the cooldown. Every H is
  reported less H_self, the mean entropy of a single sample's code, which is the
  smoothing's own floor (1.20 nats at 31 and 51 bins). Two cautions:
  - The own-code H measures how spread the residual is in log space. A baseline that
    shrinks the median but not the max widens that range, so its own-code H can rise even
    when its residual is 10x smaller (position, energy). The today's-code column compares
    the baselines like for like.
  - Both are marginal entropies: the cost for a head that knows nothing about the state.
    They do not measure the conditional entropy that a head can reach.
- **Pooling.** Every player and preset has equal weight. The quantiles come from 100 bins
  per decade, interpolated, with the exact zeros and the max kept exactly.
- **Float32.** The recorded positions are float32 at up to a few km, a resolution of about
  2-5e-4 m. E's along-track position residual has a median of 2.2e-4 m, at that resolution.
- **Held / changed**: the power and stick at t equal those at t - 1, or not. **Quiet**: no
  field band at t - 1, t or t + 1, and no damage on t - 1 or t. **Field**: in a band at t or
  t + 1. **Hit**: any damage on tick t.

## 1. One tick

### 1.1 Counts

| player | preset | one-tick pairs | E falls back | held | changed | power changed | stick changed | in a field band | hit |
|---|---|---|---|---|---|---|---|---|---|
| scripted | frontline | 3,457,823 | 0.2% | 53.4% | 46.4% | 16.5% | 35.7% | 18.5% | 1.4% |
| scripted | deathmatch | 3,602,405 | 0.2% | 46.7% | 53.1% | 20.4% | 40.2% | 15.4% | 1.6% |
| scripted | deathmatch_wide | 3,604,472 | 0.2% | 52.8% | 47.0% | 22.7% | 31.1% | 17.4% | 1.7% |
| calm-fern | frontline | 1,847,861 | 0.1% | 65.8% | 34.1% | 9.5% | 28.9% | 18.9% | 1.3% |
| calm-fern | deathmatch | 1,903,082 | 0.1% | 65.9% | 33.9% | 9.1% | 29.0% | 18.1% | 2.1% |
| calm-fern | deathmatch_wide | 1,856,775 | 0.2% | 70.5% | 29.3% | 12.6% | 20.3% | 12.2% | 3.8% |

### 1.2 Headline (pooled)

Median and p99.9 of |residual|. For position and velocity they are of the norm, which is
the same in every frame. H - H_self is under today's code: along / across the nose for
position and velocity, and for the wait and the cooldown the two-bin event. Every component
in every frame, with p90, p99, max, the zero shares and both codes, is in
`baselines-bench-tables.md`, section "One tick, pooled".

| quantity | P: median / p99.9 / H | R: median / p99.9 / H | E: median / p99.9 / H | least median | least p99.9 |
|---|---|---|---|---|---|
| position (m) | 2.60 / 6.83 / 0.50, 2.21 | 0.0066 / 0.296 / 1.36, 1.65 | 0.0019 / 0.135 / 0.88, 1.23 | E (3.4x below R) | E (2.2x below R) |
| velocity (m/s) | 0.598 / 14.1 / 1.29, 1.97 | 0.333 / 14.4 / 1.45, 1.75 | 0.116 / 8.15 / 0.96, 1.43 | E (2.9x) | E (1.7x) |
| heading (rad) | 0.0101 / 0.114 / 1.76 | 9.7e-4 / 0.110 / 1.41 | 6.3e-4 / 0.154 / 1.41 | E | R |
| nose rate (rad/s) | 0.052 / 4.80 / 1.49 | = P | 0.045 / 7.66 / 1.52 | E (barely) | P |
| angle of attack (rad) | 0.00285 / 0.077 / 1.42 | = P | 0.00142 / 0.089 / 1.30 | E (2x) | P |
| energy (of max) | 5.4e-4 / 0.0298 / 1.15 | = P | 2.1e-6 / 0.0347 / 0.90 | E (260x) | P |
| shield (of max) | 0 (93.5%) / 0.108 / 0.27 | 0 (93.5%; 98.0% below 1e-6) / 0.108 / 0.11 | 0 (95.0%) / 0.108 / 0.11 | - | equal |
| recharge wait (s) | 0 (83.6%) / 5.21 / 0.45 | 0 (98.0% below 1e-6) / 5.25 / 0.097 | 0 (96.9%) / 5.25 / 0.097 | - | equal |
| gun cooldown (ticks) | 0 (96.0%) / 2 / 0.17 | 0 (98.3%) / 2 / 0.086 | 0 (98.3%) / 2 / 0.086 | - | equal |
| ln n | 0 (96.5%) / 0.072 / 0.18 | = P | 0 (96.4%) / 0.020 / 0.12 | - | E (3.6x) |

H against P, under today's code: E is 1.0 nats lower across the nose for position but 0.4
higher along it (see below), and lower for velocity (0.3, 0.5), heading (0.35), energy (0.25), shield (0.16), the wait (0.35)
and ln n (0.06). It is 0.03 nats higher for the nose rate. Position along the nose under P
scores lowest of all (0.50), because the whole 2.6 m move falls into a few outer bins of
the 2 mm-knee code. That shows the marginal entropy's limit, not an easy target.

Per player and preset (median / p99.9; `baselines-bench-tables.md`): E's position median
runs from 0.81 mm (calm-fern wide) to 4.0 mm (scripted deathmatch), against R's 5.2-10.2 mm.
Its velocity median runs from 0.046 to 0.235 m/s, against R's 0.29-0.48. The order P > R > E
in the median holds in every group, for both of these quantities.

### 1.3 Held and changed control

Median / p99.9 / H - H_self (own code), pooled. Shares of pairs: held 59.2%, changed 40.6%
(power 15.2%, stick 30.9%), held and quiet 47.3%, changed and quiet 33.1%.

| series | held | changed | power changed | stick changed | held, quiet | changed, quiet |
|---|---|---|---|---|---|---|
| position E, along the nose (m) | 8.0e-5 / 0.019 / 1.18 | 0.0015 / 0.063 / 2.25 | 0.012 / 0.067 / 1.79 | 6.2e-4 / 0.062 / 2.15 | 8.4e-5 / 0.0056 / 1.14 | 0.0014 / 0.030 / 2.19 |
| position R, along the nose (m) | 0.0020 / 0.105 / 1.57 | 0.0028 / 0.151 / 1.77 | 0.0049 / 0.119 / 1.93 | 0.0024 / 0.170 / 1.69 | 0.0021 / 0.026 / 1.49 | 0.0028 / 0.029 / 1.67 |
| position E, across the nose (m) | 2.8e-4 / 0.090 / 1.46 | 0.0041 / 0.164 / 2.21 | 0.0016 / 0.138 / 2.03 | 0.0057 / 0.173 / 2.20 | 3.1e-4 / 0.041 / 1.48 | 0.0045 / 0.045 / 2.15 |
| position R, across the nose (m) | 9.1e-4 / 0.100 / 1.51 | 0.0122 / 0.371 / 2.31 | 0.0044 / 0.285 / 2.07 | 0.0174 / 0.402 / 2.29 | 9.6e-4 / 0.074 / 1.51 | 0.0131 / 0.133 / 2.26 |
| velocity P, along the track (m/s) | 0.120 / 6.39 / 1.63 | 0.159 / 9.12 / 1.79 | 0.289 / 7.20 / 1.96 | 0.135 / 10.2 / 1.70 | 0.127 / 1.71 / 1.55 | 0.155 / 1.66 / 1.69 |
| velocity E, along the track (m/s) | 0.0025 / 2.18 / 1.18 | 0.092 / 4.06 / 2.26 | 0.715 / 4.21 / 1.79 | 0.042 / 4.24 / 2.17 | 0.0025 / 0.571 / 1.13 | 0.083 / 1.80 / 2.20 |
| velocity R, across the track (m/s) | 0.043 / 5.53 / 1.57 | 0.497 / 16.8 / 2.29 | 0.203 / 12.9 / 2.06 | 0.790 / 18.4 / 2.32 | 0.044 / 4.22 / 1.58 | 0.542 / 5.98 / 2.25 |
| velocity E, across the track (m/s) | 0.019 / 6.12 / 1.55 | 0.202 / 8.95 / 2.17 | 0.073 / 7.89 / 1.97 | 0.281 / 9.37 / 2.18 | 0.021 / 2.64 / 1.57 | 0.224 / 2.52 / 2.11 |
| heading E (rad) | 3.2e-5 / 0.037 / 0.94 | 0.0094 / 0.209 / 1.90 | 0.0013 / 0.161 / 1.67 | 0.0126 / 0.225 / 1.70 | 3.0e-5 / 0.030 / 0.97 | 0.0099 / 0.107 / 1.85 |
| heading R (rad) | 1.3e-4 / 0.025 / 1.00 | 0.0079 / 0.145 / 1.90 | 0.0017 / 0.115 / 1.67 | 0.0106 / 0.160 / 1.77 | 1.3e-4 / 0.025 / 1.01 | 0.0084 / 0.074 / 1.85 |
| nose rate E (rad/s) | 0.0024 / 2.52 / 0.98 | 0.462 / 10.3 / 1.91 | 0.091 / 7.95 / 1.67 | 0.650 / 10.9 / 1.76 | 0.0022 / 2.07 / 1.00 | 0.491 / 5.83 / 1.88 |
| nose rate P (rad/s) | 0.0075 / 1.24 / 1.02 | 0.389 / 6.11 / 1.92 | 0.090 / 4.98 / 1.69 | 0.522 / 6.69 / 1.78 | 0.0071 / 1.23 / 1.03 | 0.411 / 3.52 / 1.88 |
| energy E (of max) | 8.5e-7 / 0.030 / 0.59 | 3.9e-4 / 0.037 / 2.05 | 0.0043 / 0.038 / 1.35 | 1.1e-4 / 0.036 / 2.01 | 9.3e-7 / 0.030 / 0.53 | 2.9e-4 / 0.036 / 2.04 |
| energy P (of max) | 4.9e-4 / 0.030 / 0.98 | 6.0e-4 / 0.030 / 1.39 | 0.0011 / 0.034 / 1.79 | 5.8e-4 / 0.030 / 1.20 | 5.4e-4 / 0.029 / 0.94 | 6.1e-4 / 0.030 / 1.34 |

When the control is held, E's residual is near zero: 0.08 mm along, 2.5 mm/s along the
track, 32 microrad of heading, 1e-6 of energy. These medians are 20-500 times below R's or
P's. In held and quiet ticks the p99.9 also collapses (velocity along 0.57 m/s against
1.4-1.7 for P and R). When the control changes, E carries the change, and its median there
is 20-400 times its held median. So the brief's expectation holds, with one qualification:

- **A power change.** E carries last tick's thrust into the new mode. Its along-track
  velocity residual (median 0.715 m/s) is then 2.5 times P's (0.289) and R's (0.284), and
  its energy residual (0.0043) 4 times P's. A stick change hurts E less: across the track,
  E's 0.281 m/s is below P's 0.642 and R's 0.790. R rotates v by the nose's swing, which
  over-rotates when the stick moves (the round-2 notes).

Win shares (`baselines-bench-tables.md`, section "Wins"), E against R on the position norm:
86% of all pairs, 97% held and quiet, 71% changed. On the velocity norm: 87 / 98 / 72%.
Heading: E beats R on 66% of pairs, but on only 44% when the control changes.

### 1.4 The nose rate under E

The nose rate is a damped oscillator, and E overshoots it. The least-squares coefficient of
the change on t on last tick's change, which E takes as 1 and P as 0:

| subset | beta | share of the change's mean square removed by the best beta | by beta = 1 (E) |
|---|---|---|---|
| all | -0.23 | 0.05 | -1.45 |
| held, quiet | +0.41 | 0.97 | -1.06 |
| changed, quiet | -0.54 | 0.21 | -1.50 |
| in a field band | -0.26 | 0.07 | -1.52 |

For comparison, the angle of attack has beta 0.54 overall and 0.94 when held and quiet
(E removes 68% there). Energy has 0.998 held and quiet (E removes 100%) and 0.20 overall.
ln n has 0.96 (E removes 92%).

- **Held control.** The nose rate's change decays geometrically by a factor of about 0.41
  per tick: 97% of its mean square is explained by 0.41 x last tick's change. E's
  coefficient of 1 overshoots by about 2.4x. E still has the lower median (0.0022 against
  0.0071 rad/s) because most held ticks are near steady state. But its mean square is 2.06
  times the change's own, and its p99.9 is 2.07 against 1.23 rad/s.
- **Changed control.** The change reverses sign (beta -0.54). E doubles the error there
  (p99.9 10.3 against 6.1 rad/s).
- **Pooled.** E beats P on 60% of pairs, but its p99.9 (7.66 rad/s) and max (18.8) are 1.6-1.8
  times P's. **E does not beat P on the nose rate**: it wins the median narrowly and loses
  the tail and the mean square. Under today's code its H is 0.03 nats above P's.

## 2. Axis correlation and the error ellipse

### 2.1 Pooled, the axes look uncorrelated

Pooled over all pairs, |Pearson| and |Spearman| are below 0.08 in every frame, for every
forecast and horizon (`baselines-bench-tables.md`, section "Axis correlation"). This
follows from symmetry and does not show independence:

- A left turn and a right turn are mirror images. The mirror flips the across component,
  so any along-across correlation cancels in the pool.
- The world frame pools over every heading, so its ellipse comes out isotropic (axis
  ratio 1.00-1.02) with every horizon and forecast.

The per-sample (conditional) correlation is what a per-axis categorical head cannot
represent.

### 2.2 With the mirror folded out

The across component below is multiplied by the sign of the nose rate at t. "Curve"
uses the sign of last tick's normal acceleration instead. Statistics come from the
subsample, 360k pairs per horizon. "World at a fixed heading" is the turn-folded nose
ellipse seen in a world frame at a random heading: the mean |Pearson| a world-frame head
would face for a ship of known heading.

Each cell: Pearson / Spearman / axis ratio / major-axis angle (deg from along).

| quantity | k (ticks, s) | forecast | nose, turn-folded | track, turn-folded | track, curve-folded | world at a fixed heading |
|---|---|---|---|---|---|---|
| position | 1 (0.03) | R | +0.02 / -0.04 / 2.33 / +89 | -0.02 / -0.03 / 2.38 / -89 | +0.06 / +0.14 / 2.57 / +88 | 0.48 |
| position | 1 | E | +0.02 / -0.14 / 2.04 / +89 | -0.02 / -0.15 / 2.03 / -89 | -0.08 / -0.22 / 1.99 / -87 | 0.42 |
| position | 3 (0.1) | CV | -0.09 / -0.00 / 2.44 / -87 | -0.39 / -0.47 / 2.48 / -78 | -0.38 / -0.44 / 2.49 / -78 | 0.51 |
| position | 3 | R | -0.06 / -0.09 / 2.07 / -88 | -0.10 / -0.07 / 2.09 / -87 | +0.02 / +0.12 / 2.25 / +89 | 0.43 |
| position | 3 | E | -0.08 / -0.30 / 1.91 / -87 | -0.09 / -0.26 / 1.85 / -86 | -0.15 / -0.32 / 1.84 / -83 | 0.39 |
| position | 15 (0.5) | CV | -0.45 / -0.38 / 2.69 / -77 | -0.62 / -0.52 / 2.69 / -69 | -0.61 / -0.51 / 2.69 / -69 | 0.55 |
| position | 15 | R | -0.51 / -0.32 / 2.64 / -74 | -0.53 / -0.35 / 2.54 / -72 | -0.27 / -0.08 / 2.30 / -81 | 0.54 |
| position | 15 | E | -0.19 / -0.28 / 1.65 / -79 | -0.21 / -0.28 / 1.56 / -76 | -0.25 / -0.35 / 1.53 / -72 | 0.31 |
| position | 30 (1) | CV | -0.55 / -0.46 / 2.32 / -68 | -0.63 / -0.51 / 2.28 / -60 | -0.63 / -0.51 / 2.29 / -60 | 0.48 |
| position | 30 | R | -0.55 / -0.51 / 2.06 / -62 | -0.53 / -0.52 / 1.95 / -59 | -0.27 / -0.24 / 1.65 / -74 | 0.42 |
| position | 30 | E | -0.21 / -0.27 / 1.38 / -69 | -0.19 / -0.24 / 1.29 / -65 | -0.21 / -0.33 / 1.27 / -58 | 0.20 |
| position | 90 (3) | CV | -0.40 / -0.38 / 1.58 / -33 | -0.36 / -0.39 / 1.63 / -25 | -0.37 / -0.40 / 1.65 / -25 | 0.28 |
| position | 90 | R | -0.29 / -0.31 / 1.37 / -34 | -0.26 / -0.28 / 1.33 / -34 | -0.04 / -0.03 / 1.04 / -33 | 0.20 |
| position | 90 | E | -0.00 / -0.05 / 1.11 / -1 | +0.03 / -0.03 / 1.12 / +6 | +0.02 / -0.11 / 1.19 / +3 | 0.06 |
| position | 150 (5) | CV | -0.19 / -0.24 / 1.49 / -14 | -0.13 / -0.14 / 1.55 / -8 | -0.14 / -0.15 / 1.56 / -9 | 0.25 |
| position | 150 | R | -0.25 / -0.27 / 1.41 / -23 | -0.24 / -0.26 / 1.39 / -23 | +0.03 / +0.00 / 1.17 / +5 | 0.21 |
| position | 150 | E | +0.05 / +0.00 / 1.17 / +8 | +0.06 / +0.02 / 1.18 / +11 | +0.10 / -0.05 / 1.27 / +12 | 0.10 |
| velocity | 1 | R | -0.01 / -0.08 / 1.83 / -90 | -0.03 / -0.04 / 1.86 / -89 | +0.06 / +0.16 / 2.01 / +88 | 0.36 |
| velocity | 1 | E | -0.07 / -0.36 / 1.69 / -86 | -0.07 / -0.31 / 1.66 / -86 | -0.13 / -0.44 / 1.70 / -83 | 0.32 |
| velocity | 15 | CV | -0.51 / -0.45 / 2.43 / -72 | -0.62 / -0.52 / 2.39 / -64 | -0.62 / -0.51 / 2.39 / -64 | 0.50 |
| velocity | 15 | R | -0.55 / -0.44 / 2.25 / -67 | -0.55 / -0.46 / 2.14 / -64 | -0.29 / -0.18 / 1.84 / -76 | 0.47 |
| velocity | 15 | E | -0.22 / -0.30 / 1.52 / -74 | -0.23 / -0.29 / 1.43 / -70 | -0.26 / -0.38 / 1.41 / -65 | 0.26 |
| velocity | 30 | CV | -0.50 / -0.45 / 1.84 / -58 | -0.53 / -0.49 / 1.82 / -49 | -0.54 / -0.49 / 1.83 / -48 | 0.37 |
| velocity | 30 | R | -0.42 / -0.55 / 1.57 / -50 | -0.38 / -0.53 / 1.50 / -47 | -0.20 / -0.26 / 1.29 / -63 | 0.28 |
| velocity | 30 | E | -0.16 / -0.21 / 1.19 / -55 | -0.13 / -0.17 / 1.14 / -47 | -0.13 / -0.26 / 1.15 / -35 | 0.11 |
| velocity | 90 | CV | -0.04 / -0.06 / 1.49 / -3 | +0.03 / +0.02 / 1.50 / +2 | +0.03 / +0.02 / 1.50 / +2 | 0.25 |
| velocity | 90 | E | +0.07 / +0.02 / 1.17 / +13 | +0.08 / +0.03 / 1.17 / +16 | +0.09 / -0.04 / 1.22 / +14 | 0.10 |
| velocity | 150 | E | +0.05 / +0.00 / 1.17 / +8 | +0.06 / +0.01 / 1.18 / +11 | +0.11 / -0.04 / 1.28 / +13 | 0.11 |

The P rows and the remaining rows are in `baselines-bench-tables.md`.

**The least-correlated frame, by horizon** (turn-folded |Pearson|, |Spearman|):

| horizon | CV | R | E |
|---|---|---|---|
| 1 tick | - | nose ~ track (both <= 0.04) | nose ~ track (Pearson 0.02; Spearman 0.14-0.15) |
| 3 ticks (0.1 s) | nose (0.09 / 0.00, against track 0.39 / 0.47) | nose (0.06 / 0.09) | nose ~ track (0.08-0.09 / 0.26-0.30) |
| 15 (0.5 s) | nose (0.45 / 0.38, track 0.62 / 0.52) | nose ~ track (0.51-0.53 / 0.32-0.35) | nose ~ track (0.19-0.21 / 0.28) |
| 30 (1 s) | nose (0.55 / 0.46, track 0.63 / 0.51) | nose ~ track (0.53-0.55 / 0.51-0.52) | nose ~ track (0.19-0.21 / 0.24-0.27) |
| 90 (3 s) | nose ~ track (0.36-0.40) | track (0.26 / 0.28) | nose ~ track (<= 0.05) |
| 150 (5 s) | track (0.13 / 0.14) | nose ~ track (0.24-0.25) | nose ~ track (<= 0.06) |

The world frame at a fixed heading is the most correlated frame at every horizon up to 1 s
(0.2-0.55): a fixed world axis sees the ship's anisotropic ellipse at an arbitrary angle. It
falls to 0.06-0.25 at 3-5 s.

Reading:

- **One tick.** For R and E the body-frame axes are nearly uncorrelated (Pearson at most
  0.08). The ellipse is axis-aligned and about twice as long across as along (ratio
  1.7-2.6, angle +-83-90 deg). E's Spearman (0.14-0.44) is above its Pearson: a correlated
  core with uncorrelated tails.
- **0.1-1 s.** CV and R build up a strong along-across correlation of 0.4-0.6 with the
  mirror folded. Its sign is negative in the folded frame: for CV, the ship that keeps
  turning ends up inside the turn and behind the straight forecast. The major axis tilts 10-30 deg off
  across, with ratio 2-2.7. E keeps the correlation at 0.2-0.3, with the ellipse nearer the
  axes (ratio 1.3-1.9).
- **3-5 s.** E's ellipse is near round and uncorrelated (ratio 1.1-1.3, |r| at most 0.1). CV
  and R stay correlated at 0.2-0.4, with ratio 1.4-1.6.
- **Nose against track.** The two body frames are within 0.02-0.07 of each other for R
  and E at every horizon. The nose frame is clearly less correlated only for CV at 0.1-1
  s.
- **Folding by the curve** (last tick's normal acceleration) instead of the turn makes R's
  error less correlated (0.27 against 0.53 at 0.5 s) and E's slightly more. The
  correlation's sign follows the turn the ship was making.

### 2.3 Error scale at the horizons (E forecast, pooled, median |component| along / across the nose)

| k | position (m) | velocity (m/s) |
|---|---|---|
| 1 | 0.00022 / 0.0011 | 0.010 / 0.064 |
| 3 | 0.0032 / 0.014 | 0.071 / 0.34 |
| 15 | 0.26 / 0.66 | 1.09 / 3.0 |
| 30 | 1.25 / 2.9 | 3.1 / 6.7 |
| 90 | 21 / 38 | 20 / 27 |
| 150 | 73 / 87 | 36 / 38 |

The same for CV and R, with p99 norms, is in `baselines-bench-tables.md`.

## 3. Reading per quantity

- **Position.** E leaves the least: a median of 1.9 mm against R's 6.6 mm and P's 2.6 m,
  and a p99.9 of 0.135 m against 0.296. Held and quiet, E's along component is at float32
  resolution (0.08 mm). The cost: when the power changes, E's along-nose median is 12 mm
  against R's 4.9 mm. E-arc and the literal E agree to within noise.
- **Velocity.** E leaves the least: a median of 0.116 m/s against R's 0.333 and P's 0.598,
  and a p99.9 of 8.2 against 14.1-14.4. R does not beat P on the tail (14.4 against 14.1:
  the over-rotation in transients). The cost is the power change, where E's along-track median (0.715 m/s) is
  2.5 times P's and R's. The field band is the tail for every baseline: velocity p99.9 7-21
  m/s in a band against 1.5-6 outside.
- **Heading.** E has the smallest median (0.63 against R's 0.97 mrad) but the largest tail
  (p99.9 0.154 against 0.110 rad), and loses to R on 56% of changed-control ticks. Under
  today's code, R and E tie on entropy (1.41 nats, against P's 1.76).
- **Nose rate.** R is P. E narrows the median by 14% and widens the p99.9 by 60%. The
  oscillator's held decay of 0.41 per tick and its reversal (-0.54) after a control change
  make last tick's rate a poor guess. A decay of about 0.4 per tick, held, would remove
  97% of the held change's mean square (section 1.4).
- **Angle of attack.** E halves the median (1.4 against 2.9 mrad) and widens the p99.9 by
  15%. It removes 68% of the mean square when held and quiet and adds 41% when the control
  changes. Entropy under today's code: 1.30 against 1.42 nats.
- **Energy.** E leaves nearly nothing when the power is held (median 1e-6 of max, beta 0.998).
  Its residual is then the shot (2.6% steps, the same for every baseline) and the power
  change, where E's median is 4 times P's. E has 0.25 nats less entropy than P under today's
  code, and a p99.9 of 0.035 against 0.030 (about a shot's 0.030 plus a carried rate of the wrong sign).
- **Shield, recharge wait, gun cooldown.** R and E are the same to within float rounding.
  Both leave exactly the events (hits, resets, shots): 98.0% below 1e-6 for the shield, the
  same p99.9, and 0.08-0.11 nats. Away from hits the shield residual is below 4e-8. Against
  P, R and E halve the event entropy of the cooldown and remove the wait's -dt countdown
  (0.45 -> 0.10 nats).
- **ln n.** R is P. E cuts the p99 5.6x (0.0035 against 0.0195) and the p99.9 3.6x (0.020
  against 0.072). Everything is inside field bands, where E removes 92% of the change's
  mean square.

## 4. Notes and open points

- **The pooled correlation is zero by symmetry.** The brief's Pearson and Spearman, taken
  literally on the pooled residual, show |r| < 0.08 everywhere. That figure says nothing
  about the per-sample correlation that independent categoricals cannot hold. Section 2.2
  folds the left-right mirror out with the turn sign. A head conditions on more than the
  turn sign, so its residual correlation could be lower than these. They are the
  correlation of the baseline's error given only the turn direction.
- **E-arc.** The brief's literal E position is not exact on a steady turn. E-arc is, and
  the two measure the same on the data.
- **The R here.** R is `delta.py`, so it is persistence for ln n and energy. The state-deltas
  report's ln n dead reckoning along the gradient (p99.9 0.013) did better than E (0.020)
  on ln n.
- **Old recordings.** The segments were recorded on 2026-11-07, before the commits on
  `feat/resim-shot-labels` that change the guns, the map size and spawn shields. Shield
  and shot rates on that branch would differ; flight would not.
- **Float32 recordings.** The held-control E residuals of position and energy are at the
  recording's float32 resolution. Below about 1e-4 m and 1e-6 of max, the true residual is
  not resolved.
- **The horizon frames** are those at t (the last true state). Errors expressed in the
  forecast's own frame at t + k were not measured.
- **The E forecast** holds the acceleration in the velocity frame. On a decelerating ship
  it can carry the speed through zero; the long-horizon medians include such cases.
- **Run.** Measure 8 min on 8 CPU threads, about 2 GB peak, under an 8 GB scope. It was
  measured twice: the first pass lacked the mirror signs; the one-tick numbers are the same.
