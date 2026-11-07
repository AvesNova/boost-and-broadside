# How much each ship-state quantity changes per tick, and how to code its delta target

Brief: `state-deltas-brief.md`, with the author's two additions (body frames; then the
refocus: hard bounds first, then strong structure, then codes sized to the bounds). No
model change. Scripts, all CPU:

| Script | What it does |
|---|---|
| `bench/state_deltas_collect.py` | self-play rollouts; one `.npz` per 1024-tick segment of the privileged ship rows, the actions and the event flags |
| `bench/state_deltas_measure.py` | the changes (1, 8, 32 ticks; every baseline; world and body frames), statistics, situation splits, histograms, subsamples |
| `bench/state_deltas_bounds.py` | the flight envelope and the hard one-tick bounds, from the game's own `flight_tick` |
| `bench/state_deltas_structure.py` | point masses, discrete steps, power and stick clusters, event rates |
| `bench/state_deltas_codes.py` | the candidate codes compared at equal bin budgets, and a bin-count sweep |
| `bench/state_deltas_recommend.py` | the recommended codes evaluated |
| `bench/state_deltas_report.py` | the full tables (`state-deltas-tables.md`) |

Tests: `tests/sections/test_state_deltas.py` (the pair mask, the kinematic baselines on an
exact turn, the codes' round trips). Histograms (log-spaced |change| bins, 10 per decade
from 1e-9, signed counts, per component and baseline):
`/tmp/bnb-scratch-deltas/histograms/<player>/<preset>/<quantity>.json`.

## Summary

| Quantity | Target (baseline) | Hard bound, one tick | Observed max (p99.9) | Structure | Recommended delta code |
|---|---|---|---|---|---|
| position | x_{t+1} - (x_t + arc dead reckoning) | 8.7 m (11 m without a baseline) | 0.69 m (0.38) | none at 0; field edges make the tail | nose frame, per axis, symlog, knee 2-3 mm, range +-10 m, **37 / 35 bins** |
| velocity | v_{t+1} - R(omega_t dt) v_t | 375 m/s (box; see 1.2) | 32 m/s (14) | along the nose: three clusters by power; across: near 0 with the stick centred | nose frame, per axis, symlog, knee 0.1 m/s, range +-400, **35 + 35 bins** |
| heading | h_{t+1} - h_t - omega_t dt, wrapped | 0.90 rad | 0.25 (0.13) | none | symlog, knee 1 mrad, range +-1 rad, **29 bins** (not circular at one tick) |
| nose rate | persistence | 24.7 rad/s (box) | 10.2 (5.6) | refraction drives the tail | symlog, knee 0.05, range +-25, **27 bins** |
| angle of attack | persistence | 0.49 rad (box) | 0.16 (0.087) | none | symlog, knee 0.003, range +-0.5, **23 bins** |
| energy (fraction of max) | persistence | 0.038 flight + 0.030 a shot = 0.068 | 0.046 (0.030) | a shot is a step of -0.026 (fighter -0.030, heavy -0.026); power sets the sign of the rest | symlog, knee 5e-4, range +-0.07, **21 bins** |
| shield (fraction of max) | the deterministic recharge | 1 (a broken shield) | 0.36 (0.12) | 93-98% exactly 0; hits are steps of 0.075-0.16 per bullet; boundary drain 0.001-0.002 per tick | symlog, knee 0.001, range +-1, **29 bins** |
| shield recharge wait | the countdown max(w - dt, 0) | reset to 6 s | 6 s | exactly 0 except a reset on any damage (2.0% of ticks) | **2 bins**: reset or not (the value after a reset is the delay, known) |
| gun cooldown | the countdown max(c - 1, 0) | interval - 1 ticks (1 fighter, 2 heavy) | 2 ticks | exactly 0 except a shot (1.7% of ticks) | **2 bins**: fired or not (the value is the type's interval, known) |
| ln n | ln n_t + grad ln n_t . v_t dt | 0.33 (box) | 0.085 (0.013) | 95-99% exactly 0 (out of every field band) | symlog, knee 6e-4, range +-0.35, **27 bins** |

Every symlog code here is two-hot with an exact-zero centre and a bin of about one knee at
the knee (K = 1 + 4 ln(1 + range / knee), odd). It covers the hard bound with no clipping
on the 2.4M pooled samples, and its marginal entropy is 1.9-2.5 nats per continuous component. The
event quantities' is under 0.1 nats.

Today's absolute codes are 10-1000 times coarser than the typical one-tick change for
every continuous quantity (table 3.1). The nose rate goes past today's code range of
+-8 rad/s: the largest observed is 9.1 rad/s, on 5e-6 of ticks.

## 0. Samples and today's code

Players: the scripted agent (default ship card) and calm-fern
`players-calm-fern/step_100663296-exponential_average`, each in self-play. Both play the
same card; the two cards were compared field by field. calm-fern's build record lacks four
newer Yemong fields (`forecast_feedback`, `rotary`, `order`, `rotary_readout`). They load
at their defaults, which were the behaviour before the fields existed, and the weights load
strictly. Fog on, staggered clocks, the CPU, team 0's privileged rows (every ship, sim
team-major). A pair (t, t + k) counts only if the ship is alive at every tick from t to
t + k and no reset falls in between.

| player | preset | environments | ticks | episode ends | games touched | living ship-ticks (1 / 8 / 32-tick pairs) |
|---|---|---|---|---|---|---|
| scripted | frontline | 128 | 3 x 1024 | 53 | 181 | 3,455,509 / 3,413,860 / 3,272,844 |
| scripted | deathmatch | 128 | 3 x 1024 | 45 | 173 | 3,600,081 / 3,557,299 / 3,412,360 |
| scripted | deathmatch_wide | 128 | 3 x 1024 | 45 | 173 | 3,602,121 / 3,559,119 / 3,413,424 |
| calm-fern | frontline | 64 | 3 x 1024 | 22 | 86 | 1,846,666 / 1,829,100 / 1,769,359 |
| calm-fern | deathmatch | 64 | 3 x 1024 | 13 | 77 | 1,901,843 / 1,884,998 / 1,827,596 |
| calm-fern | deathmatch_wide | 64 | 3 x 1024 | 26 | 90 | 1,855,557 / 1,835,906 / 1,769,046 |

The code comparison pools 400,000 one-tick samples from each player and preset: 2.4M at
equal weight.

**Today's code** (`state_code.py`, `codes.py`):

- Position: per axis, a 3-colour circular code at 9 levels with periods 26 244 m / 3^l,
  joined as the outer product (9 cells per level, 81 units). The finest level's period is
  4 m, so its colour spacing is 4/3 m.
- Velocity: the velocity is compressed radially, v -> v knee ln(1 + |v| / knee) / |v|
  (knee 50 m/s). It is then projected on three axes at 0, 120 and 240 degrees, each an
  **81-bin two-hot scalar over +-100 compressed m/s (spacing 2.5)**, 3 x 81 = 243 units. The reach is 50 (e^2 - 1) = 319 m/s. At the median speed (70-80 m/s)
  one bin is about 2.5 (1 + v / 50) = 6.4 m/s radially and 4.1 m/s tangentially.
- Heading: a circular code, 4 colours at 4 levels (period 2 pi / 4^l), finest spacing
  2 pi / 256 = 0.0245 rad.
- Nose rate: 41 bins over +-8 rad/s (0.4 rad/s spacing).
- Angle of attack (+-30 deg), recharge wait (0-6 s), shield and energy (0-1 of the type's
  maximum), ln n (+-ln 2): 21 bins each.
- Gun cooldown: one bin per tick, 4 bins.

The head adds log(code(state_t) + 1e-3), persistence, to its logits.

## 1. Hard bounds of the one-tick change

### 1.1 Envelope, from the game's `flight_tick`

| | fighter | heavy |
|---|---|---|
| corner speed sqrt(a_max / cl_max) | 89.4 m/s | 85.3 m/s |
| terminal proper speed, coasting | 103.7 m/s | 106.0 m/s |
| terminal proper speed, boosting with the store held full (an upper bound: boosting spends the store) | 210.7 m/s | 195.3 m/s |
| world speed u / n, n >= 1/2 (level -2) | <= 421 m/s | <= 391 m/s |
| observed: proper / world speed | 120 / 232 m/s | 130 / 232 m/s |
| observed: abs(nose rate), abs(angle of attack) | 9.13 rad/s, 0.27 rad (1.03 stall angles) | |
| abs(grad ln n): one level-2 field 2 ln sqrt2 x 1.875 / 40 m = 0.0325/m; observed 0.0425/m (two fields' bands overlapping) | | |

The terminal-speed bound exceeds today's velocity reach (319 m/s) inside a level -2 field.
No ship came close (232 m/s).

### 1.2 Bounds

The one-tick bounds are maxima of `flight_tick` over a box, in `bench/state_deltas_bounds.py`.
The box spans proper speed from 0 to the terminal speed; nose rate within +-1.25 x 9.13
rad/s; angle of attack within +-1.25 x 0.27 rad; any store, all 21 stick and power
pairs; and positions over two overlapping fields of every level pair. The box is set from
the observed nose rate and angle of attack because both are states of the nose oscillator,
which has no simple closed bound. The box holds states no ship reaches (a 11 rad/s nose
rate at walking speed), so its bounds are loose. A tighter box, proper speed up to
1.25 x the observed 130 m/s, is in the second column. The leading terms, as formulas:

| Quantity | Leading terms | Bound: box (box at observed speeds) | Observed max, both players | Beyond? |
|---|---|---|---|---|
| abs(dx) | abs(v) dt, abs(v) <= u_terminal / n_min | 14.8 m (11.2) | 7.99 m | no |
| arc residual | ~ abs(dv) dt / 2 | 8.70 m (5.46) | 0.69 m | no |
| abs(dv) | dt (a_lateral + a_along) / n + abs(v) abs(d ln n) + abs(v) abs(d alpha); a_lateral <= a_max = 216 / 160, a_along <= T + D + 27 (exchange); abs(v) abs(d ln n) dominates at a field edge | 258 m/s (148) | 40.6 m/s | no |
| rotated residual | the same less the turn | 375 m/s (240) | 34.8 m/s | no |
| abs(d heading) | abs(omega_mid) dt; omega ~ (C u^2 + T sin alpha) / (u n) + (u / n) abs(grad ln n) cos | 0.55 rad (0.43) | 0.31 rad | no |
| heading residual | abs(omega_mid - omega_t) dt | 0.90 rad (0.75) | 0.25 rad | no |
| abs(d omega) | the oscillator: Omega = omega_c u / (u_c n), damping 0.7 | 24.7 rad/s (23.5) | 10.2 rad/s | no |
| abs(d alpha) | the oscillator's turn | 0.49 rad (0.49) | 0.16 rad | no |
| abs(d energy) / E_max | flight: m u a / n dt, a <= 27, = 27 x 211 x 2 / 30 / 10 000 = 0.038 (the box gives 0.0379); a shot: -300 / 10 000 or -360 / 14 000 | 0.038 + 0.030 = 0.068 | 0.046 | no |
| abs(d ln n) | abs(grad ln n) abs(dx), at most 2 ln 2 | 0.48 (0.36) | 0.23 | no |
| ln n residual | the gradient's change across the move | 0.33 (0.22) | 0.085 | no |
| shield | recharge +12 dt / max = +0.004 (fighter), +0.0025 (heavy); a hit -12 or -16 per bullet (0.12 / 0.16 of a fighter, 0.075 / 0.10 of a heavy); boundary -(5 + 0.05 depth) dt / max; clamped at 0 | -1 to +0.004 | -0.36 | no |
| recharge wait | -dt, or reset to the delay (6 s) on any damage | residual in {0} or (dt, 6] | 6 s | no |
| gun cooldown | -1 tick, or set to the interval on a shot (2 fighter, 3 heavy), then the clock | residual {0, 1} fighter, {0, 2} heavy | 2 ticks | no |

No observation went past its bound. The bounds are maxima of the flight step alone. The
shot's energy and the damage are added from the rules.

## 2. Strong structure

### 2.1 Point masses and discrete steps (one tick, living ships)

| | scripted frontline / deathmatch / wide | calm-fern frontline / deathmatch / wide |
|---|---|---|
| gun cooldown residual 0 / 1 / 2 ticks | 98.3 / 1.1 / 0.6% (every preset within 0.2%) | 98.4 / 1.0 / 0.6%; 98.8 / 0.7 / 0.5%; 97.8 / 1.3 / 0.9% |
| recharge wait residual exactly 0 (else a reset) | 98.6 / 98.4 / 98.3% | 98.7 / 97.9 / 96.1% |
| shield residual exactly 0 (after the recharge) | 98.6 / 98.4 / 98.3% | 98.7 / 97.9 / 96.1% |
| ... a hit (loss > 0.05) | 0.47 / 0.52 / 0.50% | 0.21 / 0.13 / 0.24% |
| ... a small loss (boundary drain) | 0.9 / 1.1 / 1.2% | 1.1 / 1.9 / 3.6% |
| energy change on a shot (median) | -0.026 / -0.025 / -0.025 | -0.026 / -0.025 / -0.025 |
| energy change without a shot, p0.1 / p99.9 | +-0.0094 / +-0.0094 / +-0.010 | -0.0088 / +0.0091 ... -0.0068 / +0.0087 |
| ln n change exactly 0 | 95.2 / 95.8 / 98.4% | 95.0 / 95.6 / 98.8% |
| angle-type quantities exactly 0 (heading, nose rate, angle of attack) | 0-4% | 0-4% |
| position, velocity, energy exactly 0 | 0% | 0% |

A shot, a hit and a boundary tick each move several quantities at once. A shot steps the
cooldown and the energy. Any damage resets the recharge wait and drops the shield.

### 2.2 Clusters by power and stick (velocity residual in the nose frame, one tick, no shot)

| player, preset | coast: mean / p5 / p95 (m/s) | boost | reverse |
|---|---|---|---|
| scripted deathmatch | +0.04 / -0.48 / +0.34 | +0.72 / -0.04 / +1.34 | -0.48 / -1.19 / +0.04 |
| calm-fern deathmatch | +0.04 / -0.63 / +0.38 | +0.74 / -0.01 / +1.38 | -0.42 / -0.96 / +0.14 |

Every preset shows the same three clusters (the structure section of
`state-deltas-tables.md`). The energy
change has the opposite sign: about -0.004 per tick on a boost, +0.003 on a reverse and
+0.0004 coasting. The stick sets the across component. With the stick centred, calm-fern's
across residual has a median of 1.8e-4 m/s, a near point mass, against 0.44 m/s at full
stick (scripted: 0.15 against 0.19). Along the nose, power moves the mean; across it, the
stick moves the spread.

### 2.3 Rare large events

| Event | Share of living ticks |
|---|---|
| fires | 1.2-2.2% |
| shot hit | 0.75-1.7% |
| boundary damage (outside the arena) | scripted 0-0.16%; calm-fern 0.4-2.6% (it flies outside the arena far more) |
| in a field or its band | 12-19% |
| in a Frontline zone | 34-37% |
| within 1 s of a spawn | 1.9-3.1% |
| abs(d ln n) > 0.05 (a band crossed fast) | 0.06-0.39% |
| abs(d nose rate) > 2 rad/s | 0.7-4.0%, of which in a field: scripted 49-68%, calm-fern 12-22% |
| velocity residual > 3 m/s | 2.3-6.4%, of which in a field: scripted 54-68%, calm-fern 23-33% |

There are no ship collisions in v2. Respawns are not living pairs (a dead ship's tick is
excluded). The largest velocity changes (30-40 m/s) are field edges: v times d ln n at
230 m/s.

### 2.4 One-tick changes, brief (p50 / p99.9 / max)

| quantity | baseline | scripted frontline | scripted deathmatch | scripted wide | calm-fern frontline | calm-fern deathmatch | calm-fern wide |
|---|---|---|---|---|---|---|---|
| position (m) | persistence | 2.59 / 6.86 / 7.51 | 2.68 / 6.92 / 7.7 | 2.31 / 7.25 / 7.99 | 2.86 / 6.45 / 7.25 | 2.69 / 6.63 / 7.49 | 2.36 / 5.98 / 7.1 |
| position (m) | arc | 0.0068 / 0.38 / 0.68 | 0.010 / 0.26 / 0.69 | 0.0057 / 0.32 / 0.68 | 0.0056 / 0.31 / 0.65 | 0.0075 / 0.29 / 0.65 | 0.0052 / 0.15 / 0.62 |
| velocity (m/s) | persistence | 0.64 / 15.3 / 37.2 | 0.63 / 14.7 / 40.6 | 0.70 / 12.3 / 37.4 | 0.44 / 14.9 / 34.6 | 0.57 / 15.4 / 38.4 | 0.62 / 10.4 / 30.9 |
| velocity (m/s) | rotated | 0.34 / 18.6 / 34.8 | 0.48 / 13.3 / 32.2 | 0.30 / 15.2 / 34.2 | 0.30 / 14.7 / 30.6 | 0.35 / 13.9 / 34.5 | 0.29 / 7.38 / 28.4 |
| heading (rad) | persistence | 0.010 / 0.12 / 0.29 | 0.011 / 0.10 / 0.31 | 0.012 / 0.10 / 0.24 | 0.0053 / 0.13 / 0.27 | 0.0097 / 0.12 / 0.24 | 0.010 / 0.099 / 0.22 |
| heading (rad) | dead reckoning | 0.0015 / 0.13 / 0.25 | 0.0028 / 0.085 / 0.25 | 8.3e-4 / 0.12 / 0.25 | 6.2e-4 / 0.12 / 0.25 | 8.6e-4 / 0.11 / 0.25 | 1.6e-4 / 0.073 / 0.24 |
| nose rate (rad/s) | persistence | 0.080 / 5.59 / 10.2 | 0.14 / 3.56 / 9.67 | 0.043 / 4.89 / 9.78 | 0.034 / 5.42 / 9.89 | 0.048 / 4.87 / 9.93 | 0.0089 / 3.43 / 9.72 |
| angle of attack (rad) | persistence | 0.0034 / 0.084 / 0.16 | 0.0054 / 0.065 / 0.16 | 0.0026 / 0.070 / 0.16 | 0.0022 / 0.087 / 0.16 | 0.0028 / 0.080 / 0.16 | 8.6e-4 / 0.070 / 0.15 |
| energy (of max) | persistence | 5.8e-4 / 0.030 / 0.045 | 6.6e-4 / 0.030 / 0.045 | 6.3e-4 / 0.030 / 0.044 | 4.5e-4 / 0.030 / 0.045 | 3.6e-4 / 0.030 / 0.043 | 4.8e-4 / 0.030 / 0.046 |
| shield (of max) | recharge | 0 / 0.12 / 0.32 | 0 / 0.12 / 0.36 | 0 / 0.12 / 0.33 | 0 / 0.081 / 0.36 | 0 / 0.060 / 0.28 | 0 / 0.098 / 0.32 |
| ln n | persistence | 0 / 0.085 / 0.19 | 0 / 0.079 / 0.19 | 0 / 0.049 / 0.21 | 0 / 0.075 / 0.18 | 0 / 0.078 / 0.23 | 0 / 0.041 / 0.16 |
| ln n | dead reckoning | 0 / 0.013 / 0.085 | 0 / 0.012 / 0.082 | 0 / 0.0057 / 0.058 | 0 / 0.011 / 0.073 | 0 / 0.011 / 0.077 | 0 / 0.0041 / 0.045 |

Dead reckoning and the integrator:

- **Position.** The game integrates kick-drift-kick. The drift moves along the nose arc
  with the half-kicked velocity. So x_t + (v_t + v_{t+1}) dt / 2 matches it: its residual
  has a median of 3e-4 m, 65-76% of it below float32 position resolution, but it reads
  v_{t+1}. Of the baselines known at t, the arc x_t + R(omega_t dt / 2) v_t dt sinc(...)
  is best. Its median of 0.005-0.010 m is 2.5 times better than the straight v_t dt
  (0.024 m) and about 300 times better than persistence. The tails are the same for all
  three: field edges.
- **Heading.** heading_t + omega_t dt cuts the median about 10 times (persistence 0.010 ->
  0.001 rad), but not the p99.9 (0.07-0.13 rad either way: refraction transients of the
  nose).
- **Velocity.** Rotating v_t by omega_t dt halves the median (0.6 -> 0.33 m/s) and leaves
  the tail.
- **The other quantities.** The baselines chosen were the countdowns (recharge wait, gun
  cooldown), the recharge (shield) and ln n + grad ln n . v dt (ln n, which cuts the p99.9
  about 7 times).

The 8- and 32-tick changes, all baselines and frames, the body-frame tables and the splits
are in `state-deltas-tables.md`.

## 3. Codes

### 3.1 Today's resolution against the change

| quantity | today's spacing | median one-tick change (best baseline) | ratio |
|---|---|---|---|
| position | 1.33 m (finest colour) | 0.0066 m (arc) | 200 |
| velocity | ~4-6 m/s per bin at 80 m/s | 0.33 m/s (rotated) | 12-19 |
| heading | 0.0245 rad | 0.00097 rad (dead reckoning) | 25 |
| nose rate | 0.4 rad/s | 0.052 rad/s | 8 |
| angle of attack | 0.052 rad | 0.0029 rad | 18 |
| energy | 0.05 | 0.00054 | 90 |
| ln n | 0.069 | 0 (96%); nonzero median 0.0006 | 115 |
| shield, recharge wait, cooldown | 0.05, 0.3 s, 1 tick | exactly 0 on 96-98% of ticks | - |

Two-hot decodes a value exactly at any spacing. The spacing sets how narrow a predicted
distribution the code can hold. At today's spacing, the next state of most quantities is
one bin's two-hot.

### 3.2 Candidates (pooled 2.4M samples; 1-D at 21 bins, 2-D at 54)

The full tables are in `state-deltas-tables.md`, section "Codes". Ranges are the pooled
max |change| rounded up to 1, 2 or 5 x 10^k. H is the marginal entropy of the code: the
cross-entropy of a predictor that knows only the change's distribution, maximised by
equal-mass bins. H_self is the mean entropy of one two-hot target: the floor for a
predictor that knows each change exactly.

| Change | uniform (max range): width at p50 (x median), H | symlog (knee = median): width, H | quantile (+ zero): width, width at p99.9, H | hierarchy 7 x 3: width, H (H_self) |
|---|---|---|---|---|
| heading, dead reckoning | 0.05 (51x), 0.44 | 0.0016 (1.6x), 2.13 | 0.0015 (1.5x), 0.23 rad, 2.96 | 4.6e-4 (0.47x), 6.31 (3.6) |
| nose rate | 1.0 (19x), 0.83 | 0.062 (1.2x), 2.26 | 0.074 (1.4x), 9.1, 2.96 | 0.009 (0.18x), 6.63 |
| angle of attack | 0.02 (7x), 0.95 | 0.0023 (0.8x), 2.28 | 0.0024 (0.9x), 0.14, 2.96 | 1.8e-4, 6.76 |
| energy | 0.005 (9x), 0.67 | 4.9e-4 (0.9x), 1.89 | 9.4e-4 (1.7x), 0.043, 2.97 | 4.6e-5, 6.41 |
| ln n, dead reckoning | 0.01, 0.04 | 4.0e-4, 0.17 | 1.7e-5, 0.045, 0.25 | 9.1e-5, 5.03 |

| 2-D change, 54 units | world x, y uniform | world x, y symlog | nose frame symlog | velocity frame symlog | polar (nose) | three directions | joint hierarchy 6 x 9 |
|---|---|---|---|---|---|---|---|
| position, arc: width at p50 (x median) / H | 0.077 (12x) / 0.93 | 0.0030 (0.45x) / 4.80 | 0.0039 (0.59x) / 4.53 | 0.0031 (0.47x) / 4.49 | 0.0030 (0.45x) / 5.54 | 0.0058 (0.88x) / 6.11 | 0.0027 (0.42x) / 10.37 |
| velocity, rotated | 3.85 (12x) / 0.93 | 0.155 (0.46x) / 4.80 | 0.20 (0.60x) / 4.54 | 0.20 (0.60x) / 4.51 | 0.148 (0.45x) / 5.53 | 0.30 (0.90x) / 6.10 | 0.137 (0.41x) / 10.38 |

Read-out:

- **Uniform over the measured range** is 7-50 times too coarse at the median. Clipping to
  the 99.9% range only halves that.
- **Quantile bins** have the most entropy, by construction. They give up the tail: one bin
  holds everything past p99.9 (0.23 rad for the heading, 9.1 rad/s for the nose rate). They
  also move with the player's distribution, which training changes.
- **The multi-level hierarchy** resolves 10-100 times finer than needed. It pays 5-7 nats
  of marginal entropy and a 3.6-7 nat floor for an oracle, because its fine levels hold
  near-uniform noise that the model would have to learn to ignore.
- **Symlog** gives about one bin per knee at the median and stays log-spaced through the
  tail. It uses its range at no cost to the bulk, so it can be sized to the hard bound.
- **Frames.** The body frames reach the same resolution as world x, y with 0.3 nats less
  marginal entropy, and their axes match the controls (2.2). Nose and velocity frames are
  within 0.05 nats and a few percent of width of each other: the angle of attack is small
  (at most 0.27 rad), and no living ship was below the 1 m/s speed threshold. The nose
  frame is defined at every speed.
- **Polar** is finest at the median, but its angle is undefined at zero change and mixes
  the power and stick axes.
- **Three directions** cost 3 groups for 2 degrees of freedom and are twice as coarse.
- **Circular codes** buy nothing at one tick: the heading residual is at most 0.25 rad,
  far from wrapping. At 32 ticks it wraps (max pi), and a symlog over +-pi with a wrapped
  two-hot at the ends would do.

### 3.3 Bins

The symlog bin count was swept from 11 to 63 per component. The width at the median falls
as about 1/K in log-space. The marginal entropy grows by about 0.3 nats per 1.5x bins. With
the knee at the median and the range at the hard bound, K = 1 + 4 ln(1 + range / knee)
gives one knee-wide bin at the knee. That is enough for a distribution, since two-hot
decodes the value exactly, and the count lands at 21-37.

## 4. Recommendation

One code shape for every continuous quantity: **a symlog two-hot over the change from the
physics baseline**, with an exact-zero centre, the knee at the median nonzero change
(rounded to one figure) and the range at the hard bound. Two event quantities use **two
bins**. The physics baseline is added back at decode. With the target in the change frame,
the head's "nothing changes" prior is a constant one-hot at the zero centre, so the
persistence term log(code(state_t) + floor) becomes a learnable bias. Evaluated on the
pooled samples (`bench/state_deltas_recommend.py`):

| component | knee | range | bins | at 0 | width at p50 / p99 / p99.9 / max | clipped | H (nats) |
|---|---|---|---|---|---|---|---|
| position arc residual, along the nose (m) | 0.002 | +-10 | 37 | 0% | 0.0019 / 0.021 / 0.053 / 0.22 | 0 | 2.18 |
| position arc residual, across the nose (m) | 0.003 | +-10 | 35 | 0% | 0.0030 / 0.052 / 0.13 / 0.35 | 0 | 2.37 |
| velocity rotated residual, along the nose (m/s) | 0.1 | +-400 | 35 | 0% | 0.10 / 1.17 / 3.12 / 13.5 | 0 | 2.22 |
| velocity rotated residual, across the nose (m/s) | 0.1 | +-400 | 35 | 0% | 0.10 / 1.91 / 5.08 / 13.5 | 0 | 2.48 |
| heading dead-reckoning residual (rad) | 0.001 | +-1 | 29 | 0.3% | 0.0011 / 0.033 / 0.054 / 0.15 | 0 | 2.32 |
| nose rate change (rad/s) | 0.05 | +-25 | 27 | 0.4% | 0.050 / 1.41 / 2.27 / 5.9 | 0 | 2.36 |
| angle of attack change (rad) | 0.003 | +-0.5 | 23 | 0.4% | 0.0028 / 0.029 / 0.046 / 0.074 | 0 | 2.18 |
| energy change (of max) | 0.0005 | +-0.07 | 21 | 0.6% | 5.3e-4 / 0.010 / 0.017 / 0.028 | 0 | 1.86 |
| ln n dead-reckoning residual | 0.0006 | +-0.35 | 27 | 96.5% | 3.8e-4 / 0.0010 / 0.0044 / 0.031 | 0 | 0.18 |
| shield recharge residual (of max) | 0.001 (the heavy's boundary drain at the edge, 5 dt / 160) | +-1 | 29 | 93.5% | 6.4e-4 / 0.012 / 0.054 / 0.15 | 0 | 0.14 |
| recharge wait: reset to the delay | - | - | 2 | 98.0% no | - | - | 0.097 |
| gun cooldown: fired | - | - | 2 | 98.3% no | - | - | 0.086 |

**Frame per quantity.** Position and velocity: the nose frame (along, across heading_t).
It is defined at every speed, aligned with power (along) and stick (across), and within
noise of the velocity frame. Every other quantity is a scalar, frame-free.

**Bins.** 21-37 per component, 2 for the events: 302 units in all against today's 490.

**The belief's horizons** (8 and 32 ticks), with the same shape, if the belief adopts it:
arc position residual median 0.40 / 7.3 m, max 40 / 327 m; rotated velocity residual 3.1 /
16 m/s, max 286 / 454 m/s; heading dead-reckoning residual 0.035 / 0.21 rad, max 2.4 /
pi. At 32 ticks the heading needs wrapped ends.

**What to check in training:**

1. The bias at init. The zero centre should start with the share the table gives (93-98%
   for shield and ln n, 0-0.6% elsewhere), so the first loss is near H and not near ln K.
2. The tail. Cross-entropy by situation (in a field band, a hit, boost against coast)
   should fall as fast as overall: the tail is where the old code and the baselines both
   fail.
3. Decoded error against the one-step persistence and dead-reckoning baselines of
   section 2.4, per quantity. A model should beat dead reckoning, not just persistence.
4. Drift. The ranges are the physical bounds and no one player's distribution, but the
   knees are measured. Re-measure them after a few hundred million steps: a knee that moves
   3x changes the bin at the median by 3x and nothing else.
5. The velocity tail. A refraction term in the baseline, v_t exp(-grad ln n . v dt),
   would remove most field-edge jumps (30-40 m/s). That is one line, worth measuring before
   committing the range.

## 5. Notes and issues

- **Nose rate past today's code.** The nose rate exceeds today's code range of +-8 rad/s:
  observed to 9.1 rad/s, on 5e-6 of ticks, from refraction ((u / n) abs(grad ln n)). The
  code clamps it.
- **Velocity reach.** The world-speed bound inside a level -2 field (421 m/s) is above
  today's velocity reach (319 m/s). No ship came close (232 m/s).
- **calm-fern's boundary time.** calm-fern spends 0.4-2.6% of its ticks taking boundary
  damage, against 0-0.16% for the scripted agent.
- **Heading dead reckoning.** heading + omega_t dt has a worse p99.9 than persistence in
  scripted frontline and wide (0.134 against 0.116 rad). It removes the median, not the
  refraction transients.
- **Threads.** The first scripted measurement ran at 4 threads alongside the 8-thread
  calm-fern collection, briefly above the 8-thread cap. Everything later ran after the
  collection ended.
