# Look-ahead personas, round 3: the top of the hill and a curvature-shaped sampler

## Summary (2026-11-28 04:30)

Measured (laptop RTX 4070, performance profile; about 90 minutes of GPU, 02:55-04:27):

- **The summit was not at round 2's centre.** Of 96 Sobol points over an enlarged top region,
  69% beat the round-2 robust centre (mean over sizes; best +46 / +185 / +303 at 1v1 / 4v4 /
  16v16). The re-measured robust centre is level with round 2's at 1v1 (-8 +- 12) and above it
  at 4v4 (+102 +- 12) and 16v16 (+194 +- 14) in a fresh ladder.
- **Bounded rows** (skill falls off both ways within +-1.25 box units): horizon, speed weight,
  angles weight, preferred speed, energy weight, aggression, stickiness, and the temperature
  upward. **Plateaus up to a hard end:** collision weight (0 is fine: +81 / +51 / +5 against
  round 2's centre, no ramming sign), temperature downward (0.001: -11 / -29 / -60), fire
  tolerance upward (20: three times the shots, the same hits), threat weight, power
  temperature and power horizon (flat over +-2 box units). In these rows the "peak" sits
  wherever the data stops, in round 3 as in round 2.
- **Curvature at the peak** (Elo per box^2, mean surface): -332 (energy weight against angles
  weight, with speed weight), -192 (stickiness against speed weight and temperature), -143
  (horizon), -117 (preferred speed), -94; then 15 directions within +-30, and five slightly
  upward (+10 to +44: the plateau rows, threat, disengaged energy, power rows, fire tolerance).
- **The quadratic surface does not predict individual personas near the top**: r 0.04 (all
  rounds) and 0.14 (round-3 games only) between predicted and measured Elo drops of the
  sampler's personas. It ranks the round-2 sweep and design well (r 0.91, 0.95) because those
  span hundreds of Elo.
- **Sampler.** The curvature Gaussian at c for a predicted drop of 50 lost 79 Elo on average
  (17 / 80 / 141), 31% of personas more than 100 down. The main cause is one-sided: the
  temperature is a plateau below the centre and a cliff above it (measured Elo drop correlates
  with the temperature offset, r 0.42 / 0.59 / 0.64). With the temperature reflected to the cold
  side (same 24 personas otherwise): drop 23 (1 / 32 / 37), 4% more than 100 down, the same
  style spread and the same label blur (15 / 31 / 23% of the centre's label entropy).
- **League personas** (5, from the sampler): -51 to +39 Elo of the new centre at 1v1, -53 to
  +12 at 4v4, -49 to +70 at 16v16; team style store 0.34-0.52 (4v4), dive 0.52-0.85 (16v16),
  against round 2's set 0.15-0.23 store at 4v4.

Proposed (section "Recommendations"): the round-3 robust centre as the yardstick
(`candidates/robust-centre-3.json`), the sampler of `proposal-3.json` (curvature Gaussian with
each direction's sd capped at 0.35 box units, the temperature drawn only colder than the
centre, stickiness <= 0.15 and horizon >= 0.35 s), the five league personas `league3-1..5`.

Branch `personas-3` from core 902d3811. Scripts here: `round3.py` (agent groups, play, report,
ladder), `surface3.py` (step-1 summary, design, refit, summit, sampler), `local3.py`,
`sampler2.py`, `reflect3.py`, `blur3.py`, the loops `round3_loop.sh`, `ladder3.sh`,
`locked3.sh`. Raw games: `~/avesnova/boost-and-broadside/data/lookahead-personas-3/`
(`round3.sqlite`, `ladder.sqlite`, the agent groups `agents-*.json`). Conditions as round 2:
1v1 / 4v4 / 16v16, the game's defaults (fields, fog), the study clocks (400 / 90 / 80 s), every
agent the look-ahead agent. Elo is lean.eval's Bradley-Terry fit. "Box units": the round-1
spread-1 box width of each row (a log row's value x e^(2 width x offset)).

## Measurements

### 1. One-at-a-time from the round-2 robust centre

98 agents against the round-2 centre (`oat-3.json`; Elo with the centre at 0; 256 / 256 / 128
games each, SE about 25 / 25 / 35):

| row | -2 | -1.25 | -0.5 | +0.5 | +1.25 | +2 |
|---|---|---|---|---|---|---|
| threat_weight | +12/-24/+27 | +11/-12/+5 | +18/-42/+0 | -9/-16/+27 | +57/-23/-5 | +35/-23/-80 |
| collision_weight | +115/+57/+118 | +87/+46/+60 | +12/+60/+71 | -18/-38/-16 | -123/-106/-94 | -171/-111/-213 |
| horizon | -231/-248/-303 | -19/-35/+69 | -11/+31/+83 | -9/-41/-163 | -72/-100/-268 | -48/-155/-427 |
| temperature | +38/-45/-38 | -27/+1/-35 | +22/-8/+22 | -42/-51/-89 | -184/-187/-292 | -394/-410/-818 |
| speed_weight | -204/-142/-213 | -123/-120/-38 | -69/+1/-11 | -44/-30/-66 | -198/-202/-191 | -410/-332/-469 |
| power_temperature | -40/+0/+5 | +4/-20/-33 | -8/+4/+11 | +19/-8/-5 | +4/+53/-3 | -26/-30/-44 |
| fire_tolerance | -40/-168/-287 | -19/-96/-143 | -16/-20/-44 | -24/-1/-49 | +7/+5/+22 | -18/+23/+16 |
| power_horizon | +18/+1/-60 | +14/-9/+11 | -38/+11/+14 | +4/+35/+66 | -34/+0/-11 | +11/+11/-27 |
| preferred_speed | -109/-206/-592 | -96/-100/-170 | -34/-3/+5 | -57/-48/-38 | -121/+33/+22 | -114/-31/+44 |
| angles_weight | -280/-153/-163 | -231/-92/-44 | -87/-38/+8 | +16/-18/-124 | -20/-95/-233 | -52/-264/-553 |
| caution | +15/-63/-170 | +14/-38/-143 | -4/+19/-71 | -12/+35/-5 | +3/-3/+124 | -55/+67/+44 |
| reply_temperature | -5/-18/-83 | +16/-33/-83 | -4/+11/-16 | +12/+3/-55 | -51/-23/+27 | -66/+11/+77 |

At +-0.75: energy weight -33/-4/+11 and -23/-73/-83; aggression -95/+12/+60 and -22/-30/-49;
stickiness -44/+8/+14 and -8/-41/+11; store weight -29/+3/-16 and -23/-7/-55; power stickiness
+44/+14/+5 and -30/-24/+27; disengaged energy -8/-57/-27 and -31/+29/-22. Crowd ratios at +-1
(4v4 / 16v16): the energy crowd ratio up -52/-112, the speed crowd ratio up -80/-106, the
preferred-speed crowd ratio down +16/-137; the others within about 60.

Hard ends: collision weight 0: +81/+51/+5; temperature 0.001: -11/-29/-60; fire tolerance 20:
-15/-31/-131; horizon 0.1 s: -936/-1231/-1126.

- **Hard bounds in the code**: none; every row takes any positive value (no clamps). The
  look-ahead points are fixed at 0.25, 0.5, 1, 1.5, 2 s and the power look at 1, 2.5, 5 s, so
  a horizon well above 2 s (power horizon above 5 s) only flattens the time weights, and one
  below 0.25 s leaves one point that counts.
- **Degenerate play** (flags: shots under half the centre's, speed under 0.8 x, draws over 20%,
  Elo under 700): horizon 0.1 s (shots a tenth of the centre's), horizon 0.24 s (-231 to -303),
  the temperature at +2 (0.11; shots under half, Elo -394 to -818), the speed weight at +2,
  the fire tolerance at -2 (0.46: shots a third), the preferred speed at -2 (speed 0.54-0.66
  of corner). No agent drew over 20%.
- **Ramming**: not counted separately (a collision is a hit). Collision weight 0 shows no sign
  of it: the mean hit range 0.13 against the centre's 0.17 gun ranges at 1v1 (0.24 / 0.28
  against 0.27 / 0.30 in team games), hits per ship-minute 7.4 against 4.9 at 1v1.
- **Round 2's stall rule revisited.** Round 2's stalls (4v4 Elo near the floor) were all at
  stickiness 0.17-0.29; a ratio to the temperature of 8.6 at stickiness 0.106 played normally
  (921 at 4v4), and here the temperature 0.001 (ratio 44 at stickiness 0.044) and 0.0045
  (ratio 10) play normally. The stall follows the stickiness itself, not its ratio to the
  temperature.

### 2. The surface refitted

**Design** (`design-3.json`): 96 Sobol points over offsets from the round-2 centre per row
(`design-3-ranges.json`: e.g. collision -3..0, temperature -2.5..+0.25, fire tolerance
-0.5..+2, caution -0.5..+2, power horizon and temperature +-1.5 to 2; crowd ratios -1..+1),
256 / 256 / 128 games each against the round-2 centre. 69% beat it (mean over sizes); the best
five +134 to +178; median +41.

**Fit** (`surface-r3.json`; surface.py's model unchanged: per size a quadratic with every
product, ridge lambda 100): 711 agents, 2,207 pairings, 748,416 games (round 2's 627,456 plus
121,000 here; the collision-0 agent left out, off the log scale). Held-out deviance (5-fold
over agents) 75.9% of coin to saturated (round 2: 82.7% on its narrower range). It predicts
the step-1 agents with r 0.91 (RMSE 78 Elo) and the design with r 0.95 (RMSE 30), in sample.

**The summit** (`top-r3.json`; peak and robust centre as round 2: the centroid of the region
within 50 Elo of the peak, mean over sizes, inside the data's 2-98% range per size; boundary,
seek and field weight held at round 2's values):

| row | round-2 centre | round-3 centre | round-3 peak | 50-Elo region (5-95%, box units from the centre) |
|---|---|---|---|---|
| threat_weight | 0.573 | 1.17 | 1.26 | -0.31 .. +0.13 |
| angles_weight | 0.509 | 0.350 | 0.344 | -0.16 .. +0.17 |
| energy_weight | 0.305 | 0.227 | 0.191 | -0.29 .. +0.45 |
| aggression | 0.924 | 1.42 | 1.47 | -0.10 .. +0.05 |
| speed_weight | 0.623 | 0.542 | 0.531 | -0.42 .. +0.44 |
| collision_weight | 1.63 | 0.359 | 0.351 | -0.05 .. +0.09 |
| horizon (s) | 0.806 | 0.580 | 0.536 | -0.32 .. +0.29 |
| preferred_speed | 0.870 | 0.687 | 0.678 | -0.06 .. +0.11 |
| reply_temperature | 0.0857 | 0.287 | 0.307 | -0.14 .. +0.08 |
| caution | 1.66 | 7.85 | 8.29 | -0.22 .. +0.11 |
| temperature | 0.0223 | 0.0249 | 0.0240 | -0.45 .. +0.48 |
| stickiness | 0.0437 | 0.0258 | 0.0246 | -0.05 .. +0.08 |
| store_weight | 0.283 | 0.463 | 0.586 | -0.47 .. +0.42 |
| power_horizon (s) | 5.13 | 9.29 | 9.91 | -0.23 .. +0.12 |
| power_temperature | 0.105 | 0.0331 | 0.0312 | -0.08 .. +0.17 |
| power_stickiness | 0.200 | 0.202 | 0.141 | -0.57 .. +0.69 |
| disengaged_energy_weight | 0.358 | 0.212 | 0.201 | -0.06 .. +0.12 |
| fire_tolerance | 1.85 | 4.89 | 5.04 | -0.11 .. +0.06 |
| crowd ratios: angles / energy / aggression | 0.52 / 0.65 / 0.16 | 0.38 / 0.40 / 0.070 | 0.37 / 0.45 / 0.063 | |
| crowd ratios: speed / preferred speed / reply temperature | 2.0 / 1.44 / 6.1 | 1.98 / 2.43 / 5.28 | 1.90 / 2.60 / 5.62 | |
| store_crowd | 0.38 | 0.14 | 0.53 | -3.7 .. +2.6 (flat) |

- **Interior?** No. The peak sits at the data's edge in threat, collision, caution, stickiness
  (low), power horizon, power temperature, disengaged energy and fire tolerance at every size,
  and in a few more at one size (aggression, reply temperature, preferred speed, energy,
  angles, store). These are the step-1 plateau rows; the narrow 50-Elo region in them (e.g.
  collision -0.05..+0.09) is the data edge, not curvature. The steep rows (angles, energy,
  speed weight, horizon, temperature, preferred speed) are interior.
- **Curvature at the peak** (eigenvalues of the Hessian of the mean surface over the 25 free
  coordinates, Elo per box^2): -332, -192, -143, -117, -94, -81, -67, -55, -39, -29, -29, -22,
  then -12 to +1 (7 directions), and +10, +12, +21, +32, +44. Steep directions: energy +0.53 /
  angles -0.47 / speed +0.35; stickiness +0.65 / speed weight -0.54 / temperature -0.40; horizon
  -0.76 / store +0.37; preferred speed +0.70 / horizon -0.37 / preferred-speed crowd ratio
  +0.37; energy -0.47 / angles -0.45 / fire tolerance -0.42. Upward (a saddle in the fit):
  fire tolerance / power temperature; threat / disengaged energy / power stickiness; power
  horizon / power temperature / reply temperature; disengaged energy / power stickiness /
  caution: all plateau rows.
- **Couplings** (largest products, Elo per box^2): 1v1 angles x aggression -76, angles x speed
  +70, preferred speed x stickiness +50; 4v4 temperature x stickiness +67, angles x stickiness
  +47; 16v16 angles x energy +113, angles x stickiness +60, energy x speed -59, temperature x
  stickiness +57. As round 2: angles x energy at 16v16 and angles x aggression at 1v1 lead.
- **Far side per row** (step 1, measured): the horizon falls 230-300 at 0.24 s and 160-430 at
  +1.25 to +2 box units at 16v16; the speed weight 120-200 at +-1.25 and 330-470 at +2; the
  temperature 180-290 at +1.25 (0.061) and 390-820 at +2 (0.11); the angles weight 150-280 at -2
  and 260-550 at +2 (team sizes); the preferred speed 590 at -2 at 16v16. No far side within
  +-2 box units (or at the hard end) for collision (downward), temperature (downward), fire
  tolerance (upward), threat, power temperature and power horizon.
- **Robust centre shift**: the centre moved most in collision weight (1.63 -> 0.36), caution
  (1.66 -> 7.85), fire tolerance (1.85 -> 4.89), power horizon (5.1 -> 9.3 s), power
  temperature (0.105 -> 0.033), threat (0.57 -> 1.17), aggression (0.92 -> 1.42), reply
  temperature (0.086 -> 0.29), horizon (0.81 -> 0.58 s). The surface's predicted gain
  over the round-2 centre is +453 / +334 / +533; the ladder measures -8 / +102 / +194 (the
  surface extrapolates at the data edge).

### 3. The persona sampler

**Construction** (`surface3.py sampler`, `sampler-3.json`): N(round-3 centre, c Sigma) over the
25 free coordinates (21 rows less boundary, seek and field, held; 6 crowd ratios; store_crowd),
Sigma = V diag(1 / max(lambda, 10)) V^T with lambda the eigenvalues of -H of the mean surface
(1v1, 4v4, 16v16 averaged: one persona plays every size, and the yardstick is chosen on the
mean) at the centre, each direction's sd capped at 0.75 box units (the data's reach); c by
bisection for a predicted mean drop on the surface. Rejected: stickiness > 0.15, horizon <
0.35 s (step 1); 0.8 / 1.6 / 4.2% of draws at drops 25 / 50 / 100. The old ratio rule
(stickiness > 5 x temperature) was not used (section 1).

At the centre lambda = -44 .. +332 (12 of 25 under the floor of 10): c 7.4 / 11.5 / 21.8 for
predicted drops 25 / 50 / 100, with 13 / 13 / 16 directions at the 0.75 cap. Revised sampler
(`sampler2.py`): the cap at 0.35 box units: c 14.3 for 50 (22 directions capped; every row's sd
0.31-0.35: nearly isotropic), drop 100 not reachable under the cap.

**Measured** (personas against the round-3 centre, 128 / 128 / 64 games each, Elo SE about
35 / 35 / 50 per persona; `sampler-games-3.json`, `sampler2-games-3.json`,
`sampler2c-games-3.json`):

| set | personas | predicted drop | measured drop 1v1 / 4v4 / 16v16 (mean) | persona Elo sd | mean Elo q10 / q50 / q90 | > 100 below |
|---|---|---|---|---|---|---|
| Gaussian, cap 0.75, drop 25 | 16 | 25 | 38 / 82 / 192 (104) | 65 / 88 / 208 | -226 / -73 / +11 | 44% |
| Gaussian, cap 0.75, drop 50 | 32 | 50 | 17 / 80 / 141 (79) | 58 / 74 / 126 | -171 / -73 / +6 | 31% |
| Gaussian, cap 0.75, drop 100 | 16 | 100 | 20 / 115 / 173 (103) | 68 / 94 / 212 | -252 / -92 / +26 | 44% |
| Gaussian, cap 0.35, drop 50 | 24 | 50 | 18 / 65 / 147 (77) | 34 / 55 / 157 | -164 / -52 / -6 | 29% |
| the same, temperature reflected cold | 24 | - | 1 / 32 / 37 (23) | 35 / 53 / 92 | -56 / -13 / +20 | 4% |
| round 2's box x 1.55 (surface drop 50) | 32 | 50 | 1 / 50 / 73 (41) | 31 / 47 / 85 | -77 / -34 / -6 | 6% |
| along flat directions (league candidates) | 12 | 30 each | 1 / 16 / 11 (9) | 23 / 39 / 38 | -27 / -4 / +9 | 0% |

- The predicted drop is not what the Gaussian delivers: 79-104 measured for 25-100 predicted,
  hardly changing with c (the capped directions dominate at every c).
- Per persona the surface does not predict the drop: r 0.04 (all-rounds surface), 0.14 (a
  local surface fitted on round-3 games without the sampler personas; held-out deviance
  33%), `local-3.json`.
- What does: the temperature offset (r with the drop 0.42 / 0.59 / 0.64 at 1v1 / 4v4 / 16v16,
  over the 120 Gaussian and box personas); then disengaged energy (-0.31 / -0.34 at team
  sizes) and the speed weight (-0.32 at 4v4). The surface puts the temperature in a wide,
  symmetric region (-0.45..+0.48) where step 1 measured a plateau below and a cliff above.
- **Temperature reflected** (`reflect3.py`: the 18 of 24 draws above the centre's temperature
  mirrored below it, the other 6 unchanged): those 18 gain +71 Elo on average (mean over sizes);
  the set's drop falls 77 -> 23.

**Style spread** (sd across the personas of the style counters against the centre; 1v1 / 4v4
/ 16v16):

| set | store | speed | climb | dive | turning | shots per ship-minute |
|---|---|---|---|---|---|---|
| Gaussian cap 0.75, drop 50 | 0.196 / 0.145 / 0.125 | 0.098 / 0.063 / 0.073 | 0.097 / 0.101 / 0.095 | 0.071 / 0.127 / 0.203 | 0.032 / 0.027 / 0.034 | 5.8 / 26.5 / 58.4 |
| Gaussian cap 0.35, drop 50 | 0.022 / 0.143 / 0.129 | 0.054 / 0.053 / 0.058 | 0.077 / 0.096 / 0.056 | 0.036 / 0.119 / 0.191 | 0.016 / 0.019 / 0.025 | 3.6 / 19.7 / 32.3 |
| the same, temperature cold | 0.021 / 0.122 / 0.127 | 0.051 / 0.050 / 0.058 | 0.088 / 0.089 / 0.055 | 0.035 / 0.104 / 0.189 | 0.015 / 0.015 / 0.025 | 4.1 / 18.7 / 34.3 |
| round 2's box x 1.55 | 0.025 / 0.125 / 0.095 | 0.067 / 0.066 / 0.056 | 0.082 / 0.081 / 0.047 | 0.027 / 0.111 / 0.165 | 0.018 / 0.015 / 0.017 | 4.6 / 21.2 / 45.5 |

The centre's style: store 0.91 / 0.49 / 0.08, speed 0.83 / 0.82 / 0.94, climb 0.66 / 0.34 /
0.07, dive 0.07 / 0.28 / 0.68, shots 36 / 183 / 336 per ship-minute.

**Label blur and label spread** (`blur-3.json`, `blur-3b.json`, `blur-3c.json`; round 2's state
sets, which were collected around the round-2 centre: 12,012 / 21,494 / 21,999 ship-states;
the round-3 centre's label entropy there 1.56 / 1.12 / 0.92 nats; the round-2 and round-3
centres differ by a JS of 0.41 / 0.21 / 0.10, so these states are only near the new centre):

| set | blur (share of the centre's entropy) 1v1 / 4v4 / 16v16 | mean JS from the centre | mean pairwise JS |
|---|---|---|---|
| Gaussian cap 0.75, drop 25 | 24 / 29 / 22% | 0.12 / 0.10 / 0.05 | 0.22 / 0.18 / 0.08 |
| Gaussian cap 0.75, drop 50 | 24 / 42 / 34% | 0.11 / 0.15 / 0.08 | 0.22 / 0.24 / 0.13 |
| Gaussian cap 0.75, drop 100 | 29 / 47 / 42% | 0.15 / 0.20 / 0.12 | 0.26 / 0.29 / 0.17 |
| Gaussian cap 0.35, drop 50 | 15 / 31 / 23% | 0.07 / 0.11 / 0.06 | 0.17 / 0.19 / 0.09 |
| the same, temperature cold | 15 / 31 / 23% | 0.08 / 0.11 / 0.06 | 0.17 / 0.19 / 0.09 |
| round 2's box x 1.55 | 18 / 28 / 18% | 0.09 / 0.10 / 0.05 | 0.18 / 0.17 / 0.06 |

Round 2's blur at its box spread 1 was 13-19% (around its own centre on its own states).

### 4. League personas

Five picked from the Gaussian sampler's personas (the cap-0.75 sets and the flat-direction
candidates): measured mean Elo at least -25 of the centre and no size below -60, then
farthest-point in team style (store, speed, climb, dive, turning, two-circle share, hit range
at 4v4 and 16v16, standardised), starting from the farthest from the centre's style
(`league-3-picks.json`; `candidates/league3-1..5.json`). None of the flat-direction
candidates was picked: they barely moved team style.

| persona | source | 4v4 store / speed / climb / dive / turning | 16v16 store / speed / climb / dive / turning |
|---|---|---|---|
| robust-centre-3 | - | 0.44 / 0.82 / 0.30 / 0.32 / 0.51 | 0.07 / 0.94 / 0.07 / 0.67 / 0.45 |
| league3-1 | gauss100-00 | 0.49 / 0.84 / 0.30 / 0.22 / 0.45 | 0.11 / 0.98 / 0.14 / 0.52 / 0.39 |
| league3-2 | gauss50-13 | 0.34 / 0.86 / 0.23 / 0.33 / 0.49 | 0.06 / 1.01 / 0.05 / 0.56 / 0.39 |
| league3-3 | gauss50-27 | 0.52 / 0.74 / 0.30 / 0.19 / 0.47 | 0.09 / 0.95 / 0.08 / 0.53 / 0.40 |
| league3-4 | gauss100-14 | 0.49 / 0.80 / 0.26 / 0.21 / 0.47 | 0.15 / 0.95 / 0.18 / 0.52 / 0.41 |
| league3-5 | gauss50-31 | 0.49 / 0.80 / 0.30 / 0.29 / 0.49 | 0.10 / 0.93 / 0.06 / 0.85 / 0.43 |

(Ladder style, every opponent; their Elo in section 5.) Against the centre (sampler games):
store at 4v4 0.38-0.57, dive at 16v16 0.54-0.85. Labels: pairwise JS between the five 0.07-0.29
(1v1), 0.08-0.23 (4v4), 0.05-0.11 (16v16); each from the round-3 centre 0.07-0.22 / 0.08-0.18 /
0.03-0.11. Round 2's league: 0.02-0.11 between them, store 0.15-0.23 at 4v4.

### 5. Ladder

10 agents, every pair, adaptive to a score SE of 0.05 per pair (worst 0.037 reached): 8,192
games per size (`ladder-3.json`, store `ladder.sqlite`). Elo with the round-2 centre at 0 (the
SE is against it), and with the old yardstick at 1000:

| agent | 1v1 | 4v4 | 16v16 | old yardstick = 1000: 1v1 / 4v4 / 16v16 |
|---|---|---|---|---|
| robust-centre-3 | -8 +- 12 | +102 +- 12 | +194 +- 14 | 1140 / 1587 / 2407 |
| robust-centre-2 | 0 | 0 | 0 | 1148 / 1485 / 2213 |
| surface-peak-3 | -2 +- 12 | +76 +- 12 | +175 +- 14 | 1146 / 1561 / 2388 |
| league3-1 | +31 +- 12 | +114 +- 13 | +244 +- 14 | 1179 / 1599 / 2457 |
| league3-2 | -59 +- 12 | +69 +- 12 | +179 +- 14 | 1089 / 1554 / 2392 |
| league3-3 | -15 +- 12 | +52 +- 12 | +264 +- 14 | 1133 / 1537 / 2476 |
| league3-4 | -8 +- 12 | +49 +- 12 | +145 +- 14 | 1140 / 1534 / 2358 |
| league3-5 | +19 +- 12 | +101 +- 12 | +208 +- 14 | 1167 / 1586 / 2421 |
| crowd-known | -87 +- 12 | -167 +- 14 | -387 +- 24 | 1062 / 1318 / 1829 |
| old yardstick | -148 +- 12 | -487 +- 22 | -1569 +- 316 | 1000 |

- Direct games, round-3 centre against round-2 centre (184 per size): 87-7-90 at 1v1, 109-5-70
  at 4v4, 141-0-43 at 16v16.
- The old yardstick loses almost every 16v16 game against the top group: its 16v16 rating (SE
  316 against the round-2 centre) rests on crowd-known.
- Pool dependence: with the old yardstick at 1000, round 2's ladder put the round-2 centre at
  1278 at 1v1, this ladder at 1148 (crowd-known 1171 -> 1062); 4v4 matches (1482 / 1485). The
  1v1 ratings against the old yardstick depend on which agents are in the pool; comparisons
  among the top agents within one ladder do not need the yardstick.

## Recommendations (not measured; for the author)

- **Yardstick: the round-3 robust centre** (`candidates/robust-centre-3.json`): level with
  round 2's at 1v1, +102 at 4v4, +194 at 16v16, above the surface peak. league3-1 measured
  higher (+39 / +12 / +50 over it) but was one draw among 64 with 128 games per size; I would
  not promote a sampled point.
- **The summit is bounded where it matters and open where it does not.** The steep rows
  (angles / energy / speed weights, horizon, preferred speed, temperature upward) now have both
  sides measured. The plateau rows (collision, threat, fire tolerance, caution, the power rows,
  temperature downward) have no top: skill there is flat to the end tried. More range in them
  would move the "peak" further without finding one; the quadratic surface is the wrong model
  for them (it curves upward there). I would stop extending.
- **Sampler** (`proposal-3.json`): the curvature Gaussian with each direction's sd capped at
  0.35 box units, centred on the round-3 centre, the temperature drawn only on the cold side
  (reflected), stickiness <= 0.15 and horizon >= 0.35 s rejected, boundary / seek / field
  held. Measured: drop 23 (1 / 32 / 37), 4% of personas more than 100 down, blur 15 / 31 / 23%.
  Honest reading: with the surface's resolved curvature (five steep directions) and the cap,
  the result is nearly isotropic (every row's sd 0.31-0.35 box units); the curvature shaping
  barely changes it, and its predicted drop did not hold (77 measured for 50 predicted until
  the temperature fix). The c for 25 / 100 variants (5.4 / unreachable under the cap) does not
  control the drop in practice; I would scale the cap instead (rough: 0.25 for a smaller
  drop, 0.5 for a larger one, untested) and re-measure.
- **Same-plateau rule for other one-sided rows**: the collision weight (lower is never worse)
  and fire tolerance (higher is never worse to about 4 x the centre's) could be reflected the
  same way; untested.
- **BC**: blur at the proposed sampler is 15 / 31 / 23% of the centre's entropy on round 2's
  states (larger than round 2's 13-19% at spread 1; the round-3 centre's own labels are less
  peaked here, 1.56 against 1.04 nats at 1v1). If BC needs sharper labels, halve the cap or
  tell the student the persona.
- **League** (`league3-1..5`): within -51..+39 (1v1), -53..+12 (4v4), -49..+70 (16v16) of the
  round-3 centre; more team-style variety than round 2's set (store at 4v4 0.34-0.52 in the
  ladder against 0.15-0.23; dive at 16v16 0.52-0.85; label JS 0.05-0.29 against 0.02-0.11).
  league3-2 is the weakest at 1v1 (-51 below the centre).
- **Round 2's stall rule** (stickiness > 5 x temperature) should become stickiness > 0.15:
  round 2's stalls and round 3's normal agents both fit the absolute rule, not the ratio.
