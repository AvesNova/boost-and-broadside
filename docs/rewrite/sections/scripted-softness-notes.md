# Scripted softness: notes

Brief: `scripted-softness-brief.md`. Part 1 landed as `bnb head-to-head` players
`scripted@T=<t>,floor=<f>` (`bnb.league.matches.teacher_of`; drawn by
`bnb.learn.bc.TeacherOpponent`; keys optional, T=1 and floor=0 by default;
`scripted@floor=0.01` is BC's teacher, `TeacherConfig()`).

## Part 2: scores against the raw scripted agent

128 games per preset (384 per row), the three default presets, split evenly between seats.
Score = wins + half draws, over games; standard error from the per-game variance. Each row
is "the strength a perfect imitator of that distribution would have".

| Player | Total | frontline | deathmatch | deathmatch_wide |
|---|---|---|---|---|
| `scripted@T=1` (floor 0; the same code path as scripted: sanity) | 0.509 ± 0.024 | 0.492 | 0.516 | 0.520 |
| BC's teacher `scripted@floor=0.01` (T=1) | 0.398 ± 0.023 | 0.316 | 0.402 | 0.477 |
| `scripted@T=0.5` | 0.539 ± 0.024 | 0.648 | 0.480 | 0.488 |
| `scripted@T=2` | 0.470 ± 0.024 | 0.359 | 0.523 | 0.527 |
| `scripted@T=4` | 0.414 ± 0.023 | 0.324 | 0.449 | 0.469 |
| `scripted@T=1,floor=0.05` | 0.061 ± 0.012 | 0.023 | 0.016 | 0.145 |
| `rung_0.99` | 0.357 ± 0.023 | 0.332 | 0.344 | 0.395 |
| `rung_0.95` | 0.035 ± 0.008 | 0.039 | 0.008 | 0.059 |

Per-preset standard errors are about 0.04 to 0.05. The floor 0.01 teacher is the T=1 row of
`TeacherConfig()`; "floor 0.01 at T=1" is therefore the same row.

The clones (50M `2026-11-05-mixed-sunny-ember`, 100M `2026-11-04-mixed-wandering-quill`):

| Clone | against | Total | frontline | deathmatch | deathmatch_wide |
|---|---|---|---|---|---|
| sunny-ember | scripted | 0.177 ± 0.018 | 0.195 | 0.254 | 0.082 |
| wandering-quill | scripted | 0.271 ± 0.021 | 0.332 | 0.348 | 0.133 |
| sunny-ember | teacher | 0.264 ± 0.021 | 0.344 | 0.348 | 0.102 |
| wandering-quill | teacher | 0.367 ± 0.023 | 0.441 | 0.484 | 0.176 |

(The brief's 0.190 and 0.250 are other draws of the same match; the differences are within
about 1 standard error.)

## Part 3: softness audit

Scripted self-play on the CPU (the audit reads tensors on the host), 16 games per preset,
3,000 decisions per seat: about 430,000 to 440,000 living ship steps per preset.
`/tmp/bnb-scratch-scripted-softness/audit.py` (a `Scripted` subclass that records the
marginals and, by wrapping `ramp`, every ramp's input).

### Probability of the chosen value, and entropy

Share of living ship steps by the probability of the chosen factor value (exactly 1; above
0.99 and below 1; 0.99 or less), and the mean entropy in nats per living ship step. A chosen
value never has probability 0.

| Preset | Factor | = 1 | (0.99, 1) | <= 0.99 | Entropy |
|---|---|---|---|---|---|
| frontline | power | 0.602 | 0.029 | 0.369 | 0.114 |
| frontline | stick | 0.538 | 0.014 | 0.448 | 0.231 |
| frontline | shoot | 0.990 | 0.000 | 0.009 | 0.005 |
| deathmatch | power | 0.740 | 0.015 | 0.245 | 0.092 |
| deathmatch | stick | 0.511 | 0.016 | 0.473 | 0.214 |
| deathmatch | shoot | 0.990 | 0.000 | 0.010 | 0.005 |
| deathmatch_wide | power | 0.732 | 0.011 | 0.257 | 0.100 |
| deathmatch_wide | stick | 0.624 | 0.013 | 0.363 | 0.164 |
| deathmatch_wide | shoot | 0.988 | 0.000 | 0.012 | 0.006 |

Joint entropy (the factors are independent): 0.35, 0.31 and 0.27 nats per ship step.
A uniform floor of 0.01 adds about 0.07 nats of its own. A ship's shot decision is
deterministic about 99% of the time. The shoot factor is at 0 or 1 whenever the ship is
not in the narrow range-and-aim window; where in range the aim ramp is a 0.7-target-size
window.

### Ramps

Each ramp's input as observed in the same games (living ships; the conditions under which
the ramp is read: with a target for the target ramps, flying outward for the boundary).
`width` is `high - low`. `site`: the combat layer, or the Frontline layer's own call (the
Frontline preset only). Quantiles are 5%, 50%, 95%. `below` / `inside` / `above`: the share
of observations at or below `low`, strictly inside, and at or above `high`.

| Ramp (unit) | Preset (site) | end values | width | std of input | width / std | 5% / 50% / 95% | below / inside / above |
|---|---|---|---|---|---|---|---|
| shoot range (gun ranges) | frontline | 1 → 0 | 0.103 | 2.7 | 0.04 | 0.17 / 2.7 / 8.5 | 0.28 / 0.02 / 0.69 |
| shoot range | deathmatch | 1 → 0 | 0.103 | 4.5 | 0.02 | 0.20 / 3.6 / 12.8 | 0.22 / 0.02 / 0.76 |
| shoot range | deathmatch_wide | 1 → 0 | 0.103 | 8.3 | 0.01 | 0.14 / 1.8 / 42.2 | 0.36 / 0.03 / 0.61 |
| reverse (gun ranges) | frontline | 1 → 0 | 0.144 | 2.7 | 0.05 | same input | 0.14 / 0.06 / 0.81 |
| reverse | deathmatch | 1 → 0 | 0.144 | 4.5 | 0.03 | same input | 0.10 / 0.04 / 0.86 |
| reverse | deathmatch_wide | 1 → 0 | 0.144 | 8.3 | 0.01 | same input | 0.16 / 0.07 / 0.77 |
| aim (angle / target size) | frontline | 1 → 0 | 0.755 | 38 | 0.02 | 0.11 / 11.9 / 119 | 0.10 / 0.05 / 0.85 |
| aim | deathmatch | 1 → 0 | 0.755 | 19 | 0.04 | 0.03 / 0.50 / 40 | 0.40 / 0.27 / 0.33 |
| aim | deathmatch_wide | 1 → 0 | 0.755 | 20 | 0.04 | 0.03 / 0.74 / 38.5 | 0.35 / 0.21 / 0.44 |
| turn (stall angles) | frontline, combat | 0 → 1 | 0.142 | 3.7 | 0.04 | 0.06 / 2.7 / 10.6 | 0.02 / 0.10 / 0.88 |
| turn | frontline, Frontline | 0 → 1 | 0.142 | 3.8 | 0.04 | 0.00 / 0.12 / 9.8 | 0.23 / 0.29 / 0.49 |
| turn | deathmatch | 0 → 1 | 0.142 | 2.8 | 0.05 | 0.00 / 0.05 / 9.2 | 0.23 / 0.46 / 0.31 |
| turn | deathmatch_wide | 0 → 1 | 0.142 | 3.4 | 0.04 | 0.01 / 0.07 / 9.8 | 0.19 / 0.39 / 0.43 |
| level two (stall angles) | frontline, combat | 0 → 1 | 0.247 | 3.7 | 0.07 | as turn | 0.11 / 0.08 / 0.81 |
| level two | frontline, Frontline | 0 → 1 | 0.247 | 3.8 | 0.07 | as turn | 0.51 / 0.03 / 0.46 |
| level two | deathmatch | 0 → 1 | 0.247 | 2.8 | 0.08 | as turn | 0.69 / 0.03 / 0.28 |
| level two | deathmatch_wide | 0 → 1 | 0.247 | 3.4 | 0.07 | as turn | 0.57 / 0.02 / 0.41 |
| level three (stall angles) | frontline, combat | 0 → 1 | 1.05 | 3.5 | 0.30 | as turn | 0.30 / 0.14 / 0.56 |
| level three | frontline, Frontline | 0 → 1 | 1.05 | 3.4 | 0.31 | as turn | 0.59 / 0.06 / 0.35 |
| level three | deathmatch | 0 → 1 | 1.05 | 3.0 | 0.35 | as turn | 0.75 / 0.04 / 0.21 |
| level three | deathmatch_wide | 0 → 1 | 1.05 | 3.4 | 0.31 | as turn | 0.63 / 0.05 / 0.33 |
| boost slow (corner speeds) | frontline | 1 → 0 | 0.061 | 0.28 | 0.22 | 0.49 / 0.92 / 1.28 | 0.04 / 0.09 / 0.87 |
| boost slow | deathmatch | 1 → 0 | 0.061 | 0.25 | 0.24 | 0.51 / 1.01 / 1.26 | 0.04 / 0.05 / 0.92 |
| boost slow | deathmatch_wide | 1 → 0 | 0.061 | 0.30 | 0.20 | 0.45 / 0.93 / 1.33 | 0.06 / 0.09 / 0.85 |
| boost far (store x gun ranges) | frontline, combat | 0 → 1 | 2.0 | 1.4 | 1.46 | 0.00 / 0.87 / 4.2 | 0.87 / 0.11 / 0.02 |
| boost far | frontline, Frontline | 0 → 1 | 2.0 | 1.4 | 1.40 | 0.03 / 1.7 / 3.5 | 0.68 / 0.31 / 0.02 |
| boost far | deathmatch | 0 → 1 | 2.0 | 1.3 | 1.49 | 0.01 / 1.7 / 3.6 | 0.75 / 0.24 / 0.01 |
| boost far | deathmatch_wide | 0 → 1 | 2.0 | 1.9 | 1.08 | 0.00 / 0.49 / 3.3 | 0.78 / 0.20 / 0.02 |
| boundary steer (margins deep) | all | 0 → 1 | 0.466 | 40 to 95 | 0.01 | median about -20 to -56 | about 1.00 / 0.00 / 0.00 |

(The std of boundary depth and the "inside" row are over ships flying outward anywhere in the
arena, so almost every observation is far from the margin; the ramp is read only near the
edge. Its share inside the ramp is under 0.5% of outward-flying ship steps.)

The reverse and shoot ramps read the same input (the target's distance in gun ranges); the
stick ramps read the angle error (for the combat layer, to the aim point) in stall angles.

### Step-like ramps

A ramp acts as a step function when it ends at exactly 0 and 1 and its width is a small
fraction of its input's spread, so that few observations land inside it. By that reading:

- **Step-like:** shoot range (width 0.10 gun ranges; 2 to 3% of observations inside),
  reverse (0.14; 4 to 7% inside), aim (0.76 target sizes against an input whose median is
  0.5 to 12 target sizes and whose 95% is 40 to 119; 5 to 27% inside), turn (0.14 stall
  angles; 10% inside in combat, up to 46% in the Frontline layer and deathmatch, where the
  input is near zero), level two (0.25 stall angles; 2 to 8% inside), and boost slow (0.06
  corner speeds; 5 to 9% inside).
- **Not step-like:** boost far (2 store-gun-ranges wide against a std of 1.4; 11 to 31%
  inside), and level three (1.05 stall angles, 4 to 14% inside; wide but also ends at 0
  and 1).
- **Rarely read:** boundary steer, over a narrow depth range near the arena's edge.

All ramps end at exactly 0 and 1; v1's ended at 0.04 to 0.95 and were 200 to 500 units
wide for shooting by distance. Here the shoot-by-distance ramp spans 0.10 gun ranges.
