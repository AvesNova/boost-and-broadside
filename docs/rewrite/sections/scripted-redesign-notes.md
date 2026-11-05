# Scripted redesign: notes

Brief: `scripted-redesign-brief.md`. Evidence: `scripted-softness-notes.md`.

## The scale floors

Each logistic's smallest scale is 0.1 times the standard deviation of the natural logarithm
of its input, measured over the earlier (linear-ramp) agent's self-play on the CPU, 16
games per preset, 1500 decisions per seat (the softness audit's recorder; inputs below
their 1% quantile read as it, so an exact 0 does not dominate). The smallest over the three
presets, at the combat layer's site:

| Logistic (input) | std of ln(input) by preset (frontline, deathmatch, deathmatch_wide) | Floor |
|---|---|---|
| turn, level two, level three (angle error / stall angle) | 1.58, 2.46, 2.73 | 0.158 |
| boost slow (speed / corner speed) | 0.33, 0.31, 0.35 | 0.031 |
| boost far (store x distance / gun range) | 2.15 (1.43 at the Frontline layer), 1.83, 2.43 | 0.143 |
| reverse, shoot (distance / gun range) | 1.27, 1.35, 1.94 | 0.127 |
| aim (angle error / target size) | 2.05, 2.18, 2.31 | 0.205 |

The logistics are in the logarithm of their input: the inputs are positive ratios with
heavy upper tails (the aim error's 95% is 40 to 119 target sizes against a median of 0.5
to 12), so a floor of 0.1 x the linear standard deviation would have been 1.9 target sizes
for the aim, wider than the whole useful aiming window.

## The floors per decision

The brief's single floor, ε = 0.04 on every binary decision, lost almost every game: the
search's first generation (32 candidates at the converted starting point and perturbations
of it) scored about 0 against the earlier raw agent, 16 games per preset each. An ablation
against the earlier raw agent at T = 1 (128 games per preset, 384 per row, standard error
about 0.02; the converted starting parameters):

| Floors on | Score |
|---|---|
| none | 0.380 |
| shoot 0.04 only | 0.025 |
| power (reverse, boost) 0.04 only | 0.313 |
| stick (turn, levels, side) 0.04 only | 0.314 |
| all 0.04 | 0.013 |

Then each decision's floor alone, the others at 0.005 (first block), and the shot's with
the stick at 0.02 and power at 0.04 (second block):

| Floors | Score |
|---|---|
| all 0.005 | 0.268 |
| shoot 0.01 | 0.180 |
| shoot 0.02 | 0.094 |
| reverse and boost 0.02 | 0.283 |
| reverse and boost 0.04 | 0.267 |
| stick 0.02 | 0.251 |
| stick 0.04 | 0.240 |
| stick 0.02, power 0.04, shoot 0 | 0.367 |
| stick 0.02, power 0.04, shoot 0.001 | 0.312 |
| stick 0.02, power 0.04, shoot 0.002 | 0.314 |

The shot carries the cost: a per-tick floor fires an energy-costly shot (and starts the
gun's cooldown) at 30 decisions a second, so even 0.001 costs about 0.05. Power at 0.04 and
the stick at 0.02 cost nothing measurable. The floors chosen (`EPSILONS`): turn, level
two, level three and side 0.02; reverse and boost 0.04; shoot 0 (the coordinator's call):
its softness is its two logistics', which are graded near their edges and, with every
logistic's argument held within ±9.2 (`LOGIT_LIMIT`), never reach exactly 0 or 1. The
shot's only exact value is P(shoot) = 0 at its gates: no target, or a gun cooling or short
of energy (where a shot does nothing).

## The search

`bench/scripted_search.py` from a scratch wrapper that replaced the defaults reference by
the earlier raw agent (frozen from 672364f9): 32 candidates, 128 games per candidate and
preset (a quarter against the references, 16 each), the three presets, candidate seats at
T drawn from {1, 1.5, 2, 3}, the top 8 kept, sigma 0.25 x 0.85 per generation. 17
generations, about 18 minutes of GPU time (60 s a generation, 12,288 games); the plateau
rule (patience 4, 10 points) stopped it.

| Generation | Top | Median | Last |
|---|---|---|---|
| 0 | -191 | -395 | -569 |
| 2 | -171 | -270 | -449 |
| 4 | -130 | -205 | -368 |
| 6 | -82 | -132 | -200 |
| 8 | -72 | -123 | -219 |
| 10 | -45 | -89 | -126 |
| 12 | -32 | -75 | -116 |
| 14 | -55 | -81 | -109 |
| 16 | -39 | -72 | -104 |

(Ratings against the earlier raw agent, 0, of candidates playing at the drawn
temperatures; the population's spread closed from 380 to 65 points.) Every candidate beat
the chaser throughout. Five finalists (kept at least twice, the best mean rating over
their generations) replayed at T = 1 against the earlier raw agent, 384 games each: 0.543,
0.518, 0.488, 0.531, 0.504 (±0.024), within noise of each other; the first became the
defaults. What moved most: the shoot logistic's centre in from 0.82 to 0.51 gun ranges
(its scale at the floor), the aim's centre from 0.60 to 0.41 target sizes, boost-far out of
play (centre 10 store-gun-ranges), the levels' centres down (level three 1.47 to 0.73 stall
angles), and in Frontline more aggression (0.63 to 1.81) and urgency (1.82 to 3.87) with a
lower defensive floor (0.81 to 0.30).

## Acceptance

128 games per preset (384 per row), fresh seeds above the search's, the seat drawn per
game; score from the first player's side, ± its standard error.

| First | Second | Total | frontline | deathmatch | deathmatch_wide |
|---|---|---|---|---|---|
| new, T = 1 | earlier raw agent | 0.483 ± 0.025 | 0.574 | 0.336 | 0.539 |
| new, T = 1 | earlier teacher (`scripted@floor=0.01`) | 0.630 ± 0.024 | 0.746 | 0.461 | 0.684 |
| new, T = 1 | chaser | 1.000 | 1.000 | 1.000 | 1.000 |
| new, T = 1.5 | new, T = 1 | 0.477 ± 0.024 | 0.508 | 0.504 | 0.418 |
| new, T = 2 | new, T = 1 | 0.408 ± 0.024 | 0.402 | 0.461 | 0.359 |
| new, T = 3 | new, T = 1 | 0.172 ± 0.018 | 0.195 | 0.195 | 0.125 |
| new, T = 4 | new, T = 1 | 0.034 ± 0.008 | 0.039 | 0.031 | 0.031 |
| new, T = 6 | new, T = 1 | 0.007 ± 0.003 | 0.016 | 0.000 | 0.004 |
| new, T = 8 | new, T = 1 | 0.007 ± 0.003 | 0.020 | 0.000 | 0.000 |
| new, T = 16 | new, T = 1 | 0.007 ± 0.003 | 0.020 | 0.000 | 0.000 |
| new, T = 1, 1% uniform floor | new, T = 1 | 0.406 ± 0.024 | 0.316 | 0.465 | 0.438 |
| new, T = 8 | random | 1.000 | 1.000 | 1.000 | 1.000 |
| new, T = 16 | random | 1.000 | 1.000 | 1.000 | 1.000 |
| new, T = 32 | random | 0.999 | 1.000 | 1.000 | 0.996 |

- **Good enough:** the new agent beats the earlier teacher (0.630, where that teacher
  scored 0.398 against the earlier raw agent) and is level with the earlier raw agent
  (0.483 ± 0.025; it loses deathmatch, 0.336, and wins Frontline).
- **The temperature curve** is flat to T = 1.5, falls through T = 2 and 3, and is
  saturated by T = 4.
- **A 1% uniform floor costs 0.094**, about what it cost the earlier agent (0.10): a
  uniform action fires half the time, the shot's cost again.
- **The agent at any temperature is far from random:** its floors and structure remain as
  T grows (every logistic tends to 1/2; the gates and the floors stay), so even T = 32 beats
  random play in 383.5 of 384 games. The ladder (`LADDER`) is therefore random, then
  T = 8, 4, 3, 2.5, 2, 1.5 and the agent: the temperature rungs span the agent's side of the
  curve, and the step from random to T = 8 is one the temperature cannot bridge.

## Softness audit

The earlier audit's method (`/tmp/bnb-scratch-scripted-softness/audit.py`, rewritten for
the logistics as `/tmp/bnb-scratch-scripted-redesign/audit_new.py`): scripted self-play on
the CPU, 16 games per preset, 3,000 decisions per seat, 427,000 to 445,000 living ship
steps per preset.

Share of living ship steps whose chosen value has probability exactly 1, and the mean
entropy (nats per living ship step) at T = 1, 2 and 4 on the same observations:

| Preset | Factor | = 1 | Entropy T = 1 | T = 2 | T = 4 |
|---|---|---|---|---|---|
| frontline | power | 0 | 0.406 | 0.529 | 0.696 |
| frontline | stick | 0 | 0.497 | 0.622 | 0.812 |
| frontline | shoot | 0.728 | 0.013 | 0.028 | 0.060 |
| deathmatch | power | 0 | 0.384 | 0.471 | 0.641 |
| deathmatch | stick | 0 | 0.444 | 0.579 | 0.762 |
| deathmatch | shoot | 0.333 | 0.016 | 0.036 | 0.090 |
| deathmatch_wide | power | 0 | 0.433 | 0.528 | 0.654 |
| deathmatch_wide | stick | 0 | 0.398 | 0.505 | 0.669 |
| deathmatch_wide | shoot | 0.473 | 0.013 | 0.032 | 0.077 |

Joint entropy (the factors are independent), T = 1 / 2 / 4: frontline 0.92 / 1.18 / 1.57,
deathmatch 0.84 / 1.09 / 1.49, deathmatch_wide 0.84 / 1.06 / 1.40 (the earlier agent: 0.35,
0.31, 0.27 at its only temperature). Power and stick are never certain; the shot is
certain (not shooting) exactly at its gates, 33 to 73% of living ship steps, mostly the gun
cooling between shots, where a shot does nothing.

Each logistic's 10–90% transition (`width`, from `centre exp(-2.2 scale)` to
`centre exp(2.2 scale)`, in the input's unit) against its input over the same games, and
the share of observations inside it (the combat layer's site):

| Logistic | width | width / std (fl, dm, dmw) | 4.39 scale / std of ln input | inside (fl, dm, dmw) |
|---|---|---|---|---|
| turn | 0.058 stall angles | 0.017, 0.019, 0.017 | 0.49 to 0.76 | 0.06, 0.35, 0.29 |
| level two | 0.131 | 0.039, 0.043, 0.037 | 0.27 to 0.42 | 0.05, 0.04, 0.03 |
| level three | 0.565 | 0.17, 0.19, 0.16 | 0.27 to 0.42 | 0.10, 0.02, 0.03 |
| boost slow | 0.081 corner speeds | 0.34, 0.35, 0.29 | 0.45 to 0.55 | 0.12, 0.08, 0.12 |
| boost far | 7.3 store gun ranges | 4.7, 4.0, 2.4 | 0.27 to 0.38 | 0.00, 0.00, 0.09 |
| reverse | 0.218 gun ranges | 0.07, 0.05, 0.02 | 0.31 to 0.46 | 0.08, 0.08, 0.12 |
| aim | 0.447 target sizes | 0.012, 0.022, 0.025 | 0.45 to 0.47 | 0.04, 0.24, 0.18 |
| shoot | 0.290 gun ranges | 0.09, 0.07, 0.02 | 0.29 to 0.43 | 0.10, 0.10, 0.15 |

Against the earlier agent's linear ramps, the shot, reverse and boost-slow transitions are
wider (shoot 0.29 against 0.10 gun ranges, reverse 0.22 against 0.14, boost slow 0.081
against 0.061 corner speeds); the turn, the levels and the aim are narrower in their input's
unit (turn 0.058 against 0.14 stall angles, level two 0.13 against 0.25, level three 0.57
against 1.05, aim 0.45 against 0.76 target sizes), because the search moved their centres
down and a log-input logistic's width scales with its centre. None ends at exactly 0 or 1:
outside its transition a logistic keeps grading (a factor of 10 in the odds per 2.3 scales)
until the clamp at ±9.2. By the earlier audit's reading (observations inside the
transition) most stay narrow against their inputs' spread: the search took the scales it was
allowed, several at or near their floors (shoot at its floor 0.127, reverse 0.135, boost
slow 0.0345 against 0.031). The floors (0.1 x the std of ln input) are the lever if a softer
teacher is wanted.
