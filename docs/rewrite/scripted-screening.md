# Screening of the scripted agent's parameters (2026-11-24)

The aim (core-decisions.md "Evaluation runner and scripted search"): understand the space of
the scripted agent's 34 sampled rows (lean/scripted/parameters.py `SAMPLED`, each on its own
box), favouring a robust region over a sharp peak. Scripts, panel and per-configuration
results: docs/rewrite/sections/scripted-screening/ (`screen.py`, `analyse.py`, `panel.json`,
`skills.json`, `analysis.json`). All games are kept in one lean.eval store
(~/avesnova/boost-and-broadside/data/scripted-screening/screen.sqlite, 1.56M games), and
every step resumes from it.

## Findings

1. The yardstick scores 0.609 +- 0.007 on the measure below. That is better than 95.8% of the
   4,000 configurations drawn from the full boxes, whose median is 0.448. The design-value
   BFM persona scores 0.332 +- 0.007 (10th percentile).
2. Nine rows carry almost all of the effect. By total Sobol index of the surrogate:
   lead_turn_scale 0.58, lead_gain 0.14, yo_yo_time 0.12, fight_speed 0.05, closing_time
   0.05, plan_threshold 0.04, in_range 0.03, nose_tail 0.03, aligned 0.01. Morris agrees on
   the set.
3. Two of the three largest effects are features that hurt whenever they are on. Measured through the
   yardstick, any lead turn and any yo-yo time lowers skill, step by step: lead_turn_scale
   0.25 gives 0.556, and yo_yo_time 0.125 gives 0.540. The yardstick has both at 0, the bottom
   of their boxes.
4. 22 rows have no measurable effect in their boxes: total index at most 0.001 and mean
   |SHAP| at most 0.002 score. They are the candidates to fix or delete (below).
5. The defaults sit on a broad plateau, not at a cliff: 256 configurations at a quarter of
   the box spread score a mean of 0.607, the yardstick's own skill. The yardstick is not at
   the peak, though. It is on a slope for lead_gain (0.9 gives 0.533, 1.1 gives 0.661) and
   for fight_speed (skill still rising at the top of its box).
6. A configuration's ranking carries over across arena radius (3,000 to 6,000 m), team size
   (3v3 to 5v5) and spawn rule. Corrected for noise, the correlations are 0.97 to 1.0. The
   skill difference between 6,000 m and 3,000 m arenas is noise: its spread is 0.071 against
   0.070 expected from noise alone, and a surrogate of it has a cross-validated R^2 of 0.02.
   No retuning for these changes is indicated.

## Skill measure

The panel has 8 members: the yardstick (the current defaults); the tests' design-value BFM
persona (lean/scripted/tests/test_tactics.py `BFM`: yo_yo_time 0.5, lead_turn_scale 1.0,
relax_angle 1.5 pi, defend_range 1.0); and 6 personas drawn once from the full boxes (seed
20261124). Their vectors are in `panel.json`. The pre-BFM agent is no longer in the code.

There are 18 conditions: arena radius 3,000, 4,200 or 6,000 m; 3v3, 4v4 or 5v5; mixed or
sides spawn. Fog and fields stay at the defaults (10 fields), with the standard spawn and the
5,400-tick clock.

The skill of a configuration is its mean score (wins + draws / 2) over the 144 (panel
member, condition) cells, each cell weighted equally. Its standard error comes from each
cell's unbiased game variance.

| run | games per config | median error |
|---|---|---|
| Sobol, local | 288 (2 a cell) | 0.028 |
| Morris | 576 (4 a cell) | 0.020 |
| sweep | 1,152 (8 a cell) | 0.014 |
| references | 5,184 (32+ a cell) | 0.007 |

The yardstick plays itself as one panel member, scoring 0.517 there over 576 games.

## Morris screening

SALib Morris: 10 trajectories, 4 levels, 350 configurations, every sampled row on its box in
unit-cube coordinates (log rows in log space). mu* and sigma are in score per whole box
width. Noise alone gives mu* about 0.034 (an elementary effect's noise is sqrt(2) x 0.020 /
(2/3), and E|N| = 0.8 of that) and sigma about 0.043.

| row | mu* | sigma |
|---|---|---|
| lead_gain | 0.159 | 0.085 |
| lead_turn_scale | 0.108 | 0.097 |
| yo_yo_time | 0.102 | 0.095 |
| fight_speed | 0.087 | 0.034 |
| plan_threshold | 0.079 | 0.092 |
| closing_time | 0.077 | 0.061 |
| in_range | 0.066 | 0.084 |
| temperature | 0.055 | 0.067 |
| nose_tail | 0.054 | 0.052 |
| three_nine_line | 0.049 | 0.062 |

The other 24 rows have mu* between 0.022 and 0.045, close to the noise floor, with sigma at
the noise level. plan_threshold, in_range and lead_turn_scale show sigma at or above mu*,
which means interaction or non-linearity. fight_speed is close to linear (sigma 0.034).

## Sobol-sequence sample and surrogate

4,000 configurations come from the store's scrambled Sobol stream `screen` (seed 0) on the
full boxes. Skill: mean 0.451, sd 0.092; quantiles 5% 0.305, 50% 0.448, 95% 0.604, 99%
0.656; highest 0.747 (one noisy draw). 168 configurations score above the yardstick's 0.609
and 48 above 0.65, each with an error of about 0.028.

The surrogate is a gradient-boosted tree model (scikit-learn; 800 trees, depth 4, rate
0.025, settings picked by 5-fold CV on the training 80%).

- **Held-out R^2: 0.862** on 800 configurations. The noise ceiling is 0.904 (1 minus the
  noise variance over the skills' variance), so the model explains about 95% of the
  explainable variance. The first-order indices sum to 0.935: the space is mostly additive.
- Its prediction at the yardstick's point is 0.566, against 0.609 measured. The yardstick
  sits at the edge of the sample (three upward rows at 0), where the trees are coarse.

Sobol indices come from SALib (Saltelli, 8,192 base points) on the surrogate; SHAP is the
mean |SHAP value| (TreeExplainer) over the 4,000 configurations.

| row | S1 / ST | mean abs SHAP |
|---|---|---|
| lead_turn_scale | 0.544 / 0.581 | 0.053 |
| lead_gain | 0.126 / 0.144 | 0.026 |
| yo_yo_time | 0.100 / 0.115 | 0.023 |
| fight_speed | 0.045 / 0.047 | 0.015 |
| closing_time | 0.044 / 0.046 | 0.015 |
| plan_threshold | 0.004 / 0.039 | 0.007 |
| in_range | 0.026 / 0.030 | 0.011 |
| nose_tail | 0.022 / 0.025 | 0.011 |
| aligned | 0.012 / 0.014 | 0.008 |
| three_nine_line | 0.003 / 0.008 | 0.004 |
| temperature | 0.003 / 0.004 | 0.004 |
| level_two | 0.003 / 0.003 | 0.004 |

Every other row has ST at most 0.001.

Interacting pairs, by mean |SHAP interaction| (score):

| pair | interaction |
|---|---|
| plan_threshold x lead_turn_scale | 0.0097 |
| plan_threshold x lead_gain | 0.0066 |
| lead_turn_scale x yo_yo_time | 0.0043 |
| three_nine_line x lead_turn_scale | 0.0033 |
| lead_turn_scale x lead_gain | 0.0032 |
| yo_yo_time x in_range | 0.0031 |
| closing_time x yo_yo_time | 0.0027 |

plan_threshold acts almost only through interactions (S1 0.004, ST 0.039). Its effect
depends on the lead turn and the gun's lead. All interactions are small next to the main
effects.

## Robustness: where the yardstick sits

Measured one row at a time through the yardstick (the others at their defaults), at 0,
1/4, 1/2, 3/4 and the whole of each box. The default point is the yardstick: 0.609 +- 0.007;
every other point is +- 0.014 or 0.015.

| row: value | skill |
|---|---|
| lead_gain 0.8 / 0.9 | 0.444 / 0.533 |
| lead_gain 1.1 / 1.2 | 0.661 / 0.657 |
| fight_speed 0.82 / 0.90 | 0.551 / 0.596 |
| fight_speed 1.11 / 1.22 | 0.622 / 0.649 |
| lead_turn_scale 0.25 / 1 | 0.556 / 0.434 |
| yo_yo_time 0.125 / 0.5 | 0.540 / 0.426 |
| plan_threshold -0.25 / 0.25 | 0.633 / 0.553 |
| aligned 0.68 / 0.83 | 0.552 / 0.622 |
| in_range 0.70 / 1.55 | 0.549 / 0.594 |
| closing_time 0.67 / 1.49 | 0.599 / 0.589 |
| three_nine_line 1.37 / 1.77 | 0.595 / 0.632 |
| nose_tail 0.45 / 1.0 | 0.604 / 0.615 |
| temperature 0.67 / 1.49 | 0.609 / 0.582 |

(Full sweeps in `analysis.json`, "sweep".)

- **Local spread.** 256 configurations drawn at 0.25 of the box spread around the yardstick
  (stream `local25`) score a mean of 0.607, sd 0.041. Noise accounts for 0.028 of that sd,
  leaving about 0.03 of real spread; half score above 0.609, and the 5% quantile is 0.538.
  At 0.5 of the spread (`local50`) the mean falls to 0.551. That drop follows the upward
  rows: the correlation with lead_turn_scale is -0.58.
- **Verdict.** Every row except three sits in a broad, flat region: within the noise across
  the quarter-spread box, and gentle across the whole box. The three exceptions:
  - lead_turn_scale and yo_yo_time are best at 0. That is their floor, so the "cliff" is only
    in the direction of turning them on.
  - lead_gain falls steeply below 1 (-0.076 at 0.9) and gains above it (+0.05 at 1.1 to 1.2).
    The default is on a slope, close to the edge where skill drops.
  - fight_speed still rises at the top of its box (+0.04 at 1.22).
  No row is near a cliff in the direction a small error would take it, except lead_gain
  downward.

## Conditions

Mean skill of the 4,000 configurations by level: radius 3,000 / 4,200 / 6,000 m, 0.455 /
0.449 / 0.447; 3v3 / 4v4 / 5v5, 0.457 / 0.451 / 0.443; mixed / sides, 0.454 / 0.447.

Per-configuration differences between levels are near the noise. Their spread against the
spread expected from noise alone:

| difference | spread | noise |
|---|---|---|
| 6,000 m minus 3,000 m | 0.0713 | 0.0696 |
| 5v5 minus 3v3 | 0.0715 | 0.0696 |
| sides minus mixed | 0.0614 | 0.0568 |

These leave real spreads of about 0.015, 0.016 and 0.023. The yardstick by level: radius
0.585 / 0.624 / 0.619, sizes 0.586 / 0.599 / 0.643, spawn 0.603 / 0.615.

## Style measures

Only game length is recorded. It is free: the runner returns it and the store keeps it. The
mean over the 4,000 is 1,416 ticks (sd 70 across configurations), and it correlates with
skill at -0.46: stronger configurations end games sooner. Store use, mean speed and firing
share are not recorded. The runner returns only the result and the length per game, and
collecting them would mean per-tick accumulators in the compiled tick in lean/eval/runner.py
(which is also part of a network's identity hash). Skipped.

## Recommendations

- **Keep (search with CMA-ES):** lead_gain, fight_speed, closing_time, in_range, nose_tail,
  aligned, plan_threshold, three_nine_line, temperature. Widen lead_gain's and fight_speed's
  boxes upward: their best measured points are at or near the top of their boxes. A robust
  search should keep lead_gain away from below 1.
- **Delete as features (both at 0; every value above 0 measured worse):**
  lead_turn_scale (the lead turn; 26 operations per call, morning-notes "E2b operations") and
  yo_yo_time (the yo-yo's lag and lead). This is the one case where "no effect" is not the
  reason: the effect is large and always negative.
- **Fix at the default and stop sampling (no measurable effect in the box):** weakest_weight,
  turning_floor, engage_range, defend_range, cash_angle, relax_angle, scissors_range,
  switch_gain, overshoot_range, overshoot_rate, overshoot_crossing, yo_yo_scale,
  merge_speed, speed_floor, spend_share, spend_need, spend_range, speed_band, power_scale,
  turn, level_three, boundary_margin. level_two is borderline (ST 0.003).
  Behaviour (scripted-divergence.md) narrows this: engage_range changes behaviour clearly and
  five more a little; only 16 of the 22 are flat in both.
- **Delete the logic too?** Being flat in its box shows that a threshold does not matter
  between its bounds. It does not show that the mechanism is useless. Whole mechanisms whose
  every row is flat are the candidates to remove from the 439-operation call: the overshoot
  rule (3 rows), the store-spending gates (3 rows), weakest-target choice, the plan switch
  on gain, the scissors slow-down, the two-circle cash and the break relaxation. Each needs
  one ablation game set (the yardstick with the mechanism removed, against the same panel)
  before deletion. That was not measured here.
- Note that merge_speed and speed_floor have narrow boxes (width 0.1 in log units), so their
  flatness says little outside them.

## Method notes

- The runner, store and Evaluator are lean/eval's. Each call plays one (radius, spawn)
  config for a batch of configurations against the whole panel at the three team sizes, with
  sides alternating in each (configuration, member, size). The store is written after each
  call: up to 96,000 games, about 3 minutes.
- Two host-time fixes were made on the way (committed with this study):
  - Evaluator._play_group was quadratic in the requests: a list-membership search per agent
    and a whole-queue mask per entry. 24,000 requests took 19 s of host time; it now uses a
    dict and one sort.
  - The store waited on the disk at each commit (3.5 ms each on the laptop's NVMe) and now
    uses a write-ahead log with normal sync (0.03 ms).
  Together these took a 96,000-game call from 261-321 s down to 163-209 s.
- GPU unit time, laptop RTX 4070, performance profile, one job at a time:
  - references 1.5 min
  - Morris 8.5 min (210,000 games incl. references, 320-450 games/s at E 2,100)
  - Sobol 4,000 about 44 min (1.15M games, of which about 5 min was host time before the
    fixes and one partial call lost to a restart)
  - local spreads 7.5 min
  - sweep 3.4 min
  - total about 64 minutes.
- Reproduce: `screen.py STORE panel | reference --games 32 | morris --games 4 | sobol --count
  4000 --batch 2000 --games 2 | local --spread 0.25 --count 256 | local --spread 0.5 --count
  256 | sweep --games 8 --rows ... | skills`, then `analyse.py`. Both run with the
  repository root on PYTHONPATH and the tools extra (SALib, scikit-learn, shap) installed.

## CMA-ES search (2026-11-24 morning)

A short search for a better yardstick (`search.py`; record `search.jsonl`; result
`cma_candidate.json`). The code's defaults are unchanged: the candidate is a file for the
author to choose from.

**Result.** The search's final mean scores 0.690 +- 0.005 against the yardstick's 0.608 +-
0.005 (9,216 games each, the measure above). The gain holds at a quarter-box spread, across
every condition, and against personas outside the panel.

### Set-up

- pycma, ten rows: the nine that matter plus temperature. Each is in the table's units (log
  rows in log space) and scaled by its box width. It starts at the yardstick with step 0.3
  box half-widths. Every other row stays at its default.
- Bounds are the screening's boxes, with two widened upward to three half-widths: lead_gain
  to 0.8-1.6 (was 0.8-1.2) and fight_speed to 0.82-1.82 (was 0.82-1.22).
- Population 24 and 6 games a cell (864 games a candidate, error about 0.017). A generation
  took 130-155 s. Step floor 0.1 half-widths (pycma `minstd`).
- Stop: no rise of the population's mean skill beyond its error over 4 generations (minimum
  8), or 42 minutes. The rule stopped it after generation 12 (26.7 min). The best population
  mean, 0.648 +- 0.009, came at generation 8; generation 12 gave 0.654.
- The step did not shrink toward the floor. The overall step rose from 0.30 to 0.36, and the
  per-row steps at the end were 0.17 to 0.44 half-widths. So the search ended in a broad
  region rather than converging, and the plateau evidence is the local spread below.

### The candidate

| row | default | candidate | box |
|---|---|---|---|
| lead_turn_scale | 0 | 0.0013 | 0-1 |
| lead_gain | 1.0 | 1.124 | 0.8-1.6 |
| yo_yo_time | 0 | 0.0008 | 0-0.5 |
| fight_speed | 1.0 | 1.114 | 0.82-1.82 |
| closing_time | 1.0 | 0.930 | 0.67-1.49 |
| plan_threshold | 0 | -0.250 | -0.25-0.25 |
| in_range | 1.04 | 1.276 | 0.70-1.55 |
| nose_tail | 0.67 | 0.459 | 0.45-1.0 |
| aligned | 1.01 | 0.876 | 0.68-1.51 |
| temperature | 1.0 | 0.865 | 0.67-1.49 |

- lead_turn_scale and yo_yo_time stayed at their floor of 0.
- plan_threshold ended on the bottom of its box and nose_tail just above its bottom. The
  best region may lie beyond those edges; the boxes were not widened there.

### Skill

| | candidate | yardstick |
|---|---|---|
| skill (9,216 games) | 0.690 +- 0.005 | 0.608 +- 0.005 |
| local25 mean | 0.676 | 0.607 |
| local25 sd | 0.037 | 0.041 |
| local25 5% quantile | 0.614 | 0.538 |
| held-out personas | 0.735 +- 0.006 | 0.643 +- 0.007 |

- **local25** is the screening's quarter-spread: the same 256 Sobol points (stream
  `local25`), all 34 sampled rows, centred on each, 288 games a point (noise sd about 0.027).
  The candidate's neighbourhood scores 0.070 +- 0.003 above the yardstick's, paired by point.
  It falls 0.014 from its centre, against 0.002 for the yardstick. That is a slightly rounder
  top, but still a plateau: 96% of the candidate's neighbours score above the yardstick.
- **Held-out personas** are 8 drawn from the full boxes (seed 20261125), none of them in the
  panel, 32 games a cell (2,304 games each). The gain is not the panel's.
- Head to head (the yardstick as a panel member, 1,152 games), the candidate scores 0.589.

### By condition

| level | candidate | yardstick |
|---|---|---|
| 3,000 m | 0.682 | 0.591 |
| 4,200 m | 0.695 | 0.614 |
| 6,000 m | 0.693 | 0.619 |
| 3v3 | 0.666 | 0.590 |
| 4v4 | 0.698 | 0.603 |
| 5v5 | 0.706 | 0.632 |
| mixed | 0.689 | 0.602 |
| sides | 0.691 | 0.615 |

The candidate is ahead in all 18 conditions, by 0.029 to 0.136.

### GPU time

Laptop RTX 4070, performance profile: search 26.7 min, confirmation 4.8 min, held-out 1.5
min, about 33 minutes in all. 345,024 new games, 248,832 of them the search's.

Reproduce: `search.py STORE search`, then `confirm`, then `held-out` (needs pycma, in the
tools extra).
