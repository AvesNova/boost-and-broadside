# Temperature sweep (2026-11-11)

**Question.** Does the trained RL policy have too little or too much entropy for the
opponents it meets? If entropy collapse had cost strength, a hotter policy (T > 1) would
rate higher; if the policy is still too soft, a colder one (T < 1) would. Nothing is
trained. Each player samples its joint action from logits / T
(`<policy>@T=<t>`, `bnb.league.matches.temperature_of`).

## Setup

- **Policies:** the uniform averages (the variant tournament 7 rated along the curve) of
  both RL headline runs at their 121.6M checkpoint (`step_121634816`) and at the one
  nearest 60M (`step_60817408`), downloaded from `julianzia/boost-and-broadside-runs`.
  Their weight hashes match their records, and they load on this tree with no warnings.
  The null run is `2026-11-09-rl-absolute` and the champion
  `2026-11-09-rl-deadreckon-split-hiddenphysics`.
- **Field:** the 4 policies at T = 0.7, 1 and 1.4 (12 players), plus `scripted`, which
  anchors the scale at 1000.
- **Presets:** frontline and deathmatch, each rated separately, as in tournament 7. The RL
  runs trained on the mixed preset.
- **Allocation:** A-optimal to a worst standard error of 12, with a budget of 80,000
  games per preset. Blocks were 8 x 512 games (4,096 per batch, 8.0 GB of GPU memory).
  The first batch per preset took 18–27 min, mostly compile; each later batch took
  285–320 s.
- **Played:** frontline took 6 batches (24,576 games, worst error 11.4); deathmatch took
  11 batches (45,056 games, worst error 11.4).
- **Records:** the per-batch records with each block's statistics are in
  `temperature-sweep/<preset>.batches.jsonl`, and the tables in
  `temperature-sweep/sweep.json`.
- **Reproduce:** `python -m bench.temperature_sweep report --out
  docs/rewrite/sections/temperature-sweep`.
- **Entropy:** the mean, over living ship-decisions, of the joint (42-way) entropy of the
  distribution actually sampled, in nats (maximum ln 42 = 3.74). It is measured in the
  rating games themselves (`STATISTICS`' `entropy` / `ship_ticks`).

## Elo +- standard error (mean entropy, nats)

Frontline:

| policy | T = 0.7 | T = 1 | T = 1.4 |
|---|---|---|---|
| champion 121.6M | 1741 +- 11 (0.43) | **1755** +- 11 (0.88) | 1667 +- 11 (1.56) |
| champion 60.8M | 1576 +- 10 (0.61) | **1579** +- 10 (1.12) | 1499 +- 10 (1.76) |
| null 121.6M | **1661** +- 11 (0.53) | 1612 +- 11 (1.01) | 1478 +- 10 (1.66) |
| null 60.8M | **1559** +- 11 (0.70) | 1516 +- 11 (1.23) | 1387 +- 8 (1.82) |

Deathmatch:

| policy | T = 0.7 | T = 1 | T = 1.4 |
|---|---|---|---|
| champion 121.6M | 2128 +- 11 (0.41) | **2155** +- 11 (0.82) | 2032 +- 11 (1.51) |
| champion 60.8M | **1858** +- 10 (0.57) | 1851 +- 10 (1.04) | 1752 +- 10 (1.66) |
| null 121.6M | 1937 +- 11 (0.51) | **1954** +- 11 (0.96) | 1876 +- 10 (1.58) |
| null 60.8M | **1706** +- 10 (0.68) | 1684 +- 10 (1.14) | 1553 +- 7 (1.70) |

Each temperature less the same policy at T = 1. The error is the difference's own, from
a fit anchored on the T = 1 player:

| policy | frontline T = 0.7 | frontline T = 1.4 | deathmatch T = 0.7 | deathmatch T = 1.4 |
|---|---|---|---|---|
| champion 121.6M | -13 +- 10 | -86 +- 10 | -26 +- 9 | -122 +- 9 |
| champion 60.8M | -2 +- 10 | -80 +- 10 | +8 +- 8 | -98 +- 8 |
| null 121.6M | +49 +- 10 | -134 +- 12 | -16 +- 9 | -77 +- 9 |
| null 60.8M | +43 +- 12 | -130 +- 10 | +21 +- 10 | -132 +- 8 |

For reference, the training's live policy entropy (`policy/entropy/joint`, per update) was:

| run | 60.8M | 121.6M |
|---|---|---|
| null | 1.03 | 0.70 |
| champion | 0.91 | 0.66 |

`alerts/entropy_collapse` was set at both.

## What it says

- **More entropy loses everywhere.** At T = 1.4 every policy, in both modes, rates 77–134
  below itself at T = 1 (6–14 standard errors). The entropy collapse the alert flags has
  not cost strength against this field: these policies do not want more entropy.
- **Less entropy is roughly neutral.**
  - Mean of the two modes at T = 0.7 against T = 1: champion 121.6M -20, champion 60.8M
    +3, null 121.6M +17, null 60.8M +32.
  - The null gains clearly in frontline (+43, +49) and the champion 121.6M loses slightly
    in deathmatch (-26).
  - The best temperature lies between 0.7 and 1, and nearer 1 for the stronger policy.
    Sampling sharper buys at most a few tens of Elo, and only for the weaker, softer
    policies.
- **Entropy at the best temperature.**
  - Joint entropy is about 0.4–1.0 nats per ship-decision: 0.82–0.88 for the champion at
    121.6M (T = 1), and 0.53–0.70 for the null (T = 0.7 in frontline).
  - The policies played well throughout about 0.5–0.9 nats, and badly at 1.5 and above.
  - As a target entropy, that band is what the live policies already reach (0.66–0.70 at
    121.6M). It gives no reason to raise entropy, and a target near 0.7 nats would hold
    the policy where it plays best, not loosen it.

## Caveats

- **Uniform averages, not the live policy.**
  - At T = 1 the averages play softer (0.82–1.23 nats) than the live policy trained
    (0.66–1.03). A colder sample of an average partly stands in for the live policy's
    sharpness.
  - The live policy itself was not swept. The finding for it may differ; it is the
    natural next field, since tournament 7 rated the champion's live policy at 121.6M
    highest.
- **A temperature is not training to an entropy.** Dividing the logits flattens or
  sharpens every state alike. A policy trained to a target entropy would put its entropy
  where it matters. The sweep bounds the effect; it does not predict a trained one.
- **The scale is this field's.**
  - The absolute Elos run 70–110 below tournament 7's for the same checkpoints (for
    example champion 121.6M frontline 1755 here, 1825 there).
  - The field has nothing between the scripted agent (1000) and about 1400, so the
    anchor's link runs through lopsided games. The scripted agent may also have changed
    since tournament 7.
  - The comparisons within the field (the difference table) are what this sweep
    measures.
- **The opponents are this field.** "Too little entropy" against exploiters or a
  different league could read otherwise; here every opponent is one of these policies or
  the scripted agent.
- **One seed per run, two checkpoints per run.**
