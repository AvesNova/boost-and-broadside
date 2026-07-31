# Evaluation and results

The central result is zero-shot transfer across team size: a policy trained in 4-vs-4
combat remains effective without retraining as the learned fleet grows from one to 64
ships. Crossover against the scripted controller provides the quantitative view; seeded
replays show what that transfer looks like in motion.

This page distinguishes stored measurements from interpretations and describes the raw
artifacts behind every headline number. Known gaps are collected in
[remaining limitations](#remaining-limitations).

## Reference run and provenance

All headline figures refer to `resilient-resonance-682` (`chpl40cj`). Its preserved
artifacts include:

- [training configuration](../checkpoints/resilient-resonance-682/wandb_export/config.json);
- [sampled metric history](../checkpoints/resilient-resonance-682/wandb_export/history.jsonl)
  and [final summary](../checkpoints/resilient-resonance-682/wandb_export/summary.json);
- [raw online match history](../checkpoints/resilient-resonance-682/elo_history.jsonl);
- [post-hoc calibration result](../checkpoints/resilient-resonance-682/elo_calibrated.json);
- [crossover sweep](crossover/crossover.json);
- [curated seeded replays](results/replays/).

The run targeted one billion environment steps and logged 999,424,000. The W&B summary
records 7.50 training hours at a final logged throughput of about 296,000 ship-tokens/s
(37,000 environment steps/s at 8 ships per environment);
[hardware metadata](../checkpoints/resilient-resonance-682/wandb_export/files/wandb-metadata.json)
identifies one RTX 5090. Ship-tokens/s is the primary throughput measure here because it
stays comparable when ships-per-environment changes; steps/s does not.

## Zero-shot crossover

![Policy-controlled vs scripted-controlled crossover](results/crossover_phase.png)

*The learned fleet's 50% crossover stays above equal numbers throughout the displayed
1-to-64 range.*

[`crossover.py`](../src/boost_and_broadside/modes/crossover.py) fixes the learned policy to
team 0 and the stochastic scripted controller to team 1. For each learned-team size `T`,
it searches increasing scripted-team counts `S` and records:

- `beats_up_to`: the largest `S` with learned-team win rate at least 50%;
- `crossover`: the first adjacent `S` below 50%;
- the measured rates at those points and every search probe.

Ties count against the learned team. The plot draws the visual boundary halfway between
the two adjacent integer counts; it is not a continuous fitted threshold.

The same checkpoint is loaded once and evaluated across the grid. At each matchup, the
runtime ship-token count changes but no policy weight is updated. The stored sweep covers
every learned-team size from 1 through 64.

Selected measurements:

| Policy-controlled ships | Largest scripted team still beaten | Win rate there | First below 50% | Win rate there |
|---:|---:|---:|---:|---:|
| 4 | 5 | 81.6% | 6 | 40.2% |
| 8 | 11 | 69.5% | 12 | 42.2% |
| 16 | 24 | 52.7% | 25 | 33.6% |
| 32 | 47 | 55.9% | 48 | 47.3% |
| 64 | 87 | 53.1% | 88 | 46.8% |

This supports an empirical claim: the 4-vs-4-trained controller remains effective without
retraining at much larger and asymmetric sizes, and the recorded crossover stays above
numerical parity. It does **not** by itself establish a universal scale-invariance law —
the scripted controller is one opponent family, individual boundary rates carry binomial
sampling noise, and the search assumes a locally monotone boundary.

The stored sweep predates count-preserving output: it records win rates and a run-level
maximum of 256 parallel games, not per-point win/loss/tie counts (current evaluator runs
store all of those per cell, plus mean episode length). The historical rates are
therefore reported as stored, without inventing more precision.

![Scripted ships beaten per policy-controlled ship](results/crossover_ratio.png)

The ratio view describes where the measured numerical advantage is larger. Its shape is
descriptive: no ablation separates coordination, per-ship tactical skill, and weaknesses
of the scripted controller.

## Qualitative crossover evidence

![Eight policy-controlled ships versus eleven scripted-controlled ships](results/replays/vs_scripted_8v11_seed03.gif)

Outnumbered 11 ships to 8, the learned fleet wins with three ships to spare. See
[replays](replays.md) for larger battles and capture details. A clip is one qualitative
realization, not the source of the aggregate rates above.

## Post-hoc Elo calibration

![Calibrated live-policy Elo](results/elo_curve.png)

Online Elo is useful during training, but its location can drift with sequential K-factor
updates and changing opponents. [`elo_calibrate.py`](../src/boost_and_broadside/modes/elo_calibrate.py)
constructs a post-hoc scale in two stages:

1. run an adaptive tournament among stationary players — random, scripted, and frozen
   ladder checkpoints — until their fitted uncertainty reaches the target;
2. refit each historical live policy from that update's saved win/loss/tie record against
   the now-calibrated opponents.

The fit uses [Bradley-Terry](https://doi.org/10.2307/2334029) expected scores. The
primary convention treats a draw as half a win for each side; a decisive-games-only fit
is retained as a diagnostic. Ratings are reported with the scripted controller fixed at
1000 — the same convention as the fleet-scale view below. On this scale the random
baseline reads about −240, and 400 points correspond to ten-to-one odds.

Key values from [`elo_calibrated.json`](../checkpoints/resilient-resonance-682/elo_calibrated.json):

| Measurement | Elo | Conditional SE | Games |
|---|---:|---:|---:|
| Final live policy, 999.424M steps | 1812.9 | 18.4 | 627 recorded games |
| Last frozen checkpoint, 876.495M steps | 1816.8 | 9.5 | 6,134 tournament games |
| Scripted controller | 1000 (anchor) | 6.2 | 7,895 tournament games |
| Random baseline | −240.0 | 32.8 | 16,182 tournament games |

Pinning the scale to scripted is uncertain by ±6 Elo, shared by every rating; that
common shift cancels when comparing two players, so the final policy's lead over
scripted is about 813 points however the scale is pinned. The final *online* training
rating was 1547.3; it is a different, drifting estimator and should not be mixed with
the calibrated curve.

The compact plot above omits uncertainty bands. The calibration directory preserves
methodology plots with error bands and convergence diagnostics under
[`checkpoints/resilient-resonance-682/elo_calibration/`](../checkpoints/resilient-resonance-682/elo_calibration/).

## Symmetric fleet-scale ratings

The frozen training ladder and final checkpoint were replayed in symmetric battles from
1-vs-1 through 64-vs-64. Each size has its own stationary tournament containing random,
scripted, the frozen ladder checkpoints, and the final checkpoint. The evaluator swaps
team roles and stores directed win/loss/tie counts in
[`elo_scale.json`](../checkpoints/resilient-resonance-682/elo_scale.json).

Ratings again fix the scripted controller at 1000, so the plotted quantity — the
checkpoint's advantage over a consistent opponent — is directly comparable with the
calibrated training curve above.

![Fleet-scale Elo with scripted anchored at 1000](results/elo_scale_scripted_1000.png)

*The 4-vs-4 checkpoint strengthens as the fleet grows zero-shot, peaking at 16-vs-16 in
this evaluation. The 32- and 64-ship estimates have visibly wider uncertainty.*

| Ships per team | Checkpoint-tournament games | Final checkpoint Elo (±1 conditional SE) |
|---:|---:|---:|
| 1 | 143,488 | 1539 ± 6 |
| 2 | 270,336 | 1680 ± 5 |
| 4 | 507,904 | 1822 ± 4 |
| 8 | 1,549,152 | 1996 ± 3 |
| 16 | 1,460,844 | 2173 ± 4 |
| 32 | 5,856 | 2152 ± 49 |
| 64 | 1,464 | 1923 ± 82 |

At the native 4-vs-4 scale this tournament rates the final checkpoint 1822 ± 4 — within
uncertainty of the calibrated final-live rating of about 1813 ± 18. The two measurements
use different final-policy evidence, so exact equality is not expected.

### Reference-ladder conditioning

Random play is extremely far from scripted play at large fleet sizes, making a direct
random-to-scripted link statistically inefficient. The calibration therefore adds
intermediate controllers. For each ship on each simulation step, a controller with
probability `P` uses the complete scripted action; otherwise it samples a complete
action uniformly at random. The refined ladder uses
`P = 0, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100%`.

![Reference-ladder connectivity](results/semi_random_connectivity.png)

*Across seven fleet sizes, the stronger rung scores between 58.2% and 84.8% against its
neighbor. Most adjacent comparisons fall in the shaded 20–80% target region; none is
close to a deterministic matchup.*

The saved [reference tournament](../checkpoints/resilient-resonance-682/semi_random_tournament.json)
contains 128 side-balanced games for every unordered pair at every scale. Its outcome
matrices are joined to the checkpoint tournament through the shared random and scripted
endpoints before fitting, which improves graph connectivity without treating an
interpolated controller as a trained checkpoint.

## Scripted benchmark over training

![Win rate against the scripted controller](results/win_rate_vs_scripted.png)

Across the 999 sampled history points with this metric, the policy first reaches 95% at
127.9M environment steps and 99% at 221.9M; the final point is 100%. The series is not
monotone — the minimum sampled value after 200M is 89% — so it is best read as a benchmark
becoming a weak discriminator, not permanent saturation at a precise step.

The calibrated Elo continues to improve after the scripted curve becomes less informative.
That is consistent with continued learning from self-play and league opponents, but it
does not isolate which mechanism caused the improvement.

## Auxiliary dynamics learning

![Normalized next-state prediction error](results/next_state_error.png)

The auxiliary head's normalized errors fall strongly for predictable dynamics channels.
For example, position-x falls from 1.49 at the first sampled update to 0.0015 at the
last; velocity-x from 3.04 to 0.019; power from 1.47 to 0.019. Health changes much less,
from 0.57 to 0.49.

These are direct measurements from the W&B history rendered by
[`scripts/render_charts.py`](../scripts/render_charts.py). A plausible reading is that
health is dominated by sparse, hard-to-forecast damage events, but that is an
interpretation — no ablation isolates the cause. Deeper sequence behavior is in
the [autoregressive reports](ar_report/) and [noise calibration](noise_calibration/).

## Training health

![Training diagnostics](results/training_health.png)

The final aggregate critic explained variance is 0.939, with a sampled maximum of 0.943.
The panel also shows reward, KL, and clip-fraction trajectories. These are optimization
diagnostics rather than headline task-performance measures.

## Reproduce the figures

The README/evaluation charts share one renderer for W&B-format history and calibrated
history:

```bash
uv run scripts/render_charts.py \
  --run resilient-resonance-682 \
  --out docs/results

uv run scripts/render_crossover.py \
  --data docs/crossover/crossover.json \
  --out docs/results

uv run scripts/render_elo_scale.py \
  --data checkpoints/resilient-resonance-682/elo_scale.json \
  --reference-data checkpoints/resilient-resonance-682/semi_random_tournament.json \
  --out docs/results

uv run scripts/render_semi_random.py \
  --data checkpoints/resilient-resonance-682/semi_random_tournament.json \
  --out docs/results
```

The renderer is the source of the axis labels and equal-scale geometry; the tracked
rasters are regenerated from it rather than edited independently. Refitting the
calibration itself (for example under a different anchor or draw convention) needs no
GPU: `uv run main.py --mode elo_calibrate --run resilient-resonance-682 --refit` refits
the stored win/tie matrices.

To rerun the underlying evaluations, see [getting started](getting-started.md#evaluate).
They can require substantial GPU time; rendering from included artifacts does not.

## Remaining limitations

- The stored crossover sweep lacks per-point game counts, seeds, checkpoint hash, source
  commit, and confidence intervals; newer evaluator runs record these.
- The 32- and 64-vs-64 checkpoint ratings have materially wider uncertainty than the
  smaller-fleet tournaments.
- Curated GIFs lack sidecar metadata tying them to an exact checkpoint hash and capture
  command.
- Results come from one training run against one scripted opponent family; there is no
  multi-run seed study.
- The transfer results demonstrate execution and performance across observed sizes, not
  invariance under arbitrary maps, physics, observation changes, or unbounded team size.
- No causal ablation assigns the observed transfer to attention, recurrence, auxiliary
  prediction, reward decomposition, or league play individually.
