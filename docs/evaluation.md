# Evaluation and results

The central result is zero-shot transfer across team size: one recurrent policy trained in
4-vs-4 is evaluated without retraining while jointly controlling learned teams from one to
64 ships. Crossover against the scripted controller provides the quantitative view;
seeded replays provide qualitative evidence.

This page distinguishes stored measurements from interpretations and describes the raw
artifacts behind every headline. The maintainer-level claim ledger is
[`evidence.md`](evidence.md).

## Landmark run and provenance

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
records 7.50 training hours, a final logged 37,000 environment steps/s and 296,002 ship
tokens/s; [hardware metadata](../checkpoints/resilient-resonance-682/wandb_export/files/wandb-metadata.json)
identifies one RTX 5090. These are run-specific measurements.

## Zero-shot crossover

![Policy-controlled vs scripted-controlled crossover](results/crossover_phase.png)

*Each axis unit in the existing plot is a ship, despite the older “agents” labels. One
recurrent policy jointly controls every ship on the learned team. The dashed diagonal is
equal ship counts.*

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

The result supports an empirical claim: the 4-vs-4-trained controller remains effective
without retraining at much larger and asymmetric sizes, and the recorded crossover stays
above numerical parity.

It does **not** by itself establish a universal scale-invariance law. The scripted
controller is one opponent family, individual boundary rates carry binomial sampling
noise, and the search assumes a locally monotone boundary even though finite samples can
fluctuate.

### Sample-size limitation

The JSON stores a run-level maximum of 256 parallel games, not the exact game count beside
each curve point. Current evaluator logic uses all 256 for the 8-vs-11 matchup and reduces
the count for large battles to fit a `B×N²` collision-memory budget. Because the artifact
does not preserve per-point counts, checkpoint hash, seed policy, source commit, or
confidence intervals, this documentation reports stored rates without inventing more
precision. Those fields are high-priority additions in the [deferred asset
ledger](evidence.md#deferred-asset-and-analysis-ledger).

![Scripted ships beaten per policy-controlled ship](results/crossover_ratio.png)

The ratio view is useful for describing where the measured numerical advantage is larger,
but its shape is descriptive. No ablation separates coordination, per-ship tactical skill,
and weaknesses of the scripted controller.

## Qualitative crossover evidence

![Eight policy-controlled ships versus eleven scripted-controlled ships](results/replays/vs_scripted_8v11_seed03.gif)

The preferred replay matches the most legible headline row: eight blue ships share one
learned policy, while 11 red ships use the scripted controller. The terminal frame has
three blue survivors and no red ships. See [replays](replays.md) for capture semantics,
larger battles, and the current provenance limitation.

## Post-hoc ELO calibration

![Calibrated live-policy ELO](results/elo_curve.png)

Online ELO is useful during training, but its location can drift with sequential K-factor
updates and changing opponents. [`elo_calibrate.py`](../src/boost_and_broadside/modes/elo_calibrate.py)
constructs a post-hoc scale in two stages:

1. run an adaptive tournament among stationary players—random, scripted, and frozen
   ladder checkpoints—until their fitted uncertainty reaches the target;
2. refit each historical live policy from that update's saved win/loss/tie record against
   the now-calibrated opponents.

The fit uses Bradley-Terry expected scores. The primary convention treats a draw as half a
win for each side; a decisive-games-only fit is retained as a diagnostic. Ratings are
shifted so random reads zero.

Key values from [`elo_calibrated.json`](../checkpoints/resilient-resonance-682/elo_calibrated.json):

| Measurement | ELO | Conditional SE | Games |
|---|---:|---:|---:|
| Final live policy, 999.424M steps | 2052.95 | 18.41 | 627 recorded games |
| Last frozen checkpoint, 876.495M steps | 2056.79 | 9.50 | 6,134 tournament games |
| Scripted controller | 1240.03 | 6.20 | 7,895 tournament games |

The absolute zero point carries an additional ±32.81 ELO uncertainty shared by every
rating after random is shifted to zero. That common shift cancels when comparing two
players, so the final live policy's difference from scripted is about 813 ELO. The final
online training rating was 1547.28; it is a different, drifting estimator and should not
be mixed with the calibrated curve.

The compact plot above omits uncertainty bands. The calibration directory preserves
methodology plots with error bands and convergence diagnostics under
[`checkpoints/resilient-resonance-682/elo_calibration/`](../checkpoints/resilient-resonance-682/elo_calibration/).

## Scripted benchmark over training

![Win rate against the scripted controller](results/win_rate_vs_scripted.png)

Across the 999 sampled history points with this metric, the policy first reaches 95% at
127.9M environment steps and 99% at 221.9M; the final point is 100%. The series is not
monotone—the minimum sampled value after 200M is 89%—so it is best read as a benchmark
becoming a weak discriminator, not permanent saturation at a precise step.

The calibrated ELO continues to improve after the scripted curve becomes less informative.
That is consistent with continued learning from self-play and league opponents, but it
does not isolate which mechanism caused the improvement.

## Auxiliary dynamics learning

![Normalized next-state prediction error](results/next_state_error.png)

The auxiliary head's normalized errors fall strongly for predictable dynamics channels.
For example, position-x falls from 1.492 at the first sampled update to 0.00148 at the
last; velocity-x from 3.042 to 0.0192; power from 1.465 to 0.0186. Health changes much
less, from 0.570 to 0.487.

These are direct measurements from the W&B history rendered by
[`scripts/render_charts.py`](../scripts/render_charts.py). The plausible explanation that
health is dominated by sparse, hard-to-forecast damage events remains an interpretation;
the repository contains no ablation proving that cause. Deeper sequence behavior is in
the [autoregressive reports](ar_report/) and [noise calibration](noise_calibration/).

## Training health

![Training diagnostics](results/training_health.png)

The final aggregate critic explained variance is 0.939, with a sampled maximum of 0.943.
The panel also exposes reward, KL, and clip-fraction trajectories. These are optimization
diagnostics rather than headline task-performance measures, so they remain in the deeper
evaluation page.

## Reproduce the figures

The README/evaluation charts share one renderer for W&B-format history and calibrated
history:

```bash
uv run --no-sync scripts/render_charts.py \
  --run resilient-resonance-682 \
  --out docs/results

uv run --no-sync scripts/render_crossover.py \
  --data docs/crossover/crossover.json \
  --out docs/results
```

The second command currently regenerates the older “agents” axis labels. Relabeling them
to “policy-controlled ships” and “scripted-controlled ships” is intentionally tracked as
deferred plot work rather than silently editing only the raster.

To rerun the underlying evaluations, see [getting started](getting-started.md#evaluate).
They can require substantial GPU time; rendering from included artifacts does not.

## Remaining limitations

- Crossover lacks per-point provenance and uncertainty fields.
- Curated GIFs lack sidecar metadata tying them to an exact checkpoint hash and capture
  command.
- Results use one landmark training run and one scripted opponent family; no multi-run
  seed study is included.
- The transfer results demonstrate execution and performance across observed sizes, not
  invariance under arbitrary maps, physics, observation changes, or unbounded team size.
- No causal ablation assigns the observed transfer to attention, recurrence, auxiliary
  prediction, reward decomposition, or league play individually.
