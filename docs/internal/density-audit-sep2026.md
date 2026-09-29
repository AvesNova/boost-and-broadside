# Is the hex density target actually learned? — September 29, 2026

Subject: the global density head of `pious-butterfly-748`, checkpoint
`step_000084541440.pt`. The head's loss fell 0.222 → 0.009 against a target
power of 0.208, apparently 96% of the field explained — including the *enemy*
field, most of whose ships the observer cannot see. That is the kind of number
that is usually an artefact, so it was audited before anything was built on it.

Evidence in this directory: [`density-baselines-748-sep2026.json`](density-baselines-748-sep2026.json)
and the four figures `density-748-sep2026-{fields,ladder,hidden,rings}.png`.
Tools: [`benchmarks/density_baselines.py`](../../benchmarks/density_baselines.py),
[`benchmarks/density_plots.py`](../../benchmarks/density_plots.py).

## Verdict

**The head is doing real work, and the suspicion was wrong on all three counts.**
Nothing here argues for changing the target before the rebalance.

## The three ways it could have been an artefact

**Quoted against power rather than variance.** It was, and that did flatter it:
the target's mean square is 0.208 but its variance is 0.085, so a predictor of
zero scores R² = −1.46 rather than 0. Corrected, the head reads **0.925 on the
ally half and 0.908 on the enemy half** — lower than 0.956, still high.

**One mean over both halves, letting the easy half carry the hard one.** Split,
the two halves are almost the same difficulty: 0.925 ally against 0.908 enemy.
The half the observer cannot see is barely harder than the half it can.

**"Mirror of the front", a feature the head reads off its own token.** Refuted.
A per-cell linear model in the signed front position scores **R² 0.004 / −0.003**
— the front alone says essentially nothing about where either fleet is. (The
first run of this measurement read the front off token 0, which is zero-padding
on that channel; the script now asserts the feature varies, because a constant
feature silently makes its baseline identical to the mean predictor.)

## The ladder

Held out by environment, not by row: steps within an episode are strongly
correlated and a random row split would leak the answer into the fit. 122,880
samples, 331 cells per half, after 24 warm-up rollouts (visible-enemy fraction
0.574 against the run's logged `fog/visible_fraction` of 0.577).

| predictor | ally R² | enemy R² |
|---|---:|---:|
| zero (where the head starts) | −1.46 | −1.47 |
| per-cell mean — the constant heat map | 0.000 | 0.000 |
| + front position | 0.004 | −0.003 |
| + the visible ally field — *everything inferable without belief* | — | **0.167** |
| the head | **0.925** | **0.908** |

The row that matters is the fourth: front plus the observer's complete, visible
ally density explains 17% of the enemy field's variance. The head explains 91%.

## Does the enemy field come from belief or from bookkeeping?

`density-748-sep2026-hidden.png`. Enemy-half error against the number of
enemies currently out of sight, all six levels well sampled (4.5k–16.5k frames
each):

| enemies hidden | 0 | 5 | growth |
|---|---:|---:|---:|
| enemy half MSE | 0.0053 | 0.0144 | **2.7×** |
| ally half MSE | 0.0059 | 0.0101 | 1.7× |

The enemy field degrades as the enemy disappears, which bookkeeping over
visible ships would not do. But the ally half degrades too, and allies are
always visible — frames with many hidden enemies are simply harder frames. So
the belief-specific effect is the *excess*, 2.7× against a 1.7× baseline drift,
not the raw 2.7×. Even fully blind the head holds R² ≈ 0.83.

The field panels show the same thing structurally: the errors are coherent
dipoles, not noise — the predicted blob is in slightly the wrong place — and
the prediction is visibly smoother than the truth as occlusion rises, which is
a head hedging under uncertainty rather than one guessing.

## One design observation

`density-748-sep2026-rings.png`. Error falls steeply from the centre outward and
is near zero by ring 8. Rings 7–10 are **204 of 331 cells (62%) but only 14% of
the target's variance** — most of the grid is empty most of the time, because
play concentrates near the front.

That is *not* inflating the score: restricted to rings 0–6 the head reads
**higher**, 0.923 enemy against 0.908 over all rings. So the outer rings cost a
little accuracy rather than buying free credit. But 62% of the head's output
width is carrying 14% of the signal, which is worth knowing if the target is
ever revisited for capacity reasons.

## What this changes

Nothing immediately. The head earns its place, `global_density_coef` does not
need rethinking on these grounds, and the "make the target harder rather than
louder" suggestion is not supported by this evidence — the target is not too
easy. The open question it leaves is capacity, not difficulty: whether 331
cells with 62% of them near-empty is the right shape.
