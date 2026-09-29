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

## Kernel width — the target is smoother than the grid

> **Superseded September 30 2026.** This section argued the Gaussian target was
> over-smoothed. It was, and the conclusion was to remove the Gaussian rather
> than retune it: the target is now barycentric ship counts with no kernel at
> all (see *Replaced* at the end). What follows is the evidence that led there,
> and it still describes `local_presence`, which keeps the kernel.

`presence-kernel-sep2026-{single,profile,pairs}.png`, from
[`benchmarks/density_kernel_widths.py`](../../benchmarks/density_kernel_widths.py).
No checkpoint involved: this is a property of the target's definition.

`PRESENCE_RADIUS` is 500 px against a 2600 px playable radius and a 260 px ring
spacing. One ship on its own is therefore not a point on this field:

| kernel radius | half-max | cells above 10% of peak |
|---|---:|---:|
| 125 px | 0.6 rings | 2% |
| 250 px | 1.3 rings | 6% |
| **500 px (production)** | **2.6 rings** | **22%** |
| 1000 px | 5.1 rings | 85% |
| 2000 px | 10.2 rings | 100% |

At the production radius a single ship lights roughly a fifth of the 331 cells,
and its half-max reaches 2.6 ring spacings — the grid samples the field about
five times finer than the field varies.

The resolution consequence is sharper than the spread. Two ships at 500 px do
not read as two until they are about **2000 px apart, 8 ring spacings, roughly
40% of the playable diameter**; at 1040 px (4 rings) the midpoint is still the
field's maximum, so the pair reads as one elongated blob. A 5v5 fleet spread
across a quarter of the map is, to this target, the same object as a 5v5 fleet
stacked on one point.

That does not contradict the audit above — the head predicts what it is asked
to predict, and predicts it well. It bears on what the target is *worth*: a
field this smooth cannot express formation, only bulk position, so a head that
scores 0.91 on it has learned where the fleets roughly are and nothing about
how they are arranged.

### Resolved: 200 px, September 30 2026

Narrowed to 200 px, and kept coupled to `local_presence` rather than split, so
"crowded" still means one thing at a ship and at a map cell.

The prior basis for 500 px did not survive re-measurement. It rested on "at
250 px the 5v5 enemy channel is dead, median 0.18 with most ships reading
zero"; the frontline mechanic has since changed that scene distribution by
holding both fleets in contact, and the same 250 px now reads a median of 0.77
([`presence-radius-sep2026.json`](presence-radius-sep2026.json)). At 200 px the
5v5 enemy channel carries a median of 0.55 with a 10th-to-90th spread of 1.31 —
live and discriminating.

| | one ship lights | two ships separate at |
|---|---:|---:|
| 500 px | 22% of cells | ~2080 px (8 rings) |
| **200 px** | **4% of cells** | **520 px (2 rings)** |

A four-fold gain in resolution, which is what the change was for.

The cost is separation *across* fleet sizes, not within 5v5: the enemy median
between 5v5 and 50v50 on one map falls from ×1.98 to ×1.18. That bears on the
zero-shot crossover sweep, which is not what this run is doing, and is the
thing to re-check if scale transfer matters again.

Two consequences to carry forward. `RELATION_RADIUS` aliases this constant, so
the relational bias narrows with it — inert while `relational_bias` is off, as
it is in every current profile, but not free later. And `local_presence` is a
policy *input*, so run 748's checkpoint now reads a feature distribution it was
not trained on; this is one more reason the next run starts fresh.

## What this changes

Nothing immediately. The head earns its place, `global_density_coef` does not
need rethinking on these grounds, and the "make the target harder rather than
louder" suggestion is not supported by this evidence — the target is not too
easy. The open question it leaves is capacity, not difficulty: whether 331
cells with 62% of them near-empty is the right shape.


## Replaced — barycentric counts, September 30 2026

The Gaussian target is gone. Each living ship now deposits exactly one unit of
mass, split across the three cells whose centres form the triangle containing
it, by barycentric weight; the field sums to the living ship count per side and
the head predicts it as a Poisson log-rate.

No radius, and nothing to tune. Three weights summing to one are the exact 2D
analogue of two-hot encoding on a line, and they are lossless — a ship's three
weights invert to its exact position inside the triangle, which is precisely
what the smoothing above was throwing away.

Off the grid, the lattice is treated as infinite: a ship's containing triangle
is found wherever it is, and only those vertices that are cells of the finite
grid receive their weight. A ship at 1.02 playable radii still lands partly on
the rim; by 1.10 it is gone. Continuous, with no clamp and no special case.

The loss is Poisson negative log likelihood, whose gradient in the logit is
`exp(l) - y`. That is bounded and vanishes exactly when the rate matches the
count — unlike the `r / sigma^2` of the next-state head, which is what
concentrated 76% of the trunk gradient on the tokens it already predicted best.

**The audit above does not carry over.** Its R² ladder, its baselines and its
`-fields`/`-ladder`/`-hidden`/`-rings` figures all score a regression against a
smoothed field. They need re-running against the count target before any of
those numbers mean anything again.
