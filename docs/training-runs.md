# Training runs

A short log of the runs that trained to completion. The current reference run is
[`good-leaf-719`](#good-leaf-719); its measurements are what
[evaluation](evaluation.md) reports and what every published figure comes from.
Earlier runs are kept because their checkpoints and artifacts are still on disk
and still load.

Ratings below are post-hoc calibrated Elo with the scripted controller at 1000,
each measured within its own run. They are not comparable across runs that
trained under different physics — see the note at the end.

## good-leaf-719

**Current reference run.** 999M environment steps, 1004 updates, about four days
on one RTX 4070 Laptop.

Trained 4-vs-4 with four refractive fields. Final policy calibrates to **1748
Elo**, and holds a numerical advantage over the scripted controller from 2.0x at
1-vs-1 down to 1.25x at 64-vs-64 — sixteen times its training width. Learning is
front-loaded: roughly two thirds of the final rating is reached by 150M steps,
and the curve is flat from about 450M onward.

Reward weights: win 1.5, kill/assist 1.0, deaths 1.0 (combat and field), damage
dealt and taken 0.5, facing and closing speed 0.1. Shoot-quality shaping off.
Peak learning rate 4.5e-4.

[Artifacts](../checkpoints/good-leaf-719/artifacts/) ·
[figures](../checkpoints/good-leaf-719/artifacts/figures/)

## lunar-cosmos-716

500M steps, 502 updates. Same profile and physics as 719 and a useful check on
it: the two agree closely on the stationary reference ladder, so their ratings
*are* comparable, and 719's extra 500M steps are worth about 114 Elo.

Differs from 719 in carrying shoot-quality shaping at 0.1 and a lower peak
learning rate. Final policy calibrates to 1634 Elo.

[Calibration artifact](../checkpoints/lunar-cosmos-716/artifacts/elo-calibration/)

## resilient-resonance-682

999M steps. The previous reference run, and the source of every published figure
before 719. Trained without fields, under an earlier reward schema — win 4.0,
a single undifferentiated death and damage-taken term — and materially different
combat physics.

Final policy calibrates to 1772 Elo *on its own scale*, which is not the same
scale as 719's; see below.

[Artifacts](../checkpoints/resilient-resonance-682/artifacts/)

## A note on comparing ratings

Elo is defined within a pool playing one game. 682 trained with glancing bullets
doing a tenth of full damage, twice the decision rate, and no refractive fields;
719 trained with every bullet at full damage and half the decision rate. Under
719's rules an untrained policy is far more dangerous, so the whole ladder
compresses: the distance from random to the scripted controller is 1335 Elo in
682's world and 862 in 719's.

Anchoring the scripted controller at 1000 pins one point without fixing that
spacing. So 682's 1772 and 719's 1748 are not two measurements of the same
quantity, and the difference between them says nothing about which policy is
stronger. Runs that share physics — 716 and 719 — are directly comparable, and
are compared above.
