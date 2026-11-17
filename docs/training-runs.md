# Training runs

A short log of the runs whose checkpoints and artifacts are kept. The current reference
run is [`wild-universe-731`](#wild-universe-731); its measurements are what
[evaluation](evaluation.md) reports and what every published figure comes from. Earlier
runs are kept as history; their checkpoints are still on disk and still load.

Ratings below are post-hoc calibrated Elo with the scripted controller at 1000, each
measured within its own run, and each is the final live checkpoint's. They are not
comparable across runs that trained under different physics — see the note at the end.

## wild-universe-731

**Current reference run.** 672M of a planned
1,000M environment steps, 674 updates, about
2.9 days on one RTX 4070 Laptop; the run was stopped there (status
`interrupted`). W&B run `mwvvm4kb`.

The first run under frontal armour, and the only one trained under this version's physics,
in which a glancing bullet does 30% of full damage.
Trained 4-vs-4 with 4 refractive fields. The final live policy calibrates
to **2,110 Elo** and the best averaged checkpoint to
2,052. Against the scripted controller the final live
policy holds a numerical advantage from 2.00x at 1-vs-1 and
1.75x at its training size down to
1.19x at 64-vs-64 — sixteen times its training width. Learning
is front-loaded: the calibrated curve passes 1,780 by
150M steps and stays within 50 Elo of the final rating from about
457M onward.

Reward weights, in the derived form this version uses: win 1, death
0.283, damage 0.274, with kill and damage payout ratios
of 2 and a kill-shot fraction of
0.5; facing 0.09 and closing speed
0.08. Shoot-quality shaping off. Peak learning rate
4.5e-04.

[Artifacts](../checkpoints/wild-universe-731/artifacts/) ·
[figures](../checkpoints/wild-universe-731/artifacts/figures/)

## good-leaf-719

The previous reference run. 999M environment
steps, 1004 updates, about four days on one RTX 4070 Laptop, under the earlier full-damage
physics: every bullet did full damage.

Trained 4-vs-4 with four refractive fields. Final live policy calibrates to
1,748 Elo *on its own scale*. Its own crossover
sweep has it beating 1.50x its number of
scripted ships at 4 ships and 1.23x at
64, but only an equal number at 1-vs-1
(1.00x).

Reward weights: win 1.5, kill/assist 1.0, deaths 1.0 (combat and field), damage dealt and
taken 0.5, facing and closing speed 0.1. Shoot-quality shaping off. Peak learning rate
4.5e-4.

[Artifacts](../checkpoints/good-leaf-719/artifacts/) ·
[figures](../checkpoints/good-leaf-719/artifacts/figures/)

## lunar-cosmos-716

500M steps, 502 updates, under the same profile and physics as 719. On the stationary
reference ladder the two agreed closely, so their ratings are comparable, and 719's extra
500M steps were worth about 114 Elo. It carried shoot-quality shaping at 0.1 and a lower
peak learning rate, and its final policy calibrated to 1634 Elo. Its checkpoints and
artifacts are not kept in this repository.

## resilient-resonance-682

999M steps. The reference run before
719. Trained without fields, under an earlier reward schema — win 4.0, a single
undifferentiated death and damage-taken term — and materially different combat physics.

Final live policy calibrates to 1,772
Elo *on its own scale*; see below.

[Artifacts](../checkpoints/resilient-resonance-682/artifacts/)

## A note on comparing ratings

Elo is defined within a pool playing one game. 682 trained with glancing bullets doing a
tenth of full damage, twice the decision rate, and no refractive fields; 719 trained with
every bullet at full damage and half the decision rate; wild-universe-731 trained with frontal
armour and glancing bullets at 30% of full damage.
The rules change how dangerous an untrained policy is, so they stretch or compress the
whole ladder: the distance from random to the scripted controller is
1,335 Elo in 682's world,
863 in 719's and
914 in 731's.

Anchoring the scripted controller at 1000 pins one point without fixing that spacing. So
682's 1,772, 719's
1,748 and 731's
2,110 are not three measurements of the same quantity, and
the differences between them say nothing about which policy is stronger. Only runs that
share physics — 716 and 719 — are directly comparable, and are compared above.
