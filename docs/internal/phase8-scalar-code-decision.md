# Phase 8: scalar encoding and the round-trip contract

Status: decided 2026-10-06, implemented in `train/rl/categorical_codes.py`
(`ScalarCode`) and specified in `frontline-redesign-plan.md` §8.3.

## The problem

§8.3 originally asked for a discretised Gaussian rebuilt from the stored mean
and sigma, a sharp two-hot at sigma 0, decoding by the histogram's mean and
variance, and an exact round trip. Those cannot all hold. For health 52.5 at
sigma 0 the two-hot puts 0.5 on bins 50 and 55: its mean is 52.5 but its
variance is 6.25, so the decoder returns sigma 2.5 and a hidden ship's belief
widens on its own every decision. Near either end of a range, clipping or
folding the Gaussian moves both moments as well.

## The decision

* **Encode:** the exact two-hot of the mean, convolved with a discrete Gaussian
  kernel of variance sigma². The mean is exact because the kernel is
  symmetric; the histogram variance is sigma² plus the two-hot's own
  `h² t (1 − t)`, because variances add under convolution. The kernel is a
  sampled Gaussian at or above one square bin (variance exact to 1e-8 there)
  and the spike mixed with the unit sampled Gaussian below it.
* **Decode:** the histogram mean, and its variance less `h² t (1 − t)`. That is
  never negative: on a lattice the two-hot is the least-variance distribution
  with a given mean.
* **Edges, without padding bins:** mass past the end is folded into the end bin,
  then one closed-form drift-and-diffusion step on the interior bins restores
  the mean and the variance. End bins never send, so nothing spills again.
* **Accepted inaccuracy:** where the end bin holds too much mass for that step
  to reach the target, the moments are projected once and then held.

Correcting only the mean was considered and rejected: the mean then holds while
sigma collapses to zero over repeated steps (97, 5 → 97, 0 after 50 steps),
which makes a hidden ship near full health falsely certain.

## Measured (5-unit bins on [0, 100])

| sigma ≤ 10, random mean | exact in one step | 30-step drift p99 | max |
|---|---:|---:|---:|
| fold only | 30% | 7.2 | 8.3 |
| fold + correction | 92% | 3.0 | 3.8 |

| start (mean, sigma) | fold only, 50 steps | fold + correction, 50 steps |
|---|---|---|
| 97, 5 | 94.9, 1.0 | 96.7, 4.6 |
| 90, 10 | 87.4, 3.5 | 89.5, 9.2 |
| 3, 4 | 4.5, 0.1 | 3.0, 4.0 |
| 10, 8 | 11.7, 3.2 | 10.0, 8.0 |

The tests in `tests/train/test_categorical_codes.py` pin exactness in the
interior and at sigma 0, the corrected edge cases, and that edge drift settles.
