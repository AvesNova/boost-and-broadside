# The belief diagnostic is aligned; it was leaking teleports — September 27, 2026

Subject: `belief/visible/position_px` read 4.45 px through run 747 while an
offline probe of the same checkpoint measured the same quantity at 0.20 px — a
factor of 22, on the one channel of nine where the two disagreed.
[`next-state-head-audit-sep2026.md`](next-state-head-audit-sep2026.md) named a
one-decision misalignment in `_precompute_belief_diagnostics` as the hypothesis
worth one test.

It is not a misalignment. Evidence:
[`belief-diagnostic-alignment-sep2026.json`](belief-diagnostic-alignment-sep2026.json)
and [`belief-diagnostic-alignment-sep2026-probe.py`](belief-diagnostic-alignment-sep2026-probe.py),
which runs the production rollout and the production diagnostic and then swaps
one filter at a time.

Conditions: `checkpoints/bright-forest-747/step_000221184000.pt` on the `bc`
profile (5v5 Frontline, 1024 px sight, opaque zones, `action_repeat` 1), 48
environments, 3000 warm-up decisions then 128 scored, RTX 4070 Laptop.

## The pairing is correct

Replace the head's forecast with the *believed target vector itself* — the
identity, which for an absolute Fourier-moment predictor is not a zero
prediction vector but the current target — and score it against
`privileged_targets` at three offsets. Own ships are never hidden, so their
believed state is truth exactly, and the number is a pure statement about
indices:

| scored against | mean position error |
|---|---|
| `privileged_targets[t]` | **0.019 px** |
| `privileged_targets[t+1]` | 8.77 px |
| `privileged_targets[t+2]` | 17.61 px |

0.019 px is the bf16 storage and Fourier-ladder round-trip floor. So
`buf.obs[t]` and `buf.privileged_targets[t]` describe the same instant, and the
production statistic — which scores the forecast made at `t` against
`truth_targets[1:]`, i.e. `t+1` — is comparing the right two things. The clean
8.77/17.61 doubling confirms there is no half-step anywhere either.

## What the 22× actually was

| cell | mean | tokens |
|---|---|---|
| production visible-enemy | **0.208 px** | 16,962 |
| the same, contiguity filter dropped | **3.862 px** | 16,998 |
| the same, team id from state rather than observation | 0.208 px | 17,070 |
| production hidden-enemy | 553 px | 13,717 |

0.208 px reproduces the audit probe's 0.20 px through production code, and 553
px reproduces its 527 px. The whole inflation comes from the 36 tokens — 0.2% —
that `transition_contiguous` removes: they average 1726 px each, which is enough
to move the mean of seventeen thousand samples by a factor of eighteen. The
audit's own `enemy_visible_anyalive` cell, which drops the same filter, recorded
3.55 px.

The production cell's distribution says the same thing from the other side: p50
0.181, p90 0.335, p99 0.725, max 6.08 px. There is no tail here at all. A mean
of 4.45 is not reachable from this distribution without samples from a different
process.

So the historical series was measuring a small number of lifecycle
discontinuities that reached it, on a statistic where a handful of them outweigh
everything else. Which ones, in which commit, is not recoverable from the
checkpoint — that series belongs to a run whose observations were also affected
by the [seat-symmetry bug](seat-symmetry-bug-sep2026.md), and run 747 is being
discarded for that reason anyway.

## What changed

`belief/<cell>/position_beyond_legal_frac`: the fraction of the cell whose
one-decision position error exceeds ten times the furthest a ship can legally
travel in one decision (`max_speed · dt · action_repeat`, so 30 px at the
current numbers).

On the visible cell that quantity is unreachable by physics, so it is exactly
the rate at which lifecycle discontinuities are leaking past the transition
filters — 0.0000 on this checkpoint. On the belief cells it is not a leak but a
reading of how far the recursion has drifted — 0.906, consistent with the
audit's finding that the hidden belief collapses onto an age-independent ~500 px
prior.

The point is that the mean is no longer the only thing on the chart. An
eighteen-fold inflation from two tokens in a thousand and a model that has
stopped working look identical in the mean and completely different here.

Alignment is pinned by
`tests/train/test_ppo.py::TestBeliefDiagnosticAlignment`, which asserts the
index agreement above, asserts that the statistic ignores `privileged_targets[0]`
and responds to `privileged_targets[T]`, and asserts that a leaked teleport
raises the new series.
