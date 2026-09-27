# Next-state auxiliary head audit — September 25, 2026

Subject: `checkpoints/bright-forest-747/step_000221184000.pt` — the BC run's final
checkpoint (profile `bc`, update 449, 221.2M steps, live Elo 1017), measured at
commit `30790f6`.

Evidence in this directory: [`next-state-head-audit-sep2026.json`](next-state-head-audit-sep2026.json)
(every aggregate below) and [`next-state-head-audit-sep2026-probe.py`](next-state-head-audit-sep2026-probe.py)
(the measurement; `RUN_DIR` at its top selects the run, and it takes
`envs steps warmup device` as arguments). The rendered tables it produced are in
[`next-state-head-audit-sep2026-tables.txt`](next-state-head-audit-sep2026-tables.txt).

## What was measured

96 environments × 3000 decisions of the checkpoint playing itself in its own
training environment (5v5 Frontline, 1024 px sight, opaque zones, `action_repeat`
1), the first 200 decisions discarded as belief warm-up. At every decision the
head's forecast is taken from the same forward pass the policy acts on, applied to
the composed observation the policy saw, decoded to physical units, and compared
against the authoritative next state. 2.68M scored (ship, step) pairs.

Transitions are filtered as training filters them: alive at both ends, no episode
boundary, and no respawn — Frontline respawn is instantaneous, so a ship is alive
on both sides of a teleport, and leaving those in inflates mean position error
from 0.24 px to 2.2 px and RMSE to 71 px.

Two baselines share the head's information exactly:

* **identity** — nothing changed: the current *believed* state carried forward.
  For a visible ship that is the observation; for a hidden one it is the belief as
  the recursion left it.
* **linear** — position advanced by the believed velocity over one decision
  (`pos + vel·dt`); attitude advanced by the believed angular velocity. Other
  channels fall back to identity.

For the belief cells two further baselines use only the last sighting:
**dead-reckon** (`last_seen_pos + last_seen_vel · age`) and **frozen**
(`last_seen_pos`).

Cells are (ally | enemy) × (visible | belief). **Ally-belief is empty by
construction** — `team_visibility_from_state` gives every team its own ships
unconditionally, so an ally is never hidden and there are no ally beliefs to
audit. Of the enemy tokens, 58% are visible and 42% are beliefs (mean age 3.9 s).

The probe reproduces the training-time `belief/hidden_age_*/position_px` series to
within 10–20% in every age bucket, so what follows describes the signal the run
actually trained on rather than a property of this harness.

## Headline

The head is doing two things well, several things not at all, and one thing
actively worse than doing nothing.

| | ally visible | enemy visible | enemy belief |
|---|---|---|---|
| position | 0.24 px — **4.0× worse than linear** | 0.20 px — 3.9× worse than linear | 527 px — **equal to identity** |
| velocity | 4.78 px/s — **1.5× worse than identity** | 4.69 px/s — 1.7× worse | 83.5 px/s — equal to identity |
| attitude | 0.0037 rad — 15× better than identity | 0.063 rad — 1.05× better | 0.78 rad — equal to identity |
| angular velocity | 0.048 rad/s — 34× better | 2.28 rad/s — **1.23× worse** | 1.99 rad/s — equal to identity |
| health | 1.26 hp — **4.5× worse** | 1.27 hp — 3.5× worse | 3.3 hp — equal to identity |
| power | 0.33 — 1.3× better | 0.64 — 1.3× worse | 7.8 — equal to identity |
| cooldown | 0.0091 s — 1.15× worse | 0.0149 s — 1.19× worse | 0.0017 s — equal to identity |
| shield delay | 0.039 s — 1.17× worse | 0.051 s — 1.17× worse | 0.17 s — equal to identity |
| log index | 0.0061 — 2.7× worse | 0.0089 — 5.0× worse | 0.28 — equal to identity |

Mean absolute error. Position medians are tighter and tell the same story:
ally visible p50 0.17 px (head) against 0.035 px (linear) and 3.07 px (identity).

Three findings follow from this, in descending order of how much they cost.

## 1. The belief is not a forecast — it is a static, decohered prior

For a hidden enemy the head's one-step correction is within 1% of identity on
**every** channel, position and velocity included. The belief is therefore whatever
the recursion has already made it, and the head neither corrects it toward truth —
which is exactly what the re-based labels were introduced to teach — nor dead-reckons
it forward.

What the belief *is* shows up against the last sighting:

| hidden age | n | belief | dead-reckon | frozen |
|---|---|---|---|---|
| ≤ 0.1 s | 16,380 | 6.6 px | **0.27 px** | 6.0 px |
| 0.1–0.5 s | 61,590 | 445 px | **4.8 px** | 22.9 px |
| 0.5–1 s | 52,960 | 519 px | **25.6 px** | 62.8 px |
| 1–2 s | 85,730 | 450 px | **79.9 px** | 138 px |
| 2–5 s | 171,175 | 493 px | **254 px** | 351 px |
| 5–10 s | 138,241 | 657 px | **587 px** | 760 px |
| 10–30 s | 34,296 | 775 px | 863 px | 1130 px |

Mean absolute position error. The learned belief is **93× worse than dead reckoning
at 0.1–0.5 s**, 20× worse at 0.5–1 s, and only overtakes it after about 10 s of
occlusion. 70% of belief tokens are under 5 s old.

The belief does not drift — it collapses, within three to fifteen decisions, onto a
roughly age-independent ~500 px estimate, and then stays there. The mechanism is
visible in the predicted moments. Mean resultant length per position harmonic, hidden
cell (1.0 = certain, 0 = uniform):

```
harmonic      0      1      2      3      4      5      6      7      8      9
period (px) 65536  32768  16384   8192   4096   2048   1024    512    256    128
resultant   0.988  0.974  0.939  0.910  0.820  0.580  0.415  0.216  0.204  0.247
```

Everything below about 2048 px has decohered. Pushing exactly these resultants and
their implied phase spreads back through the project's own decoder reproduces the
measured error (293 px mean, 161 px median, against 527/401 measured — the real
errors are correlated across harmonics, which makes them somewhat worse than this
independent-noise model), and localises the cause: making harmonics ≥ 4 exact
collapses decoded error to 0.8 px, while making harmonics < 4 exact leaves 271 px.
The belief's position uncertainty lives entirely in the 512–4096 px band.

That is a coherent and honest belief — "somewhere in this few-hundred-pixel region"
— and it is what a point estimate with no dynamics model should look like. It is not
worth what it costs: for the first several seconds of occlusion, the most common
case, seeding the same representation from last-seen position plus last-seen
velocity would be one to two orders of magnitude sharper.

Worth noting for interpretation: the decoded point is robust to coarse-harmonic
error, because the dyadic ladder re-snaps at each finer scale. Injecting the
measured coarse-harmonic noise (0.105 RMS) while leaving the finest harmonic exact
decodes to 0.00 px of error. Coarse-harmonic noise hurts the *encoder input* — a
hidden ship's encoded channels are the head's own output, copied in — not the
decoded position.

## 2. Only the channels carrying a σ are learned

Per-dimension MSE in the scaled label space, ally-visible cell, with the head's own
reported variance beside it. `label²` is the null model that predicts zero.

| dim | head | identity | linear | label² | σ² |
|---|---|---|---|---|---|
| position_x harmonic 0 | 0.0143 | 2.5e-8 | 2.7e-11 | 0.505 | — |
| position_x harmonic 4 | 0.0115 | 6.1e-6 | 5.8e-9 | 0.540 | — |
| position_x harmonic 8 | 0.0098 | 1.7e-3 | 1.7e-6 | 0.499 | — |
| **position_x harmonic 9** | **1.4e-4** | 6.7e-3 | 6.7e-6 | 0.498 | 7.6e-4 |
| attitude harmonic 0 | 0.0030 | 4.6e-3 | 0.0145 | 0.514 | — |
| attitude harmonic 2 | 0.0162 | 0.068 | 0.192 | 0.493 | — |
| **attitude harmonic 3** | **0.0020** | 0.214 | 0.474 | 0.500 | 2.1e-3 |
| velocity x / y | 0.0071 / 0.0024 | 0.019 / 0.019 | — | 9.6 / 10.2 | 0.021 / 0.016 |
| angular velocity | 6.2e-4 | 1.005 | — | 1.504 | 7.4e-4 |
| health | 2.9e-4 | 1.2e-4 | — | 0.457 | 9.4e-4 |
| power | 3.4e-5 | 5.5e-5 | — | 0.019 | 1.2e-4 |
| cooldown | 0.0700 | 0.0488 | — | 0.0713 | 0.0834 |
| shield delay | 0.0075 | 0.0076 | — | 1.055 | 9.7e-3 |
| log index | 1.1e-4 | 1.9e-4 | — | 0.295 | 3.3e-4 |

`FourierMomentPredictor` reports one spread, for its finest harmonic only
(`uncertainty_gather` puts it at index `n-1` of each block), and the argument for
that was gradient share: nine of ten harmonics need no σ because the moment's
magnitude already carries the confidence. The measurement says those nine harmonics
are, for a ship in plain sight, **10× to 500,000× worse than doing nothing**, at a
uniform noise floor of about 0.105 RMS per dimension, while the one harmonic that
does carry a σ is 50× better than identity. The same split appears in attitude:
harmonics 0–2 are within a factor of 4 of identity, harmonic 3 beats it 110×.

The floor is flat across harmonics and across visibility (0.010–0.014 visible,
0.0086–0.016 hidden at the coarse end), which is what an untrained output looks
like, not a hard prediction problem — identity proves the targets are nearly
constant over one decision. The σ-bearing channels are trained; the rest are
starved, exactly the failure the design note anticipated in the other direction.

Two channels are worse than *both* baselines in an absolute sense:

* **cooldown** — head 0.0700 against identity 0.0488 and a null model of 0.0713.
  The head is at chance on a quantity that, for an ally, is a deterministic function
  of the pending shoot action already in the observation. It has been degrading
  monotonically through the run (0.058 → 0.071 in the W&B per-dimension series).
* **health** — 2.4× worse than identity for allies, 1.8× for visible enemies. Damage
  is rare and unpredictable; the head pays for guessing rather than reporting no
  change.

## 3. Ally-versus-enemy separates known actions from unknown ones, as it should

The ally/enemy split is the cleanest signal in the audit, and it is the one place
where the head behaves exactly as designed:

* **angular velocity**: 0.048 rad/s for an ally (34× better than identity) against
  2.28 rad/s for a visible enemy (23% *worse* than identity). An ally's commanded
  turn is in the observation; an enemy's is not, and a decision-rate turn command is
  essentially unpredictable. The head correctly exploits the one and correctly fails
  at the other — but it should fall back to identity there rather than beat it by
  -23%.
* **attitude**: 15× better than identity for an ally, 1.05× for a visible enemy. Same
  cause, same conclusion. The linear (extrapolate by angular velocity) baseline is
  *worse* than identity for both, which is a real fact about the plant: heading
  change is commanded, not continued.
* **position**: essentially identical for ally and enemy (0.24 px and 0.20 px), and
  in both cases about 4× worse than `pos + vel·dt`. Position over one decision is a
  matter of integrating known velocity, which needs no knowledge of intent, so the
  absence of a split is expected. Being beaten 4× by two lines of arithmetic is not.

## Calibration of the reported σ

Within a factor of a few, and correctly ordered between visible and hidden — the
head does know when it is blind. For the hidden cell the finest position harmonic
reports σ² 0.443 against a realised 0.440, which is as good as calibration gets.

For visible ships it is systematically **conservative by 3–6×**: position finest
harmonic 7.6e-4 reported against 1.4e-4 realised, velocity 0.021 against 0.0071,
health 9.4e-4 against 2.9e-4. Attitude (2.1e-3 vs 2.0e-3), angular velocity
(7.4e-4 vs 6.2e-4), cooldown and shield delay are close. Since the Gaussian
likelihood weights the gradient on the mean by 1/σ², an inflated σ on the visible
tokens throttles learning precisely where the learnable dynamics are — a
self-limiting loop that plausibly explains why visible position sits at
linear-extrapolation parity rather than below it.

Overconfidence appears only in the hidden cell's absolute channels: velocity
reports σ² 5.5/2.8 against realised 8.0/4.9, angular velocity 0.80 against 1.31.

## An anomaly in the `belief/visible/position_px` series

The training-time series reads 4.45 px for visible enemies. This probe measures the
same quantity, on the same checkpoint, at 0.20 px — a factor of 22 — while every
other channel of that series agrees with the probe to within 20% (velocity 4.31 vs
4.69, attitude 0.072 vs 0.063, angular velocity 2.48 vs 2.28, power 0.63 vs 0.64,
cooldown 0.017 vs 0.015, log index 0.0098 vs 0.0089).

Ruled out: bf16 storage of the position target vector (full-ladder round-trip error
is 0.012 px mean, 0.06 px max, and the difference of two independently quantised
states one step apart is 0.017 px); respawn contamination (including respawns raises
the probe to 2.2 px, not 4.5); and the alive filter (no effect — Frontline respawn is
instant, so `dead_frac` is 0.001).

Hypothesis worth one test: a one-decision misalignment in
`_precompute_belief_diagnostics`' pairing of `rollout_predictions[t]` with
`privileged_targets[t+1]`. Position is the only channel whose per-decision change
(≈3 px, the identity error) is large relative to the head's own error (0.2 px), so
such a slip would inflate position by roughly an order of magnitude and stay
invisible in every other channel — which is the pattern. It would also explain why
the series has been pinned at 4.1–4.5 px since 30M steps while `next_state_visible/*`
kept improving: a constant offset, not a plateau. Test by logging the same statistic
against `privileged_targets[t]` and `[t+2]` for one update.

**Tested and refuted**, September 27:
[`belief-diagnostic-alignment-sep2026.md`](belief-diagnostic-alignment-sep2026.md).
The pairing is correct to 0.019 px. The inflation is the 0.2% of tokens that
`transition_contiguous` removes, averaging 1726 px each — which is this
document's own `enemy_visible_anyalive` cell at 3.55 px, dismissed above as "2.2
px, not 4.5" from a different cell.

## Harness bug found while measuring this

**Fixed**, September 27, in `TensorEnv.step`; see
[`tests/env/test_spawn_lifecycle.py`](../../tests/env/test_spawn_lifecycle.py).
The description below is of the defect as it stood at `30790f6`, and every
number in this document was measured with the probe's own workaround for it.

`TensorEnv.step()` never clears `state.ship_spawned`; only `YemongEnvWrapper` does,
at the start of each decision ([wrapper.py:361](../../src/boost_and_broadside/env/wrapper.py#L361)).
`team_visibility_from_state` ORs that latch into both teams' sight when
`spawn_reveal` is on, so once it is set it never clears and **every ship stays
permanently visible to both teams**.

Training is unaffected — it steps through the wrapper. Everything that drives
`TensorEnv` directly is affected and has been measuring a fog-free world:

* the live Elo evaluator ([elo_eval.py:230](../../src/boost_and_broadside/train/rl/elo_eval.py#L230))
  — so this run's live Elo, and the scripted anchor it is calibrated against, are
  omniscient-world numbers;
* every `MatchRunner` consumer: `evaluation/tournament.py`, `noise-calibration`,
  `ar-report`, `capture`, `feature-stats`.

Measured with the latch left alone: 100% enemy visibility and zero belief tokens
over 3000 decisions. Cleared the way the wrapper clears it: 58% visible, 42%
beliefs. The probe in this directory clears it explicitly; a fix belongs in
`TensorEnv.step`, which is the level at which "a decision has passed" is known.

## What the audit suggests, in order

1. **Seed the belief from dead reckoning, or make the head's hidden-ship update
   learn it.** A one-to-two order of magnitude position error, on 70% of belief
   tokens, against `last_pos + last_vel · age`.
2. **Give the coarse Fourier harmonics a σ**, or otherwise fix their gradient share.
   Nine of ten position harmonics and three of four attitude harmonics are worse
   than doing nothing on a visible ship, and they are what the trunk reads for a
   hidden one.
3. **Let identity be reachable.** A head that cannot express "no change" cheaply
   pays on cooldown, health, shield delay and log index, and on enemy angular
   velocity. A gated or residual update would make the no-change prediction free.
4. **Deflate the visible-token σ**, or condition it on visibility explicitly. Being
   3–6× conservative there throttles the gradient exactly where the dynamics are
   learnable.
5. **Check the `belief/visible/position_px` pairing** before reading that series
   again.
6. **Fix the spawn-reveal latch** in `TensorEnv.step`, then re-read every Elo number
   this run produced.
