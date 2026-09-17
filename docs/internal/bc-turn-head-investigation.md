# Behaviour cloning: why the turn head does not close

Run `icy-energy-741` (W&B `xs3whcqt`), profile `bc`, source commit `85f2bf3`,
checkpoint `checkpoints/icy-energy-741/step_000167608320.pt` (update 341,
167.6M steps). All measurements were taken on 2026-09-17 on the dev
RTX 4070 Laptop (8 GB) at `num_envs=128`, `microbatch_tokens=12288`,
`num_steps=128`, `num_ships=8`. The checkpoint was never modified.

Two sessions are recorded here. Session 1 (Exp 1–6) established *what* the
residual is. Session 2 (Exp 7–13) establishes *where it comes from*, and its
conclusion supersedes session 1's "likely interpretations".

## Executive summary

**The residual BC loss is a bearing-precision deficit in the shared trunk, and
the deficit is caused by the positional representation — not by the action
head, not by capacity, not by optimisation, not by recurrence, and not by data
volume.**

The evidence chain, all on fresh held-out rollouts under the production mask
`bc_valid & actor_mask & alive`:

1. **The action head is exonerated.** Append the teacher's true
   `sin/cos` bearings to the *frozen* final latent and fit a small turn head:
   held-out turn KL falls `0.511 → 0.036`. From the bearings *alone*, with no
   latent at all, it reaches `0.011`. The map from bearing to the teacher's turn
   distribution is trivially learnable; the only thing missing from the latent
   is the bearing (Exp 8, Exp 11).
2. **The trunk never acquires the bearing to the needed precision, and never
   plateaus.** Layerwise probes on the frozen checkpoint reduce frontline-bearing
   error monotonically from `17.0°` (encoder output) to `7.0°` median at the
   final latent — still improving at the last layer. The tail is worse than the
   median suggests: p90 `76°` (Exp 7).
3. **That error quantitatively explains the observed KL.** A turn head *fitted*
   on a bearing corrupted to median `7.7°` / p90 `18.9°` scores held-out turn KL
   `0.477`; the checkpoint scores `0.511`. To reach KL `0.14` the bearing must be
   known to median `1.9°` / p90 `4.7°` (Exp 12). This replaces session 1's
   "≈11° of equivalent Gaussian uncertainty", which was an assumed-hedging
   estimate rather than a measured one.
4. **The representation, not the capacity, is the binding constraint.** A
   ~200k-parameter sum-pooling probe over *ego-relative* tokens — displacement
   rotated into the ego frame plus an explicit unit vector — trained on 49k
   tokens reaches median `5.4°` / p90 `21.6°`. The 1.945M-parameter trunk,
   trained on 167.6M environment steps, reaches `7.0°` / p90 `76°`. The **same
   probe** fed the trunk's own absolute-Fourier encoding (with the ego's absolute
   position handed over for free) reaches only `29.4°` / p90 `129°`. Withholding
   just the unit vector from the ego-relative cell costs `5.4° → 13.8°`
   (Exp 9, Exp 10).

**The mechanism.** The teacher's bearing is a masked weighted sum of *pairwise*
unit vectors,

```
force_i = Σ_j w_j · unit(p_j − p_i)          bearing_i = angle(force_i) − attitude_i
```

over all 8 ships and all zones (`frontline_strategy`, plus `predict_interception`
for the combat term). Self-attention cannot form that sum: a value vector `v_j`
depends on `j` alone, so `Σ_j a_ij v_j` can never contain `unit(p_j − p_i)`,
which depends on the *pair*. And the trunk has no relative positional encoding
of any kind — position enters only as an **absolute** 8-frequency Fourier
expansion of world `x` and `y` per token
(`features.py:829-844`), over a 16384-px world in which play occupies a
2600-radius disc, and attention is plain dot-product with no positional bias
(`attention.py:40-140`). The trunk must therefore approximate the entire
relational reduction inside FFNs acting on pooled vectors. That is exactly why
depth helps monotonically but slowly, and why a network ten times smaller with
the pairwise term supplied outright beats it on a rounding error of the data.

**Honest limit of the demonstration.** Feeding the sum-pooling probe's
*predicted* bearing (median `5.8°`) into a turn head moves held-out turn KL only
`0.619 → 0.556` against its own latent-only baseline. The link function is steep
enough that the median is the wrong statistic — the p90 is what sets the KL. A
better representation is therefore *necessary but not by itself sufficient*: the
recommendation below is an end-to-end architecture change, where the trunk gets
167M steps to exploit it, not a bolt-on probe (Exp 11).

**Session 1 facts that still stand.** The old `0.81` was an evaluator missing
`actor_mask`; the residual is a turn-head, and within it a direction, phenomenon;
the policy under-turns and hedges; power and shoot are solved; `p_team ≡ 0`;
next-state and entropy are orthogonal and small; fp32/bf16 and rollout-start
hidden state are noise; fixed-rollout fitting is memorisation (fresh-rollout KL
`2.6`). None of session 2's measurements contradict these.

**Confirmed defect (was "unresolved").** `loss/behavioral_cloning_kl` is the mean
over *all four update epochs*, not a pre-update value. `accum_scalar` is created
once per `_update_epochs` (`ppo.py:2586`) and `bc_kl` sits in the `_additive`
table (`ppo.py:2711`), so every micro-batch of every epoch contributes. With
Exp 5's measured descent (`0.4335 →``0.3303` over four epochs) this fully accounts
for the run logging ≈`0.316` where the checkpoint measures `0.44–0.57`.

**Remaining uncertainties.** Whether an end-to-end run with relative-position
attention actually reaches p90 < 5°; the ~`0.08` nat left/right gap (Exp 13
rules out a geometric cause); whether the probe is a tight lower bound on the
error the action head actually suffers.

## Session 1 summary (Exp 1–6)

**Established facts (direct measurement).**

1. The `0.34` vs `0.81` discrepancy is a **masking defect in one of the
   evaluators, not a property of the checkpoint**. On one identical rollout,
   the same weights and the same stored teacher labels give
   turn KL `0.4393` over the production mask (`bc_valid & actor_mask & alive`,
   65 536 tokens), `0.7313` over `alive` alone (117 663 tokens) and `0.8330`
   over all tokens (131 072 tokens). The denominators are the giveaway:
   131 072 is exactly 2× the production denominator, because `ego_pass` makes
   only team 0 actor tokens while the scripted teacher labels *every* ship.
   Opponent-team tokens carry turn KL `1.0983`, dead tokens `1.7254`.
2. Precision and recurrent state are **not** the cause. bf16 vs fp32
   evaluation differs by `≤0.0015` KL; stored rollout-start hidden vs zero
   hidden differs by `≤0.010` KL (and is bit-identical on the first rollout,
   whose stored hidden legitimately *is* zero).
3. Genuine rollout-to-rollout variation is large: over 12 consecutive rollouts
   from one reset, total BC KL runs `0.390 → 0.654 → 0.470` (steady state
   ≈ `0.44–0.50`), and the 8 env shards inside a single rollout span
   `0.31–0.58`. So a single aggregate number is worth ±0.1 at best.
4. Residual KL is a turn-head phenomenon and, within the turn head, a
   **direction** phenomenon. Exact chain-rule split of the turn KL:
   coarse straight/left/right `0.3845`, sharp-vs-normal `0.0548`. The teacher
   puts **zero** mass on both air-brake actions, always.
5. The dominant error is **under-turning**: where the teacher's argmax is a
   normal left/right turn (33.2% of tokens), the policy's argmax is
   `GO_STRAIGHT` more often than it is the correct turn (17.8% of all tokens
   vs 12.7%). Conditional turn KL is `0.63` (left) / `0.77` (right) vs `0.30`
   (straight) and `0.31–0.34` (sharp). Policy turn entropy `0.772` exceeds
   teacher turn entropy `0.473`: the policy hedges.
6. The teacher's turn head is **exactly reconstructable from one scalar**. For
   the 64.3% of tokens at `alpha > 0.99` (pure frontline navigation), rebuilding
   the distribution from the recorded bearing reproduces the stored
   `expert_probs` to a mean max-abs error of `8.4e-4` — which is the bf16
   storage noise of the buffer, not a modelling gap. Imitation is therefore a
   *bearing regression* problem, nothing more.
7. That bearing must be resolved to a couple of degrees. Feeding the teacher a
   perturbed bearing on the same tokens: `±0.02 rad` (1.1°) costs KL `0.11`,
   `±0.05 rad` (2.9°) costs `0.48`. The measured policy KL on those tokens is
   `0.4723`. Under a Gaussian bearing uncertainty the equivalent is
   `σ ≈ 0.20 rad` (11°) → KL `0.504`.
8. Next-state prediction is **not** interfering, by two independent measures.
   Gradients: trunk shares BC `0.885`, next-state `0.065`, windowed next-state
   `0.006`, with `cos(bc, next_state) = -0.022` on the trunk — orthogonal, not
   conflicting (BC trunk gradient norm `16.25` vs next-state `1.05`). Training:
   with `next_state_coef = 0`, matched seed, data and optimizer steps, turn KL
   after 4 epochs is `0.3743` vs `0.3303` with it on, and after 12 epochs
   `0.1854` vs `0.1958` — no advantage either way beyond noise.
9. `p_team` (the team-target blend inside the legacy combat controller) is
   **identically 0** in this configuration: its ramp is
   `team_target_distance_ramp=(60, 100)` world units and the mean nearest-enemy
   distance is 710–1100. The team-target-blending hypothesis is dead *for this
   run*; the global computation that actually matters is
   `frontline_strategy().bearing` (`alpha` mean 0.74, 64% of tokens at
   `alpha = 1`).

**Likely interpretations — superseded by session 2.** Kept for the record. The
direction of these guesses was right (angular precision on a global bearing) but
the attribution was not: they left capacity, representation and sample
complexity open, and Exp 9/10 settle it on representation.

* The limiting factor is **angular precision on a globally-computed bearing**,
  not action-head capacity, not optimisation interference, not recurrence.
  The teacher's `turn_angle_ramp=(0.03, 0.12)` maps a 3.9°-wide window of
  |bearing| onto a probability swing of 0.068 → 0.95, on top of a hard
  left/right sign step at zero. Yemong reproduces the teacher's bearing to
  roughly ±11° of effective uncertainty; the ramp converts that into ~0.45 nats.
* The turn KL profile across |bearing| is exactly the ramp's sensitivity
  profile: `0.28` below the ramp, peaking at `0.70–0.80` in `[0.09, 0.2)` rad
  where the ramp saturates, and falling to `0.15–0.24` beyond 0.7 rad where the
  teacher is saturated and sign is easy. That is a signature of an estimator
  error passed through a steep link function, not of a missing concept.
* Frontline-bearing tokens (`alpha ≈ 1`, KL `0.472`) are harder than
  personal-intercept tokens (`alpha ≈ 0`, KL `0.375`), consistent with the
  frontline bearing requiring a zone/fleet-wide reduction, but the gap (0.10)
  is much smaller than the overall level, so *both* bearings are imprecise.

**Unresolved at the end of session 1** (both since closed — see the executive
summary):

* Why the online logged `loss/behavioral_cloning_kl ≈ 0.316` sits below the
  pre-update checkpoint value measured here (`0.44–0.50` in steady state,
  `0.39` immediately after a reset). Exp 5 gives the likely mechanism rather
  than a proof: four epochs of the production update move turn KL from `0.4335`
  to `0.3303` **on the rollout being scored**, and the logged metric averages
  over all four epochs, so it reports a policy already partly fitted to its own
  scoring data. Confirming this needs one logging change (item 5 below).
* Whether the bearing error is a capacity limit, an input-representation limit
  (toroidal geometry, zone tokens), or a sample-complexity limit. No
  architecture cell was run this session — deliberately, since the evaluator had
  to be trusted first.

## Experiment log

### Exp 1 — evaluator reconciliation on identical stored samples
*Question:* do bf16/fp32, stored/zero hidden, or aggregation explain `0.34` vs
`0.81`? *Method:* `scratchpad/bc/exp1_reconcile.py` — build the BC trainer at
the checkpoint, collect 3 rollouts, and evaluate each with the production mask
under three paths, aggregating numerators and denominators separately.
*Runtime:* ~3 min. *Sample:* 65 536 valid tokens per rollout.

| rollout | path | den | KL total | turn | power | shoot | H(teacher) |
|---|---|---|---|---|---|---|---|
| 0 | bf16 / stored | 65536 | 0.3904 | 0.3910 | −0.0006 | 0.0000 | 0.6994 |
| 0 | fp32 / stored | 65536 | 0.3890 | 0.3896 | −0.0006 | 0.0000 | 0.6994 |
| 0 | fp32 / zero   | 65536 | 0.3890 | 0.3896 | −0.0006 | 0.0000 | 0.6994 |
| 1 | bf16 / stored | 65536 | 0.4418 | 0.4407 | −0.0001 | 0.0011 | 0.7077 |
| 1 | fp32 / zero   | 65536 | 0.4475 | 0.4464 | −0.0001 | 0.0011 | 0.7077 |
| 2 | bf16 / stored | 65536 | 0.5361 | 0.5144 | 0.0124 | 0.0092 | 0.8402 |
| 2 | fp32 / zero   | 65536 | 0.5247 | 0.5028 | 0.0124 | 0.0096 | 0.8402 |

Per-shard (8 shards, correct num/den aggregation), rollout 0:
`0.424 0.336 0.328 0.314 0.455 0.426 0.371 0.458`.

*Interpretation:* dtype and hidden state are noise at this scale; shard spread
is not. Rollout 0's stored-vs-zero identity is expected (the first rollout after
a reset stores the zero hidden). The small negative power KL is the bf16
`expert_probs` storage (see Defects). *Next:* measure the distribution drift and
test mask variants.

### Exp 2 — KL vs rollout index, plus per-token dump
*Question:* can distribution variation alone produce 0.34 → 0.81? *Method:*
`scratchpad/bc/exp2_drift.py` — 12 consecutive rollouts from one reset under the
frozen checkpoint, env-ordered evaluation (no shuffling), teacher internals
recorded on rollouts 0 and 11, per-token dump written for rollout 11.
*Runtime:* ~4 min. *Sample:* 65 536 valid tokens × 12.

| rollout | KL total | turn | power | shoot | H(teacher) |
|---|---|---|---|---|---|
| 0 | 0.390 | 0.391 | −0.001 | 0.000 | 0.699 |
| 1 | 0.434 | 0.433 | −0.000 | 0.001 | 0.705 |
| 2 | 0.544 | 0.523 | 0.011 | 0.010 | 0.841 |
| 3 | **0.654** | 0.609 | 0.033 | 0.012 | 0.924 |
| 4 | 0.590 | 0.551 | 0.019 | 0.020 | 0.880 |
| 5 | 0.553 | 0.510 | 0.015 | 0.027 | 0.891 |
| 6 | 0.488 | 0.447 | 0.016 | 0.026 | 0.891 |
| 7 | 0.491 | 0.449 | 0.016 | 0.026 | 0.877 |
| 8 | 0.446 | 0.413 | 0.012 | 0.021 | 0.847 |
| 9 | 0.462 | 0.434 | 0.008 | 0.020 | 0.852 |
| 10 | 0.502 | 0.464 | 0.012 | 0.026 | 0.863 |
| 11 | 0.470 | 0.439 | 0.009 | 0.022 | 0.842 |

Rollout 0 `mean_closest_dist=1101`, `alpha=1.00`, `p_team=0.000`;
rollout 11 `mean_closest_dist=710`, `alpha=0.744`, `p_team=0.000`.

*Interpretation:* real variation covers `0.39–0.65`, so `0.34` is reachable
(early/low-contact states) but `0.81` is not — something else produced it.
`p_team = 0` kills the team-target-blend hypothesis for this config. *Next:*
test mask variants directly on the dumped tokens.

### Exp 3 — turn-head decomposition and teacher-variable stratification
*Question:* where in the turn head, and in what states, does the KL live?
*Method:* `scratchpad/bc/exp3_turn.py`, offline on `dump_rollout11.pt`.
*Runtime:* ~20 s. *Sample:* 65 536 valid tokens. Full output:
`scratchpad/bc/exp3_out.txt`.

Mean turn KL `0.4393`; teacher turn entropy `0.4728`; policy turn entropy
`0.7722`.

| action | teacher mass | teacher argmax % | policy mass | KL contribution | conditional KL | argmax agree % | mean q@teacher argmax |
|---|---|---|---|---|---|---|---|
| GO_STRAIGHT | 0.3961 | 40.23 | 0.4590 | 0.0399 | 0.2961 | 86.77 | 0.6922 |
| TURN_LEFT | 0.1514 | 14.35 | 0.1444 | 0.1152 | 0.6317 | 40.59 | 0.3933 |
| TURN_RIGHT | 0.1908 | 18.88 | 0.1506 | **0.1720** | **0.7689** | 36.19 | 0.3632 |
| SHARP_LEFT | 0.1090 | 11.01 | 0.1084 | 0.0470 | 0.3351 | 79.49 | 0.6675 |
| SHARP_RIGHT | 0.1528 | 15.54 | 0.1376 | 0.0653 | 0.3063 | 81.00 | 0.6901 |
| AIR_BRAKE | 0.0000 | 0.00 | 0.0000 | −0.0000 | — | — | — |
| SHARP_AIR_BRAKE | 0.0000 | 0.00 | 0.0000 | −0.0000 | — | — | — |

Air-brake actions carry **no teacher probability at all** (`p_air_brake` and
`p_sharp_air_brake` are hard zeros in `_compute_action_probs`); the policy has
learned to put no mass there. Nothing should be read into them.

Confusion matrix (% of all valid tokens; rows teacher argmax, cols policy argmax):

|  | straight | left | right | sharp_L | sharp_R |
|---|---|---|---|---|---|
| straight | **34.90** | 2.45 | 2.13 | 0.43 | 0.31 |
| left | **7.25** | 5.82 | 0.36 | 0.74 | 0.17 |
| right | **10.53** | 0.53 | 6.83 | 0.24 | 0.75 |
| sharp_L | 0.68 | 0.85 | 0.09 | **8.75** | 0.64 |
| sharp_R | 0.60 | 0.15 | 1.33 | 0.87 | **12.59** |

Error characterisation:
* **straight-vs-turn** dominates: teacher-turn → policy-straight is 17.8% of all
  tokens; teacher-straight → policy-turn is only 5.3%. The policy under-turns.
* **normal-vs-sharp** is small: 0.0548 of 0.4393 by the chain rule; sharp
  argmax agreement is 79–81%.
* **direction (left/right)** errors: the policy puts more mass on the opposite
  side on 21.3% of tokens, and those tokens carry mean turn KL `0.853`. Outright
  left↔right argmax flips are rarer (0.36 + 0.53 + 0.17 + 0.09 + 0.75 + 1.33 ≈
  3.2% of tokens).

Stratification by the teacher's own intermediates (effective bearing = the
frontline bearing when `alpha > 0.5`, else the combat bearing):

| \|bearing\| (rad) | tokens | % | KL | KL direction | KL sharp | H(teacher) |
|---|---|---|---|---|---|---|
| [0, 0.03) | 9887 | 15.1 | 0.3128 | 0.3115 | 0.0013 | 0.2588 |
| [0.03, 0.06) | 8971 | 13.7 | 0.2824 | 0.2790 | 0.0034 | 0.5130 |
| [0.06, 0.09) | 7008 | 10.7 | 0.4168 | 0.4053 | 0.0115 | 0.7386 |
| [0.09, 0.12) | 5456 | 8.3 | **0.7020** | 0.6798 | 0.0222 | 0.6258 |
| [0.12, 0.2) | 8952 | 13.7 | **0.8028** | 0.7492 | 0.0537 | 0.4014 |
| [0.2, 0.3) | 4961 | 7.6 | 0.5646 | 0.4450 | 0.1196 | 0.5212 |
| [0.3, 0.42) | 3085 | 4.7 | 0.5760 | 0.3091 | 0.2670 | 0.7244 |
| [0.42, 0.7) | 4021 | 6.1 | 0.5059 | 0.2357 | 0.2702 | 0.4021 |
| [0.7, 1.2) | 3900 | 6.0 | 0.1964 | 0.1441 | 0.0523 | 0.4040 |
| [1.2, 2) | 4613 | 7.0 | 0.1536 | 0.1322 | 0.0214 | 0.3936 |
| [2, 3.2) | 4682 | 7.1 | 0.2439 | 0.2307 | 0.0132 | 0.3870 |

`turn_angle_ramp = (0.03, 0.12)` and `sharp_turn_angle_ramp = (0.24, 0.42)`:
the direction KL peaks exactly across the turn ramp and the sharp-vs-normal KL
peaks exactly across the sharp ramp (`0.27` in `[0.3, 0.7)`). Tokens inside the
sharp ramp average KL `0.5578` vs `0.4254` outside both ramps.

| stratum | KL |
|---|---|
| `alpha = 0` (pure personal intercept, 18.6%) | 0.3745 |
| `alpha = 1` (pure frontline navigation, 64.3%) | 0.4723 |
| front-vs-combat bearing disagreement < 0.05 rad | 0.3468 |
| disagreement 0.05–0.2 rad | 0.5250 |
| disagreement > 2 rad | 0.4275 |
| nearest enemy < 100 | 0.3774 |
| nearest enemy 600–1000 | 0.5025 |
| speed 20–60 | 0.2950 |
| speed 100–140 | 0.4821 |
| episode step < 500 | 0.3953 |
| episode step > 6000 | 0.4469 |

No stratum is clean. Nothing gets below ~0.28, and nothing exceeds ~0.80. This
is a broad precision deficit, not a localized discontinuity: target-identity
changes, team blending and hard target selection do **not** produce isolated
spikes here (and `p_team ≡ 0` means hard team-target selection is not even
active).

Mask variants on the identical tokens — **the reconciliation**:

| mask | denominator | turn KL |
|---|---|---|
| `bc_valid & actor_mask & alive` (production) | 65 536 | **0.4393** |
| `actor_mask & alive` | 65 536 | 0.4393 |
| `alive` only | 117 663 | **0.7313** |
| no mask at all | 131 072 | **0.8330** |
| `alive & ~actor` (opponent team only) | 52 127 | 1.0983 |
| dead tokens only | 13 409 | 1.7254 |

*Interpretation:* an evaluator that forgets `actor_mask` reports ~0.73–0.83 on a
checkpoint whose true value is ~0.44. That reproduces the reported `0.814`
almost exactly. *Next:* quantify how precisely the bearing must be known.

### Exp 4 — BC vs next-state gradient relationship
*Question:* is the next-state auxiliary interfering with BC? *Method:*
`scratchpad/bc/exp4_grad.py` — the repo's own `top_level` gradient diagnostics,
4 diagnosed minibatches, after 3 burn-in rollouts so the env distribution is the
steady state. *Runtime:* ~2 min. Raw: `scratchpad/bc/exp4_grad.json`.

| quantity (last diagnosed minibatch) | value |
|---|---|
| `grad_norm/trunk_top_level/bc` | 16.25 |
| `grad_norm/trunk_top_level/next_state` | 1.05 |
| `grad_norm/trunk_top_level/windowed_next_state` | 0.098 |
| `grad_share/trunk_top_level/bc` | 0.897 |
| `grad_share/trunk_top_level/next_state` | 0.058 |
| `grad_cos/trunk_top_level/bc__next_state` | **−0.022** |
| `grad_cos/trunk_top_level/bc__windowed_next_state` | −0.020 |
| `grad_cos/trunk_top_level/next_state__windowed_next_state` | 0.742 |
| `grad_cos/trunk_top_level/entropy__bc` | 0.029 |

*Interpretation:* the earlier claim that next-state contributes a *substantial*
trunk gradient does not hold at this checkpoint — it is 6% of the trunk
gradient, essentially orthogonal to BC (|cos| ≈ 0.02). Removing it would free
almost nothing and would cost the trunk its dynamics signal. The entropy bonus
is likewise negligible (cos 0.03, and `entropy_coef = 0.005`). *Next:* confirm
with a matched training ablation (Exp 5) rather than gradients alone.

### Exp 5 — matched offline fitting, `next_state_coef` 1 vs 0
*Question:* does the next-state auxiliary slow BC down in practice, not just in
gradient geometry? *Method:* `scratchpad/bc/exp5_ns.py` — two independent
trainers from the same checkpoint and the same seed, each burning in 2 rollouts
(so the env distribution matches), GAE computed, then `next_state_coef` set to
1.0 or 0.0 as the **only** difference, then three calls to the production
`_update_epochs` (4 epochs × 32 minibatches each = 12 epochs over the same
frozen rollout). Turn KL re-measured on that rollout after every call, plus one
fresh on-policy rollout at the end. *Runtime:* ~34 s of optimisation per cell
(~4 min total with collection). *Sample:* 65 536 valid tokens per measurement.

| epochs on the frozen rollout | 0 | 4 | 8 | 12 | fresh on-policy rollout |
|---|---|---|---|---|---|
| `next_state_coef = 1` (production) | 0.4335 | 0.3303 | 0.2246 | 0.1958 | 2.5885 |
| `next_state_coef = 0` | 0.4335 | 0.3743 | 0.2320 | 0.1854 | 3.0050 |

*Interpretation:* turning the next-state auxiliary off does **not** speed BC up.
After 4 epochs the BC-only cell is *behind* (0.3743 vs 0.3303); by 12 epochs the
two are within 0.01 of each other, which is inside the per-shard noise. This
matches the gradient geometry of Exp 4 (6% of trunk gradient, cosine −0.02) and
closes the auxiliary-interference hypothesis: **next-state prediction is not
what is limiting the turn head**, and removing it buys nothing.

Two side findings, both important:

* **Repeated fitting on one rollout is fitting, not learning.** Turn KL on the
  frozen batch falls 0.43 → 0.20 in 12 epochs, reproducing the earlier
  "frozen-batch KL collapses" observation — while the very next on-policy
  rollout scores **2.59** (and 3.01 for the BC-only cell), six times worse than
  the checkpoint it started from. The frozen-batch collapse is therefore not
  evidence that the architecture can fit the teacher; it is evidence that 128
  envs × 128 steps is a small enough batch to memorise. Any conclusion drawn
  from long offline optimisation on a fixed dataset — including the earlier
  `0.343 → 0.008` result — has to be read with this in mind.
* **The online metric is measured mid-update.** Four epochs move turn KL from
  0.4335 to 0.3303 on the rollout being scored. `loss/behavioral_cloning_kl` is
  accumulated across every minibatch of all four epochs
  (`ppo.py:2700-2745`), so it reports roughly the *average over that descent*,
  not the checkpoint's value. That is the most likely reason the run logs ≈0.316
  where the checkpoint measures ≈0.44 pre-update.

### Exp 6 — teacher reconstruction and the price of a bearing error
*Question:* is the teacher observable/reproducible, and how precise must the
bearing be? *Method:* `scratchpad/bc/exp6_angle.py` — rebuild the teacher's turn
distribution analytically from the recorded bearing using
`StochasticAgentConfig`'s ramps, on the 42 158 `alpha > 0.99` tokens, then
perturb the bearing. *Runtime:* ~15 s.

Reconstruction error vs the stored `expert_probs`: mean max-abs `8.36e-4`,
max `3.26e-2` — consistent with bf16 buffer storage, i.e. **exact**. The turn
teacher is a deterministic function of one scalar bearing; nothing about it is
unobservable in principle.

| perturbation | resulting KL |
|---|---|
| bearing +0.005 rad (0.29°) | 0.0192 |
| bearing +0.01 rad (0.57°) | 0.0422 |
| bearing +0.02 rad (1.15°) | 0.1046 |
| bearing +0.03 rad (1.72°) | 0.1863 |
| bearing +0.05 rad (2.87°) | 0.4836 |
| bearing +0.08 rad (4.58°) | 1.2320 |
| bearing +0.20 rad (11.5°) | 3.9259 |
| random-sign ±0.05 rad | 0.4825 |
| Gaussian σ = 0.05 rad, marginalized | 0.0567 |
| Gaussian σ = 0.10 rad, marginalized | 0.2045 |
| Gaussian σ = 0.20 rad, marginalized | 0.5042 |
| **measured policy, same tokens** | **0.4723** |

*Interpretation:* the entire residual is worth ~3° of systematic bearing error,
or ~11° of bearing *uncertainty* if the policy is hedging optimally (which its
elevated turn entropy, 0.772 vs 0.473, says it is). The teacher is not complex;
it is *steep*. Any imitator that estimates the frontline bearing to ±10° will
report exactly this KL.

### Exp 7 — layerwise bearing probes on the frozen checkpoint

*Question:* where in the trunk does the teacher's bearing appear, improve,
degrade or plateau? *Method:* `exp7_collect.py` freezes the checkpoint, burns in
3 rollouts to reach the steady-state distribution, then collects 3 more while
forward hooks capture every ship token's activation at eight tap points. Probes
(linear, and a 2×512 GELU MLP) are fitted on one set of rollouts to predict
`(sin θ, cos θ)` and scored on an **independent** set collected from a different
seed — no probe ever sees its own test states. Loss is `1 − cos(error)`; error is
reported in degrees because that is the unit the teacher's ramps are steep in.
*Runtime:* ~3 min collection per tag, ~6 min per target for 16 probes.
*Sample:* 49 152 valid tokens train, 49 152 held out.

Temporal sublayers are invoked through `forward_sequence`, not `forward`, so a
plain forward hook never fires on them; the collector wraps the method instead.
They also emit `(B·N, T, D)` where the spatial sublayers emit `(T·B, N+M, D)` —
reshaping one as the other silently shuffles states against their labels and
would report a real signal as noise.

MLP probe, held-out, frontline bearing (`frontline_strategy().bearing`):

| tap | median | mean | p90 | <3° | <10° | median @ `alpha≈1` | median @ turn ramp |
|---|---|---|---|---|---|---|---|
| raw ego input features (101 ch) | 14.25 | 36.71 | 115.5 | 0.148 | 0.407 | 9.45 | 5.88 |
| encoder output | 17.04 | 38.72 | 117.8 | 0.127 | 0.363 | 11.33 | 7.47 |
| b0.spatial0 | 14.13 | 36.79 | 115.7 | 0.149 | 0.411 | 9.18 | 6.07 |
| b0.spatial1 | 11.23 | 29.53 | 92.3 | 0.170 | 0.467 | 7.59 | 5.16 |
| b0.temporal0 | 10.38 | 28.40 | 90.0 | 0.188 | 0.489 | 6.97 | 4.67 |
| b1.spatial0 | 7.84 | 25.00 | 82.4 | 0.242 | 0.570 | 5.24 | 3.84 |
| b1.spatial1 | 7.19 | 23.47 | 76.9 | 0.259 | 0.590 | 4.83 | 3.46 |
| **b1.temporal0 (final latent)** | **7.00** | 23.20 | **76.1** | 0.272 | 0.596 | **4.69** | 3.29 |

MLP probe, held-out, personal intercept bearing (`predict_interception`):

| tap | median | p90 | <3° | median @ `alpha≈1` |
|---|---|---|---|---|
| raw ego input features | 30.74 | 139.6 | 0.082 | 26.20 |
| encoder output | 31.96 | 143.9 | 0.067 | 27.46 |
| b0.spatial0 | 21.97 | 118.1 | 0.107 | 16.89 |
| b0.spatial1 | 9.97 | 64.9 | 0.208 | 8.63 |
| b0.temporal0 | 9.07 | 63.5 | 0.233 | 7.53 |
| b1.spatial0 | 8.33 | 58.5 | 0.247 | 7.21 |
| b1.spatial1 | 8.12 | 55.1 | 0.262 | 6.86 |
| **b1.temporal0 (final latent)** | **7.92** | **55.4** | 0.264 | 7.09 |

*Replication.* The whole pipeline was re-run end to end — fresh rollout
collection, fresh probe fits — and reproduces every tap within `0.8°`:
`15.08 / 16.52 / 14.72 / 11.03 / 10.38 / 7.81 / 7.04 / 6.84` against the
`14.25 / 17.04 / 14.13 / 11.23 / 10.38 / 7.84 / 7.19 / 7.00` tabulated above.
The monotone curve, the absence of a plateau at the last layer, and the
encoder's apparent regression (`15.08 → 16.52`) all survive. `exp7_rows.json`
holds the replication run. Given session 1's finding that whole-rollout KL
varies by ±0.1 between rollouts, this stability is worth noting: probe error is
a much quieter measurement than KL, which is part of why it is the better
instrument for judging an intervention.

Linear probes are far worse everywhere (final latent: `12.1°` frontline, `27.0°`
personal) — the bearing is present but nonlinearly encoded, so every number above
is the MLP.

*Interpretation.*

* **No plateau.** Frontline error is still falling at the last layer
  (`7.19 → 7.00`), and every spatial sublayer in block 1 still pays. Whatever the
  trunk is doing, two blocks is not enough of it. This is a depth-of-computation
  signal, and it is the one piece of evidence that pointed at capacity before
  Exp 10 reframed it.
* **The spatial sublayers do the work; the temporal ones add almost nothing.**
  `b0.spatial1` buys `2.9°`, `b1.spatial0` buys `2.5°`; `b0.temporal0` buys
  `0.85°` and `b1.temporal0` buys `0.19°`. Even those small gains are not
  necessarily *recurrence* — a Griffin temporal block contains a gated MLP path,
  so part of any gain is simply more nonlinearity. Consistent with the teacher
  being memoryless: there is nothing for recurrence to contribute.
* **The encoder appears to lose precision.** The 101-channel raw feature vector
  probes to `14.25°` and its own 128-d encoding to `17.04°` (frontline); the
  personal bearing shows the same ordering. Modest and near the probe's noise
  floor, but it is the wrong direction for a widening projection and worth one
  cheap check.
* **`alpha≈0` is uninformative for the frontline target** (`61°` at the final
  latent). When `alpha = 0` the teacher does not use the frontline bearing, so
  nothing ever pressed the trunk to represent it there. Not a defect; read the
  `alpha≈1` column instead.
* The personal bearing plateaus early (`9.97°` at `b0.spatial1`, `7.92°` at the
  end); the frontline bearing keeps improving. Consistent with the frontline
  bearing needing a fleet-wide and zone-wide reduction while the intercept
  bearing needs one target.

### Exp 8 — oracle-bearing control and the achievable floor

*Question:* is the residual in the action head or in the bearing? *Method:*
`exp8_oracle.py` fits 2×512 MLP turn heads on the **frozen** final latent with
and without the teacher's true bearings appended, trains on one rollout set and
scores held-out turn KL on another. Oracle features are the teacher's complete
sufficient statistic: `sin/cos` of both bearings, `alpha`, and both `|bearing|`.
*Runtime:* ~4 min. *Sample:* 49 152 / 49 152.

| turn head | held-out turn KL |
|---|---|
| frozen policy action head (the checkpoint) | 0.511 |
| fresh head, frozen final latent only | 0.640 |
| **fresh head, frozen final latent + true bearings** | **0.040** |
| fresh head, **true bearings only** (no latent) | **0.011** |
| fresh head, raw ego input features only | 1.348 |

*Interpretation.* Turn KL collapses by 16× the moment the bearing is supplied,
and a head that sees *only* the bearings — no latent, no observation — reaches
`0.011` — the same order as the bf16 storage noise of `expert_probs` (D1) plus
the fitted head's own slack, though this session did not separate the two. So the
achievable floor is zero for practical purposes, the action head has ample capacity, and
the entire residual is the bearing. The `0.640` for latent-only against the real
head's `0.511` is the probe's handicap, not a finding: 49k samples cannot match a
head trained for 167.6M steps. Comparisons are therefore made against `0.640`
wherever a fitted head is involved.

Measured angular error against outcome, on the `alpha > 0.99` tokens where the
frontline bearing is the whole teacher (error from a final-latent probe, so the
error attached to each token is an honest held-out error):

| probe error | tokens | turn KL | teacher-turn → policy-straight | left↔right flip | sharp↔normal |
|---|---|---|---|---|---|
| [0°, 1°) | 4071 | 0.353 | 0.201 | 0.006 | 0.016 |
| [1°, 2°) | 3779 | 0.354 | 0.211 | 0.004 | 0.015 |
| [2°, 3°) | 3459 | 0.403 | 0.224 | 0.007 | 0.023 |
| [3°, 5°) | 5560 | 0.452 | 0.239 | 0.009 | 0.028 |
| [5°, 8°) | 5156 | 0.594 | 0.304 | 0.019 | 0.039 |
| [8°, 12°) | 3648 | 0.757 | 0.329 | 0.039 | 0.073 |
| [12°, 20°) | 2890 | 0.847 | 0.247 | 0.093 | 0.153 |
| [20°, 45°) | 2275 | 0.855 | 0.090 | 0.193 | 0.214 |
| [45°, 180°] | 1316 | 1.048 | 0.011 | 0.441 | 0.131 |

Turn KL rises monotonically with measured bearing error, and the *kind* of error
changes with it: under-turning dominates up to ~12° and left/right flips take
over past 20°. Two cautions. The floor of `0.35` in the lowest bin is not a
contradiction of Exp 8's `0.040` — the probe's accuracy on a token is not the
head's accuracy on that token, and the two extract different, partially
overlapping approximations. And plugging the probe's *point estimate* into the
analytic teacher scores `2.54`, far worse than the policy's `0.568`: a steep ramp
punishes point estimates, and the policy is correctly hedging instead. The
fitted-head calibration in Exp 12 is the sound way to price an error, and is what
the executive summary quotes.

### Exp 9 — representation audit: absolute Fourier vs ego-relative geometry

*Question:* is the geometry lost before the trunk, or does the trunk fail to
compute it? *Method:* `exp9_collect_obs.py` dumps the policy's own observation
channels unencoded; `exp9_repr.py` builds two feature sets carrying **identical
information** and fits the same flat MLP probe to each, held out as before.
Set A is every token's absolute position through the model's own 8-frequency
Fourier expansion plus its scalar channels — what the ship encoder actually
receives. Set B is the same tokens with the toroidal displacement to the ego ship
rotated into the ego frame, handed over as `(dx, dy, distance, unit vector)`.
*Runtime:* ~8 min. *Sample:* 49 152 / 49 152.

| target | features | dim | median | p90 | median @ `alpha≈1` |
|---|---|---|---|---|---|
| frontline | A absolute Fourier | 1248 | 27.94 | 131.9 | 19.22 |
| frontline | **B ego-relative** | 624 | **16.69** | **84.3** | **13.25** |
| personal | A absolute Fourier | 1248 | 35.33 | 164.6 | 16.56 |
| personal | **B ego-relative** | 624 | **23.58** | **160.0** | **15.86** |

*Interpretation.* Same information, same probe, same budget: the ego-relative
encoding is ~1.6× more accurate. But note that **both** are far worse than the
trunk's own final latent (`7.0°`), so this flat probe is sample-complexity
limited and its absolute levels say nothing about the observation's ceiling. Only
the A-vs-B gap is informative here. Exp 10 removes the confound by giving the
probe the teacher's own inductive bias.

### Exp 10 — sum-pooling probe: the decisive representation result

*Question:* with the right structure, how precisely can the bearing be recovered
from the observation — and does the encoding matter once structure is available?
*Method:* `exp10_deepsets.py` fits a masked sum-pooling probe (per-token φ →
masked mean and max → ρ, ~200k parameters, `h=192`) over the 24 entity tokens.
Three cells differ **only** in each token's geometric channels. `abs_fourier`
gets absolute Fourier positions *plus the ego's own absolute position broadcast
to every token* — the friendliest possible absolute encoding. `ego_relative` gets
ego-frame displacement, distance and an explicit unit vector. `ego_rel_nounit`
gets the same minus the unit vector. Held out as before.
*Runtime:* ~6 min. *Sample:* 49 152 / 49 152.

| cell | token dim | target | median | mean | p90 | <3° | <10° | median @ `alpha≈1` |
|---|---|---|---|---|---|---|---|---|
| abs_fourier | 84 | frontline | 29.36 | 48.02 | 129.3 | 0.069 | 0.217 | 22.62 |
| **ego_relative** | 26 | frontline | **5.37** | **10.22** | **21.6** | 0.307 | 0.725 | **4.71** |
| ego_rel_nounit | 24 | frontline | 13.77 | 29.42 | 84.7 | 0.137 | 0.397 | 10.71 |
| abs_fourier | 84 | personal | 23.39 | 46.85 | 135.3 | 0.167 | 0.363 | 10.49 |
| **ego_relative** | 26 | personal | **5.49** | 16.52 | **40.7** | 0.411 | 0.623 | **2.42** |
| ego_rel_nounit | 24 | personal | 7.76 | 21.09 | 57.6 | 0.319 | 0.548 | 4.21 |

For reference, the trunk's final latent (Exp 7): frontline `7.00` median / `76.1`
p90; personal `7.92` / `55.4`, from 1.945M parameters and 167.6M environment
steps.

*Interpretation.* This is the session's strongest result.

* A **200k-parameter** probe trained on **49k tokens** matches or beats the full
  trunk: frontline median `5.37` vs `7.00`, and p90 `21.6` vs `76.1` — a 3.5×
  smaller tail, which is where the KL lives. On the personal bearing at
  `alpha≈1` it reaches `2.42°` against the trunk's `7.09°`. The one stratum where
  it does not win is the frontline bearing at `alpha≈1` (`4.71°` vs the trunk's
  `4.69°`) — a tie, and worth stating plainly: on its best-served states the
  trunk is already as good as the probe, and the probe's advantage is
  concentrated in the tail and in the states the trunk handles worst.
* The **same probe** on the trunk's own absolute-Fourier encoding gets `29.36°`.
  The gap between `29.36` and `5.37` is caused by nothing but the geometric
  channels, holding architecture, data and budget fixed.
* The **unit vector alone is worth 2.6×** (`5.37` vs `13.77`). This is the
  pairwise term `unit(p_j − p_i)` that attention structurally cannot produce, and
  it is the single most load-bearing feature in the comparison.

Together these say the observation contains the geometry, the trunk's encoding
does not make it cheaply available, and the deficit is representational rather
than a matter of parameters or samples.

### Exp 11 — does the better representation actually buy turn KL?

*Question:* chain Exp 10 into the metric that matters. *Method:*
`exp11_intervention.py` fits the ego-relative sum-pooling bearing probe on the
`train` rollouts, reads its bearing estimate out on `heldout` and `heldout2`
(out-of-sample in both), fits a turn head on `heldout` and scores it on
`heldout2`. Every number is an unseen-state number.
*Runtime:* ~6 min. *Sample:* three independent 49 152-token sets.

Bearing probe on `heldout2`: frontline median `5.82` / p90 `23.1`; personal
median `4.58` / p90 `37.9`.

| turn head | held-out turn KL |
|---|---|
| frozen policy action head | 0.511 |
| fresh head, latent only | 0.619 |
| fresh head, latent + **predicted** bearing | 0.556 |
| fresh head, predicted bearing only | 0.710 |
| fresh head, latent + **true** bearing | 0.035 |
| fresh head, true bearing only | 0.011 |

*Interpretation.* The predicted bearing improves its own baseline by 10%
(`0.619 → 0.556`) and lands nowhere near the oracle's `0.035`. A median of `5.8°`
is simply not accurate enough: Exp 12 shows that KL is set by the p90, and `23°`
at p90 costs most of the residual on its own. **A better representation is
necessary but not sufficient on its own at probe budget** — which is why the
recommendation is an end-to-end architecture change, where the trunk gets 167M
steps to exploit the representation rather than 49k tokens. Reporting this cell
as a success would be the easiest mistake available here.

### Exp 12 — how accurate must the bearing be, and the left/right gap

*Question (part 1):* price a bearing error properly, replacing Exp 6's
assumed-hedging estimate. *Method:* `exp12_calib.py` corrupts the teacher's true
bearings with a wrapped Gaussian of known scale, **fits** a turn head on the
corrupted bearing (so optimal hedging is learned, not assumed), and scores
held-out turn KL. *Runtime:* ~5 min.

| σ (rad) | median error | p90 error | held-out turn KL |
|---|---|---|---|
| 0 | 0.00° | 0.00° | 0.011 |
| 0.005 | 0.19° | 0.47° | 0.013 |
| 0.01 | 0.38° | 0.94° | 0.018 |
| 0.02 | 0.77° | 1.88° | 0.039 |
| 0.035 | 1.35° | 3.31° | 0.085 |
| 0.05 | 1.94° | 4.72° | 0.138 |
| 0.08 | 3.08° | 7.51° | 0.237 |
| 0.12 | 4.61° | 11.27° | 0.340 |
| 0.20 | 7.70° | 18.88° | 0.477 |
| 0.35 | 13.51° | 32.83° | 0.620 |

The checkpoint measures turn KL `0.511`, which sits between the `σ = 0.20` and
`σ = 0.35` rows — i.e. an equivalent bearing error of roughly median `8.5°` /
p90 `21°`. The direct layerwise probe measures the final latent at median `7.0°`.
Those agree as well as they can: a probe is a *lower bound* on the error the head
suffers, so the head's effective error must be at least the probe's, and it is.
**Measured bearing error quantitatively explains the observed turn KL.**

Read as a requirement: turn KL `0.24` needs p90 `7.5°`; turn KL `0.14` needs p90
`4.7°`; turn KL `0.04` needs p90 `1.9°`. The ego-relative sum probe's p90 of
`21.6°` interpolates to ≈`0.51`, against the `0.556` Exp 11 measured — close, but
the agreement should not be read as tighter than it is: this curve is generated
by Gaussian corruption, whose p90 is always 2.45× its median, and neither the
trunk's error distribution nor the probe's has that shape. The curve is reliable
for *how steep* the requirement is and unreliable as a point predictor for any
particular estimator.

*Question (part 2):* does the left/right gap survive matching on `|bearing|`?
Re-weighting the right-hand bands to the left-hand `|bearing|` histogram:

| band (rad) | n left | n right | KL left | KL right |
|---|---|---|---|---|
| [0, 0.03) | 3038 | 3065 | 0.392 | 0.413 |
| [0.03, 0.06) | 2637 | 2923 | 0.343 | 0.346 |
| [0.06, 0.09) | 2235 | 2437 | 0.441 | 0.462 |
| [0.09, 0.12) | 1774 | 2073 | 0.701 | 0.819 |
| [0.12, 0.2) | 2973 | 3871 | 0.817 | 0.959 |
| [0.2, 0.3) | 1814 | 2447 | 0.622 | 0.733 |
| [0.3, 0.42) | 1136 | 1556 | 0.552 | 0.754 |
| [0.42, 0.7) | 1310 | 1707 | 0.475 | 0.697 |
| [0.7, 1.2) | 1586 | 1888 | 0.230 | 0.315 |
| [1.2, 2.0) | 2342 | 2323 | 0.144 | 0.193 |
| [2.0, 3.2) | 2007 | 2010 | 0.267 | 0.355 |

Matched: left `0.454`, right `0.537` — a `0.083` nat gap that survives and is
present in every band, widening with `|bearing|`.

### Exp 13 — is the left/right gap geometric?

*Method:* `exp13_side.py` splits the final-latent bearing probe's held-out error
by side, matched to the band `0.03 < |bearing| < 0.7`. *Runtime:* ~2 min.

| target | side | n | median error | p90 error |
|---|---|---|---|---|
| frontline | left | 13 494 | 5.33° | 24.08° |
| frontline | right | 16 080 | 5.14° | 23.32° |
| personal | left | 4 417 | 8.87° | 39.08° |
| personal | right | 5 515 | 8.44° | 36.32° |

Signed error (positive = predicted anticlockwise of truth): frontline median
`+0.12°`, mean `−0.73°`; personal median `−0.20°`, mean `−1.51°`.

*Interpretation.* The representation is **symmetric in error magnitude** — if
anything marginally better on the right — so the `0.083` nat gap is not a
geometric or sign-convention defect. Exp 6 already verified the teacher
reconstruction to `8.4e-4`, which rules out action indexing independently. Two
things push in the observed direction instead: a small **leftward signed bias**
of `0.7–1.5°` in the latent's bearing (at this ramp's steepness a `0.57°`
systematic offset is worth `0.042` nats, per Exp 6), and a genuinely
**right-skewed teacher** — 54% of turning tokens in the matched band have a
positive bearing. A leftward-biased estimator on a right-skewed target costs
more on the right. Plausible but not proven; it is a cheap follow-up, not a
blocker, and it is worth ≤ 20% of the residual either way.

## Recurrent-state findings

The rollout-start hidden state barely matters for this metric. On identical
samples (Exp 1): stored vs zero hidden differs by `0.0000` (rollout 0, where the
stored hidden legitimately is zero), `0.0066` (rollout 1) and `0.0099`
(rollout 2) in total KL — i.e. under 2% of the value, and in inconsistent
directions. Two reasons: the initial state only affects the head of a 128-step
sequence, and episode boundaries re-zero it anyway. Consequence for methodology:
a frozen-dataset offline experiment that reuses checkpoint-generated hidden
states is **not** meaningfully contaminated by staleness at this scale — but
that is because the model is nearly memoryless *for the BC target*, which is
itself a finding: the teacher's turn head is a pure function of the current
state, so there is nothing for recurrence to contribute.

Session 2 checked this the way the brief asked — by probing *before and after*
each temporal sublayer rather than inferring memorylessness from a zeroed
rollout-start state. The recurrent stack still runs through the whole sequence
in both cases, so this is the sound test. `b0.temporal0` improves the
frontline-bearing probe from `11.23°` to `10.38°`; `b1.temporal0` improves it
from `7.19°` to `7.00°` (Exp 7). Both gains are small, and neither is
attributable to recurrence as such — a Griffin temporal block carries a gated
MLP path, so some of it is simply more nonlinearity. Recurrence does not
meaningfully improve bearing estimation, and the spatial sublayers account for
essentially all of the trunk's progress on it (`2.9°` and `2.5°` per layer).

## Auxiliary-loss findings

See Exp 4 for the gradient geometry (BC 89% of trunk gradient, next-state 6%,
cosine −0.02) and Exp 5 below for the matched training ablation.

## Defects found

### D1 — `expert_probs` are stored in bf16 (minor, not fixed)

*Symptom:* `bc_kl` is slightly negative on heads the policy has solved — e.g.
power KL `−0.00061` (Exp 1, rollout 0), which is impossible for a true KL.
*Root cause:* `RolloutBuffer.expert_probs` uses `_STORAGE_FLOAT = torch.bfloat16`
(`src/boost_and_broadside/train/rl/buffer.py:216,656`). bf16 has an 8-bit
mantissa, so a stored teacher distribution no longer sums to 1; `bc_loss` uses
the raw stored probabilities while the entropy floor uses `p.clamp(min=1e-8)`,
and the difference of the two can go negative by ~1e-3.
*Measured effect:* order `1e-3` nats — three orders below the turn KL under
investigation. It also caps how exactly BC can ever fit (target noise of
~`8e-4` per probability, Exp 6).
*Fix:* not applied. Changing the storage dtype changes rollout memory for every
run and is not justified by a 1e-3 bias; it is recorded so nobody mistakes the
negative power KL for an evaluator bug.

### D2 — unmasked KL evaluator (the `0.81`) — *in a diagnostic, not in production*

*Symptom:* the same checkpoint scoring `0.814` in one offline evaluation and
`0.344` in another. *Root cause:* omitting `actor_mask` when reducing. Under
`paradigm = ego_pass` the live policy acts only for team 0, so `actor_mask`
selects exactly half the ship tokens, while the scripted teacher labels all of
them; the unmasked mean therefore mixes in 52 127 opponent-team tokens at KL
`1.098` and 13 409 dead-ship tokens at KL `1.725`.
*Measured effect:* `0.4393 → 0.8330`, a 1.90× inflation, with the denominator
moving `65 536 → 131 072`.
*Fix:* none needed in production — `_minibatch_denominators` and
`_compute_minibatch_loss` both apply `bc_valid & actor_mask & alive`
(`ppo.py:1596-1600`, `1743-1744`), and this session's measurements reproduce the
production value. Any future offline evaluator must use the same three masks;
the check is cheap — the valid denominator must equal
`num_steps × num_envs × num_ships / 2` in `ego_pass`.

### D3 — `loss/behavioral_cloning_kl` averages over the update epochs (confirmed)

*Symptom:* the run logs BC KL ≈`0.316` while the checkpoint measures `0.44–0.57`
pre-update on fresh rollouts.
*Root cause:* `accum_scalar` is created once per `_update_epochs`
(`ppo.py:2586`) and `bc_kl` is listed in the `_additive` diagnostics table
(`ppo.py:2711`), so every micro-batch of every minibatch of all four epochs
contributes to one mean. The logged number is the average over the descent, not
the policy's value before it.
*Measured effect:* Exp 5 measured four production epochs moving turn KL from
`0.4335` to `0.3303` **on the rollout being scored**; the mean over that descent
is ≈`0.32`, which is what the run reports. So the headline BC number has been
flattering itself by roughly `0.13` nats.
*Fix:* not applied — it is a logging change with no effect on training, and
applying it mid-investigation would break comparability with the run's own
history. Log epoch 0 separately and use *that* for plateau detection.

## Recommended fix

Ranked by evidence, not by cost.

### 1. Give the spatial sublayers relative position — the primary recommendation

Supply `unit(p_j − p_i)` and `log(1 + |p_j − p_i|)`, toroidally wrapped and
rotated into the query ship's frame, as a **pairwise** term inside spatial
attention: project it and add it to the value (and/or as an attention bias), the
way a geometric transformer does. This is the one quantity the current
architecture structurally cannot form, and it is exactly the term the teacher
sums.

*Evidence:* Exp 10. Holding architecture, data and budget fixed, the ego-relative
cell reaches median `5.37°` / p90 `21.6°` against `29.36°` / `129°` for the
absolute-Fourier cell; withholding only the unit vector costs `5.37 → 13.77`. A
200k-parameter probe on 49k tokens beats the 1.945M-parameter trunk trained on
167.6M steps. Exp 12 converts an improvement in p90 into turn KL directly.

*Expected effect, with its caveat:* Exp 12's curve says p90 `7.5°` is worth turn
KL ≈`0.24` and p90 `4.7°` ≈`0.14`. Read that as an order of magnitude, not a
forecast — Exp 12 corrupts the bearing with a *Gaussian*, whose p90 is fixed at
2.45× its median, while the trunk's error distribution is far heavier-tailed
(p90 `76°` against median `7.0°`, a ratio of 10.9). The mapping from an error
*distribution* to KL is therefore only calibrated in the middle of its range,
which is why Exp 12's equivalent-noise reading of the checkpoint (median `8.5°`)
sits above the probe's direct measurement (`7.0°`). What the evidence supports
firmly is the direction and the rough size: the tail is what costs, and the
ego-relative probe cuts the tail by 3.5× using four orders of magnitude less
data than the run had.

*Cost:* one extra projection per spatial sublayer plus an `(N+M)²` displacement
tensor — 24 tokens, so negligible. It does change `ModelConfig` and the
checkpoint schema.

### 2. Bearing auxiliary loss — cheap, do it alongside, not instead

Add a diagnostic-turned-auxiliary head predicting `sin/cos` of
`frontline_strategy().bearing` and of the intercept bearing, and log the angular
error in degrees. Exp 7 shows the signal is learnable from the latent at every
depth, so this is a real training signal and a permanent instrument. But it
cannot create information the encoding does not afford — it presses the trunk to
spend capacity on the bearing without making the bearing cheaper to compute.
Pair it with (1); do not run it alone and conclude anything.

### 3. More spatial depth — supported, but the slow axis

Exp 7's curve is monotone and unsaturated at the last layer, so
`n_spatial_per_block` `2 → 4` will help. But two spatial sublayers bought
`10.38° → 7.19°`, so extrapolating to `3°` needs many more, at a linear cost in
compute. Worth running as the control cell *against* (1) on matched data and
matched optimizer steps — if (1) at depth 2 beats depth 4 without it, the
representational claim is confirmed in the only way that counts.

### Explicitly not recommended

* **Widening the model.** Nothing in Exp 7–10 implicates width; the winning probe
  is an order of magnitude smaller than the trunk.
* **More or more diverse BC data.** The winning probe used 49k tokens where the
  run had 167.6M environment steps. Sample complexity is not the constraint.
* **Removing the next-state auxiliary.** Closed in session 1 (Exp 4, Exp 5):
  orthogonal, 6% of trunk gradient, and removing it is not faster.
* **Touching the action head, or reweighting/resampling difficult angular
  regions.** Exp 8 and Exp 11 exonerate the head at `0.011–0.040` given the
  bearing. Reweighting redistributes a precision deficit; it does not fix one.
* **Softening the teacher's ramp.** Session 1 suggested this as a diagnostic. It
  is no longer needed — Exp 12 measures the ramp's price directly, with hedging
  fitted rather than assumed — and changing the teacher would change what is
  being imitated.

### How to measure any of these

Both metrics, on fresh rollouts, or the result is not interpretable:

1. **held-out turn KL** over `bc_valid & actor_mask & alive`, on rollouts the
   update has not touched (`exp7_collect.py` + the KL block of `exp8_oracle.py`);
2. **held-out bearing-probe error** at the final latent, median **and p90**
   (`exp7_probe.py`).

A change that improves both is real. A change that improves turn KL without
improving probe error is fitting the rollout it is scored on — Exp 5 measured
that failure mode at `0.43 → 0.20` on the frozen batch while the next fresh
rollout scored `2.59`.

## Remaining uncertainties and the cheapest next experiments

1. **Does (1) actually reach p90 < 5° end to end?** (~1–2 GPU-hours.) One `bc`
   run with relative-position attention in the spatial sublayers against a
   matched baseline, same seed, same data, same optimizer steps, tracking the
   turn-KL curve and the probe error rather than a final number. This is the
   experiment that settles the recommendation, and nothing above substitutes for
   it.
2. **Is the probe a tight lower bound on the head's error?** (~20 min.) Fit a
   turn head on `[frozen latent + probe-predicted bearing]` *with the probe
   trained to convergence on much more data* than 49k tokens. If the gap between
   Exp 11's `0.556` and Exp 8's `0.040` closes as the probe improves, the whole
   chain is confirmed; if it does not, the latent's bearing is less usable than
   the probe suggests and (1) matters even more.
3. **The `0.083` nat left/right gap.** (~30 min.) Test the leftward-bias
   explanation directly: subtract the measured signed bias from the probe's
   bearing and recompute the matched left/right KL. If the gap shrinks, it is a
   calibration artefact of the estimator, not a defect.
4. **Does the encoder really lose precision?** (~15 min.) Exp 7 shows the
   101-channel raw features probing better than their own 128-d encoding
   (`14.25°` vs `17.04°`). Re-run with more probe capacity and several seeds; if
   it holds, the encoder's two RMSNorms on a 101→256→128 path are worth a look.
5. **Log the pre-update BC KL separately** (~15 min, now a confirmed defect
   rather than a hypothesis). `loss/behavioral_cloning_kl` averages all four
   update epochs (`ppo.py:2586`, `2711`). Logging epoch 0 separately would move
   the run's headline BC number by roughly `0.13` nats and is what plateau
   detection should use.

## Raw-data appendix

Scripts live in [`benchmarks/bc_diagnostics/`](../../benchmarks/bc_diagnostics):

```
harness.py          trainer construction at the checkpoint + teacher-internals recorder
exp1_reconcile.py   bf16/fp32 x stored/zero hidden x per-shard KL
exp2_drift.py       12-rollout drift curve; writes dump_rollout11.pt
exp3_turn.py        turn decomposition, stratification, mask variants
exp4_grad.py        top_level gradient diagnostics
exp5_ns.py          next_state_coef 1 vs 0 matched fitting
exp6_angle.py       teacher reconstruction + bearing-error pricing
```

Session 2's scripts are listed below; Exp 6's bearing-error pricing is superseded
by Exp 12, which fits the hedging rather than assuming it.

`dump_rollout11.pt` (18 MB, not committed) holds the per-token evidence for
Exp 3 and Exp 6: teacher probabilities, policy turn log-probs, the three masks,
and the teacher's own intermediates (`alpha`, `p_team`, personal/team/frontline
bearings, nearest-enemy distance, speed, step count, alive count). Regenerate it
with `exp2_drift.py`; `BC_DIAG_DIR` chooses where it is written and read.

Commands (all from the repo root):

```
uv run --no-sync python benchmarks/bc_diagnostics/exp1_reconcile.py 128
uv run --no-sync python benchmarks/bc_diagnostics/exp2_drift.py 12
uv run --no-sync python benchmarks/bc_diagnostics/exp3_turn.py
uv run --no-sync python benchmarks/bc_diagnostics/exp4_grad.py
uv run --no-sync python benchmarks/bc_diagnostics/exp5_ns.py
uv run --no-sync python benchmarks/bc_diagnostics/exp6_angle.py
```

Common configuration for every run above:

```
checkpoint      checkpoints/icy-energy-741/step_000167608320.pt   (unmodified)
profile         bc          (policy_gradient_coef=0, behavior_cloning_coef=1,
                             next_state_coef=1, entropy_coef=0.005, Adam 3e-4)
launch          resolve_training_launch(profile="bc", vram="off", device="cuda",
                                        seed=0, compile_mode=None, wandb=False,
                                        num_envs=128, microbatch_tokens=12288)
env             num_ships=8, num_fields=10, 24 entity tokens, T=128, ego_pass,
                frontline zones active (zone_ring_radius=1200, zone_radius=330)
model           d_model=128, n_heads=4, 2 Yemong blocks, 2 spatial + 1 temporal
                per block, ~1.945M policy parameters
teacher         StochasticScriptedAgent(StochasticAgentConfig()) —
                turn_angle_ramp=(0.03,0.12) prob (0.068,0.95),
                sharp_turn_angle_ramp=(0.24,0.42) prob (0.05,0.95),
                team_target_distance_ramp=(60,100)  [inactive: p_team ≡ 0],
                frontline_combat_radius=600, shoot_distance_ramp=(200,500)
device          RTX 4070 Laptop 8 GB, torch autocast bf16 for forwards
```

`num_envs=128` rather than the run's 1280 is the only deviation from the
training configuration, forced by the 8 GB dev card. It changes the sample size
per rollout, not the distribution: each rollout still contributes 65 536 valid
tokens, and the per-shard spread (Exp 1) bounds the sampling error at ±0.06.

Caveat on Elo: `runtime.elo_eval.step/flush` are stubbed out in every probe, so
no ladder games were played and no rating was written. The checkpoint, its
`roster.json` and its `elo_history.jsonl` are untouched.

### Session 2 scripts, sample sizes and runtimes

```
exp7_collect.py     freeze checkpoint, burn in, collect rollouts, capture the 8
                    trunk tap points + raw observation channels -> probe_<tag>.pt
exp7_probe.py       layerwise (sin, cos) bearing probes, train/held-out split
exp8_oracle.py      oracle-bearing turn heads, achievable floor, error -> KL bins
exp9_collect_obs.py raw observation dump (superseded: exp7_collect.py now does this)
exp9_repr.py        flat-MLP probe, absolute Fourier vs ego-relative features
exp10_deepsets.py   sum-pooling probe; the decisive representation comparison
exp11_intervention.py  predicted bearing -> turn KL, three-way split
exp12_calib.py      fitted-head bearing-error calibration; left/right matching
exp13_side.py       probe error and signed bias by side
```

Reproduction, in order, from the repo root:

```
uv run --no-sync python benchmarks/bc_diagnostics/exp7_collect.py train    3 3 0   # ~3 min
uv run --no-sync python benchmarks/bc_diagnostics/exp7_collect.py heldout  3 3 1   # ~3 min
uv run --no-sync python benchmarks/bc_diagnostics/exp7_collect.py heldout2 3 3 2   # ~3 min
uv run --no-sync python benchmarks/bc_diagnostics/exp7_probe.py frontline          # ~6 min
uv run --no-sync python benchmarks/bc_diagnostics/exp7_probe.py personal           # ~6 min
uv run --no-sync python benchmarks/bc_diagnostics/exp8_oracle.py                   # ~4 min
uv run --no-sync python benchmarks/bc_diagnostics/exp9_collect_obs.py train   3 3 0
uv run --no-sync python benchmarks/bc_diagnostics/exp9_collect_obs.py heldout 3 3 1
uv run --no-sync python benchmarks/bc_diagnostics/exp9_repr.py                     # ~8 min
uv run --no-sync python benchmarks/bc_diagnostics/exp10_deepsets.py                # ~6 min
uv run --no-sync python benchmarks/bc_diagnostics/exp11_intervention.py            # ~6 min
uv run --no-sync python benchmarks/bc_diagnostics/exp12_calib.py                   # ~5 min
uv run --no-sync python benchmarks/bc_diagnostics/exp13_side.py                    # ~2 min
```

Arguments are `<tag> <n_rollouts> <n_burn_in> <seed>`. Total ≈55 GPU-minutes on
the 8 GB dev card. Each `probe_<tag>.pt` is 213 MB and each `obs_<tag>.pt` 25 MB;
both patterns are gitignored. Every `exp*_rows.json` alongside them is the raw
output backing the tables above and *is* committed — they are a few kB each.

Sample sizes. Each tag is 3 rollouts after 3 burn-in rollouts at `num_envs=128`,
`num_steps=128`, subsampled to every 4th timestep (`TSTRIDE = 4`; consecutive
steps are near-duplicates). That is 32 × 384 × 8 = 98 304 ship tokens per tag, of
which exactly **49 152** pass `bc_valid & actor_mask & alive` — half, because
`ego_pass` makes only team 0 actor tokens. Probes train on `train` and report on
`heldout` (Exp 7–10) or on `heldout2` (Exp 11–13, where `heldout` is spent
fitting the turn head). The three tags come from seeds 0/1/2 and are independent
draws, so no probe or head is ever scored on a state it was fitted on.

Two collection details that are easy to get wrong and silently fatal:

* Temporal sublayers are reached through `forward_sequence`, not `forward`, so a
  `register_forward_hook` never fires on them. `exp7_collect.py` wraps the method
  and restores it afterwards.
* Spatial sublayers and the encoder emit `(T·B, N+M, D)`; a temporal sublayer in
  sequence mode emits `(B·N, T, D)`. Reshaping one as the other misaligns states
  against labels, and the probe reports the result as "this layer carries no
  information" rather than failing.

Configuration is otherwise identical to session 1 (see the block above):
`d_model=128`, 2 Yemong blocks × (2 spatial + 1 temporal), `full_attention` map
reads, 24 entity tokens, world size 16384 px with play inside a 2600-px radius,
position encoded as 8 base-2 Fourier frequencies per axis (finest period 128 px),
`n_bullet_cross_per_block=0`. `runtime.elo_eval.step/flush` are stubbed in every
probe, so no ladder games were played and no rating was written; the checkpoint,
its `roster.json` and its `elo_history.jsonl` are untouched.
