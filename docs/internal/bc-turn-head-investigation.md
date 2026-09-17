# Behaviour cloning: why the turn head does not close

Run `icy-energy-741` (W&B `xs3whcqt`), profile `bc`, source commit `85f2bf3`,
checkpoint `checkpoints/icy-energy-741/step_000167608320.pt` (update 341,
167.6M steps). All measurements below were taken on 2026-09-17 on the dev
RTX 4070 Laptop (8 GB) at `num_envs=128`, `microbatch_tokens=12288`,
`num_steps=128`, `num_ships=8`, seed 0. The checkpoint was never modified.

## Executive summary

**Established facts (direct measurement, this session).**

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

**Likely interpretations.**

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

**Unresolved.**

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

## Recommended next experiments

Ordered by information per GPU-minute, given what the hour established.

1. **Regress the bearing directly (1 GPU-hour).** Add a diagnostic head that
   predicts `sin/cos` of the teacher's `frontline_strategy().bearing` and of the
   personal intercept bearing, trained alongside BC, and log the angular error
   in degrees. The turn KL is now a known monotone function of that error
   (Exp 6 table), so this converts an opaque nats number into a degrees number
   that can be compared against the ~3° needed. It also separates "the trunk
   does not encode the bearing" from "the action head cannot express it".
2. **Bearing-error attribution by input ablation (30 min).** Recompute the
   frontline bearing from the observation alone offline and compare with the
   privileged `TensorState` computation. `frontline_strategy` uses toroidal
   displacements over *all* ships plus zone geometry; if the observation's
   encoding loses precision in any of those (quantisation, vision range,
   belief-imputed enemies), the ceiling is a representation problem, not a
   capacity problem. This is the largest unresolved question and it is cheap.
3. **Spatial-depth cell, matched data (1–2 GPU-hours).** Only after (2). Compare
   `n_spatial_per_block` 2 → 4 at `d_model=128` against baseline, same rollouts,
   same optimizer steps, tracking the *turn* KL learning curve rather than a
   final value. Global bearing is a relational reduction over 24 entity tokens,
   so spatial depth is the right first knob, ahead of width.
4. **Teacher-steepness sensitivity (30 min, diagnostic only).** Re-run BC
   briefly against a diagnostic teacher with `turn_angle_ramp=(0.03, 0.30)` —
   the same decision boundary, a gentler link function. If the achievable KL
   scales with the ramp width as Exp 6 predicts, the residual is confirmed as
   estimator precision passed through a steep ramp, and the number to quote for
   the production teacher becomes "±N degrees", not "0.44 nats". Do **not**
   change the production teacher.
5. **Reconcile the online metric (15 min).** Log `loss/behavioral_cloning_kl`
   for epoch 0 separately from the 4-epoch mean. If the pre-update value is
   ~0.45 and the logged mean is ~0.32, the run's headline BC number has been
   flattering itself by roughly 0.13 nats all along, and plateau detection
   should use the epoch-0 value.

Deliberately **not** recommended: removing the next-state auxiliary (Exp 4/5
show it is orthogonal and small), widening the model before (2), or repeating
the dataset-size sweep.

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
