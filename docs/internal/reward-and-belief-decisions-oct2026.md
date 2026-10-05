# Reward, critic and belief redesign — decisions

October 4, 2026. Base: `dev` at `4eba8a1`. Status: **agreed design, not implemented.**

This records what was decided in review of the "semantic rewards, categorical state and
map-scale training" plan. The review itself, with the reasoning behind each objection,
is [`semantic-rewards-plan-audit-oct2026.md`](semantic-rewards-plan-audit-oct2026.md).
Where this note and the audit disagree, this note wins: several of the audit's
recommendations were revised in discussion and are recorded here in their final form.

The guiding rule is to follow published systems rather than invent: DreamerV3 and
Dreamer 4 for the critic and advantage plumbing, Honor of Kings for the reward
structure, OpenAI Five for the zero-sum construction and team mixing. Sections marked
**Open** were discussed but not settled.

---

## 1. Rewards

### 1.1 Structure: five heads, one per level

Rewards stay per ship, scalar, and attributed to the ship that earned them. Each of the
plan's five levels becomes **one critic head**. The events inside a level are summed
into that head's reward; they are still computed and logged separately, but they do not
get their own value heads.

| Head | Events summed into it | Gamma | GAE lambda | Read from |
|---|---|---|---|---|
| outcome | match result | 0.9997 | 0.97 | global token, categorical (§2.2) |
| zone capture | capture completion ledger (`zone_gain`, `zone_loss`) | 0.999 | 0.97 | ship token |
| zone progress | live meter movement (captured / reset, ally / enemy zone) | ~0.995 | 0.95 | ship token |
| kill and death | `death`, `kill_enemy`, `kill_ally` | 0.995 | 0.95 | ship token |
| damage | `damage_taken`, `damage_dealt_enemy`, `damage_dealt_ally`, `shield_recharge`, `healed_damage` | 0.991 | 0.90 | ship token |

Gammas and lambdas are in the existing per-tick convention of
`COMPONENT_GAMMAS_PER_TICK` / `COMPONENT_LAMBDAS_PER_TICK` in `config/defaults.py`.

Why five and not fifteen or one:

* **Not fifteen.** GAE is linear in rewards and values, so at a shared gamma and lambda
  the advantage of a sum equals the sum of the advantages. Splitting events inside a
  level does not change the policy gradient; it changes only how the critic is fitted.
  The case the split used to serve, "uncertain ±1" versus "certainly 0", is represented
  by the categorical critic (§2) as a distribution with the same mean.
  Known loss: a merged kill-and-death head cannot tell a mutual trade (kill and die,
  net ≈ 0) from a quiet spell, because the realised return is the same. This does not
  affect the policy gradient.
* **Not one.** The levels need different horizons (0.9997 for the match, 0.991 for an
  exchange), the outcome is read from the global token while the rest are per ship, and
  a sparse ±1 outcome summed with dense damage would dominate the critic's error.
* **Zone capture and zone progress are separate** because they have different time
  scales, the same relationship as death to damage: progress pays on the tick a ship
  stands on the point, a completion pays only once the meter fills, at least
  `capture_seconds` later. Both currently share gamma 0.999 and lambda 0.97 by accident
  of tier, not by decision.

This is the Honor of Kings layout: zero-sum reward groups, per-agent rewards, one value
head per group, advantages combined by weight.

### 1.2 Balance: zero sum per level, weights derived

* Within a level, every event charges the side it happens to `w` and pays the side
  that caused it `ratio × w`. `kill_payout_ratio`, `damage_payout_ratio` and
  `capture_payout_ratio` start at 2.0 and `offensive_bias` anneals them to 1.0.
* **Invariant:** at all ratios 1, each head's reward summed over both teams is zero on
  every tick. The test asserts this per head (per level), not per event component.
* **Fallback payee.** When nobody on the opposing side caused an event, the counterpart
  is split evenly over the opposing team with `enemy_team_share`, dividing by the full
  team slot count including dead ships. The only non-bullet damage source is the
  boundary, so this covers boundary damage, boundary death, an enemy's friendly fire,
  and healing of boundary damage.
* **Empty ledger.** A bullet kill on a ship whose outstanding-damage ledger is empty
  (its shield was emptied by the boundary) gives the whole kill share to the raw final
  hit.
* **Free parameters:** one weight per level plus the offensive bias. Everything else is
  derived. `kill_shot_fraction` is removed (fixed at 0.5 by the attribution rule).

### 1.3 Event definitions

Unchanged from the plan unless stated:

* Zone capture uses the presence and absence ledgers (§1.5 of the plan), half
  final-tick presence and half ledger share. Ledgers clear on completion, when roles
  rotate (which also zeroes the zone that was not captured), and when the meter returns
  to zero.
* Kills use the outstanding-damage matrix with pro-rata decrement on recharge, shared
  half by raw final impact and half by ledger share.
* `damage_taken` counts every source, boundary included. This is what stops recharge
  farming.
* `healed_damage` is the charge-back to attackers when their damage is recharged. It
  arrives at least 150 ticks after the hit, so at the damage gamma it is worth roughly a
  quarter of the payment it reverses. It is correct bookkeeping, not the anti-farming
  mechanism. Log its realised share of the damage head.
* Removed: `ally_win`, `enemy_win`, the old `outcome` probe, `boundary`,
  `boundary_damage`, the `ally_/enemy_combat_*` pair, and the shaping components.

### 1.4 Lambda matrix and team mixing

* Delete the lambda matrix (`_lambda_matrix`, `_precompute_lambda_aggregates`,
  `enemy_neg_k`, `ally_zero_k`, `_LOCAL_COMPONENTS`). Every nonzero component is
  already diagonal or bit-identical across a side, so this is behaviour-preserving.
* Keep one scalar per head in its place, OpenAI Five's "team spirit":
  `adv_i + s × mean_{j≠i, same side} adv_j`. Default **0** (pure per-ship credit). It
  exists so that a selfish equilibrium has a cheap remedy.
* Rewire the reward-decomposed gradient diagnostic to per-head advantages.

---

## 2. Critic and advantages

### 2.1 Raw rewards, categorical critic per head

Follows DreamerV3 / Dreamer 4.

* **Rewards are stored raw.** Remove the `symlog` applied at storage time in
  `RolloutBuffer` (`buffer.py`). GAE runs on raw rewards.
* **Everything the policy and GAE see is still a scalar.** Rewards, lambda-returns and
  advantages are scalars per ship per head, computed by GAE exactly as now.
* **Only the critic's output is a distribution.** Each head emits logits over fixed
  bins whose locations are symlog-spaced (DreamerV3: symexp of a uniform grid). The
  head's value V is the expectation, sum of probability times bin value, in raw units;
  that V is what GAE uses.
* **Training target:** the realised scalar lambda-return, converted to a two-hot (or
  HL-Gauss) distribution over the bins, trained with cross-entropy. The target's mean
  equals the scalar return, so what the value means does not change.
* **Return scaler:** no longer needed for the critic; the symlog-spaced support handles
  scale. The compression lives in exactly one place, the bin spacing.
* **51 bins** per head. DreamerV3 and Dreamer 4 use 255 because they span many
  domains; this project does not need to. DreamerV3's support is
  `symexp(linspace(-20, 20))`; at 51 bins that range would leave a spacing of 0.8 symlog
  units, with the first bins off zero at ±1.2 and ±4.0 in raw units, far too coarse
  for these returns. **Narrow the range** to cover each head's measured maximum |return|
  with margin (for example ±5 symlog units, ±147 raw, spacing 0.2). Two-hot target as in
  DreamerV3.
* Explained variance is computed on the expectation so the series stays comparable
  with runs 748 and 750. The Huber value loss and any value clipping go away.

### 2.2 Outcome head: four classes

* Classes **win / tie / loss / unresolved**. Value on the ±1 scale is
  V = P(win) − P(loss).
* Gamma stays 0.9997 (run 737 showed gamma 1 bootstrapping collapsing). The discount is
  modelled as probability mass leaking into **unresolved**, worth 0, so **tie** remains
  the honest tie probability:

  ```
  Q_t = gamma · [(1 − lambda) · p(s_{t+1}) + lambda · Q_{t+1}] + (1 − gamma) · e_unresolved
  Q_t = onehot(result)    on the terminal transition (a timeout is a real tie)
  ```

  Its expectation equals the scalar lambda-return. Loss: cross-entropy against Q_t.
* Calibration diagnostic: P(win | resolved) against realised wins.

### 2.3 Combining heads into one advantage

* Weight each head's scalar advantage by its level weight, **sum, then normalise
  once**, exactly as DreamerV3 does. Verified against the official source
  (`github.com/danijar/dreamerv3`, `dreamerv3/configs.yaml`, `dreamerv3/agent.py`
  `imag_loss`, `embodied/jax/utils.py` `Normalize`):

  | setting | DreamerV3 value |
  |---|---|
  | `retnorm` | `impl: perc, rate: 0.01, limit: 1.0, perclo: 5.0, perchi: 95.0, debias: False` |
  | scale | `max(limit, hi − lo)` of EMA percentiles of the lambda-returns (EMA rate 0.01, i.e. decay 0.99) |
  | advantage | `(ret − value) / scale`: the mean is **not** subtracted |
  | `advnorm`, `valnorm` | `impl: none` |
  | critic / reward heads | `symexp_twohot`, 255 bins, `symexp(linspace(-20, 20))` |
  | discount | `horizon: 333`, so gamma = 1 − 1/333 ≈ 0.997; lambda 0.95 |
  | critic regulariser | `slowreg: 1.0` toward an EMA copy of the critic (`slowvalue rate 0.02`) |
  | actor entropy | `actent: 3e-4` |

  So: the summed return's 5th–95th percentile spread, EMA decay 0.99, floored at 1;
  divide the summed advantage by it; **drop PPO's per-minibatch advantage
  standardisation** (no mean subtraction). The slow-critic regulariser is not adopted;
  PPO's fixed rollout targets do that job here.
* **The unit, "1".** DreamerV3's floor is in raw reward units: advantages are never
  amplified when the return spread is under one unit. **A win is 1** (`win_weight` is
  already 1.0, and the win is the objective and the largest single event). The floor
  then binds only when outcomes barely vary across the batch, as in the run-735
  stalemate where 90% of games timed out; in that regime small kill and damage
  differences stay at their natural size instead of being blown up.
* This replaces the per-component `AdvantageScaler`, which normalised each component
  to unit RMS before weighting and so cancelled the derived weights. After the change
  the level weights reach the policy exactly as derived, and splitting or merging events
  inside a level cannot change a level's gradient share.
* Gate for the reward build step: measure per-head gradient shares with
  `benchmarks/gradient_decomposition.py` before comparing to earlier runs.

### 2.4 Policy objective

PPO is unchanged. Dreamer 4's PMPO (advantage sign only, alpha 0.5, reverse KL to the
behaviour-cloned prior at beta 0.3) was considered and is not part of this change.

---

## 3. Next-state head and belief

### 3.1 Targets and residual head

* The head outputs residual logits on a baseline built from the believed state:
  `z = log p̃ + f(h)`, last layer zero-initialised, trained by cross-entropy.
* **Targets are sharp for every channel and every token, hidden ones included.**
  Cross-entropy against the sharp truth is a proper scoring rule; at a level the head
  cannot resolve its optimum is the conditional distribution, which is near uniform. No
  hindsight blur of targets.
* **Inputs and baselines are blurred** (§3.2), so a calibrated head outputs zeros at
  a level it cannot resolve. This is the reason for the residual design: the head never
  has to memorise and negate a stale sharp pattern.

### 3.2 Belief storage: moments, not distributions

Storing full categorical distributions per ship cost about 1.1 GB extra GPU buffer per
resident shard and 3.2 GB host RAM for the stored shards. Decided instead:

* The belief and the rollout buffer store **moments**, about today's footprint:
  * position: mean (2, fp32) and one sigma;
  * velocity: 2D mean and 2x2 covariance;
  * attitude: mean angle and spread;
  * health, power, shield delay, local log index: mean and sigma each;
  * cooldown: its 4-bin distribution directly (cheaper and exact);
  * angular velocity: taken from the enemy-action head's distribution, already stored.
* The categorical **input and baseline** are rebuilt inside the forward pass from the
  moments: position by the per-level blur curve λ(σ / d_ℓ); each velocity axis as the
  exact 1D Gaussian with variance `nᵀ Σ n` projected onto that axis, then binned. These
  activations are transient and micro-batched.
* The moments for the next step are decoded from the head's predicted distribution
  by **closed-form least squares only: no sampling, no branching search over
  candidates.** Per level, the mean is the probability-weighted circular mean of the
  colour positions (a complex phase); levels combine coarse to fine by unwrapping each
  finer phase against the coarser estimate, as the Fourier ladder decode does today.
  Sigma is a weighted least-squares fit of log per-level sharpness against the blur
  curve. Velocity: the 2D mean is the least-squares solution from the three axis
  means; the three axis variances give the 2x2 covariance exactly. Scalars: mean and
  variance of the histogram. Sharp
  targets keep the head calibrated, which is what makes the decoded sigma meaningful.
* Visible ships: mean = truth, sigma = floor.
* Accepted cost: the belief cannot represent multimodal positions ("left or right of
  the zone"). The sigma-calibration diagnostic will show whether that matters.

### 3.3 Position code: 9 colours

The plan's 5-colour Gaussian-integer code (base 2+i) is replaced by a **9-colour code,
base 3 on the square lattice**: level-0 lattice `(65536/3)·Z[i]`, spacing
`d_ℓ = 65536 / 3^(ℓ+1)`, colour = (x mod 3, y mod 3) of the cell corner at that level,
bilinear ≤4-hot over the cell's four corners (always distinct colours).

| | 5 colours (base 2+i) | 9 colours (base 3) |
|---|---:|---:|
| levels to ~4 px | 12 | 9 |
| logits | 60 | 81 |
| finest cell | 4.2 px | 3.3 px |
| scale step per level | 2.24x | 3x |
| target information per level, mean | 0.61 nats | 1.20 nats |
| same, at a cell centre | 0.22 nats | 0.81 nats |

Information is log N minus the entropy of the bilinear target, averaged over positions
in a cell. The dyadic 4-colour code (the analogue of today's Fourier ladder) averages
0.39 nats and carries none at a cell centre.

Reasons: twice the training signal per level for 21 more logits; both codes wrap the
65 536 px torus exactly; the 9-colour code is separable per axis, so decoding is
closed form per axis. Trade-offs accepted: coarser blur steps (3x against 2.24x) and the
same square-lattice axis bias at every level (the 5-colour code rotates 26.6° per level).
Both share the residue-code failure mode: losing a middle level costs about one coarser
cell of error.

**Why 9 among all possible colour counts.** A code that wraps the square torus and is
self-similar from level to level must step by multiplication by a Gaussian integer
`β = a + bi`. The colour count is then `N = a² + b²` and each level rotates by
`atan(b/a)`. So the possible counts are 4, 5, 8, 9, 10, 13, 16, 17, 18, 20, 25, ...;
3, 6, 7 and 12 are hexagonal-lattice counts and cannot tile a square torus. A real
base (2, 3, 4, 5) has no rotation and splits into independent x and y codes; a base
with `gcd(a, b) = 1` (2+i, 3+i, 3+2i, 4+i) rotates and its colours form one cycle of
length N. N = 2 fails because two corners of a cell share a colour.

| N | base | rotation | per-axis | levels to ~4 px | finest | logits | step | info mean | info min | info per logit |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2 | 0° | yes | 14 | 4.0 px | 56 | 2.00x | 0.39 | 0.00 | 0.097 |
| 5 | 2+i | 26.6° | no | 12 | 4.2 px | 60 | 2.24x | 0.61 | 0.22 | 0.122 |
| 8 | 2+2i | 45° | no | 10 | 2.0 px | 80 | 2.83x | 1.08 | 0.69 | 0.135 |
| 9 | 3 | 0° | yes | 9 | 3.3 px | 81 | 3.00x | 1.20 | 0.81 | 0.133 |
| 10 | 3+i | 18.4° | no | 9 | 2.1 px | 90 | 3.16x | 1.30 | 0.92 | 0.130 |
| 13 | 3+2i | 33.7° | no | 8 | 2.3 px | 104 | 3.61x | 1.57 | 1.18 | 0.120 |
| 16 | 4 | 0° | yes | 7 | 4.0 px | 112 | 4.00x | 1.77 | 1.39 | 0.111 |
| 17 | 4+i | 14.0° | no | 7 | 3.2 px | 119 | 4.12x | 1.83 | 1.45 | 0.108 |
| 25 | 5 | 0° | yes | 6 | 4.2 px | 150 | 5.00x | 2.22 | 1.83 | 0.089 |

Information per logit peaks at N = 8 to 10. Above that, logits grow faster than the
information they carry, the error from losing one level grows with the step (3.6x at
13, 4x at 16), and blur steps coarsen. 16 is the dyadic code with pairs of levels fused.
9 is kept for its per-axis decode; 10 (3+i) is the swap-in if square-lattice axis bias
shows up, since its 18.4° rotation per level averages the bias out.

Joint 9-way softmax per level, 9 levels, finest 3.3 px. A factorised 2×3-way variant
(54 logits) is the cheaper fallback.

### 3.4 Attitude: 4-colour, 4-level circular code

The 1D analogue of the position code. Level ℓ has 4 colours spaced
`360° / 4^(ℓ+1)` apart and repeats every `360° / 4^ℓ`; each level is a two-hot between
the two neighbouring colours. It wraps the circle exactly. Colour spacings by level:
90°, 22.5°, 5.6°, 1.4°. **16 logits, 1.4° finest**, against the plan's flat 72 bins at
5°. Aiming needs about 1.5°: a 10 px hull at 200–400 px subtends 1.4°–2.9°.

| colours per level | levels | logits | finest | info mean | info at midpoint | per logit |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 8 | 16 | 1.4° | 0.19 | 0 | 0.10 |
| 3 | 5 | 15 | 1.5° | 0.60 | 0.41 | 0.20 |
| **4** | **4** | **16** | **1.4°** | **0.89** | **0.69** | **0.22** |
| 6 | 3 | 18 | 1.7° | 1.29 | 1.10 | 0.22 |

Information is `ln k` minus the entropy of the two-hot target, in nats.

Known wrinkle: an enemy's next attitude is a five-spike mixture (heading plus 0°, ±5°,
±15°) spanning 30°, wider than the two finer levels' periods (22.5°, 5.6°), so those
levels sit near uniform for enemies. Training is unaffected (per-level cross-entropy
against sharp truth is proper) and the belief collapses to mean and spread anyway.

Attitude is stored and predicted like every other channel; it is **not** derived from
the velocity heading and turn offset (see §3.6).

### 3.5 Other channels

Sharp targets everywhere. Bins are sized to resolve the events that matter (a hit, a
shield-delay reset, a field transition) with natural values on bin centres; slow
deterministic drift is cheap because the residual baseline makes "no change" free.

| channel | range | bins | spacing | event resolved | drift per decision |
|---|---|---:|---|---|---|
| health | 0–100 | 21 | 5 hp | 10 hp hit = 2 bins | recharge 0.5 hp, 0.1 bin |
| power | 0–100 | 21 | 5 | boost drain | ~0.15 bin |
| shield delay | 0–5 s | 21 | 0.25 s | reset to 5 s is the last bin | countdown, 0.13 bin |
| local log index | −ln 2 to ln 2 | 21 | 0.069 | the five plateaus land on bins 0, 5, 10, 15, 20 | ~0.7 bin |
| cooldown | 0–0.1 s | 4 | 1 tick | exactly 3 ticks | — |
| angular velocity | 5 values | 5 | — | exact | — |

* **Local log index** goes from 5 to 21 bins because transitions between plateaus
  matter and the per-decision change (about 0.05) moved a 5-bin two-hot by only 0.14
  bin. It is stored and predicted, not computed from the field map (see §3.6).
* **Angular velocity** stays at 5 bins: in both flight paths the stored value is exactly
  `turn_offset / dt` (0, ±5°, ±15° per tick; 0 when stalled) and does not depend on speed
  or index. The rotation of the velocity heading under lift is carried by the velocity
  and attitude channels. Predicted by the next-state head like every other channel
  (see §3.6); the enemy-action head's turn marginal is a diagnostic to compare it with,
  not its source. (Raised in review as possibly 21 bins; kept at 5 on this evidence.)
* **Velocity** keeps the symlog-with-knee encoding at v₀ ≈ 100 px/s, as **three 1D axes**
  at 0°/120°/240°, 81 bins each (243 logits). Decided over a hex barycentric two-hot:
  with moment storage the belief is unimodal anyway. The three axis variances determine
  the 2x2 covariance exactly, so the moment round trip is exact.
* Going finer than 21 (e.g. 51 for health and shield delay) is worth trying only if
  those channels sit at the identity baseline after training.
* Gaussian uncertainty terms are removed; `time_since_observation` stays.

### 3.6 Hidden ships: model outputs and last sighting only

A hidden ship's input is built **only** from the next-state head's own outputs, carried
forward from the last visible state. Nothing is computed from physics or the map for it:
no attitude from heading plus turn offset, no log index from the field map at the
believed position, no angular velocity from the action head. This keeps the head's
job, learning the dynamics, as the owner's earlier decision required (no physics prior).

Before the first run, set `next_state_coef` from a measured trunk gradient share; the
head's norm will differ from the Gaussian head's (76% of the trunk in run 748).

---

## 4. Map scale

* **No training at 50v50 or any other fleet size.** Zero-shot fleet transfer is part of
  the project's point; it worked for dogfighting and is the goal for Frontline.
* Map-scale randomisation at fixed ship count is for **representation coverage only**:
  it exercises the coarse code levels and low-frequency RoPE pairs that 50v50 positions
  use and 5v5 never does. At ten ships, scale s divides areal density by s², so per-scale
  metrics are sparse-map metrics, not transfer evidence. The crossover evaluation is the
  transfer evidence.
* **Open:** the scale range and distribution. The review suggested capping near s = 4
  (covers 50v50's s = 3.16 with margin; above s = 6 nearly every episode times out) and
  logging ships-in-sight alongside s.
* Implementation gaps to close if kept: field generation reads scalar radii
  (`field_generation.py`), and the scripted strategy reads a scalar zone radius
  (`frontline_strategy.py`, scaled once in `evaluation/agents.py`).

---

## 5. Build order and gates

**Decided: everything lands together and is tested once.** Rewards, critic, next-state
head, belief and map scale go in as one change with one schema bump and one cold start.
No per-change config switches, no intermediate A/Bs (five heads against fifteen and
PMPO are not run). Coefficients set from measured gradient norms (`next_state_coef`, the
BC loss balance) are re-measured **once, at the end**.

Gates before the run: the per-head zero-sum test at ratio 1; the encode/decode
round-trip tests (§6). Read the run against the dead-reckoning and persistence baselines
already logged, not against the Gaussian head; check cooldown first, where a free "no
change" was the audit's diagnosis.


---

## 6. Review outcomes and remaining concerns

Resolved in review:

| # | item | decision |
|---|---|---|
| 1 | bundling many changes into one step | accepted: everything at once, tested once (§5) |
| 2 | first hidden step starts sharp and frozen | accepted risk; watch hidden-age position error against dead reckoning from the first updates |
| 3 | encode/decode round trip | must be exact; closed-form least squares, no sampling or branching (§3.2). Unit test each code: 9-colour position, 4-colour attitude, three velocity axes, 21-bin scalars near their range edges where clipping distorts the Gaussian |
| 4 | advantage floor unit | follow DreamerV3 exactly, verified (§2.3); a win is 1 |
| 5 | critic bins | 51, with a narrowed range (§2.1) |
| 6 | zone progress gamma | see below |
| 7 | map scale breaks the scripted agents | **required work**: field generation and the scripted strategy must read per-env geometry (`state.zone_radius`, per-env field radii) before scale randomisation is on |
| 8 | coarse position levels near-constant at s = 1 | accepted |
| 9 | bimodal hidden health and shield delay | accepted limitation; log hidden health and shield-delay error by hidden age |
| 10 | retuning after schema changes | once, at the end |
| 11 | unverified DreamerV3 numbers | verified from the official repository (§2.3) |
| 12 | five-vs-fifteen heads, PMPO | not run |

**Zone progress gamma.** The payment lag is one tick when the zone is uncontested:
progress is paid on the tick a ship's presence moves the meter. But gamma sets how far
back credit reaches, and the decision that earns progress is choosing a zone and flying
to it, several seconds before the first payment (adjacent zones on the 1 200 px ring
are about 1 400 px apart, roughly 14 s at cruise). Repeated payments after arrival also
credit the approach. That is an engagement-length horizon, so **0.995 / lambda 0.95**
(the kill-and-death values) stays; damage's 0.991 would be too short.

Still open:

* **Map-scale range and distribution** (§4).

---

## Sources

Most research was done through search summaries because arxiv and several publisher
sites were blocked by the network proxy. The DreamerV3 settings in §2.3 were then
verified directly against `github.com/danijar/dreamerv3` (main branch, fetched
October 5, 2026).

* Hafner et al., *Mastering Diverse Domains through World Models* (DreamerV3).
* Hafner, Yan et al., *Training Agents Inside of Scalable World Models* (Dreamer 4),
  arXiv 2509.24527; settings cross-checked against
  github.com/vijayabhaskar-ev/dreamer_v4 (255 symexp two-hot bins for reward and value,
  gamma 0.997, lambda 0.95, PMPO alpha 0.5, beta 0.3, reverse KL).
* Ye et al., *Towards Playing Full MOBA Games with Deep RL* (Honor of Kings,
  multi-head value over zero-sum reward groups), NeurIPS 2020.
* Berner et al., *Dota 2 with Large Scale Deep RL* (OpenAI Five: zero-sum by
  subtracting the enemy mean, team spirit).
* Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*
  (MAPPO, value normalisation).
* Farebrother et al., *Stop Regressing: Training Value Functions via Classification*
  (HL-Gauss), 2024.
* van Seijen et al., *Hybrid Reward Architecture*, 2017 (many fine heads; the counter-case).
