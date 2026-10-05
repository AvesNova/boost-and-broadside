# Frontline reward, critic and belief design

Status: **agreed design, not implemented.** Base: `dev` at `4eba8a1`.

One change set covering four areas:

1. rewards: five per-level critic heads with derived, zero-sum weights;
2. the critic and advantages: raw rewards, a categorical critic per head, DreamerV3's
   return normalisation;
3. the next-state head and belief: categorical predictions with sharp targets, a
   belief stored as moments, and new position, velocity and attitude codes;
4. per-episode map-scale randomisation.

The guiding rule is to follow published systems rather than invent: DreamerV3 and
Dreamer 4 for the critic and advantage plumbing, Honor of Kings for the reward
structure, OpenAI Five for the zero-sum construction and team mixing (§2.5).

Everything lands together, with one checkpoint-schema bump and one cold start, and is
tested once (§6). Physics numbers used for sizing are in Appendix A.

---

## 1. Rewards

### 1.1 Principles

* **Rewards own spatial credit.** Every reward is written onto the ship that earned it.
  PPO solves temporal credit only, through per-head GAE. The only quantity broadcast
  across a team is the match outcome.
* **Each level is zero-sum.** An event charges the side it happens to and pays the side
  that caused it (§1.3).
* **Rewards, returns and advantages are scalars.** Only the critic's output is a
  distribution (§2).

### 1.2 Five heads

Events are grouped into five levels. Each level is **one critic head** whose reward is
the sum of that level's events. Events are still computed and logged one by one; they
do not get their own value heads.

| head | events summed into it | gamma | GAE lambda | horizon | read from |
|---|---|---:|---:|---|---|
| outcome | match result | 0.9997 | 0.97 | ~3 300 decisions, the match | global token, categorical (§2.2) |
| zone capture | completion ledger | 0.999 | 0.97 | ~1 000 decisions | ship token |
| zone progress | live meter movement | 0.995 | 0.95 | ~200 decisions | ship token |
| kill and death | death, kill of an enemy, kill of an ally | 0.995 | 0.95 | ~200 decisions, an engagement | ship token |
| damage | damage taken and dealt, friendly fire, recharge, charge-back | 0.991 | 0.90 | ~110 decisions, an exchange | ship token |

Values use the per-tick convention of `COMPONENT_GAMMAS_PER_TICK` in
`config/defaults.py`; Frontline runs `action_repeat` 1 at 30 Hz, so per tick is per
decision. The horizon is 1 / (1 − gamma).

**Why five, not one per event.** GAE is linear in rewards and values, so at a shared
gamma and lambda the advantage of a sum equals the sum of the advantages. Splitting
events inside a level therefore does not change the policy gradient; it only changes
how the critic is fitted. The reason split components existed, letting a scalar critic
tell "an uncertain ±1 is coming" from "certainly nothing", is covered by the
categorical critic, which represents those as different distributions with the same
mean. One thing a merged kill-and-death head cannot tell apart is a mutual trade (kill
and die, net about zero) from a quiet spell, because the realised return is the same.
That does not affect the policy gradient.

**Why not one head.** The levels need different horizons, the outcome is read from the
global token while the rest are per ship, and a sparse ±1 outcome summed with dense
damage would dominate the critic's error.

**Why zone capture and zone progress are separate.** They have different time scales,
the same relationship as death to damage. Progress is paid on the tick a ship's
presence moves the meter (one tick when uncontested). A completion is paid only when
the meter fills, at least `capture_seconds` (10 s, 300 decisions) later at a one-ship
lead. Gamma sets how far back credit reaches, and the decision that earns progress is
choosing a zone and flying to it: adjacent zones on the 1 200 px ring are about
1 400 px apart, roughly 14 s at cruise. That is an engagement-length horizon, so
progress takes the kill-and-death values; damage's 0.991 would be too short.

This is the Honor of Kings layout: zero-sum reward groups, per-agent rewards, one value
head per group, advantages combined by weight.

### 1.3 Balance

* **Rule.** An event charges the side it happens to `w` and pays the side that caused
  it `ratio × w`. `kill_payout_ratio`, `damage_payout_ratio` and `capture_payout_ratio`
  start at 2.0; the schedule's `offensive_bias` holds them to 50M steps, anneals them
  linearly to 1.0 by 300M, then holds 1:1. At 1:1 every event is exactly zero-sum.
  The premium exists because priced evenly, an even trade is worth nothing and a policy
  that cannot reliably win it declines it.
* **Payee.** If a ship on the opposing side caused the event, that ship is paid. If
  nobody on the opposing side did, the counterpart is split evenly over the opposing
  team with `enemy_team_share`, dividing by the full team slot count including dead
  ships. The only non-bullet damage source is the play-area boundary, so this covers
  boundary damage and death, an enemy's friendly fire, and recharge of boundary damage.
* **Invariant, tested.** At all ratios 1, each head's reward summed over both teams is
  zero on every tick.
* **Free parameters.** One weight per level, plus the offensive bias. Everything else
  is derived. Starting values carry the current balance: win 1.0, zone capture 2.0,
  zone progress 2.0, death 0.283 per death, damage 0.274. The schedule's tier scales
  multiply them as today (outcome scale for the outcome and both zone heads, kill/death
  scale, damage scale).
* **The unit.** A win is 1. This is the unit DreamerV3's advantage floor is measured in
  (§2.3).

### 1.4 Events

**Outcome.** Win, tie or loss at the terminal transition; a 9 000-step timeout is a real
tie. Read and valued per environment from the global token and broadcast to every
allied ship.

**Zone capture.** Per active zone, two ledgers over the current capture attempt, with
progress P running from 0 to 1:

* attacker presence C_i: each tick the meter moves by Δ > 0 toward the attacker,
  `C_i += Δ / n_present` for each present attacking ship;
* defender absence A_j: on the same tick, `A_j += Δ / n_absent` for each absent defending
  ship, dead ships counting as absent.

So Σ C_i = Σ A_j = P. On a reversal by r both ledgers scale pro rata,
`C, A ← C, A · (1 − r/P)`. On completion:

```
zone_gain_i = ratio · w · [ ½ · present_i / n_present + ½ · C_i / P ]
zone_loss_j =       − w · [ ½ · absent_j  / n_absent  + ½ · A_j / P ]
```

Ledgers clear on completion, whenever roles rotate (which also zeroes every other
zone's meter), and when the meter returns to zero (guard P = 0). If the charged set is
empty the whole opposing side is charged; this is never reached in practice, because
`_advance_capture_state` moves a meter only on a strict alive majority inside the zone,
so the other side always has a ship dead or elsewhere.

**Zone progress.** Today's `_ZoneCreditReward` split, summed into one head: on every
tick a meter moves, the favoured side's ships inside the zone share `ratio × Δ`, and the
other side's ships outside it, dead included, share the charge `Δ`. Ships of the losing
side who are inside are contesting and are not charged. The completion tick is excluded
(the zone capture head pays it).

**Kill and death.**

* Death: `−U` to the dying ship, any cause.
* Outstanding-damage ledger O[a, t]: applied damage from attacker a to target t, the
  existing `cumulative_damage_matrix`, cleared at respawn. When t recharges by h, with
  S_t = Σ_a O[a, t]: `O[a, t] ← O[a, t] · (1 − h / S_t)` (pro rata, no queue).
* Share of a death of t: `share_a = ½ · raw_final[a, t] / Σ raw_final[·, t] + ½ · O[a, t] / S_t`.
  `raw_final` is the raw impact on the death tick: the killing hit lands on zero shield,
  so its *applied* damage is 0. If S_t = 0 (the shield was emptied by the boundary)
  the whole share goes to the raw final hit.
* Kill of an enemy: `+ratio · U · share_a`. Kill of an ally: `−ratio · U · share_a`, with
  the counterpart paid to the opposing team. A boundary death pays the opposing team by
  `enemy_team_share`. A ship is never blamed for its own death.

**Damage.**

* Damage taken: `−V · applied`, every source including the boundary. This is what stops
  boundary recharge farming.
* Damage dealt to an enemy: `+ratio · V` per unit, shared among simultaneous hits by
  applied damage. Boundary damage pays this to the opposing team.
* Damage dealt to an ally: `−ratio · V` per unit to the shooter; the counterpart is paid
  to the opposing team.
* Shield recharge: `+V · h` to the recharging ship.
* Charge-back: `−ratio · V · h · O[a, t] / S_t` to each attacker, taken before the
  ledger update, exactly reversing the payment for damage that no longer exists.
  Recharge of unattributed damage charges the opposing team. Recharge starts 150
  decisions after the last hit (5 s delay), so at gamma 0.991 the charge-back is worth
  0.991^150 ≈ 0.26 of the payment it reverses: it is correct bookkeeping, not the
  anti-farming mechanism. Log its realised share of the damage head.

**Removed:** `ally_win`, `enemy_win`, the token-weight `outcome` probe, `boundary`,
`boundary_damage`, the `ally_/enemy_combat_*` pair, the shaping components (facing,
closing speed, shoot quality, shooting penalty, speed), and `kill_shot_fraction`
(fixed at ½ by the share rule).

### 1.5 Team mixing

The lambda matrix (`_lambda_matrix`, `_precompute_lambda_aggregates`, `enemy_neg_k`,
`ally_zero_k`, `_LOCAL_COMPONENTS`) is deleted. Removing it does not change behaviour:
every component with nonzero weight is already diagonal, or bit-identical across a side
(`tests/train/test_global_value.py` asserts the latter). In its place each head keeps
one scalar, OpenAI Five's "team spirit":
`adv_i + s · mean_{j ≠ i, same side} adv_j`, default **0** (pure per-ship credit), so a
selfish equilibrium has a cheap remedy. The reward-decomposed gradient diagnostic reads
per-head advantages directly.

---

## 2. Critic and advantages

### 2.1 Raw rewards and a categorical critic

* **Rewards are stored raw.** The `symlog` applied at storage in `RolloutBuffer` is
  removed; GAE runs on raw rewards. Today's storage symlog is not what DreamerV3 does:
  a discounted sum of symlogged rewards is not the symlog of the return, and it
  compresses dense and sparse components differently.
* **The critic's output is a distribution; everything else stays scalar.** Each head
  emits logits over fixed bins whose locations are symlog-spaced. Its value V is the
  expectation, Σ probability × bin value, in raw units; that V is what GAE uses.
* **Target.** The realised scalar lambda-return, two-hot over the bins (DreamerV3),
  trained by cross-entropy. The target's mean equals the return, so what the value means
  is unchanged; only how the critic is fitted changes.
* **51 bins per head.** DreamerV3 uses 255 over `symexp(linspace(-20, 20))` because it
  spans many domains. At 51 bins that span would give a spacing of 0.8 symlog units,
  with the first bins at ±1.2 and ±4.0 raw: far too coarse for these returns. The range
  is narrowed to each head's measured largest |return| with margin, for example ±5
  symlog units (±147 raw), spacing 0.2.
* **Removed:** the per-component `ReturnScaler` for the critic (the bin spacing is the
  single place values are compressed), the Huber value loss and any value clipping.
  Explained variance is computed on the expectation.

### 2.2 Outcome head: four classes

Classes **win, tie, loss, unresolved**. Value on the ±1 scale is V = P(win) − P(loss).

Gamma stays 0.9997: at gamma 1, with ~8 600-step episodes and 128-step rollouts, the
value bootstraps about 67 times before any terminal grounds it with no contraction to
damp error, and run 737 measured win explained variance falling from 0.994 to 0.42.
The discount is modelled as probability mass leaking into **unresolved**, worth 0:

```
Q_t = gamma · [(1 − lambda) · p(s_{t+1}) + lambda · Q_{t+1}] + (1 − gamma) · e_unresolved
Q_t = onehot(result)        on the terminal transition
```

Its expectation equals the scalar lambda-return, so the categorical target and the
scalar GAE advantage agree exactly. Loss: cross-entropy against Q_t.

The fourth class keeps **tie** honest. Without it the discount mass would have to go
into tie, and at gamma 0.9997 most of the target would be discount rather than outcome:

| decisions remaining | mass on win + loss | mass from discount alone |
|---:|---:|---:|
| 9 000 | 0.07 | 0.93 |
| 4 500 | 0.26 | 0.74 |
| 1 000 | 0.74 | 0.26 |
| 100 | 0.97 | 0.03 |

Calibration is then checkable as P(win | resolved) against realised wins.

### 2.3 One advantage, normalised once

Each head's scalar advantage is weighted by its level weight, summed, and the sum is
normalised once, exactly as DreamerV3 does. Settings verified against the official
source (`github.com/danijar/dreamerv3`, main branch, fetched October 5, 2026:
`dreamerv3/configs.yaml`, `imag_loss` in `dreamerv3/agent.py`, `Normalize` in
`embodied/jax/utils.py`):

| setting | DreamerV3 |
|---|---|
| return normaliser | `impl: perc, rate: 0.01, limit: 1.0, perclo: 5.0, perchi: 95.0, debias: False` |
| scale | `max(limit, hi − lo)` of EMA 5th and 95th percentiles of the lambda-returns (EMA decay 0.99) |
| advantage | `(return − value) / scale`; the mean is **not** subtracted |
| value and advantage normalisers | none |
| critic and reward heads | symexp two-hot, 255 bins, `symexp(linspace(-20, 20))` |
| discount, lambda | horizon 333 (gamma ≈ 0.997), 0.95 |
| actor entropy | 3e-4 |
| critic regulariser | toward an EMA copy of the critic, rate 0.02 (not adopted: PPO's fixed rollout targets do that job) |

So the scale is the summed return's 5th–95th percentile spread, EMA decay 0.99, floored
at 1; the summed advantage is divided by it; PPO's per-minibatch advantage
standardisation is dropped.

This replaces the per-component `AdvantageScaler`, which normalised each component to
unit RMS *before* weighting and so cancelled the derived weights. After the change the
level weights reach the policy exactly as derived, and splitting or merging events
inside a level cannot change its gradient share.

**Why a win is the unit.** The floor means advantages are never amplified while the
return spread is under one unit. With a win at 1 the floor binds only when outcomes
barely vary across the batch, as in run 735's stalemate where 90% of games timed out; in
that regime small kill and damage differences stay at their natural size rather than
being blown up into the whole update.

### 2.4 Policy objective

PPO is unchanged apart from the advantage normalisation above.

### 2.5 Precedent

| system | reward and credit | critic | advantage scaling | policy loss |
|---|---|---|---|---|
| DreamerV3 | one raw scalar reward | symexp two-hot, 255 bins | percentile spread, max(1, S) | actor-critic |
| Dreamer 4 | same | symexp two-hot reward and value | none | PMPO, sign of the advantage, KL to the BC prior |
| Honor of Kings | five zero-sum groups (farming, KDA, damage, pushing, win/lose), per hero | one value head per group | per-group weights | dual-clip PPO |
| OpenAI Five | zero-sum by subtracting the enemy mean; "team spirit" mixes own and team reward | single | — | PPO |
| MAPPO | shared team reward | centralised | running mean and std of value targets | PPO |

Learned credit-assignment methods (COMA, QMIX, partial reward decoupling, learned
internal rewards in Capture the Flag) target cooperative games with one shared reward.
This environment attributes every event to a ship already; the strongest results in
team-versus-team games use hand-built zero-sum per-agent rewards.

---

## 3. Next-state head and belief

### 3.1 What the current head does

Measured on run 747's final checkpoint (`next-state-head-audit-sep2026.md`, 2.68M scored
transitions), mean absolute error:

| | ally visible | enemy visible | enemy hidden |
|---|---|---|---|
| position | 0.24 px, 4× worse than `pos + vel·dt` | 0.20 px, 3.9× worse | 527 px, equal to "nothing changed" |
| velocity | 4.78 px/s, 1.5× worse than "nothing changed" | 4.69 px/s, 1.7× worse | 83.5 px/s, equal |
| health | 1.26 hp, 4.5× worse | 1.27 hp, 3.5× worse | 3.3 hp, equal |
| cooldown | 1.15× worse | 1.19× worse | equal |

For hidden enemies the belief was 93× worse than dead reckoning at 0.1–0.5 s of
occlusion and only overtook it after about 10 s; 70% of hidden tokens are under 5 s
old. Nine of ten position Fourier harmonics on a visible ship were 10× to 500 000× worse
than predicting no change; only the one harmonic carrying a sigma trained, and the
visible-ship sigmas were 3–6× too wide, which throttled the Gaussian likelihood's
gradient where the dynamics are learnable.

The design below addresses the gradient starvation (per-level cross-entropy with no
learned sigma), the cost of "no change" (a residual head) and the sigma calibration.
It deliberately gives the head **no physics prior**: the head exists as a hard
auxiliary task, and learning the dynamics is its job. Dead reckoning is the bar it is
measured against, not a shortcut.

### 3.2 The head

* Residual logits on a baseline built from the current input:
  `z_{t+1} = log p̃_t + f(h_t)`, with `f`'s last layer zero-initialised so the head starts
  as "nothing changes", and `p̃` floored by a small epsilon.
* **Targets are sharp for every channel and every token, hidden ones included.**
  Cross-entropy against the sharp truth is a proper scoring rule: at a level the head
  cannot resolve, its optimum is the conditional distribution, which is near uniform.
  Targets are built on the fly from truth and never stored.
* **Inputs and baselines are smoothed** (§3.3). With a calibrated sigma the smoothed
  baseline already equals the best prediction at an unresolvable level, so the head
  outputs zeros there instead of memorising and cancelling a stale sharp pattern.
  Smoothing the input, not the target, keeps the loss proper and lets the head beat its
  prior wherever it has information.
* The 13 Gaussian uncertainty terms are removed. `time_since_observation` stays.
* The ship-token input becomes mostly sparse multi-hot codes; initialise the encoder's
  first layer per input unit, like an embedding table.
* `next_state_coef` is set from a measured trunk-gradient share (the Gaussian head held
  76% of the trunk in run 748).

### 3.3 Belief

**Storage: moments, not distributions.** Storing the full categorical distribution of
every channel would add about 1.1 GB of GPU buffer per resident rollout shard and
3.2 GB of host RAM for the stored shards (one shard is 128 steps × 960 envs × 10 ships =
1.23M ship tokens, about 440 extra bf16 channels each). The belief and the rollout
buffer store moments instead:

| channel | stored | floats |
|---|---|---:|
| position | mean (fp32) and one sigma | 3 |
| velocity | 2D mean and 2×2 covariance | 5 |
| attitude | mean angle and spread | 2 |
| health, power, shield delay, local log index | mean and sigma each | 8 |
| angular velocity, cooldown | the distribution (5 and 4 bins) | 9 |

About 27 floats per ship, against today's 24.

**Rebuilding the input.** Inside the forward pass the categorical input, which is also
the head's baseline, is rebuilt from the moments: position and attitude by the per-level
blur curve, each velocity axis as the 1D Gaussian with variance `nᵀ Σ n` projected onto
that axis, scalars as a discretised Gaussian over their bins. These activations are
transient and micro-batched; the buffer stays at today's size.

**Visible ships:** mean = truth, sigma = floor. Allies are always visible.

**Hidden ships use only model outputs, carried forward from the last sighting.** Nothing
is computed from physics or the map: no attitude from heading plus turn offset, no log
index from the field map, no angular velocity from the enemy-action head.

**Decoding is closed-form least squares only: no sampling, no branching search over
candidates.**

* Coded channels (position, attitude): per level, the probability-weighted circular
  mean of the colour positions as a complex phase; levels combine coarse to fine by
  unwrapping each finer phase against the coarser estimate. Sigma is a weighted
  least-squares fit of log per-level sharpness against the blur curve.
* Velocity: the 2D mean is the least-squares solution from the three axis means; the
  three axis variances determine the 2×2 covariance exactly.
* Scalars: mean and variance of the histogram.

**Round trip must be exact.** With a zero residual, decoding the rebuilt input must
return the same moments, or the belief drifts by itself every step. Each code has a
round-trip unit test, including the scalars near their range edges where clipping
distorts the Gaussian.

**Accepted limitations.**

* The belief is unimodal: it cannot say "left or right of the zone".
* Hidden health and shield delay are really bimodal (the ship took a hit or did not;
  the delay is counting down or reset to 5 s). A mean and sigma blurs that into one bump.
  Log hidden health and shield-delay error by hidden age.
* When a ship disappears its belief is the last visible state at the sigma floor, a
  sharp baseline. Widening and moving it is a learned skill with no prior; this is where
  the design is most likely to fall short. Watch hidden-age position error against dead
  reckoning from the first updates.

### 3.4 Position: 9-colour nested code

**Construction.** Base 3 on the square lattice. Level ℓ has cell spacing
`d_ℓ = 65 536 / 3^(ℓ+1)` px. A point's colour at level ℓ is (x mod 3, y mod 3) of its
cell corner in units of d_ℓ, nine colours. Each level is a bilinear multi-hot over the
cell's four corners, which always have distinct colours. Every level wraps the
65 536 px torus exactly. Nine levels, 81 logits:

| level | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spacing (px) | 21 845 | 7 282 | 2 427 | 809 | 270 | 90 | 30 | 10.0 | 3.3 |

The finest cell, 3.3 px, matches cruise motion per decision (3.3 px at 100 px/s), so
small moves shift the bilinear weights enough to train on. The code splits by axis:
each level is a pair of independent base-3 codes, so decoding runs per axis.

**Why 9 colours.** A code that wraps the square torus and is self-similar from level to
level must step by multiplication by a Gaussian integer a + bi. The colour count is then
a² + b² and each level rotates by atan(b/a), so the possible counts are 4, 5, 8, 9, 10,
13, 16, 17, 18, 20, 25, … (3, 6, 7 and 12 belong to hexagonal lattices, which cannot tile
a square torus; 2 fails because two cell corners share a colour). Information is
ln N minus the entropy of the bilinear target, averaged over positions in a cell, in
nats:

| colours | base | rotation | per axis | levels to ~4 px | finest | logits | step | info mean | info min | per logit |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2 | 0° | yes | 14 | 4.0 px | 56 | 2.00× | 0.39 | 0.00 | 0.097 |
| 5 | 2+i | 26.6° | no | 12 | 4.2 px | 60 | 2.24× | 0.61 | 0.22 | 0.122 |
| 8 | 2+2i | 45° | no | 10 | 2.0 px | 80 | 2.83× | 1.08 | 0.69 | 0.135 |
| **9** | **3** | **0°** | **yes** | **9** | **3.3 px** | **81** | **3.00×** | **1.20** | **0.81** | **0.133** |
| 10 | 3+i | 18.4° | no | 9 | 2.1 px | 90 | 3.16× | 1.30 | 0.92 | 0.130 |
| 13 | 3+2i | 33.7° | no | 8 | 2.3 px | 104 | 3.61× | 1.57 | 1.18 | 0.120 |
| 16 | 4 | 0° | yes | 7 | 4.0 px | 112 | 4.00× | 1.77 | 1.39 | 0.111 |
| 17 | 4+i | 14.0° | no | 7 | 3.2 px | 119 | 4.12× | 1.83 | 1.45 | 0.108 |
| 25 | 5 | 0° | yes | 6 | 4.2 px | 150 | 5.00× | 2.22 | 1.83 | 0.089 |

With few colours the bilinear target lights nearly all of them, so a level says little
(the 4-colour code, the analogue of today's dyadic Fourier ladder, carries nothing at a
cell centre). With many, logits grow faster than the information they carry, the error
from losing one level grows with the step, and the belief blur moves in coarser steps.
Information per logit peaks at 8–10 colours; 9 sits at the peak and decodes per axis.
10 (3+i) is the fallback if square-lattice axis bias shows up, since its 18.4° rotation
per level averages the bias out.

**Known properties.** Losing a middle level costs about one coarser cell of error (the
residue-code failure mode; the old Fourier ladder had it at 512–4 096 px). Levels 0–1
barely change within the 2 600 px playable radius at map scale 1 and only encode the
random map centre; they matter on large maps.

**RoPE** stays on physical coordinates: truth for visible ships, the decoded mean for
hidden ones.

### 3.5 Velocity: three axes

`u = v̂ · v₀ · log(1 + |v| / v₀)` with v₀ ≈ 100 px/s: roughly linear across the playing
band (about 20–215 px/s) and logarithmic only in the unlimited-boost tail. Cells are
nearly isotropic (about 1.4× elongation). u is projected onto unit vectors at 0°, 120°
and 240°; each projection is a 1D two-hot over 81 bins of 5 u-units, covering |u| ≤ 199
(631 px/s, the highest unlimited-boost speed). 243 logits.

| speed (px/s) | 20 | 100 | 215 | 631 |
|---|---:|---:|---:|---:|
| spacing along / across velocity (px/s) | 6 / 6 | 10 / 7 | 16 / 9 | 37 / 16 |

Today's knee at 1 px/s gives 152 / 33 px/s at 100 px/s. The measured per-decision Δv is
about 3.9 px/s RMS, about 0.4 bin.

Three marginals cannot represent a bimodal joint (turned left or right). With moment
storage the belief is unimodal anyway, and the three axis variances recover the 2×2
covariance exactly, so the round trip loses nothing.

### 3.6 Attitude: 4-colour circular code

The 1D analogue of the position code. Level ℓ has 4 colours spaced `360° / 4^(ℓ+1)`
apart and repeats every `360° / 4^ℓ`; each level is a two-hot between the two
neighbouring colours, and the code wraps the circle exactly. Four levels with spacings
90°, 22.5°, 5.6° and 1.4°: **16 logits, 1.4° finest.**

Aiming needs about 1.5°: a 10 px hull at 200–400 px subtends 1.4°–2.9°. A flat 72-bin
code (5°) carries the angle only as interpolation weights, and today's input is four
Fourier harmonics whose finest period is 45°.

| colours per level | levels | logits | finest | info mean | info at midpoint | per logit |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 8 | 16 | 1.4° | 0.19 | 0 | 0.10 |
| 3 | 5 | 15 | 1.5° | 0.60 | 0.41 | 0.20 |
| **4** | **4** | **16** | **1.4°** | **0.89** | **0.69** | **0.22** |
| 6 | 3 | 18 | 1.7° | 1.29 | 1.10 | 0.22 |

An enemy's next attitude is a mixture of five spikes (heading plus 0°, ±5°, ±15°) spanning
30°, wider than the two finer levels' periods, so those levels sit near uniform for
enemies. Training is unaffected (per-level cross-entropy is proper); for allies the turn
is known and every level is predictable.

### 3.7 Other channels

Bins are sized to resolve the events that matter with natural values on bin centres;
slow deterministic drift is cheap because the residual baseline makes "no change" free.

| channel | range | bins | spacing | event resolved | drift per decision |
|---|---|---:|---|---|---|
| health (shield) | 0–100 | 21 | 5 hp | a 10 hp hit is 2 bins | recharge 0.5 hp, 0.1 bin |
| power | 0–100 | 21 | 5 | boost drain | ~0.15 bin |
| shield delay | 0–5 s | 21 | 0.25 s | the reset to 5 s is the last bin | countdown, 0.13 bin |
| local log index | −ln 2 to ln 2 | 21 | 0.069 | the five plateaus (n = ½ … 2 in steps of √2) on bins 0, 5, 10, 15, 20 | ~0.05, 0.7 bin |
| cooldown | 0–0.1 s | 4 | 1 tick | exactly 3 ticks | — |
| angular velocity | 5 values | 5 | — | exact | — |

* **Local log index** transitions between plateaus matter; at 5 bins the per-decision
  change moved a two-hot by only 0.14 bin.
* **Angular velocity** is exactly `turn_offset / dt` in both flight paths (0, ±5°, ±15°
  per tick; 0 when stalled), independent of speed and index (`physics.py`). The rotation
  of the velocity heading under lift is carried by the velocity and attitude channels.
  The enemy-action head's turn prediction is a diagnostic to compare against, not its
  source.
* Finer bins (51) for health and shield delay are worth trying only if those channels
  sit at the "no change" baseline after training.

---

## 4. Map-scale randomisation

* **No training at other fleet sizes.** Zero-shot fleet-size transfer is part of the
  project's point: it worked for dogfighting and is the goal for Frontline. The
  crossover evaluation is the transfer evidence.
* **Purpose: representation coverage.** Positions at 50v50 span about 16 000 px and use
  coarse position levels and low-frequency RoPE pairs that the 5v5 map never exercises.
  Only the map size changes: ship count, episode length (9 000 steps), physics, vision
  and rewards are fixed. At ten ships, scale s divides areal density by s², so per-scale
  metrics are sparse-map metrics, not transfer evidence. Log ships-in-sight alongside s.
* **Mechanism.** s is sampled per environment at reset and applied through
  `scaled_frontline_geometry` (playable radius, zone radius, zone ring, field sizes and
  widths), whose quantities already live per environment in `TensorState`. s ≤ 12 keeps
  the playable radius (31 200 px) under the 32 768 px half-period.
* **Required before it is turned on.** Field generation (`field_generation.py`) and the
  scripted strategy (`frontline_strategy.py`, scaled once in `evaluation/agents.py`) read
  scalar radii; on a rescaled environment the behaviour-cloning teacher and the scripted
  opponent would be wrong. Both must read per-environment geometry.
* **Range and distribution: open.** One candidate: probability 0.5 at s = 1, otherwise
  s = exp(x) with x exponential at rate 1.5, truncated to [0, ln 12]:

  | | P |
  |---|---:|
  | s = 1 | 0.50 |
  | s ≤ 1.25 | 0.65 |
  | s ≥ 2 | 0.17 |
  | s ≥ 3.16 (50v50 scale) | 0.08 |
  | s ≥ 6 | 0.023 |

  At s = 12 the zone ring radius is 14 400 px, about 106 s at cruise against a 300 s
  episode, so large maps are nearly all timeouts at full cost. A cap near s = 4 covers
  the 50v50 scale with margin.

---

## 5. Diagnostics

* Position cross-entropy per level, visible against hidden; hidden bucketed by realised
  error. Watch the finest level for fast visible ships.
* Sigma calibration against realised error, by hidden age, for every channel.
* Hidden-age position and velocity error against dead reckoning and persistence (already
  logged as production diagnostics).
* Hidden health and shield-delay error by hidden age.
* Zone occupancy, front advances and timeout rate (run 735's collapse signature: zone
  occupancy fell from 0.075 to 0.0006 of live ship-steps and 90% of games timed out),
  bucketed by map scale and ships-in-sight.
* Per-head gradient share and per-head zero-sum residual at ratio 1.
* The charge-back's realised share of the damage head.
* Outcome calibration: P(win | resolved) against realised wins.

---

## 6. Implementation

* **One change set, tested once.** One schema bump and one cold start; no weight
  migration; no intermediate comparisons.
* **Gates before the run:** the per-head zero-sum test at ratio 1; the encode/decode
  round-trip tests (§3.3).
* **Retune once, at the end:** coefficients set from measured gradient norms
  (`next_state_coef`, the BC loss balance) are re-measured with
  `benchmarks/gradient_decomposition.py`.
* **Read the run** against the dead-reckoning and persistence baselines, not against the
  Gaussian head. Check cooldown first: making "no change" free was the audit's
  diagnosis for it.

Main code touched: `env/rewards.py` and `env/frontline.py` (events and ledgers),
`train/rl/ppo.py` (lambda matrix, advantage combination, value loss), `train/rl/buffer.py`
(raw rewards, normaliser), `models/yemong/policy.py` (heads),
`train/rl/physical_belief.py`, `train/rl/belief.py`, `train/rl/features.py`,
`env/observation.py` (codes and belief), `env/field_generation.py` and
`agents/frontline_strategy.py` (per-environment geometry), `config/defaults.py`, the
checkpoint schema and the tests.

---

## Appendix A. Physics calibration

Measured at 60 Hz (dt = 1/60 s), px/frame unless noted. Frontline runs at dt = 1/30, so
px/s values carry over and per-tick values double per decision; re-check at 30 Hz once.

**Steady-state speed** (px/frame; ×60 for px/s):

| n | coast straight | coast normal | coast sharp | boost (energy-limited) straight | normal | sharp | boost (unlimited) straight | normal | sharp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.500 | 3.331 | 2.715 | 1.310 | 3.586 | 3.131 | 1.940 | 10.519 | 8.576 | 4.138 |
| 0.707 | 2.356 | 1.920 | 0.926 | 2.847 | 2.486 | 1.541 | 7.442 | 6.068 | 2.927 |
| 1.000 | 1.666 | 1.358 | 0.655 | 2.260 | 1.974 | 1.224 | 5.265 | 4.292 | 2.071 |
| 1.414 | 1.178 | 0.960 | 0.463 | 1.794 | 1.567 | 0.972 | 3.724 | 3.036 | 1.465 |
| 2.000 | 0.833 | 0.679 | 0.328 | 1.424 | 1.244 | 0.772 | 2.634 | 2.147 | 1.036 |

Anchors: n = 1 coast cruise 100 px/s; n = 1 energy-limited boost 136 px/s; highest
energy-limited 215 px/s (n = 0.5); highest unlimited 631 px/s (n = 0.5); lowest sustained
sharp-turn coast 19.7 px/s (n = 2). Normal play spans about 20–215 px/s. `max_speed` 180
in `ShipConfig` is not a cap.

**Turning.** Sustained turn radius, independent of n and power: normal 66.2 px, sharp
35.3 px. Largest sustained angular rate 7.40°/frame = 444°/s (unlimited boost, n = 0.5,
normal turn). Commanded turn offsets 0, ±5°, ±15°; attitude = velocity heading + offset;
angular velocity = offset / dt. One-tick lateral acceleration model (median error 1.3%,
worst 8.6%): `a_lat ≈ lift · v² + thrust · sin(offset) / n²`, lift 0.015 (normal) / 0.027
(sharp), coast thrust 8 px/s², boost thrust 80 px/s².

**Acceleration and response.** Largest sustained lateral acceleration: energy-limited
0.148 px/frame² (533 px/s², n = 0.5, normal); unlimited 1.109 px/frame² (3 992 px/s²).
Initial unlimited boost acceleration `a₀ = 1 / (45 n²)` px/frame², at most 0.0889
(320 px/s², n = 0.5). Unlimited boost response `v(t) = (5.270 / n) · tanh(t / (237 n))`
px/frame, time constant 3.95 n s (n = 1: 50/90/99% at 130/349/627 frames). No-energy
response (n = 1): terminal 2.26 px/frame; 50/90/99% at 74/360/782 frames.

**Per-decision quantities at 30 Hz:**

| quantity | value |
|---|---|
| displacement per decision | 0.7 px (20 px/s), 3.3 px (100), 7.2 px (215), 21 px (631) |
| largest forward Δv per decision | ~10.7 px/s |
| largest lateral Δv per decision | ~17.8 px/s energy-limited; ~133 px/s unlimited |
| measured Δv per decision (allies) | RMS ~3.9 px/s, p99.9 ~18.6 px/s |
| cooldown | 3 ticks (0.1 s) |
| shield recharge | 5 s delay (150 ticks), then 0.5 hp per tick |

**Geometry.** World torus 65 536 × 65 536 px; map centre uniformly random per episode.
5v5: playable radius 2 600 px, zone ring 1 200 px, zone radius 330 px. Lengths scale with
`s = sqrt(ships / 10)` for fixed density (50v50: s = 3.16, playable radius 8 222 px).
Vision range 1 024 px.

---

## Appendix B. Sources

* Hafner et al., *Mastering Diverse Domains through World Models* (DreamerV3); settings
  in §2.3 verified against `github.com/danijar/dreamerv3`.
* Hafner, Yan et al., *Training Agents Inside of Scalable World Models* (Dreamer 4),
  arXiv 2509.24527.
* Ye et al., *Towards Playing Full MOBA Games with Deep RL* (Honor of Kings), NeurIPS 2020.
* Berner et al., *Dota 2 with Large Scale Deep RL* (OpenAI Five), 2019.
* Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*.
* Farebrother et al., *Stop Regressing: Training Value Functions via Classification*, 2024.
* Sreenivasan and Fiete, *Grid cells generate an analog error-correcting code*, 2011
  (residue codes and their failure mode).
* In this repository: `next-state-head-audit-sep2026.md`,
  `perception-belief-audit-sep2026.md`,
  `docs/engineering/frontline-density-and-front-dynamics.md`.
