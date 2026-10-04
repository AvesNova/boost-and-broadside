# Audit of the "semantic rewards, categorical state and map-scale training" plan

October 4, 2026. Against `dev` at `4eba8a1`, the September audits in this directory,
and current practice in distributional and categorical RL. Verdict first, then the four
sections in the plan's order, then the build order.

## Verdict

The plan is coherent and mostly grounded in measured failures, but five points need
to change before implementation, and two of the four changes are solving a different
problem from the one they are motivated by.

1. **The zero-sum test is mis-specified.** With opposing outcomes in separate
   components, no single component sums to zero over both teams; only an *event's*
   pair does. The test as written would fail on the first tick.
2. **The categorical outcome target leaks most of its mass into the tie class.** At
   gamma 0.9997 over 9 000-step episodes, 93% of the target mass is "tie" at the
   first decision and 74% at mid-episode. The §5 calibration diagnostic is meaningless
   under that construction. Use a fourth absorbing class instead.
3. **Splitting components changes the policy gradient even at identical weights.**
   `AdvantageScaler` normalises every component to unit RMS before the weighted
   sum, so the outcome tier goes from four unit-RMS streams to seven. The plan says
   the tier scales are kept; that is a silent 1.5–1.75x reweight toward zone play.
4. **Blurred position targets are not a proper scoring rule and undo the point of
   going categorical.** Cross-entropy against the sharp truth already produces a
   calibrated distribution; blurring by realised error forbids the head from using
   velocity to beat its own prior and biases it toward underconfidence. Velocity and
   every other channel in the plan already use sharp targets; position should too.
5. **The position belief stays a Gaussian.** The plan decodes the 60-way code to
   (mu, sigma) and re-encodes an isotropic blur. That is *less* expressive than the
   current full 2D covariance with correlation, and it discards the multimodality that
   was the argument for a categorical head. Store the 60 probabilities as the belief
   (as §2.4 does for every other channel) and decode mu only for RoPE.

And the two scope mismatches:

- **Map-scale randomisation at fixed ship count does not train the 50v50 condition.**
  `frontline_scale` exists to hold areal density constant as the fleet grows. Scaling
  the map with ten ships divides density by s², so the "50v50 eval scale" row in the
  plan's table is a 5v5 game on a map ten times too empty. The trainer already
  supports several `ScaleConfig`s with different ship counts sharing one policy
  (`config/training.py`, `scales[1:]` run self-play). A second scale at 20 ships and
  s = 1.41 trains large maps *and* the density the transfer target has.
- **The next-state head's measured failures are mostly not representational.** The
  September 25 audit attributes the visible-ship deficit to Gaussian-NLL gradient
  starvation of the coarse harmonics and a 3–6x conservative sigma; the hidden-ship
  deficit to the belief collapsing with no dynamics prior. The plan fixes the first two
  by construction (cross-entropy per level, no sigma) and does nothing about the
  third. That is fine as long as the A/B at build step 2 is read against the
  dead-reckoning baseline, not against the current head.

---

## 1. Rewards

### 1.1 Lambda-matrix removal: correct, with one thing to keep

The claim that this is behaviour-preserving checks out. Every component in
`REWARD_COMPONENT_NAMES` with nonzero weight is in `_LOCAL_COMPONENTS` except
`ally_win`, `enemy_win` and `outcome`, and `tests/train/test_global_value.py` already
asserts those are bit-identical across a side. `ally_combat_*` and `enemy_combat_*`,
the only components that ever used the team path with a sign, carry weight zero.

What is lost is not behaviour but a knob. The matrix was the only place a
team-spirit mix (OpenAI Five's `tau`, annealed from 0.3 to 1.0) could be expressed,
and the plan's own rule, "rewards own spatial credit", means the only cooperative
signal left is the outcome token and the zone absence charges. Both are slow. If a
selfish kill-trading equilibrium appears, there will be no cheap remedy. A single
scalar per component (`team_share_k`, applied as `adv_i + s_k * mean_j!=i adv_j`
within a side) keeps the remedy at near-zero cost without the (T, B, N, N, K)
tensor. I would keep that scalar and set it to zero, rather than delete the concept.

The reward-decomposed gradient diagnostic (`grad_diagnostics.py`,
`tests/train/test_grad_diagnostics.py`) and `test_ppo.py`'s lambda-row tests go with
the matrix; budget for that.

### 1.2 Balance rule and the zero-sum test

The rule itself (charge w, pay ratio x w, team-split when no opponent caused it) is
what the code does today and is fine. The **test is mis-specified**: "every
component's per-env sum over both teams is zero on every tick" cannot hold when
`death` is strictly negative and `kill_enemy` strictly positive by construction.
The invariant is per *event*: at all ratios 1, `death + kill_enemy + kill_ally`
sums to zero over both teams, `damage_taken + damage_dealt_enemy + damage_dealt_ally`
sums to zero, `shield_recharge + healed_damage` sums to zero, `zone_gain +
zone_loss` and the four live-progress components sum to zero, and `outcome` sums
to zero. `tests/env/test_shields.py` already states the invariant this way for the
recharge pair; write the new test the same way.

### 1.3 Component count and the advantage scaler

The plan treats weights as the only thing that sets gradient share. They are not.
`AdvantageScaler` (buffer.py) normalises each component's advantage to unit RMS
*before* `comp_weights` is applied and the components are summed, so each
component contributes roughly one unit-RMS stream times its weight, independent of
how much reward it carries. Consequences of the proposed split:

| tier | components today (nonzero weight) | components in the plan |
|---|---:|---:|
| outcome | ally_win, enemy_win, capture_progress, front_advance (+outcome at 0.01) | outcome, zone_gain, zone_loss, 4 live progress = 7 |
| kill/death | combat_death, kill_shot, kill_assist, kill_ally_shot, kill_ally_assist, boundary | death, kill_enemy, kill_ally = 3 |
| damage | combat_damage_taken, damage_dealt_enemy, damage_dealt_ally, shield_recharge, boundary_damage | damage_taken, damage_dealt_enemy, damage_dealt_ally, shield_recharge, healed_damage = 5 |

If the streams were uncorrelated the outcome tier's share would rise about 1.3x in
RMS (sqrt(7/4)) and the kill tier's fall about 1.4x, with the tier scales untouched.
They are correlated (the four live-progress components are mutually exclusive on a
tick, so they add in RMS more like one stream than four), which makes the effect
smaller but unmeasured. `core.py` already records that doubling the damage payout
moved the damage tier from 25% to 34% of the policy gradient. The plan's §5 lists
"per-component gradient share after the reward refactor" as a diagnostic; it needs
to be a *gate* at build step 1, with the tier scales re-derived from
`benchmarks/gradient_decomposition.py` before any comparison to run 748 or 750 is
made.

Related: merging `kill_shot` and `kill_assist` into one component halves the
kill side's stream count against the death side. The current `RewardConfig`
docstring notes the kill side already delivers about 87% of the death side's
gradient because its weight is spread over two correlated streams; after the merge
it will be one stream. That is probably an improvement, but it moves the
offence/defence balance in the same direction as the payout ratio, so the two
cannot be attributed separately in one run.

### 1.4 Level 1: the categorical outcome head

**The discount leak into T is the main defect in this section.** Gamma per
decision is 0.9997 and an episode is 9 000 decisions. The plan's recursion puts
(1 - gamma) of mass into the tie class each step, so the target at a state with
tau decisions left holds gamma^tau of real outcome mass and 1 - gamma^tau in T:

| decisions remaining | target mass in W+L | target mass in T (from discount alone) |
|---:|---:|---:|
| 9 000 | 0.07 | 0.93 |
| 4 500 | 0.26 | 0.74 |
| 1 000 | 0.74 | 0.26 |
| 100 | 0.97 | 0.03 |

The scalar value V = P(W) - P(L) is still the correctly discounted return, but the
three-way distribution is not an outcome distribution for most of the episode, and
the §5 "predicted P(W/T/L) against realised outcomes" diagnostic cannot be read
without un-doing the leak. The current code avoids this by keeping the categorical
head at gamma 1 with nearest-future-terminal labels (`_outcome_categorical_loss`),
which is a plain classifier of the realised result and is what the run-748 fix was
about.

Two ways out, in order of preference:

1. Add a fourth absorbing class U ("unresolved within horizon") worth 0, and leak
   the discount mass into U instead of T. T stays the honest tie probability,
   V = P(W) - P(L) is unchanged, the recursion is the same, and calibration is
   checkable: P(W | resolved) against realised wins. Cost is one logit.
2. Keep the critic's scalar value head for `outcome` as today (gamma 0.9997, GAE,
   ReturnScaler) and keep the gamma-1 classifier as the probe it already is. This
   is the status quo; the plan gains nothing from merging them.

Either way, confirm that the backward recursion cuts on `terminated` the way
`compute_gae` does (a 9 000-step timeout is a true DRAW terminal, so this is
consistent), and that the bootstrap at the chunk end uses the stored rollout-time
p(s_{T+1}), which is three floats per env per step.

Relative to practice: a win/draw/loss head trained by cross-entropy on terminal
labels is what AlphaZero-family and most game critics do; a *discounted*
categorical target that moves mass to a neutral class is non-standard, and the
usual fix for "I need a discount for stability but want calibrated outcome
probabilities" is exactly the absorbing-class trick (it is the categorical form of
treating 1 - gamma as a per-step termination probability).

### 1.5–1.6 Zone ledgers and the four live components

The ledger algebra is sound: with C_i += Delta/n_present and A_j += Delta/n_absent
on every forward tick, and both ledgers scaled by (1 - r/P) on a reversal by r, the
identities sum(C) = sum(A) = P hold, and the completion split (half final-tick
presence, half ledger share) is well defined because P = 1 at completion. The
non-empty fallback is never reached: `_advance_capture_state` moves a meter only
on a strict alive majority inside the zone, so the other side always has at least
one ship dead or elsewhere, which is the current code's own argument.

Three implementation notes the plan does not mention:

- The meter is zeroed when roles rotate (`progress = where(active_defense,
  progress, 0)`), and the roles rotate on *every* capture of *either* zone, so
  the ledgers of the zone that was not captured must clear too. The plan says
  "when roles rotate", which covers it, but the test should pin it.
- `zone_capture_progress` is clamped to [0, 1]. A reversal that hits 0 leaves
  P = 0 and the pro-rata scale undefined; clearing both ledgers at P = 0 is the
  right behaviour and needs a guard.
- Four live components keyed on ownership x direction cost four value heads for
  a signal that is one stream today. Since a zone is captured, reset or idle and
  never two at once, the four are mutually exclusive on any tick; for the critic
  this is four sparser targets, not more information. The honest reason to split
  is the cancellation argument in §1.1, which applies to *captured vs reset*. The
  *ally vs enemy* zone split is not a cancellation case (a ship is on one side of
  each zone at a time), so two components (`zone_progress_for`, `zone_progress_
  against`) would carry the same information with half the heads. Worth an A/B
  before committing four names to the checkpoint schema.

### 1.7 Kills and deaths

The outstanding-damage matrix O[a,t] is `cumulative_damage_matrix` plus a pro-rata
decrement on heal; that is a small change to `frontline.py`. The half/half split
between raw final impact and ledger share is reasonable and matches the current
`kill_shot_fraction` 0.5.

One gap: S_t = sum_a O[a,t] can be zero on a death with a raw final hit. A ship
depleted by boundary damage (not in O) and then finished by one projectile has a
raw-final share but no ledger. The plan's fallback covers "no enemy cause"; this
is "enemy cause, empty ledger". Give the whole share to raw_final when S_t = 0.

Starting the merged lambda at 0.95 is fine. Note the `kill_ally` component now
blames with both halves on the same horizon; today the shot half is at 0.87.

### 1.8 Damage and the charge-back

`damage_taken` from every source including boundary is the right fix for recharge
farming, and it replaces two components (`boundary`, `boundary_damage`) with zero.

The `healed_damage` charge-back is correct accounting but a weak deterrent, and the
plan should not expect it to do more than bookkeeping. Recharge begins 150 ticks
after the last hit (`shield_recharge_delay` 5 s at 30 Hz) and the damage tier's
gamma is 0.991 per decision, so by the time the charge-back arrives it is worth
0.991^150 = 0.26 of the payment it reverses. Chip damage that is fully healed still
nets about 74% of its payout to the attacker. That is acceptable, since the
exploit it exists for (deliberate damage/recharge cycles) is already closed by
`damage_taken`, but it means the attacker's critic has to predict a small,
delayed negative conditioned on the target's shield-delay state, which it can only
see if the target is visible. Worth logging the charge-back's realised share of
the damage tier so a surprising kill/damage balance can be traced to it.

Today's `shield_recharge` charges the *opposing team* equally in total; the plan
charges the specific attackers through the ledger. That is a better attribution and
changes the sign structure of `shield_recharge` from mixed to pure positive, which
is one of the cancellation cases the plan is fixing.

### 1.9 Weights

Removing `kill_shot_fraction` is right. `enemy_team_share` stays for the
no-cause fallbacks; make sure it still divides by the recipient team's full slot
count including dead ships, as it does now, or the zero-sum test will fail on
ticks with a dead recipient.

---

## 2. Next-state head and state representation

### 2.1 What the audit actually found, and what the plan fixes

The September 25 audit's three findings were: (1) the hidden-ship belief is a
static decohered prior, 93x worse than dead reckoning at 0.1–0.5 s; (2) only the
sigma-bearing harmonics train, the other nine are 10x to 500 000x worse than
identity on a visible ship; (3) identity is not cheaply reachable. The plan's
residual zero-initialised logits fix (3) directly. Per-level cross-entropy fixes (2)
in the sense that every level gets the same nats-weighted gradient and there is no
sigma to miscalibrate. Nothing in the plan addresses (1): the owner's recorded
decision (`perception-belief-audit-sep2026.md`, item 1) is that the head must learn
the dynamics rather than be given a prior, and the plan keeps that. So the
expected result of build step 2 is a sharper *visible* forecast and a better
calibrated hidden one, not a hidden belief that beats dead reckoning. State that in
the step-2 success criterion so the run is not read as a failure of the
representation.

The gradient-share history matters here. In run 748 the next-state term held 76%
of the trunk's pre-clip gradient (`grad-decomposition-748-sep2026.json`), and the
BC coefficients were re-set from measured norms twice (`db55887`). A head with
~440 logits summed over nine channels will have a different norm again. Measure
with `benchmarks/gradient_decomposition.py` before the first run and set
`next_state_coef` from it; do not carry 0.17 or 0.2 across.

### 2.2 Position

**The code.** The Gaussian-integer construction is correct: |2+i|^2 = 5, so each
level has exactly five cosets; the four corners of a level-l cell are 0, 1, i, 1+i
in L_l units, whose differences have norm 1 or 2 and are never multiples of (2+i),
so the corners are always distinct colours; and 65 536 / (2+i) is a sublattice of
Z[i] scaled to the torus, so level 0 tiles it exactly. The 12-level table is right
(29 309 x 5^(-11/2) = 4.2 px). This is a 2D residue-number / grid-cell code.

**The risk is the one grid codes are known for.** Each level repeats every
sqrt(5) cells, so a level alone localises to within about 2.2 of its own cells and
*every* finer level depends on every coarser level being right. Sreenivasan and
Fiete (2011) showed such codes have a catastrophic error mode: once noise at some
level exceeds the ambiguity radius, the decode jumps by a coarse cell rather than
degrading gracefully. The September audit measured exactly that band failing in the
Fourier version: resultant length 0.2–0.6 for periods 512–4 096 px. A categorical
code does not change the geometry of the failure; it only changes the loss. The
coarse-to-fine argmin decode in §2.2 will, when a middle level is near uniform,
pick among aliases and land a few hundred pixels off, which is the audit's current
527 px.

Two cheaper alternatives are worth putting beside it in the same A/B:

- **A two-level map-relative code.** The map centre is uniformly random on the
  torus each episode and the playable circle never approaches the half-period, so
  absolute torus coordinates carry no game information above the playable diameter;
  with spatial RoPE already supplying relative geometry, levels 0–2 (29 309 to
  5 862 px) are per-episode noise at s = 1. A coarse barycentric 3-hot on the hex
  grid `hex_density.py` already implements (331 cells over the playable circle,
  cell pitch about 260 px at s = 1, scaled per env) plus one fine two-hot per axis
  inside the cell gives a flat, non-aliased joint distribution at the coarse level
  and sub-pixel resolution at the fine one, with no residue decoding. About 331 +
  2 x 64 logits. Its weakness is that the fine level still aliases at the cell
  pitch, but only one level does, and the coarse level is a true distribution over
  the map.
- **The current dyadic Fourier basis with per-harmonic categorical phase.** Keeps
  every encoder and RoPE table, and isolates the "loss, not geometry" hypothesis.

If the nested-5 code is kept, two changes:

- **Use sharp targets everywhere, including hidden ships.** Cross-entropy against
  the true next cell is a proper scoring rule; its minimiser *is* the conditional
  distribution given the inputs, which already includes "I cannot tell which alias"
  as a near-uniform level. Blurring the target by realised error e does two bad
  things: it hands the head its own previous error as a label, so a bad belief
  teaches uniform at the fine levels and the loop has a uniform fixed point at
  exactly the levels that need to recover (the same shape as the run-748 outcome
  fixed point); and it forbids the head from using velocity, attitude and the
  ledger of what it has seen to beat the prior, which is the entire point of a
  learned belief. The deadzone-and-ramp is a hand-designed calibration curve
  standing in for what the softmax does on its own. If the worry is that
  unpredictable fine levels add gradient noise on old hidden tokens, weight those
  levels down by hidden age or mask them, which changes the variance and not the
  target.
- **Store the distribution, not (mu, sigma).** §2.4 stores the predicted
  distribution for every other channel and feeds it back; §2.2 alone decodes to
  three numbers and re-encodes an isotropic Gaussian blur. The current belief
  carries a full 2D covariance with correlation (`UNCERTAINTY_NAMES`); the plan's
  position belief is strictly less expressive than today's, and it throws away the
  one thing a categorical head can represent that a Gaussian cannot, which is "it
  went left or right of the zone". Keep the 60 probabilities as the belief state
  (60 floats per ship per observer, trivial), use them as the residual baseline
  directly, and run the decode only to get mu for RoPE and the diagnostics. The
  "near-uniform baseline at an uncertain fine level" property the plan wants from
  re-encoding holds automatically. Drop the lambda-curve, the least-squares sigma
  fit and the Monte Carlo blur: that is most of §2.2's machinery.

On the finest level: 4.2 px against 3.3 px per decision at cruise is right for
visible ships and matches the audit's note that the level-9 (128 px) harmonic was
the only one that trained. The §5 instruction to gate level 11 by speed if it
sits at baseline is sensible.

### 2.3 Velocity

The symlog-with-knee at v0 = 100 px/s is a real improvement over the current
`SymlogVelocity`, whose knee at 1 px/s compresses the whole playing band; the
plan's own figure (152/33 px/s effective spacing at 100 px/s today) is the right
indictment. The 81-bin, 5 u-unit spacing gives 10/7 px/s cells at cruise against a
measured per-decision delta-v RMS of 3.9 px/s, so a sharp two-hot moves by about
0.4 bin per decision, which is enough to learn from.

Three axes at 120 degrees are a Radon-type projection, not a joint distribution:
the three marginals are linearly dependent in the mean (they sum to zero), so the
third adds consistency but no new first moment, and no set of 1D marginals can
represent a bimodal joint (turned left or right). For a *visible* ship over one
decision that does not matter. For a hidden ship after a few seconds it is the
case that matters, and it is the same multimodality argument as for position. The
codebase already has the right primitive: the barycentric 3-hot on a hex lattice in
`hex_density.py` is exactly a lossless 2D two-hot. On the u-disc of radius 199 a
hex pitch of 10 u is about 1 400 cells and 15 u about 640; the plan rejected "about
4 900" at the 1D resolution, but the 2D cell does not need to match the 1D spacing
to beat three marginals. Put a 640-cell hex two-hot in the A/B.

HL-Gauss smoothing at half a bin is reasonable and cheap; the "sense of distance
between bins" it buys is the documented reason Farebrother et al. (2024) prefer it
to two-hot.

### 2.4 Other channels

Fine as stated. Two notes:

- Angular velocity is exactly `turn_offset / dt` with turn_offset in {0, +-5, +-15}
  degrees (`physics.py:160`), so the five-bin head is predicting the enemy's turn
  command one step ahead. The enemy-action head already predicts the 42-way joint
  action, whose turn marginal is this quantity. Tying them (the plan's own "consider")
  removes a redundant head and gives the belief the enemy-action head's prediction,
  which is trained on the same label. Do it.
- Cooldown is 3 ticks and deterministic given the pending shoot action for allies.
  The audit found the Gaussian head at chance on it, degrading through the run; a
  4-way categorical with a residual baseline makes "no change" free, which was the
  diagnosis. This is the cheapest confirmation that the residual-logit mechanism
  works and should be the first thing read off the step-2 run.

The encoder's ship-token input grows from roughly 70 floats to about 440
(60 + 243 + 72 + 5 + 11 + 21 + 21 + 4 + 5), and the head's output from 24 to the
same. At d_model 128 that is a wider first linear layer and a larger head; neither
is a throughput concern, but the input is now mostly zeros with a handful of
non-zero entries per channel, so initialise the first layer like an embedding
table (per-input-unit variance) rather than orthogonally over the whole row.

---

## 3. Categorical critic

Keeping scalar bf16 returns in the rollout and converting to a categorical target
only at loss time is the right division; it leaves `compute_gae`, the scalers and
the lambda-free aggregation untouched. The choice is consistent with practice:
Farebrother et al. (2024, "Stop Regressing") found HL-Gauss consistently better
than two-hot and MSE across value regression in online RL, and DreamerV3 and
TD-MPC2 both use symlog two-hot critics.

Three corrections:

- **Do not symlog-space the bins.** Rewards are already symlog-compressed at
   storage (`symlog #1` in `buffer.py`) and the ReturnScaler then maps two standard
   deviations to one. A symlog-spaced support on top compresses a third time, which
   is the same "symlog of a log" mistake `features.py` warns about for the
   uncertainty input. Use uniform bins in the already-normalised space.
- **21 bins is thin for HL-Gauss.** The method's advantage over two-hot is the
   Gaussian spread over several bins; with sigma at 0.75 bin width and 21 bins the
   support is 1.5 sigma of a normal component's mass wide and the tails to +-10
   normalised units would be a few very wide bins carrying no resolution where
   sparse components actually live. Farebrother et al. use 101 bins; 51–101 uniform
   bins over +-3 normalised units, with sigma = 0.75 bin, is the standard setting.
   Clipping at +-3 loses less than the current Huber loss does on the same tails.
- **Say what happens to value clipping and explained variance.** `_huber` on
   `new_value - target_norm` goes away; PPO value clipping (if enabled) has no
   categorical analogue and should simply be dropped; explained variance should be
   computed on the expectation so the critic series stay comparable with runs 748
   and 750.

Per-component categorical heads are 15 x N_bins outputs per ship; at 51 bins that
is 765 logits per ship token, which is fine. The ReturnScaler's EMA moving under
the bins makes the target non-stationary in bin space, exactly as it does for the
scalar target today, so nothing new there.

---

## 4. Map-scale randomisation

### 4.1 What it trains is not what is being transferred to

`frontline_scale(num_ships)` and the density doc are explicit that the invariant
behind zero-shot fleet transfer is *areal density*, so that a ship sees the same
number of ships inside its 1 024 px sight at every fleet size. Sampling s per
episode with the ship count fixed at ten gives density 1/s². At s = 3.16 the
policy plays ten ships on the 50v50 map: one tenth of the density it would meet at
50v50, and one tenth of the engagement rate it was tuned for. The plan's table
labels that row "50v50 eval scale"; it is not. What the randomisation does train is
long travel, long occlusion, the coarse code levels and the low-frequency RoPE pairs,
which is the plan's own stated purpose, and that is worth something; but it should
not be sold as training the transfer target, and the §5 metrics bucketed by s
should be read as "sparse-map" metrics.

The trainer already has the density-preserving mechanism. `TrainConfig.scales` is a
tuple of `ScaleConfig(env_config, num_envs)` sharing one policy, optimiser and
return scaler, with `scales[1:]` running self-play and the batch sized "inversely
proportional to N so total ships-per-update stays constant" (`config/training.py`).
Only one scale is configured today. A second scale at 20 ships (s = 1.41, playable
radius 3 677 px) or 40 ships (s = 2, 5 200 px) gives bigger maps at the right
density, exercises the same coarse levels and RoPE pairs, and tests the fleet-size
transfer during training instead of after. Memory is the constraint (tokens per
decision scale with N), which is why the plan may have avoided it; but it is the
mechanism the mission is about.

If per-episode map scale is kept anyway, a smaller range (s in [1, 2]) at fixed N
costs little density and still varies the geometry.

### 4.2 Implementation gaps

`scaled_frontline_geometry` returns new *config dataclasses*, applied once at
resolve time; per-episode scale moves the multiplication into the Frontline reset
path. The per-env tensors for zones and the boundary exist (`zone_radius`,
`playable_boundary_radius`, `map_center` in `TensorState`), but:

- `field_generation.py:68` reads `ship_config.field_radius_min/max` and the
  transition widths as scalars; the fields need per-env scale at generation.
- The scripted opponent's `frontline_zone_radius` is a scalar parameter
  (`frontline_strategy.py:53`, scaled once in `evaluation/agents.py:89`). On a
  per-env scaled batch the teacher is wrong on every env except s = 1. For the BC
  profile that is a teacher that cannot find the zones; for the RL profile the
  scripted opponent in `scales[0]` is similarly broken. Either read
  `state.zone_radius` in the strategy or keep the scripted slot's envs at s = 1.
- The hex density target is already per-env in units of the playable radius, so it
  survives; the `radius` observation feature is normalised by world size, which is
  fine; `zone_offensive_distance` and friends are symlog of absolute pixels, which
  is fine.
- `frontline_rules_match` excludes geometry, so checkpoints survive.

### 4.3 Outcome sparsity

At s = 12 the ring radius is 14 400 px, 106 s at cruise against a 300 s episode
with capture at 10 s per net ship. Expect nearly all s >= 6 episodes to time out.
At rate 1.5 that is 2.3% of episodes contributing ties to the outcome head and
nothing to the zone tiers. The exponential's density near s = 1 is also high enough
that "s <= 1.25 with probability 0.65" means most non-standard episodes are nearly
standard; the distribution is doing less than its range suggests. If the point is
occlusion and coarse levels, a log-uniform s on [1, 4] with p(s = 1) = 0.5 is simpler
and spends the 50% non-standard budget where it changes something.

---

## 5. Diagnostics and build order

The §5 list is good. Add:

- the per-event zero-sum residual (§1.2), not per-component;
- tier gradient shares *before and after* step 1 as a gate, with tier scales
  re-derived (§1.3);
- the next-state head's trunk gradient share at the first update of step 2, with
  `next_state_coef` set from it (§2.1);
- for the outcome head, P(W | resolved) against realised wins, which is only
  readable with the absorbing class (§1.4);
- for step 4, every per-update metric bucketed by s *and* by ships-in-sight, so
  the density mismatch is visible in the plot rather than hidden in s.

Build order: one variable at a time is right, and the A/B at each step must name
the baseline. Step 1 against run 750's reward shares; step 2 against *dead reckoning
and persistence*, not against the Gaussian head (the audit's baselines are already
production diagnostics); step 3 against step 2's explained variance on the
expectation; step 4 against step 3 at s = 1 only, since nothing before it saw
another scale.

Each step retires the checkpoint schema. Four schema bumps in sequence, each
without migration, means no run from step k can seed step k+1. If the steps are to
be A/B'd on equal compute, that is four cold starts of the full BC-then-RL budget.
Reordering so the schema-stable changes come first does not help here because all
four touch the schema. The one cheap mitigation is to land the reward refactor and
the categorical critic together behind a config switch (the critic conversion is a
loss-time change with no rollout effect), which makes steps 1 and 3 one schema and
one cold start.

---

## References used for the state-of-the-art comparison

- Farebrother et al., "Stop Regressing: Training Value Functions via
  Classification for Scalable Deep RL", 2024 (HL-Gauss vs two-hot vs MSE; 101 bins).
- Hafner et al., DreamerV3, 2023 (symlog two-hot critic; return normalisation by
  percentile, which `AdvantageScaler` cites).
- Hansen et al., TD-MPC2, 2024 (symlog discrete regression, 101 bins).
- Imani and White, "Improving Regression Performance with Distributional Losses",
  2018 (the HL-Gauss origin).
- Berner et al., OpenAI Five, 2019 (team spirit annealing; the knob the lambda
  matrix was).
- Sreenivasan and Fiete, "Grid cells generate an analog error-correcting code for
  singularly precise neural computation", Nature Neuroscience 2011, and Fiete,
  Burak and Brookings, "What grid cells convey about rat location", 2008 (residue
  codes, their capacity and their catastrophic error mode).
- Bellemare, Dabney and Munos, C51, 2017, and Dabney et al., QR-DQN, 2018
  (distributional critics; the projection step the plan's absorbing-class
  recursion avoids needing).
