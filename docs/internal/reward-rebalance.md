# Reward weighting: bug fixes, rebalance, and the balance rule

Working notes. Delete once the follow-up run has landed.

Numbers come from two runs: `good-leaf-719` (the reference, 1B steps) and
`silvery-pond-720` (this work's first attempt, interrupted at 127M). Detailed
evidence is in the appendix.

## What landed

Two bugs meant the reward weights in the config had never been the weights any run
trained under. Fixing them made the weights real, which then made it worth asking
what they should be -- and that turned out to be a structural question rather than
a tuning one.

| # | Change | Commit |
|---|---|---|
| 1 | **Lambda fix.** Rows were normalized by their own *weighted* sum, which divides out the weight just applied: local weights saturated at `min(w, 1)` and global ones cancelled entirely. `ally_win_weight=1.5` trained identically to `0.25`. Normalizing the unweighted pattern first keeps the weight linear; bit-identical at `w = 1`, which is the test. | `e27b73b` |
| 2 | **Return scaler.** `return_min_span=1.0` bound 8 of 12 components on every update, compressing critic targets up to 121x and critic gradients up to 1.5e4x. Replaced p5/p95 with a masked mean/std estimator (2σ → 1), dropped the floor to 1e-3, added a Huber value loss for the tails the floor had been suppressing. Dead ships now excluded from both scalers. | `6a95e0e` |
| 3 | **Friendly kills as their own components.** `kill_shot` and `kill_assist` each carried the friendly-fire penalty inside their own positive signal, so one critic head predicted the sum of two opposite things and the friendly half was unweightable and invisible to every diagnostic. Now `kill_ally_shot` and `kill_ally_assist`, mirroring the enemy pair on both horizons. | `81e3e9e`, `d60f44f` |
| 4 | **Tier scales.** The three group scales were dead (all held at 1.0; `global_scale` applied only to zero-weight components). Replaced with four tier scales — outcome, kill/death, damage, shaping — the same partition the per-component gammas and lambdas already follow. Locality became its own registry, since it had only been derivable from the old groups by accident. | `f392417` |
| 5 | **Weights derived, not chosen.** Four free numbers; every event component follows from the balance rule below. | `0a7708c` |

Not done, deliberately: `behavior_cloning_coef` stays at 2.0. During BC 100% of
updates hit `max_grad_norm` (median total norm 2.18, max 13.3), so every other
term trained at 0.1-0.5x its nominal step for the first 38M steps. The mechanism
is measured, but that it *harmed* the outcome is not, and the intervention is
unpredictable: BC's unweighted gradient rises as its coefficient falls (CV 97%,
roughly 2 → 13 over the decay), because a smaller coefficient lets the policy
drift further from the teacher. Left as the control, to be measured against.

## The balance rule

Weights were sixteen independent choices, and most of them were not independent.
**An event pays one side exactly what it charges the other.** A death costs the
dying ship's team `U` and pays whoever caused it `U` between them; damage does the
same with `V`; a win pays `W` and charges `W`. `f` — how `U` splits between
landing the finishing blow and contributing damage — is the only ratio the rule
leaves free.

| free number | value | fixes |
|---|---:|---|
| `win_weight` (W) | 1.00 | `ally_win`, `enemy_win` |
| `death_weight` (U) | 1.00 | `combat_death`, `field_death` |
| `damage_weight` (V) | 0.40 | all four damage components, plus `enemy_field_damage` |
| `kill_shot_fraction` (f) | 0.50 | splits U across the four kill components and `enemy_field_death` |

Shaping stays individually weighted (`facing` 0.10, `closing_speed` 0.10). Facing
a target is a state, not something that happens to somebody, so there is no
opposing side to charge and the rule does not apply.

Three consequences that look like coincidences and are not:

* **`enemy_field_death` must equal `kill_shot`.** A ship killed by a field was shot
  by nobody on its fatal step — verified: `kill_shot` returns exactly 0.0 there
  while `kill_assist` returns 1.0 — so the offensive side is short by precisely
  `kill_shot`, and something has to make it up or a field kill pays half a combat
  kill. `enemy_field_damage == V` by the same argument. These are the only two
  source-split components with a non-zero weight, and that is their purpose: to
  supply the offensive side of events with no shooter to attribute to.
  `enemy_field_damage` is also the principled form of "reward for forcing an enemy
  into a field" — attributed by team rather than by proximity, so unlike a
  nearest-enemy heuristic it survives a change of fleet size.
* **Killing a teammate costs the team twice**, once for the death and once for
  having caused it, with the enemy paid nothing. Friendly fire is structurally
  twice as expensive as being killed by an opponent, with no special case saying
  so. Friendly-fire pressure falls 19.2% → 12.8% as a side effect.
* **Offence and defence land at 25.0% and 25.9%** with nothing targeting them. The
  rule makes the balance an identity rather than a goal.

The remaining `ally_*` and `enemy_combat_*` components stay at zero: their events
are already paid for by the local per-ship components and by damage attribution,
so turning them on would charge the same event twice.

### Weight parity is not gradient parity

The kill side spends `U` across two correlated components (+0.509 cosine) while
the death side spends it on one, so the kill side delivers about 87% of the death
side's gradient magnitude. Left alone on purpose. Weights state what an event
*means*; pressure is allowed to follow how coherent each signal actually is.

That is the point, not a defect. Pressure is `weight × coherence`, and coherence
falls as a behaviour is solved, because rare events produce gradients that cancel
across the batch. In run 719 `field_death`'s coherence halved (15.5 → 7.9) while
field deaths fell eightfold (152 → 28 per million). A component whose problem the
policy has solved should fade on its own.

Which is why share-targeting was the wrong instrument. Solving `w = share / d`
means that when `d` falls, `w` rises to compensate — a standing instruction to
keep pressuring a solved problem. The four unit values are still solved against
measured `d`, but only at the *tier* level, where the allocation is a genuine
strategic choice.

Tier targets: **31% outcome / 32% kill-death / 31% damage / 5% shaping**, flat
across the top three. The win pair is two near-duplicate signals (+0.536 cosine)
and half of all games are self-play, where the outcome is a coin flip by
construction, so the tiers below carry per-step information it cannot. It still
takes the largest single share, because everything below it is a proxy and proxies
are what a policy learns to farm.

Shaping tapers: `(0, 1.0) → (100M, 1.0, exponential) → (400M, 0.05, hold)`. It has
to be pushed down rather than left alone — its realised share *grows* 1.58x under
static weights. `facing` and `closing_speed` are not potential-based, so they bias
the optimum for as long as they are on, and they oppose the objective directly
(`closing_speed` against `field_damage_taken`: mean cosine −0.446, negative in
99.9% of samples). The 0.05 floor is instrumentation, not correctness — a tier at
zero keeps its components registered, but a floor keeps their gradient share and
explained variance readable to the end.

## What run 720 showed

Interrupted at 127.4M steps, update 127, live_elo 1417.

Confirmed:

* The floor never bound. `floor_bound_span_count` 0 for the whole run. The two
  narrowest components clear 1e-3 by 31x and 34x; at the 1e-2 first proposed they
  would have cleared by only ~3x, which is why the floor went lower.
* Starved critics recovered. Against 719 at the matched 111M step: `field_death`
  0.714 → **0.858**, `kill_assist` 0.712 → **0.860**, `kill_shot` 0.898 → 0.922,
  the win pair 0.873 → 0.913 at 41.8M. The fix did not raise the ceiling so much
  as reach it roughly twenty times sooner.
* Elo tracked and then exceeded the reference: 1417 at the stop against 719's
  95-125M band of mean 1327, sd 53 over 30-55M.

Not confirmed:

* **`field_death`'s coherence did not rise.** Predicted to climb toward the pack
  once its critic was fed; normalized, it went 0.77 → 0.79 while its EV rose 0.14.
  So the chain "starved critic → noisy advantage → incoherent gradient" does not
  hold for this component: its low coherence is intrinsic to rarity. That removes
  the argument for raising its weight.
* `damage_dealt_ally` EV fell (0.773 → 0.699). Possibly the friendly-kill split
  diluting it; unresolved.
* A friendly-fire overshoot to 19.2% was diagnosed mid-run from an Elo reading
  that later turned out to be a low sample rather than a trend. The derivation
  fixes the overshoot structurally, so it was never isolated.

## What to check on the first diagnostic update of the next run

1. `enemy_field_death` and `enemy_field_damage` are active and their coherence is
   measured for the first time. Their unit values rest on estimates (15.0 and 17.0,
   from structural analogues) and are the least trustworthy numbers in the config.
2. Realised tier shares against 31/32/31/5. The four unit values were solved
   against `d` measured under the *old* weight vector, so expect drift.
3. `scaler/floor_bound_span_count` stays 0 with two new active components.
4. Friendly-fire pressure lands near 12.8% rather than 19.2%.
5. Whether `damage_dealt_ally`'s EV recovers now that the friendly-kill weights are
   halved relative to run 720.

## Out of scope

Tier scheduling (the mechanism exists and ships flat; design its keypoints from a
clean run's EV curves). Team-spirit annealing on the local-vs-shared lambda axis —
the faithful OpenAI Five analogue, and a different axis from the tier ladder.
Joint 2x8x3 action space, with the caveat that a joint entropy bonus penalizes the
inter-dimension correlation the joint head exists to express. EV-gated dynamic
weights.

The one item this list carried that has since been built is the latent multi-step
rollout head, which is where the balance analysis pointed: the old `next_state`
term held 27-31% of the trunk gradient for an objective the architecture docs
already called a weak representation signal. It and the windowed loss are now the
[predictive belief state](../architecture.md#predictive-belief-state), whose two
loss families appear separately in the gradient decomposition. Whether that
redistributes the trunk gradient usefully is unmeasured; the share numbers above
describe the objective it replaced.

# Appendix: evidence

## Bug 1 — the lambda clamp cancels reward weights

`PPOTrainer._lambda_matrix` ends with

```python
return lambda_ij / lambda_ij.abs().sum(dim=3, keepdim=True).clamp(min=1.0)
```

The row sum already contains the component weight, so dividing by it removes the
weight again. Local (diagonal) components come out at `min(w, 1)`; global ones at
`1/n_alive_allies` per ally whenever `w * n_alive > 1`, i.e. the weight cancels
entirely. `ally_win_weight=1.5` trains identically to `0.25`.

Measured confirmation. Define `d = share / w_eff`, the policy-gradient share a
component delivers per unit of effective weight, measured post-BC (step > 50M):

| component | w cfg | w_eff | share | d | d CV |
|---|---:|---:|---:|---:|---:|
| combat_death | 1.0 | 1.0 | 13.69% | 13.69 | 8.5% |
| combat_damage_taken | 0.5 | 0.5 | 6.84% | 13.69 | 9.6% |
| field_damage_taken | 0.5 | 0.5 | 6.88% | 13.77 | 10.6% |
| kill_shot | 1.0 | 1.0 | 13.36% | 13.36 | 6.4% |
| damage_dealt_enemy | 0.5 | 0.5 | 7.43% | 14.86 | 10.5% |
| kill_assist | 1.0 | 1.0 | 11.85% | 11.85 | 5.8% |
| damage_dealt_ally | 0.5 | 0.5 | 5.80% | 11.59 | 11.7% |
| ally_win | 1.5 | 1.0 | 11.23% | 11.23 | 10.8% |
| enemy_win | 1.5 | 1.0 | 10.41% | 10.41 | 11.5% |
| facing | 0.1 | 0.1 | 1.62% | 16.24 | 11.0% |
| closing_speed | 0.1 | 0.1 | 2.60% | 25.97 | 15.4% |
| field_death | 1.0 | 1.0 | 8.28% | 8.28 | 28.6% |

Components at 2x different configured weights land on the same `d` to four
significant figures (combat_death vs combat_damage_taken: 13.69 both). The win
pair, configured at 1.5, delivers 0.84x the share of kill_shot at 1.0 — the
effective-weight-1.0 prediction, not the 1.5x one.

`d` is stable to 6-15% CV across 950M steps, so target shares can be chosen and
weights solved as `w = share_target / d`.

### Fix

Normalize the unweighted pattern, then apply the weight:

```python
pattern = torch.where(self.local_k, local_lambda, global_lambda)   # +-1 / 0
norm    = pattern.abs().sum(dim=3, keepdim=True).clamp(min=1.0)
return  pattern / norm * comp_weights * alive_j
```

At `w = 1` this is bit-identical to current behaviour — that equivalence is the
test. Local becomes `w` unclamped; win becomes `w/n_alive` per ally, `w` total.

## Bug 2 — `return_min_span=1.0` starves 8 of 12 critics

`ReturnScaler._half_span` clamps to `min_span/2`. The RL profile sets
`return_min_span=1.0` while the class docstring requires it to "sit far below
every active component's real span". Exact spans from the final checkpoint:

| component | true p5-p95 span | floored | target / | loss / |
|---|---:|:--:|---:|---:|
| field_death | 0.00827 | yes | 121x | 14614x |
| kill_assist | 0.08791 | yes | 11.4x | 129x |
| kill_shot | 0.09234 | yes | 10.8x | 117x |
| ally_win / enemy_win | 0.1067 | yes | 9.4x | 88x |
| damage_dealt_ally | 0.1396 | yes | 7.2x | 51x |
| combat_death | 0.2129 | yes | 4.7x | 22x |
| field_damage_taken | 0.4252 | yes | 2.4x | 5.5x |
| facing / closing_speed / combat_damage_taken / damage_dealt_enemy | 2.44-4.21 | no | 1x | 1x |

EV is a variance ratio in one space, so the floor cannot depress EV mechanically —
only through undertraining. The two critics whose EV *falls* over training
(field_death 0.633->0.492, damage_dealt_ally 0.708->0.557) rank #1 and #5 in
suppression.

### The naive fix over-corrects

p5/p95 measures the width of the zero-spike for sparse components, not the range
of the signal. Reconstructed from the logged return histograms:

| component | p5-p95 | p0.5-p99.5 | full range |
|---|---:|---:|---:|
| damage_dealt_ally | 0.176 | 0.725 | 2.462 |
| field_damage_taken | 0.574 | 1.654 | 3.178 |
| field_death | 0.0059 | 0.0126 | 0.137 |
| combat_damage_taken | 4.650 | 5.890 | 7.294 |
| ally_win | 0.106 | 0.120 | 0.133 |

The estimator disagrees with itself by up to 4x on exactly the components where
the floor binds hardest, and barely at all elsewhere. So the profile's defence of
the floor ("critic outliers dominate the squared value loss") was pointing at a
real failure — it just patched the scale instead of the estimator or the loss.

### Fix — three coupled changes, not one  [LANDED]

1. Tail-aware scale estimator: masked mean/std, two sigma to one, matching the
   second-moment statistics `AdvantageScaler` already uses. Dead ships are now
   excluded from both scalers, having been counted in neither's favour before.
2. `return_min_span` down to **1e-3**, not the 1e-2 first sketched: measured
   against run 719's logged histograms, the narrowest live component
   (`field_death`, 4-sigma span 0.0127) clears 1e-3 by 12x but 1e-2 by only 1.3x,
   which is not what "far below every active component's real span" means.
3. Huber value loss (`value_huber_delta=1.0`), scaled to agree with squared error
   inside delta so the switch reshapes only the tails.

Doing (2) alone destabilizes the critic; doing (3) alone leaves the starvation.
Resume is safe: `ReturnScaler.load_state_dict` re-seeds when `min_span` changes,
and now also when handed percentile-era state it cannot interpret.

`return_quantile_samples` is gone — it existed only to bound CPU sorting for
`torch.quantile`, and moments need no subsampling.

Amplification is smaller than the p5/p95 arithmetic implied, because the new
estimator is itself tail-aware: `field_death` gains ~79x rather than 121x,
`damage_dealt_ally` ~2.5x rather than 7.2x, `field_damage_taken` ~1.0x rather
than 2.4x. The predicted value-share table below is therefore an overestimate of
the redistribution; re-measure rather than trusting it.

### Predicted effect

Value-gradient shares, assuming EV unchanged (an upper bound on a transient that
relaxes as the fed critics improve):

| component | amp | now | after |
|---|---:|---:|---:|
| field_damage_taken | 2.4 | 34.4% | 27.2% |
| damage_dealt_ally | 7.2 | 10.8% | 26.6% |
| field_death | 121 | 0.2% | 10.6% |
| damage_dealt_enemy | 1.0 | 18.4% | 9.1% |
| combat_damage_taken | 1.0 | 13.7% | 6.5% |
| kill_shot + kill_assist | ~11 | 1.5% | 5.4% |
| ally_win + enemy_win | 9.4 | 1.4% | 4.0% |

Aggregate value loss rises ~14x (0.0125 -> 0.174); the critic's top-level gradient
share moves 2.4% -> ~5-6%. `max_grad_norm` still would not bind (it binds on 2.6%
of post-BC updates today).

## Falsifiable predictions (outcomes recorded in "What run 720 showed")

1. field_death's `d` rises from 8.28 toward the pack (~12-14) and its CV drops
   from 28.6% toward ~10%. It is both the most suppressed critic and the least
   coherent policy gradient. If `d` does not move, field deaths really are just
   rare and the component should be demoted.
2. field_death EV rises above 0.49 and damage_dealt_ally above 0.56, and both stop
   declining.
3. Win-pair and kill critics' EV barely moves (already 0.91-0.92 under ~10x
   suppression). Their value-share rise is reallocation, not repair.
4. `ally_win_weight` becomes a linear knob, testable on the first diagnostic
   update.

## Missing component — friendly kills (landed)

There is no `kill_ally`. The friendly-kill penalty is folded *inside*
`KillShotReward` and `KillAssistReward` (`reward -= dm_friendly/total_friendly`),
so one critic head predicts the sum of a positive enemy-kill signal and a negative
friendly-kill signal under one gamma and one weight. Extract it as its own
component so it can be weighted and learned separately. Note `K` grows, which
dilutes the per-component value-loss divisor, and `EnvWrapper._active_names` is
frozen at init from `weight != 0`, so it must launch with a non-zero weight.

## Tier scheduling (an OpenAI-Five-style curriculum)

Measured first sustained crossing of each critic's explained variance, run 719
(under the min_span suppression, so re-measure after the scaler fix):

| component | EV>0.7 | EV>0.85 | EV>0.92 |
|---|---:|---:|---:|
| ally_win / enemy_win | 7M | 32M | 104M |
| kill_shot | 21M | 53M | — |
| kill_assist | 101M | — | — |
| combat_death | 14M | 20M | 226M |
| field_death | 59M | — | — |
| damage_dealt_enemy | 7M | 13M | 17M |
| combat_damage_taken | 8M | 13M | 16M |
| field_damage_taken | 29M | 49M | 111M |
| damage_dealt_ally | 22M | — | — |
| facing / closing_speed | 7-10M | 9-12M | 11-14M |

Tier-mean EV over training:

| step | win | kill/death | damage | shaping |
|---:|---:|---:|---:|---:|
| 5M | 0.313 | 0.170 | 0.116 | 0.307 |
| 20M | 0.850 | 0.565 | 0.749 | 0.971 |
| 50M | 0.876 | 0.756 | 0.898 | 0.982 |
| 100M | 0.915 | 0.809 | 0.909 | 0.982 |
| 200M | 0.929 | 0.806 | 0.877 | 0.981 |
| 999M | 0.928 | 0.777 | 0.853 | 0.980 |

Two findings that constrain the curriculum:

1. The win critic is reliable **early** — EV 0.85 by 32M, ahead of the whole
   kill/death tier. There is no window where it is unusable, so "start
   damage-heavy because win is not learnable yet" is not supported.
2. The proxy tiers **decay** after ~100M (damage 0.909 -> 0.853, kill/death 0.809
   -> 0.777) while win holds flat at 0.93. Late in training the outcome signal is
   the most reliable one, not the least. That supports a win-heavy endgame, for
   the opposite reason to the usual one.

Recommendation: build the mechanism now, ship the first run flat. Replace the
three dead group scales (`true_reward_scale`, `global_scale`, `local_scale`, all
`hold(1.0)`, and `global_scale` applies only to zero-weight components) with four
tier scales. Run 1 sets every tier scale to `hold(1.0)` except `shaping_scale`,
so the bug fixes and the new weight vector have an interpretable baseline. Design
the curriculum keypoints from run 1's EV curves, which will be the first ones
measured without the suppression.

Sketch for run 2, if run 1 supports it: damage 1.0 -> 0.6 and win 1.0 -> 1.5 over
roughly 50M-300M, kill/death held flat as the bridge. Avoid a mid-training hump on
kill/death — the tiers are +0.5 correlated, so a hump is unlikely to be
distinguishable from a flat hold in one run.

Separately: OpenAI Five's team-spirit anneal maps onto a **different** axis than
the tier ladder — self-only credit vs ally-shared credit, which here is the
diagonal-vs-team lambda in `_LOCAL_COMPONENTS`, currently a static per-component
property. Annealing that is the faithful analogue and is more directly aimed at
coordination than the tier schedule is.

## Static weights, not a controller

Question: hold the *designed* shares constant with a feedback controller, or fix
the weights and let realised shares drift?

Projected drift of the tier shares under the final static weights, using `d`
measured per window in run 719:

| tier | 20-50M | 50-150M | 150-400M | 400-1000M | drift |
|---|---:|---:|---:|---:|---:|
| win | 30.1% | 31.0% | 34.6% | 38.6% | 1.29x |
| kill/death | 32.3% | 28.1% | 24.1% | 23.4% | 0.73x |
| damage | 34.2% | 35.0% | 34.7% | 32.6% | 0.95x |
| shaping | 3.4% | 5.9% | 6.6% | 5.4% | 1.58x |

Static weights already produce the OpenAI-style curriculum: win rises 1.29x,
kill/death falls to 0.73x, damage stays flat. A constant-pressure controller would
cancel exactly the schedule we decided we wanted.

Three reasons to stay static for run 1:

1. The drift is the curriculum (above).
2. `d` drift is the instrument. Prediction 1 is that field_death's `d` rises from
   8.28 toward the pack once its critic is fed; under a controller that shows up
   as a silently lowered weight instead of a visible measurement.
3. Cost. Reward-level diagnostics ran ~29 s on a measured update against a ~36 s
   update at level 3 / 15 microbatches. Continuous readings for a controller are
   not free, and a low-duty-cycle controller acts on a sparse noisy estimate.

The cheap version of the same idea is an outer loop: measure `d`, re-solve
`w = share_target / d` between runs. Same fixed point, each run interpretable.

Note shaping drifts *up* 1.58x under static weights, so the `shaping_scale` taper
has to fight the drift as well as deliver the intended decay — taper harder than
the nominal target suggests.

If a controller is ever built: burn in past ~100M (`d` is erratic before that -
facing reads 14.7 / 7.8 / 14.0 across the first three windows), EMA over hundreds
of updates, clamp to +-30% of the static weight, target the designed share vector
rather than equal shares, log its state, and default it off.

## Outcome: the flat allocation was wrong, and the auxiliary was never the variable

The recommendation above — build the tier mechanism, ship run 1 flat at roughly
31/32/31/5 — was executed as `lyric-durian-722` and `breezy-cloud-723`. Both
finished behind `good-leaf-719`. A three-point round robin, 2048 games per pair
with both orderings sharing a seed:

| pair | 11.9M (exact) | ~72M | ~140M |
|---|---:|---:|---:|
| 719 vs 722 | −49.7 | +57.0 | +42.6 |
| 719 vs 723 | −76.4 | −10.6 | +13.2 |
| 722 vs 723 | −28.9 | −58.5 | −3.5 |

±15.4 Elo at 2σ. Note the crossover: at 12M, 719 was the *worst* of the three by a
wide margin. The flat allocation bootstraps faster and then plateaus, which is
what an outcome-weighted reward should do — sparse, high-variance, weakly
attributed. 719's kill-weighted reward is slower to exploit and keeps paying.

The plateau is visible in play, not just in Elo: 722 and 723 drew 25.0% of their
games against each other against 12.7% against 719, with lifespan 383 against ~250
and field deaths 50/M against 193/M. Paid mostly for the outcome, the policy
learns not to lose.

Two hypotheses died here, and the second was mine:

* **Auxiliary share does not explain the ranking.** 719's two state losses took
  0.371 of the trunk gradient — between 722's 0.430 and well above 723's 0.096 —
  and 719 beat both. 722 and 723 span a 4.5x range in auxiliary share and are
  separated by 3.5 Elo, inside the measurement error. Cutting
  `predictive_state_coef` from 0.2 to 0.02 was argued for from three converging
  lines of evidence and produced no strength difference. A large auxiliary share
  is normal for this trunk.
* **Flat tier allocation was a judgement without a check.** 719 is now that check.
  The weights are re-solved against its measured tier shares (17/54/25/4), which
  puts `U` at 1.0 rather than 0.38 — kill/death pressure up about 2.5x relative
  to the win pair.

What the balance rule cannot reproduce, and this is worth stating plainly: 719
paid `kill_shot = 1.0` *and* `combat_death = 1.0`, a 2:1 payout-to-charge ratio —
exactly the imbalance the rule exists to remove. So 719's `kill_shot` share of
0.145 becomes 0.071 spread across `kill_shot`, `kill_ally_shot` and
`enemy_field_death`. Tier totals match, and so does the deaths-vs-kill-credit
split *within* the kill/death tier (0.514/0.486 for both, which was not fitted),
but the concentration does not. If the mechanism behind 719's advantage was
finishing-blow credit specifically rather than kill/death pressure in aggregate,
the rule structurally cannot test that.

Two differences remain uncontrolled. `return_min_span` is the larger one: 719's
critic put essentially zero gradient on the sparse components (`ally_win` 0.001,
`kill_shot` 0.000) and spent everything on dense shaping, which is the starvation
fix #2 removed. The branch fixed a real bug and the run got weaker; the starvation
may have been load-bearing. That is the next suspect if the re-solved weights do
not recover the gap. The other is the predictive objective replacing the two state
losses, which the 722/723 comparison argues is second-order.

`shaping_scale` is back to `hold(1.0)` for the same reason the tiers are: 719
carried its shaping undecayed, and the taper would be one more difference than
this comparison can carry. The argument for tapering is unaffected and still
recorded above — restore it once the reward split is settled.
