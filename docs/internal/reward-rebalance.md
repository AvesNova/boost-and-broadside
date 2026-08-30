# Reward weighting: bug fixes and rebalance

Working notes for the next training run. Delete once it has landed.
Detailed evidence is in the appendix below; every number was measured from
`good-leaf-719` (its `artifacts/wandb-export/*/history.jsonl` and the
`scaler_state_dict` in `step_000999309312.pt`).

## What this run changes and why

Two bugs in the weighting path mean the reward weights in the config have never
been the weights the run trained under.

- **Bug 1 — the lambda clamp cancels weights.** Local weights saturate at
  `min(w, 1)`; global ones cancel entirely. `ally_win_weight=1.5` trains exactly
  like `0.25`. Confirmed: components at 2x different configured weights deliver
  identical gradient share per unit weight.
- **Bug 2 — `return_min_span=1.0` starves 8 of 12 critics.** Critic targets are
  compressed up to 121x and critic gradients up to 1.5e4x. The two critics whose
  explained variance *falls* over training are the first and fifth most suppressed.

With those fixed, weights become a real knob, so the run also adopts a designed
weight vector, extracts a reward component that is currently hidden inside two
others, and stops the behaviour-cloning term from eating the gradient clip.

## Phases

Each is one commit. Order matters: the weight vector is meaningless before
phase 1, and its `d` calibration assumes the critics of phase 2.

| # | Change | Files |
|---|---|---|
| 1 **done** | Lambda fix: normalize the *unweighted* pattern, then apply the weight. Bit-identical at `w = 1` — that equivalence is the test. | `train/rl/ppo.py` |
| 2 **done** | Return scaler: masked mean/std estimator, `return_min_span` 1.0 -> **1e-3**, Huber value loss. All three together. | `train/rl/buffer.py`, `train/rl/ppo.py`, `profiles/rl.py` |
| 3 **done** | Extract `kill_ally_shot`/`kill_ally_assist`; replace the three dead group scales with four tier scales; add the shaping taper. | `env/rewards.py`, `config/core.py`, `config/defaults.py`, `train/rl/ppo.py` |
| 4 **done** | New weight vector. | `config/defaults.py` |
| 5 | `behavior_cloning_coef` 2.0 -> 0.5. During BC, 100% of updates hit `max_grad_norm` (median total norm 2.18, max 13.3), so every other term trained at 0.1-0.5x its nominal step. After BC, 2.6% clip. | `config/defaults.py` |

Each phase: `pytest`, then `--smoke`, tests added for new behaviour, docs updated,
commit at the phase boundary.

## Weight table

Tiers are win / kill-death / damage / shaping. Offence is dealing damage, defence
is taking it; the friendly-fire pair counts in both. Weights are solved as
`w = share_target / d`, where `d` is the measured policy-gradient share a
component delivers per unit of effective weight (stable to 6-15% CV across the
run). The win pair is pinned at 1.00 by choice.

| component | tier | side | w now | **w new** | share now | share new |
|---|---|:--:|---:|---:|---:|---:|
| ally_win | win | - | 1.5 *(eff 1.0)* | **1.00** | 11.2% | 16.2% |
| enemy_win | win | - | 1.5 *(eff 1.0)* | **1.00** | 10.4% | 15.0% |
| kill_shot | kill/death | O | 1.0 | **0.28** | 13.4% | 5.4% |
| kill_assist | kill/death | O | 1.0 | **0.31** | 11.9% | 5.3% |
| kill_ally_shot | kill/death | both | - | **0.28** | - | 5.4% |
| kill_ally_assist | kill/death | both | - | **0.28** | - | 5.4% |
| combat_death | kill/death | D | 1.0 | **0.27** | 13.7% | 5.3% |
| field_death | kill/death | D | 1.0 | **0.28** | 8.3% | 5.4% |
| damage_dealt_enemy | damage | O | 0.5 | **0.54** | 7.4% | 11.6% |
| combat_damage_taken | damage | D | 0.5 | **0.32** | 6.8% | 6.3% |
| field_damage_taken | damage | D | 0.5 | **0.26** | 6.9% | 5.2% |
| damage_dealt_ally | damage | both | 0.5 | **0.50** | 5.8% | 8.4% |
| facing | shaping | - | 0.1 | **0.09** | 1.6% | 2.1% |
| closing_speed | shaping | - | 0.1 | **0.08** | 2.6% | 3.0% |

Tier totals **31.2 / 32.2 / 31.4 / 5.1**. Offence **41.4%**, defence **41.4%**,
friendly fire **19.2%**. Both win components are pinned at 1.00 by choice; their
realised shares differ by 1.2 points only because their measured `d` differs by
8%, which is almost certainly measurement asymmetry on what is one event seen
two ways.

The friendly-kill signal is two components mirroring the enemy pair on both
horizons, so the kill/death tier is six components at equal share rather than
five. That lands friendly fire at 19.2% across the two tiers -- the "counted in
both offence and defence" weighting -- without breaking equality inside the tier.

`field_death`, `kill_ally_shot` and `kill_ally_assist` use the pack-median `d`.
None has a trustworthy measurement: field_death's 8.28 is the lowest of the
twelve and prediction 1 says it moves, and the friendly-kill pair has never
existed standalone. Re-solve all three off the first diagnostic update.

### Constant, except shaping

Weights are static for the whole run. Under static weights the realised tier
shares drift on their own — win 1.29x up, kill/death 0.73x down, damage flat —
which is the curriculum we wanted, arriving without a schedule. A
constant-pressure controller would cancel it. `d` drift is also the instrument
that tells us the phase-2 fix worked, and a controller would absorb that signal
into the weight instead of showing it.

Shaping is the exception. It drifts *up* 1.58x under static weights, so its taper
has to overcome the drift as well as deliver the intended decay:

```
shaping_scale = (0, 1.0, "hold"), (100_000_000, 1.0, "exponential"),
                (400_000_000, 0.05, "hold")
```

That lands realised shaping share at roughly 5% early and under 0.5% late. The
floor is 0.05 rather than 0 so the components stay measurable — `d` and explained
variance for `facing` and `closing_speed` remain readable all run. (Zero is safe
for `_active_names`, which freezes at init from the *initial* weight, and the
returns do not collapse either, because component rewards are stored unweighted
and the weight only enters through lambda.)

## What to check on the first diagnostic update

1. `field_death`'s `d` rises from 8.28 toward the pack (~12-14) and its CV drops
   from 28.6%. If it does not move, field deaths really are just rare and the
   component should be demoted.
2. `field_death` explained variance rises above 0.49 and `damage_dealt_ally` above
   0.56, and both stop declining.
3. Win-pair and kill critics' EV barely moves (already 0.91-0.92 under ~10x
   suppression); their value-share rise is reallocation, not repair.
4. `ally_win_weight` is a linear knob — realised win share should land near 32%.
5. Aggregate value loss rises ~14x and the critic's top-level gradient share moves
   2.4% -> ~5-6%. `max_grad_norm` should still rarely bind.

## Out of scope for this run

Tier scheduling (build the mechanism in phase 3, ship it flat; design keypoints
from this run's EV curves). Team-spirit annealing on the local-vs-shared lambda
axis. Joint 2x8x3 action space. Latent multi-step rollout head. EV-gated dynamic
weights.

---

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

## Falsifiable predictions

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

## Missing component — friendly kills

There is no `kill_ally`. The friendly-kill penalty is folded *inside*
`KillShotReward` and `KillAssistReward` (`reward -= dm_friendly/total_friendly`),
so one critic head predicts the sum of a positive enemy-kill signal and a negative
friendly-kill signal under one gamma and one weight. Extract it as its own
component so it can be weighted and learned separately. Note `K` grows, which
dilutes the per-component value-loss divisor, and `EnvWrapper._active_names` is
frozen at init from `weight != 0`, so it must launch with a non-zero weight.

## Later, not started

Joint 2x8x3 action space (own ablation; note a joint entropy bonus penalizes the
inter-dimension correlation the joint head exists to express). Latent multi-step
rollout head. EV-gated dynamic weights.

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

# Outcome: the derivation was right, the allocation was not

Everything above was written before any of it had run. Five runs later, on a
single Elo scale, this is what it bought. Ratings are from one Bradley-Terry fit
over all six runs' ladder checkpoints and a shared random / scripted /
semi-scripted reference field — 34 players, 49,152 games, every rating inside
±15 Elo — in `artifacts/elo-calibration/20260828T203651Z-be16de50/`. Per-run
calibrations are each on their own scale and cannot be compared; this one can.

Gap to 719 at matched steps, interpolating 719's ladder in log(step), ±26 at 2σ:

| run | what it changed against its predecessor | final | mid-run |
|---|---|---:|---|
| 720 | the four bug fixes, weights solved against measured share | **+58** | +60 @72M, +66 @124M |
| 721 | weights re-derived from the balance rule | −16 | −47 @83M, −2 @145M |
| 722 | predictive belief state replaces the one-step head | −53 | −37 @71M, −37 @115M |
| 723 | predictive coefficients cut 4.5x | −87 | −12 @74M, −46 @150M |
| 724 | weights re-solved against 719's own tier split (`U` 0.38 → 1.00) | **−98** | −35 @33M, −33 @69M |

**720 is the only configuration on this branch that beats 719**, and it beats it
everywhere it was measured. **721 is the cleanest single-variable result**: it
differs from 720 only in how the weights were derived, and that derivation cost
about 60–70 Elo. Everything after 721 inherits 721's weights, so nothing
downstream ever got to run on the vector that worked.

## What separates 720, and it is not a tier share

Share-targeting was abandoned above for a reason that turned out to be right for
the wrong reason. The reason given was that `w = share / d` keeps pressuring a
solved problem. The reason it actually failed is simpler: **tier share is not the
quantity that decides how a policy plays.** Run 724 is the proof — it hit 719's
measured split to within 5% at the tier level and within noise at the top level,
and produced the most passive policy of the set: lifespan 421 at matched steps
against 719's 259, and the worst rating on the board.

The quantity that tracks strength is the ratio between what a death charges and
what a kill pays:

| run | death charge | kill payout | ratio | gap to 719 |
|---|---:|---|---:|---:|
| 719 | 1.00 | `kill_shot` 1.00 + `kill_assist` 1.00 | **2.00** | — |
| 720 | 0.27 | 0.28 + 0.31 | **2.19** | +58 |
| 721 / 722 / 723 | 0.38 | 0.19 + 0.19 | 1.00 | −16 / −53 / −87 |
| 724 | 1.00 | 0.50 + 0.50 | 1.00 | −98 |

720 kept a 2:1 payout by accident, the same way 719 did — its weights were solved
per component against measured `d`, before the balance rule tied the two sides
together. Every run that sits at 1:1 is more passive than 719 and rates below it.
That is five runs agreeing on one number, and it was visible in the play as well:
lifespans at matched steps run 208 (720), 259 (719), 282, 307, 315, 421 (724),
in almost the same order as the ratings.

The mechanism is not subtle. Charging a death `U` and paying a kill `U` makes an
even trade worth exactly nothing, so a policy that cannot reliably win the trade
declines it. Paying the aggressor more is what makes the trade worth taking.

## What this branch does about it

The balance rule is kept — it is a good derivation and it is why only a handful of
numbers are set — and given the one knob it structurally cannot express:
`kill_payout_ratio`, kill tier only, default 1.0.

The free numbers then stop chasing a share target and reproduce 719's effective
vector directly: `W` 1.0, `U` 1.0, `V` 0.5, `f` 0.5, `k` 2.0. That is an exact
match on all twelve components 719 carried — see
`TestShippedWeightsReproduceRun719`, which is the test that fails if this drifts.
Effective, not configured: 719's config said `ally_win_weight=1.5`, but the
lambda normalization order divided the weight back out of every global component,
so the win pair trained at a total of 1.0. The rest of 719's vector is its config.

Only two components are added, both required by the rule and both zero in 719:
`enemy_field_death` at 1.0 and `enemy_field_damage` at 0.5, so a ship killed by a
field pays its opponents something. The `kill_ally_*` pair is not an addition —
719 penalized friendly kills at an unscaled −1.0 share folded inside
`KillShotReward` under `kill_shot`'s own weight, which is 1.0, exactly where the
derivation puts it.

`shaping_scale` returns to `hold(1.0)` for the same reason: 719 carried its
shaping undecayed, and the taper would be one more difference than this
comparison can carry. The argument for it is unaffected and still recorded above.

## What is still uncontrolled, and what to check

`return_min_span` is the one difference from 719 that nobody has isolated, and it
is the largest. 719's critic put essentially zero gradient on the sparse
components (`ally_win` 0.001, `kill_shot` 0.000) and spent everything on dense
shaping; the scaler fix removed that starvation, and the branch got weaker. The
branch fixed a real bug and the starvation may have been load-bearing. If this
run does not recover 720's margin, that is the next suspect and it is a
one-variable test.

Not carried here on purpose: the predictive belief state. It costs 26% throughput
(2066 sps against 719's 2707, and 722–724 ran gradient diagnostics at interval 5
rather than 1, so the true cost is larger), and across 722 and 723 it never
showed a strength gain to pay for that. The reward change and the objective
change have never been separated in one run; this branch is the reward arm, on
the fast code path.

On the first diagnostic update past the behavior-cloning decay:

1. Realised tier shares. They are no longer targeted, so they are a reading
   rather than a check — but record them, because the whole point is that this
   vector is expected to land somewhere the share-solve would have rejected.
2. `enemy_field_death` and `enemy_field_damage` are active for the first time at
   a weight this high. Their coherence has never been measured.
3. `scaler/floor_bound_span_count` stays 0.
4. Lifespan. It is the cheapest early read on whether the 2:1 payout is doing
   what five runs of evidence say it should — 720 sat at 208 against 724's 421.

## Run 725: the kill ratio reproduces 719, and only 719

725 ran 719's effective vector with `k = 2` on the pre-predictive code path. A
joint Bradley-Terry fit over all seven runs — 39 players, 49,152 games, every
rating within ±15 Elo, in
`artifacts/elo-calibration/20260829T151130Z-33da087c/` — puts it here against
719 at matched steps:

| step | 30.9M | 82.6M | 133.4M | 153.8M |
|---|---:|---:|---:|---:|
| 725 − 719 | −38 | −52 | **−1** | **+3** |

Parity, reached from below at about 130M. The design goal was to rebuild 719 and
it rebuilt 719, behaviour included: lifespan 268 over 100–155M against 719's 267,
realised tier shares 13.9/57.3/25.0/3.8 against 719's measured 17/54/25/4 without
targeting them, friendly-fire pressure 12.3% against the 12.8% the derivation
predicted, `floor_bound_span_count` 0, and 2,764 sps on the fast path.

It did **not** reach 720. The fit's reference player is `r720_ladder_71.7M`, so
each stderr above is the error on the gap to that exact point: 725 at 82.6M rates
1142 against 720's 1188 at 71.7M — **−46 ± 19 at 2σ, with 15% more steps**. At
the next band, 725 at 133.4M against 720 at 124.4M is −32 [±27]. Marginal
individually, consistent in sign.

So the kill payout ratio explains 725 ≈ 719 and does not explain 720 > 719.

## The damage tier was tilted in 720 too

720's weights were solved per component against measured coherence, and that left
the damage tier asymmetric in the same direction as its kill tier:

| run | damage dealt | damage taken | ratio | vs 719 |
|---|---:|---:|---:|---:|
| 719 | 0.50 | 0.50 | 1.00 | — |
| **720** | 0.54 | 0.32 | **1.69** | +58 |
| 725 | 0.50 | 0.50 | 1.00 | ~0 |

The kill ratio was scoped to its own tier on the grounds that "damage and win
were balanced in 719 as well, so the evidence for an asymmetry is specific to
this tier." That reasoning had one run in it. 720 is a second and it disagrees.

The obvious alternative does not hold: 720's shaping taper does not begin until
100M, and 720 was already +40 at 71.7M.

Hence `damage_payout_ratio`, damage tier only, default 1.0, set to 2.0. Win stays
balanced — it is one signal to each side of the same event, with no third party
to pay.

### Two caveats on this run, recorded before it starts

1. **2.0 is symmetry, not measurement.** 720 measured 1.69. One run cannot
   resolve the difference, and a round number that matches `k` is easier to
   reason about, but nothing measured says 2.0.
2. **The tier share moves with the knob, and that is a confound.** Doubling the
   paid side takes the damage tier from about 25% of the policy gradient to about
   34% and dilutes kill/death from 57% toward 50% — a move *toward* 720's flat
   31/32/31/5. Since 720 and 721 shared that flat allocation and differed only in
   payout ratio, allocation is a live alternative explanation for 720's margin,
   and this run moves both at once. A win here is evidence for "720's direction"
   rather than for the damage ratio specifically. Separating them needs a second
   run holding tier shares fixed while the ratio moves, which the tier scales can
   do.

## Reconstructing 720 instead of 719

Run 726 tested 719's vector plus a damage tilt. It was stopped at 103.5M once a
config diff made the point clearer than the run would have: **726 was never a
reconstruction of 720.** It was 719 with one thing changed, and the kill/death
level relative to the win pair stayed at 719's, which is 3.5x heavier than 720's.
Its measured tier shares said so — 11.8 outcome / 52.0 kill-death / 32.4 damage
against 720's 29.3 / 30.8 / 36.1 at the same steps.

So the profile now fits the derivation to 720 directly, by least squares on log
weights. Log space because the weights span 0.08 to 1.0 and only ratios matter.
The problem separates in logs, so the fit is closed-form; it was checked against
a coordinate-descent optimiser with random restarts and agrees to every digit.

| fit | W | U | V | f | k | d | RMS | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eyeballed first pass | 1.000 | 0.2800 | 0.2900 | 0.500 | 2.00 | 2.00 | 7.3% | 16.0% |
| ratios pinned at 2 | 1.000 | 0.2831 | 0.2738 | 0.4875 | 2.00 | 2.00 | 5.9% | 14.5% |
| all six free | 1.000 | 0.2750 | 0.2884 | 0.4873 | 2.09 | 1.80 | 5.0% | 10.9% |
| **shared ratio, f even (shipped)** | **1.000** | **0.283** | **0.274** | **0.500** | **2.00** | **2.00** | **6.0%** | 14.4% |

Freeing the two ratios separately buys 5.9% → 5.0% and returns 2.09 and 1.80.
Tying them to one number and solving returns **1.96**, which the fit cannot
distinguish from 2.0 (5.97% against 6.01% RMS). One shared ratio is the smaller
claim and keeps a single number across runs 725, 726 and this one, so 2.0 it is.
Pinning `f` even costs 0.2% RMS and keeps the standing principle.

At `r = 2` and `f = 0.5` the kill payout `r·U·f` equals `U`, so the whole
kill/death tier lands on 0.283. That is arithmetic, not a coincidence.

The ~6% residual is irreducible. The rule forces pairs equal that 720 had
unequal: `combat_damage_taken` 0.32 against `field_damage_taken` 0.26 (23%
spread, the worst), `kill_assist` 0.31 against `kill_shot` 0.28, `damage_dealt_ally`
0.50 against `damage_dealt_enemy` 0.54. Those spreads fell out of 720's own
`w = share / d` solve rather than out of a principle, so some of that residual is
fitting 720's noise.

### Everything else that differs from 720

A full flattened diff of the resolved config against
`checkpoints/silvery-pond-720/config.json` now reports **no non-reward
differences at all**. `shaping_scale` was the last one and is restored to 720's
taper. Physics, model, env, every schedule, `microbatch_tokens`, `num_envs`,
`total_timesteps`, all per-component gammas and lambdas and the other three tier
scales were already identical.

Code: between 720's commit `3d7f86d` and here, the only training-path changes are
the weight derivation, the two ratio knobs, and one plumbing edit — `ppo.py`
reads `self._component_weights[name]` where it read
`getattr(cfg.rewards, f"{name}_weight")`. For a fixed weight vector that is a
no-op. Nothing touches the loss, the lambda matrix, the scalers or the rollout.

What remains, and neither is a choice:

1. **K goes 14 → 16.** `enemy_field_death` and `enemy_field_damage` are zero in
   720 and non-zero under the rule, so the critic widens and every component's
   share of the per-component value loss is diluted by 14/16. A field kill pays
   0.283 where 720 paid nothing for the shot half.
2. **720's taper is untested past 127M.** It ran 27M steps into the decay and
   ended near 0.76. Everything the taper does after that is inherited on faith.

And the seed differs, which is worth stating plainly: 720's whole +58 rests on a
single run with no replicate.
