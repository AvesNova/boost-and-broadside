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
| 1 | Lambda fix: normalize the *unweighted* pattern, then apply the weight. Bit-identical at `w = 1` — that equivalence is the test. | `train/rl/ppo.py` |
| 2 | Return scaler: RMS scale estimator (matching `AdvantageScaler`), `return_min_span` 1.0 -> 1e-2, Huber value loss. All three together. | `train/rl/buffer.py`, `train/rl/ppo.py`, `profiles/rl.py` |
| 3 | Extract `kill_ally`; replace the three dead group scales with four tier scales; add the shaping taper. | `env/rewards.py`, `config/core.py`, `config/defaults.py`, `train/rl/ppo.py` |
| 4 | New weight vector. | `config/defaults.py` |
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
| ally_win | win | - | 1.5 *(eff 1.0)* | **1.00** | 11.2% | 16.6% |
| enemy_win | win | - | 1.5 *(eff 1.0)* | **1.00** | 10.4% | 15.4% |
| kill_shot | kill/death | O | 1.0 | **0.32** | 13.4% | 6.3% |
| kill_assist | kill/death | O | 1.0 | **0.36** | 11.9% | 6.3% |
| combat_death | kill/death | D | 1.0 | **0.31** | 13.7% | 6.3% |
| field_death | kill/death | D | 1.0 | **0.32** | 8.3% | 6.3% |
| kill_ally | kill/death | both | — | **0.32** | — | 6.3% |
| damage_dealt_enemy | damage | O | 0.5 | **0.52** | 7.4% | 11.4% |
| combat_damage_taken | damage | D | 0.5 | **0.31** | 6.8% | 6.3% |
| field_damage_taken | damage | D | 0.5 | **0.25** | 6.9% | 5.1% |
| damage_dealt_ally | damage | both | 0.5 | **0.48** | 5.8% | 8.2% |
| facing | shaping | - | 0.1 | **0.09** | 1.6% | 2.2% |
| closing_speed | shaping | - | 0.1 | **0.08** | 2.6% | 3.1% |

Tier totals **32.1 / 31.6 / 31.1 / 5.2**. Offence **38.7%**, defence **38.6%**,
friendly fire **14.6%**.

`field_death` and `kill_ally` use the pack-median `d` because neither has a
trustworthy measurement — `field_death`'s current `d` is the thing prediction 1
says will move, and `kill_ally` has never existed standalone. Both are provisional;
re-solve after the first diagnostic update of the new run.

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
