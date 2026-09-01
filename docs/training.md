# Training system

Boost and Broadside trains a recurrent centralized team policy with clipped
[PPO](https://arxiv.org/abs/1707.06347). The training loop combines scripted-opponent
bootstrapping, self-play, a running-average policy, frozen historical opponents,
decomposed rewards, and continuous evaluation.

This page explains the current implementation and calls out reference-run settings where
they matter. Exact results and post-hoc methodology are in [evaluation](evaluation.md).

## Reference run at a glance

The reference run is `good-leaf-719`. Its
[stored configuration](../checkpoints/good-leaf-719/artifacts/wandb-export/) records:

| Setting | Value |
|---|---:|
| Target environment steps | 1,000,000,000 |
| Parallel environments | 2,592 × 3 rollouts per update |
| Ships per environment | 8 total (4-vs-4), plus 4 refractive fields |
| Rollout length | 128 steps |
| Decisions per second | 30 (`action_repeat=2`) |
| PPO minibatches | 32 |
| Token width / attention heads / blocks | 128 / 4 / 2 |
| Episode horizon | 1,024 physics ticks (17.1 s) |
| Opponent paradigm | `ego_pass` |
| Elo evaluation games per matchup slot | 512 |

The run logged 999,309,312 steps over 1,004 updates before finishing. Where this page
and the stored configuration disagree, the stored configuration is what actually ran.
Earlier runs are listed in [training runs](training-runs.md).

## Recurrent PPO lifecycle

[`PPOTrainer`](../src/boost_and_broadside/train/rl/ppo.py) owns environment groups,
policies, rollout collection, [generalized advantage
estimation](https://arxiv.org/abs/1506.02438), update-time sequence re-evaluation,
evaluation, logging, and checkpoints.

### Decision rate

Physics always runs at `ShipConfig.dt` = 1/60 s. `EnvConfig.action_repeat` sets how many
of those ticks each chosen action is held for, so collision and projectile integration
are unaffected and only the rate at which the policy may change its mind moves. The
primary profile holds for 2 ticks, giving **30 Hz decisions**.

That rate is set by the plant, not by the renderer:

| timescale | seconds | decisions @ 30 Hz |
|---|---:|---:|
| firing cooldown | 0.10 | 3 |
| bullet flight to ~200 px | 0.40 | 12 |
| full 360° turn | 1.3–2.3 | 39–69 |
| mean episode | ~4.7 | ~140 |
| `num_steps=128` rollout | 4.3 | 128 |

At 60 Hz five of every six shoot decisions were no-ops against the cooldown, consecutive
observations differed by 17 ms, and a 128-step rollout spanned 2.1 s against a ~4.7 s
episode, so the recurrent policy never saw close to a whole episode inside one BPTT window.

The rate is a real trade, not a free win. Combat damage per live ship-step falls
monotonically as the hold grows. The measurement uses the *fixed* scripted controller over
equal game time, so the policy cannot adapt and the whole difference is the environment:

| | 60 Hz | 30 Hz | 20 Hz | 15 Hz |
|---|---:|---:|---:|---:|
| combat damage / live ship-step | 0.2965 | 0.2814 | 0.2656 | 0.2475 |

30 Hz gives up 5% of combat effectiveness and halves the tokens per second of game time.
20 Hz gives up 10% for a third, which did not pay.

**`action_repeat` is honoured by `TensorEnv.step`, not by the wrapper**, so evaluation runs
at the same rate as training. It was briefly the other way round, and the failure was
silent: a policy trained to hold an action for N ticks but evaluated one tick per action
turns a fraction of its intended amount per decision, mistimes every lead, and advances its
recurrent state N times too fast for the game clock. `YemongEnvWrapper` opts out via
`tick`, because it has to accumulate rewards and episode statistics per physics tick.

Rewards are summed across the held ticks. That is scale-preserving: over a fixed span of
game time both the dense per-tick terms and the one-off event terms total exactly what
they would at repeat 1, so the component ratios in `RewardConfig` are untouched. Episode
lengths and ship ages stay in physics ticks so they remain comparable across rates.

Discounts do **not** carry over unchanged. They were chosen as horizons in seconds, so
moving the rate requires `gamma_new = gamma_old ** (rate_old / rate_new)`, and the same
for the GAE lambdas, since variance accumulates per unit of game time rather than per
decision. The horizons those values encode are tabulated in
[`config/defaults.py`](../src/boost_and_broadside/config/defaults.py).

Ships also spawn with randomised health, power and cooldown
(`EnvConfig.spawn_resource_spread`). Spawning at full resources every episode made health
an almost deterministic function of elapsed time, which the critic can read off the clock
instead of the state, and meant damaged-fleet positions were only reachable by playing
two hundred steps into them. Draws are per-ship but balanced in expectation across teams,
so no outcome variance enters the win signal that a policy could not have influenced.

### Action timing

Environment and policy run concurrently on separate CUDA streams, which costs one
decision of action latency: the step that advances the environment applies the action
chosen on the *previous* decision, while the policy computes the next one from the
current observation.

The observation is what makes that Markov. `previous_action` does not hold the action
that already ran. It holds the action **about to be applied**, written into the
observation as it is handed forward. So the stored transition is
`(state, pending action) → action`, and a chosen action shows up in the reward one step
later, which GAE handles through the value function.

Two consequences worth knowing before touching the auxiliary losses:

- The channel is `(B, N+M, 3)`, so spatial attention lets **every ship read every other
  ship's pending action**, opponents included. One decision is 1/30 s against a ~0.4 s
  bullet flight, so the lookahead is small, and it is symmetric.
- One-step next-state prediction is therefore a *deterministic* function of the
  observation (up to `bullet_spread`), not merely a short-horizon one. That is why it is
  a weak representation signal and why longer-horizon prediction is the useful version.

A logical update proceeds as follows:

1. collect `T` actions while preserving `T+1` observations for bootstrap and next-state
   labels;
2. record actions, factored log probabilities, per-component values/rewards, masks, and
   recurrent boundary state in the [`RolloutBuffer`](../src/boost_and_broadside/train/rl/buffer.py);
3. compute per-component GAE and return-normalization statistics;
4. re-evaluate complete recurrent sequences with the stored initial hidden state;
5. optimize clipped policy, value, entropy, behavior-cloning, and auxiliary prediction
   objectives;
6. update average/league/evaluation state and save scheduled checkpoints.

The buffer stores precision-tolerant leaves in bfloat16 and categorical channels in small
integer dtypes, while reductions and running statistics use float32. Current profiles can
also collect multiple fixed-width rollout shards into a host-backed logical batch, a
newer design documented in [memory optimization](engineering/memory-optimization.md);
the reference run predates it.

## `ego_pass` and team perspective

The combat policy is trained from a canonical team-0 perspective. During rollout,
[`_rollout_policy_pass`](../src/boost_and_broadside/train/rl/ppo.py) batches the raw
observation with a team-flipped view. The same weights therefore produce candidate actions
for both perspectives in one evaluation.

[`opponents.py`](../src/boost_and_broadside/train/rl/opponents.py) then composes the action
tensor according to each environment group's assigned opponent. In self-play, the learned
weights act for both teams, with team 1 using the flipped observation. In scripted or
league games, the opponent's actions replace the corresponding side.

Only the ego-side actor mask contributes the policy-gradient and behavior-cloning losses.
This keeps the learning convention consistent while still generating both sides of a
self-play battle efficiently.

## PPO and auxiliary losses

The total update combines:

- clipped PPO policy loss over the sum of the three action-factor log probabilities;
- normalized per-component critic mean-squared error;
- entropy bonuses for power, turn, and shoot distributions;
- behavior cloning from the scripted controller, gated down as scripted win rate rises;
- one-step next-state prediction error;
- a cumulative triangle-window position/velocity drift loss;
- optional sketched isotropic Gaussian regularization of the embedding space
  (SIGReg, from [LeJEPA](https://arxiv.org/abs/2511.08544)), disabled in the
  reference configuration.

Two gates key off the same signal: the raw win rate against the scripted controller,
taken from the evaluation battery rather than from training envs. It decays the
behavior-cloning weight to zero at `bc_winrate_target`, and it tightens `target_kl` at
`high_winrate_threshold`. Using one measure of "is the policy strong yet" rather than two
also keeps the trust region independent of the Elo gauge, which would otherwise need
re-deriving whenever the anchor or the environment moved.

Advantages and returns are both scaled per component with running second-moment
statistics in symlog reward space, which keeps critic targets in a stable range without
forcing components with different natural scales into one value head. Returns map two
standard deviations to one. A standard deviation rather than a percentile span, because
for a sparse component — a death, a win, a friendly kill — the return distribution is a
spike at zero with rare large excursions, and p5/p95 measures the width of the spike:
run 719's `field_death` returns had a p5–p95 span of 0.0059 against a full range of
0.137, while dense components disagreed by 1.1–1.3x. Dead ships are excluded from both
scalers, since their returns sit at zero and would narrow exactly the components that can
least afford it. The exact loss assembly and logging proxies live in
[`ppo.py`](../src/boost_and_broadside/train/rl/ppo.py).

The critic's loss is squared error out to `value_huber_delta` normalized units and linear
beyond, matching plain squared error in the bulk so the switch reshapes only the tails.
Per-component normalization necessarily produces those tails, and bounding them here is
what lets `return_min_span` be a true epsilon.

Both scalers carry a floor, and a floor that binds on an active component replaces that
component's own scale with the guard's. Both are true epsilons:
the terminal win signal's advantage RMS is around 0.008, two orders of magnitude below a
per-step damage signal, and an earlier floor of 0.1 was downweighting it roughly
thirteenfold in the policy gradient. `return_min_span` was likewise held at 1.0 for a
time, where it bound eight of twelve components on every update of the reference run and
suppressed `field_death`'s critic gradient by four orders of magnitude; it now sits at
1e-3, more than an order of magnitude below the narrowest live component's spread.
`scaler/floor_bound_span/*` and
`scaler/floor_bound_rms/*` report which components each floor is currently holding up.

## Reward decomposition

Rewards are emitted as named components by [`rewards.py`](../src/boost_and_broadside/env/rewards.py).
Each active component receives its own critic output and can have its own GAE gamma/lambda
horizon. Weights are magnitudes; each component carries its own sign, noted below. The
reference policy activated these components:

| Component | Weight | Tier | Role |
|---|---:|---|---|
| `ally_win` | `W` = 1.00 | outcome | +1 to each surviving teammate on a win |
| `enemy_win` | `W` = 1.00 | outcome | opponent's win signal, seen as −1 through a negative enemy lambda |
| `combat_death` | `U` = 0.283 | kill/death | −1 when projectile damage kills this ship |
| `field_death` | `U` = 0.283 | kill/death | −1 when boundary damage kills this ship |
| `kill_shot` | `r·U·f` = 0.283 | kill/death | fatal-step credit (+), proportional to that step's damage |
| `kill_assist` | `r·U(1−f)` = 0.283 | kill/death | assist credit (+), proportional to cumulative episode damage |
| `kill_ally_shot` | `r·U·f` = 0.283 | kill/death | blame (−) for a teammate's death, by that step's damage |
| `kill_ally_assist` | `r·U(1−f)` = 0.283 | kill/death | blame (−) for a teammate's death, by cumulative damage |
| `enemy_field_death` | `r·U·f` = 0.283 | kill/death | credit (+) to the enemy team when a ship dies to a field |
| `combat_damage_taken` | `V` = 0.274 | damage | −applied projectile health loss |
| `field_damage_taken` | `V` = 0.274 | damage | −applied boundary health loss |
| `damage_dealt_enemy` | `r·V` = 0.548 | damage | +proportional to damage dealt to enemies |
| `damage_dealt_ally` | `r·V` = 0.548 | damage | −proportional to friendly fire dealt |
| `enemy_field_damage` | `r·V` = 0.548 | damage | +to the enemy team when a ship takes field damage |
| `facing` | 0.09 | shaping | dense aim geometry (+) |
| `closing_speed` | 0.08 | shaping | dense approach geometry (+) |
| `shoot_quality` | off | shaping | firing opportunity quality (+); head retained at zero weight |

Only five of those numbers are set. The rest follow from one rule: **an event pays one
side exactly what it charges the other.** A death costs the dying ship's team `U` and pays
whoever caused it `U` between them; damage does the same with `V`; a win pays `W` and
charges `W`. `f` — how the kill budget splits between "landed the finishing blow" and
"contributed damage" — is the only ratio the rule leaves free, and it is even.

`r` is the one named exception, and it applies to both event tiers. At `r = 1` the rule
holds exactly; above it, the side that *caused* an event is paid more than the side it
happened to is charged — at `r = 2` a death still charges the dying team `U` but pays its
killers `2U`, and damage still charges its victim `V` but pays whoever dealt it `2V`.
Nothing else in the system can express that, because raising `U` or `V` raises the charge
and the payout together. It is carried as two config fields, `kill_payout_ratio` and
`damage_payout_ratio`, so either tier can be moved alone; they are set to the same number
because one shared ratio is the smaller claim, and because solving them separately gave
2.09 and 1.80 — a difference the fit cannot resolve.

Win stays balanced: it is one signal to each side of the same event, with no third party
to pay.

At `r = 2` and `f = 0.5` the kill payout `r·U·f` equals `U`, so every component in the
kill/death tier lands on the same number. That looks like a coincidence and is not.

The friendly-fire components follow the payout rather than the charge, so neither ratio
makes harming a teammate quietly cheaper as it makes harming an opponent more attractive.

Three things follow that look like coincidences and are not.

`enemy_field_death` has to equal `kill_shot`. A ship killed by a field was shot by nobody
on its fatal step, so `kill_shot` reads zero there and only `kill_assist` fires; the
shortfall is exactly `kill_shot`, and something has to make it up or field kills would pay
less than combat kills. `enemy_field_damage` equals `V` by the same argument. These are the
only two source-split components with a non-zero weight, and that is what they are for:
supplying the offensive side of events that have no shooter to attribute to. It is also the
principled form of "reward for forcing an enemy into a field" — attributed by team rather
than by proximity, so it survives a change of fleet size.

Killing a teammate costs the team twice: the ally is charged `U` for dying and the shooter
is charged `U` for causing it, while the enemy is paid nothing. Friendly fire is
structurally twice as expensive as being killed by an opponent, with no special case saying
so.

The remaining `ally_*` and `enemy_combat_*` components stay at zero. Their events are
already fully paid for by the local per-ship components and by damage attribution, so
turning them on would charge the same event twice.

Equal weight is not equal gradient, and is not meant to be. The kill side spends its weight
across two correlated components (+0.509 cosine) while the death side spends it on one, so
the kill side delivers roughly 87% of the death side's gradient magnitude. Weights state
what an event *means*; how much pressure it exerts is allowed to follow how coherent the
signal actually is. That is deliberate — a component whose behaviour the policy has already
solved produces gradients that increasingly cancel, and its influence should fade on its
own rather than be propped up. In the reference run `field_death`'s coherence halved as
field deaths fell eightfold.

The numbers are **solved, not chosen.** The target is the one configuration measured that
beat the reference run, by +58 Elo at matched steps on a joint Bradley-Terry fit. That
run's weights were not derived — they came from `w = share / d` against measured gradient
coherence, which is why no two of them are equal — so these five numbers are the closest
the derivation can come to that vector, by least squares on log weights. Log space because
the weights span 0.08 to 1.0 and only ratios matter, so a 10% error on `facing` should
count like a 10% error on `ally_win`. The fit has a closed form and was checked against a
numeric optimiser.

It lands within 6% RMS. The residual is irreducible rather than a tuning failure: the rule
forces pairs equal that the target had unequal — `combat_damage_taken` 0.32 against
`field_damage_taken` 0.26 is the worst of them, and that spread came out of a
per-component solve rather than out of any principle.

Two earlier vectors are worth knowing about, because they are what this replaced. Setting
the numbers to a *tier-share target* was tried twice and lost both times. Copying the
reference run's own vector exactly was tried once: it reproduced that run — parity at 133M
and 154M steps — and came nowhere near the run being reconstructed here. Matching the
reference is evidently enough to match it and not enough to beat it.

Two components are new against the target rather than fitted to it. `enemy_field_death` and
`enemy_field_damage` were both zero there, so a ship killed by a field paid its opponents
nothing at all; the rule requires them. They also widen the critic from 14 heads to 16,
which dilutes every other component's share of the per-component value loss by 14/16.

The wrapper divides component rewards by total ship count for team-size normalization.
A lambda aggregation matrix then maps local event signals to training targets:

- local components use diagonal/self-only credit;
- global outcome components aggregate across live teammates;
- selected enemy-perspective components use negative enemy coefficients to recover
  zero-sum outcome structure.

Each row of that matrix is normalized to a mean over the contributors it actually has —
one ship for a local component, the live teammates for a global one — and the component
weight is applied afterwards. The order matters: normalizing a row that already carries
its weight divides the weight back out, which is how the reference run's `ally_win`
weight of 1.5 came to train identically to 0.25 per ally — an effective total of 1.0 across
a four-ship team, whatever the configured number was. Because advantages are already
per-component unit-RMS before aggregation, the weight is then a pure importance term,
and only the ratios between weights affect training — the aggregate advantage is
divided by its own RMS, so scaling every weight together is a no-op.

Components are grouped into four tiers — outcome, kill/death, damage, shaping — and each
tier carries a schedule scale applied on top of the per-component weights. The tiers are a
credit-assignment ladder, and the per-component gammas and lambdas already follow the same
partition: an outcome is discounted over a whole episode, a kill over an engagement, damage
over an exchange, geometry over the next moment.

Three of the four scales hold flat, and that is the settled answer for them: the realised
tier shares already drift the way a curriculum would move them — measured on the reference run,
the outcome tier's share of the policy gradient rises about 1.29x over a run while the
kill/death tier falls to 0.73x — so scheduling them would fight a trend rather than create
one.

Shaping is the exception, and it is also the last remaining config difference between this
profile and the run it reconstructs — that run carried this taper, so this one does too. It
has to be pushed down rather than left alone: shaping's realised share *grows* about 1.58x
over a run, and `facing` and `closing_speed` are not
[potential-based](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf),
so they bias the optimum for as long as they are on, and they oppose the objective
directly: `closing_speed` against `field_damage_taken` measures a mean gradient cosine of
−0.446, negative in 99.9% of samples. They exist to stop early passive collapse, which is
finished long before the budget is. `shaping_scale` therefore decays from 100M steps to a
floor of 0.05 at 400M. The floor is not zero, so the components stay measurable to the end:
their gradient share and explained variance remain readable, which is how the next run
learns whether shaping was still buying anything.

One caveat on that inheritance. The reconstructed run stopped at 127M, so it ran barely 27M
steps into the taper and ended near 0.76. Everything the taper does past that point is
untested by the evidence this profile is built on.

Note that `kill_shot` is not winner-take-all: when several ships damage a target on its
fatal step, each earns credit proportional to that step's damage. `kill_assist` remains
proportional to cumulative episode damage even when a field delivers the final blow;
that preserves partial credit for attacks that force a dangerous navigation choice.

`kill_ally_shot` and `kill_ally_assist` mirror that pair for friendly fire, on the same two
horizons and with the same attribution. They are components in their own right rather than
negative terms inside the enemy pair. Folded in, one critic head had to predict the sum of
a positive enemy-kill signal and a negative friendly-kill one, the friendly half could not
be weighted separately, and it was invisible to every per-component diagnostic — it read as
part of `kill_shot`'s gradient share. All four share one implementation, differing only in
which damage matrix they read and whether they credit enemy deaths or blame friendly ones.

The former solid-obstacle death, proximity, closing-speed, and time-to-impact components
have been removed: refractive interfaces are traversable and should not receive universal
wall-avoidance shaping. Applied interface and projectile health loss, plus their exclusive
death causes, are recorded separately so neither source can double-count overkill.
Interfaces also reduce projectile damage potential, but that
barrier loss is not credited to a ship; only damage that reaches a target enters combat
attribution. See [`config/defaults.py`](../src/boost_and_broadside/config/defaults.py) for
current component horizons and schedules, and the preserved run config for historical weights.

## Behavior-cloning profile

[`profiles/bc.py`](../src/boost_and_broadside/profiles/bc.py) pretrains against the
scripted controller before any policy gradient is taken. The controller supplies
supervised action targets on every environment and never takes a side, so
`policy_gradient_coef` and `league_fraction` are zero for the whole budget and no roster
entry plays a rollout. The critic and the next-state head train alongside the actor, so RL inherits a warm trunk
and a critic that has already seen the full reward decomposition.

It trains in the environment RL continues in (eight ships, the same decision rate,
spawn spread, logical batch, minibatching, and component discount horizons), so the
handoff in `bnb train --profile rl --pretrain-from <bc-checkpoint>` does not also change
the task. Five things differ, and only where the objective requires it: no policy
gradient, no league, no KL trust region (under supervision, moving away from the rollout
policy is the objective rather than a reason to stop), full-strength next-state
prediction, and its own budget. That list is enforced by a test rather than by
convention, because the profile is written independently and does not inherit from
`rl`.

Evaluation still runs during BC: the raw win rate against the scripted controller is
what decays the cloning weight to zero at `bc_winrate_target`, and the run rates on the
same live gauge RL continues on.

Once that weight reaches zero the entropy bonus goes with it. Entropy regularizes an
objective; it is not one. With no policy gradient and no cloning term it would be the only
gradient reaching the actor, and its optimum is the uniform distribution, so a run that
had finished cloning would spend the rest of its budget undoing it. This was measured at a reduced launch width. A policy that had been cloned to a KL of 1.12
against the scripted controller, at 60% of maximum action entropy, was back to 99.8% of
maximum and a KL of 2.66 within 400 updates of the cutoff. Those are its untrained values.
A control arm that kept its cloning weight held steady over the same span. After
the cutoff the actor is held where cloning left it and the critic, next-state, and SIGReg
terms carry on through the shared trunk. RL is unaffected: its policy gradient is positive
throughout, so it keeps the scheduled entropy bonus.

## Fields

[`profiles/rl.py`](../src/boost_and_broadside/profiles/rl.py) trains in four cached static
fields, with the two local field reward heads active and the environment count reduced to
offset the extra attention tokens. The scripted controller ignores fields entirely: it aims
and manoeuvres as if the medium were uniform.

Fields are not a profile variant. `num_fields` is a sequence length -- it sets the token
count `N + M`, and no weight shape depends on it -- so a run at zero fields uses the same
network, and run 682 is still rated with the same evaluation stack.

It used to carry a mild stay-on-your-side steering bias, on the theory that behavior
cloning needed field-dependent targets to warm up the attention trunk. Measurement killed
it. Against a uniform-random agent the bias produced *more* interface crossings (2.24
against 1.60 per thousand ship-steps) and left ships in higher-index (slower) medium
more often (mean log index +0.159 against +0.108). Both were occupancy artifacts rather
than decisions, and since crossing an interface costs health that
`field_damage_taken` then penalises, behaviour cloning was imprinting a habit RL had to
unlearn.

Field representation does not depend on the scripted agent in any case. The auxiliary
next-state head predicts `local_log_index` directly, which cannot be done without locating
the ship relative to every field, and that pressure is always on and never decays with the
behavior-cloning weight.

Field maps are regenerated every rollout rather than drawn from a bank fixed at startup.
Generation is fully vectorised on device: it loops over fields rather than over maps or
retries, proposing `max_generation_attempts` placements for every map at once and taking
the first that fits, so a whole bank costs `num_fields` iterations of fixed-shape tensor
work with no host synchronisation. A 512-map bank of four fields refreshes in about 4 ms
against a rollout of tens of seconds.

The reason to bother: a fixed bank is a small distribution that a full run draws from
thousands of times per map, whereas a bank replaced each rollout supplies roughly one
distinct map per episode. Maps are laminar because candidates are rejected against
already-placed fields before acceptance, which matters because
`validate_field_layout` costs eight device-to-host syncs and raises, so it cannot run on
the hot path. Rows that exhaust their proposal budget keep their previous map rather than
ending the run, and `physics/field_map_generation_failures` reports how many, so a
too-tight radius/width/count combination shows up as a number instead of silently thinning
the distribution.

Per-update physics diagnostics report field/combat damage per live ship-step, source death
rates, the fraction of steps taking boundary damage, time in non-ambient media, and the
field share of total applied damage. These metrics are independent of reward weights.

A recommended curriculum for a dedicated field run is:

1. low/high index with no interface damage;
2. all four log-symmetric index levels with no damage;
3. add standard damage;
4. add severe damage;
5. enable nesting and larger parent/child index ratios.

The current profile samples all index and damage combinations and nested maps directly;
the staged curriculum is guidance, not a separate navigation-task implementation. Field
utility is learned from combat outcome, navigation, speed, handling, and health tradeoffs.

## Opponent curriculum

The primary scale has two environment groups: self-play, and a league whose width is
`league_fraction` (0.5 in
[`profiles/rl.py`](../src/boost_and_broadside/profiles/rl.py)). The league half is divided
into `league_slots` contiguous slots, and each slot draws its own opponent from the roster at
every rollout boundary.

Every opponent is an ordinary roster entry on one Elo scale:

- **scripted:** a stochastic hand-built controller, also the behavior-cloning target and a
  stable evaluation benchmark;
- **average:** a uniform running mean of eligible live-policy snapshots, joining the roster
  at the scripted performance cutoff;
- **checkpoint:** frozen historical snapshots, joining at each Elo milestone.

Self-play is the other half of the batch: the live weights viewed from the other team
perspective.

Sampling is proportional to `exp(-abs(opponent_elo - live_elo) / temperature)`, excluding
the random agent. That exclusion is load-bearing: an untrained policy sits at random's own
rating, so including it would make the early league mostly random play, which self-play
already provides at twice the actor tokens.

Semi-random rungs cover that early range instead. They are the same interior references
the ladder uses (below), so proximity sampling always has a well-matched candidate rather
than a choice between an opponent it beats every time and one it never beats.

There is no per-opponent schedule, because the ratings already encode the curriculum. At
step zero the scripted agent is the only entry a slot can draw, so training begins as an
even split of self-play and scripted games. The average policy joins at the BC cutoff,
checkpoints join as milestones are crossed, and the scripted agent stops being drawn once
the live rating leaves it behind. Ratings for the scripted and average entries are synced
from the continuous evaluator every update, since a stale rating misdirects every draw.

The [`EloRoster`](../src/boost_and_broadside/train/rl/roster.py) retains historical entries
rather than evicting the weakest; `league_size` only bounds the GPU-resident LRU policy
cache. Some entries a given run cannot host at all: a bullet-reading policy in a
bullet-free run, for instance, whose rollout observation shape was fixed when the wrapper
was built. Those are retired from sampling with their ratings intact, and the run
continues.

## Continuous rating and the frozen ladder

[`elo_eval.py`](../src/boost_and_broadside/train/rl/elo_eval.py) advances dedicated
evaluation environments alongside training. The evaluator has five logical slots:

1. live vs fixed anchor;
2. live vs floating checkpoint;
3. live vs scripted controller;
4. live vs running-average policy;
5. floating checkpoint vs fixed anchor.

Ratings live on the **live gauge**: an approximate scale, defined rather than measured,
that pins the uniform-random agent at 0 and the scripted controller at 1000. Slot 2
therefore updates the live policy rather than scripted: the player defining the scale must
not drift under the one being measured against it. Everything on this scale is logged
under `live_elo/` and is never the number a result quotes; see
[live Elo versus calibrated Elo](#live-elo-versus-calibrated-elo) below.

The anchor pool has two parts. **Stationary references** (the random agent, the
semi-random rungs, and the scripted controller) sit at its head and never age out,
because their strength is a fixed property and their ratings are fixed constants.
**Checkpoint anchors** follow: the newest `MAX_CHECKPOINT_ANCHORS` frozen ladder
snapshots, which do rotate as the live policy leaves them behind.

### The reference ladder

With only random and scripted as fixed references, the live policy saturates both for the
whole early climb: ~100% wins against one, ~100% losses against the other. Its rating is
therefore barely identified at exactly the point where opponent selection depends on
it. A ladder of
semi-random rungs fills that range. Each rung takes the scripted action with probability
`p` and a uniform one otherwise, and the gauge assigns it **1000·p**. A profile therefore
declares only which rungs exist (`TrainConfig.live_reference_probabilities`); the ratings
follow from the definition in
[`config/live_elo.py`](../src/boost_and_broadside/config/live_elo.py).

The linear placement is an approximation and a rough one at the weak end. Measured against
ladders fitted by `bnb semi-random` and regauged to the same two endpoints, it rates the
`p=0.2` rung about 106 Elo too high in the zero-field environment and 77 too high with
fields; from `p=0.6` upward it is within 40 points in both. Two consequences come with it:
proximity sampling meets an over-rated rung slightly earlier than it otherwise would, and
the milestone grid that decides when checkpoints freeze is read on these numbers.

What it buys is that nothing has to be re-fitted. A fitted gauge is a property of the
environment its rungs played in (tick rate, field count, ship config, fleet size), so it
silently expires whenever any of those move, and the two shipped environments had fitted
gauges that disagreed sharply (random at −364 with no fields, +132 with four). The defined
gauge is the same in both, which is precisely why a live rating from one environment is
not comparable with a live rating from the other.

Per-episode assignment is a multinomial draw over the information weights, so the pool
can be any size at no extra environment cost: the slot's envs simply redistribute, and
saturated references draw almost no games. Stationary references also cost no forward
pass: every semi-random rung is a Bernoulli blend of the same two action tensors, so the
whole stationary ladder is computed from one scripted call and one random call however
many rungs it holds.

Ties count as half a win. These live ratings steer opponent selection and training
decisions, but they remain a filtered online estimate on an approximate scale.

At configured rating milestones, the trainer writes unpruned ladder snapshots. After
training, [`elo_calibrate.py`](../src/boost_and_broadside/modes/elo_calibrate.py) replays
stationary players and refits historical match records to construct the more rigorous
reported curve. The [evaluation guide](evaluation.md#post-hoc-elo-calibration) explains why
the two rating series differ.

### Live Elo versus calibrated Elo

They are different estimators of different things and are never substituted for one
another:

| | live Elo | calibrated Elo |
|---|---|---|
| Produced by | the trainer, continuously | `bnb elo-calibrate`, after the run |
| Scale | defined: random 0, scripted 1000, rung 1000·p | fitted Bradley-Terry, shifted so scripted reads 1000 |
| Purpose | opponent sampling, progress, milestone placement | reported results |
| Metric keys | `live_elo/policy`, `live_elo/ladder/<label>` | `calibrated_elo/live`, `calibrated_elo/ckpt_<step>` |
| Stored in | `elo_history.jsonl`, checkpoint payloads | the `elo-calibration` artifact |

The naming is the enforcement. Nothing the trainer emits sits under a bare `elo/` prefix
that could be read as either, and the calibration chart files deliberately do not reuse
the trainer's key names even though they share its file shape. Published figures and prose
quote the calibrated series only.

`bnb semi-random` checks the approximation rather than supplying it: each scale in its
artifact carries `live_gauge_error`, the per-rung distance between the fitted ladder and
the rating training actually uses. No profile has to wait for it.

### Watching the live estimator

The live rating is a K-factor tracking filter, and a filter can settle somewhere the games
do not support. The `elo_diag/*` keys instrument it. They are read-only: nothing consumes
them, and opponent sampling, promotion, and `best_training` selection all behave exactly as
they did before the keys existed.

| Key | Reads |
|---|---|
| `elo_diag/implied_gauge_elo` | the rating that best explains the live policy's recent record against the *defined* references — random, the semi-random rungs, scripted — pooled over the last 8 updates |
| `elo_diag/drift_vs_gauge` | `live_elo` minus that. The headline alarm |
| `elo_diag/implied_scripted_elo`, `elo_diag/drift_vs_scripted` | the same against the scripted controller alone |
| `elo_diag/se_live_vs_scripted` | how precisely the floor-anchored offset is known, as effective resistance to the anchor in the pool's information graph |
| `elo_diag/movement_z` | this update's rating change divided by the standard error one update's games can support |
| `elo_diag/fiedler` | algebraic connectivity of that graph; zero means the pool has split and the ratings across the split are unidentified |
| `elo_diag/max_abs_rating` | trips first if a fit ever runs away to a separation corner |

The drift keys are what these exist for, but they are a **paired** instrument: compare two
runs at matched steps, and never read one run's number as its error.

The bias is large and shared. Replayed over run 719 — whose filter the post-hoc calibration
shows tracking the truth to −4 Elo on average — `drift_vs_gauge` still averages +53 and
climbs past +75. A rating fitted only against the floor saturates: once the policy beats every
defined reference almost always, the record can no longer say how far above them it sits, so
the estimate settles below the truth and the apparent drift grows with training. Runs under the
same physics share this almost exactly, which is what makes their difference informative. Run
727 against 719 reads +4.6 ± 33 Elo before 727's resume at 128M and +60.5 ± 30 after it.

Sign convention: positive drift means `live_elo` reads *above* what the record supports.

`elo_diag/movement_z` is likewise a distribution to watch, not a threshold to trip. A healthy
run sits near a median of 1.3 with a 95th percentile around 4.3 and occasional single updates
past 14.

Non-finite values are dropped rather than logged, so a saturated window shows up as a gap
in the series instead of a spike that ruins the axis.

### The accumulated ladder record

`match_matrix.json`, beside `roster.json` in the run directory, holds the running win/loss/tie
totals between players whose *weights* never change: the random agent, the semi-random rungs,
the scripted controller, and every checkpoint snapshot including the newest one whose rating
is still settling. The live and averaged policies are excluded, because they change strength
under the record and a count matrix cannot say when a game was played.

The games come from the evaluation battery's floating-vs-anchor slot, which the run already
plays every update to settle the floating rating and which previously discarded its results.
No evaluation budget moves to fill this file. Two keys track it, `elo_diag/ladder_matrix_games`
and `elo_diag/ladder_matrix_pairs`.

It is a sidecar, not a checkpoint payload key, so a checkpoint written before it existed stays
loadable. A resume that finds no file starts counting from zero: the accumulation buys
precision in the ladder's ratings and nothing depends on it being complete.

### The two-stage rating, running alongside

`two_stage/*` carries a candidate replacement for the K-factor filter. **It gates nothing.**
Swapping the estimator would change promotion timing, which changes the pool, which changes
every rating, so the filter keeps gating for a full comparison run and this one only observes.

Stage 1 refits every accumulated player from `match_matrix.json` each update, with the scripted
controller pinned at its gauge value — pinned rather than centred, so the scale holds still when
a promotion changes the pool. Stage 2 then solves for the rating that best explains the live
policy's record *this update* against those refit ratings, and reports a standard error with it.

Two measured facts shaped it, both from replaying 719's recorded counts against its post-hoc
calibrated curve. Fitting against the whole pool scores 15.4 RMS where fitting against the
gauge's defined references alone scores 65.9 — a floor-only fit collapses to one saturated edge
late in a run, and saturated edges carry almost no information. And a single update beats every
longer window tested (15.4 at one, 21.8 at two, 36.7 at eight), because the policy improves fast
enough that pooling costs more in lag than it buys in variance.

The structural difference does not show up in either number. A K-factor filter carries state and
settles where competing pulls cancel, so a shift that gets in stays in — run 727 held one for
270M steps after a resume. A per-update solve has no memory of its own previous value.

### Choosing which ladder games to play

The evaluation battery's floating-vs-anchor slot used to draw its opponent by `p(1−p)`, which
picks the most evenly matched opponent available. That answers "which game is most informative
about some rating", not "which game most sharpens this rung's offset from the floor" — and it
ranks the single most valuable game, the rung against the anchor, last, because it is saturated.

The replacement weights each candidate by `p(1−p)·b²`, where `b` is the potential drop across
that edge under unit current injected at the rung and drawn off at the anchor. Local information
times global position: play the matches that carry the most current between the rung and the
floor. In simulation, standard error of the rung's floor offset after equal budgets:

| rule | 10 batches | 40 | 160 |
|---|---:|---:|---:|
| `p(1−p)`, the old rule | 39.5 | 26.1 | 14.5 |
| uniform | 33.9 | 20.4 | 10.9 |
| **current-flow weighted** | **23.6** | **12.2** | **6.1** |

About three times fewer games for the same precision, and note the old rule is worse than
spreading games evenly — targeting information without asking what the information is *about*
is actively counterproductive.

A fixed share of the budget is spread evenly regardless of score. That is not a hedge against
the rule being wrong; it is insurance against the failure this class of rule causes, where
starving an edge splits the graph and leaves the ratings either side of the split unidentified.
The rule falls back to the old one whenever the accumulated graph cannot yet reach the anchor,
which is the normal state until a new rung has played its first games.

This is separate from `elo_history.jsonl` by design. That file records what cannot be
recovered — the live and averaged policies exist in one form for one update. Everything in the
match matrix can be replayed from disk afterwards; it is kept because the training run needs it
during the run.

## Checkpoints and reproducibility

Every payload family (full `step_<N>.pt` resumes, best-model snapshots, and the ladder
snapshots the league and calibrator reload) carries the same provenance block: the
observation schema, the weights, critic width, `team_pma_k`, step and rating, the training
paradigm, and the model, environment, and ship configs it was trained under. Full
checkpoints add optimizer, scaler and averaging state on top; ladder snapshots add nothing.
Saves are prepared asynchronously and written through a temporary file before rename.
[`checkpoint.py`](../src/boost_and_broadside/train/rl/checkpoint.py) defines the filenames.
(The included reference-run directory retains `recent_avg.pt` from an older naming
convention.)

Checkpoints are rebuilt from their own recorded configs rather than from whatever the
reader is running, by
[`policy_io.load_policy_bundle`](../src/boost_and_broadside/train/rl/policy_io.py), the
single loading path behind the league roster, the ladder evaluator, and every eval mode.
`build_policy` derives the feature pipelines from `ship_config` instead of accepting them,
so a bullet-reading config always gets its bullet encoder and there is no argument a caller
can omit to produce a policy whose inputs disagree with its weights.

Three compatibility rules follow from that:

- **Observation schema.** The refractive-field contract adds encoder inputs and a
  local-index auxiliary target; radius is shared by ship and field tokens and normalized by
  half the shorter world dimension, and the ship's local `grad(n)` widens the encoder's
  first projection. Payloads carry `observation_schema=refractive_fields_v3`. Earlier
  schemas have no faithful weight-only migration, so they are rejected and retraining is
  required.
- **Physics constants.** Eleven `ShipConfig` fields set the encoders' normalizers, so
  weights trained under different ones were fitted to differently-scaled inputs. A
  mismatch is refused by name; `--allow-config-drift` downgrades it to a warning, and the
  policy then reads the world through the constants it trained on. This is the only thing
  that flag now does.
- **Architecture.** Nothing needs to match. Nothing in the policy is sized by ship count,
  and each entry is rebuilt from its own config, so a league or rating field can hold
  checkpoints of different widths and depths. An entry whose architecture differs from the
  live run's simply runs eager rather than claiming a compiled graph nothing else reuses.
  The one exception is the training rollout, whose observation shape is fixed when the
  wrapper is built: a bullet-reading opponent in a bullet-free run is refused rather than
  left to play blind.

Payloads written before provenance existed still load; the loader falls back to the
caller's configs and warns, naming what it assumed.

Full checkpoints also carry the complete resolved launch: every configuration value, the
source that chose each value, and the execution and VRAM record. That makes the memory
decision part of a run's history rather than a property of whichever machine happened to
start it. It is recorded, not enforced: resuming onto a card that sizes the launch
differently is expected, and so is resuming with a deliberately changed hyperparameter.

Resuming is stricter than reloading a policy. `load_checkpoint` requires every field a full
payload writes and refuses one that lacks any of them, naming it. A resume restores the
complete training state (weights, optimizer, both scalers, the averaging accumulator, the
live rating and its running average, the milestone grid, and the evaluation windows) or it
does not happen. Only the resolved-config and launch blocks are optional, and both are
provenance rather than state. Policy-only files (`best_*.pt`, `ladder_step_*.pt`) are
consequently not resumable; `--pretrain-from` is the path that takes one.

Environment state is deliberately *not* in that list. A resume respawns every
environment and re-staggers truncation by writing a random `step_count`, exactly
as a cold start does, so the first episode in most environments is a fragment:
ships at spawn health, nothing in flight, and a forced truncation a few steps
later. Those fragments are withheld from the episode statistics until their
environment recycles — the same rule `elo_eval` applies to rated games, that an
episode counts only if it ran the full horizon from step 0. Without the gate the
first update of every run and every resume reported a reward mean, win rate and
lifespan that measured the seeding rather than the policy; on one measured resume
that was reward −8%, win rate −0.07 and lifespan −10% for a single update. The
gate costs nothing after the first episode length, and it never applies to the
per-step source metrics: a fragment's steps are real steps.

`--seed` seeds Python's `random` and Torch on the CPU and every CUDA device. Every draw
the trainer makes comes from Torch, including the permutation that orders PPO
minibatches, so one seed covers the run; a second RNG would be a second thing to
remember to seed. This is not bit-exact reproducibility: nothing forces deterministic
CUDA kernels, and reductions on the GPU are free to reassociate. It removes the variance
that is avoidable, which is what makes two runs of one configuration comparable.

W&B logging runs off the main training path. The reference run's sampled metric history,
configuration, summary, and run metadata are exported under
[`wandb_export/`](../checkpoints/resilient-resonance-682/wandb_export/) so the published
charts can be rebuilt without relying on a hosted dashboard.

### What `--vram` may and may not change

A profile fixes its logical batch, minibatch count, and fleet, and `--vram` only decides
how that fixed batch is laid out in memory. It may enable gradient checkpointing and set
the microbatch, which give the same objective within floating-point tolerance. It may
redistribute the batch across a different number of rollout shards, which keeps the nominal
token count and optimizer-step count but changes the env-stream count, the temporal
correlation within a shard, and how minibatches are composed. It may never resize the
batch, the minibatch count, or the fleet.
[Memory optimization](engineering/memory-optimization.md#resolving-a-launch-for-a-card---vram)
describes the policies, the preset rows, and what the probe measures.

## Gradient diagnostics

Several named losses land on one shared trunk, and `max_grad_norm` renormalizes whatever
arrives together: the term that sends more gradient takes a larger share of every clipped
step and the others lose it. The total gradient norm cannot show that split, and inferring
it after the fact is how a 3.4x actor/critic imbalance survived three full runs. Gradient
diagnostics measure it directly.

The instrument is off by default and changes nothing about what a run optimizes. It is
selected per launch, alongside `--device` rather than in the profile, and recorded in the
launch block of every checkpoint.

```
bnb train --profile rl --gradient-diagnostics top_level
bnb train --profile rl --gradient-diagnostics reward_policy --gradient-diagnostics-interval 25
bnb train --profile rl --gradient-diagnostics reward_full \
    --gradient-diagnostics-interval 100 --gradient-diagnostics-minibatches 2
```

| Level | Decomposes | Extra backward traversals per diagnosed micro-batch | Update phase | Peak VRAM |
|---|---|---:|---:|---:|
| `off` | nothing; no diagnostic code runs | 0 | 1.00x | 4051 MiB |
| `top_level` | the weighted PPO loss terms | one per active term | 1.02x | 4361 MiB |
| `reward_policy` | also the policy gradient, by reward component | plus one per weighted component | 1.09x | 4441 MiB |
| `reward_full` | also the critic gradient, by reward component | plus one per active component | 1.17x | 4529 MiB |

Cost measured by
[`benchmarks/gradient_diagnostics.py`](../benchmarks/gradient_diagnostics.py) on an RTX 4070
Laptop with one diagnosed minibatch every update, at `--num-envs 1952` -- a half-width shard
of the `rl` profile as it stood before it carried fields, and not a width that resolves
today. The multipliers are ratios between diagnostic levels at one width, which is what
makes them worth quoting; the absolute MiB are of that launch. The multiplier is on the update phase alone; a training step also collects a
rollout and runs Elo evaluation, so the share of total wall clock is smaller. Raising
`--gradient-diagnostics-interval` divides the overhead by the cadence.

`--gradient-diagnostics-interval` sets the cadence in PPO updates and
`--gradient-diagnostics-minibatches` how many complete optimizer minibatches each diagnostic
update measures. Measurement happens in the first epoch, so every reading describes a step
taken against a comparably fresh policy.

### What is measured

Gradients are accumulated over every micro-batch of an optimizer minibatch before any
statistic is taken, so what is reported describes a real optimizer step. The cosine is

    cos(sum_microbatches g_a, sum_microbatches g_b)

and not the mean of per-micro-batch cosines, which is a different number and is not the
direction the step follows.

- `grad_norm/<group>/<term>` — the length of that term's accumulated gradient.
- `grad_share/<group>/<term>` — its norm over the summed term norms: the share of the
  pre-clip gradient budget it is asking for, and so roughly the share of the clipped step it
  takes. This is the generalization of `train/actor_grad_share`.
- `grad_cos/<group>/<a>__<b>` — whether two terms pull the same way. Near +1 they reinforce,
  near −1 they are cancelling and both are being wasted, near 0 they are independent.
- `grad_diag/total_norm/<group>` — the norm of the combined gradient, and
  `grad_diag/agreement/<group>` that over the summed norms. Agreement near 1 means the terms
  point the same way; near 0 means the group is largely cancelling itself out.

Groups are `top_level`, `reward_policy`, and `reward_value`, each also reported over the
shared trunk under a `trunk_` prefix. The trunk scope is the informative one for cosines
between terms owned by different heads: the action head, the two value heads, and the
next-state head have disjoint parameters, which drags every whole-model pairing toward zero
whatever the terms are doing to the weights they share. The trunk is defined by module
reference (`YemongPolicy.trunk_modules`), not by matching parameter names.

`grad_diag/microbatches`, `grad_diag/terms`, `grad_diag/level`, and `grad_diag/seconds`
record how the measurement was taken.

### Reward decomposition

`reward_policy` splits the policy gradient across active reward components. The split is an
exact attribution rather than a model: the per-component advantages are aggregated through
the same lambda matrix the live gradient used, and PPO's clipping branch is taken from the
*aggregate* objective and reused for every component. Choosing a branch per component would
let each reward take whichever branch flatters it, and the parts would no longer sum to the
update being run — not subtly, but by orders of magnitude. Components sum back to the
aggregate policy gradient to floating-point tolerance, which
[`test_grad_diagnostics.py`](../tests/train/test_grad_diagnostics.py) asserts with clipping
active. Components scheduled to zero weight contribute no gradient and are not logged.

`reward_full` adds the same split for the critic, which is already a sum of independent
per-component squared errors. It is substantially more expensive than the other levels — one
extra backward traversal per component on top of the policy split, and it holds one full
gradient copy per term in memory simultaneously — and it emits O(K²) cosine series. Use it
on a cadence, not every update.

On a diagnostic update the actor/critic split (`train/grad_norm_actor`,
`train/grad_norm_critic`, `train/actor_grad_share`) is read off the accumulated top-level
terms — the same quantity over a whole minibatch instead of one micro-batch of it — and the
cheap histogram-cadence probe stands down rather than measuring it twice.

## Engineering validation

Training behavior is covered across:

- [`test_ppo.py`](../tests/train/test_ppo.py) for loss, masking, rollout, and schedule logic;
- [`test_grad_diagnostics.py`](../tests/train/test_grad_diagnostics.py) for gradient
  decomposition: that the components sum to the aggregate gradient with clipping active,
  that micro-batch accumulation matches the unsplit minibatch, and that measuring leaves the
  applied gradient bit-identical;
- [`test_buffer.py`](../tests/train/test_buffer.py) for recurrent storage, GAE, precision,
  sharding, and scalers;
- [`test_roster.py`](../tests/train/test_roster.py) and
  [`test_elo_eval.py`](../tests/train/test_elo_eval.py) for opponent/rating behavior;
- [`test_checkpoint.py`](../tests/train/test_checkpoint.py) for save/resume state and
  mixed-architecture league play;
- [`test_policy_io.py`](../tests/train/test_policy_io.py) for construction, provenance, and
  the physics-drift check;
- [`test_match.py`](../tests/modes/test_match.py) for perspective, the bullet axis, and side
  assignment in the shared match loop;
- [`test_bradley_terry.py`](../tests/train/test_bradley_terry.py) for calibrated fitting and
  uncertainty.
