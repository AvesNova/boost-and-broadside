# Training-efficiency plan

Working plan for the sample-efficiency and wall-clock work started on
`fix/league-unification-scaler-floors`. Internal: it records reasoning and
measurements, not reader-facing documentation. Reader-facing material lives in
[training.md](../training.md).

Written so a fresh session can pick this up without re-deriving anything.
Current as of `e467120`.

**Read [What went wrong twice](#what-went-wrong-twice) before changing anything
other code consumes.** Two of the three suspects in the first regression turned
out to be measurement bugs introduced here, not training problems.

---

## Where the time goes

Measured on the 8 GB dev card ([memory-optimization.md](../engineering/memory-optimization.md)),
and consistent with a first-principles token count:

| phase | share |
|---|---:|
| rollout (env + opponent forwards + Elo battery) | **15%** |
| PPO update (epochs × minibatches × micro-batches) | **85%** |

Consequences that drive everything below:

- Wall-clock work belongs in the update phase. The Elo battery, the scripted
  agent and the stream overlap all live inside the 15%.
- Sample efficiency and wall clock are the same lever: anything that extracts
  more learning per token also shortens the run at fixed final Elo.

---

## Done

Oldest first. Each entry records *why*, because the reasoning is
harder to recover than the diff.

### `8f8e612`: advantage scaler floor

`AdvantageScaler.min_rms` defaulted to 0.1 with no config path, against true
per-component advantage RMS of 0.0075 (win) to 0.27 (damage). The floor bound
permanently on **six of eleven** components, so `normalize()` divided them by the
guard rather than their own statistics, exactly what the scaler exists to
prevent. Win components were taking ~22% of the policy gradient against the ~62%
their weights ask for.

Dropped to a 1e-4 epsilon, plumbed as `TrainConfig.advantage_min_rms`. No
loss-magnitude risk: the aggregated advantage is renormalized to unit RMS *after*
lambda aggregation, so this changes the mix, not the scale. Verified against
`--smoke`: total loss unchanged at 8.2.

**`return_min_span` was deliberately left at 1.0** at the time. Lowering it was
not the same kind of fix: `ReturnScaler` divides the whole return distribution by
a robust p5–p95 half-span, so sparse components with heavy tails produce very
large normalized targets and the squared value loss follows. Measured **~11× on
`loss/value` at production spans and ~400× in `--smoke`**, which against
`max_grad_norm=1.0` (grad norm 0.65) would make clipping bind every step and
silently cut the effective learning rate. **Resolved by `62da40d`**: the
categorical critic's loss is bounded per sample and scale-free in the target, so
the floor is back to a 1e-3 epsilon.

### `c6610b2`: unified opponent league

Three opponent env groups were each sized at their peak scheduled fraction and
then only ever switched fully on or off, so `scripted_fraction` stepping 0.5 →
0.3 at 50M **never actually stepped down**. The reference run trained against
scripted in 50% of envs for its whole length with self-play at 10% rather than
30%, and by then the scripted matchup sat at a 100% win rate, so half the batch
produced no outcome signal.

Replaced with self-play plus one league whose width is a single `league_fraction`,
split into `league_slots` slots that each draw by Elo proximity every rollout.
The scripted agent became an ordinary roster entry.

Two more bugs fixed here: the `avg` entry's rating was set once at creation and
**never updated**, and a bullet-reading league entry in a bullet-free run *raised
mid-rollout* instead of being retired.

### `11a00ea`: GAE truncation and conv-buffer boundaries

**C1:** GAE keyed its non-terminal mask on physics `dones`, but the wrapper
auto-resets before returning the observation, so a *truncated* episode
bootstrapped off the freshly spawned one. At γ=0.999 that leaked the length of
the trace. Now keyed on `terminated`.

**C2:** `reset_hidden_for_envs` zeroes the whole packed hidden (RG-LRU state
*and* conv buffer), but `forward_sequence` applied `done_mask` only inside the
RG-LRU, so re-evaluation read the previous episode's inputs for `CONV_KERNEL-1`
steps after every boundary and the PPO ratio was wrong there. The boundary mask
depends on the output step **and** the tap's lag, so no input-side mask expresses
it; the conv is now an explicit sum of K shifted taps with a per-(step, lag)
validity mask. **Verified by reverting the fix with the new test in place.**

**M2:** removed the blocking `.cpu()` once per optimizer step.

### `c9e2c41`: generalized measurement ladder

`MAX_ANCHORS = 2` was hardwired into a binary `torch.where` and a single-threshold
draw, with a comment warning that raising it fails *silently*. Now a multinomial
draw over information weights and a gather.

Anchors split into **stationary references** (permanent prefix) and **checkpoint
anchors** (rotating). Stationary references cost no forward pass: every
semi-random rung is a Bernoulli blend of the same two action tensors, so a ladder
of any length is one scripted call plus one random call.

### `33dbef4`: action repeat (20 Hz; **superseded**, now 30 Hz)

**Superseded by `e467120`**: the rate is 30 Hz and the evaluation path is fixed.
The reasoning below is the 60 Hz baseline and still holds; only the chosen rate
and the discount exponent changed.

Physics ran at 1/60 and the policy chose a new action every tick, far finer than
the plant responds to. `EnvConfig.action_repeat=3` holds each action three
physics ticks. Physics, collisions and projectile integration are untouched.

| timescale | seconds | decisions @ 20 Hz |
|---|---:|---:|
| firing cooldown | 0.10 | 2 |
| bullet flight to ~200 px | 0.40 | 8 |
| full 360° turn | 1.3–2.3 | 26–46 |
| mean episode | ~4.7 | ~93 |
| `num_steps=128` rollout | 6.4 | 128 |

At 60 Hz five of six shoot decisions were no-ops against the cooldown, and a
128-step rollout spanned 2.1 s against a ~4.7 s episode, so **BPTT never covered a
whole episode**. It now does, and a token buys three times the game time.

- Rewards **summed** across held ticks (scale-preserving over fixed game time).
- Every γ and λ re-derived as `value ** (60/20)`; horizons in seconds are now
  written down in `src/boost_and_broadside/config/defaults.py`.
- `max_episode_steps` needed **no** change, because it counts physics ticks.
- Spawn health/power/cooldown randomised, balanced in expectation across teams.
- `total_timesteps` → 333M: same game time as the 1e9-step reference run.

### `6f8e61e`: reference ladder and scripted-anchored gauge

With only random and scripted as fixed references the live policy saturates both
for the whole early climb, so its rating is barely identified exactly when
opponent sampling and the milestone grid depend on it. Nine semi-random rungs are
now fixed roster entries and anchor-pool members.

**Ratings are environment-specific, and the recompute proved how much:**

Current values (30 Hz, refit in `e467120`), against the original 60 Hz fit:

| | old 60 Hz | `rl` @30 Hz | `rl_fields` @30 Hz |
|---|---:|---:|---:|
| random | −550.7 | **−363.9** | **+132.3** |
| 0.5 rung | 193.9 | 270.7 | 550.4 |

Fields move random ~500 Elo (they compress the skill scale); tick rate and map
generation each move it tens to low hundreds. Each profile carries its own
ladder. Re-run `bnb semi-random --profile <name>` whenever tick rate, field
count, ship config, fleet size, map generation **or the scripted agent** moves.

Gauge is now **scripted pinned at 1000**, matching post-hoc calibration, so
normalized Elo collapses to the rating itself and the milestone grid is absolute.
Slot 2 updates the live policy rather than scripted.

### `a32e178`: scripted controller ignores fields

Its stay-on-your-side steering measured **net-negative**: 2.24 interface
crossings per thousand ship-steps against uniform-random's 1.60, and mean log
index +0.159 against +0.108. Both occupancy artifacts (a ship at index *n* takes
*n*× as long to cross the same ground), not decisions, and the bias capped at 35% of
turn intent and only acted within ~120 px of an interface, so pursuit overwhelmed
it. Since crossing costs health that `field_damage_taken` penalises, BC was
imprinting a habit RL had to unlearn.

Two replacements were tried and both failed: `grad(n)` is zero throughout a flat
core so it never reaches a ship in slow medium; the index-difference version
measured worse still. **The metric is also confounded**: time-weighted index is
mechanically biased toward slow media, so a *path-weighted* measure is needed
before attempting this again.

Field representation never depended on it: the auxiliary next-state head predicts
`local_log_index` directly (prediction dim 10 of 10), which cannot be done
without locating the ship against every field, and unlike the BC weight that
pressure never decays.

### `38a5c22`: win-rate KL gate, derived aux names, tournament defaults

- `high_elo_threshold` → `high_winrate_threshold` at 0.8. Both gates now read the
  raw scripted win rate, so the trust region needs no re-deriving when the gauge
  moves. 0.8 reproduces the original firing point.
- `_NS_FEAT_NAMES` hand-listed 9 names against a computed width of 10, silently
  dropping `local_log_index`. Names now come from `coordinator.get_feature_names()`.
- The old semi-random dispatcher defaulted to a hardcoded run name; `--profile rl` now.

### `55eeaed`: vectorized field maps, refreshed per rollout

Maps came from a bank of 512 built once by a CPU loop with an `.item()` per draw;
a full run drew each thousands of times. Generation now loops over **fields**,
not maps or retries: every field proposes `max_generation_attempts` placements
for all maps at once and takes the first that fits. No data-dependent control
flow, so no host sync. **4 ms per refresh**, called at the top of every rollout,
roughly one distinct map per episode.

`validate_field_layout` costs **eight** device-to-host syncs and raises, so it
cannot run on the hot path. Startup still validates (an unplaceable config is an
error); refreshes keep the previous row for any map that exhausts its budget and
report `physics/field_map_generation_failures`.

Verified: 25 consecutive refreshes × 512 maps all pass the strict validator;
three refreshes yield 1536 distinct layouts.

---

### `e467120`: evaluation rate, anchor labels, reward mix

Four fixes from investigating a regression that was mostly instrumentation. Run
708 (20 Hz) looked far worse than 707/705 on both charts.

**The eval-rate bug, the serious one.** `action_repeat` was honoured only in
`YemongEnvWrapper.step`. The Elo battery and every evaluation mode step
`TensorEnv` directly, so **708 trained at 20 Hz and was evaluated at 60 Hz**, and
its reference ladders were rated at 60 Hz regardless of profile. A policy holding
an action for N ticks but given one turns a fraction of its intended amount per
decision, mistimes every lead, and advances its recurrent state N times too fast
for the game clock. It still plays, just far worse, and nothing says why.
`TensorEnv.step` is now the decision-level call; the wrapper opts out via `tick`.

**The anchor-label bug.** `eval/win_rate_vs_random`, then named
`elo/training_vs_random`, classified slot-0 games by
whether the anchor carried weights, but *every* stationary reference is
policy-free: random, all nine rungs, and scripted. The whole ladder landed in
the random bucket, so the chart reported the win rate against an
information-weighted mix of near-level opponents (~0.85) while
`matches/random/win_rate` was **1.000**. The apparent intransitivity, beating
random 85% while beating scripted 70% with scripted beating random 99%, was
entirely this.

**Win weights 4.0 → 1.5.** With `AdvantageScaler` normalizing every component to
unit RMS, the effective policy-gradient share is just `weight/sum(weights)`. 4.0
put the win pair at **56%**, against ~21% before the floor was removed. Measured
consequence: combat damage per live ship-step halved (0.154 vs 0.285), combat
deaths fell a third, episodes ran 39% longer, and 13% drew. 1.5 lands at 32%.

**20 Hz → 30 Hz.** Measured on the *fixed* scripted controller at equal game time,
so the policy cannot adapt and any change is the environment alone:

| | 60 Hz | 30 Hz | 20 Hz | 15 Hz |
|---|---:|---:|---:|---:|
| combat damage / live ship-step | 0.2965 | 0.2814 | 0.2656 | 0.2475 |

20 Hz gave up 10% for a third of the tokens; 30 Hz gives up 5% for half.

**Resource metrics added**: `physics/mean_power`, `mean_speed`,
`out_of_power_fraction`. The learned policy runs at **34.8/100 mean power with
7.8% of live ship-steps at zero**, against scripted's 56.2 and 1.2%. It is *not*
slower than random (97.0 vs 46.4 px/s); random is the slow one.

Run 708 is **uninterpretable** as evidence about the tick rate or the reward mix:
two measurement bugs were active throughout it.

### Block C: categorical critic (**tried and reverted**)

Shipped as `ec3463a` (drop TeamPMA), `62da40d` (categorical critic),
`bd0c215` (merge the win pair), then `a5cb5dd` (rebalance the value
coefficient). Three training runs. **All of it reverted** except the EV
logging fix and the actor/critic gradient instrument.

**Runs, all against `iconic-shadow-709` (MSE + TeamPMA + split win):**

| run | config | Elo @80–100M |
|---|---|---:|
| `iconic-shadow-709` | MSE, TeamPMA, split win | **1221** |
| `ruby-puddle-710` | CE, `return_min_span=1e-3` | (representation bug) |
| `charmed-moon-711` | CE, span floor restored | 1156 |
| `youthful-spaceship-712` | + `value_function_coef=0.29` | 1103 |

**The premise was wrong.** The block was justified by a 281× ratio in
`critic/value_loss` between `ally_win` and the dense components, read as an
undertrained win baseline. That ratio is what target compression does to a
*squared* loss even with a perfect critic, since c=23 compression gives c²=526,
and it carries no information about fit quality. The metric that does is
scale-free, and `ally_win`'s explained variance in 709 was **0.88**. There
was never an underfit baseline to fix.

**What the critic change actually did.** In 711 the critic improved by
+0.043 mean EV, concentrated entirely in the two components that had
headroom (`field_death` 0.49→0.80, `kill_assist` 0.46→0.68) and flat on the
other ten, which were already at 0.85–0.98. The policy was 65 Elo and 9
win-rate points *worse*. **Improving a component's baseline does not imply a
better policy**, and that is the finding worth keeping.

**Three controls were broken, in sequence:**

1. **Representation.** The ±5 bin support was sized from a z-distribution
   measured on a checkpoint whose scaler still had `min_span=1.0`, then the
   same commit dropped the floor, changing the distribution it was sized
   against by up to 73×. For a sparse component the p5–p95 span measures the
   noise floor of nothing happening (`field_death`: central 90% spans 0.0009,
   events reach −0.12), so events landed at |z| up to 2000 and all encoded to
   the same end bin. Measured round-trip ceilings: 0.085 / 0.151 / 0.415
   against observed EV of 0.024 / 0.032 / 0.109. That is run 710.
2. **Effective learning rate.** CE sends **3.44× the trunk gradient** MSE does
   at convergence (measured offline; its loss *value* is 24× larger, but that
   is mostly the CE entropy floor, and only 1.38× reaches the head's own
   parameters). Solving the observed gradient norms for an actor/critic split
   gives actor 0.68 / critic 0.73 under MSE (total 1.00 vs observed 0.98) and
   actor 0.68 / critic 2.51 under CE (total 2.60 vs observed 2.58). Against
   `max_grad_norm=1.0` the actor's share fell 68% → 26%. Run 712 corrected it
   (gradient norm and share both landed on target) and Elo got *worse*. The
   mechanism was real and was not the cause.
3. **TeamPMA.** `ec3463a` removed the attention-pooled win head on the
   argument that "the trunk already runs spatial attention, so a ship's
   embedding is already team-aggregated." That was never measured, and it is
   not equivalent: attention over all tokens is permutation-equivariant across
   the whole set, while PMA computes an explicit per-team aggregate masked to
   that team's living ships, and P(win) is a set function over a team. **Every
   run that lost to 709 was missing it.** Perfectly confounded with the critic
   change across all three runs, and never isolated.

**Kept:** the EV logging fix (`93a4638`, where explained variance was gated on the
final epoch *index*, so the whole family was silently dropped whenever
`target_kl` broke the loop early, losing ~4% of points biased toward the
updates where the policy moved furthest), and `train/actor_grad_share`, which
makes control 2 visible on update 1 instead of three runs later.

**Settled by `sage-silence-713`** (MSE + no TeamPMA, single variable against
709). Both changes cost, and they are roughly additive:

| run | config | Elo @80–100M | vs 709 |
|---|---|---:|---:|
| `iconic-shadow-709` | MSE + TeamPMA | 1221 | n/a |
| `sage-silence-713` | MSE, **no TeamPMA** | 1185 | **−35** |
| `charmed-moon-711` | CE, no TeamPMA | 1156 | −65 |
| `youthful-spaceship-712` | CE, no TeamPMA, coef 0.29 | 1103 | −117 |

Two conclusions, both from matched comparisons rather than inference:

1. **TeamPMA is load-bearing**, worth ~35 Elo on its own, about a third of
   the gap. It is not the redundant re-derivation the removal argued it was.
   Restored, and it stays.
2. **The categorical critic was independently negative.** 713 and 711 differ
   only in the critic (plus the win merge), both without TeamPMA, and CE is
   29 Elo worse against that matched baseline. So the Block C verdict does
   *not* need re-reading: the confound was real, and so was the effect it was
   hiding. Both changes were bad, which is why 711 looked twice as bad as
   either.

The methodological point stands regardless: this cost three runs to learn and
would have cost one had TeamPMA been tested separately, as its own
"expected to do nothing" claim implied it should be.

## What went wrong

Two distinct failure modes. The first is the earlier pair of regressions; the
second is Block C, which failed a different way and more expensively.

### Shape 1: a representation changed and its consumers were not audited

- Making scripted and the rungs *stationary references* (`policy=None`) was
  correct. But `anchor_is_random` had encoded "stationary" as "policy-free", and
  that encoding silently became wrong.
- Putting `action_repeat` in the wrapper was correct for reward accumulation. But
  five other call sites step `TensorEnv` directly and silently kept the old
  meaning of "a step".

Both stayed invisible because the system kept running and produced plausible
numbers. Generalisable guards:

1. When you change what a type *means*, grep every consumer of that type, not
   only the ones you are editing.
2. Prefer making the default correct (`TensorEnv.step` honours the repeat; the
   one caller needing otherwise opts out loudly) over making every caller
   remember.
3. Derive labels and counts from the source of truth rather than restating them;
   the same class of bug produced `_NS_FEAT_NAMES` (9 names, 10 dimensions).
4. **A fixed-policy control separates environment from learning.** The
   scripted-vs-scripted tick sweep answered "did the env get harder or did
   learning break?" in one run. Reach for it first.

### Shape 2: an argument was recorded as if it were a measurement

Block C cost three runs and reverted. Every step of it was defensible in
isolation and the aggregate was not, because unmeasured claims kept entering
the record as settled facts and later work was built on them.

1. **A ratio is not a finding.** The 281× `critic/value_loss` gap was read as
   an undertrained baseline. It is what a squared loss does to targets
   compressed 23×. One look at `critic/explained_variance` (0.88) would have
   ended the block before it started. *Check whether the metric you are
   reasoning from is scale-free before drawing a conclusion about fit.*
2. **Don't measure a distribution and then change it in the same commit.** The
   bin support was sized against a scaler state that the same commit altered by
   up to 73×.
3. **Bundling a "small, sign-uncertain" change into a block destroys the
   baseline.** TeamPMA removal was never isolated, so all three runs compared
   two variables against 709 and none of them can attribute anything. If a
   change is genuinely expected to do nothing, that is an argument for testing
   it separately and cheaply, not for hiding it inside something else.
4. **Anything sharing `max_grad_norm` shares an effective learning rate.** A
   3.4× shift in the critic's trunk gradient went unnoticed for three runs
   because only the total norm was logged.
5. **Offline round-trips are cheap and would have caught two of the three.**
   The representation ceiling (encode → decode real returns, no learning) and
   the head-only loss comparison both run in minutes. Run them before, not
   after.
6. **A better critic is not the goal.** EV rose +0.31 on `field_death` and
   +0.22 on `kill_assist` and the policy got worse. Auxiliary metrics justify a
   change only when the objective moves with them.

## Remaining blocks

**Block C was tried and reverted**; see the post-mortem in Done. The short
version: its premise (an undertrained win baseline, inferred from a 281×
value-loss ratio) was arithmetic rather than a finding, and `ally_win`'s
explained variance in 709 was 0.88. Three runs, none beat the baseline.

**A and B remain, and are independent of each other.** D no longer has a
dependency on C. The baseline is now understood: `sage-silence-713` isolated
the TeamPMA question and the head is load-bearing (~35 Elo), so it is
restored and the tree is 709's configuration plus two logging additions.

Roughly one run each. Budget is tight, so each block is a coherent theme that
can be judged as one thing.

### Block A: action distribution

**Joint 2×3×7 = 42-way action head.** Product distributions are a strict subset:
a factored policy *cannot* represent "boost-straight or coast-turn, never
boost-turn", and real dogfighting policies are full of that. Cost is 30 extra
logits.

- `flat_action_sampling=True` **already exists** on `StochasticScriptedAgent` and
  produces the exact joint as the outer product of the three independent
  marginals, so the BC target is exact and already implemented.
- `RolloutBuffer.expert_probs` widens 12 → 42; `POWER_SLICE`/`TURN_SLICE`/
  `SHOOT_SLICE` become a joint layout.

**Per-factor entropy control.** Measured collapse by 10% of the run: power 14% of
max, turn 19%, **shoot 9%**. One coefficient cannot hold three distributions with
maxima `ln 3`, `ln 7`, `ln 2` at sensible floors.

With a joint head, compute marginals per state by reshaping 42 → (3,7,2) and
summing the other two axes, then take entropy of each. Two notes:

- A single `H(joint)` bonus is **insufficient**: max is `ln 42 = 3.74`, and the
  optimizer can satisfy any target on it while shoot stays pinned.
- Marginals are **coupled** under a joint head (moving shoot mass perturbs turn
  and power), so three dual controllers can oscillate. **Start with fixed
  per-factor coefficients**, not duals. `H(marginal_i) ≤ H(joint) ≤ Σ H(marginal_i)`,
  so per-factor floors imply a loose joint floor.
- Regularizing marginals does *not* push toward independence (uniform marginals
  are compatible with strong coupling), so the correlation modelling survives.

**M3** folds in here: the per-head softmaxes stop being throwaway diagnostics.

**Watch:** per-factor marginal entropies at their floors; `policy/kl` rising off
~0.005 (it was suppressed *by* the collapse, since a near-deterministic categorical
has small KL under logit perturbation); `policy/clip_fraction` near 0.08.

This block is what makes the win-rate KL gate meaningful: neither KL threshold
has ever bound.

### Block B: token allocation

**`shared_pass`.** In `ego_pass` only team-0 ships carry actor gradient. That is
*correct* in league envs (team 1 is the opponent) but wastes half the ships in
self-play envs. Under `shared_pass` a self-play env yields `N` actor tokens
against `N/2`:

| | league envs | self-play envs | actor tokens @ 75/25 |
|---|---|---|---:|
| `ego_pass` | N/2 | N/2 | **0.5N** |
| `shared_pass` | N/2 | N | **0.875N** |

**1.75×**, plus the rollout forward halves (2B → B). `_opp_team_flag` plumbing
already exists and is tested.

**`league_fraction` → 0.25.** Coupled to `shared_pass`: only under it does a
self-play env yield 2× the actor tokens of a league env. Under `ego_pass` they
are equal and the tilt has no efficiency basis.

**Caveat:** both teams in a self-play env come from *one* episode and are
anticorrelated on outcome, so the win-component gain is well under 1.75×. If Elo
regresses, `shared_pass` is the suspect (it forfeits the canonical team-0 prior).

**Not doing:** hiding enemy pending actions. The leak is 16.7 ms against a ~0.4 s
bullet flight, symmetric, and it would forfeit `shared_pass`. Block D's
next-action head gets the opponent-modeling benefit without paying that.

### Block D: representation

The one-step next-state prediction is a **deterministic function of the
observation** (up to `bullet_spread`), not merely short-horizon: `previous_action`
holds the action *about to be applied* and is visible for every ship, so via
spatial attention each ship sees every pending action. It is nearly trivially
solvable, which is why it is a weak representation signal.

**Iterated action-conditioned k-step latent prediction.** One learned step
function applied k times, *not* k independent heads, which each learn a shortcut
and never form a coherent dynamics model. Condition on the ship's own actions
from the buffer; what remains unpredictable is other agents' behaviour, which is
the signal wanted.

- All ship tokens, both teams. Fields excluded (static within an episode, so
  trivially predictable).
- Target `z[t+k].detach()`, masked by `alive` and by episode boundaries; reuse
  the `max_pool1d` window-validity trick from `_triangle_conv_loss`.
- Cosine loss on L2-normalized latents, substantially more collapse-resistant
  than raw MSE.
- **SIGReg on** at a small coefficient as the collapse guard. Already
  implemented, sitting at `sigreg_coef=0`.
- Keep the 1-step explicit state prediction: grounds the latent in physics,
  nearly free.

**Cost: ~zero extra trunk compute.** `z_0 … z_{T-1}` already exists from the
single sequence forward; this adds one small MLP applied k times and a detached
slice. Best compute-to-signal ratio on the list.

**Enemy next-action prediction head.** Predict the *simultaneous* action `a_t`,
genuinely unknown, with `buffer.actions[t]` as the label. Opponent modeling with
**no observation change**, so it survives `shared_pass`. The degeneracy worry,
that ships learn to act predictably to satisfy the head, is unfounded: a sampled
discrete action index is a constant w.r.t. parameters, so no gradient path rewards
being predictable.

**Not doing: dreaming.** Dreamer's premise is that the env is expensive. Here the
env is 15% of wall clock and the update is 85%, so a learned world model would be
*slower* than the real simulator.

---

## Deferred, with reasons

| | why |
|---|---|
| **Team-size / token-split randomization** | "trained only on 4v4" is the stronger claim. Revisit behind an env-fraction gate; note `comp_rewards /= _n_ships` uses the static config count and would need to be per-env. The lambda aggregation is already alive-aware and needs no change. |
| **Variable per-agent step size** | Batching kills the payoff: all ships share one forward, so a subset cannot be skipped. Zero throughput gain, and a ragged buffer breaks the dense `(T,B,N)` layout GAE, the lambda einsum and the scan all assume. |
| **Hiding enemy pending actions** | 33 ms at `action_repeat=2`, symmetric, and it forfeits `shared_pass`. |
| **Scripted field interaction** | Two attempts measured worse than nothing. Needs a *path-weighted* index metric before retrying; the time-weighted one is confounded. |
| **M1 (Hillis–Steele scan)** | Highest ceiling, most work. **Profile before writing anything**: a `torch.profiler` trace of one update settles whether the bandwidth estimate is right. O(T log T) with ~40 full passes per temporal sublayer; a Blelloch or chunked scan is O(T). |
| **M5** | Check `TORCH_LOGS=graph_breaks` first; Inductor probably already fuses it. |
| **M6** | `torch._foreach_nan_to_num_` does not exist in this build. |
| **Bullet axis cost** | The biggest single lever measured so far, and untested. Bullets are **~26% of forward FLOPs** (23.5 GFLOP `kv_bullet` + 3.4 encoder of 103.7) and **~45% of persistent VRAM** (612 MiB of 1363, of which `bullet_pos` alone is 204 MiB in fp32), against `n_bullet_cross_per_block=1`. Five times the cost of Block C. Three sub-levers: fp32 `bullet_pos` (a shooter-relative encoding would make bf16 viable, −102 MiB); the field channels on bullets (153 MiB); and 80 fixed slots/env stored regardless of occupancy, where compaction is a real project, it breaks the dense `(T,B,·)` layout. **Ablate `n_bullet_cross_per_block=0` first** to find out whether any of it earns its keep. |
| **Ladder rung spacing** | `elo_milestone_gap=200` and the grid seeds from the live gauge's zero, so the first checkpoint snapshot fires at 200 Elo, deep inside rung territory, where the 9 semi-random rungs cover 200–950 densely. Intended shape is rungs below the scripted anchor and checkpoints above it: set the gap to 100 and seed the grid from 1000. Note `min_games_to_freeze=1000` will defer milestones during fast climbs, so the effective gap is wider than the nominal one. **`MAX_CHECKPOINT_ANCHORS=2` is not the same kind of knob**: stateless rungs are free (the whole stationary ladder costs one scripted call and one random call), but each *policy* anchor is a full forward over 512 envs in slot 0 and again in slot 4, ~1024 env-forwards/step against the rollout's 2592. Keeping the 2 newest is near-equivalent to keeping the 2 nearest anyway, since live Elo climbs roughly monotonically and the free rungs cover everything below. |

---

## Invariants a fresh session must not break

- **The live gauge is defined, not fitted.** Random is 0, scripted is 1000, a
  semi-random rung is 1000·p, and `config/live_elo` is the single derivation
  site. A profile chooses which rungs exist and nothing else about the scale, so
  no environment change can leave a stale ladder behind. `bnb semi-random`
  measures how far that placement sits from a fitted one, validating the
  gauge and is never a prerequisite for training.
- **The live gauge is absolute**, scripted pinned at 1000. Normalized Elo is the
  rating itself. Milestone grid and any rating threshold are absolute.
- **Live Elo is not calibrated Elo.** The trainer logs `live_elo/*`;
  `bnb elo-calibrate` writes `calibrated_elo/*`; published results quote the
  latter only.
- **Both "is it strong yet" gates read the scripted win rate**, not Elo. Keep it
  that way, because an Elo threshold needs re-deriving every time the gauge moves.
- **`validate_field_layout` must never run on the hot path**: eight syncs and it
  raises. Generated maps are laminar by construction.
- **`advantage_min_rms` is an epsilon. `return_min_span` is NOT, and must stay
  at 1.0.** For a sparse component the p5–p95 span measures the noise floor of
  nothing happening, not the signal: `field_death`'s central 90% spans 0.0009
  while its events reach −0.12. Dropping it to an epsilon does not "fix" a
  floor-bound component, it hands that component a divisor 20–70× too small.
  `scaler/floor_bound_span/*` binding is *expected*;
  `scaler/floor_bound_rms/*` binding is still a bug. The 281× ratio in
  `critic/value_loss` between a floored and an unfloored component is what a
  squared loss does to compressed targets and says nothing about fit; read
  `critic/explained_variance` instead.
- **`max_grad_norm` renormalizes the actor and the critic together.** Any
  change to the value loss changes the actor's share of every clipped step.
  `train/actor_grad_share` reports it directly; a change that moves it is a
  change to the effective learning rate, not just to the critic.
- **Discounts encode horizons in seconds.** Changing the tick rate means
  `γ ** (rate_old/rate_new)`, not reusing the number.
- **`max_episode_steps` counts physics ticks**, not decisions.
- **`TensorEnv.step` is the decision-level call** and honours `action_repeat`.
  Only `YemongEnvWrapper` may use `tick`, because it accumulates rewards per
  physics tick. Anything else stepping at tick granularity evaluates the policy
  at a rate it was never trained for, silently.
- **Stationary references are policy-free, so `policy is None` does not identify
  the random agent.** Classify anchors by which reference they are.
- **Weights are the reward mix.** `AdvantageScaler` normalizes every component to
  unit RMS, so the effective policy-gradient share is `weight/sum(weights)` and
  nothing else. Changing a weight changes the objective directly.

## Definition of done

`uv run --no-sync pytest`, the registered `bnb smoke` profile cases, and ruff
check and format on changed files. Add tests, update docs, commit at logical
milestones.
