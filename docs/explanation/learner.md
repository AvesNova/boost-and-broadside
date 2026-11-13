---
covers: [src/bnb/learn/ppo.py, src/bnb/learn/returns.py, src/bnb/learn/losses.py, src/bnb/learn/labels.py, src/bnb/learn/shots.py, src/bnb/runtime/store.py, src/bnb/models/relational.py, src/bnb/models/nearest.py]
reviewed: 2026-11-02
---

# The learner

<!-- generated: section-status -->

The learner is PPO with v1's math over any model that has an `unroll`: per-level returns,
the categorical critic, the return scale and the auxiliary losses with their labels. It is
`bnb.learn.ppo.Learner`, configured by `bnb.learn.ppo.LearnerConfig`. The training loop
around it is in [The training loop](training-loop.md); the design record is
`docs/rewrite/sections/learner-spec.md`.

## The model protocol

The learner trains a model that

- acts through its controller in the rollout, and
- has `unroll(carry, obs, replay, reset)`, which recomputes a stored segment from the
  [carry](glossary.md#carry) at its start and returns every head's output over it.

`reset` is the flag a controller's `step` received: the episode ended before this
observation. The learner reads it off the stored observation's `reset` column, so
`reset[t]` is the stored `done[t - 1]`
(`tests/sections/test_learner.py::test_unroll_receives_the_reset_before_each_step`).

A head the model lacks is a missing field of its output, and its loss is skipped. Yemong
(`bnb.models.yemong`) has every head. The toy model, `bnb.models.relational.Relational`,
has the policy and the critic only.

## Where the segments live

An update consumes `shards` segments of `length` ticks each, collected one after another,
each continuing from the carry the last one ended with, and the lookahead after them: the
ticks the rollout plays after the last shard, never trained on, of which the store keeps the
returns' records (rewards, values, outcome probabilities with an outcome head, `done`,
the win/loss weights) and, for a
bullet's flight, the shot capture and the observation. The rollout records the learner's
views into a staging ring of two graph blocks on the device, and after each block copies the
block into a `bnb.runtime.store.SegmentStore` on a copy stream. The store keeps the segments
in pinned host memory (`host_store`, the default) or on the device. The update stages
micro-batches of views back the same way, gathering and copying the next one while the
current one computes. A rollout from the same seed writes the same records into either
store, with one shard or several, and the update's counts, losses and gradients are the
same bit for bit
(`tests/sections/test_learner_streaming.py::test_the_update_through_pinned_memory_equals_the_device_path`).
Each micro-batch size has two staging buffers that alternate, allocated on the caller's
thread before any copy into them is issued: the caching allocator may hand a new buffer memory
the compute stream freed with kernels still queued, and the first copy into it, on the copy
stream, waits for them
(`tests/sections/test_store_staging.py::test_a_new_staging_buffer_is_filled_after_the_compute_that_last_used_its_memory`).
Readers wait on the store's last write: the GPU waits where the data stays on the device,
and the host only where it reads pinned memory itself. An update synchronises with the
host once per epoch, for the early stop, and once at its end, for its results
(`tests/sections/test_learner_streaming.py::test_an_update_synchronises_only_to_read_its_kl_and_its_results`).

A segment holds, per view and tick:

- the observation parts the model reads, every row's action in view order, the
  log-probability, value, values per value level, outcome probabilities (with an outcome
  head), rewards per reward level and `done`, and the model's replay record;
- the label inputs: the enemy rows of the fog-off privileged view (`truth`; an own ship is
  always visible, so its row is the observation's), every row's `alive` flag and each own
  ship's `valued` flag;
- the shot capture (`bnb.learn.shots.ShotCapture`).

Per segment it holds the learner's carry at the start and the privileged rows of the
observation after the last tick. After the lookahead's last tick (the last shard's, without
a lookahead), `bnb.runtime.rollout.Rollout.bootstrap` evaluates the learner's controller on
the next observation without acting, for the values and outcome probabilities the returns
bootstrap from.

## Returns

Over the whole update's ticks, on the device: the shards in order, each row's segments one
after another, then the lookahead (`bnb.learn.ppo.Learner.chained_returns`). A shard's end
is not a cut: its last ticks read the next shard's rewards, and only the lookahead's end
bootstraps, so every trained tick has the lookahead's ticks, at least, after it. N shards of
L ticks give the returns of one segment of N L ticks
(`tests/sections/test_lanes.py::test_chained_returns_over_shards_equal_one_segments`).

- **The value levels' rewards** (`bnb.learn.returns.value_rewards`). The returns run per
  value level of the critic being trained (`YemongConfig.levels`): by default one per
  reward level, each its own weighted reward. The single value arm has one level,
  `summed_reward`, whose reward is the sum of every level's weighted reward
  ([below](#the-single-value-arm)).
- **Values in reward units.** A per-ship level's value is its bin expectation; the outcome
  level's is `P(win) - P(loss)` times the outcome's reward weight.
- **Per-level GAE** (`bnb.learn.returns.per_level_gae`), each level with its own discount and
  trace decay (`LearnerConfig.discount`, `LearnerConfig.trace_decay`). `done[t]` cuts the
  trace after `t`; a timeout is a real end.
- **Team mixing and the level sum** (`bnb.learn.returns.team_aggregate`): a ship's advantage
  is the sum over levels of its own advantage plus each level's team spirit times the mean
  over its living teammates. Team spirit is 0 on every level by default.
- **The outcome's targets** (with an outcome level): lambda targets of the outcome
  distribution (`bnb.models.yemong.critic.outcome_targets`), with the view's final outcome
  at an episode end (`critic.final_outcome` of the view's `Trajectory.win_loss`). In
  Elimination and Frontline that is the result one-hot (win, draw or loss). Team deathmatch's
  is soft: `P(win) = share`, `P(loss) = 1 - share`, `P(draw) = 0`, share the view's kill
  share. The head is read as `P(win) - P(loss)`, so its value at the end is `2 share - 1`,
  the outcome reward's.

`LearnerConfig.discount`, `trace_decay` and `team_spirit` hold one value per value level,
in the critic's order; the learner refuses a model whose levels they do not match. The
defaults are per level (discount / trace decay): damage 0.982 / 0.81, kill_death 0.990 /
0.90, objective_progress 0.995 / 0.95, objective 0.999 / 0.97, outcome 0.9997 / 0.97. They
are the 2026-11-09 champion's with the two combat levels' squared, because the x2 game
fights in half the ticks. One value for every level (tried 2026-11-13) let a run learn to
stop fighting.

### The single value arm

`--set yemong.levels='["summed_reward"]' --set yemong.outcome_level=null` builds Yemong with
one value output: one per-ship categorical value head (the same bins, limit and two-hot
cross-entropy as a per-ship level's) and no outcome head. Its reward is one stream, the sum
of the five weighted reward levels, and it has one discount, trace decay and team spirit,
which the run sets (`--set learner.discount='[0.999]' --set learner.trace_decay='[0.97]'
--set learner.team_spirit='[0.0]'`, for example). The rewards themselves, their weights and
the per-level reward series stay per reward level; only the value side is one level. The
critic's series name the level `summed_reward` (`critic/value_loss/summed_reward`,
`critic/explained_variance/summed_reward`, ...), and the outcome ledger's series and the
outcome figure are absent. The build record holds `levels` and `outcome_level`, so a policy
file says which layout it has (`tests/sections/test_single_value.py`).

The return scale is the gap between exponential moving averages of the 5th and 95th
percentiles of the summed returns of living own ships, at least 1, updated once per update
(`bnb.models.yemong.critic.ReturnScale`). Advantages are divided by it, with no mean
subtracted, so a factor common to every reward weight leaves the policy's gradient as it
was. The weights set the units the critic works in: each per-ship level's value is
categorical over 77 symlog bins to +-3 (+-19 in reward units, `YemongConfig.value_bins` and
`value_limit`), the bins 0.079 apart in symlog, so returns of order 1 span many of them. A
different factor per level would change the levels' mix, which is the reward's design, not
its units.

The explained variance per level (`critic/explained_variance/{level}`) is against the
lambda-returns the critic trains on, which are its own values plus the advantage, so a
smooth critic scores high whatever it predicts. `critic/explained_variance_monte_carlo/{level}`
is against the lambda 1 returns, the discounted rewards bootstrapped only after the update's
last tick: whether the value predicts the rewards. Ticks near that last tick are bootstrapped
sooner, most of all for the outcome's long horizon.

## Losses

Every term is a mean in nats per token. Each divides its masked sum by the mask summed over
the whole minibatch, so the micro-batches of a minibatch add up to the unsplit minibatch
(`tests/sections/test_learner.py::test_micro_batches_accumulate_to_the_unsplit_minibatch`).
The coefficients are the fields of `bnb.learn.ppo.LearnerConfig` named after the terms.

| Term | Tokens | Definition |
|---|---|---|
| `policy` | own ships alive at `t` | the surrogate on the scaled advantage under the trust region ([below](#the-trust-region)) |
| `entropy_power`, `entropy_stick`, `entropy_shoot` | the same | the joint action distribution's entropy by the chain rule in the joint's order: H(power), H(stick given power), H(shoot given both). They sum to the joint entropy, so equal coefficients weigh the joint as one coefficient would; a negative coefficient rewards it |
| `value` | own ships alive or waiting to respawn; each view-step for the outcome | per value level, the cross-entropy against the two-hot of the return, the outcome against its targets |
| `next_state` | rows whose belief is valid at `t`, where `t` did not end an episode (delta: and whose ship is alive at `t` and `t + 1`; with `YemongConfig.delta_split`, of those only the own rows and the enemies seen at `t`; with `YemongConfig.physics_hidden_decay` tau the hidden enemies too, past the segment's first tick, each weighted `exp(-age / tau)`, against the truth-to-truth target) | absolute: the state code's cross-entropy against the sharp code of the true state at `t + 1`, in the model's shifted frame; delta: the delta code's against the code of the true change from `t` to `t + 1` |
| `correction` | with `YemongConfig.delta_split` only: the next-state rows whose belief at `t` is a forecast (hidden enemies) | the error code's cross-entropy against the code of the physics forecast's error, `truth(t + 1) - forecast(t + 1)` |
| `enemy_action` | enemy rows alive at `t` | the cross-entropy of the enemy's actual joint action |
| `density` | each view-step, over its count of valued own ships | Poisson terms over the hex cells, against the counts of living ships at `t` |
| `shot_trajectory` | valid shots of own ships and of enemies visible at `t` | the trajectory code's cross-entropy, averaged over the shot's 4 queried ages ([below](#shot-labels)) |
| `shot_outcome` | the same shots | the outcome code's cross-entropy |
| `geometry` | with `YemongConfig.geometry_head` only: pairs of an own ship alive at `t` and a partner alive and seen at `t` (`bnb.models.yemong.geometry.pairs`) | the cross-entropy of the pair's five geometry codes, averaged over them (nats per target) |

With `YemongConfig.delta_split` the `next_state` and `correction` terms both divide by the count
of every next-state row (seen and hidden), not by their own rows: a row then weighs the
same in either term at equal coefficients, as it did in the one term before the split. So
`next_state/loss` and `correction/loss` are each a part of a mean over every next-state
row, and add up to it.

Each term matches an independent computation on a stored segment
(`tests/sections/test_learner.py::test_each_loss_term_matches_an_independent_computation`,
and `tests/sections/test_learner_geometry.py` for the geometry head).

The geometry term divides by its own count, the pairs (`bnb.learn.ppo.COUNTS` "pairs"): a view with
more ships has more pairs, as it has more next-state rows. A model without the head gives
no such output, and the term stays 0 (and out of the gradient decomposition).
`geometry/loss` is the term; `geometry/cross_entropy/{range, antenna_train_angle, aspect,
heading_crossing_angle, closure}` its parts.
GEOMETRY_COEFFICIENT_NOTE

**The persistence delta head's coefficient, 0.0033** (`LearnerConfig.next_state`'s default
since 2026-11-12, with a new run's delta head; the absolute head's was 0.0041, `--set
learner.next_state=0.0041` with the absolute head, `--set yemong.next_state=absolute` and
the split off: [the next-state head](yemong.md#the-next-state-head-absolute-or-delta)). Nats per
token change with the code, so it is set to match the next-state term's trunk gradient in
the absolute arm it is compared with (arm 1: `yemong.order=temporal_first`,
`yemong.acceleration_inputs=true`, `spread_fit="coarsest"`, BC). At initialisation the
share is exactly 0 for both heads (their last weight starts at zero; 0.007-0.012 after the
first update). Measured on a short CPU run instead: arm 1's config scaled to 64
environments of 32 ticks, 8 minibatches, float32, seed 1, the gradient decomposition at
every update, averaged over updates 10 to 20. At 0.0041 the absolute head's trunk norm
averages 0.0143 (median 0.0120, share 0.17) and the persistence head's 0.0177 (median
0.0146, share 0.24), 1.24 times more (1.21 on the medians); 0.0041 / 1.23 = 0.0033, and at
0.0033 the persistence head averages 0.0144 (median 0.0123, share 0.18). The shares at this
scale are noisy, so the run's own `gradient/share/trunk/next_state` is the check. (Round
1's delta head, now removed, had 0.0025 by the same method.)

**The error head's coefficient** (`LearnerConfig.correction`, default 0.0033, used only with
`YemongConfig.delta_split`) starts equal to the physics head's per row (both 0.0033; the shared
denominator above makes equal coefficients equal per row). Measured on arm 3's short CPU
run (arm 2's BC config scaled to 64 environments of 32 ticks, 8 minibatches, float32, seed
1, the gradient decomposition at every update, updates 10 to 20): the physics head's trunk
norm averages 0.0146 (share 0.19; arm 2's one head 0.0144 at the same coefficient) and the
error head's 0.0065 (share 0.036), over token shares of 0.64 seen rows (0.49 own, 0.15
visible enemies) and 0.36 hidden rows. Per row the error head's trunk gradient is 0.78
times the physics head's, inside the 1.5x band, so its coefficient stays 0.0033. On arm 4's short
run (the extrapolation baseline, the same scale) the physics head's trunk norm averages
0.0119 and the error head's 0.0086, 1.32 times per row, also inside the band. The run's
`gradient/norm/trunk/correction` is the check.

An update makes `epochs` passes of `minibatches` minibatches over the views; a view keeps
its whole segment, which `unroll` needs. It clips the gradient norm, steps Adam with a
linear warmup of the learning rate over the first 10M environment steps (a run warm-started
from a clone warms up again; the uniform average of the weights starts when the warm-up
ends, `TrainConfig.uniform_start`, unless `LeagueConfig.uniform_from` is set), and stops early after
an epoch whose mean approximate KL divergence exceeds `target_kl` (None: never). The
policy's terms and statistics are one method, `Learner._policy`{.plain}; behaviour cloning
(`bnb.learn.bc.BehaviourCloning`) replaces it and keeps the rest of the update, with 256
minibatches instead of 32: its progress follows its optimizer steps. The minibatch order and
the shot spread come from the learner's own random streams (`bnb.learn.ppo.random_stream`,
seeded from the run's `TrainConfig.seed`), so the global generator does not change an update. The old log-probabilities are
the rollout's, or recomputed by `unroll` before the first epoch
(`LearnerConfig.old_log_probabilities`).

## The trust region

`LearnerConfig.trust_region` chooses how the policy term limits a step
(`bnb.learn.ppo.Learner._policy`{.plain}, `bnb.learn.losses`), with r the ratio of the
sampled joint action's probability now to the rollout's and A the scaled advantage:

| `trust_region` | The policy term per token |
|---|---|
| `"clip"` (default) | PPO's: max(-A r, -A clip(r, 1 - `clip`, 1 + `clip`)), as before the option (`tests/sections/test_learner_trust_region.py::test_the_clip_is_todays`) |
| `"clip_higher"` | DAPO's asymmetric clip: the ratio in [1 - `clip`, 1 + `clip_high`], 0.2 and 0.28 |
| `"dppo_tv"` | DPPO's mask (Qi et al., "Rethinking the Trust Region in LLM Reinforcement Learning", ICML 2026): the unclipped -A r, its gradient zeroed where D > `dppo_delta` (0.15) and the step moves further away (A > 0 and r > 1, or A < 0 and r < 1). D is the exact total variation between the rollout's and the current joint action distribution at the state, 1/2 sum over the 42 joint actions of \|mu(a) - pi(a)\| |
| `"dppo_binary_tv"` | the same mask with D = \|mu(a) - pi(a)\| of the sampled action, the paper's binary approximation; a lower bound of the exact form, so it masks less at the same delta |

Yemong's policy is one categorical over the 42 joint actions per ship (not three heads),
so the exact total variation needs no product of marginals: the rollout records each own
ship's joint distribution as float16 (`Outputs.probabilities`, `Trajectory.probabilities`),
only under `"dppo_tv"`. That is 42 × 2 bytes per own ship per tick, 420 bytes per learner
view per tick at 5v5, about a tenth of the segment store's per-tick records (observation,
replay, shot capture, truth and the rest, about 4.2 kB per view-tick at 5v5), and nothing
on the device beyond a micro-batch's slice. `"dppo_tv"` needs
`old_log_probabilities = "rollout"`: its mu is the rollout's.

Every mode logs `policy/clip_fraction`{.plain} as its trust region's rate (the ratio past
the clip's bounds, or the samples DPPO masks) and the same rate by the policy's joint
entropy, in thirds of ln 42 (`policy/clip_fraction/entropy_low`{.plain}, `policy/clip_fraction/entropy_middle`{.plain},
`policy/clip_fraction/entropy_high`{.plain});
and the divergence D (the binary form under the clip modes): its mean, its share past
`dppo_delta` and its quantiles p50, p90 and p99 (`policy/divergence/*`{.plain}, from a
log histogram over [1e-4, 1]).

<!-- not-runnable: needs a run -->
```bash
uv run --no-sync bnb train ... --set learner.trust_region=dppo_tv --set learner.dppo_delta=0.15
```

## Shot labels

Every tick, each living ship's shot is labelled along its whole flight, by firing it again
with the game's own code (`bnb.learn.shots.resimulate`):

- **Capture.** The rollout stores each ship's launch state as `fire` reads it
  (`bnb.game.physics.bullets.launch_state`), its move that tick, and whether it was alive.
- **Scan.** One bullet per row and age, newest first. Each tick every row fires a fresh
  bullet with `bnb.game.physics.bullets.launch`, every bullet moves with
  `bnb.game.physics.bullets.move_bullets`, and the game's swept test
  (`bnb.game.physics.combat.contacts`, `bnb.game.physics.combat.first_contact`) runs it against
  the tick's stored ship motion. The spread is drawn from the learner's own generator, once
  per update, and so are the queried ages' uniform draws (below). The re-simulation runs once per update, before the epochs, and its labels go
  into the store beside the segments; every epoch reads them back.
- **Each shot records** at every age its displacement and velocity in the launch heading's
  frame and the refractive index, and its first hit (an enemy or an ally of the shooter), or
  else its closest pass and that ship's side. Every age is labelled, past a hit too: the
  bullet flies all its moves in the re-simulation, and past its hit the snapshots are its
  straight-on flight (the trajectory code has no hit component; the outcome code keeps the
  first hit), so a barely-hit and a barely-missed shot train the same path. A shot whose
  whole flight crosses an episode end is invalid, and so is every shot launched within the
  longest flight of the ticks played after it, whatever it hit: keeping the ones that hit in
  time would bias the labels toward hits. The re-simulation flies on into the bullet flight
  after the segment (the next shard's first ticks, or the lookahead's after the last shard;
  `bnb.runtime.store.SegmentStore.following`), so a shot fired in a segment's last ticks is
  labelled; only where nothing was played after the segment (no lookahead) are they
  invalid.

A re-simulated shot is the bullet the game fires from that state, position, velocity and
hit, tick by tick, including friendly fire and fields (`tests/sections/test_learner_shots.py`).
The targets are Yemong's shot codes (`bnb.models.yemong.shots`). The trajectory head is
queried at 4 ages per shot (`bnb.learn.shots.query_ages`), chosen with the labels: the last
move; one uniform over the whole flight (it may repeat another); the closest pass to
another living ship, of either side; the first move into a field's band, where the
refractive index first changes and the path bends. Each of the last two is a uniform age of
its own when the shot passes no ship or meets no band. The loss is the mean over the 4,
and the head runs the 4 queries in one kernel (`bnb.models.yemong.kernels.heads`, each
row's own ages from a table of every age).

## The other labels

Built per micro-batch on the device, from the stored rows (`bnb.learn.labels`), row-aligned
with the view (as are the shot labels' codes; expanded, the codes would be many times the
size of the segment):

- **Next state:** the sharp code of each row's truth at `t + 1`, its position shifted by the
  view's offset; the last step's truth is the observation after the segment. With the delta
  head (`YemongConfig.next_state = "delta"`), the code of each row's true change from `t`
  to `t + 1` (persistence; `bnb.models.yemong.delta.DeltaCode.targets`), which changes
  were clamped to the code's padded range (`next_state_clipped`), and whether the frame
  was the heading (slower than 1 m/s, `next_state_slow`). Under the extrapolation
  baseline the change is from the baseline of the true state at `t` under the rates the
  model read (`bnb.learn.labels.rates`, `Yemong.segment_rates`). With `YemongConfig.delta_split`, the
  next-state mask split by sight at `t` (`bnb.learn.labels.split_masks`): `physics_mask`,
  the own rows and the enemies seen at `t`, and `correction_mask`, the enemies whose belief
  at `t` is a forecast; with `YemongConfig.physics_hidden_decay` also `physics_weight`, the
  physics head's weight per row (1 seen, `exp(-age / tau)` hidden), and a hidden row's
  rates in the target are the truth's (`bnb.learn.labels.true_rates`) instead of the
  belief's held ones.
- **Error (with `YemongConfig.delta_split`):** built inside the loss, after the unroll, since it
  reads the physics head (`bnb.learn.labels.correction_targets`): the code of each row's
  `e = truth(t + 1) - forecast(t + 1)`, the forecast the physics advance of the row as the
  model read it at `t` (`bnb.models.yemong.inputs.believed`) under the physics head's
  detached logits, as `step` made it (`Yemong.forecast`: with
  `YemongConfig.physics_hidden_fade`, a hidden row's forecast faded by its age).
- **Enemy action:** each enemy row's joint action at `t`.
- **Density:** the hex counts of the living ships of each side at `t`, in world coordinates
  scaled by each environment's arena radius.
- **Geometry:** the pairs the view sees at `t` (`bnb.learn.labels.geometry_pairs`), each
  pair's range, angles and closure from their true states.

## Compilation and precision

Section code does not compile itself. The learner takes a `stage` function from the runtime
(`bnb.runtime.train.compiler`), which compiles the micro-batch loss and the
re-simulation's tick (`bnb.learn.shots.tick`). The compile is of fixed shapes, so every
micro-batch size compiles each stage once more. The micro-batches take as few sizes as the
views allow (`bnb.learn.ppo.micro_batch_sizes`): one when the minibatches divide the views,
two when they do not (`tensor_split` gives minibatches of two sizes that differ by one, and
no size above one divides both). A run with compiling on prints its sizes before the first
update (`bnb.runs.session.micro_batch_note`); the cadence's halves of the first minibatch
can add their own. Two diagnostics run eager. The heads' sums (`bnb.learn.head_statistics`)
are measured on cadence updates over a sample: the first epoch's first micro-batches until
they hold `HEAD_SUM_VIEWS` (256) views. Over every micro-batch they cost about 8% of RL's
steady time. The step/unroll drift unrolls the update's first micro-batch
(`Learner.first_micro_batch`). `TrainConfig.precision` defaults to
`"bfloat16"`: the model's matmuls run under bf16 autocast, in the rollout and in the update
alike. `"float32"` (strict) and `"tf32"` (a process-wide setting the entry point applies with
`bnb.cli.set_precision`) are the alternatives. The residual stream, norms,
softmax, the recurrent state, the codes, the belief, the critic's expectations, every loss,
the weights and the optimizer stay float32 in all of them.

Each update logs the signals that show a precision problem: `policy/ratio_drift/max`{.plain}
and `policy/ratio_drift/mean`{.plain} (how far the update's `unroll` moves the policy from
the rollout's `step`, before any optimizer step; bf16 rounding makes it nonzero),
`policy/clip_fraction`{.plain}, `train/skipped_steps`{.plain} (minibatch steps whose
gradient norm was not finite; the step is skipped, leaving the weights and the optimizer's
state as they were) and `counters/nonfinite_beliefs`{.plain} (non-finite belief entries the
learner's carry has counted since its rollouts began).

## The toy learner

`bnb.models.relational.Relational` is a fixed-size relational model for small team
deathmatch. Each own ship reads its own state and, for every other row of the view, that
ship relative to it in its heading frame: position, velocity, distance and relative heading
where it is visible, and its team, `alive` and `visible` flags. Two hidden layers feed the
joint 42-way policy, the per-ship categorical critic, and the outcome from the mean of the
own ships' hidden states. It is stateless, so its `unroll` is its forward over every step.
Trained from scratch against the chaser at 2v2, it must win at least 60% of 256 games after
the budget in `tests/loop/test_learning.py`; the [training guide](../guides/training.md#the-toy-learner)
shows how to run it.

## The nearest-neighbour MLP

`bnb.models.nearest.Nearest` (`--model nearest`, settings `TrainConfig.nearest`,
`NearestConfig`) is the relational model's outputs over an input whose width depends on
neither the fleet nor the field count, so a model trained at 5v5 plays any size. It is
stateless and has no memory or belief: it sees only what is visible now. Each own ship
reads (widths at the defaults, 196 in all, `nearest.input_width`):

| Block | Width | Contents |
|---|---|---|
| own state | 15 + 10 | the relational model's own features; the globals (episode progress, arena radius, fog, mode one-hot, scores own, enemy and target, front and threshold), scaled as the MLP's `GLOBAL_SCALE` |
| allies | 3 x 11 | the nearest living allies, nearest first: the relational model's 10 pair features and a presence flag |
| enemies | 4 x 11 | the nearest living visible enemies (a hidden one is never a candidate), the same 11 |
| fields | 3 x 8 | the nearest fields by the distance to their centre: centre offset in the heading frame, centre distance, signed edge distance (centre distance minus radius), radius, band width, level, presence |
| zones | 5 x 14 | one slot per role (own spawn, own defense, neutral, enemy defense, enemy spawn): centre offset in the heading frame, distance, radius, role one-hot, progress, direction, symlog of the captures to win and to lose, presence |

A missing ally, enemy or field, and a role no present zone has (team deathmatch), are zero
with presence 0. The own ships are the first half of the view's rows, so the build record
(`NearestBuild`) holds the config alone. Two hidden GELU layers of `hidden` (256) feed the
joint policy, the critic and the outcome (the mean of the own ships' hidden states); at the
defaults it has 180,474 parameters. It plays through
`bnb.controllers.relational.RelationalController`, and like the relational model has no
auxiliary heads, so neither PPO nor BC computes those terms (`Learner.auxiliary` is False).
