---
covers: [src/bnb/learn/ppo.py, src/bnb/learn/returns.py, src/bnb/learn/losses.py, src/bnb/learn/labels.py, src/bnb/learn/shots.py, src/bnb/runtime/store.py, src/bnb/models/relational.py]
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
returns' records (rewards, values, outcome probabilities, `done`, results) and, for a
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
  log-probability, value, values per level, outcome probabilities, rewards and `done`, and
  the model's replay record;
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

- **Values in reward units.** A per-ship level's value is its bin expectation; the outcome
  level's is `P(win) - P(loss)` times the outcome's reward weight.
- **Per-level GAE** (`bnb.learn.returns.per_level_gae`), each level with its own discount and
  trace decay (`LearnerConfig.discount`, `LearnerConfig.trace_decay`). `done[t]` cuts the
  trace after `t`; a timeout is a real end.
- **Team mixing and the level sum** (`bnb.learn.returns.team_aggregate`): a ship's advantage
  is the sum over levels of its own advantage plus each level's team spirit times the mean
  over its living teammates. Team spirit is 0 on every level by default.
- **The outcome's targets:** lambda targets of the outcome distribution
  (`bnb.models.yemong.critic.outcome_targets`), with the view's result at an episode end.

The return scale is the gap between exponential moving averages of the 5th and 95th
percentiles of the summed returns of living own ships, at least 1, updated once per update
(`bnb.models.yemong.critic.ReturnScale`). Advantages are divided by it, with no mean
subtracted.

## Losses

Every term is a mean in nats per token. Each divides its masked sum by the mask summed over
the whole minibatch, so the micro-batches of a minibatch add up to the unsplit minibatch
(`tests/sections/test_learner.py::test_micro_batches_accumulate_to_the_unsplit_minibatch`).
The coefficients are the fields of `bnb.learn.ppo.LearnerConfig` named after the terms.

| Term | Tokens | Definition |
|---|---|---|
| `policy` | own ships alive at `t` | the surrogate on the scaled advantage under the trust region ([below](#the-trust-region)) |
| `entropy_power`, `entropy_stick`, `entropy_shoot` | the same | the joint action distribution's entropy by the chain rule in the joint's order: H(power), H(stick given power), H(shoot given both). They sum to the joint entropy, so equal coefficients weigh the joint as one coefficient would; a negative coefficient rewards it |
| `value` | own ships alive or waiting to respawn; each view-step for the outcome | per level, the cross-entropy against the two-hot of the return, the outcome against its targets |
| `next_state` | rows whose belief is valid at `t`, where `t` did not end an episode | absolute: the state code's cross-entropy against the sharp code of the true state at `t + 1`, in the model's shifted frame; delta: the delta code's against the code of the true change from `t` to `t + 1` |
| `enemy_action` | enemy rows alive at `t` | the cross-entropy of the enemy's actual joint action |
| `density` | each view-step, over its count of valued own ships | Poisson terms over the hex cells, against the counts of living ships at `t` |
| `shot_trajectory` | valid shots of own ships and of enemies visible at `t` | the trajectory code's cross-entropy, averaged over the shot's valid ages |
| `shot_outcome` | the same shots | the outcome code's cross-entropy |
| `rotary_readout` | every ship alive at `t`, visible or hidden (with Yemong's rotary read-out only) | the squared distance between a block's read-out (cos, sin) pair and the true one, averaged over the rotary frequencies (`bnb.learn.losses.phasor_square_error`), then over the blocks' read-outs; not in nats |

Each term matches an independent computation on a stored segment
(`tests/sections/test_learner.py::test_each_loss_term_matches_an_independent_computation`,
and `tests/sections/test_learner_rotary_readout.py` for the read-out).

The read-out's term is the one term not in nats: a per-ship mean over the frequencies of
`|p - p_true|^2`, between 0 and 4, averaged over the blocks (each block has its own
read-out). An all-zero read-out scores 1. Its minimum over the
model's uncertainty is the expected phasor, so the read-out learns to shorten its pairs
where it is unsure. A model without the read-out gives no such output, and the term stays 0
(and out of the gradient decomposition). `rotary_readout/loss` and the
`rotary_readout/error/*` population and hidden-age series are the blocks' mean;
`rotary_readout/error/block_<k>` is block k's error over the living ships.
**The read-out's coefficient, 0.007.** Set so the read-out's trunk gradient norm at
initialisation is about 0.02, the level the next-state term settles at in a trained BC run.
Measured on the CPU (arm D, production initialisation, 5v5, 16 environments of 128 ticks,
BC's gradient decomposition): at coefficient 1 the read-out's trunk norm is 2.97, so 0.007
gives about 0.021. Calibrating its *share* at initialisation would mislead: there the other
terms' trunk norms are about 0.002, because their heads start at or near zero, while in a
trained BC run they settle at 0.02-0.10 (policy 0.04-0.07, density 0.05-0.08, enemy action
0.05-0.10, next state about 0.02, value 0.02-0.07). The arm's own logged
`gradient/norm/trunk/rotary_readout` is the check. The measurement predates the read-out
in every block (it was one head, on the first block) and was not repeated for it.

An update makes `epochs` passes of `minibatches` minibatches over the views; a view keeps
its whole segment, which `unroll` needs. It clips the gradient norm, steps Adam with a
linear warmup of the learning rate over the first 10M environment steps (a run warm-started
from a clone warms up again), and stops early after
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
| `"clip"` (default) | PPO's: max(-A r, -A clip(r, 1 - `clip`, 1 + `clip`)), as before the option (`tests/sections/test_learner_trust_region.py::test_the_clip_is_todays_bit_for_bit`) |
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

Every tick, each living ship's shot is labelled at every age of its flight, by firing it
again with the game's own code (`bnb.learn.shots.resimulate`):

- **Capture.** The rollout stores each ship's launch state as `fire` reads it
  (`bnb.game.physics.bullets.launch_state`), its move that tick, and whether it was alive.
- **Scan.** One bullet per row and age, newest first. Each tick every row fires a fresh
  bullet with `bnb.game.physics.bullets.launch`, every bullet moves with
  `bnb.game.physics.bullets.move_bullets`, and the game's swept test
  (`bnb.game.physics.combat.contacts`, `bnb.game.physics.combat.first_contact`) runs it against
  the tick's stored ship motion. The spread is drawn from the learner's own generator, once
  per update. The re-simulation runs once per update, before the epochs, and its labels go
  into the store beside the segments; every epoch reads them back.
- **Each shot records** at every age its displacement and velocity in the launch heading's
  frame and the refractive index, and its first hit (an enemy or an ally of the shooter), or
  else its closest pass and that ship's side. Ages after a hit are invalid. A shot whose
  flight crosses an episode end is invalid, and so is every shot launched within the
  longest flight of the ticks played after it, whatever it hit: keeping the ones that hit in
  time would bias the labels toward hits. The re-simulation flies on into the bullet flight
  after the segment (the next shard's first ticks, or the lookahead's after the last shard;
  `bnb.runtime.store.SegmentStore.following`), so a shot fired in a segment's last ticks is
  labelled; only where nothing was played after the segment (no lookahead) are they
  invalid.

A re-simulated shot is the bullet the game fires from that state, position, velocity and
hit, tick by tick, including friendly fire and fields (`tests/sections/test_learner_shots.py`).
The targets are Yemong's shot codes (`bnb.models.yemong.shots`), and the trajectory head is
queried at every age at once.

## The other labels

Built per micro-batch on the device, from the stored rows (`bnb.learn.labels`), row-aligned
with the view (as are the shot labels' codes; expanded, the codes would be many times the
size of the segment):

- **Next state:** the sharp code of each row's truth at `t + 1`, its position shifted by the
  view's offset; the last step's truth is the observation after the segment. With the delta
  head (`YemongConfig.next_state = "delta"`), the code of each row's true change from `t`
  to `t + 1`, from the baseline of its truth at `t`
  (`bnb.models.yemong.delta.DeltaCode.targets`), and which changes were clamped to the
  code's padded range (`next_state_clipped`).
- **Enemy action:** each enemy row's joint action at `t`.
- **Density:** the hex counts of the living ships of each side at `t`, in world coordinates
  scaled by each environment's arena radius.
- **Rotary read-out:** the true (cos, sin) of every row's rotary angles at `t`
  (`bnb.learn.labels.rotary_targets`): its true position at `t`, shifted by the view's
  offset, and its true heading. These are the angles the row's tables would have if the ship
  were visible, which is what the read-out stands in for. The term masks dead rows.

## Compilation and precision

Section code does not compile itself. The learner takes a `stage` function from the runtime
(`bnb.runtime.train.compiler`), which compiles the micro-batch loss and the
re-simulation's tick (`bnb.learn.shots.tick`). `TrainConfig.precision` defaults to
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
