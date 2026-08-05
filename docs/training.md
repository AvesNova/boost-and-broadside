# Training system

Boost and Broadside trains a recurrent centralized team policy with clipped
[PPO](https://arxiv.org/abs/1707.06347). The training loop combines scripted-opponent
bootstrapping, self-play, a running-average policy, frozen historical opponents,
decomposed rewards, and continuous evaluation.

This page explains the current implementation and calls out reference-run settings where
they matter. Exact results and post-hoc methodology are in [evaluation](evaluation.md).

## Reference run at a glance

The preserved [run configuration](../checkpoints/resilient-resonance-682/wandb_export/config.json)
records:

| Setting | Value |
|---|---:|
| Target environment steps | 1,000,000,000 |
| Parallel environments | 7,808 |
| Ships per environment | 8 total (4-vs-4) |
| Rollout length | 128 steps |
| Decisions per second | 60 (`action_repeat=1`) |
| PPO minibatches | 32 |
| Token width / attention heads / blocks | 128 / 4 / 2 |
| Episode horizon | 1,024 physics ticks (17.1 s) |
| Opponent paradigm | `ego_pass` |
| Elo evaluation games per matchup slot | 512 |

The run logged 999,424,000 steps before finishing. Today's profiles have continued to
evolve, so where this page and the export disagree about that run, the export is what
actually ran. In particular the reference run decided at 60 Hz; the current profile holds
each action for three physics ticks (see [decision rate](#decision-rate)), so its step
counts and discounts are not directly comparable.

## Recurrent PPO lifecycle

[`PPOTrainer`](../src/boost_and_broadside/train/rl/ppo.py) owns environment groups,
policies, rollout collection, [generalized advantage
estimation](https://arxiv.org/abs/1506.02438), update-time sequence re-evaluation,
evaluation, logging, and checkpoints.

### Decision rate

Physics always runs at `ShipConfig.dt` = 1/60 s. `EnvConfig.action_repeat` sets how many
of those ticks each chosen action is held for, so collision and projectile integration
are unaffected and only the rate at which the policy may change its mind moves. The
primary profile holds for 3 ticks — **20 Hz decisions**.

That rate is set by the plant, not by the renderer:

| timescale | seconds | decisions @ 20 Hz |
|---|---:|---:|
| firing cooldown | 0.10 | 2 |
| bullet flight to ~200 px | 0.40 | 8 |
| full 360° turn | 1.3–2.3 | 26–46 |
| mean episode | ~4.7 | ~93 |
| `num_steps=128` rollout | 6.4 | 128 |

At 60 Hz five of every six shoot decisions were no-ops against the cooldown, consecutive
observations differed by 17 ms, and a 128-step rollout spanned 2.1 s against a ~4.7 s
episode — so the recurrent policy never saw a whole episode inside one BPTT window. At
20 Hz it does, and a token buys three times the game time.

Rewards are summed across the held ticks. That is scale-preserving: over a fixed span of
game time both the dense per-tick terms and the one-off event terms total exactly what
they would at repeat 1, so the component ratios in `RewardConfig` are untouched. Episode
lengths and ship ages stay in physics ticks so they remain comparable across rates.

Discounts do **not** carry over unchanged. They were chosen as horizons in seconds, so
moving the rate requires `gamma_new = gamma_old ** (rate_old / rate_new)` — and the same
for the GAE lambdas, since variance accumulates per unit of game time rather than per
decision. The horizons those values encode are tabulated in
[`runs/shared.py`](../runs/shared.py).

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
that already ran — it holds the action **about to be applied**, written into the
observation as it is handed forward. So the stored transition is
`(state, pending action) → action`, and a chosen action shows up in the reward one step
later, which GAE handles through the value function.

Two consequences worth knowing before touching the auxiliary losses:

- The channel is `(B, N+M, 3)`, so spatial attention lets **every ship read every other
  ship's pending action**, opponents included. One step is 1/60 s against a ~0.4 s bullet
  flight, so the lookahead is small, and it is symmetric.
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

Two gates key off the same signal — the raw win rate against the scripted controller,
taken from the evaluation battery rather than from training envs. It decays the
behavior-cloning weight to zero at `bc_winrate_target`, and it tightens `target_kl` at
`high_winrate_threshold`. Using one measure of "is the policy strong yet" rather than two
also keeps the trust region independent of the Elo gauge, which would otherwise need
re-deriving whenever the anchor or the environment moved.

Advantages are scaled per component with running RMS statistics. Returns use a
per-component percentile scaler in symlog reward space, which keeps critic targets in a
stable range without forcing components with different natural scales into one value
head. The exact loss assembly and logging proxies live in
[`ppo.py`](../src/boost_and_broadside/train/rl/ppo.py).

Both scalers carry a floor, and a floor that binds on an active component replaces that
component's own scale with the guard's. `advantage_min_rms` is therefore a true epsilon:
the terminal win signal's advantage RMS is around 0.008, two orders of magnitude below a
per-step damage signal, and an earlier floor of 0.1 was downweighting it roughly
thirteenfold in the policy gradient. `return_min_span` is *not* an epsilon and is held at
1.0 on purpose — see the note in [`runs/rl.py`](../runs/rl.py) for why lowering it needs
the critic's outlier sensitivity addressed first. `scaler/floor_bound_span/*` and
`scaler/floor_bound_rms/*` report which components each floor is currently holding up.

## Reward decomposition

Rewards are emitted as named components by [`rewards.py`](../src/boost_and_broadside/env/rewards.py).
Each active component receives its own critic output and can have its own GAE gamma/lambda
horizon. Weights are magnitudes; each component carries its own sign, noted below. The
reference policy activated these components:

| Component | RL / fields weight | Role |
|---|---:|---|
| `ally_win` | 4.0 | +1 to each surviving teammate on a win |
| `enemy_win` | 4.0 | opponent's win signal, seen as −1 through a negative enemy lambda |
| `facing` | 0.1 | dense aim geometry (+) |
| `closing_speed` | 0.1 | dense approach geometry (+) |
| `shoot_quality` | 0.1 | firing opportunity quality (+) |
| `kill_shot` | 1.0 | fatal-step credit (+), proportional to that step's damage; killing a friendly earns the negative share |
| `kill_assist` | 1.0 | assist credit (+), proportional to cumulative episode damage |
| `combat_damage_taken` | 0.5 / 0.5 | −applied projectile health loss |
| `field_damage_taken` | off / 0.5 | −applied boundary health loss |
| `damage_dealt_enemy` | 0.5 | +proportional to damage dealt to enemies |
| `damage_dealt_ally` | 0.5 | −proportional to friendly fire dealt |
| `combat_death` | 1.0 / 1.0 | −1 when projectile damage kills this ship |
| `field_death` | off / 1.0 | −1 when boundary damage kills this ship |

Weights are normalized by their absolute sum, and the wrapper divides component rewards
by total ship count for team-size normalization. A lambda aggregation matrix then maps
local event signals to training targets:

- local components use diagonal/self-only credit;
- global outcome components aggregate across live teammates;
- selected enemy-perspective components use negative enemy coefficients to recover
  zero-sum outcome structure.

Note that `kill_shot` is not winner-take-all: when several ships damage a target on its
fatal step, each earns credit proportional to that step's damage. `kill_assist` remains
proportional to cumulative episode damage even when a field delivers the final blow;
that preserves partial credit for attacks that force a dangerous navigation choice.

The former solid-obstacle death, proximity, closing-speed, and time-to-impact components
have been removed: refractive interfaces are traversable and should not receive universal
wall-avoidance shaping. Applied interface and projectile health loss, plus their exclusive
death causes, are recorded separately so neither source can double-count overkill.
Interfaces also reduce projectile damage potential, but that
barrier loss is not credited to a ship; only damage that reaches a target enters combat
attribution. See [`runs/shared.py`](../runs/shared.py) for current component horizons and
schedules, and the preserved run config for historical weights.

## Field training profile

The primary [`runs/rl.py`](../runs/rl.py) profile remains an exact zero-field combat
baseline. [`runs/rl_fields.py`](../runs/rl_fields.py) adds four cached static fields,
activates the two local field reward heads, and reduces environment count to offset the
extra attention tokens. The scripted controller ignores fields entirely: it aims and
manoeuvres as if the medium were uniform.

It used to carry a mild stay-on-your-side steering bias, on the theory that behavior
cloning needed field-dependent targets to warm up the attention trunk. Measurement killed
it. Against a uniform-random agent the bias produced *more* interface crossings (2.24
against 1.60 per thousand ship-steps) and left ships in higher-index — slower — medium
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
distinct map per episode. Maps are laminar by construction — candidates are rejected
against already-placed fields before acceptance — which matters because
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
`league_fraction` (0.5 in [`runs/rl.py`](../runs/rl.py)). The league half is divided into
`league_slots` contiguous slots, and each slot draws its own opponent from the roster at
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
rating, so including it would make the early league mostly random play — which self-play
already provides, at twice the actor tokens.

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
cache. An entry this run cannot host — a bullet-reading policy in a bullet-free run, whose
rollout observation shape is fixed when the wrapper is built — is retired from sampling
with its rating intact, rather than ending the run.

## Continuous rating and the frozen ladder

[`elo_eval.py`](../src/boost_and_broadside/train/rl/elo_eval.py) advances dedicated
evaluation environments alongside training. The evaluator has five logical slots:

1. live vs fixed anchor;
2. live vs floating checkpoint;
3. live vs scripted controller;
4. live vs running-average policy;
5. floating checkpoint vs fixed anchor.

Ratings live on an **absolute gauge with the scripted controller pinned at 1000**, the
same convention the post-hoc calibration reports, so in-training and calibrated numbers no
longer need re-basing against each other. Slot 2 therefore updates the live policy rather
than scripted: the player defining the scale must not drift under the one being measured
against it.

The anchor pool has two parts. **Stationary references** — the random agent, the
semi-random rungs, and the scripted controller — sit at its head and never age out,
because their strength is a fixed property and their ratings are measured constants.
**Checkpoint anchors** follow: the newest `MAX_CHECKPOINT_ANCHORS` frozen ladder
snapshots, which do rotate as the live policy leaves them behind.

### The reference ladder

With only random and scripted as fixed references, the live policy saturates both for the
whole early climb — winning ~100% against one and losing ~100% against the other — so its
rating is barely identified exactly when opponent selection depends on it. A ladder of
semi-random rungs (`TrainConfig.reference_ladder`) fills that range. Each rung takes the
scripted action with probability `p` and a uniform one otherwise, and their ratings are
fitted offline by `--mode semi_random --profile <name>`.

Those ratings are a property of the environment the rungs play in, so a ladder is valid
only for the tick rate, field count, ship config and fleet size it was measured under.
The two shipped profiles differ sharply — on the scripted-anchored gauge the random agent
sits at **−351** in `rl` and **+170** in `rl_fields`, because refractive fields compress
the skill scale — so each profile carries its own ladder and re-running the tournament is
mandatory whenever the environment moves.

Per-episode assignment is a multinomial draw over the information weights, so the pool
can be any size at no extra environment cost — the slot's envs simply redistribute, and
saturated references draw almost no games. Stationary references also cost no forward
pass: every semi-random rung is a Bernoulli blend of the same two action tensors, so the
whole stationary ladder is computed from one scripted call and one random call however
many rungs it holds.

Ties count as half a win. These live ratings steer opponent selection and training
decisions, but they remain a filtered online estimate.

At configured rating milestones, the trainer writes unpruned ladder snapshots. After
training, [`elo_calibrate.py`](../src/boost_and_broadside/modes/elo_calibrate.py) replays
stationary players and refits historical match records to construct the more rigorous
reported curve. The [evaluation guide](evaluation.md#post-hoc-elo-calibration) explains why
the two rating series differ.

## Checkpoints and reproducibility

Every payload family — full `step_<N>.pt` resumes, best-model snapshots, and the ladder
snapshots the league and calibrator reload — carries the same provenance block: the
observation schema, the weights, critic width, `team_pma_k`, step and rating, the training
paradigm, and the model, environment, and ship configs it was trained under. Full
checkpoints add optimizer, scaler and averaging state on top; ladder snapshots add nothing.
Saves are prepared asynchronously and written through a temporary file before rename.
[`checkpoint.py`](../src/boost_and_broadside/train/rl/checkpoint.py) defines the filenames.
(The included reference-run directory retains `recent_avg.pt` from an older naming
convention.)

Checkpoints are rebuilt from their own recorded configs rather than from whatever the
reader is running, by
[`policy_io.load_policy_bundle`](../src/boost_and_broadside/train/rl/policy_io.py) — the
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
  policy then reads the world through the constants it trained on.
- **Architecture.** Nothing needs to match. Nothing in the policy is sized by ship count,
  and each entry is rebuilt from its own config, so a league or rating field can hold
  checkpoints of different widths and depths — an entry whose architecture differs from the
  live run's simply runs eager rather than claiming a compiled graph nothing else reuses.
  The one exception is the training rollout, whose observation shape is fixed when the
  wrapper is built: a bullet-reading opponent in a bullet-free run is refused rather than
  left to play blind.

Payloads written before provenance existed still load; the loader falls back to the
caller's configs and warns, naming what it assumed.

W&B logging runs off the main training path. The reference run's sampled metric history,
configuration, summary, and run metadata are exported under
[`wandb_export/`](../checkpoints/resilient-resonance-682/wandb_export/) so the published
charts can be rebuilt without relying on a hosted dashboard.

## Engineering validation

Training behavior is covered across:

- [`test_ppo.py`](../tests/train/test_ppo.py) for loss, masking, rollout, and schedule logic;
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
