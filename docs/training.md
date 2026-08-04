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
| PPO minibatches | 32 |
| Token width / attention heads / blocks | 128 / 4 / 2 |
| Episode horizon | 1,024 steps |
| Opponent paradigm | `ego_pass` |
| Elo evaluation games per matchup slot | 512 |

The run logged 999,424,000 steps before finishing. Today's profiles have continued to
evolve, so where this page and the export disagree about that run, the export is what
actually ran.

## Recurrent PPO lifecycle

[`PPOTrainer`](../src/boost_and_broadside/train/rl/ppo.py) owns environment groups,
policies, rollout collection, [generalized advantage
estimation](https://arxiv.org/abs/1506.02438), update-time sequence re-evaluation,
evaluation, logging, and checkpoints.

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
extra attention tokens. The scripted controller applies a mild material-aware steering
bias near an interface. It favors remaining on the current side, with stronger influence
from index contrast and boundary damage, but caps the blend at 35% so fields do not become
impenetrable walls in behavior-cloning targets.

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
the fixed random anchor. That exclusion is load-bearing — the live rating and random's both
start at zero, so including it would make the early league mostly random play.

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

Anchor games use the two newest frozen ladder checkpoints, with per-episode assignment
weighted toward the matchup that carries the most rating information. Ties count as half
a win. These live ratings steer opponent selection and training decisions, but they
remain a filtered online estimate.

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
