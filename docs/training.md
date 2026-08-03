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

The current primary profile starts with a high scripted-opponent fraction, then introduces
average-policy and league games while retaining self-play. Fractions are schedules over
the global environment step in [`runs/rl.py`](../runs/rl.py).

The main opponent types are:

- **scripted:** a stochastic hand-built controller, used for early supervision, direct
  opposition, and a stable evaluation benchmark;
- **self:** the live weights viewed from the other team perspective;
- **average:** a uniform running mean of eligible live-policy snapshots after the scripted
  performance cutoff;
- **league:** frozen historical checkpoint policies sampled near the live rating.

The scripted controller is scheduled directly; it is not a sampled roster entry. The
[`EloRoster`](../src/boost_and_broadside/train/rl/roster.py) retains historical entries
rather than evicting the weakest; `league_size` only bounds the GPU-resident LRU policy
cache. Historical sampling is proportional to
`exp(-abs(opponent_elo - live_elo) / temperature)`, excluding the fixed random anchor.

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

Full scheduled `step_<N>.pt` checkpoints contain policy and optimizer state, return and
advantage scalers, average-policy state, rating/evaluation state, counters, and serialized
environment/model/training configuration. Saves are prepared asynchronously and written
through a temporary file before rename.

The checkpoint subsystem also maintains current average/best snapshots and unpruned
ladder checkpoints; [`checkpoint.py`](../src/boost_and_broadside/train/rl/checkpoint.py)
defines the current filenames. (The included reference-run directory retains
`recent_avg.pt` from an older naming convention.)

The refractive-field observation contract adds encoder inputs and a local-index auxiliary
target. Radius is shared by ship and field tokens and normalized by half the shorter world
dimension. New checkpoint payloads carry `observation_schema=refractive_fields_v2`.
Earlier schemas have no faithful weight-only migration because their radius semantics
differ, so they are rejected clearly and retraining is required.

The schema pins the observation contract, not the layers above it, so weights written by
a different model architecture pass that check and fail on load instead. Those failures
name the offending file and the differing parameter keys. Where the caller asked for a
specific file — resume, `--pretrain_from`, an explicit evaluation checkpoint — the failure
is fatal. Where the file is only a league opponent or ladder anchor, typically a roster
restored from an earlier run, the entry is retired for the remainder of the run with a
warning and training continues against the rest of the pool.

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
- [`test_checkpoint.py`](../tests/train/test_checkpoint.py) for save/resume state;
- [`test_bradley_terry.py`](../tests/train/test_bradley_terry.py) for calibrated fitting and
  uncertainty.
