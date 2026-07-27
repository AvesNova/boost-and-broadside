# Training system

Boost and Broadside trains a recurrent centralized team policy with clipped PPO. The
training loop combines scripted-opponent bootstrapping, self-play, a running-average
policy, frozen historical opponents, decomposed rewards, and continuous evaluation.

This page explains the current implementation and calls out landmark-run settings where
they matter. Exact results and post-hoc methodology are in [evaluation](evaluation.md).

## Landmark experiment at a glance

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
| ELO evaluation games per matchup slot | 512 |

The run logged 999,424,000 steps before finishing. Today's profiles have continued to
evolve, so use the artifact above—not current constants—for claims about that experiment.

## Recurrent PPO lifecycle

[`PPOTrainer`](../src/boost_and_broadside/train/rl/ppo.py) owns environment groups,
policies, rollout collection, generalized advantage estimation, update-time sequence
re-evaluation, evaluation, logging, and checkpoints.

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
also collect multiple fixed-width rollout shards into a host-backed logical batch. That
newer sharding design is documented in [memory optimization](engineering/memory-optimization.md)
and should not be retroactively attributed to the landmark run, whose exported config
predates the setting.

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
- optional SIGReg embedding regularization, disabled in the landmark configuration.

Advantages are scaled per component with running RMS statistics. Returns use a
per-component percentile scaler in symlog reward space, which keeps critic targets in a
stable range without forcing components with different natural scales into one value
head. The exact loss assembly and logging proxies live in
[`ppo.py`](../src/boost_and_broadside/train/rl/ppo.py).

## Reward decomposition

Rewards are emitted as named components by [`rewards.py`](../src/boost_and_broadside/env/rewards.py).
Each active component receives its own critic output and can have its own GAE gamma/lambda
horizon. The landmark combat policy activated these components:

| Component | Weight | Role |
|---|---:|---|
| `ally_win` | 4.0 | own-team terminal outcome |
| `enemy_win` | 4.0 | opponent outcome, aggregated with negative enemy lambda |
| `facing` | 0.1 | dense aim geometry |
| `closing_speed` | 0.1 | dense approach geometry |
| `shoot_quality` | 0.1 | firing opportunity quality |
| `kill_shot` | 1.0 | fatal-step credit proportional to damage on that step |
| `kill_assist` | 1.0 | cumulative-damage assist credit |
| `damage_taken` | 0.5 | local incoming damage accounting |
| `damage_dealt_enemy` | 0.5 | local hostile damage accounting |
| `damage_dealt_ally` | 0.5 | local friendly-fire accounting |
| `death` | 1.0 | local ship death |

Weights are normalized by their absolute sum, and the wrapper divides component rewards
by total ship count for team-size normalization. A lambda aggregation matrix then maps
local event signals to training targets:

- local components use diagonal/self-only credit;
- global outcome components aggregate across live teammates;
- selected enemy-perspective components use negative enemy coefficients to recover
  zero-sum outcome structure.

The implementation, not component names alone, defines credit semantics. In particular,
`kill_shot` is not winner-take-all when multiple ships deal damage on the fatal step;
credit is proportional to that step's damage. `kill_assist` is proportional to cumulative
episode damage.

Available obstacle and behavior-shaping components remain in the registry but had zero
weight in the landmark combat run. See [`runs/shared.py`](../runs/shared.py) for current
component horizons and schedules, and the preserved run config for the historical weights.

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

The scripted controller is scheduled directly; it is not currently a sampled roster
entry. The [`EloRoster`](../src/boost_and_broadside/train/rl/roster.py) retains historical
entries rather than evicting the weakest. `league_size` controls the GPU-resident LRU
policy cache. Historical sampling is proportional to
`exp(-abs(opponent_elo - live_elo) / temperature)`, excluding the fixed random anchor.

## Continuous rating and the frozen ladder

[`elo_eval.py`](../src/boost_and_broadside/train/rl/elo_eval.py) advances dedicated
evaluation environments alongside training. The current evaluator has five logical slots:

1. live vs fixed anchor;
2. live vs floating checkpoint;
3. live vs scripted controller;
4. live vs running-average policy;
5. floating checkpoint vs fixed anchor.

Two newest frozen anchors and information-weighted matchup allocation stabilize the online
scale. Ties count as half a win for online ELO. These live ratings support opponent
selection and training decisions, but they remain a filtered online estimate.

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
ladder checkpoints. Refer to [`checkpoint.py`](../src/boost_and_broadside/train/rl/checkpoint.py)
for current names. The included landmark directory contains `recent_avg.pt` from an older
convention; it is an artifact, not the current general filename.

W&B logging runs off the main training path. The landmark's sampled metric history,
configuration, summary, and run metadata are exported under
[`wandb_export/`](../checkpoints/resilient-resonance-682/wandb_export/) so the published
charts can be rebuilt without relying solely on a hosted dashboard.

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
