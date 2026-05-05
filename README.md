# Boost and Broadside

GPU-accelerated multi-agent RL environment. Teams of ships compete in 2D dogfights. Trained via PPO with ELO-rated league play, a decomposed distributional critic, and a composable observation pipeline.

## Quick Start

```bash
uv sync

# RL training from scratch (W&B + checkpoints)
uv run main.py --mode rl

# Watch: human (WASD + Shift + Space) vs latest checkpoint
uv run main.py --mode watch

# Crash-test a config change without W&B
uv run main.py --mode rl --smoke
```

## Modes

| Mode | Description |
|---|---|
| `rl` | PPO RL training. Optional `--pretrain_from <path.pt>` to warm-start. |
| `rl_obstacles` | RL with dynamic orbiting obstacles. |
| `bc` | Behavior cloning pretraining from the scripted agent. |
| `bc_warmstart` | BC for 50M steps, then immediately switch to RL. |
| `watch` | Render live gameplay at 60fps. `--fast-cache` skips obstacle convergence animation. |
| `collect_stats` | Run a matchup and print win-rate statistics. |
| `elo_stats` | ELO tournament across scripted agents and/or checkpoints. |

All modes accept `--smoke` for a tiny crash-test run (4 envs, no W&B, exits after a few updates).

Agent specs for `--team0` / `--team1`: `null` (human), `random`, `scripted`, `latest`, or a path to a `.pt` checkpoint.

## Project Structure

```
runs/                       # Hyperparameter profiles (one file per experiment)
  shared.py                 # MODEL_CONFIG, OBS_CONFIG, REWARDS, SHIP_CONFIG
  rl.py                     # Primary RL run
  rl_obstacles.py           # RL with obstacles
  bc.py / bc_warmstart.py   # Pretraining profiles
  rl_hpc.py                 # High-core-count server profile

src/boost_and_broadside/
  config/
    core.py                 # Frozen dataclasses: ShipConfig, EnvConfig, ModelConfig, RewardConfig
    training.py             # TrainConfig, ScaleConfig, ObstacleCacheConfig
    schedule.py             # Schedule primitives: constant, linear, stepped, exponential, join
    obs_spec.py             # ObsConfig + transform blocks (Fourier, Symlog, OneHot, ...)
  env/
    state.py                # TensorState — mutable GPU state for B parallel envs
    physics.py              # Pure physics (kinematics, shooting, collisions)
    env.py                  # TensorEnv — vectorized physics, no rewards
    rewards.py              # Decomposed reward components (19 components, K=19 critic heads)
    wrapper.py              # MVPEnvWrapper — obs, rewards, auto-reset
    obstacle_physics.py     # Harmonic gravity + PBD obstacle dynamics
    obstacle_cache.py       # Pre-converged obstacle map cache
  models/mvp/
    encoder.py              # ShipEncoder — generic obs pipeline driven by ObsConfig
    attention.py            # TransformerBlock (MHSA + GatedMLP) with alive masking
    griffin.py              # YemongBlock: TransformerBlock + Griffin RG-LRU temporal block
    policy.py               # MVPPolicy — encoder → YemongBlocks → action/value heads
  agents/                   # Scripted agents (stochastic_scripted, jouster, boom_zoom, ...)
  modes/
    agent_factory.py        # Resolve agent specs (null/random/scripted/latest/path.pt)
    interactive.py          # run_watch_mode + keyboard input
    collect.py              # run_collect_stats_mode
    elo_stats.py            # run_elo_stats_mode
  train/rl/
    buffer.py               # RolloutBuffer — pre-allocated GAE buffer, twohot utilities
    ppo.py                  # PPOTrainer — rollout, GAE, PPO epochs, W&B, checkpointing
    roster.py               # EloRoster — ELO-rated league pool with proximity-weighted sampling
    sigreg.py               # Sigma regularization loss
  ui/
    renderer.py             # GameRenderer — pygame renderer reading TensorState directly

tests/                      # 140 tests across env, models, and train
```

## Architecture

### Policy (`MVPPolicy`)

The actor-critic policy uses a shared trunk to process both ships and obstacles, using a spatial-temporal architecture based on YemongBlocks.

```text
obs dict → ShipEncoder → N+M YemongBlocks → slice [:N] ships → ActionHead + NextStateHead + TeamPMA + ValueHead
```

- **ShipEncoder**: Encodes each entity (ship or obstacle) into a `d_model`-dimensional token. It acts as a generic pipeline driven by `ObsConfig` — it iterates over feature specs, applies transform chains, concatenates the resulting features, and passes them through a 2-layer MLP (with RMSNorm and GELU) to project to `d_model`.
- **YemongBlock**: The core backbone, which combines spatial and temporal processing:
  - **Spatial Block (`TransformerBlock`)**: A standard Multi-Head Self-Attention (MHSA) layer followed by a GatedMLP. Ships and obstacles attend to each other within the same timestep (cross-entity attention), with dead entities dynamically masked out.
  - **Temporal Block (`GriffinTemporalBlock`)**: Applied independently per entity across time. It uses a Real-Gated Linear Recurrent Unit (RG-LRU) with learnable decay rates, replacing traditional GRUs. The architecture follows the Griffin paper: `norm → (linear₁ → causal_conv → RG-LRU) × GeLU(linear₂) → linear_out → 1st residual → GatedMLP(norm) → 2nd residual`. The causal depthwise convolution uses a stored buffer of the last `kernel-1` inputs so that step-by-step rollout (T=1) and PPO re-evaluation (T=128) use an identical causal context.
  - *Note*: Obstacle tokens participate in spatial attention and carry temporal hidden state, but they are sliced out before the policy heads (they receive no action, value, or auxiliary heads).
- **ActionHead**: A 2-layer MLP producing factored categorical logits for 3 power × 7 turn × 2 shoot actions. The joint log-prob is the sum of the three sub-actions.
- **NextStateHead**: An auxiliary 2-layer MLP that predicts normalized dynamic features of the next state (position, velocity, heading, angular velocity, health, power, cooldown, and an alive logit) for each ship. This provides a dense dynamics-learning signal.
- **TeamPMA**: Pooling by Multi-head Attention. For each team, a learned seed attends over the embeddings of all alive ships on that team. The pooled team embedding is then broadcast back to each ship on that team, ensuring the value head has a global team context.
- **ValueHead**: A 2-layer MLP producing K=19 distributional heads (one per reward component). It uses 255 categorical bins in double-symlog space `[-20, 20]`, trained with cross-entropy on twohot targets (DreamerV3-style). A `ReturnScaler` normalizes between symlog-reward space (GAE) and the value head's input/output range.

**Hidden state**: `(n_layers, B*(N+M), CONV_KERNEL * D)` — The RG-LRU recurrent state and the causal convolution buffer are packed together.

### Observations (`ObsConfig`)

Features are specified as `(source, transform_chain)` pairs. Each transform block is a frozen dataclass:

| Block | Type | Output dims |
|---|---|---|
| `Fourier(n, period)` | S→V | 2n |
| `FourierAngle(n)` | V→V | 2n |
| `SymlogVec()` | V→V | 2 |
| `VecMag()` | V→S | 1 |
| `Symlog()` | S→S | 1 |
| `Normalize(scale)` | S→S | 1 |
| `Clamp(lo, hi)` | S→S | 1 |
| `Bucketize(n)` | S→S | 1 (int) |
| `OneHot(n)` | S→V | n |
| `AsFloat()` | S→S | 1 |

Default config (`OBS_CONFIG`, raw_dim=92): Fourier position (10 freqs, period=world size), FourierAngle velocity direction (10 freqs), symlog velocity speed, symlog angular velocity, one-hot health (11 classes), normalized power, clamped cooldown, team one-hot, alive flag, previous action one-hots (power/turn/shoot), normalized radius. The Fourier period matches the map boundary exactly so toroidal position wraps correctly.

### Rewards (decomposed critic)

19 independent reward components, each with its own value head. Zero-sum accounting is handled at PPO update time via a lambda aggregation matrix — no explicit reward negation for enemies.

**Global (lambda-aggregated across ships):** `ally_damage`, `enemy_damage`, `ally_death`, `enemy_death`, `ally_win`, `enemy_win`, `facing`, `closing_speed`, `shoot_quality`

**Local (self-only, lambda=0 for all others):** `kill_shot`, `kill_assist`, `damage_taken`, `damage_dealt_enemy`, `damage_dealt_ally`, `death`, `obstacle_death`, `obstacle_proximity`, `obstacle_closing_speed`, `obstacle_tti`

### League Play

`EloRoster` maintains a pool of rated agents (past checkpoints, avg-policy, scripted). Opponents are sampled by ELO proximity (`exp(-|elo_i - training_elo| / temperature)`). New checkpoints are added at ELO milestones; weakest are pruned when the roster exceeds capacity. ELO ratings are zero-sum and updated after each eval batch.

## Configuration

All hyperparameters live in `runs/`. The shared constants (`MODEL_CONFIG`, `OBS_CONFIG`, `REWARDS`, `SHIP_CONFIG`) are in `runs/shared.py`. Each run profile imports from shared and overrides only what differs.

Time-varying parameters (learning rate, loss coefficients, opponent fractions) are expressed as schedules using primitives in `config/schedule.py`:

```python
learning_rate = linear((0, 1e-7), (5_000_000, 3e-4))   # warmup
scripted_fraction = stepped((0, 0.5), (50_000_000, 0.3)) # step at 50M
```

## Checkpoints

Saved to `checkpoints/<run-name>/step_<N>.pt` on a configurable interval. Each file contains policy weights, optimizer state, ELO roster, return scaler, and the serialized `ObsConfig` (so the correct observation pipeline is always reconstructed on load).

## Logging

W&B logging runs asynchronously in a background thread. All configs (model, obs, rewards, schedule snapshots) are serialized into the W&B run config. Disable with `--smoke`.

## Development

```bash
uv run pytest          # 140 tests
uv run pytest -x -q    # stop on first failure
```

See [STYLE_GUIDE.md](STYLE_GUIDE.md) for code conventions.
