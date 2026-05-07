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
obs dict → ShipEncoder → n_layers × YemongBlock → slice [:N] ships → ActionHead + NextStateHead + TeamPMA + ValueHead
```

- **ShipEncoder**: Encodes each entity (ship or obstacle) into a `d_model`-dimensional token. It acts as a generic pipeline driven by `ObsConfig` — it iterates over feature specs, applies transform chains, concatenates the resulting features, and passes them through a 2-layer MLP (`Linear → RMSNorm → GELU → Linear → RMSNorm`) to project to `d_model`.
- **YemongBlock**: The core backbone, which combines spatial and temporal processing:
  - **Spatial Block (`TransformerBlock`)**: Pre-norm transformer: `RMSNorm → MHSA → Residual → RMSNorm → GatedMLP → Residual`. The GatedMLP is SwiGLU-style (`down_proj(gelu(gate_proj(x)) × up_proj(x))`, 4× hidden expansion). Ships and obstacles attend to each other within the same timestep (cross-entity attention), with dead entities masked out as keys.
  - **Temporal Block (`GriffinTemporalBlock`)**: Applied independently per entity across time. It uses a Real-Gated Linear Recurrent Unit (RG-LRU) with learnable decay rates, replacing traditional GRUs. The architecture follows the Griffin paper: `norm → (linear₁ → causal_conv → RG-LRU) × GeLU(linear₂) → linear_out → 1st residual → GatedMLP(norm) → 2nd residual`. The causal depthwise convolution uses a stored buffer of the last `kernel-1` inputs so that step-by-step rollout (T=1) and PPO re-evaluation (T=128) use an identical causal context.
  - *Note*: Obstacle tokens participate in spatial attention and carry temporal hidden state, but they are sliced out before the policy heads (they receive no action, value, or auxiliary heads).
- **ActionHead**: A 2-layer MLP producing factored categorical logits for 3 power × 7 turn × 2 shoot actions. The joint log-prob is the sum of the three sub-actions.
- **NextStateHead**: An auxiliary MLP (`Linear → RMSNorm → GELU → Linear`, output dim=10) that predicts **deltas and phase shifts** for the next ship state, providing a dense dynamics-learning signal. Output layout (`AUX_PRED_DIM=10`):
  - `[0:2]` **pos_phase** — `(Δφ_x, Δφ_y)`: phase shifts for the toroidal position Fourier encoding.
  - `[2:4]` **vel_delta** — `(Δvel_x, Δvel_y)`: additive delta for velocity.
  - `[4]` **att_phase** — `Δφ_att`: phase shift for the heading Fourier encoding.
  - `[5]` **ang_vel_delta** — additive delta for symlog angular velocity.
  - `[6:9]` **health/power/cooldown phase** — phase shifts for the quarter-wave Fourier encodings `[sin(π/2·x), cos(π/2·x)]`.
  - `[9]` **alive logit** — BCE loss against the ground-truth alive flag.

  Fourier phase shifts are applied via a rotation matrix `(sin(φ+Δφ), cos(φ+Δφ))`, guaranteeing `sin²+cos²=1` on the predicted features. Continuous outputs use weighted MSE loss; the alive logit uses BCE.
- **TeamPMA**: Pooling by Multi-head Attention. For each team, a learned seed attends over the embeddings of all alive ships on that team. The pooled team embedding is then broadcast back to each ship on that team, ensuring the value head has a global team context.
- **ValueHead**: A 2-layer MLP producing K=19 scalar heads (one per reward component), trained with MSE. Outputs are in a normalized space maintained by `ReturnScaler`, which tracks a per-component EMA of the p5/p95 percentiles of the GAE returns and maps symlog-reward space to approximately `[-1, 1]` per component, keeping loss magnitudes comparable across components with very different natural scales.

**Hidden state**: `(n_layers, B*(N+M), CONV_KERNEL * D)` — The RG-LRU recurrent state and the causal convolution buffer are packed together.

### Observations (`ObsConfig`)

Features are specified as `(source, transform_chain)` pairs in `runs/shared.py`. The encoder iterates the list in order, applies each chain, concatenates all outputs into a raw feature vector, and projects it to `d_model` via the ShipEncoder MLP.

#### Feature table (`OBS_CONFIG`, raw_dim = 105)

| Feature | Source | Transform chain | Dims | Notes |
|---|---|---|---|---|
| `pos_x` | `pos[0]` | `Fourier(10, world_w)` | 20 | Base-2 power freqs; period = map width so toroidal wrap is exact |
| `pos_y` | `pos[1]` | `Fourier(10, world_h)` | 20 | Same for y |
| `att` | `att` | `FourierAngle(10)` | 20 | Heading as 2D unit vector → Fourier angle encoding |
| `vel_dir` | `vel` | `FourierAngle(10)` | 20 | Velocity direction angle; magnitude encoded separately |
| `vel_speed` | `vel` | `VecMag() → Symlog()` | 1 | ‖vel‖ compressed with symlog |
| `ang_vel` | `ang_vel` | `Symlog()` | 1 | Unbounded scalar; symlog keeps gradient well-scaled |
| `health` | `health` | `Normalize(100) → QuarterWaveFourier()` | 2 | Normalized to [0,1], then `[sin(π/2·x), cos(π/2·x)]` |
| `power` | `power` | `Normalize(100) → QuarterWaveFourier()` | 2 | Same encoding as health |
| `cooldown` | `cooldown` | `Normalize(0.1) → Clamp(0,1) → QuarterWaveFourier()` | 2 | Clamped before encoding; 0.1 s max cooldown |
| `team` | `team_id` | `OneHot(3)` | 3 | 0=team A, 1=team B, 2=obstacle |
| `alive` | `alive` | `AsFloat()` | 1 | Boolean cast to 0.0/1.0 |
| `act_power` | `prev_power` | `OneHot(3)` | 3 | Previous power action |
| `act_turn` | `prev_turn` | `OneHot(7)` | 7 | Previous turn action |
| `act_shoot` | `prev_shoot` | `OneHot(2)` | 2 | Previous shoot action |
| `radius` | `radius` | `Normalize(40.0)` | 1 | Entity collision radius normalized by 40 px |

#### Transform block reference

| Block | Type | Output dims | Description |
|---|---|---|---|
| `Fourier(n, period)` | S→V | 2n | Base-2 power freqs: `[sin(2π/T · 2^k · x), cos(...)]` for k=0..n-1 |
| `FourierAngle(n)` | V→V | 2n | `atan2(y,x)` of a 2D vector, then same Fourier encoding as above |
| `QuarterWaveFourier()` | S→V | 2 | `[sin(π/2·x), cos(π/2·x)]` for x∈[0,1]; preserves sin²+cos²=1 |
| `SymlogVec()` | V→V | 2 | Scales a 2D vector so ‖out‖=symlog(‖in‖), direction preserved |
| `VecMag()` | V→S | 1 | Euclidean norm ‖v‖ |
| `Symlog()` | S→S | 1 | `sign(x)·log(1+|x|)` — compresses unbounded scalars |
| `Normalize(scale)` | S→S | 1 | `x / scale` |
| `Clamp(lo, hi)` | S→S | 1 | `clamp(x, lo, hi)` |
| `Bucketize(n)` | S→S | 1 (int) | Maps [0,1] → integer bucket in [0,n-1]; usually followed by `OneHot` |
| `OneHot(n)` | S→V | n | Integer → one-hot float vector |
| `AsFloat()` | S→S | 1 | Bool/int cast to float |

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
