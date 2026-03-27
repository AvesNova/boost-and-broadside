# Boost and Broadside

A GPU-accelerated multi-agent RL environment where teams of ships compete in 2D naval dogfights. The current focus is an **MVP RL pipeline** training a shared recurrent policy via self-play PPO.

**Game Details**: See [docs/game_design.md](docs/game_design.md) for rules, physics, and action space.

## Quick Start

```bash
# Install dependencies
uv sync

# Train with W&B logging and checkpointing
uv run --no-sync main.py

# Play against a fresh AI (WASD + Shift for sharp turns, Space to shoot)
uv run --no-sync main.py --mode play

# Watch a checkpointed agent play itself
uv run --no-sync main.py --mode watch --checkpoint checkpoints/checkpoint_000500.pt
```

## Modes

| Flag | Description |
|---|---|
| `--mode train` (default) | PPO self-play training with async W&B logging and periodic checkpoints |
| `--mode play` | Human controls ship 0; AI (fresh policy) controls ships 1–7 |
| `--mode watch --checkpoint <path>` | Load a checkpoint and watch self-play at 60fps |

## Project Structure

```
src/boost_and_broadside/
├── config.py          # Frozen dataclass configs (ShipConfig, EnvConfig, ModelConfig, ...)
├── constants.py       # Action space constants and slices
├── env/
│   ├── state.py       # TensorState — mutable GPU state for B parallel envs
│   ├── physics.py     # Pure physics functions (kinematics, shooting, collisions)
│   ├── env.py         # TensorEnv — vectorized physics engine, no rewards
│   ├── rewards.py     # Modular reward components, REWARD_COMPONENT_NAMES, compute_per_component_rewards
│   └── wrapper.py     # MVPEnvWrapper — obs construction, reward orchestration, auto-reset
├── models/mvp/
│   ├── encoder.py     # ShipEncoder — Fourier position + symlog vel + team embed
│   ├── attention.py   # RelationalSelfAttention — MHSA with alive masking
│   └── policy.py      # MVPPolicy — encoder → attention → per-ship GRU → action/distributional-value heads
├── modes/
│   └── interactive.py # run_play_mode, run_watch_mode — shared render loop + keyboard input
├── train/rl/
│   ├── buffer.py      # RolloutBuffer — pre-allocated GAE buffer, (T,B,N,K) shapes, symlog/twohot utilities
│   └── ppo.py         # PPOTrainer — rollout collection, GAE, PPO epochs, async W&B, checkpointing
└── ui/
    └── renderer.py    # GameRenderer — pygame renderer reading TensorState directly

tests/
├── env/               # physics, rewards, env + wrapper tests
├── models/            # encoder, attention, policy tests
└── train/             # buffer, GAE, minibatch iterator tests

old_code/              # Archived prior codebase (world model / Hydra era)
```

## Architecture

### Environment

- **`TensorEnv`**: Pure physics, no rewards. `step()` returns `(dones, truncated)` only.
- **`MVPEnvWrapper`**: Owns observation construction, reward computation, and auto-reset. Snapshots pre-step state to compute per-ship rewards.
- **Ships per environment** set by `EnvConfig.num_ships`; all run the same shared policy (self-play).

### Observations (per ship)

| Feature | Encoding | Dims |
|---|---|---|
| Position (x, y) | Fourier (log-spaced freqs) | `4 * n_fourier_freqs` |
| Velocity (vx, vy) | Symlog | 2 |
| Attitude (cos, sin) | Raw | 2 |
| Angular velocity | Symlog | 1 |
| Health, power, cooldown | Normalized scalars | 3 |
| Team | Learned embedding | 4 |
| Alive flag | Float | 1 |
| Previous action | One-hot (power\|turn\|shoot) | 12 |

### Policy (`MVPPolicy`)

```
obs dict → ShipEncoder → RelationalSelfAttention (alive masking) → per-ship GRU → action/value heads
```

- **Action space**: Factored categorical — 3 power × 7 turn × 2 shoot (joint log-prob = sum of three).
- **GRU hidden state**: `(1, B*N, D)`, `batch_first=False`.
- **Value**: K=12 distributional heads per ship (one per reward component); 255 categorical bins in double-symlog space `[-20, 20]`, trained with cross-entropy on twohot targets (DreamerV3-style).

### Rewards (decomposed critic + lambda aggregation)

Each ship has K=12 independent value heads, each predicting only events that happen **to that ship**. Zero-sum accounting is deferred to advantage aggregation.

| Component | Allied λ | Enemy λ | Signal |
|---|---|---|---|
| damage | +1 | −1 | damage taken this step |
| death | +1 | −1 | ship destroyed |
| victory | +1 | −1 | game outcome |
| exposure | +1 | −1 | in enemy crosshairs |
| facing | +1 | 0 | turning toward nearest enemy |
| turn_rate | +1 | 0 | angular velocity reward |
| proximity | +1 | 0 | closing distance to enemy |
| closing_speed | +1 | 0 | radial closing speed |
| positioning | +1 | 0 | offensive alignment |
| power_range | +1 | 0 | power in useful range |
| speed_range | +1 | 0 | speed in useful range |
| shoot_quality | +1 | 0 | shot quality |

**Advantage aggregation**: For ship _i_, `A_i = Σ_j Σ_k λ(same_team[i,j], k) · A_j^(k)` where lambdas are relative to the querying ship's team. Allied component advantages add; outcome component advantages subtract for enemies. This replaces explicit team-1 reward negation.

## Configuration

All hyperparameters live in [main.py](main.py) as frozen dataclasses — no config files. To experiment, edit `main.py` directly.

```python
ship_config   = ShipConfig()                    # physics defaults
env_config    = EnvConfig(num_ships=8, ...)
model_config  = ModelConfig(d_model=128, n_heads=4, n_fourier_freqs=8, num_value_components=12, num_bins=255, bin_lo=-20.0, bin_hi=20.0)
reward_config = RewardConfig(damage_weight=0.01, ...)
train_config  = TrainConfig(num_envs=128, num_steps=512, checkpoint_interval=500, ...)
```

## Logging

Training logs metrics to W&B asynchronously (background thread) to avoid GPU sync on the hot path. Run `wandb login` before training. All configs are serialized into the W&B run config for reproducibility.

## Checkpointing

Checkpoints are saved to `checkpoints/checkpoint_{update:06d}.pt` every `checkpoint_interval` updates. Each file contains policy weights, optimizer state, update index, and global step count.

## Development

- **Style Guide**: [STYLE_GUIDE.md](STYLE_GUIDE.md)
- **Tests**: 74 tests across env, models, and train modules.

```bash
uv run pytest
```
