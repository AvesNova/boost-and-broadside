# Boost and Broadside

A GPU-accelerated multi-agent RL environment where teams of ships compete in 2D naval dogfights. The current focus is an **MVP RL pipeline** training a shared recurrent policy via self-play PPO.

**Game Details**: See [docs/game_design.md](docs/game_design.md) for rules, physics, and action space.

## Quick Start

```bash
# Install dependencies
uv sync

# Train (edit main.py to configure)
uv run  main.py
```

## Project Structure

```
src/boost_and_broadside/
├── config.py          # Frozen dataclass configs (ShipConfig, EnvConfig, ModelConfig, ...)
├── constants.py       # Action space constants and slices
├── env/
│   ├── state.py       # TensorState — mutable GPU state for B parallel envs
│   ├── physics.py     # Pure physics functions (kinematics, shooting, collisions)
│   ├── env.py         # TensorEnv — vectorized physics engine, no rewards
│   ├── rewards.py     # Modular reward components, zero-sum transform
│   └── wrapper.py     # MVPEnvWrapper — obs construction, reward orchestration, auto-reset
├── models/mvp/
│   ├── encoder.py     # ShipEncoder — Fourier position + symlog vel + team embed
│   ├── attention.py   # RelationalSelfAttention — MHSA with alive masking
│   └── policy.py      # MVPPolicy — encoder → attention → per-ship GRU → action/value heads
└── train/rl/
    ├── buffer.py      # RolloutBuffer — pre-allocated GAE buffer with recurrent hidden storage
    └── ppo.py         # PPOTrainer — rollout collection, GAE, PPO epochs, async wandb logging

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
- **8 ships, 4v4**, all running the same shared policy (zero-sum self-play).

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
- **Value**: scalar per ship; GAE computed per-ship.

### Rewards (zero-sum)

| Component | Signal |
|---|---|
| Damage | +reward for damage dealt, -reward for damage taken |
| Kill / Death | +kill\_weight per kill, -death\_weight per death |
| Victory | ±victory\_weight at episode end |
| Positioning | Offensive alignment minus defensive exposure, distance-weighted |

Team-1 rewards are negated after summing to enforce zero-sum self-play.

## Configuration

All hyperparameters live in [main.py](main.py) as frozen dataclasses — no config files, no CLI flags. To experiment, edit `main.py` directly.

```python
ship_config   = ShipConfig()                    # physics defaults
env_config    = EnvConfig(num_ships=8, ...)
model_config  = ModelConfig(d_model=128, n_heads=4, n_fourier_freqs=8)
reward_config = RewardConfig(damage_weight=0.01, ...)
train_config  = TrainConfig(num_envs=128, num_steps=512, ...)
```

## Logging

Set `use_wandb=True` in `main.py` and run `wandb login`. Logging runs in a background thread to avoid GPU sync on the hot path.

## Development

- **Style Guide**: [STYLE_GUIDE.md](STYLE_GUIDE.md)
- **Tests**: 71 tests across env, models, and train modules.

```bash
uv run pytest
```
