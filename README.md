# Boost and Broadside

A high-performance codebase for **Boost and Broadside**, a compiled competitive multi-agent environment where teams of ships compete in 2D dogfights.

Now featuring a **GPU-Accelerated Environment (`env2`)** capable of simulating thousands of games in parallel for massive data collection.

**Game Details**: See [docs/game_design.md](docs/game_design.md) for rules, physics, and action space.

## Quick Start

### Installation

This project uses `uv` for dependency management.

```powershell
# Install dependencies
uv sync
```

### Playing the Game

Run the game in play mode to watch agents compete or play yourself.

```powershell
# Watch a match (BC Agent vs Scripted Agent)
uv run main.py mode=play team1=most_recent_bc team2=scripted

uv run main.py mode=play team1=most_recent_world_model team2=most_recent_world_model

# Play as a human (Use Arrow Keys + Space to shoot)
uv run main.py mode=play human_player=true
```

## Project Structure

- **`src/boost_and_broadside/`**: Main namespaced package containing all core logic.
    - **`agents/`**: Agent implementations (Scripted, Transformer, World Model).
    - **`core/`**: Core types, constants, and utilities.
    - **`env2/`**: Vectorized game environment.
    - **`train/`**: Training pipelines for World Model and RL.
    - **`modes/`**: Entry points for different execution modes (play, collect, train).
- **`tools/`**: Utility scripts for data inspection and debugging.
- **`configs/`**: Hydra configuration files.
- **`tests/`**: Unit and integration tests.

## World Model Architecture (MambaBB)

The core of the system is a **Factorized World Model (MambaBB)** that separates temporal and spatial processing for efficiency and performance.

### Architecture Highlights
- **Backbone**: `Mamba2` (State Space Model) handles temporal mixing along the time axis.
- **Spatial Layer**: `RelationalAttention` handles mixing between ships (spatial axis) at each timestep.
- **Physics Trunk**: A shared relational encoder computes analytic edge features (distance, closing speed, etc.) to bias the attention.
- **Factorization**: The model alternates between Mamba (Time) and Attention (Space) blocks.

### Design Principles
- **Continuous Stream**: Data is trained as a continuous stream of tokens with specialized `seq_idx` handling for resets, avoiding padding.
- **Strictly Optimized**: Utilizes hardware-optimized `mamba_ssm` kernels.
- **Causal**: Predicts the next state autoregressively.

### Training Objectives
- **Action Prediction ($S_t \to A_t$)**: Cross-entropy loss on discrete action indices (Power, Turn, Shoot).
- **State Prediction ($S_t, A_t \to S_{t+1}$)**: MSE loss on next-state reconstruction (Residual Delta).


## Data Structure

The collected training data is stored in **HDF5** format (`aggregated_data.h5`) for efficiency and scalability.

### Datasets (Root Group)

All datasets are aligned along the first dimension (Total Timesteps), except for `episode_lengths`.

| Dataset Name | Shape | Description |
| :--- | :--- | :--- |
| **`tokens`** | `(N, MaxShips, TokenDim)` | The observation tokens for all ships. |
| **`actions`** | `(N, MaxShips, NumActions)` | The actions taken by each ship. |
| **`expert_actions`** | `(N, MaxShips, NumActions)` | The ground-truth "optimal" actions produced by the scripted expert. |
| **`action_masks`** | `(N, MaxShips, NumActions)` | Boolean mask indicating valid actions. |
| **`rewards`** | `(N, MaxShips)` | The reward received by each ship. |
| **`returns`** | `(N, MaxShips)` | Precomputed discounted returns (Reward-to-go). |
| **`episode_ids`** | `(N,)` | ID of the episode each timestep belongs to. |
| **`agent_skills`** | `(N, MaxShips)` | Skill level of the ship (0.0=Novice, 1.0=Expert). |
| **`episode_lengths`** | `(NumEpisodes,)` | Length of each episode in the dataset. |

### Metadata (Attributes)

Global metadata is stored as HDF5 attributes on the root group:

- `num_episodes`: Total number of episodes.
- `total_timesteps`: Total number of timesteps.
- `total_sim_time`: Total simulation time in seconds.
- `max_ships`: Maximum number of ships per team.
- `token_dim`: Dimension of each token.
- `num_actions`: Number of possible actions.
## Workflows

### 1. Data Collection

Generate training data using scripted agents.

```powershell
# Collect data (CPU, single worker)
uv run main.py mode=collect

# Massive GPU Data Collection (Recommended for pretraining)
# Generates HDF5 data using the vectorized environment.
# Configured via collect.massive and collect sections in config.yaml.
uv run main.py mode=collect_massive
```

### 2. Training

Train models using the collected data.

```powershell
# Train World Model
uv run main.py mode=train_wm

# Train RL Agent (PPO) using World Model backbone
uv run main.py mode=train train.run_rl=true train.rl.policy_type=world_model
```

### 3. Evaluation

Evaluate model performance.

```powershell
# Evaluate World Model rollouts
uv run main.py mode=eval_wm
```

### 4. Combined Pipelines

Run multiple steps in a single command.

```powershell
# Collect data then train World Model
uv run main.py mode=train train.run_collect=true train.run_world_model=true

# Quick test run
uv run main.py mode=train train.run_collect=true train.run_world_model=true --config-name config_test
```

## Tools & Debugging

### Learning Rate Range Test

Use this workflow to find the optimal learning rate.

1. **Run the Range Test**:
    ```powershell
    # Run training for 1 epoch with exponential LR scheduling
    uv run main.py mode=train_wm world_model.scheduler.type=exponential_range_test world_model.epochs=1
    ```

2. **Analyze Results**:
    ```powershell
    # Find steepest descent point (limit max_lr to avoid instability)
    uv run python tools/analyze_lr_test.py --latest --max_lr 5e-4
    ```

3. **Turn It Off**:
    The range test is only active when `world_model.scheduler.type` is set to `linear_range_test` or `exponential_range_test`.
    To return to normal training, simply omit the override (defaults to `warmup_constant`).
    ```powershell
    # Normal training (defaults to warmup_constant)
    uv run main.py mode=train_wm
    ```

We provide tools in the `tools/` directory to help with development.

```powershell
# Inspect collected data files
uv run python tools/inspect_data.py data/bc_pretraining/latest/aggregated_data.h5

# Verify data loading logic
uv run python tools/verify_data_loading.py

# Generates 2D projections (PCA, PaCMAP) of world model embeddings
uv run python tools/viz_latent_space.py --latest
```

## Logging & Metrics

The training script logs various metrics to help diagnose model performance.

### Training Metrics (Micro-steps)
- **`loss`**: Total combined loss (State + Action).
- **`state_loss`**: MSE loss for state reconstruction ($S_{t+1}$).
- **`action_loss`**: Cross-entropy loss for action prediction ($A_t$).
- **`error_power/turn/shoot`**: Classification error rate for each action component on the training batch.

### Validation Metrics (Per Epoch)
- **`val_loss`**: Total loss on the validation set (Teacher Forcing).
- **`val_error_power/turn/shoot`**: Classification error rate on validation set.
- **`val_rollout_mse_state`**: **Autoregressive** MSE. We run a 50-step closed-loop simulation (feeding model predictions back in) and compare the trajectory to the ground truth. This detects drift that standard teacher-forcing validation misses.
- **`val_rollout_mse_step`**: (Heatmap) The rollout error broken down by step (0-50). Visualized in WandB as an Epoch vs Step heatmap to show stability over time.
- **`norm_latent`**: L2 norm of the latent embeddings. useful for detecting collapse or explosion.
- **`prob_*`**: Average predicted probability (confidence) of the ground truth action. Higher is better.
- **`entropy_*`**: Entropy of the action distributions. Higher entropy means the model is less confident (closer to random).

## Configuration

The project uses [Hydra](https://hydra.cc/) for configuration. Key config files are in `configs/`. Do not modify source code for parameters; use command line overrides or edit `configs/config.yaml`.

**Common Overrides:**
- `cw` (World Size): `environment.world_size=[2000,2000]`
- `max_ships`: `environment.max_ships=16`
- `render`: `collect.render_mode='human'`

## Development

- **Style Guide**: Please refer to [STYLE_GUIDE.md](STYLE_GUIDE.md) for coding standards.
- **Tests**: Run `uv run pytest` to ensure everything is working.

```powershell
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/components/test_ship.py
```
