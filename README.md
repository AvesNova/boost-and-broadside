# Boost and Broadside

A high-performance codebase for **Boost and Broadside**, a compiled competitive multi-agent environment where teams of ships compete in 2D dogfights.

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

- **`src/`**: Source code for the environment, agents, and training logic.
    - **`agents/`**: Agent implementations (Scripted, Transformer, World Model).
    - **`env/`**: The core game environment (Ship dynamics, physics, state).
    - **`train/`**: Training pipelines for BC and RL.
    - **`modes/`**: Entry points for different execution modes (play, collect, train).
- **`tools/`**: Utility scripts for data inspection and debugging.
- **`configs/`**: Hydra configuration files.
- **`tests/`**: Unit and integration tests.

## World Model Architecture

The core of the system is an **Interleaved World Model** that predicts the future state of the game.

For a detailed breakdown of the model and training pipeline, see [docs/world_model_architecture.md](docs/world_model_architecture.md).

### Sequence Structure
The model processes an interleaved sequence of **State ($S$)** and **Action ($A$)** blocks. Each block contains $N$ tokens, where $N$ is the number of ships (default 8).

$$ S_0 \xrightarrow{\text{predict}} A_0 \xrightarrow{\text{predict}} S_1 \xrightarrow{\text{predict}} A_1 \dots $$

- **State Tokens ($S_t$)**: Continuous vector encodings of ship states (position, velocity, health).
- **Action Tokens ($A_t$)**: Embeddings of discrete actions (Power, Turn, Shoot).

### Factorized Attention
To handle the multi-agent temporal structure efficiently, we use a factorized attention mechanism with alternating layers:

1.  **Temporal Attention**: Each ship attends only to its own history.
    - Causal masking ensures no peeking into the future.
    - Uses **Rotary Position Embeddings (RoPE)** for relative time awareness.
2.  **Spatial Attention**: Ships attend to each other within a local temporal window.
    - **Window**: Each token attends to all tokens in its **Current Block** and the **Previous Block**.
        - $A_t$ attends to $\{S_t, A_t\}$
        - $S_t$ attends to $\{A_{t-1}, S_t\}$
    - This allows interactions (like shooting) to propagate correctly while maintaining causality.

### Training Objectives
- **Action Prediction ($S_t \to A_t$)**: Cross-entropy loss on discrete action indices.
- **State Prediction ($A_t \to S_{t+1}$)**: MSE loss on next-state reconstruction.

## Data Structure

The collected training data is stored in **HDF5** format (`aggregated_data.h5`) for efficiency and scalability.

### Datasets (Root Group)

All datasets are aligned along the first dimension (Total Timesteps), except for `episode_lengths`.

| Dataset Name | Shape | Description |
| :--- | :--- | :--- |
| **`tokens`** | `(N, MaxShips, TokenDim)` | The observation tokens for all ships. |
| **`actions`** | `(N, MaxShips, NumActions)` | The actions taken by each ship. |
| **`action_masks`** | `(N, MaxShips, NumActions)` | Boolean mask indicating valid actions. |
| **`rewards`** | `(N,)` | The reward received at each timestep. |
| **`returns`** | `(N,)` | Precomputed discounted returns (GAE/Reward-to-go). |
| **`episode_ids`** | `(N,)` | ID of the episode each timestep belongs to. |
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
# Collect data (single worker)
uv run main.py mode=collect

# Collect with multiple workers (faster)
uv run main.py mode=collect collect.num_workers=4

# Collect with custom config
uv run main.py mode=collect --config-name config_test
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
- **Tests**: Run `uv run pytest --color=no -rf --tb=line` to ensure everything is working.

```powershell
# Run all tests
uv run pytest --color=no -rf --tb=line

# Run specific test file
uv run pytest tests/components/test_ship.py --color=no -rf --tb=line
```
