# Boost and Broadside

A high-performance codebase for **Boost and Broadside**, a compiled competitive multi-agent environment where teams of ships compete in 2D dogfights.

This project features a **GPU-Accelerated Environment (`env2`)** capable of simulating thousands of games in parallel for massive data collection, and a state-of-the-art **World Model (Yemong)** based on **Mamba2** and **Relational Attention**.

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
# Watch a match (Scripted vs Scripted)
uv run main.py mode=play team1=scripted team2=scripted

# Watch World Model Agents
uv run main.py mode=play team1=most_recent_world_model team2=most_recent_world_model

# Play as a human (Use Arrow Keys + Space to shoot)
uv run main.py mode=play human_player=true
```

**Supported Agents**:
*   `scripted`: Rule-based expert agent.
*   `most_recent_world_model`: Latest trained World Model (Yemong).
*   `best_world_model`: Best performing World Model based on validation loss.
*   `most_recent_rl`: Latest RL agent (PPO).

## Project Structure

*   **`src/boost_and_broadside/`**: Main namespaced package.
    *   **`agents/`**: Agent implementations (Scripted, Yemong/WorldModel, RL).
    *   **`core/`**: Core types, constants, and utilities.
    *   **`env2/`**: Vectorized (GPU) game environment.
    *   **`models/`**: Neural network architectures (Yemong, Heads, Encoders).
    *   **`modes/`**: Entry points (`play`, `collect`, `train`, `pretrain`).
    *   **`train/`**: Training pipelines.
*   **`tools/`**: Utility scripts for data inspection, visualization, and debugging.
*   **`configs/`**: Hydra configuration files.
*   **`tests/`**: Unit and integration tests.

## World Model Architecture (Yemong)

The core of the system is a **Factorized World Model (Yemong)** that separates temporal and spatial processing for efficiency and performance. It is designed to learn the physics and agent behaviors from the `env2` massive datasets.

### Architecture Highlights
*   **Backbone**: `Mamba2` (State Space Model) handles temporal mixing along the time axis.
*   **Spatial Layer**: `RelationalAttention` handles mixing between ships (spatial axis) at each timestep.
*   **Physics Trunk**: A shared relational encoder computes analytic edge features (distance, closing speed, etc.) to bias the attention.
*   **Factorization**: The model alternates between Mamba (Time) and Attention (Space) blocks.

### Training Objectives
*   **Action Prediction ($S_t \to A_t$)**: Cross-entropy loss on discrete action indices (Power, Turn, Shoot).
*   **State Prediction ($S_t, A_t \to S_{t+1}$)**: MSE loss on next-state reconstruction (Residual Delta).

## Workflows

### 1. Data Collection

Generate training data using scripted agents.

```powershell
# Collect data (CPU, single worker) - Good for small tests
uv run main.py mode=collect

# Massive GPU Data Collection (Recommended for pretraining)
# Generates HDF5 data using the vectorized environment (env2).
# Configured via 'collect.massive' in config.yaml.
uv run main.py mode=collect_massive
```

### 2. Training

Train models using the collected data.

```powershell
# Pretrain World Model (Yemong) on collected data
uv run main.py mode=pretrain

# Train Spatial Layer Only (Action Prediction from single state)
uv run main.py mode=pretrain model=yemong_spatial

# Train Temporal Layer Only (Next State Prediction from history)
uv run main.py mode=pretrain model=yemong_temporal

# Full Pipeline: Collect Data -> Train World Model
uv run main.py mode=train train.run_collect=true train.run_world_model=true
```

### 3. Evaluation

Evaluate model performance.

```powershell
# Evaluate World Model rollouts
uv run main.py mode=eval_wm
```

## Tools & Debugging

We provide tools in the `tools/` directory to help with development.

```powershell
# Inspect collected data files
uv run python tools/inspect_data.py data/bc_pretraining/latest/aggregated_data.h5

# Analyze Learning Rate Range Test Results
uv run python tools/analyze_lr_test.py --latest

# Generates 2D projections (PCA, PaCMAP) of world model embeddings
uv run python tools/viz_latent_space.py --latest
```

## Configuration

The project uses [Hydra](https://hydra.cc/) for configuration. Key config files are in `configs/`. Do not modify source code for parameters; use command line overrides or edit `configs/config.yaml`.

**Common Overrides:**
*   `cw` (World Size): `environment.world_size=[2000,2000]`
*   `max_ships`: `environment.max_ships=16`
*   `render`: `collect.render_mode='human'`

## Development

*   **Style Guide**: Please refer to [STYLE_GUIDE.md](STYLE_GUIDE.md) for coding standards.
*   **Tests**: Run `uv run pytest` to ensure everything is working.

```powershell
# Run all tests
uv run pytest
```
