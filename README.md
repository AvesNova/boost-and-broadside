# Boost and Broadside

A high-performance codebase for **Boost and Broadside**, a compiled competitive multi-agent environment where teams of ships compete in 2D dogfights.

## 🚀 Quick Start

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

## 📂 Project Structure

- **`src/`**: Source code for the environment, agents, and training logic.
    - **`agents/`**: Agent implementations (Scripted, Transformer, World Model).
    - **`env/`**: The core game environment (Ship dynamics, physics, state).
    - **`train/`**: Training pipelines for BC and RL.
    - **`modes/`**: Entry points for different execution modes (play, collect, train).
- **`tools/`**: Utility scripts for data inspection and debugging.
- **`configs/`**: Hydra configuration files.
- **`tests/`**: Unit and integration tests.

## 🛠️ Workflows

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

## 🔧 Tools & Debugging

We provide tools in the `tools/` directory to help with development.

```powershell
# Inspect collected data files
uv run python tools/inspect_data.py data/bc_pretraining/latest/aggregated_data.pkl

# Verify data loading logic
uv run python tools/verify_data_loading.py

# Generates 2D projections (PCA, PaCMAP) of world model embeddings
uv run python tools/viz_latent_space.py --latest
```

## ⚙️ Configuration

The project uses [Hydra](https://hydra.cc/) for configuration. Key config files are in `configs/`. Do not modify source code for parameters; use command line overrides or edit `configs/config.yaml`.

**Common Overrides:**
- `cw` (World Size): `environment.world_size=[2000,2000]`
- `max_ships`: `environment.max_ships=16`
- `render`: `collect.render_mode='human'`

## 📝 Development

- **Style Guide**: Please refer to [STYLE_GUIDE.md](STYLE_GUIDE.md) for coding standards.
- **Tests**: Run `uv run pytest --color=no -rf --tb=line` to ensure everything is working.

```powershell
# Run all tests
uv run pytest --color=no -rf --tb=line

# Run specific test file
uv run pytest tests/components/test_ship.py --color=no -rf --tb=line
```
