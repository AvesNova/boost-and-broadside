# Boost and Broadside

## Project Overview

Boost and Broadside is a reinforcement learning project for training AI agents to play a multi-ship combat game. The project uses PyTorch, Stable-Baselines3, and Hydra for configuration management.

## Development Commands

### Running the Application
```powershell
# Play mode (human playable with rendering)
uv run main.py mode=play

# Play mode options:
# Disable human player (watch only)
uv run main.py mode=play human_player=false

# Select agents (options: scripted, dummy, random, most_recent, best)
uv run main.py mode=play team1=random team2=best

# Collect training data from scripted agents (single worker)
uv run main.py mode=collect

# Collect with multiple parallel workers (set in config)
uv run main.py mode=collect collect.num_workers=4

# Train RL models
uv run main.py mode=train
```

### Testing
```powershell
# Run all tests
uv run pytest

# Run component tests
uv run pytest tests/components/

# Run pipeline tests
uv run pytest tests/pipelines/

# Run with verbose output
uv run pytest -v
```

### Inspecting Collected Data
```powershell
# Inspect all collected data files
uv run python src/inspect_data.py

# Inspect a specific file
uv run python src/inspect_data.py data/bc_pretraining/worker_0/data_final.pkl
```

### Python Environment
```powershell
# Install dependencies with uv
uv sync

# Run commands with uv (automatically manages venv)
uv run <command>
```

## Architecture

### Core Components

**GameCoordinator** (`src/game_coordinator.py`)
- Central orchestrator between the environment and agents
- Manages episode lifecycle, state history, and token generation
- Coordinates multi-agent team actions

**Environment** (`src/env/env.py`)
- Gymnasium-compatible environment for ship combat
- Supports multiple game modes: 1v1, 2v2, 3v3, 4v4, NvN
- Uses fractal positioning algorithm for squad initialization
- Physics substep system: agents act at `agent_dt` intervals, physics updates at `physics_dt`

**State** (`src/env/state.py`)
- Immutable snapshot of game state at a specific time
- Contains ship dictionary and bullet manager

**Ship** (`src/env/ship.py`)
- Complex physics model with thrust, drag, and lift forces
- Action lookup tables for efficient state updates
- Six actions: forward, backward, left, right, sharp_turn, shoot
- Power/energy management system

### Agent System

**Agent Types** (`src/agents/`)
- `ScriptedAgent`: Rule-based AI using shooting range and angle thresholds
- `TeamTransformerAgent`: Transformer-based model that outputs actions for all ships
- `HumanAgent`: Human-controlled via pygame input
- `ReplayAgent`: Replays recorded actions

**Tokenizer** (`src/agents/tokenizer.py`)
- Converts observations to token tensors for transformer input
- Uses sin/cos encoding for wrap-around position features
- Team-relative perspective encoding

**TeamTransformerAgent Architecture**:
1. Project ship tokens (10D) to transformer dimension (64D default)
2. Apply transformer encoder with self-attention
3. Output action logits for each ship (6 actions)
4. Teams select which ships they control during execution

### Configuration

Uses Hydra for hierarchical configuration (`configs/config.yaml`):
- **mode**: play, collect, or train
- **environment**: world_size, memory_size, max_ships, dt parameters
- **agents**: agent type and configuration per team
- **collect**: data collection settings for BC pretraining
- **train**: BC and RL hyperparameters, transformer model config

### Code Style

- Use type hints: `dict | None` instead of `Optional[Dict]`
- Add docstrings to all methods and files
- Prefer consistent type hints on method inputs and outputs

### Game Modes

- `1v1`, `2v2`, `3v3`, `4v4`: Fixed team sizes
- `nvn`: Random team sizes up to max_ships/2
- `reset_from_observation`: Reset to specific state for replay

### Physics System

- Dual timestep: `agent_dt` (default 0.04s) for agent decisions, `physics_dt` (default 0.02s) for physics updates
- Ships use complex numbers for 2D positions and velocities
- Lookup tables optimize force calculations
- Power regeneration/depletion based on thrust mode

### Training Pipeline

1. **Data Collection** (BC Pretraining): Scripted agents generate demonstrations
2. **Behavioral Cloning**: Pre-train transformer on collected data
3. **RL Fine-tuning**: PPO-based refinement using Stable-Baselines3

### Data Collection

**Parallel Workers**: Set `collect.num_workers` in config to run multiple collection processes in parallel using Python's multiprocessing. Each worker saves to its own subdirectory (`worker_0/`, `worker_1/`, etc.).

**Run Organization**: Each collection run creates a timestamped directory (e.g., `20251019_221740/`) to prevent collisions between runs.

**Data Format**: Episodes are aggregated into single tensors organized by team:
```
data["team_0"]
  ["tokens"]: Token sequences (T, max_ships, token_dim)
  ["actions"]: Action tensors (T, max_ships, num_actions)
  ["rewards"]: Reward sequences (T,)
data["team_1"]
  ["tokens"]: Token sequences (T, max_ships, token_dim)
  ["actions"]: Action tensors (T, max_ships, num_actions)
  ["rewards"]: Reward sequences (T,)
data["episode_ids"]: Episode ID for each timestep (T,)
data["episode_lengths"]: Length of each episode (N,)
data["metadata"]: Run info, dimensions, worker details
```

**Aggregation**: After all workers complete, data is automatically aggregated into:
- `aggregated_data.pkl`: Combined data from all workers
- `metadata.yaml`: Human-readable metadata and data shapes

**Checkpointing**: Data is periodically saved (configurable via `save_frequency`) to prevent data loss on crashes.

**Max Episode Length**: Episodes are automatically terminated after `max_episode_length` steps (default: 512) to prevent infinite games.

### Output Directories

- `data/bc_pretraining/YYYYMMDD_HHMMSS/`: Timestamped run directory
  - `worker_N/`: Individual worker data and checkpoints
  - `aggregated_data.pkl`: Combined data from all workers
  - `metadata.yaml`: Run metadata and data shapes
- `outputs/`: Hydra auto-generates timestamped output directories


### Next Steps

Implement the BC training
Implement the RL training
