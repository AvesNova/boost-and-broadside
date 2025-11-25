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

# Select agents (options: scripted, dummy, random, most_recent_bc, most_recent_rl, best_bc, best_rl)
uv run main.py mode=play team1=most_recent_bc team2=scripted

# Collect training data from scripted agents (single worker)
uv run main.py mode=collect

# Collect with multiple parallel workers (set in config)
uv run main.py mode=collect collect.num_workers=4

# Train RL models
uv run main.py mode=train

# Train with multiple parallel environments
uv run main.py mode=train train.rl.n_envs=8
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


### Unified Training Pipeline

Run the full pipeline (Collect -> BC -> RL) in a single command:
```powershell
uv run main.py mode=train train.run_collect=true train.run_bc=true train.run_rl=true
```

### Parallel RL Training
You can speed up RL training by running multiple environments in parallel. This is controlled by `train.rl.n_envs`.
```powershell
# Run with 8 parallel environments
uv run main.py mode=train train.run_rl=true train.rl.n_envs=8
```

You can also run partial pipelines:
```powershell
# BC -> RL (skip collection, use existing data)
uv run main.py mode=train train.run_bc=true train.run_rl=true train.bc_data_path="path/to/data.pkl"

# RL only (load pretrained BC model)
uv run main.py mode=train train.run_rl=true train.rl.pretrained_model_path="path/to/model.pth"

# RL from scratch (no pretraining)
uv run main.py mode=train train.run_rl=true
```

### World Model

The World Model is a transformer-based model that learns multi-ship dynamics using 2-D factorized attention, MAE-style masking, and flow-matching denoising.

**Architecture Overview**:
- **Token Structure**: Each token = `[ship_state, previous_action]` concatenated (state: 12 dims, action: 6 dims)
- **Embedding Structure**: `final_embed = content_projection + ship_id_embedding + time_embedding`
- **Attention Pattern**: Alternating spatial and temporal blocks in 3:1 ratio (S-S-S-T-S-S-S-T...)
- **Batch Length Alternation**: Alternates between short (32) and long (128) batch lengths per iteration

**Token & Embedding Details**:
1. **Content Projection**: Raw token (state + action) projected to embed_dim (128)
2. **Ship ID Embedding**: Learned embedding for each ship (0-7), preserves ship identity
3. **Time Embedding**: Learned positional embedding for each timestep (0-63)
4. **Critical Order**: Noise/masking applied to content BEFORE adding ship_id and time embeddings

**Masking (MAE-style)**:
- Randomly masks 15% of tokens during training
- Masked tokens: content replaced with learned `mask_token`, but ship_id and time embeddings preserved
- Masked tokens NEVER receive noise
- Teaches occlusion reasoning and missing-data handling

**Denoising (Flow-Matching)**:
- Applies Gaussian noise to UNMASKED tokens only
- Noise scale: `σ = sqrt(1 - τ²) * 0.1` where τ ~ U(0,1)²
- Noise added to content BEFORE structural embeddings
- Stabilizes long rollouts and reduces error compounding

**2-D Factorized Attention**:
- **Spatial Blocks**: Ships attend to each other at the SAME timestep (no temporal mixing)
  - Reshape: `(B, T, N, E)` → `(B*T, N, E)`
  - No KV caching (attention is per-timestep)
- **Temporal Blocks**: Each ship attends only to its OWN past (causal, no ship mixing)
  - Reshape: `(B, T, N, E)` → `(B*N, T, E)`
  - Always causal (even with KV cache)
  - KV cache concatenates along time dimension only
- **Pattern**: 3 spatial blocks, then 1 temporal block, repeated (8 layers total)

**Training**:
```powershell
# Train the World Model
uv run main.py mode=train_wm train.bc_data_path=data/bc_pretraining/debug/aggregated_data.pkl

# With custom parameters
uv run main.py mode=train_wm world_model.epochs=10 world_model.mask_ratio=0.15
```

**Configuration** (`world_model` in `config.yaml`):
- `embed_dim`: 128 - Transformer embedding dimension
- `n_layers`: 8 - Total transformer blocks (6 spatial, 2 temporal)
5. ✅ KV cache only for temporal blocks (not spatial)
6. ✅ Temporal attention never mixes ships
7. ✅ Spatial attention never mixes timesteps

**Evaluation**:
```powershell
# Evaluate the World Model (generate rollouts)
uv run main.py mode=eval_wm
```

