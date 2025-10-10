# Boost and Broadside

> Physics-based ship combat simulation with transformer-based AI agents

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Boost and Broadside** is a reinforcement learning research project featuring a custom Gymnasium environment where AI-controlled ships battle in 2D space using physics. The project implements a transformer-based multi-agent system that learns coordinated team tactics through behavior cloning pretraining followed by PPO reinforcement learning.

## Features

- **Ship Physics**: Complex aerodynamic model with thrust, drag, lift, and power management
- **Transformer-Based AI**: Multi-head self-attention architecture for team coordination
- **Multi-Agent Learning**: Supports 1v1 through 4v4 team battles with variable team sizes
- **Two-Phase Training**: Behavior cloning pretraining → PPO reinforcement learning pipeline
- **Human Playable**: Interactive pygame interface for playing against AI or collecting demonstrations
- **Self-Play Support**: Maintains opponent memory pool for curriculum learning
- **Episode Playback**: Record and replay episodes for analysis and debugging
- **Gymnasium Compatible**: Standard RL interface with Stable-Baselines3 integration
- **Parallel Processing**: Multi-worker data collection and training with GPU acceleration
- **Fault Tolerance**: Automatic checkpointing and recovery from interruptions

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training AI Agents](#training-ai-agents)
  - [Playing the Game](#playing-the-game)
  - [Data Collection](#data-collection)
  - [Parallel Processing](#parallel-processing)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Development](#development)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Setup

1. **Clone the repository**
   ```powershell
   git clone https://github.com/yourusername/boost-and-broadside.git
   cd boost-and-broadside
   ```

2. **Create a virtual environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install the package**
   ```powershell
   # Standard installation
   pip install -e .
   
   # With development dependencies
   pip install -e ".[dev]"
   ```

## Quick Start

### Play Against a Scripted Opponent

```powershell
python main.py play human
```

**Controls:**
- `W` or `↑`: Boost forward
- `S` or `↓`: Reverse thrust
- `A` or `←`: Turn left
- `D` or `→`: Turn right
- `Shift`: Sharp turn modifier
- `Space`: Fire

### Train an AI Agent (Full Pipeline)

```powershell
# 1. Collect behavior cloning data (with parallel processing)
python main.py collect bc --config src/unified_training.yaml

# 2. Run full training pipeline (BC pretraining + RL)
python main.py train full --config src/unified_training.yaml
```

### Parallel Data Collection and Training

```powershell
# Collect data using 4 worker processes
python main.py collect bc --config src/unified_training.yaml

# Train with parallel RL environments
python main.py train rl --config src/unified_training.yaml
```

### Watch a Trained Model

```powershell
# Replay a recorded episode
python main.py replay episode --episode-file data/bc_pretraining/1v1_episodes.pkl.gz
```

## Command Reference

The unified command-line interface provides access to all functionality through `main.py`:

### Training Commands
```bash
# Behavior cloning training
python main.py train bc [--config CONFIG] [--run-name NAME]

# Reinforcement learning training
python main.py train rl [--config CONFIG] [--run-name NAME] [--bc-model PATH]

# Full training pipeline
python main.py train full [--config CONFIG] [--run-name NAME] [--skip-bc]
```

### Data Collection Commands
```bash
# Collect behavior cloning data
python main.py collect bc [--config CONFIG] [--output DIR]

# Collect self-play data
python main.py collect selfplay [--config CONFIG] [--output DIR]
```

### Playing Commands
```bash
# Human vs AI play
python main.py play human [--config CONFIG]
```

### Replay Commands
```bash
# Replay saved episode
python main.py replay episode --episode-file PATH [--config CONFIG]

# Browse episode data
python main.py replay browse [--data-dir DIR] [--config CONFIG]
```

### Evaluation Commands
```bash
# Evaluate trained model
python main.py evaluate model --model PATH [--config CONFIG]
```

## Usage

### Training AI Agents

#### Behavior Cloning (BC) Pretraining

Collect expert demonstrations from scripted agents:

```powershell
python main.py collect bc --config src/unified_training.yaml
```

This generates training data in `data/bc_pretraining/` with episodes from various game modes (1v1, 2v2, 3v3, 4v4).

Train the BC model:

```powershell
python main.py train bc --config src/unified_training.yaml
```

#### Reinforcement Learning (RL) Training

Train with PPO from scratch:

```powershell
python main.py train rl --config src/unified_training.yaml
```

Or initialize from a BC checkpoint:

```powershell
python main.py train rl --config src/unified_training.yaml --bc-model checkpoints/bc_model.pt
```

#### Full Pipeline

Run both phases automatically:

```powershell
python main.py train full --config src/unified_training.yaml
```

**Training Modes:**
- `scripted`: Train against rule-based opponents
- `self_play`: Train against past versions of the policy
- `mixed`: Blend of scripted and self-play opponents

### Playing the Game

#### Human vs Scripted

```powershell
python main.py play human
```

#### Human vs Trained AI

```powershell
python main.py play human --opponent-model checkpoints/best_model.pt
```

### Data Collection

#### Collect Custom Episodes

```powershell
python src/collect_data.py collect_bc --config src/unified_training.yaml
```

Modify `src/unified_training.yaml` to adjust:
- Number of episodes per game mode
- Game modes to include
- Output directory

#### Episode Playback

```powershell
python src/collect_data.py playback --episode-file <path-to-episode.pkl.gz>
```

**Playback Controls:**
- `Space`: Pause/Resume
- `S`: Step forward (when paused)
- `R`: Reset to beginning
- `+`/`-`: Increase/decrease playback speed
- `ESC`: Quit

## Parallel Processing

The project supports parallel processing for faster data collection and training:

### Key Features

- **Multi-Worker Data Collection**: Distribute BC data collection across multiple processes
- **Parallel RL Environments**: Run multiple RL environments simultaneously
- **Automatic Checkpointing**: Periodic saves to prevent data loss
- **GPU Acceleration**: Utilize GPU for both BC and RL training
- **Fault Tolerance**: Resume from interruptions automatically

### Configuration

Add to your configuration file:

```yaml
parallel_processing:
  enabled: true
  num_workers: 4
  
  data_collection:
    episodes_per_worker: 250
    checkpoint_frequency: 100
    
  rl_training:
    num_envs: 4
    checkpoint_frequency: 25000
```

For detailed documentation, see [docs/parallel_processing.md](docs/parallel_processing.md).

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Environment (env.py)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Ship Physics │  │ Bullet System│  │ Collision Det│   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────┘
                            ↓
              ┌─────────────────────────────┐
              │  Observation (State Tokens) │
              └─────────────────────────────┘
                            ↓
         ┌──────────────────────────────────────┐
         │     Transformer Model                │
         │  ┌────────────────────────────────┐  │
         │  │  Token Projection              │  │
         │  │  Multi-Head Self-Attention     │  │
         │  │  Action Head                   │  │
         │  └────────────────────────────────┘  │
         └──────────────────────────────────────┘
                            ↓
              ┌─────────────────────────────┐
              │  Actions (per-ship binary)  │
              └─────────────────────────────┘
```

### Key Components

#### 1. Environment (`env.py`)
- **Gymnasium-compatible** custom environment
- **Physics simulation** at 50Hz (default), agent decisions at 25Hz
- **Variable team sizes**: 1v1, 2v2, 3v3, 4v4, or random (nvn)
- **Fractal positioning**: Ships positioned using recursive fractal patterns

#### 2. Ship Physics (`ship.py`)
- **Thrust system**: Base, boost (drains power), and reverse thrust
- **Aerodynamics**: Drag and lift forces based on turn state
- **Power management**: Balance between boosting and shooting
- **Action space**: 6 binary actions per ship (forward, backward, left, right, sharp_turn, shoot)

#### 3. Transformer Model (`team_transformer_model.py`)
- **Input**: Ship state tokens (10-dim: health, power, position, velocity, attitude)
- **Architecture**: Token projection → Multi-head attention → Action head
- **Attention masking**: Handles variable team sizes and dead ships
- **Output**: Action logits for all ships

#### 4. Agent System (`agents.py`)
Unified interface supporting:
- **Scripted agents**: Rule-based targeting and maneuvering
- **RL agents**: Transformer and PPO-based policies
- **Human agents**: Keyboard control
- **Self-play agents**: Opponent memory pool
- **Playback agents**: Episode replay

#### 5. Training Pipeline

**Phase 1: Behavior Cloning**
- Collect expert demonstrations (scripted vs scripted)
- Train transformer via supervised learning
- Learn basic combat behaviors

**Phase 2: Reinforcement Learning**
- Initialize from BC weights (optional)
- Train with PPO against scripted/self-play opponents
- Learn advanced tactics and team coordination

## Configuration

All hyperparameters are in `src/unified_training.yaml`:

```yaml
environment:
  world_size: [1200, 800]
  max_ships: 8
  agent_dt: 0.04        # Agent decision interval
  physics_dt: 0.02      # Physics simulation timestep

model:
  transformer:
    embed_dim: 64
    num_heads: 4
    num_layers: 3
  
  ppo:
    learning_rate: 0.0003
    n_steps: 4096
    batch_size: 128

training:
  rl:
    total_timesteps: 2000000
    opponent:
      type: "mixed"              # scripted, self_play, or mixed
      scripted_mix_ratio: 0.3
```

See the full configuration file for all options.

## Development

### Running Tests

```powershell
# Run all tests
pytest tests

# Run specific test file
pytest tests/test_environment_basics.py

# Run with verbose output
pytest -v tests

# Run specific test
pytest tests/test_ship_physics.py::test_thrust_calculation
```

### Code Formatting

```powershell
# Format code
black src tests

# Check formatting
black --check src tests
```

### VSCode Launch Configurations

The project includes `.vscode/launch.json` with configurations:
- **Play**: Launch human play mode
- **Replay**: Playback episodes
- **Collect BC**: Collect training data
- **Train**: Run training pipeline
- **Current File**: Debug active file

## Project Structure

```
boost-and-broadside/
├── src/
│   ├── env.py                      # Gymnasium environment
│   ├── ship.py                     # Ship physics
│   ├── state.py                    # Game state
│   ├── agents.py                   # Agent interface
│   ├── team_transformer_model.py   # Transformer architecture
│   ├── transformer_policy.py       # SB3 custom policy
│   ├── rl_wrapper.py               # RL training wrapper
│   ├── parallel_rl_wrapper.py      # Parallel RL environments
│   ├── parallel_utils.py           # Worker process utilities
│   ├── data_aggregator.py          # Data aggregation from workers
│   ├── recovery_manager.py         # Recovery from interruptions
│   ├── unified_train.py            # Training entry point
│   ├── bc_training.py              # Behavior cloning
│   ├── collect_data.py             # Data collection
│   ├── game_runner.py              # Game orchestration
│   ├── renderer.py                 # Pygame visualization
│   └── unified_training.yaml       # Configuration
├── tests/
│   ├── conftest.py                 # Test fixtures
│   ├── test_environment_*.py       # Environment tests
│   ├── test_ship_*.py              # Physics tests
│   ├── test_parallel_utils.py      # Parallel processing tests
│   ├── test_data_aggregator.py     # Data aggregation tests
│   ├── test_recovery_manager.py    # Recovery tests
│   └── test_*.py                   # Additional tests
├── docs/
│   └── parallel_processing.md      # Parallel processing documentation
├── data/                           # Training data
├── checkpoints/                    # Model checkpoints
├── logs/                           # Training logs
├── pyproject.toml                  # Package configuration
├── pytest.ini                      # Test configuration
└── README.md
```

## Technical Details

### Observation Space

Each ship is represented by a 10-dimensional token:
1. Health (0-100)
2. Power (0-100)
3-4. Position (x, y)
5-6. Velocity (vx, vy)
7-8. Attitude (cos θ, sin θ)
9-10. Additional features

Observations include tokens for all ships (padded to `max_ships`).

### Action Space

6 binary actions per ship:
- `forward` (0): Boost forward (drains power)
- `backward` (1): Reverse thrust
- `left` (2): Turn left
- `right` (3): Turn right
- `sharp_turn` (4): Sharp turn modifier (increases drag)
- `shoot` (5): Fire bullet (costs power)

### Reward Structure

Rewards are shaped to encourage:
- Dealing damage to enemies
- Surviving (avoiding damage)
- Team coordination
- Efficient power management

See `constants.py` for reward constants.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Development Guidelines:**
- Use type hints for all functions
- Follow existing code style (black formatting)
- Add tests for new features
- Update documentation as needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Gymnasium](https://gymnasium.farama.org/) for RL environment interface
- Uses [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for PPO implementation
- Transformer architecture inspired by [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Physics simulation inspired by real-world naval vessel dynamics

## Citation

If you use this project in your research, please cite:

```bibtex
@software{boost_and_broadside,
  author = {Julian Kizanis},
  title = {Boost and Broadside: Physics-Based Ship Combat with Transformer Agents},
  year = {2025},
  url = {https://github.com/avesnova/boost-and-broadside}
}
```

---

**Questions or Issues?** Please open an issue on GitHub or reach out to the maintainers.
