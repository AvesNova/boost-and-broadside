# Hydra Integration Summary

## Overview

This document summarizes the integration of Hydra/Omegaconf into the Boost and Broadside project to simplify configuration management and enable powerful command-line parameter overrides.

## Key Changes

### 1. Configuration Structure

- **Old System**: Single configuration file (`src/unified_training.yaml`) with manual loading
- **New System**: Hierarchical Hydra configuration in `src/config/` directory
  - `base.yaml`: Contains all default parameters
  - Command-specific configs in subdirectories (`train/`, `collect/`, `play/`, `replay/`, `evaluate/`)

### 2. Main Entry Point

- **File**: `main.py`
- **Changes**:
  - Integrated Hydra initialization and configuration composition
  - Automatic command detection for loading appropriate configuration
  - Support for both traditional CLI arguments and Hydra overrides
  - Preserved backward compatibility with existing command structure

### 3. Pipeline Modules

- **Location**: `src/pipelines/`
- **Changes**:
  - Updated all pipeline modules to work with Hydra's `DictConfig`
  - Added proper conversion between `DictConfig` and regular dictionaries where needed
  - Maintained existing functionality while adding Hydra support

### 4. Import Fixes

- Fixed numerous import issues throughout the codebase
- Converted absolute imports to relative imports where appropriate
- Resolved circular dependencies

## Benefits of the New System

### 1. Simplified Configuration

- No need to specify config file path for common operations
- Automatic loading of appropriate configuration based on command
- Clear separation of configuration for different operations

### 2. Powerful Overrides

- Override any parameter from the command line
- Example: `python main.py train bc model.transformer.embed_dim=128`
- No need to modify configuration files for temporary changes

### 3. Configuration Composition

- Base configuration with command-specific overrides
- Easy to maintain and extend
- Clear inheritance structure

### 4. Backward Compatibility

- Existing commands continue to work
- No breaking changes to the user interface
- Gradual adoption of new features possible

## Usage Examples

### Training

```bash
# Basic BC training
python main.py train bc

# BC training with custom parameters
python main.py train bc model.transformer.embed_dim=128 model.ppo.learning_rate=0.0001

# RL training with BC initialization
python main.py train rl --bc-model checkpoints/bc_model.pt

# Full pipeline with custom timesteps
python main.py train full training.rl.total_timesteps=5000000
```

### Data Collection

```bash
# BC data collection
python main.py collect bc

# Self-play data collection with custom episodes
python main.py collect selfplay data_collection.selfplay_data.episodes_per_mode.2v2=50

# Parallel data collection
python main.py collect bc parallel_processing.enabled=true parallel_processing.num_workers=4
```

### Playing and Evaluation

```bash
# Human play
python main.py play human

# Custom world size
python main.py play human environment.world_size=[1600,1200]

# Model evaluation
python main.py evaluate model --model checkpoints/best_model.pt

# Evaluation with custom parameters
python main.py evaluate model --model checkpoints/best_model.pt evaluation.num_episodes=50
```

## Configuration Files Structure

```
src/config/
├── base.yaml                 # Base configuration
├── train/
│   ├── bc.yaml              # BC training overrides
│   ├── rl.yaml              # RL training overrides
│   └── full.yaml            # Full pipeline overrides
├── collect/
│   ├── bc.yaml              # BC data collection overrides
│   └── selfplay.yaml        # Self-play data collection overrides
├── play/
│   └── human.yaml           # Human play overrides
├── replay/
│   ├── episode.yaml         # Episode replay overrides
│   └── browse.yaml          # Episode browsing overrides
└── evaluate/
    └── model.yaml           # Model evaluation overrides
```

## Technical Implementation Details

### Command Detection

The system automatically detects commands and loads the appropriate configuration:

```python
def get_command_specific_config(command_parts: list) -> str:
    """Get command-specific config based on command parts"""
    config_map = {
        ("train", "bc"): "train/bc",
        ("train", "rl"): "train/rl",
        ("train", "full"): "train/full",
        ("collect", "bc"): "collect/bc",
        ("collect", "selfplay"): "collect/selfplay",
        ("play", "human"): "play/human",
        ("replay", "episode"): "replay/episode",
        ("replay", "browse"): "replay/browse",
        ("evaluate", "model"): "evaluate/model",
    }
    return config_map.get(tuple(command_parts), "base")
```

### Configuration Composition

Hydra automatically composes configurations from multiple sources:

1. Base configuration (`base.yaml`)
2. Command-specific configuration (e.g., `train/bc.yaml`)
3. Runtime overrides from command line

### DictConfig Handling

Pipeline modules now work with `DictConfig` objects:

```python
@staticmethod
def execute(cfg: DictConfig) -> int:
    """Execute the appropriate command with DictConfig"""
    # Convert to regular dict for compatibility
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Use the configuration
    ...
```

## Migration Guide

### For Users

No changes required for basic usage. Existing commands continue to work:

```bash
python main.py train bc
python main.py collect bc
python main.py play human
```

### For Developers

When modifying existing code:

1. Use `cfg.get("key", default_value)` for safe access to configuration values
2. Convert `DictConfig` to regular dict for compatibility with existing functions
3. Use `OmegaConf.to_container(cfg, resolve=True)` for conversion

```python
# Old way
config = load_config(config_path)
value = config["key"]

# New way
value = cfg.get("key", default_value)
```

## Future Enhancements

1. **Configuration Validation**: Add schema validation for configuration files
2. **Environment-Specific Configs**: Add support for different environments (dev, prod)
3. **Configuration Groups**: Implement configuration groups for easier management
4. **Dynamic Configuration**: Add support for dynamic configuration based on runtime conditions

## Conclusion

The Hydra integration simplifies configuration management while maintaining backward compatibility. It provides powerful override capabilities and a clear, maintainable configuration structure that will make the project easier to use and extend.