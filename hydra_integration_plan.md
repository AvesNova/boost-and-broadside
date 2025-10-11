# Hydra Integration Plan for Boost and Broadside

## Overview
This plan outlines how to integrate Hydra/Omegaconf for configuration management while keeping the current subcommand structure (train bc, train rl, play human, etc.).

## Current State Analysis
The codebase already has partial Hydra integration:
- `main.py` has the `@hydra.main` decorator
- `unified_training.yaml` exists with comprehensive configuration
- But the full potential of Hydra is not utilized

## Proposed Architecture

### 1. Configuration Structure
```
src/config/
├── __init__.py
├── base.yaml              # Base configuration (renamed from unified_training.yaml)
├── train/
│   ├── bc.yaml           # BC training specific overrides
│   ├── rl.yaml           # RL training specific overrides
│   └── full.yaml         # Full pipeline specific overrides
├── collect/
│   ├── bc.yaml           # BC data collection overrides
│   └── selfplay.yaml     # Self-play data collection overrides
├── play/
│   └── human.yaml        # Human play overrides
├── replay/
│   ├── episode.yaml      # Episode replay overrides
│   └── browse.yaml       # Episode browsing overrides
└── evaluate/
    └── model.yaml        # Model evaluation overrides
```

### 2. Command Flow
1. User runs: `python main.py train bc --config src/config/base.yaml`
2. Hydra loads base.yaml
3. System detects command "train bc" and applies train/bc.yaml overrides
4. CLI arguments are merged with the final configuration
5. Pipeline receives the merged DictConfig

### 3. Integration Points

#### main.py Changes
- Keep current argparse structure for subcommands
- Use Hydra to load and merge configurations
- Pass DictConfig to pipeline modules
- Support Hydra overrides for any parameter

#### Pipeline Module Changes
- Accept DictConfig instead of args.config dict
- Use OmegaConf access patterns (cfg.key instead of cfg["key"])
- Remove manual config loading from files
- Keep command-specific logic but use unified config

#### Configuration Files
- Move unified_training.yaml to src/config/base.yaml
- Create command-specific override files
- Use Hydra's defaults list to compose configurations

## Benefits
1. **Unified Configuration**: All config in one place with clear overrides
2. **Type Safety**: Hydra provides validation and type checking
3. **Easy Overrides**: `python main.py train bc model.transformer.embed_dim=128`
4. **Backward Compatibility**: Existing commands still work
5. **Clear Structure**: Configuration organized by command

## Implementation Steps
1. Create config directory structure
2. Move and reorganize configuration files
3. Update main.py to properly use Hydra
4. Modify pipeline modules to work with DictConfig
5. Update CLI argument handling
6. Test all commands
7. Update documentation

## Example Usage
```bash
# Current usage (still works)
python main.py train bc --config src/config/base.yaml

# With Hydra overrides
python main.py train bc model.transformer.embed_dim=128 training.rl.total_timesteps=5000000

# Different output directory
python main.py collect bc data_collection.bc_data.output_dir=custom_data