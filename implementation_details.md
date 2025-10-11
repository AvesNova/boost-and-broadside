# Detailed Implementation Plan for Hydra Integration

## 1. Configuration Structure Design

### Base Configuration (src/config/base.yaml)
This will be the renamed `unified_training.yaml` with minor modifications to support Hydra's composition.

### Command-Specific Overrides

#### src/config/train/bc.yaml
```yaml
defaults:
  - base
  - _self_

# Override specific values for BC training
training:
  mode: "bc"
  
# BC-specific logging
logging:
  console:
    level: "INFO"
    progress_bars: true
```

#### src/config/train/rl.yaml
```yaml
defaults:
  - base
  - _self_

# Override specific values for RL training
training:
  mode: "rl"
  rl:
    total_timesteps: 2000000
```

## 2. main.py Updates

### Current Issues
- The current implementation mixes argparse and Hydra in a complex way
- It converts DictConfig back to regular dict, losing Hydra benefits
- No proper integration of command detection with config loading

### Proposed main.py Structure
```python
#!/usr/bin/env python3
"""
Boost and Broadside - Unified Command Line Interface
"""

import argparse
import sys
from typing import Dict, Any

import hydra
from omegaconf import OmegaConf, DictConfig

# Import pipeline modules
from src.pipelines import (
    TrainingPipeline,
    DataCollectionPipeline,
    PlayPipeline,
    PlaybackPipeline,
    EvaluationPipeline,
)

def create_parser():
    """Create the main argument parser with Hydra support"""
    parser = argparse.ArgumentParser(
        description="Boost and Broadside - Physics-based ship combat with transformer-based AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        # Add conflict_handler to allow Hydra overrides
        conflict_handler='resolve',
    )
    
    # Add Hydra support for config overrides
    parser.add_argument(
        "--config-path",
        type=str,
        default="src/config",
        help="Path to configuration directory"
    )
    
    parser.add_argument(
        "--config-name",
        type=str,
        default="base",
        help="Base configuration name"
    )

    # Add subparsers for each pipeline
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add pipeline subparsers
    TrainingPipeline.add_subparsers(subparsers)
    DataCollectionPipeline.add_subparsers(subparsers)
    PlayPipeline.add_subparsers(subparsers)
    PlaybackPipeline.add_subparsers(subparsers)
    EvaluationPipeline.add_subparsers(subparsers)

    return parser

def get_command_specific_config(command_parts: list) -> str:
    """Get command-specific config based on command parts"""
    if not command_parts:
        return "base"
    
    # Map commands to config files
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

@hydra.main(version_base=None, config_path="src/config", config_name="base")
def main(cfg: DictConfig) -> int:
    """Main entry point with full Hydra configuration"""
    # Store original config path and name
    original_config_path = hydra.utils.get_original_cwd() + "/src/config"
    original_config_name = "base"
    
    # Parse command line arguments to determine the command
    parser = create_parser()
    cli_args, unknown_args = parser.parse_known_args()
    
    # If no command is specified, print help
    if not hasattr(cli_args, "command") or cli_args.command is None:
        parser.print_help()
        return 1
    
    # Determine command parts for config selection
    command_parts = []
    if hasattr(cli_args, "command") and cli_args.command:
        command_parts.append(cli_args.command)
    
    # Add subcommand if exists
    subcommand_attr = f"{cli_args.command}_command"
    if hasattr(cli_args, subcommand_attr) and getattr(cli_args, subcommand_attr):
        command_parts.append(getattr(cli_args, subcommand_attr))
    
    # Load command-specific configuration
    command_config = get_command_specific_config(command_parts)
    
    # Use Hydra's compose API to load the appropriate config
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    # Initialize Hydra with the config directory
    with initialize_config_dir(config_dir=original_config_path, version_base=None):
        # Compose the configuration with command-specific overrides
        cfg = compose(config_name=command_config, overrides=unknown_args)
    
    # Merge CLI arguments into config
    # This allows CLI args to override config file values
    cli_dict = vars(cli_args)
    for key, value in cli_dict.items():
        if key != "command" and value is not None:
            # Convert args with underscores to config with dots
            config_key = key.replace("_", ".")
            OmegaConf.set(cfg, config_key, value)
    
    # Execute the appropriate pipeline with the Hydra config
    try:
        if cli_args.command == "train":
            return TrainingPipeline.execute(cfg)
        elif cli_args.command == "collect":
            return DataCollectionPipeline.execute(cfg)
        elif cli_args.command == "play":
            return PlayPipeline.execute(cfg)
        elif cli_args.command == "replay":
            return PlaybackPipeline.execute(cfg)
        elif cli_args.command == "evaluate":
            return EvaluationPipeline.execute(cfg)
        else:
            print(f"Unknown command: {cli_args.command}")
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

## 3. Pipeline Module Updates

### TrainingPipeline Changes
```python
@staticmethod
def execute(cfg: DictConfig) -> int:
    """Execute the appropriate training command with DictConfig"""
    try:
        with InterruptHandler("Training interrupted by user"):
            # Get command from config
            if cfg.get("training", {}).get("mode") == "bc":
                return TrainingPipeline._train_bc(cfg)
            elif cfg.get("training", {}).get("mode") == "rl":
                return TrainingPipeline._train_rl(cfg)
            elif cfg.get("training", {}).get("mode") == "full":
                return TrainingPipeline._train_full(cfg)
            else:
                # Fallback to detecting from command structure
                if "train_command" in cfg:
                    if cfg.train_command == "bc":
                        return TrainingPipeline._train_bc(cfg)
                    elif cfg.train_command == "rl":
                        return TrainingPipeline._train_rl(cfg)
                    elif cfg.train_command == "full":
                        return TrainingPipeline._train_full(cfg)
                
                print(f"Unknown training command")
                return 1
    except Exception as e:
        print(f"Error during training: {e}")
        return 1

@staticmethod
def _train_bc(cfg: DictConfig) -> int:
    """Execute behavior cloning training with DictConfig"""
    # Extract configuration values using OmegaConf access
    model_config = cfg.model.transformer
    bc_config = cfg.model.bc
    data_config = cfg.data_collection.bc_data
    
    # Generate run name if not provided
    run_name = cfg.get("run_name") or generate_run_name("bc")
    
    print("=" * 60)
    print("PHASE 1: BEHAVIOR CLONING PRETRAINING")
    print("=" * 60)
    print(f"Run name: {run_name}")
    
    # Setup directories and logging
    checkpoint_dir, log_dir = setup_directories(f"{run_name}_bc")
    logger = setup_logging(run_name, log_dir=log_dir)
    
    # Save config copy
    save_config_copy(OmegaConf.to_container(cfg, resolve=True), checkpoint_dir)
    
    # ... rest of the implementation using cfg values
```

## 4. Configuration Directory Creation Script

### create_config_structure.py
```python
#!/usr/bin/env python3
"""
Script to create the Hydra configuration directory structure
"""

import os
import shutil
from pathlib import Path

def create_config_structure():
    """Create the configuration directory structure"""
    base_dir = Path("src/config")
    
    # Create directories
    dirs = [
        "train",
        "collect", 
        "play",
        "replay",
        "evaluate"
    ]
    
    for dir_name in dirs:
        (base_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    # Move unified_training.yaml to base.yaml
    if Path("src/unified_training.yaml").exists():
        shutil.copy("src/unified_training.yaml", base_dir / "base.yaml")
        print("Copied unified_training.yaml to src/config/base.yaml")
    
    # Create command-specific config files
    configs = {
        "train/bc.yaml": """
defaults:
  - base
  - _self_

# BC training specific configuration
training:
  mode: "bc"
  
logging:
  console:
    level: "INFO"
    progress_bars: true
""",
        "train/rl.yaml": """
defaults:
  - base
  - _self_

# RL training specific configuration
training:
  mode: "rl"
  rl:
    total_timesteps: 2000000
""",
        # Add other config files as needed
    }
    
    for config_path, content in configs.items():
        config_file = base_dir / config_path
        if not config_file.exists():
            config_file.write_text(content)
            print(f"Created {config_file}")

if __name__ == "__main__":
    create_config_structure()
```

## 5. Migration Strategy

1. **Phase 1**: Create config structure and move files
2. **Phase 2**: Update main.py with new Hydra integration
3. **Phase 3**: Update pipeline modules one by one
4. **Phase 4**: Test all commands and fix issues
5. **Phase 5**: Update documentation

## 6. Testing Strategy

### Commands to Test
```bash
# Training commands
python main.py train bc
python main.py train rl
python main.py train full

# Data collection commands
python main.py collect bc
python main.py collect selfplay

# Play commands
python main.py play human

# Replay commands
python main.py replay episode --episode-file data/test.pkl
python main.py replay browse

# Evaluation commands
python main.py evaluate model --model checkpoints/test.pt

# Test Hydra overrides
python main.py train bc model.transformer.embed_dim=128
python main.py train rl training.rl.total_timesteps=1000000
```

This implementation provides a clean integration of Hydra while maintaining the existing command structure that users are familiar with.