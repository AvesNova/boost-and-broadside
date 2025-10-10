"""
Centralized configuration management for Boost and Broadside
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(
    config_path: str, defaults: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load and validate configuration with defaults

    Args:
        config_path: Path to the configuration file
        defaults: Default configuration values to merge

    Returns:
        Loaded configuration dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        print(f"Warning: Config file {config_path} not found, using defaults")
        return defaults or get_default_config()

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Merge with defaults if provided
        if defaults:
            config = merge_configs(defaults, config)

        # Validate configuration
        validate_config(config)

        return config

    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        print("Using default configuration")
        return defaults or get_default_config()


def get_default_config(context: str = "full") -> Dict[str, Any]:
    """
    Get default configuration for a specific context

    Args:
        context: Context for the configuration (full, training, data_collection, etc.)

    Returns:
        Default configuration dictionary
    """
    # Validate context
    valid_contexts = ["full", "training", "data_collection", "logging", "evaluation"]
    if context not in valid_contexts:
        raise ValueError(
            f"Invalid context '{context}'. Valid contexts are: {valid_contexts}"
        )

    base_config = {
        "environment": {
            "world_size": [1200, 800],
            "max_ships": 8,
            "agent_dt": 0.04,
            "physics_dt": 0.02,
            "max_episode_steps": 10000,
        },
        "scripted_agent": {
            "max_shooting_range": 500.0,
            "angle_threshold": 5.0,
            "bullet_speed": 500.0,
            "target_radius": 10.0,
            "radius_multiplier": 1.5,
        },
    }

    if context in ["full", "training"]:
        base_config.update(
            {
                "model": {
                    "transformer": {
                        "token_dim": 10,
                        "embed_dim": 64,
                        "num_heads": 4,
                        "num_layers": 3,
                        "max_ships": 8,
                        "num_actions": 6,
                        "dropout": 0.1,
                        "use_layer_norm": True,
                    },
                    "bc": {
                        "learning_rate": 0.001,
                        "batch_size": 128,
                        "epochs": 50,
                        "validation_split": 0.2,
                        "early_stopping_patience": 10,
                        "policy_weight": 1.0,
                        "value_weight": 0.5,
                    },
                    "ppo": {
                        "learning_rate": 0.0003,
                        "n_steps": 4096,
                        "batch_size": 128,
                        "n_epochs": 10,
                        "gamma": 0.99,
                        "gae_lambda": 0.95,
                        "clip_range": 0.2,
                        "ent_coef": 0.01,
                        "vf_coef": 0.5,
                        "max_grad_norm": 0.5,
                    },
                },
                "training": {
                    "rl": {
                        "total_timesteps": 2000000,
                        "learning_team_id": 0,
                        "opponent": {
                            "type": "mixed",
                            "scripted_mix_ratio": 0.3,
                            "selfplay_memory_size": 50,
                            "opponent_update_freq": 10000,
                            "scripted_config": {
                                "max_shooting_range": 500.0,
                                "angle_threshold": 5.0,
                                "bullet_speed": 500.0,
                                "target_radius": 10.0,
                                "radius_multiplier": 1.5,
                            },
                        },
                        "selfplay_update_freq": 20000,
                        "min_steps_before_selfplay": 50000,
                        "eval_freq": 25000,
                        "eval_episodes": 20,
                        "checkpoint_freq": 100000,
                    },
                },
                "evaluation": {
                    "episodes": 100,
                    "game_mode": "2v2",
                    "model_type": "transformer",
                },
            }
        )

    if context in ["full", "data_collection"]:
        base_config.update(
            {
                "data_collection": {
                    "bc_data": {
                        "episodes_per_mode": {
                            "1v1": 2500,
                            "2v2": 2500,
                            "3v3": 2500,
                            "4v4": 2500,
                        },
                        "game_modes": ["1v1", "2v2", "3v3", "4v4"],
                        "output_dir": "data/bc_pretraining",
                        "gamma": 0.99,
                        "compress": False,
                    },
                    "selfplay_data": {
                        "total_episodes": 1000,
                        "game_mode": "nvn",
                        "output_dir": "data/selfplay",
                        "model_paths": [],
                        "gamma": 0.99,
                    },
                },
            }
        )

    if context in ["full", "logging"]:
        base_config.update(
            {
                "logging": {
                    "wandb": {
                        "enabled": False,
                        "project": "ship-combat-unified",
                        "tags": ["unified", "transformer"],
                    },
                    "tensorboard": {
                        "enabled": True,
                        "log_dir": "logs",
                    },
                    "console": {
                        "level": "INFO",
                        "progress_bars": True,
                    },
                },
                "paths": {
                    "models": "checkpoints",
                    "data": "data",
                    "logs": "logs",
                    "results": "results",
                },
            }
        )

    if context in ["evaluation"]:
        base_config.update(
            {
                "evaluation": {
                    "episodes": 100,
                    "game_mode": "2v2",
                    "model_type": "transformer",
                },
            }
        )

    return base_config


def merge_configs(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries

    Args:
        default: Default configuration
        override: Override configuration

    Returns:
        Merged configuration
    """
    result = default.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration values

    Args:
        config: Configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate environment
    if "environment" in config:
        env = config["environment"]
        if "world_size" in env:
            if not isinstance(env["world_size"], list) or len(env["world_size"]) != 2:
                raise ValueError("world_size must be a list of two integers")

        if "max_ships" in env:
            if not isinstance(env["max_ships"], int) or env["max_ships"] < 1:
                raise ValueError("max_ships must be a positive integer")

    # Validate model configuration
    if "model" in config:
        model = config["model"]

        if "transformer" in model:
            tf = model["transformer"]
            required_fields = [
                "token_dim",
                "embed_dim",
                "num_heads",
                "num_layers",
                "max_ships",
                "num_actions",
            ]
            for field in required_fields:
                if field not in tf:
                    raise ValueError(f"Missing required transformer field: {field}")

        if "ppo" in model:
            ppo = model["ppo"]
            if "learning_rate" in ppo and ppo["learning_rate"] <= 0:
                raise ValueError("PPO learning_rate must be positive")

    # Validate training configuration
    if "training" in config and "rl" in config["training"]:
        rl = config["training"]["rl"]
        if "total_timesteps" in rl and rl["total_timesteps"] <= 0:
            raise ValueError("total_timesteps must be positive")

    # Validate data collection configuration
    if "data_collection" in config:
        dc = config["data_collection"]

        if "bc_data" in dc:
            bc = dc["bc_data"]
            if "episodes_per_mode" in bc:
                for mode, count in bc["episodes_per_mode"].items():
                    if not isinstance(count, int) or count < 0:
                        raise ValueError(
                            f"Episode count for {mode} must be a non-negative integer"
                        )


def save_config(config: Dict[str, Any], path: str) -> None:
    """
    Save configuration to file

    Args:
        config: Configuration to save
        path: Path to save the configuration
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
