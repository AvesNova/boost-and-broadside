import torch.nn as nn
from pathlib import Path
import logging
from omegaconf import OmegaConf

from .human import HumanAgent
from .scripted import ScriptedAgent
from .team_transformer_agent import TeamTransformerAgent
from .world_model_agent import WorldModelAgent
from .replay import ReplayAgent
from .dummy import DummyAgent
from .random_agent import RandomAgent
from utils.model_finder import find_most_recent_model, find_best_model

log = logging.getLogger(__name__)


def _load_agent_config_from_model(
    agent_config: dict, model_type: str, selection_criteria: str
) -> dict:
    """
    Helper to load configuration from a saved model.

    Args:
        agent_config: Base agent configuration.
        model_type: Type of model ("bc" or "rl").
        selection_criteria: "most_recent" or "best".

    Returns:
        Updated agent configuration dictionary with model path and transformer params.
    """
    if selection_criteria == "most_recent":
        model_path = find_most_recent_model(model_type)
    elif selection_criteria == "best":
        model_path = find_best_model(model_type)
    else:
        raise ValueError(f"Unknown selection criteria: {selection_criteria}")

    if not model_path:
        log.warning(
            f"No {model_type.upper()} model found for '{selection_criteria}_{model_type}' agent. Using random initialization."
        )
        return agent_config

    # Try to load config from run directory
    run_dir = Path(model_path).parent
    config_path = run_dir / "config.yaml"

    final_config = agent_config.copy()
    if config_path.exists():
        log.info(f"Loading agent config from {config_path}")
        saved_cfg = OmegaConf.load(config_path)
        if model_type == "world_model":
            # Extract world model config
            wm_cfg = OmegaConf.to_container(saved_cfg.world_model, resolve=True)
            # Map keys to WorldModelAgent args
            if "n_ships" in wm_cfg:
                wm_cfg["max_ships"] = wm_cfg.pop("n_ships")

            final_config.update(wm_cfg)
        else:
            # Extract transformer config for BC/RL
            transformer_cfg = OmegaConf.to_container(
                saved_cfg.train.model.transformer, resolve=True
            )
            # Remove keys that are not arguments to TeamTransformerAgent
            if "num_actions" in transformer_cfg:
                del transformer_cfg["num_actions"]
            final_config.update(transformer_cfg)
    else:
        log.warning(f"No config.yaml found in {run_dir}. Using current defaults.")

    final_config["model_path"] = model_path
    return final_config


def create_agent(agent_type: str, agent_config: dict) -> nn.Module:
    """
    Factory function to create agents based on type.

    Args:
        agent_type: String identifier for the agent type.
        agent_config: Dictionary of configuration parameters for the agent.

    Returns:
        An instantiated agent (nn.Module).

    Raises:
        TypeError: If the agent_type is unknown.
    """
    match agent_type:
        case "human":
            return HumanAgent(**agent_config)
        case "scripted":
            return ScriptedAgent(**agent_config)
        case "team_transformer_agent":
            return TeamTransformerAgent(**agent_config)
        case "replay_agent":
            return ReplayAgent(**agent_config)
        case "dummy":
            return DummyAgent(**agent_config)
        case "random":
            return RandomAgent(**agent_config)

        case "most_recent_bc":
            cfg = _load_agent_config_from_model(agent_config, "bc", "most_recent")
            return TeamTransformerAgent(**cfg)

        case "most_recent_rl":
            cfg = _load_agent_config_from_model(agent_config, "rl", "most_recent")
            return TeamTransformerAgent(**cfg)

        case "best_bc":
            cfg = _load_agent_config_from_model(agent_config, "bc", "best")
            return TeamTransformerAgent(**cfg)

        case "best_rl":
            cfg = _load_agent_config_from_model(agent_config, "rl", "best")
            return TeamTransformerAgent(**cfg)

        case "most_recent_world_model":
            cfg = _load_agent_config_from_model(
                agent_config, "world_model", "most_recent"
            )
            return WorldModelAgent(**cfg)

        case "best_world_model":
            cfg = _load_agent_config_from_model(agent_config, "world_model", "best")
            return WorldModelAgent(**cfg)

        case "world_model":
            # If model_path is not provided, we could try to find the most recent one.
            # For now, we rely on the config or default to random initialization.
            if "model_path" not in agent_config:
                pass
            return WorldModelAgent(**agent_config)

        case _:
            raise TypeError(f"Unknown agent type: {agent_type}")
