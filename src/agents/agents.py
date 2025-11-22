import torch.nn as nn

from .human import HumanAgent
from .scripted import ScriptedAgent
from .team_transformer_agent import TeamTransformerAgent
from .replay import ReplayAgent
from .dummy import DummyAgent
from .random_agent import RandomAgent
from src.utils.model_finder import find_most_recent_model, find_best_model


def create_agent(agent_type: str, agent_config: dict) -> nn.Module:
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
        case "most_recent":
            model_path = find_most_recent_model()
            if not model_path:
                print("Warning: No model found for 'most_recent' agent. Using random initialization.")
            
            # Copy config to avoid mutating original
            config = agent_config.copy()
            config["model_path"] = model_path
            return TeamTransformerAgent(**config)
        case "best":
            model_path = find_best_model()
            if not model_path:
                print("Warning: No model found for 'best' agent. Using random initialization.")
            
            config = agent_config.copy()
            config["model_path"] = model_path
            return TeamTransformerAgent(**config)
        case _:
            raise TypeError(f"Unknown agent type: : {agent_type}")
