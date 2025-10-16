import torch.nn as nn

from .human import HumanAgent
from .scripted import ScriptedAgent
from .team_transformer_agent import TeamTransformerAgent
from .replay import ReplayAgent


def get_agent(agent_name: str, agent_config: dict) -> nn.Module:
    match agent_name:
        case "human":
            return HumanAgent(**agent_config)
        case "scripted":
            return ScriptedAgent(**agent_config)
        case "team_transformer_agent":
            return TeamTransformerAgent(**agent_config)
        case "replay_agent":
            return ReplayAgent(**agent_config)
        case _:
            raise TypeError(f"Invalid agent name: {agent_name}")
