from typing import Any

from .base import Agent
from .human import HumanAgent
from .scripted import ScriptedAgent
from .model import ModelAgent
from .replay import ReplayAgent


class AgentFactory:
    """Factory for creating agents from configuration"""

    @staticmethod
    def create_agent(agent_config: dict[str, Any]) -> Agent:
        """
        Create an agent from configuration

        Args:
            agent_config: Dictionary containing agent configuration

        Returns:
            Agent instance
        """
        agent_type = agent_config["type"]
        agent_id = agent_config["id"]
        team_id = agent_config["team_id"]
        squad = agent_config["squad"]

        match agent_type:
            case "human":
                return HumanAgent(agent_id, team_id, squad)
            case "scripted":
                return ScriptedAgent(
                    agent_id, team_id, squad, agent_config.get("config", {})
                )
            case "model":
                return ModelAgent(
                    agent_id,
                    team_id,
                    squad,
                    agent_config.get("model_path"),
                    agent_config.get("config", {}),
                )
            case "replay":
                return ReplayAgent(
                    agent_id, team_id, squad, agent_config.get("replay_data")
                )
            case _:
                raise ValueError(f"Unknown agent type: {agent_type}")
