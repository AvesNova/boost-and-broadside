from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class Agent(ABC, nn.Module):
    """Base class for all agents in the game that inherits from nn.Module"""

    def __init__(self, agent_id: str, team_id: int, squad: list[int]):
        """
        Initialize an agent

        Args:
            agent_id: Unique identifier for this agent
            team_id: Which team this agent belongs to
            squad: List of ship IDs this agent controls
        """
        ABC.__init__(self)  # Initialize ABC
        nn.Module.__init__(self)  # Initialize nn.Module

        self.agent_id = agent_id
        self.team_id = team_id
        self.squad = squad  # List of ship IDs this agent controls

    @abstractmethod
    def get_actions(self, obs_dict: dict[str, torch.Tensor]) -> dict[int, torch.Tensor]:
        """Get actions for this agent"""
        pass

    @abstractmethod
    def get_agent_type(self) -> str:
        """Return agent type for logging/debugging"""
        pass

    def reset(self) -> None:
        """Reset agent state for new episode (optional override)"""
        pass
