from abc import ABC, abstractmethod
import torch


class Agent(ABC):
    """Base class for all agents in the game"""

    def __init__(self, agent_id: str, team_id: int, squad: list[int]):
        """
        Initialize an agent

        Args:
            agent_id: Unique identifier for this agent
            team_id: Which team this agent belongs to
            squad: List of ship IDs this agent controls
        """
        self.agent_id = agent_id
        self.team_id = team_id
        self.squad = squad  # List of ship IDs this agent controls

    @abstractmethod
    def get_actions(self, obs_dict: dict[str, torch.Tensor]) -> dict[int, torch.Tensor]:
        """
        Get actions for each ship in this agent's squad

        Args:
            obs_dict: Observation dictionary from environment

        Returns:
            Dictionary mapping ship_id to action tensor
        """
        pass

    @abstractmethod
    def get_agent_type(self) -> str:
        """Return agent type for logging/debugging"""
        pass

    def reset(self) -> None:
        """Reset agent state for new episode"""
        pass

    def get_ship_ids_for_obs(self, obs_dict: dict[str, torch.Tensor]) -> list[int]:
        """
        Filter ship IDs to only include those that exist in observation

        Args:
            obs_dict: Observation dictionary

        Returns:
            List of valid ship IDs from this agent's squad
        """
        valid_ships: list[int] = []
        max_ships = obs_dict["ship_id"].shape[0]

        for ship_id in self.squad:
            if ship_id < max_ships and obs_dict["alive"][ship_id, 0].item() > 0:
                valid_ships.append(ship_id)

        return valid_ships
