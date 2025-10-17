from typing import Any
import torch
import torch.nn as nn
from .base import Agent


class HumanAgent(Agent):
    """Agent that controls exactly 1 ship via human input"""

    def __init__(self, agent_id: str, team_id: int, squad: list[int]):
        """
        Initialize human agent

        Args:
            agent_id: Unique identifier for this agent
            team_id: Which team this agent belongs to
            squad: List of ship IDs this agent controls (should be exactly 1)
        """
        super().__init__(agent_id, team_id, squad)

        if len(squad) != 1:
            raise ValueError("HumanAgent must control exactly 1 ship")

        self.controlled_ship_id = squad[0]
        self.renderer = None  # Will be set by coordinator

    def set_renderer(self, renderer: Any) -> None:
        """Set the renderer for getting human input"""
        self.renderer = renderer

    def get_actions(self, obs_dict: dict[str, torch.Tensor]) -> dict[int, torch.Tensor]:
        """
        Get actions from human input

        Args:
            obs_dict: Observation dictionary from environment

        Returns:
            Dictionary mapping ship_id to action tensor
        """
        actions: dict[int, torch.Tensor] = {}

        # Only get actions for our controlled ship if it's alive
        ship_is_alive = (
            self.controlled_ship_id < obs_dict["ship_id"].shape[0]
            and obs_dict["alive"][self.controlled_ship_id, 0].item() > 0
        )

        if ship_is_alive and self.renderer:
            # Get human actions from renderer
            human_actions = self.renderer.get_human_actions()

            if self.controlled_ship_id in human_actions:
                actions[self.controlled_ship_id] = human_actions[
                    self.controlled_ship_id
                ]
            else:
                # Default to no action if no input available
                actions[self.controlled_ship_id] = torch.zeros(6, dtype=torch.float32)
        else:
            # Ship is dead or no renderer available
            actions[self.controlled_ship_id] = torch.zeros(6, dtype=torch.float32)

        return actions

    def get_agent_type(self) -> str:
        """Return agent type for logging/debugging"""
        return "human"

    def reset(self) -> None:
        """Reset agent state for new episode"""
        # Human agent doesn't need to reset any state
        pass
