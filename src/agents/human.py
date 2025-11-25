from typing import Any
import torch
import torch.nn as nn


class HumanAgent(nn.Module):
    """
    Agent that represents a human player.

    In the current architecture, human input is handled directly by the
    GameCoordinator/Environment via the Renderer. This agent serves as a
    placeholder that returns no-op actions, which are then overridden
    by the actual human input in the game loop.
    """

    def __init__(self, **kwargs: Any):
        """
        Initialize human agent.

        Args:
            **kwargs: Ignored configuration arguments.
        """
        super().__init__()

    def forward(
        self, obs_dict: dict[str, torch.Tensor], ship_ids: list[int]
    ) -> dict[int, torch.Tensor]:
        """
        Forward pass for the agent.

        Returns no-op actions (all zeros) for the controlled ships.
        The actual human control is applied in the GameCoordinator loop.

        Args:
            obs_dict: Observation dictionary from environment.
            ship_ids: List of ship IDs to control.

        Returns:
            Dictionary mapping ship_id to action tensor (all zeros).
        """
        return {ship_id: torch.zeros(6, dtype=torch.float32) for ship_id in ship_ids}
