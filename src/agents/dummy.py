from typing import Any
import torch
import torch.nn as nn

from src.env.constants import Actions


class DummyAgent(nn.Module):
    """
    Agent that performs no actions (all zeros).
    """

    def __init__(
        self, agent_id: str, team_id: int, squad: list[int], **kwargs: Any
    ):
        """
        Initialize dummy agent.

        Args:
            agent_id: Unique identifier for this agent.
            team_id: Which team this agent belongs to.
            squad: List of ship IDs this agent controls.
            **kwargs: Ignored additional arguments.
        """
        super().__init__()
        self.agent_id = agent_id
        self.team_id = team_id
        self.squad = squad
        self.num_actions = len(Actions)

    def forward(
        self, observation: dict[str, Any], ship_ids: list[int]
    ) -> dict[int, torch.Tensor]:
        """
        Get no-op actions for the specified ships.

        Args:
            observation: Observation dictionary.
            ship_ids: List of ship IDs to get actions for.

        Returns:
            Dict mapping ship_id to action tensor (all zeros).
        """
        actions = {}
        for ship_id in ship_ids:
            actions[ship_id] = torch.zeros(self.num_actions, dtype=torch.float32)
        return actions
