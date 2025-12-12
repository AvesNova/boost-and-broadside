from typing import Any
import torch
import torch.nn as nn

from env.constants import HumanActions


class RandomAgent(nn.Module):
    """
    Agent that performs random actions.
    """

    def __init__(self, agent_id: str, team_id: int, squad: list[int], **kwargs: Any):
        """
        Initialize random agent.

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
        self.num_actions = len(HumanActions)

    def forward(
        self, observation: dict[str, Any], ship_ids: list[int]
    ) -> dict[int, torch.Tensor]:
        """
        Get random actions for the specified ships.

        Args:
            observation: Observation dictionary.
            ship_ids: List of ship IDs to get actions for.

        Returns:
            Dict mapping ship_id to action tensor (random binary).
        """
        actions = {}
        for ship_id in ship_ids:
            # Generate random binary actions
            # Using 0.5 probability for each action
            random_actions = torch.randint(
                0, 2, (self.num_actions,), dtype=torch.float32
            )
            actions[ship_id] = random_actions
        return actions
