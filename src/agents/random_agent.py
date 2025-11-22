import torch
import torch.nn as nn
from env.constants import Actions


class RandomAgent(nn.Module):
    """
    Agent that performs random actions.
    """

    def __init__(self, agent_id: str, team_id: int, squad: list[int], **kwargs):
        super().__init__()
        self.agent_id = agent_id
        self.team_id = team_id
        self.squad = squad
        self.num_actions = len(Actions)

    def forward(self, observation: dict, ship_ids: list[int]) -> dict[int, torch.Tensor]:
        """
        Get random actions for the specified ships.

        Args:
            observation: Observation dictionary
            ship_ids: List of ship IDs to get actions for

        Returns:
            Dict mapping ship_id -> action tensor (random binary)
        """
        actions = {}
        for ship_id in ship_ids:
            # Generate random binary actions
            # Using 0.5 probability for each action
            random_actions = torch.randint(0, 2, (self.num_actions,), dtype=torch.float32)
            actions[ship_id] = random_actions
        return actions
