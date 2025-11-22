import torch
import torch.nn as nn
from env.constants import Actions


class DummyAgent(nn.Module):
    """
    Agent that performs no actions (all zeros).
    """

    def __init__(self, agent_id: str, team_id: int, squad: list[int], **kwargs):
        super().__init__()
        self.agent_id = agent_id
        self.team_id = team_id
        self.squad = squad
        self.num_actions = len(Actions)

    def forward(self, observation: dict, ship_ids: list[int]) -> dict[int, torch.Tensor]:
        """
        Get no-op actions for the specified ships.

        Args:
            observation: Observation dictionary
            ship_ids: List of ship IDs to get actions for

        Returns:
            Dict mapping ship_id -> action tensor (all zeros)
        """
        actions = {}
        for ship_id in ship_ids:
            actions[ship_id] = torch.zeros(self.num_actions, dtype=torch.float32)
        return actions
