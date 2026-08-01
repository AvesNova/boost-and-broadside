"""Scripted controller with independently corrupted ship actions."""

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import ShipConfig
from boost_and_broadside.constants import (
    NUM_POWER_ACTIONS,
    NUM_SHOOT_ACTIONS,
    NUM_TURN_ACTIONS,
)
from boost_and_broadside.env.state import TensorState


class SemiRandomScriptedAgent:
    """Use a complete scripted action with probability ``p_scripted``.

    One Bernoulli choice is made independently for every ship on every simulation
    step. If it is false, all three action heads for that ship are sampled
    uniformly at random. Keeping the three heads together makes ``p_scripted``
    the probability that a ship follows one coherent scripted decision, rather
    than the less interpretable probability that each individual head survives.
    """

    def __init__(
        self,
        ship_config: ShipConfig,
        p_scripted: float,
        scripted_agent: StochasticScriptedAgent | None = None,
    ) -> None:
        if not 0.0 <= p_scripted <= 1.0:
            raise ValueError("p_scripted must lie in [0, 1]")
        self.p_scripted = float(p_scripted)
        self.scripted_agent = scripted_agent or StochasticScriptedAgent(
            ship_config, StochasticAgentConfig()
        )

    @staticmethod
    def random_actions_like(action: torch.Tensor) -> torch.Tensor:
        """Sample uniform full actions with the same batch and ship dimensions."""
        batch_size, num_ships = action.shape[:2]
        return torch.stack(
            [
                torch.randint(NUM_POWER_ACTIONS, (batch_size, num_ships), device=action.device),
                torch.randint(NUM_TURN_ACTIONS, (batch_size, num_ships), device=action.device),
                torch.randint(NUM_SHOOT_ACTIONS, (batch_size, num_ships), device=action.device),
            ],
            dim=-1,
        ).to(dtype=action.dtype)

    def mix_actions(
        self, scripted_action: torch.Tensor, random_action: torch.Tensor
    ) -> torch.Tensor:
        """Choose between complete scripted and random actions per ship."""
        if self.p_scripted == 1.0:
            return scripted_action
        if self.p_scripted == 0.0:
            return random_action
        choose_scripted = (
            torch.rand(scripted_action.shape[:2], device=scripted_action.device) < self.p_scripted
        )
        return torch.where(choose_scripted.unsqueeze(-1), scripted_action, random_action)

    def get_actions(self, state: TensorState) -> torch.Tensor:
        """Generate scripted and random candidates, then mix them per ship."""
        scripted_action = self.scripted_agent.get_actions(state)
        random_action = self.random_actions_like(scripted_action)
        return self.mix_actions(scripted_action, random_action)
