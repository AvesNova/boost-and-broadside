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


def semi_random_label(probability: float) -> str:
    """Canonical player label for a scripted-action mixture probability.

    Lives with the agent rather than with the calibrator because the roster,
    the tournament and the calibrator all have to agree on it, and the roster
    must not depend on a mode module to spell its own entry labels.
    """
    if probability == 0.0:
        return "random"
    if probability == 1.0:
        return "scripted"
    digits = f"{probability:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"semi_scripted_{digits}"


def semi_random_probability(label: str) -> float | None:
    """Recover ``p_scripted`` from a label, or None if it is not a rung label.

    The inverse of :func:`semi_random_label`. It exists because a roster written
    before ``p_scripted`` was persisted carries the probability *only* in the
    label, and a resume that cannot recover it silently turns every rung into
    the uniform random agent while leaving its rating intact.
    """
    prefix = "semi_scripted_"
    if not label.startswith(prefix):
        return None
    try:
        probability = float(label[len(prefix) :].replace("p", "."))
    except ValueError:
        return None
    return probability if 0.0 < probability < 1.0 else None


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
