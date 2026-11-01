"""What a controller tells its ships to do."""

from typing import ClassVar

import torch

from .tree import contract

NUM_POWER = 3  # 0 coast, 1 boost, 2 reverse
NUM_STICK = 7  # 0 straight, then three levels to each side
NUM_SHOOT = 2
NUM_JOINT = NUM_POWER * NUM_STICK * NUM_SHOOT  # 42


@contract
class Action:
    """Factored action components, int32, each shaped (..., ships)."""

    power: torch.Tensor
    stick: torch.Tensor
    shoot: torch.Tensor
    SCHEMA_VERSION: ClassVar[int] = 1

    def packed(self) -> torch.Tensor:
        """(..., ships, 3) int32 in the order power, stick, shoot (the physics layout)."""
        return torch.stack([self.power, self.stick, self.shoot], -1).to(torch.int32)

    @staticmethod
    def unpack(packed: torch.Tensor) -> Action:
        return Action(packed[..., 0], packed[..., 1], packed[..., 2])


def to_joint(action: Action) -> torch.Tensor:
    """The joint index power * 14 + stick * 2 + shoot, in [0, 42)."""
    return action.power * (NUM_STICK * NUM_SHOOT) + action.stick * NUM_SHOOT + action.shoot


def from_joint(joint: torch.Tensor) -> Action:
    joint = joint.to(torch.int32)
    return Action(joint // (NUM_STICK * NUM_SHOOT), (joint // NUM_SHOOT) % NUM_STICK,
                  joint % NUM_SHOOT)
