"""The keyboard pilot: a pure function of an input array the shell fills each frame.

The shell (`bnb.play`) reads the keys into a bool array ordered as `INPUTS` and turns it
into one ship's action here, so the mapping is testable without a window. Power: boost or
reverse while its key is held, coast with neither or both. Stick: left or right at the
middle level, the full level with `full`, the lightest with `lightest`; both sides at once
cancel. Shoot while `shoot` is held.
"""

import torch

from bnb.contracts import Action

INPUTS = ("boost", "reverse", "left", "right", "full", "lightest", "shoot")
INPUT = {name: i for i, name in enumerate(INPUTS)}

# Power indices (bnb.game.rules.POWER_MODES): 0 coast, 1 boost, 2 reverse.
COAST, BOOST, REVERSE = 0, 1, 2
# Stick indices by level (bnb.game.rules.STICK_LEVELS): odd turn right, even turn left.
LIGHTEST, MIDDLE, FULL = 1, 2, 3


def human_action(inputs: torch.Tensor) -> Action:
    """The action of the ship flown with these inputs, (...,) int32 per component.

    inputs (..., len(INPUTS)) bool, in INPUTS order.
    """
    held = {name: inputs[..., i].bool() for name, i in INPUT.items()}
    power = torch.where(
        held["boost"] & ~held["reverse"],
        BOOST,
        torch.where(held["reverse"] & ~held["boost"], REVERSE, COAST),
    )
    level = torch.where(held["full"], FULL, torch.where(held["lightest"], LIGHTEST, MIDDLE))
    left = held["left"] & ~held["right"]
    right = held["right"] & ~held["left"]
    stick = torch.where(left, 2 * level, torch.where(right, 2 * level - 1, 0))
    return Action(power.to(torch.int32), stick.to(torch.int32), held["shoot"].to(torch.int32))
