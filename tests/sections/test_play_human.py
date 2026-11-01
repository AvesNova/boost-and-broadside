"""The keyboard pilot (`bnb.controllers.human`): every power, stick level and shoot is
reachable, and the keys mean what the play shell's key list says."""

import itertools

import torch

from bnb.contracts.action import NUM_POWER, NUM_SHOOT, NUM_STICK
from bnb.controllers.human import INPUTS, human_action
from bnb.game.rules import POWER_MODES, STICK_LEVELS


def _action(**held):
    inputs = torch.tensor([held.get(name, False) for name in INPUTS])
    action = human_action(inputs)
    return int(action.power), int(action.stick), int(action.shoot)


def test_every_action_component_is_reachable():
    every = torch.tensor(list(itertools.product([False, True], repeat=len(INPUTS))))
    action = human_action(every)
    assert set(action.power.tolist()) == set(range(NUM_POWER))
    assert set(action.stick.tolist()) == set(range(NUM_STICK))
    assert set(action.shoot.tolist()) == set(range(NUM_SHOOT))
    joint = set(zip(action.power.tolist(), action.stick.tolist(), action.shoot.tolist()))
    assert len(joint) == NUM_POWER * NUM_STICK * NUM_SHOOT


def test_keys_map_to_the_game_controls():
    power = lambda **held: POWER_MODES[_action(**held)[0]]  # noqa: E731
    stick = lambda **held: STICK_LEVELS[_action(**held)[1]]  # noqa: E731
    assert power() == 0.0 and power(boost=True, reverse=True) == 0.0
    assert power(boost=True) == POWER_MODES[1] and power(reverse=True) == POWER_MODES[2]
    assert stick() == 0.0 and stick(left=True, right=True) == 0.0
    middle, full, lightest = (stick(left=True), stick(left=True, full=True),
                              stick(left=True, lightest=True))
    assert 0.0 < lightest < middle < full  # left is the positive (counter-clockwise) side
    assert stick(right=True) == -middle and stick(right=True, full=True) == -full
    assert stick(right=True, lightest=True) == -lightest
    assert _action(shoot=True)[2] == 1 and _action()[2] == 0
