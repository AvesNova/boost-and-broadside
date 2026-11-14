"""The Yemong controller against the contracts: no privileged input, and Outputs as the
runtime stores them (spec 10)."""

import pytest
import torch

from bnb.contracts import REWARD_LEVELS, GameConfig, Obs
from bnb.contracts.action import to_joint
from bnb.controllers.yemong import YemongController
from bnb.game import api
from bnb.models.yemong.model import Yemong, YemongConfig, yemong_build

# The code under test draws from torch's default generator (model initialisation, the
# rollout, the scripted opponents): each test runs with it seeded and restored.
pytestmark = pytest.mark.usefixtures("global_generator")


def test_yemong_outputs():
    config = GameConfig(2, 3, 1200.0)
    rules, types = api.rules_of(config), api.ship_types()
    state = api.reset(rules, types, api.uniforms(rules, 2))
    _, obs, _ = api.perceive(rules, state)
    view = Obs(
        ships=obs.ships.flatten(0, 1),
        fields=obs.fields.flatten(0, 1),
        globals=obs.globals.flatten(0, 1),
        zones=obs.zones.flatten(0, 1),
    )
    controller = YemongController(Yemong(yemong_build(YemongConfig(), rules, types)), 2, 4)
    carry = controller.init_carry(4, torch.device("cpu"))
    with torch.no_grad():
        action, carry, out = controller.step(
            carry, view, torch.ones(4, dtype=torch.bool), torch.arange(view.ships.shape[0])
        )
    assert action.power.shape == (4, 2) and action.power.dtype == torch.int32
    assert out.log_probability.shape == out.entropy.shape == out.value.shape == (4, 2)
    assert out.values.shape == (4, 2, len(REWARD_LEVELS)) and out.outcome.shape == (4, 4)
    torch.testing.assert_close(out.value, out.values.sum(-1))
    torch.testing.assert_close(out.outcome.sum(-1), torch.ones(4))
    leaves = torch.utils._pytree.tree_leaves(out.replay)
    assert leaves and all(leaf.shape[0] == 4 for leaf in leaves)
    assert torch.equal(carry.previous_action, to_joint(action).to(torch.int32))
