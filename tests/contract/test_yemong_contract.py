"""The Yemong controller against the contracts: no privileged input, and Outputs as the
runtime stores them (spec 10)."""

import inspect
import typing

import torch

from bnb.contracts import GameConfig, Obs, Privileged
from bnb.controllers.yemong import YemongController
from bnb.game import api
from bnb.models.yemong.model import Yemong, YemongConfig


def test_yemong_step_takes_no_privileged():
    hints = typing.get_type_hints(YemongController.step)
    for name in inspect.signature(YemongController.step).parameters:
        assert hints.get(name) is not Privileged, f"YemongController.step({name}) takes Privileged"


def test_yemong_outputs():
    config = GameConfig(2, 3, 1200.0)
    rules, types = api.rules_of(config), api.ship_types()
    state = api.reset(rules, types, api.uniforms(rules, 2))
    _, obs, _ = api.perceive(rules, state)
    view = Obs(ships=obs.ships.flatten(0, 1), fields=obs.fields.flatten(0, 1),
               globals=obs.globals.flatten(0, 1))
    controller = YemongController(Yemong(YemongConfig(), rules, types), 2, 4)
    carry = controller.init_carry(4, torch.device("cpu"))
    with torch.no_grad():
        action, carry, out = controller.step(carry, view, torch.ones(4, dtype=torch.bool))
    assert action.power.shape == (4, 2) and action.power.dtype == torch.int32
    assert out.logp.shape == out.entropy.shape == out.value.shape == (4, 2)
    assert out.values.shape == (4, 2, 2) and out.outcome.shape == (4, 4)
    torch.testing.assert_close(out.value, out.values.sum(-1))
    torch.testing.assert_close(out.outcome.sum(-1), torch.ones(4))
    leaves = torch.utils._pytree.tree_leaves(out.replay)
    assert leaves and all(leaf.shape[0] == 4 for leaf in leaves)
    assert torch.equal(carry.previous_action, (action.power * 14 + action.stick * 2
                                               + action.shoot).to(torch.int32))
