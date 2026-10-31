"""Contract types: validation, schemas, pytree behaviour, and the privileged-data rule."""

import inspect
import typing

import pytest
import torch
from torch.utils import _pytree

from bnb.contracts import Action, GameConfig, Obs, Outputs, Pool, Privileged, ticks_of
from bnb.contracts.action import NUM_JOINT, from_joint, to_joint
from bnb.contracts.obs import SHIP_COLUMNS, SHIP_HIDDEN_ZERO
from bnb.contracts.pools import pool_sizes


def test_game_config_limits():
    assert GameConfig(5, 10, 2600.0).num_ships == 10
    with pytest.raises(ValueError):
        GameConfig(128, 10, 2600.0)  # 256 ships: indices must fit in uint8 with 255 = none
    with pytest.raises(ValueError):
        GameConfig(5, 65, 2600.0)
    with pytest.raises(ValueError):
        GameConfig(5, 10, 2600.0, num_teams=3)


def test_time_native_rules_must_be_whole_ticks():
    assert ticks_of(8.0, 1 / 30) == 240
    assert ticks_of(0.1, 1 / 30) == 3
    with pytest.raises(ValueError):
        ticks_of(0.05, 1 / 30)


def test_joint_action_round_trip():
    joint = torch.arange(NUM_JOINT)
    action = from_joint(joint)
    assert torch.equal(to_joint(action), joint.to(torch.int32))
    assert torch.equal(Action.unpack(action.packed()).stick, action.stick)


def test_observation_schema():
    assert len(SHIP_COLUMNS) == len(set(SHIP_COLUMNS)) == 19
    assert SHIP_COLUMNS[SHIP_HIDDEN_ZERO] == "alive"
    assert Obs.SCHEMA_VERSION == Action.SCHEMA_VERSION == 1


def test_contracts_are_pytrees():
    obs = Obs(ships=torch.zeros(2, 3), bullets=None, fields=torch.ones(1), globals=None)
    leaves, spec = _pytree.tree_flatten(obs)
    back = _pytree.tree_unflatten(leaves, spec)
    assert back.bullets is None and torch.equal(back.ships, obs.ships)
    out = _pytree.tree_map_only(torch.Tensor, lambda x: x + 1, Outputs(logp=torch.zeros(2)))
    assert out.value is None and out.logp.sum() == 2


def test_pool_shares_must_be_whole():
    pools = (Pool("a", 0.5, ("x", "y")), Pool("b", 0.5, ("x", "x")))
    assert pool_sizes(pools, 8) == (4, 4)
    with pytest.raises(ValueError):
        pool_sizes(pools, 7)
    with pytest.raises(ValueError):
        pool_sizes(pools[:1], 8)


def _controller_classes():
    from bnb.controllers.model import ModelController
    from bnb.controllers.scripted import Chaser, Random

    return (ModelController, Chaser, Random)


@pytest.mark.parametrize("cls", _controller_classes())
def test_no_controller_takes_privileged(cls):
    hints = typing.get_type_hints(cls.step)
    for name in inspect.signature(cls.step).parameters:
        assert hints.get(name) is not Privileged, f"{cls.__name__}.step({name}) takes Privileged"


def test_rollout_carry_holds_no_privileged():
    from bnb.controllers.scripted import Random
    from bnb.runtime.rollout import Rollout

    ro = Rollout(GameConfig(1, 1, 600.0), (Pool("p", 1.0, ("r", "r")),), {"r": Random(1)}, "r",
                 2, 4, torch.device("cpu"), compile=False)
    leaves = _pytree.tree_leaves(ro.carry, is_leaf=lambda x: isinstance(x, Privileged))
    assert not any(isinstance(x, Privileged) for x in leaves)
