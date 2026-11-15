"""A model's initial weights are the same whatever the CPU thread count around its build
(`bnb.models.initialise` draws on one thread): the orthogonal initialisation's QR rounds
differently with the thread count, which moved arm 1's 40-tick golden by 2e-2. Exact
equality: the same ops on the same thread count."""

import pytest
import torch

from bnb.contracts import GameConfig
from bnb.models.mlp import MLP, MLPBuild
from bnb.runtime.train import TrainConfig, build_model, model_build

GAME = GameConfig(5, 10, 2600.0)


def _built(name: str) -> torch.nn.Module:
    if name == "mlp":
        return MLP(MLPBuild(GAME.ships_per_team, GAME.num_ships, GAME.num_fields, 64))
    config = TrainConfig(game=GAME, model=name)
    return build_model(model_build(config), torch.device("cpu"), seed=3)


def _weights_at(threads: int, name: str) -> dict[str, torch.Tensor]:
    before = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        return {key: value.clone() for key, value in _built(name).state_dict().items()}
    finally:
        torch.set_num_threads(before)


@pytest.mark.parametrize("name", ["yemong", "relational", "nearest", "mlp"])
def test_the_initial_weights_do_not_depend_on_the_thread_count(name, global_generator):
    one, four = _weights_at(1, name), _weights_at(4, name)
    assert one.keys() == four.keys()
    for key in one:
        assert torch.equal(one[key], four[key]), key


def test_the_build_leaves_the_thread_count_as_it_was(global_generator):
    before = torch.get_num_threads()
    torch.set_num_threads(3)
    try:
        _built("relational")
        assert torch.get_num_threads() == 3
    finally:
        torch.set_num_threads(before)
