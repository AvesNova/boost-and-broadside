"""A training lineup (`TrainConfig.lineup`): every game of the batch flies the named types,
slot by slot, both teams, through resets; a lineup of the wrong size or an unknown name
fails at the config; `bnb ratings --lineup` flies it for every player."""

import pathlib

import pytest
import torch

from bnb import cli
from bnb.contracts import GameConfig, Pool
from bnb.league.matches import Player
from bnb.learn.ppo import LearnerConfig
from bnb.models.nearest import NearestConfig
from bnb.runtime.lanes import LaneConfig
from bnb.runtime.train import LEARNER, TrainConfig, Trainer, lineup_indices

CPU = torch.device("cpu")
pytestmark = pytest.mark.usefixtures("global_generator")
GAME = GameConfig(3, 2, 1200.0, episode_seconds=1.0)  # short games: resets inside the update


def _config(lineup) -> TrainConfig:
    return TrainConfig(
        game=GAME,
        model="nearest",
        nearest=NearestConfig(hidden=16, value_bins=11),
        lanes=LaneConfig(environments=0),
        precision="float32",
        pools=(Pool("chaser", 1.0, (LEARNER, "chaser")),),
        environments=8,
        length=32,
        shards=1,
        lookahead=0,
        compiled=False,
        host_store=False,
        learner=LearnerConfig(minibatches=2),
        lineup=lineup,
    )


def test_every_game_flies_the_lineup_through_resets():
    trainer = Trainer(_config(("Heavy", "Heavy", "Fighter")), CPU)
    trainer.update()
    types = trainer.rollout.carry["state"].ships.ship_type
    assert torch.equal(types, torch.tensor([[1, 1, 0] * 2] * 8, dtype=types.dtype))


def test_without_a_lineup_the_game_flies_its_own():
    trainer = Trainer(_config(None), CPU)
    types = trainer.rollout.carry["state"].ships.ship_type
    assert torch.equal(types[0], torch.tensor([0, 1, 0] * 2, dtype=types.dtype))


@pytest.mark.parametrize(
    "lineup, message",
    [(("Heavy",), "1 ships for a team of 3"), (("Heavy", "Heavy", "Wren"), "Wren")],
)
def test_a_wrong_lineup_fails_at_the_config(lineup, message):
    with pytest.raises(ValueError, match=message):
        _config(lineup)


def test_ratings_fly_the_lineup_for_every_player_without_one_of_its_own():
    field = [Player("a", policy=pathlib.Path("a")), Player("solo", ship_type=0)]
    lineup = ("P-51D", "P-51D", "Heavy", "Heavy", "Heavy")
    flown = cli.with_lineup(field, lineup, ("real_scale_frontline", "real_scale_elimination"))
    indices = lineup_indices(lineup, GameConfig(5, 10, 7800.0, ship_set="real-scale"))
    assert flown[0].ship_type == indices and flown[1].ship_type == 0
    with pytest.raises(SystemExit, match="other ship sets"):
        cli.with_lineup(field, lineup, ("frontline", "real_scale_frontline"))
