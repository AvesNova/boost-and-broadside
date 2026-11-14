"""A training lineup (`TrainConfig.lineup`): every game of the batch flies the named types,
slot by slot, both teams, through resets; a lineup of the wrong size or an unknown name
fails at the config; `bnb ratings --lineup` flies it for every player. Without one, each
episode draws its lineup (`TrainConfig.lineup_draw`, the default): every slot one of the
set's drawn types with equal chance, both teams alike, keyed by the episode."""

import pathlib

import pytest
import torch

from bnb import cli
from bnb.contracts import GameConfig, Pool
from bnb.game import api
from bnb.league.matches import Player
from bnb.learn.ppo import LearnerConfig
from bnb.models.nearest import NearestConfig
from bnb.runtime.lanes import LaneConfig
from bnb.runtime.train import LEARNER, TrainConfig, Trainer, lineup_indices

CPU = torch.device("cpu")
pytestmark = pytest.mark.usefixtures("global_generator")
GAME = GameConfig(3, 2, 1200.0, episode_seconds=1.0)  # short games: resets inside the update


def _config(lineup, lineup_draw: bool = True) -> TrainConfig:
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
        lineup_draw=lineup_draw,
    )


def test_every_game_flies_the_lineup_through_resets():
    trainer = Trainer(_config(("Heavy", "Heavy", "Fighter")), CPU)
    trainer.update()
    types = trainer.rollout.carry["state"].ships.ship_type
    assert torch.equal(types, torch.tensor([[1, 1, 0] * 2] * 8, dtype=types.dtype))


def test_without_a_lineup_or_its_draw_the_game_flies_its_own():
    trainer = Trainer(_config(None, lineup_draw=False), CPU)
    types = trainer.rollout.carry["state"].ships.ship_type
    assert torch.equal(types, torch.tensor([[0, 1, 0] * 2] * 8, dtype=types.dtype))


def test_each_episode_draws_a_mirrored_lineup_through_resets():
    """Both teams fly the episode's drawn lineup slot for slot; an episode that ends
    spawns the next one's lineup (a ship's shield is within its new type's); the
    environments' lineups differ."""
    trainer = Trainer(_config(None), CPU)
    rollout = trainer.rollout
    first = rollout.carry["state"].ships.ship_type.clone()
    trainer.update()
    carry = rollout.carry
    types = carry["state"].ships.ship_type
    assert int(carry["episode"].min()) > 0  # every environment has reset
    assert torch.equal(types, rollout._lineups(carry["episode"]))
    assert torch.equal(types[:, :3], types[:, 3:])
    assert not torch.equal(types, first)
    assert len({tuple(row) for row in types.tolist()}) > 1
    assert set(types.unique().tolist()) <= set(api.drawn_lineup_types(GAME.ship_set))
    full = api.per_ship(trainer.types.max_shield, types)
    assert (carry["state"].ships.shield <= full * (1.0 + 1e-6)).all()


def test_a_drawn_slot_is_either_type_with_equal_chance():
    trainer = Trainer(_config(None), CPU)
    episodes = torch.arange(2000, dtype=torch.int64)[:, None].expand(-1, 8).reshape(-1)
    rollout = trainer.rollout
    rollout.environment_index = torch.arange(8).repeat(2000)  # 2000 episodes of 8 games
    heavy = rollout._lineups(episodes)[:, :3].float().mean()
    stderr = (0.25 / (2000 * 8 * 3)) ** 0.5
    assert abs(float(heavy) - 0.5) < 4 * stderr


def test_drawn_lineups_need_a_set_with_drawn_types():
    assert api.drawn_lineup_types("real-scale") == (0, 1)  # Yongma, Jujak
    with pytest.raises(ValueError, match="no drawn lineup types"):
        api.drawn_lineup_types("design-seed")


@pytest.mark.parametrize(
    "lineup, message",
    [(("Heavy",), "1 ships for a team of 3"), (("Heavy", "Heavy", "Wren"), "Wren")],
)
def test_a_wrong_lineup_fails_at_the_config(lineup, message):
    with pytest.raises(ValueError, match=message):
        _config(lineup)


def test_ratings_fly_the_lineup_for_every_player_without_one_of_its_own():
    field = [Player("a", policy=pathlib.Path("a")), Player("solo", ship_type=0)]
    lineup = ("Yongma", "Yongma", "Jujak", "Jujak", "Jujak")
    flown = cli.with_lineup(field, lineup, ("real_scale_frontline", "real_scale_elimination"))
    indices = lineup_indices(lineup, GameConfig(5, 10, 7800.0, ship_set="real-scale"))
    assert flown[0].ship_type == indices and flown[1].ship_type == 0
    with pytest.raises(SystemExit, match="other ship sets"):
        cli.with_lineup(field, lineup, ("frontline", "real_scale_frontline"))


def test_ratings_play_a_preset_at_an_arena_scale_floored_at_the_mode_minimum():
    assert cli.rated_game("real_scale_deathmatch", None).arena_radius == 4200.0
    assert cli.rated_game("real_scale_elimination", 2.0).arena_radius == 15600.0
    assert cli.rated_game("real_scale_deathmatch", 0.5).arena_radius == 3000.0  # the minimum
    with pytest.raises(SystemExit, match="above 0"):
        cli.rated_game("real_scale_deathmatch", 0.0)
