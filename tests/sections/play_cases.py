"""Shared by the play shell's tests: tiny saved policies and small matches."""

import pathlib

import torch

from bnb.contracts import GameConfig
from bnb.game import api
from bnb.models.relational import Relational, RelationalBuild, RelationalConfig
from bnb.models.yemong.model import Yemong, YemongConfig, yemong_build
from bnb.runs.checkpoint import save_policy

SMALL = GameConfig(3, 4, 900.0, episode_seconds=20.0)  # quick kills in a small arena


def tiny_model(name: str, game: GameConfig, seed: int = 0) -> torch.nn.Module:
    """The smallest Yemong or relational model for this game, every weight drawn from
    `seed` (so every output depends on its inputs)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if name == "yemong":
            config = YemongConfig(
                width=8,
                blocks=1,
                spatial_per_block=1,
                heads=1,
                map_width=4,
                value_bins=11,
                hex_rings=1,
                rotary_octaves=1,
                heading_harmonics=(1,),
            )
            model = Yemong(yemong_build(config, api.rules_of(game), api.ship_types()))
        else:
            build = RelationalBuild(
                RelationalConfig(hidden=8, value_bins=11), game.ships_per_team, game.num_ships
            )
            model = Relational(build)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(0.3 * torch.randn(parameter.shape, generator=generator))
    return model.eval()


def tiny_policy(
    directory: pathlib.Path, game: GameConfig, name: str = "relational", seed: int = 0
) -> pathlib.Path:
    """A tiny policy file for `game` (and the default ship card) at `directory`."""
    return save_policy(directory, tiny_model(name, game, seed), game=game, types=api.ship_types())
